# SPDX-License-Identifier: Apache-2.0
"""Attention-backend override that stashes decode queries. The sensor half.

Queries never enter the KV cache -- they are transient activations, consumed and
freed every step. A ``KVConnector`` therefore cannot see them, which is why
capturing exact attention needs a second extension point.

vLLM provides one: ``register_backend(AttentionBackendEnum.CUSTOM)``
(``v1/attention/backends/registry.py``), documented for exactly this. We
subclass the selected backend's ``Impl``, copy the decode queries, and delegate
everything else to ``super()``. **No vLLM source is patched and no kernel is
touched** -- FlashAttention runs unmodified.

Why the impl and not a module hook
----------------------------------
``Attention.forward`` is traced through by ``torch.compile``; only the custom op
``unified_attention_with_output`` survives as a runtime node, and it dispatches
``self.impl.forward(...)`` dynamically. Overriding the *impl* is therefore the
only hook that is still present in the traced graph; an ``nn.Module`` forward
hook is erased by the trace.

Why that is not enough under CUDA graphs
---------------------------------------
Surviving the trace is not the same as surviving graph replay. A CUDA graph
records the kernels one execution of the trace produced, and a replay runs only
those kernels: no Python. So ``_capture`` runs once, while vLLM captures the
graph, and never again for that graph -- which is every decode step of a served
run, since ``vllm serve`` captures a full graph per decode batch size by
default. A registry that a step-scoped ``clear()`` empties is therefore empty
on every replayed step (issue #2: attn_sum == 0 on every served record).

What does survive replay is the recorded *copy*. So the probe owns one
persistent buffer per layer, allocated outside any capture, and ``_capture``
copies into it. During capture that copy becomes a recorded kernel; every
replay of that graph re-runs it and the buffer holds the current step's
queries with no Python involved. The buffer is never cleared -- clearing it
would be exactly the bug -- so the connector instead asks whether the write it
is about to read is fresh *for this step*:

* a graph write is fresh whenever the step replays a graph whose recorded
  kernels include that copy. ``(cudagraph_runtime_mode, batch_descriptor)``
  from the forward context is what vLLM's own dispatcher keys graphs by, and
  both the capturing forward and every replay carry it, so the probe records
  the key at capture time and the connector matches it at read time.
* an eager write is fresh only for the step it ran on, tracked by a step
  generation the connector bumps once per step.

Sizing: one buffer per layer, ``max_rows x heads x head_size``, where
``max_rows`` covers the largest captured decode batch. Not one buffer per
graph: that would be ~30x the memory (the capture sizes sum to far more than
the largest one), and it buys nothing, since every graph writes the rows it
owns and the connector only ever reads the rows the current step filled.

Why decode only
---------------
vLLM v1 orders decode requests first in the batch and exposes
``num_decode_tokens``. In decode each request contributes exactly one query
token, so batch row *i* is request slot *i* -- no ``query_start_loc`` parsing,
no token-to-request mapping. Prefill queries are skipped: capturing them is
O(T^2) work and O(T*H*D) memory per layer.

The copy is small (``num_decode_reqs x heads x head_size``) but it must happen
on the current stream: ``query`` is a buffer vLLM reuses next step, so a
deferred copy could read it after ``q_proj`` has overwritten it.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.attn_connector")
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


def graph_key_of(ctx: Any) -> Any | None:
    """The CUDA graph a forward context runs under, or None for no graph.

    ``(cudagraph_runtime_mode, batch_descriptor)`` is the pair vLLM's
    ``CudagraphDispatcher`` itself uses to pick a graph
    (``v1/cudagraph_dispatcher.py``), and ``BatchDescriptor`` is a frozen
    dataclass, so the pair is hashable and compares by value. Capture and
    replay of the same graph therefore produce equal keys, which is the whole
    point: the probe records the key while capturing and the connector
    reconstructs it while reading.

    ``NONE`` means no graph will be replayed, so the probe's Python body runs
    this step and the write is a live one. Its descriptor carries the raw token
    count and would otherwise mint a fresh key per batch size, so it collapses
    to None.
    """
    if ctx is None:
        return None
    mode = getattr(ctx, "cudagraph_runtime_mode", None)
    desc = getattr(ctx, "batch_descriptor", None)
    if mode is None or desc is None:
        return None
    if getattr(mode, "value", mode) == 0:        # CUDAGraphMode.NONE
        return None
    return (mode, desc)


def current_graph_key() -> Any | None:
    """``graph_key_of`` for the forward in progress.

    Both halves call this: the probe from inside ``impl.forward``, the
    connector from ``wait_for_save``, which vLLM runs inside the same
    ``set_forward_context`` block (``kv_connector_model_runner_mixin.py``:
    "This context manager must be used within an active forward context").
    """
    try:
        from vllm.forward_context import get_forward_context

        return graph_key_of(get_forward_context())
    except Exception:
        return None


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


class QueryRegistry:
    """Decode queries, written by the probe, read by the connector in
    ``wait_for_save`` -- which runs after the forward, so the handoff needs no
    synchronisation beyond ordinary step ordering.

    One persistent buffer per layer, keyed by layer name because that is the
    one identifier both halves share. See the module docstring for why the
    buffer persists across steps and how freshness is tracked instead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: dict[str, torch.Tensor] = {}
        # Graph keys whose recorded kernels write this layer's buffer. A
        # permanent property of the graph, so a set and not a last-value.
        self._graph_keys: dict[str, set] = {}
        # Step generation at which the probe last ran *live* for this layer.
        self._eager_gen: dict[str, int] = {}
        self._rows: dict[str, int] = {}
        self._gen = 0
        self._max_rows = 0
        self.enabled = False
        self.layers_seen: set[str] = set()

    # -- configuration ------------------------------------------------------ #

    def configure(self, max_rows: int) -> None:
        """Rows to allocate per layer, from the largest decode batch vLLM will
        capture a graph for. Must be called before graph capture: a buffer that
        has to grow afterwards leaves already-recorded kernels writing into the
        old one, which costs those graphs their capture (see ``put``).
        """
        self._max_rows = max(self._max_rows, int(max_rows))

    # -- probe side --------------------------------------------------------- #

    def put(self, layer_name: str, q: torch.Tensor, n: int,
            graph_key: Any = None) -> None:
        """Copy ``q[:n]`` into this layer's buffer and record how it got there."""
        with self._lock:
            buf = self._buf.get(layer_name)
            if (buf is None or buf.shape[0] < n or buf.shape[1:] != q.shape[1:]
                    or buf.dtype != q.dtype or buf.device != q.device):
                if _capturing():
                    # Allocating here would take the buffer from the graph's
                    # private pool, and the graph would be the only thing
                    # keeping it addressable. Skip instead: the key is not
                    # recorded, so the connector reports the layer unscored
                    # rather than reading a buffer nobody refreshes.
                    logger.warning(
                        "attn_connector: %s needs a %d-row query buffer but the "
                        "stream is capturing a CUDA graph; this graph will not "
                        "capture queries. Call REGISTRY.configure() before "
                        "capture.", layer_name, n)
                    return
                stale = self._graph_keys.pop(layer_name, None)
                if stale:
                    logger.warning(
                        "attn_connector: %s query buffer grew to %d rows after "
                        "%d CUDA graph(s) were captured against the old one; "
                        "those graphs can no longer capture queries.",
                        layer_name, max(n, self._max_rows), len(stale))
                buf = torch.empty((max(n, self._max_rows), *q.shape[1:]),
                                  dtype=q.dtype, device=q.device)
                self._buf[layer_name] = buf
            # Under capture this becomes a recorded kernel; every replay of the
            # graph re-runs it, which is what keeps the buffer current.
            buf[:n].copy_(q[:n])
            if graph_key is None:
                self._eager_gen[layer_name] = self._gen
            else:
                self._graph_keys.setdefault(layer_name, set()).add(graph_key)
            self._rows[layer_name] = n
            self.layers_seen.add(layer_name)

    # -- connector side ----------------------------------------------------- #

    def get(self, layer_name: str, graph_key: Any = None) -> torch.Tensor | None:
        """This layer's queries, or None if nothing wrote them *this step*.

        None is the signal that the step cannot be scored. Returning a stale
        buffer instead would produce records that look right and are not.
        """
        if graph_key is None:
            if self._eager_gen.get(layer_name, -1) != self._gen:
                return None
        elif graph_key not in self._graph_keys.get(layer_name, ()):
            return None
        return self._buf.get(layer_name)

    def rows(self, layer_name: str) -> int:
        """Rows the last write to this layer covered. Capture-time under a
        graph, so a padded batch size, not this step's request count."""
        return self._rows.get(layer_name, 0)

    def end_step(self) -> None:
        """Retire live writes. Buffers deliberately survive: under CUDA graphs
        the probe never runs again, so dropping them is the bug this replaced.
        """
        self._gen += 1

    def clear(self) -> None:
        """Forget everything, buffers included. Not part of the step loop."""
        with self._lock:
            self._buf.clear()
            self._graph_keys.clear()
            self._eager_gen.clear()
            self._rows.clear()


REGISTRY = QueryRegistry()


def _capture(layer: Any, query: torch.Tensor, attn_metadata: Any) -> None:
    """Copy this step's decode queries, if any."""
    if not REGISTRY.enabled or attn_metadata is None:
        return
    # `num_decode_tokens` exists on the dataclass but is only populated on some
    # paths (it was 0 on every call in testing). `max_query_len` is always set,
    # and == 1 is exactly the condition we need: one query token per request,
    # so batch row i is request slot i. Anything else -- prefill, chunked
    # prefill, speculative decoding -- breaks that identity, so skip it rather
    # than mis-attribute. It also keeps mixed prefill+decode graphs out of the
    # registry, so the connector sees them as unscorable instead of reading a
    # buffer their replay never writes.
    if getattr(attn_metadata, "max_query_len", 0) != 1:
        return
    # `query` is padded out to the CUDA-graph batch size; only the first
    # num_actual_tokens rows are real. Under capture that is the whole padded
    # batch (`_dummy_run` runs the capture size unpadded), so the recorded copy
    # covers every row any replay of this graph can fill.
    n = getattr(attn_metadata, "num_actual_tokens", 0)
    if not n:
        return
    name = getattr(layer, "layer_name", None)
    if name is None:
        return
    REGISTRY.put(name, query, n, current_graph_key())


def install(backend: str | None = None) -> bool:
    """Register a query-capturing subclass of the active attention backend.

    Must run before the engine is built: ``_cached_get_attn_backend`` is
    ``@cache``-decorated, so the class is resolved once and never re-read.

    Returns True if the override was installed.
    """
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    except ImportError as exc:  # pragma: no cover
        logger.warning("attn_connector: attention backend registry unavailable: %s", exc)
        return False

    name = backend or "FLASH_ATTN"
    try:
        member = AttentionBackendEnum[name]
    except KeyError:
        logger.warning("attn_connector: unknown attention backend %r; probe not installed", name)
        return False

    try:
        base_cls = member.get_class()
    except Exception as exc:  # pragma: no cover
        logger.warning("attn_connector: cannot resolve %s: %s", name, exc)
        return False

    base_impl = base_cls.get_impl_cls()

    class _ProbedImpl(base_impl):  # type: ignore[misc, valid-type]
        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, *args, **kwargs):
            _capture(layer, query, attn_metadata)
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, *args, **kwargs
            )

    class _ProbedBackend(base_cls):  # type: ignore[misc, valid-type]
        @staticmethod
        def get_impl_cls():
            return _ProbedImpl

    # __qualname__ matters as much as __name__: vLLM's platform re-derives the
    # class path from the class object itself (`_backend_cls_path`, cuda.py:423)
    # rather than reusing the string handed to register_backend. A class defined
    # inside a function has qualname "install.<locals>._ProbedBackend", which
    # resolves to a module path that does not exist.
    for cls, base in ((_ProbedImpl, base_impl), (_ProbedBackend, base_cls)):
        cls.__name__ = f"Probed{base.__name__}"
        cls.__qualname__ = cls.__name__
        cls.__module__ = __name__
        globals()[cls.__name__] = cls
    register_backend(member, f"{__name__}.{_ProbedBackend.__name__}")
    REGISTRY.enabled = True
    logger.info("attn_connector: query probe installed over %s", base_cls.__name__)
    return True
