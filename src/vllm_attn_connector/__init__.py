# SPDX-License-Identifier: Apache-2.0
"""Exact per-decode-step attention capture for vLLM, out-of-tree.

Two extension points, both public:
  * ``install_probe()`` registers an attention-backend override that copies the
    decode queries. Must run BEFORE the engine is constructed.
  * ``AttnConnector`` is a ``KVConnector`` that recomputes q.K^T against the
    paged prompt keys and emits the result as Flowcept provenance.

The same provenance shape is also produced from drug-response models, which
need none of the vLLM machinery because their attention is already
materialised. See ``drp_emit``, ``drp_paccmann`` and ``drp_hidra``.

Names are resolved lazily. ``AttnConnector`` and the Triton kernels require
vLLM and a CUDA device; the drug-response path requires neither, and importing
this package on a laptop without vLLM installed must not fail. Every public
name below is still reachable exactly as before -- the import simply happens on
first access rather than at module load.
"""

from typing import TYPE_CHECKING

__all__ = [
    "AttnConnector",
    "install_probe",
    "REGISTRY",
    "decode_attention",
    "decode_attention_torch",
]

_LAZY = {
    "AttnConnector": (".connector", "AttnConnector"),
    "decode_attention": (".kernels", "decode_attention"),
    "decode_attention_torch": (".kernels", "decode_attention_torch"),
    "REGISTRY": (".probe", "REGISTRY"),
    "install_probe": (".probe", "install"),
}


def __getattr__(name: str):
    """Resolve a public name on first use (PEP 562)."""
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .connector import AttnConnector
    from .kernels import decode_attention, decode_attention_torch
    from .probe import REGISTRY, install as install_probe
