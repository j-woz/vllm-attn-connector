# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree vLLM KV connector capturing one importance score per token.

Scores are emitted as Flowcept provenance via the ``vllm`` adapter
(``flowcept.flowceptor.adapters.vllm``).

vLLM resolves ``kv_connector`` as an attribute of ``kv_connector_module_path``
(``KVConnectorFactory.get_connector_class`` calls ``getattr(module, name)``), so
``KVNormConnector`` must be reachable from this module.

It is exposed lazily via :pep:`562` so the numeric modules can be imported and
tested without pulling in vLLM or Flowcept.
"""

from typing import TYPE_CHECKING

from .layout import LayerLayout, UnsupportedLayout, resolve_layer_layout

if TYPE_CHECKING:
    from .connector import KVNormConnector

__all__ = [
    "KVNormConnector",
    "LayerLayout",
    "UnsupportedLayout",
    "resolve_layer_layout",
]


def __getattr__(name: str):
    """Import the vLLM-dependent connectors only when actually requested."""
    if name == "KVNormConnector":
        from .connector import KVNormConnector

        return KVNormConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
