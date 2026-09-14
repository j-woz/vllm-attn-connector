# SPDX-License-Identifier: Apache-2.0
"""Exact per-decode-step attention capture for vLLM, out-of-tree.

Two extension points, both public:
  * ``install_probe()`` registers an attention-backend override that copies the
    decode queries. Must run BEFORE the engine is constructed.
  * ``AttnConnector`` is a ``KVConnector`` that recomputes q.K^T against the
    paged prompt keys and emits the result as Flowcept provenance.
"""

from .connector import AttnConnector
from .kernels import decode_attention, decode_attention_torch
from .probe import REGISTRY, install as install_probe

__all__ = [
    "AttnConnector",
    "install_probe",
    "REGISTRY",
    "decode_attention",
    "decode_attention_torch",
]
