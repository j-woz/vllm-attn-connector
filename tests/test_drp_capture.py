# SPDX-License-Identifier: Apache-2.0
"""Checks for the capture base, chiefly that a conf record survives the MQ.

Flowcept publishes with msgpack from a background flush thread. A value
msgpack cannot pack raises there, where it cannot reach the caller: the
publish is logged as an error, the record is dropped, and capture exits
zero having reported success. Nothing downstream notices until someone
reads the store and finds the configuration missing.

That is exactly what happened to Paccmann. Its params carry two
``pathlib.Path`` values (``modelpath``, ``smiles_language_filepath``), so
every run lost its whole conf record -- the gene count, the padding
length, the head layout -- while the 742 attention tasks landed fine.
A bare 2,128-wide vector with no record of what the axis means.

So these tests assert on msgpack-packability rather than on shape. The
record was always well-formed; it was simply unpackable, which no
structural check catches.

Run either way::

    python tests/test_drp_capture.py    # verbose, prints every case
    pytest tests/test_drp_capture.py    # same checks, quiet
"""

import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_attn_connector.drp_capture import AttentionCapture, to_jsonable

VERBOSE = __name__ == "__main__"


def say(msg):
    if VERBOSE:
        print(msg)


def packable(obj):
    """True if msgpack can serialize obj, mirroring Flowcept's MQ."""
    try:
        import msgpack
    except ImportError:  # pragma: no cover - msgpack ships with flowcept
        return None
    try:
        msgpack.packb(obj)
        return True
    except TypeError:
        return False


class _Recorder:
    """Minimal interceptor: remembers the conf instead of publishing it."""

    def __init__(self):
        self.workflows = []

    def send_model_workflow(self, workflow_id, conf, parent_workflow_id=None):
        self.workflows.append(
            {"workflow_id": workflow_id, "conf": conf,
             "parent_workflow_id": parent_workflow_id}
        )


class _PathyCapture(AttentionCapture):
    """A capture whose params carry Paths, as Paccmann's genuinely do."""

    MODEL_NAME = "Pathy"
    FRAMEWORK = "torch"
    METRIC_REFERENCE = "pathy_test_attention"

    def collect(self, **kwargs):  # pragma: no cover - not exercised here
        return []


# -- to_jsonable --------------------------------------------------------


def test_path_becomes_a_string():
    out = to_jsonable({"modelpath": Path("/tmp/out/model.pt")})
    assert out == {"modelpath": "/tmp/out/model.pt"}
    assert isinstance(out["modelpath"], str)
    say("  ok  Path -> str")


def test_pure_paths_too():
    # PurePath, not just the concrete Path, so this holds on any platform.
    assert to_jsonable(PurePosixPath("a/b")) == "a/b"
    say("  ok  PurePath -> str")


def test_nested_paths_are_reached():
    out = to_jsonable({"a": {"b": [Path("/x"), {"c": Path("/y")}]}})
    assert out == {"a": {"b": ["/x", {"c": "/y"}]}}
    say("  ok  nested Paths converted at any depth")


def test_tuples_and_sets_become_lists():
    assert to_jsonable((1, 2)) == [1, 2]
    assert sorted(to_jsonable({"a", "b"})) == ["a", "b"]
    say("  ok  tuple/set -> list")


def test_primitives_pass_through_unchanged():
    for v in (None, True, 3, 2.5, "s"):
        assert to_jsonable(v) is v or to_jsonable(v) == v
    # multiheads is a plain list of ints and must stay one
    assert to_jsonable([4, 4, 4, 4]) == [4, 4, 4, 4]
    say("  ok  primitives untouched")


def test_numpy_scalars_and_arrays():
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        say("  --  numpy absent, skipped")
        return
    assert to_jsonable(np.int64(2128)) == 2128
    assert isinstance(to_jsonable(np.int64(2128)), int)
    assert to_jsonable(np.array([1.5, 2.5])) == [1.5, 2.5]
    say("  ok  numpy scalar -> int, array -> list")


def test_unknown_objects_degrade_to_repr_not_an_exception():
    # Deliberate: an imperfect provenance record beats a dropped one.
    out = to_jsonable({"obj": object()})
    assert isinstance(out["obj"], str)
    assert packable(out) is not False
    say("  ok  unknown object -> repr, still packable")


def test_output_is_always_msgpack_packable():
    hostile = {
        "modelpath": Path("/tmp/model.pt"),
        "multiheads": (4, 4, 4, 4),
        "tags": {"a", "b"},
        "nested": {"p": Path("/y")},
        "weird": object(),
    }
    assert packable(hostile) is False, "fixture should be unpackable as-is"
    assert packable(to_jsonable(hostile)) is True
    say("  ok  unpackable conf becomes packable (the actual bug)")


# -- send_workflow ------------------------------------------------------


def test_send_workflow_coerces_paths():
    rec = _Recorder()
    cap = _PathyCapture(model=None, workflow_id="wf-1", interceptor=rec)
    cap.send_workflow({"modelpath": Path("/tmp/v/model.pt"),
                       "number_of_genes": 2128})

    conf = rec.workflows[0]["conf"]
    assert conf["modelpath"] == "/tmp/v/model.pt"
    assert not isinstance(conf["modelpath"], Path)
    assert packable(conf) is True
    say("  ok  send_workflow emits a packable conf")


def test_send_workflow_keeps_identity_and_parentage():
    rec = _Recorder()
    cap = _PathyCapture(model=None, workflow_id="wf-2", interceptor=rec)
    conf_id = cap.send_workflow({"number_of_genes": 2128})

    w = rec.workflows[0]
    assert conf_id == "wf-2:conf"
    # Filed under a child id with the caller's as parent: re-registering the
    # parent id is silently dropped by Flowcept, conf and all.
    assert w["workflow_id"] == "wf-2:conf"
    assert w["parent_workflow_id"] == "wf-2"
    assert w["conf"]["model"] == "Pathy"
    assert w["conf"]["framework"] == "torch"
    assert w["conf"]["metric_reference"] == "pathy_test_attention"
    assert w["conf"]["number_of_genes"] == 2128
    say("  ok  identity preserved, filed under a child id")


def _main():
    print("to_jsonable")
    test_path_becomes_a_string()
    test_pure_paths_too()
    test_nested_paths_are_reached()
    test_tuples_and_sets_become_lists()
    test_primitives_pass_through_unchanged()
    test_numpy_scalars_and_arrays()
    test_unknown_objects_degrade_to_repr_not_an_exception()
    test_output_is_always_msgpack_packable()
    print("\nsend_workflow")
    test_send_workflow_coerces_paths()
    test_send_workflow_keeps_identity_and_parentage()
    print("\nall checks passed")


if __name__ == "__main__":
    _main()
