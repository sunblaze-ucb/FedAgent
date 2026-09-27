"""fedagent/unified_memory.py: (1) the ref's forced CPUOffload is lifted only when active, only for
forward_only FSDP1 engines, and the torch symbol is always restored; (2) update_weights releases
the allocator cache first, keeps verl's @register metadata, and is untouched when inactive; (3) the
run_fed knob survives YAML's on/off booleans, and a patch that cannot apply leaves verl importable.
No GPU needed: the verl methods are replaced by stubs that do what the real code does."""
import asyncio
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")
import torch.distributed.fsdp as fsdp_pkg  # noqa: E402

from fedagent import unified_memory as um  # noqa: E402


def _stub_build(self, module):
    # mirrors transformer_impl.FSDPEngine._build_fsdp_module's FSDP1 ref branch
    from torch.distributed.fsdp import CPUOffload

    cpu_offload = None
    if self.engine_config.forward_only:
        cpu_offload = CPUOffload(offload_params=True)
    return types.SimpleNamespace(cpu_offload=cpu_offload)


def _engine(forward_only=True, strategy="fsdp"):
    return types.SimpleNamespace(engine_config=types.SimpleNamespace(forward_only=forward_only,
                                                                     strategy=strategy))


@pytest.mark.parametrize("m,integrated,expect_offload", [
    ("on", False, False),
    ("auto", True, False),
    ("auto", False, True),
    ("off", True, True),
])
def test_ref_offload_decision(monkeypatch, m, integrated, expect_offload):
    monkeypatch.setenv("FEDAGENT_UNIFIED_MEMORY", m)
    monkeypatch.setattr(um, "_device_is_integrated", lambda: integrated)
    orig_cls = fsdp_pkg.CPUOffload
    out = um._wrap_build_fsdp_module(_stub_build)(_engine(), None)
    assert out.cpu_offload.offload_params is expect_offload
    assert fsdp_pkg.CPUOffload is orig_cls


def test_actor_untouched(monkeypatch):
    monkeypatch.setenv("FEDAGENT_UNIFIED_MEMORY", "on")
    assert um._wrap_build_fsdp_module(_stub_build)(_engine(forward_only=False), None).cpu_offload is None


def test_fsdp2_left_stock(monkeypatch):
    monkeypatch.setenv("FEDAGENT_UNIFIED_MEMORY", "on")
    out = um._wrap_build_fsdp_module(_stub_build)(_engine(strategy="fsdp2"), None)
    assert out.cpu_offload.offload_params is True


def test_symbol_restored_on_error(monkeypatch):
    monkeypatch.setenv("FEDAGENT_UNIFIED_MEMORY", "on")
    orig_cls = fsdp_pkg.CPUOffload

    def boom(self, module):
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError):
        um._wrap_build_fsdp_module(boom)(_engine(), None)
    assert fsdp_pkg.CPUOffload is orig_cls


@pytest.mark.parametrize("m,expect_calls", [("on", 1), ("off", 0)])
def test_update_weights_releases_first(monkeypatch, m, expect_calls):
    monkeypatch.setenv("FEDAGENT_UNIFIED_MEMORY", m)
    order = []

    async def orig(self, global_steps=None, mode="auto"):
        order.append(("sync", global_steps, mode))
        return "done"

    orig.some_register_attr = "kept"
    wrapped = um._wrap_update_weights(orig, lambda force_sync: order.append(("empty", force_sync)))
    assert asyncio.run(wrapped(object(), 3, mode="naive")) == "done"
    assert order == [("empty", True)] * expect_calls + [("sync", 3, "naive")]
    assert wrapped.some_register_attr == "kept"            # functools.wraps -> MAGIC_ATTR survives


def test_bad_mode_rejected(monkeypatch):
    monkeypatch.setenv("FEDAGENT_UNIFIED_MEMORY", "maybe")
    with pytest.raises(ValueError):
        um.mode()


def test_unset_is_stock(monkeypatch):
    monkeypatch.delenv("FEDAGENT_UNIFIED_MEMORY", raising=False)
    assert um.mode() == "off"
    assert um.install_deferred_unified_memory_patch() is True   # no-op, nothing armed


@pytest.mark.parametrize("value,expect", [
    ("off", "off"), ("on", "on"), ("OFF", "off"), ("false", "off"),    # YAML 1.1 booleans
    ("auto", "auto"), ('"off"', "off"), ("null", "auto"),
])
def test_yaml_spellings_of_the_knob(tmp_path, value, expect):
    """The documented spelling, ``unified_memory: off``, reaches OmegaConf.load as False; it must
    still mean off (``str(value or "auto")`` made it auto), and ``on`` must not be rejected as
    'true'."""
    import argparse

    from fedagent.fed import run_fed

    path = tmp_path / "fed.yaml"
    path.write_text(f"unified_memory: {value}\n")
    args = argparse.Namespace(config=str(path), model_path=None, output_dir=None, rounds=None,
                              clients=None)
    assert run_fed.unified_memory_env(run_fed.load_cfg(args)) == {"FEDAGENT_UNIFIED_MEMORY": expect}


def test_unknown_knob_value_rejected():
    from omegaconf import OmegaConf

    from fedagent.fed import run_fed

    with pytest.raises(ValueError, match=r"auto\|on\|off"):
        run_fed.unified_memory_env(OmegaConf.create({"unified_memory": "maybe"}))


def test_patch_that_cannot_apply_leaves_verl_importable(monkeypatch, capsys):
    """_apply runs inside verl's own import and, under the default auto, on every platform: a seam
    that a verl upgrade moved must log and leave that target stock, not fail the import of verl."""
    def moved():
        raise ImportError("cannot import name 'aggressive_empty_cache'")

    monkeypatch.setitem(um._PATCHES, "verl.moved_target", moved)
    monkeypatch.setattr(um, "_attempted", set())
    um._apply("verl.moved_target")                      # must not raise
    um._apply("verl.moved_target")                      # one attempt per process
    err = capsys.readouterr().err
    assert err.count("could not patch verl.moved_target") == 1 and "ImportError" in err


def test_deferred_hook_patches_real_verl():
    """Fresh interpreter: arm, then import verl the way a Ray worker does; both targets patched,
    update_weights still carries verl's @register metadata."""
    pytest.importorskip("verl")
    code = (
        "import os; os.environ['FEDAGENT_UNIFIED_MEMORY']='on'\n"
        "from fedagent import unified_memory as um\n"
        "assert um.install_deferred_unified_memory_patch()\n"
        "from verl.workers.engine_workers import ActorRolloutRefWorker as W\n"
        "from verl.workers.engine.fsdp.transformer_impl import FSDPEngine as E\n"
        "from verl.single_controller.base.decorator import MAGIC_ATTR\n"
        "assert getattr(W.update_weights, '__fedagent_unified_memory__', False)\n"
        "assert getattr(W.update_weights, MAGIC_ATTR, None) is not None\n"
        "assert getattr(E._build_fsdp_module, '__fedagent_unified_memory__', False)\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
