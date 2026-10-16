"""Infrastructure-only retry boundary; no GPU or simulator imports."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "native_launcher", Path(__file__).with_name("native_launcher.py")
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_child_mps_isolation_preserves_parent_and_logs_bounded_startup(
    tmp_path, monkeypatch
):
    runner = tmp_path / "native_eval.py"
    runner.write_text("# frozen stub\n")
    protocol = tmp_path / "native_protocol.json"
    protocol.write_text("{}\n")
    monkeypatch.setattr(
        launcher,
        "FROZEN_RUNNER_SHA256",
        hashlib.sha256(runner.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        launcher,
        "FROZEN_PROTOCOL_SHA256",
        hashlib.sha256(protocol.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/test-parent-mps")
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "70")
    parent_env = dict(launcher.os.environ)
    calls = []

    class FakeEnv:
        def __init__(self, *a, **kw):
            calls.append(kw)

    def env_factory(*a, **kw):
        return {**parent_env, "RRP_SIMPLE_AUTHKEY": "test-auth", "MUJOCO_GL": "egl"}

    simple = SimpleNamespace(worker_env=env_factory, SimpleEnv=FakeEnv)
    envs = ModuleType("rrp.envs")
    envs.simple = simple
    monkeypatch.setitem(sys.modules, "rrp", ModuleType("rrp"))
    monkeypatch.setitem(sys.modules, "rrp.envs", envs)
    diag = tmp_path / "diagnostics"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launcher",
            "--runner",
            str(runner),
            "--diagnostics",
            str(diag),
            "--",
            "--arm",
            "psi01",
        ],
    )

    def run(path, run_name):
        assert path == str(runner) and run_name == "__main__"
        assert sys.argv == [str(runner), "--arm", "psi01"]
        child = simple.worker_env(b"auth")
        assert not any(k.startswith("CUDA_MPS_") for k in child)
        assert child["RRP_SIMPLE_AUTHKEY"] == "test-auth"
        assert child["MUJOCO_GL"] == "egl"
        assert dict(launcher.os.environ) == parent_env
        FakeEnv("task", start_timeout=1800)
        FakeEnv("task")

    monkeypatch.setattr(launcher.runpy, "run_path", run)
    launcher.main()
    assert [c["start_timeout"] for c in calls] == [120.0, 120.0]
    assert [c["log"].name for c in calls] == [
        "isaac-worker-0.log",
        "isaac-worker-1.log",
    ]
    assert dict(launcher.os.environ) == parent_env
    amendment = json.loads((diag / "amendment.json").read_text())
    assert amendment["parent_mps"]["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "70"


def test_frozen_mismatch_fails_before_external_import_or_diagnostics(
    tmp_path, monkeypatch
):
    runner = tmp_path / "native_eval.py"
    runner.write_text("changed")
    diag = tmp_path / "diagnostics"
    monkeypatch.setattr(
        sys, "argv", ["launcher", "--runner", str(runner), "--diagnostics", str(diag)]
    )
    imported = []
    import builtins

    original = builtins.__import__

    def traced_import(name, *a, **kw):
        if name.startswith("rrp"):
            imported.append(name)
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", traced_import)
    with pytest.raises(SystemExit, match="Frozen source mismatch"):
        launcher.main()
    assert imported == []
    assert not diag.exists()
