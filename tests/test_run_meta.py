import json
import threading

import pytest

from rag.telemetry import run_meta
from rag.telemetry.run_meta import (
    JsonlWriter,
    collect_run_metadata,
    config_hash,
    create_run_dir,
    write_run_files,
)


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": {"c": 2}}) == config_hash({"b": {"c": 2}, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_run_dir_and_files_are_never_overwritten(tmp_path):
    d = create_run_dir("SYS-P", output_dir=tmp_path, run_id="r1")
    assert d == tmp_path / "SYS-P" / "r1"
    with pytest.raises(FileExistsError):
        create_run_dir("SYS-P", output_dir=tmp_path, run_id="r1")
    write_run_files(d, {"seed": 42}, {"x": 1})
    assert json.loads((d / "run_meta.json").read_text(encoding="utf-8")) == {"x": 1}
    with pytest.raises(FileExistsError):
        write_run_files(d, {"seed": 1}, {"x": 2})


def test_metadata_without_gpu_and_secrets_hidden(monkeypatch):
    monkeypatch.setattr(run_meta.shutil, "which", lambda name: None)
    monkeypatch.setenv("VLLM_API_KEY", "super-secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    class Eng:
        def describe(self):
            return {"engine": "fake"}

    meta = collect_run_metadata({"experiment": "SYS-P", "seed": 42}, engines={"main": Eng()})
    assert meta["gpu"] == {"available": False, "reason": "nvidia-smi not found"}
    assert meta["env"]["VLLM_API_KEY"] == "<set>"
    assert meta["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert "super-secret" not in json.dumps(meta, default=str)
    assert meta["engines"]["main"] == {"engine": "fake"}
    assert meta["seed"] == 42 and len(meta["config_hash"]) == 16
    assert "commit" in meta["git"]


def test_jsonl_writer_is_thread_safe(tmp_path):
    w = JsonlWriter(tmp_path / "q.jsonl")
    threads = [
        threading.Thread(target=lambda i=i: [w.write({"i": i, "j": j}) for j in range(50)])
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [json.loads(x) for x in (tmp_path / "q.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 400
