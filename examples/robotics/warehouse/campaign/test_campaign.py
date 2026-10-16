"""Safety-contract tests for the campaign scheduler (no workloads launched)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from .prepare import prepare
from .run import validate


class CampaignContracts(unittest.TestCase):
    def config(self):
        return {
            "cpus": [min(os.sched_getaffinity(0))],
            "memory_gib": 1,
            "deadline_utc": "2026-10-16T23:54:00+00:00",
            "jobs": [],
        }

    def test_rejects_ambiguous_deadline(self):
        config = self.config()
        config["deadline_utc"] = "2026-10-16T23:54:00"
        with self.assertRaises(ValueError):
            validate(config)

    def test_cuda_requires_private_owned_path(self):
        config = self.config()
        config["jobs"] = [{"id": "cuda", "command": ["python"], "backend": "cuda"}]
        with self.assertRaises(ValueError):
            validate(config)

    def test_duplicate_jobs_rejected(self):
        config = self.config()
        config["jobs"] = [{"id": "same", "command": ["python"]}] * 2
        with self.assertRaises(ValueError):
            validate(config)

    def test_paired_runs_share_host_and_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            for name in ("examples", "src"):
                (repo / name).mkdir(parents=True)
                (repo / name / "test.py").write_text("# frozen\n")
            robustness, collection = root / "robustness", root / "collection"
            robustness.mkdir()
            collection.mkdir()
            cases = [
                {"id": "a-on", "pair": "a"},
                {"id": "a-off", "pair": "a"},
                {"id": "b", "pair": "b"},
            ]
            (robustness / "protocol.json").write_text(json.dumps({"cases": cases}))
            (collection / "protocol.json").write_text(json.dumps({"cases": []}))
            output = root / "output"
            prepare(repo, output, robustness, collection)
            manifest = json.loads((output / "manifest.json").read_text())
            lane = next(lane for lane in manifest["lanes"] if "a-on" in lane)
            self.assertIn("a-off", lane)
            self.assertTrue((output / "source.zip").is_file())
            with self.assertRaises(FileExistsError):
                prepare(repo, output, robustness, collection)


if __name__ == "__main__":
    unittest.main()
