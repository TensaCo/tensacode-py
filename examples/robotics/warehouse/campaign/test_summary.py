"""A finished process, a successful process, and physical evidence are distinct."""

import json
import tempfile
import unittest
from pathlib import Path

from examples.robotics.warehouse.campaign.summarize import summarize


class SummaryTests(unittest.TestCase):
    def test_runtime_error_and_interruption_are_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "physical"
            report.mkdir()
            (report / "probe.json").write_text(json.dumps({"success": False}))
            rows = []
            for index, (job, code, reason, output) in enumerate(
                [
                    ("physical_failure", 0, "process_exit", report),
                    ("runtime_error", 1, "process_exit", root / "missing"),
                    ("interrupted", 0, "signal", root / "stopped"),
                ]
            ):
                rows.extend(
                    [
                        {
                            "event": "started",
                            "job": job,
                            "unit": job,
                            "host": "fixture",
                            "time": index * 10,
                            "backend": "egl",
                            "command": ["run", "--output", str(output)],
                        },
                        {
                            "event": "finished",
                            "job": job,
                            "time": index * 10 + 1,
                            "exit_code": code,
                            "reason": reason,
                        },
                    ]
                )
            (root / "ledger.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            result = summarize([root])
            self.assertEqual(
                result["process_outcomes"],
                {
                    "finished": 3,
                    "succeeded": 1,
                    "nonzero_exit": 1,
                    "interrupted": 1,
                    "physical_reports_present": 1,
                    "independently_valid_physical_reports": None,
                },
            )
            self.assertFalse(result["jobs"][0]["reported_task_success"])
            self.assertIsNone(result["jobs"][1]["reported_task_success"])
            self.assertEqual(len(summarize([root], start_time=10)["jobs"]), 2)


if __name__ == "__main__":
    unittest.main()
