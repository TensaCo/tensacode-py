"""Verify the bounded MPS CUDA context, record it, then exec the reviewed job."""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-sms", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.physical_sms <= 0:
        parser.error(
            "A reviewed command and positive physical SM inventory are required"
        )
    properties = torch.cuda.get_device_properties(0)
    fraction = properties.multi_processor_count / args.physical_sms
    receipt = {
        "time": time.time(),
        "device": properties.name,
        "visible_multiprocessors": properties.multi_processor_count,
        "physical_multiprocessors_inventory": args.physical_sms,
        "visible_sm_fraction": fraction,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mps_pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "mps_requested_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        "affinity": sorted(os.sched_getaffinity(0)),
        "scope": "Fresh CUDA context in the same bounded unit and inherited environment as the exec target.",
    }
    if fraction > 0.8 or not receipt["mps_pipe"]:
        receipt["verified"] = False
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        raise RuntimeError("CUDA context exceeds the allowed equivalent-SM execution allocation")
    receipt["cuda_operation_result"] = torch.ones(4, device="cuda").sum().item()
    receipt["verified"] = True
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
