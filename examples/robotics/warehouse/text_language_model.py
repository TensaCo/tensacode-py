"""Reusable pinned, cache-only text model for proposal generation.

No download, training, robot execution, or semantic admission happens here.
Applications must validate and independently certify each proposal, then ground
it against fresh sensor evidence. CUDA requires an externally allocated lane.
"""

import hashlib
import time
from pathlib import Path

from .model_language import ModelProposal

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


class PinnedTextLanguageModel:
    def __init__(self, device="cpu", threads=2):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Only explicitly allocated CPU/CUDA devices are supported")
        if type(threads) is not int or not 1 <= threads <= 2:
            raise ValueError("Text adapter is bounded to one or two CPU threads")
        source = Path(__file__)
        self._source_hashes = {
            "adapter_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "proposal_schema_source_sha256": hashlib.sha256(
                source.with_name("model_language.py").read_bytes()
            ).hexdigest(),
        }
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            .to(device)
            .eval()
        )

    def propose(self, instruction, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if inputs.input_ids.shape[1] > 4096:
            raise ValueError("Prompt exceeds 4096 token bound")
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=384,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs.input_ids.shape[1] :]
        return ModelProposal(
            self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            int(generated.numel()),
            time.monotonic() - started,
            hashlib.sha256(text.encode()).hexdigest(),
        )

    def provenance(self):
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "device": self.device,
            "dtype": "bfloat16",
            "torch": self.torch.__version__,
            "local_files_only": True,
            **self._source_hashes,
        }
