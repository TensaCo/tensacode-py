"""Inspect one caller-supplied image with an explicit multimodal model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import mimetypes
import os
from pathlib import Path
from typing import Any

from tensorcode.integrations import LocalModel, OpenAICompatibleModel
from tensorcode.ops import llm
from tensorcode.tools.agents import Chatbot


@dataclass(frozen=True)
class InspectionResult:
    answer: str
    source_ref: str
    media_type: str


def inspect_image(
    image_path: str | Path,
    question: str,
    *,
    model: Any,
    detail: str | None = "auto",
) -> InspectionResult:
    """Send real image bytes and a question through the public chatbot path."""

    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"image path is not a regular file: {path}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty")
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None or not media_type.startswith("image/"):
        raise ValueError(f"could not infer an image MIME type from {path.name!r}")

    resolved = path.resolve()
    source_ref = f"file:{resolved}"
    encode_image = llm.ImageEncoder(
        media_type=media_type,
        source_ref=source_ref,
        detail=detail,
    )
    bot = Chatbot(model=model, encode_image=encode_image)
    answer = bot(question.strip(), images=(resolved.read_bytes(),))
    return InspectionResult(answer, source_ref, media_type)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ask an explicit local or OpenAI-compatible multimodal model about "
            "one image. The answer is model output, not independently verified grounding."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("question")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local-model", help="Hugging Face model ID or local directory")
    mode.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--model", help="Remote provider model name")
    parser.add_argument("--revision", help="Exact local model revision")
    parser.add_argument("--device", default="cpu", help="Local torch device")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the local Transformers loader to download missing files",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--api", choices=("chat_completions", "responses"), default="chat_completions"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--detail", choices=("low", "high", "auto"), default="auto")
    return parser


def _configured_model(args):
    if args.local_model:
        if args.model is not None:
            raise ValueError("--model is only valid with --base-url")
        return LocalModel.from_pretrained(
            args.local_model,
            revision=args.revision,
            local_files_only=not args.allow_download,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    if not args.model:
        raise ValueError("--model is required with --base-url")
    if args.revision is not None:
        raise ValueError("--revision is only valid with --local-model")
    return OpenAICompatibleModel(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get(args.api_key_env),
        timeout=args.timeout,
        api=args.api,
    )


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        model = _configured_model(args)
        result = inspect_image(
            args.image,
            args.question,
            model=model,
            detail=args.detail,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
