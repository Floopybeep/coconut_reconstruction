#!/usr/bin/env python3
"""Evaluate Qwen2.5-1.5B on final-answer-only math datasets.

The expected dataset format is a JSON list of objects with at least:

    {"question": "...", "answer": "..."}

Intermediate steps, if present, are ignored. The model is expected to put the
final answer in ``\\boxed{...}``.

The following commands were used to evaluate the model.
python ablations/infer_qwen.py --batch-size 64 --num-shots 0 --temperature 0.0 --seed 123
python ablations/infer_qwen.py --batch-size 64 --num-shots 1 --temperature 0.0 --seed 123
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = [
    "gsm=data/gsm_test.json",
    "svamp=data/svamp_test.json",
    "multiarith=data/multiarith_train_test.json",
]

DEFAULT_FEW_SHOT_PATHS = [
    "data/gsm_valid.json",
    "data/svamp_train.json",
    "data/multiarith_train_test.json"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct-answer Qwen2.5-1.5B evaluation on math datasets."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=(
            "Dataset paths, optionally as name=path. Defaults to GSM8K, SVAMP, "
            "and MultiArith test-style files."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="ablations/results/qwen",
        help="Directory for summary JSON and per-sample JSONL outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--num-shots",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use zero-shot or one-shot direct-answer prompting.",
    )
    parser.add_argument(
        "--few-shot-path",
        default="data/gsm_train.json",
        help="Dataset used to build the one-shot example.",
    )
    parser.add_argument("--few-shot-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of examples per dataset.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype. auto uses bfloat16 on CUDA when available.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed to the model. Use 'auto', 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 uses greedy decoding; positive values enable sampling.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional transformers attention implementation, e.g. sdpa.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model/tokenizer files already present locally.",
    )
    parser.add_argument(
        "--no-save-generations",
        action="store_true",
        help="Skip writing per-example JSONL files.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_dataset_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
    else:
        path = spec
        name = Path(path).stem
    if not name:
        raise ValueError(f"Dataset name is empty in spec: {spec!r}")
    return name, Path(path)


def load_json_dataset(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path} row {idx} is not a JSON object")
        if "question" not in row or "answer" not in row:
            raise ValueError(f"{path} row {idx} must have 'question' and 'answer'")
        rows.append(
            {
                "idx": idx,
                "question": str(row["question"]).strip(),
                "answer": str(row["answer"]).replace(",", "").strip(),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_few_shot_prefix(path: Path, index: int) -> str:
    examples = load_json_dataset(path)
    if not examples:
        raise ValueError(f"No examples found in few-shot path: {path}")
    if index < 0 or index >= len(examples):
        raise ValueError(
            f"--few-shot-index {index} is out of range for {path} "
            f"(size={len(examples)})"
        )
    example = examples[index]
    return f"Example: {example['question']}\n\\boxed{{{example['answer']}}}\n\n"


def build_prompt(question: str, few_shot_prefix: str = "", system_prompt: str = "") -> str:
    return f"{system_prompt}{few_shot_prefix}{question}\n"


def choose_torch_dtype(dtype: str) -> Any:
    import torch

    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def choose_device(device: str) -> Any:
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize_text_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(answer|final answer)\s*[:=]\s*", "", text, flags=re.I)
    text = text.replace(",", "")
    text = text.strip()
    text = text.rstrip(".")
    return re.sub(r"\s+", " ", text)


def maybe_decimal(text: str) -> Decimal | None:
    cleaned = normalize_text_answer(text)
    cleaned = cleaned.replace("$", "").replace("%", "")
    if re.fullmatch(r"[-+]?\d+/\d+", cleaned):
        numerator, denominator = cleaned.split("/", 1)
        denominator_decimal = Decimal(denominator)
        if denominator_decimal == 0:
            return None
        return Decimal(numerator) / denominator_decimal
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


def answers_match(predicted: str, expected: str, numeric_tol: Decimal) -> bool:
    predicted_norm = normalize_text_answer(predicted)
    expected_norm = normalize_text_answer(expected)

    if predicted_norm == expected_norm:
        return True
    if predicted_norm.lower() == expected_norm.lower():
        return True

    predicted_decimal = maybe_decimal(predicted_norm)
    expected_decimal = maybe_decimal(expected_norm)
    if predicted_decimal is None or expected_decimal is None:
        return False
    return abs(predicted_decimal - expected_decimal) <= numeric_tol


def extract_boxed_answer(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start == -1:
        return None

    content_start = start + len(r"\boxed{")
    depth = 1
    for pos in range(content_start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:pos]
    return None


def extract_answer(generated_suffix: str, _full_text: str) -> str:
    boxed = extract_boxed_answer(generated_suffix)
    if boxed is not None:
        return normalize_text_answer(boxed)

    text = generated_suffix
    text = text.replace("<|endoftext|>", "").strip()
    text = re.split(r"\n\s*(?:Question|Q)\s*[:#]", text, maxsplit=1)[0]
    text = text.splitlines()[-1] if text.splitlines() else text
    return normalize_text_answer(text)


def count_generated_tokens(token_ids, eos_token_id: int | None, pad_token_id: int | None) -> int:
    count = 0
    for token_id in token_ids.tolist():
        if pad_token_id is not None and token_id == pad_token_id:
            continue
        if eos_token_id is not None and token_id == eos_token_id:
            break
        count += 1
    return count


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_model_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = choose_torch_dtype(args.dtype)
    device = choose_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        padding_side="left",
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def evaluate_dataset(
    *,
    name: str,
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    device: Any,
    args: argparse.Namespace,
    few_shot_prefix: str,
    output_path: Path | None,
) -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    system_prompt = "Solve the following math problem. You may show your reasoning. At the end, give the final answer on its own line in the exact format:\n\\boxed{answer}\n\n"
    # system_prompt = """
    # A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
    # The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
    # The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \boxed{...}.
    # Output only those two tagged blocks, in that order, nothing else.
    # i.e., <think>reasoning process here</think><answer>\boxed{...}</answer>
    # """

    correct = 0
    generated_token_total = 0
    numeric_tol = Decimal("1e-6")
    do_sample = args.temperature > 0

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_fp = output_path.open("w", encoding="utf-8")
    else:
        out_fp = None

    try:
        for batch_rows in tqdm(
            batched(rows, args.batch_size),
            total=math.ceil(len(rows) / args.batch_size),
            desc=name,
            dynamic_ncols=True,
        ):
            prompts = [
                build_prompt(row["question"], few_shot_prefix, system_prompt) for row in batch_rows
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            ).to(device)

            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p

            with torch.no_grad():
                generated = model.generate(**inputs, **generation_kwargs)

            prompt_width = inputs["input_ids"].shape[1]
            suffixes = tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
            )
            full_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            generated_token_counts = [
                count_generated_tokens(
                    token_ids,
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                )
                for token_ids in generated[:, prompt_width:]
            ]

            for row, suffix, full_text, generated_tokens in zip(
                batch_rows, suffixes, full_texts, generated_token_counts
            ):
                prediction = extract_answer(suffix, full_text)
                is_correct = answers_match(prediction, row["answer"], numeric_tol)
                correct += int(is_correct)
                generated_token_total += generated_tokens
                result = {
                    "dataset": name,
                    "idx": row["idx"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "correct": is_correct,
                    "generated_tokens": generated_tokens,
                    "generated": suffix.strip(),
                }
                if out_fp is not None:
                    out_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
    finally:
        if out_fp is not None:
            out_fp.close()


    total = len(rows)

    if name == "multiarith":
        correct -= 1
        total -= 1

    return {
        "dataset": name,
        "path": str(output_path) if output_path is not None else None,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "avg_generated_tokens": generated_token_total / total if total else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    run_name = dt.datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    few_shot_prefix = ""
    if args.num_shots == 1:
        few_shot_prefix = [
            build_few_shot_prefix(
            # Path(args.few_shot_path), args.few_shot_index
            Path(few_shot_path), 0
        ) for few_shot_path in DEFAULT_FEW_SHOT_PATHS]

    model, tokenizer, device = load_model_and_tokenizer(args)

    summaries = []
    for spec in args.datasets:
        name, path = parse_dataset_spec(spec)
        rows = load_json_dataset(path, limit=args.limit)
        generation_path = None if args.no_save_generations else run_dir / f"{name}.jsonl"
        summary = evaluate_dataset(
            name=name,
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args,
            few_shot_prefix=few_shot_prefix,
            output_path=generation_path,
        )
        summary["dataset_path"] = str(path)
        summaries.append(summary)
        print(
            f"{name}: {summary['correct']} / {summary['total']} = "
            f"{summary['accuracy']:.4f}; "
            f"avg generated tokens = {summary['avg_generated_tokens']:.2f}"
        )

    overall_correct = sum(item["correct"] for item in summaries)
    overall_total = sum(item["total"] for item in summaries)
    overall_generated_tokens = sum(
        item["avg_generated_tokens"] * item["total"] for item in summaries
    )
    summary_payload = {
        "model_id": args.model_id,
        "num_shots": args.num_shots,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall_correct / overall_total if overall_total else 0.0,
            "avg_generated_tokens": (
                overall_generated_tokens / overall_total if overall_total else 0.0
            ),
        },
        "datasets": summaries,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, ensure_ascii=False)

    print(
        f"overall: {overall_correct} / {overall_total} = "
        f"{summary_payload['overall']['accuracy']:.4f}; "
        f"avg generated tokens = "
        f"{summary_payload['overall']['avg_generated_tokens']:.2f}"
    )
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
