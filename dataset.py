
import re
import itertools
from dataclasses import dataclass
from typing import Any, Optional
from datasets import load_dataset

import torch
from torch.utils.data import Dataset


def _as_step_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(step) for step in value]
    return [str(value)]


_EQUATION_RE = re.compile(r"<<(.*?)>>", re.DOTALL)


def _extract_equation_steps(rationale: str) -> list[str]:
    return [
        f"<<{match.group(1).strip()}>>"
        for match in _EQUATION_RE.finditer(rationale)
        if match.group(1).strip()
    ]


def _extract_cot_steps(rationale: str) -> list[str]:
    equation_dot_count = sum(
        match.group(1).count(".") for match in _EQUATION_RE.finditer(rationale)
    )
    sentence_count = max(0, rationale.count(".") - equation_dot_count)

    rationale_without_equations = _EQUATION_RE.sub("", rationale)
    rationale_without_equations = re.sub(r"\s+", " ", rationale_without_equations)

    steps = [
        sentence.strip() + "."
        for sentence in re.split(r"\.(?=\s|$)", rationale_without_equations)
        if sentence.strip()
    ]
    if sentence_count > 0 and len(steps) > sentence_count:
        steps = steps[: sentence_count - 1] + [
            " ".join(step.rstrip(".") for step in steps[sentence_count - 1 :]) + "."
        ]

    if steps:
        return steps

    fallback = rationale_without_equations.strip()
    return [fallback] if fallback else []


def _split_gsm8k_answer(
    value: Any,
    step_extraction_method: str = "equations",
) -> tuple[list[str], str]:
    text = str(value)
    if "####" not in text:
        return [], text.strip()

    rationale, answer = text.rsplit("####", 1)
    if step_extraction_method in {"equation", "equations"}:
        steps = _extract_equation_steps(rationale)
    elif step_extraction_method in {"cot", "cot_sentences", "text"}:
        steps = _extract_cot_steps(rationale)
    else:
        raise ValueError(
            "`step_extraction_method` must be one of: equations, cot."
        )

    return steps, answer.strip()


def _load_hf_split(
    dataset_name: str,
    split: str,
    dataset_config_name: Optional[str] = None,
):

    if not dataset_name:
        raise ValueError("`dataset_name` is required. Load data from Hugging Face only.")

    if dataset_config_name:
        return load_dataset(dataset_name, dataset_config_name, split=split)
    return load_dataset(dataset_name, split=split)


class LatentDataset(Dataset):
    """
    Hugging Face backed Coconut-style dataset.

    Rows are normalized to question/steps/answer, then each reasoning step is
    replaced at construction time with `c_thought` copies of `latent_id`.
    With `c_thought == 0`, the original textual CoT steps are kept.
    """

    def __init__(
        self,
        tokenizer,
        latent_id: int,
        start_id: int,
        end_id: int,
        c_thought: int,
        stage: int,
        split: str = "train",
        dataset_name: str = "openai/gsm8k",
        dataset_config_name: Optional[str] = None,
        max_size: Optional[int] = None,
        question_column: str = "question",
        steps_column: str = "steps",
        answer_column: str = "answer",
        final_answer_column: Optional[str] = None,
        step_extraction_method: str = "equations",
        add_special_tokens: bool = True,
        single_sample: bool = False,
        prompt_only: bool = False,
        pad_latent_to_max: bool = True,
    ):
        self.tokenizer = tokenizer
        self.latent_id = latent_id
        self.start_id = start_id
        self.end_id = end_id
        self.c_thought = c_thought
        self.stage = stage
        self.add_special_tokens = add_special_tokens
        self.question_column = question_column
        self.steps_column = steps_column
        self.answer_column = answer_column
        self.final_answer_column = final_answer_column
        self.step_extraction_method = step_extraction_method
        self.single_sample = single_sample
        self.prompt_only = prompt_only
        self.pad_latent_to_max = pad_latent_to_max

        raw_dataset = _load_hf_split(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            split=split,
        )
        if max_size is not None:
            raw_dataset = raw_dataset.select(range(min(max_size, len(raw_dataset))))

        self.samples = [
            self._tokenize_sample(raw_sample, idx)
            for idx, raw_sample in enumerate(raw_dataset)
        ]

    @classmethod
    def from_config(
        cls,
        configs,
        tokenizer,
        latent_id: int,
        start_id: int,
        end_id: int,
        stage: int,
        split: str,
        max_size: Optional[int] = None,
        prompt_only: bool = False,
    ):
        return cls(
            tokenizer=tokenizer,
            latent_id=latent_id,
            start_id=start_id,
            end_id=end_id,
            c_thought=getattr(configs, "c_thought", 0),
            stage=stage,
            split=split,
            dataset_name=getattr(configs, "dataset_name", "openai/gsm8k"),
            dataset_config_name=getattr(configs, "dataset_config_name", None),
            max_size=max_size,
            question_column=getattr(configs, "question_column", "question"),
            steps_column=getattr(configs, "steps_column", "steps"),
            answer_column=getattr(configs, "answer_column", "answer"),
            final_answer_column=getattr(configs, "final_answer_column", None),
            step_extraction_method=getattr(
                configs, "step_extraction_method", "equations"
            ),
            add_special_tokens=getattr(
                configs,
                "add_special_tokens",
                not getattr(configs, "no_bot_tokens", False),
            ),
            single_sample=getattr(configs, "single_sample", False),
            prompt_only=prompt_only,
            pad_latent_to_max=getattr(configs, "pad_latent_to_max", True),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _normalize_sample(self, sample, idx: int) -> dict[str, Any]:
        if self.question_column not in sample:
            raise KeyError(f"Missing question column `{self.question_column}`.")
        if self.answer_column not in sample:
            raise KeyError(f"Missing answer column `{self.answer_column}`.")

        question = str(sample[self.question_column])
        raw_answer = sample[self.answer_column]

        if self.steps_column in sample and sample[self.steps_column] is not None:
            steps = _as_step_list(sample[self.steps_column])
            answer = (
                str(sample[self.final_answer_column])
                if self.final_answer_column
                else str(raw_answer)
            )
        else:
            steps, answer = _split_gsm8k_answer(
                raw_answer,
                step_extraction_method=self.step_extraction_method,
            )

        return {
            "question": question,
            "steps": steps,
            "answer": answer,
            "idx": sample["idx"] if "idx" in sample else idx,
        }

    def _tokenize_sample(self, sample, idx: int) -> dict[str, Any]:
        normalized = self._normalize_sample(sample, idx)

        question_tokenized = self.tokenizer.encode(
            normalized["question"] + "\n",
            add_special_tokens=True,
        )
        steps_tokenized = [
            self.tokenizer.encode(step + "\n", add_special_tokens=False)
            for step in normalized["steps"]
        ]
        answer_tokenized = self.tokenizer.encode(
            "### " + normalized["answer"],
            add_special_tokens=False,
        ) + [self.tokenizer.eos_token_id]

        if self.prompt_only:
            if self.single_sample:
                latent_count = self.c_thought
            elif self.pad_latent_to_max:
                latent_count = self.stage * self.c_thought
            else:
                latent_count = min(self.stage, len(steps_tokenized)) * self.c_thought

            latent_tokens = [self.latent_id] * latent_count
            if self.add_special_tokens:
                reasoning_tokens = [self.start_id] + latent_tokens + [self.end_id]
            else:
                reasoning_tokens = latent_tokens

            input_ids = question_tokenized + reasoning_tokens
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": [-100] * len(input_ids),
                "idx": normalized["idx"],
                "latent_tokens": latent_count,
                "answer": normalized["answer"],
                "question_length": len(question_tokenized),
            }

        if self.single_sample:
            latent_count = self.c_thought
            latent_tokens = [self.latent_id] * latent_count
            remaining_step_tokens = []
        else:
            n_replaced_steps = min(self.stage, len(steps_tokenized))
            latent_count = n_replaced_steps * self.c_thought
            latent_tokens = [self.latent_id] * latent_count
            remaining_step_tokens = list(
                itertools.chain.from_iterable(steps_tokenized[n_replaced_steps:])
            )

        if self.add_special_tokens:
            reasoning_tokens = (
                [self.start_id]
                + latent_tokens
                + [self.end_id]
                + remaining_step_tokens
            )
            masked_prefix_length = len(question_tokenized) + latent_count + 2
        else:
            reasoning_tokens = latent_tokens + remaining_step_tokens
            masked_prefix_length = len(question_tokenized) + latent_count

        input_ids = question_tokenized + reasoning_tokens + answer_tokenized
        labels = [-100] * masked_prefix_length + remaining_step_tokens + answer_tokenized

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "idx": normalized["idx"],
            "latent_tokens": latent_count,
            "answer": normalized["answer"],
            "question_length": len(question_tokenized),
        }


@dataclass
class LatentCollator:
    pad_token_id: int
    start_id: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        bot_positions = [
            feature["input_ids"].index(self.start_id)
            for feature in features
            if self.start_id is not None and self.start_id in feature["input_ids"]
        ]
        target_bot_position = max(bot_positions) if bot_positions else None

        aligned_features = []
        for feature in features:
            feature = dict(feature)
            left_pad = 0
            if (
                target_bot_position is not None
                and self.start_id in feature["input_ids"]
            ):
                left_pad = target_bot_position - feature["input_ids"].index(
                    self.start_id
                )

            if left_pad > 0:
                feature["input_ids"] = [self.pad_token_id] * left_pad + feature[
                    "input_ids"
                ]
                feature["attention_mask"] = [0] * left_pad + feature["attention_mask"]
                feature["labels"] = [self.label_pad_token_id] * left_pad + feature[
                    "labels"
                ]
            feature["question_length"] = feature["question_length"] + left_pad
            aligned_features.append(feature)

        max_length = max(len(feature["input_ids"]) for feature in aligned_features)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "position_ids": [],
            "idx": [],
            "latent_tokens": [],
            "answer": [],
            "question_length": [],
        }
        for feature in aligned_features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * pad_length
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * pad_length
            )
            batch["labels"].append(
                feature["labels"] + [self.label_pad_token_id] * pad_length
            )
            position_ids = []
            cur_position = 0
            for mask_value in batch["attention_mask"][-1]:
                if mask_value:
                    position_ids.append(cur_position)
                    cur_position += 1
                else:
                    position_ids.append(0)
            batch["position_ids"].append(position_ids)
            batch["idx"].append(feature["idx"])
            batch["latent_tokens"].append(feature["latent_tokens"])
            batch["answer"].append(feature["answer"])
            batch["question_length"].append(feature["question_length"])

        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "position_ids": torch.tensor(batch["position_ids"], dtype=torch.long),
            "idx": torch.tensor(batch["idx"], dtype=torch.long),
            "latent_tokens": torch.tensor(batch["latent_tokens"], dtype=torch.long),
            "answer": batch["answer"],
            "question_length": torch.tensor(batch["question_length"], dtype=torch.long),
        }
