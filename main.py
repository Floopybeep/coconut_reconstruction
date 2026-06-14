
import os
import sys
import yaml
import wandb
import random
import datetime
import argparse

import torch
import torch.nn as nn

import numpy as np
import bitsandbytes as bnb
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from dataset import LatentCollator, LatentDataset
from model import Coconut


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def _single_sample_subset(dataset, sample_index, repeat_count):
    if len(dataset) == 0:
        raise ValueError("Cannot select a single sample from an empty dataset.")

    if sample_index < 0:
        sample_index += len(dataset)
    if sample_index < 0 or sample_index >= len(dataset):
        raise ValueError(
            f"single_sample_index={sample_index} is out of range for "
            f"dataset with {len(dataset)} samples."
        )

    return torch.utils.data.Subset(dataset, [sample_index] * repeat_count)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    # Parse args to configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filepath', help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)

    # Set up
    set_seed(configs.seed)
    configs.__dict__["start_time"] = current_time

    save_dir = os.path.join(configs.save_path, configs.run_name, current_time)
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.add_tokens("<bot>")
    tokenizer.add_tokens("<eot>")
    tokenizer.add_tokens("<latent>")
    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id = tokenizer.convert_tokens_to_ids("<bot>")
    end_id = tokenizer.convert_tokens_to_ids("<eot>")

    llm_model_config = AutoConfig.from_pretrained(configs.model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(configs.model_id, config=llm_model_config, torch_dtype=torch.bfloat16)
    llm_model.resize_token_embeddings(len(tokenizer))
    model = Coconut(llm_model, latent_id, start_id, end_id, tokenizer.eos_token_id, device)
    model.to(device=device, dtype=torch.bfloat16)

    # embeddings = model.get_input_embeddings()
    # for token_id in [latent_id, start_id, end_id]:


    # Load model configs if settings indicate so
    if configs.load_model_path != "None":
        saved_checkpoint = torch.load(configs.load_model_path, weights_only=False)
        saved_weights = saved_checkpoint["model_state_dict"]
        model.load_state_dict(saved_weights, strict=False)
        model.to(device=device, dtype=torch.bfloat16)

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss
    
    # Load dataset
    max_dataset_size = 32 if configs.debug else None
    dataset_name = getattr(configs, "dataset_name", "openai/gsm8k")
    data_collator = LatentCollator(
        pad_token_id=tokenizer.pad_token_id,
        start_id=start_id,
        label_pad_token_id=-100,
    )

    # If debug, train with 32 dataset samples, and 1 epoch per stage
    if configs.debug:
        configs.epochs_per_stage = 1

    if configs.single_sample:
        max_dataset_size = None
        configs.__dict__.setdefault("single_sample_index", 0)
        configs.__dict__.setdefault("single_sample_steps_per_epoch", 1)

    if configs.load_model_path is None:
        configs.resume_from_epoch = 0

    # Initialize wandb
    if not configs.only_eval and not configs.debug:
        wandb_run = wandb.init(project=configs.wandb_project, name=configs.run_name)
        wandb_run.config.update(configs, allow_val_change=True)

    # If CoT, train with stage = 0, add_special_tokens = False
    # If coconut, stage = epoch // 3, add_special_tokens = True
    best_eval_acc = 0.0
    for epoch in range(configs.resume_from_epoch, configs.num_epochs):
        if configs.coconut:
            stage = min(epoch // configs.epochs_per_stage, configs.max_latent_stage)
        elif configs.cot:
            stage = 0

        train_dataset = None
        if not configs.only_eval:
            base_train_dataset = LatentDataset.from_config(
                configs=configs,
                tokenizer=tokenizer,
                latent_id=latent_id,
                start_id=start_id,
                end_id=end_id,
                stage=stage,
                split=getattr(configs, "train_split", "train"),
                max_size=max_dataset_size,
            )
            shuffle_train = True
            if configs.single_sample:
                single_sample_steps = int(configs.single_sample_steps_per_epoch)
                if single_sample_steps <= 0:
                    raise ValueError("single_sample_steps_per_epoch must be positive.")
                repeat_count = (
                    configs.batch_size_training
                    * configs.gradient_accumulation_steps
                    * single_sample_steps
                )
                train_dataset = _single_sample_subset(
                    base_train_dataset,
                    int(configs.single_sample_index),
                    repeat_count,
                )
                shuffle_train = False
            else:
                train_dataset = base_train_dataset

            train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=configs.batch_size_training,
                                                           shuffle=shuffle_train, collate_fn=data_collator)

        val_split = getattr(
            configs,
            "val_split",
            "test" if dataset_name == "openai/gsm8k" else "validation",
        )
        if configs.single_sample:
            val_split = getattr(configs, "train_split", "train")

        val_dataset = LatentDataset.from_config(
            configs=configs,
            tokenizer=tokenizer,
            latent_id=latent_id,
            start_id=start_id,
            end_id=end_id,
            stage=stage,
            split=val_split,
            max_size=max_dataset_size,
        )
        if configs.single_sample:
            val_dataset = _single_sample_subset(
                val_dataset,
                int(configs.single_sample_index),
                1,
            )
        val_dataloader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=1, 
                                                           shuffle=False, collate_fn=data_collator)

        if train_dataset is not None:
            print(f"Loaded train dataset: {len(train_dataset)} samples")
            if configs.single_sample:
                print(
                    "Single-sample training: "
                    f"train split index {configs.single_sample_index}, "
                    f"{len(train_dataset)} copies, "
                    f"{configs.single_sample_steps_per_epoch} optimizer steps per epoch"
                )
        print(f"Loaded validation dataset: {len(val_dataset)} samples")

        path_log = os.path.join(save_dir, f"log_eval_{epoch}.txt")

        if not configs.only_eval:

            train_steps = 0
            total_len = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Train Epoch {epoch+1}", total=total_len)

            # Training step
            model.train()
            for step, batch in enumerate(train_dataloader):
                train_steps += 1

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask", "position_ids", "labels"}
                }
                outputs = model(**batch)
                loss = outputs.loss / configs.gradient_accumulation_steps
                display_loss = (
                    loss.detach().float().item()
                    * configs.gradient_accumulation_steps
                )
                pbar.set_postfix(loss=f"{display_loss:.4f}")
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                

                if not configs.only_eval and not configs.debug:
                    wandb_run.log({
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float() * configs.gradient_accumulation_steps
                    })
            
            if not configs.save_only_improve and not configs.debug and not configs.only_eval:
                payload = {
                    "model_state_dict": model.state_dict(),
                    "configs": configs,
                }
                ckpt_save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}.pt")
                torch.save(payload, ckpt_save_path)
                print(f"\nModel saved for epoch {epoch + 1}")
        
        # Eval step
        with torch.no_grad():
            model.eval()
            correct = 0

            for idx, batch in tqdm(enumerate(val_dataloader), colour="blue", desc=f"Test accuracy for epoch {epoch + 1}"):
                gt_answer = batch["answer"][0].split("#")[-1].strip()
                input_ids = batch["input_ids"].to(device)

                # Remove language tokens after <eot> to evaluate generation accuracy
                if configs.cot:
                    question_length = batch["question_length"][0].item()
                    eval_input_ids = input_ids[:, :question_length]
                else:
                    eot_positions = (input_ids[0] == end_id).nonzero(as_tuple=False)
                    if len(eot_positions) == 0:
                        raise ValueError(
                            "Cannot build eval prompt because <eot> is missing."
                        )
                    eval_input_ids = input_ids[:, : eot_positions[0].item() + 1]

                output_tokens = model.generate(eval_input_ids, max_new_tokens=configs.max_new_tokens)
                output_text = tokenizer.decode(output_tokens)
                output_extracted = output_text.split("#")[-1].replace("<|im_end|>", "").strip()

                with open(path_log, "a") as f:
                    f.write(f"Question # {idx + 1}\n")
                    f.write(f"GT output:\t\t{gt_answer}\n")
                    f.write(f"Model output:\t{output_extracted}\n")
                
                correct += output_extracted == gt_answer
            
            eval_acc = correct / len(val_dataloader)
            print(f"Accuracy on validation set for epoch {epoch + 1}: {eval_acc:.2%}  ({correct} / {len(val_dataloader)})")

            with open(path_log, "a") as f:
                f.write(f"\n\nTotal accuracy: {eval_acc:.2%}")

            if not configs.only_eval and not configs.debug:
                wandb_run.log({"eval/accuracy": eval_acc})

        if configs.save_only_improve and not configs.debug and eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            payload = {
                "model_state_dict": model.state_dict(),
                "configs": configs,
            }
            ckpt_save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}_{eval_acc:.4f}.pt")
            torch.save(payload, ckpt_save_path)
            print("\nModel saved after eval")

                

        

if __name__ == "__main__":
    main()
