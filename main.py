
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


def _eval_prompt_input_ids(batch, device):
    input_ids = batch["input_ids"]
    labels = batch.get("labels")

    if labels is not None:
        answer_positions = (labels[0] != -100).nonzero(as_tuple=False)
        if len(answer_positions) > 0:
            input_ids = input_ids[:, : answer_positions[0].item()]

    return input_ids.to(device)


def _check_prompt_only_dataset(dataset, name):
    if len(dataset) == 0:
        return

    labels = dataset[0].get("labels")
    if labels is not None and any(label != -100 for label in labels):
        raise ValueError(
            f"{name} is expected to be prompt-only, but it contains answer labels."
        )


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
        saved_checkpoint = torch.load(
            configs.load_model_path,
            map_location="cpu",
            weights_only=False,
        )
        saved_weights = saved_checkpoint["model_state_dict"]
        model.load_state_dict(saved_weights, strict=False)
        del saved_checkpoint, saved_weights
        model.to(device=device, dtype=torch.bfloat16)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    if not configs.only_eval:
        for epoch in range(configs.resume_from_epoch, configs.num_epochs):
            if configs.coconut:
                stage = epoch // configs.epochs_per_stage
            elif configs.cot:
                stage = 0

            if configs.reset_optimizer and epoch % configs.epochs_per_stage == 0:
                optimizer = bnb.optim.Adam8bit(
                    model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay
                )

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
                dataset_mode="validation" if configs.coconut else "test",
            )
            if configs.single_sample:
                val_dataset = _single_sample_subset(
                    val_dataset,
                    int(configs.single_sample_index),
                    1,
                )
            _check_prompt_only_dataset(val_dataset, "Validation dataset")
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
                    print(tokenizer.decode(train_dataset[0]["input_ids"]))
            print(f"Loaded validation dataset: {len(val_dataset)} samples")

            path_log = os.path.join(save_dir, f"log_eval_{epoch}.txt")

            train_steps = 0
            total_len = (
                len(train_dataloader) + configs.gradient_accumulation_steps - 1
            ) // configs.gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Train Epoch {epoch+1}", total=total_len)

            # Training step
            model.train()

            total_valid_tokens = 0
            loss = 0
            for step, batch in enumerate(train_dataloader):
                train_steps += 1

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask", "position_ids", "labels"}
                }
                outputs = model(**batch)

                total_valid_tokens += (batch["labels"][..., 1:] != -100).sum().item()

                loss += outputs.loss 
                display_loss = loss.detach().float().item() / total_valid_tokens
                pbar.set_postfix(loss=f"{display_loss:.4f}")

                del batch, outputs

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step + 1 == len(train_dataloader):
                    loss = loss / total_valid_tokens
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                    del loss
                    loss = 0
                    total_valid_tokens = 0
                
                if not configs.only_eval and not configs.debug:
                    wandb_run.log({
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": display_loss
                    })
            
            if not configs.save_only_improve and not configs.debug and not configs.only_eval:
                payload = {
                    "model_state_dict": model.state_dict(),
                    "configs": configs,
                }
                ckpt_save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}.pt")
                torch.save(payload, ckpt_save_path)
                print(f"\nModel saved for epoch {epoch + 1}")

            # train_iter = iter(train_dataloader)
            # for update_step in range(total_len):
            #     accum_batches = []
            #     total_valid_tokens = 0

            #     for _ in range(configs.gradient_accumulation_steps):
            #         try:
            #             batch = next(train_iter)
            #         except StopIteration:
            #             break

            #         total_valid_tokens += (
            #             batch["labels"][..., 1:] != -100
            #         ).sum().item()
            #         accum_batches.append(batch)
            #         train_steps += 1

            #     if not accum_batches:
            #         break
            #     if total_valid_tokens == 0:
            #         raise ValueError(
            #             "No valid target tokens found in the accumulation window."
            #         )

            #     optimizer.zero_grad(set_to_none=True)
            #     loss_sum_for_display = 0.0

            #     for batch in accum_batches:
            #         batch = {
            #             key: value.to(device)
            #             for key, value in batch.items()
            #             if key in {"input_ids", "attention_mask", "position_ids", "labels"}
            #         }
            #         outputs = model(**batch)
            #         loss_sum_for_display += outputs.loss.detach().float().item()
            #         loss = outputs.loss / total_valid_tokens
            #         loss.backward()

            #         del outputs, loss, batch

            #     optimizer.step()
            #     optimizer.zero_grad(set_to_none=True)
            #     pbar.update(1)

            #     display_loss = loss_sum_for_display / total_valid_tokens
            #     pbar.set_postfix(loss=f"{display_loss:.4f}")
                
            #     if not configs.only_eval and not configs.debug:
            #         wandb_run.log({
            #             "train/epoch": epoch + 1,
            #             "train/step": epoch * total_len + update_step,
            #             "train/loss": display_loss
            #         })
            
            # if not configs.save_only_improve and not configs.debug and not configs.only_eval:
            #     payload = {
            #         "model_state_dict": model.state_dict(),
            #         "configs": configs,
            #     }
            #     ckpt_save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}.pt")
            #     torch.save(payload, ckpt_save_path)
            #     print(f"\nModel saved for epoch {epoch + 1}")
            
            # Eval step
            with torch.no_grad():
                model.eval()
                correct = 0

                for idx, batch in tqdm(enumerate(val_dataloader), colour="blue", desc=f"Test accuracy for epoch {epoch + 1}"):
                    gt_answer = batch["answer"][0].split("#")[-1].strip()
                    eval_input_ids = _eval_prompt_input_ids(batch, device)

                    output_tokens = model.generate(eval_input_ids, max_new_tokens=configs.max_new_tokens)
                    output_text = tokenizer.decode(output_tokens)
                    output_extracted = output_text.split("#")[-1].replace("<|im_end|>", "").strip()

                    with open(path_log, "a") as f:
                        f.write(f"Question # {idx + 1}\n")
                        if configs.single_sample:
                            f.write(f"Question:\t\t{tokenizer.decode(batch['input_ids'][0])}\n")
                        f.write(f"GT output:\t\t{gt_answer}\n")
                        f.write(f"Model output:\t{output_extracted}\n")
                        if configs.single_sample:
                            f.write(f"Full output:\t{output_text}\n")
                    
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

    # Test step
    test_split = getattr(configs, "test_split", "test")
    if configs.single_sample:
        test_split = getattr(configs, "train_split", "train")

    test_dataset = LatentDataset.from_config(
        configs=configs,
        tokenizer=tokenizer,
        latent_id=latent_id,
        start_id=start_id,
        end_id=end_id,
        stage=configs.max_latent_stage,
        split=test_split,
        max_size=max_dataset_size,
        dataset_mode="test",
    )
    if configs.single_sample:
        test_dataset = _single_sample_subset(test_dataset, int(configs.single_sample_index), 1
                                             )
    _check_prompt_only_dataset(test_dataset, "Test dataset")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=8, batch_size=1, 
                                                        shuffle=False, collate_fn=data_collator)
    
    path_log = os.path.join(save_dir, f"log_test.txt")

    with torch.no_grad():
        model.eval()
        correct = 0

        for idx, batch in enumerate(tqdm(test_dataloader, colour="blue", desc=f"Test accuracy")):
            gt_answer = batch["answer"][0].split("#")[-1].strip()
            eval_input_ids = _eval_prompt_input_ids(batch, device)

            print(f"{tokenizer.decode(eval_input_ids)}")

            output_tokens = model.generate(eval_input_ids, max_new_tokens=configs.max_new_tokens)
            output_text = tokenizer.decode(output_tokens)
            output_extracted = output_text.split("#")[-1].replace("<|im_end|>", "").strip()

            with open(path_log, "a") as f:
                f.write(f"Question # {idx + 1}\n")
                f.write(f"Question:\t\t{tokenizer.decode(batch['input_ids'][0])}\n")
                f.write(f"GT output:\t\t{gt_answer}\n")
                f.write(f"Model output:\t{output_extracted}\n")
                f.write(f"Full output:\t{output_text}\n")
            
            correct += output_extracted == gt_answer
        
        eval_acc = correct / len(test_dataloader)
        print(f"Accuracy on test set: {eval_acc:.2%}  ({correct} / {len(test_dataloader)})")

        with open(path_log, "a") as f:
            f.write(f"\n\nTotal accuracy: {eval_acc:.2%}")


if __name__ == "__main__":
    main()
