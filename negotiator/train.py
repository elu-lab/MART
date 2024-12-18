import argparse
import os
import json
import torch
import time
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from huggingface_hub import login

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--huggingface_token', type=str, defalut='Your HuggingFace token')

    parser.add_argument('--train_data_fp', type=str, defalut='./data/negotiator_train.json')
    parser.add_argument('--test_data_fp', type=str, defalut='./data/negotiator_test.json')

    parser.add_argument('--base_model', type=str, defalut='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--checkpoint_dir', type=str, defalut='./llama-3.1-8b/checkpoints')
    parser.add_argument('--output_dir', type=str, defalut='./llama-3.1-8b/results')
    args = parser.parse_args()

    # Login to HuggingFace and W&B
    login(token=args.huggingface_token)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16,
    )

    data_files = {
        "train": args.train_data_fp,
        "test": args.test_data_fp
    }
    dataset = load_dataset("json", data_files=data_files)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    checkpoint_number = max([int(d.split('-')[-1]) for d in os.listdir(args.checkpoint_dir) if d.startswith('checkpoint')], default=0)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint-{checkpoint_number}")

    if os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, quantization_config=quant_config).to(device)
    else:
        print(f"Loading base model: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=quant_config).to(device)

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        warmup_steps=10,
        learning_rate=2e-4,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        push_to_hub=False,
        gradient_checkpointing=True,
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=0.2,
        per_device_eval_batch_size=8,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_params,
        args=training_params,
        train_dataset=train_dataset, 
        eval_dataset=test_dataset    
    )

    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(args.output_dir) 
    print(f"Model saved at {args.output_dir}")
