from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, TaskType
from datasets import Dataset
import pandas as pd
import torch

# === Load New Data ===
df2 = pd.read_csv("training_dataset.csv")
combined_df = pd.concat([df2]).reset_index(drop=True)
dataset = Dataset.from_pandas(combined_df)
dataset = dataset.shuffle(seed=42)

# === Load Tokenizer ===
model_id = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map={"": 0},
    torch_dtype=torch.float16
)

base_model = prepare_model_for_kbit_training(base_model)
model = PeftModel.from_pretrained(base_model, "./lora-openhermes-output", is_trainable=True)

# === Tokenization ===
def tokenize_function(batch):
    combined_texts = [p + c for p, c in zip(batch["prompt"], batch["completion"])]
    tokenized = tokenizer(
        combined_texts,
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# === Data Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./finetuned-openhermes-rpg-contd",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    warmup_steps=50,
    max_steps=71406,
    bf16=True,
    fp16=False,
    optim="adamw_torch",
    report_to="none"
)


# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Train ===
trainer.train()

# === Save Final Model ===
model.save_pretrained("./lora-openhermes-output")
tokenizer.save_pretrained("./lora-openhermes-output")
