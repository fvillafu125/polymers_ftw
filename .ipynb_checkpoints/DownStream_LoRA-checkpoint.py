import torch
import yaml
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import LoraConfig, get_peft_model
from dataset import Downstream_Dataset
from PolymerSmilesTokenization import PolymerSmilesTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# Load config, data, tokenizer
finetune_config = yaml.load(open("config_finetune.yaml", "r"), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_properties = 1  # Set this to your number of regression targets

# Model and tokenizer
model = RobertaForSequenceClassification.from_pretrained(
    finetune_config['model_path'],
    num_labels=num_properties,
    problem_type="regression"
)
tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=finetune_config['blocksize'])

# LoRA setup
if finetune_config['lora_flag']:
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

model.to(device)

# Data
train_data = pd.read_csv(finetune_config['train_file'])
train_dataset = Downstream_Dataset(train_data, tokenizer, finetune_config['blocksize'])
# train_dataloader = DataLoader(train_dataset, batch_size=finetune_config['batch_size'], shuffle=True)

# You need a validation set for early stopping
val_data = pd.read_csv(finetune_config['test_file'])
val_dataset = Downstream_Dataset(val_data, tokenizer, finetune_config['blocksize'])

# TrainingArguments
training_args = TrainingArguments(
    output_dir=finetune_config['save_path'],
    num_train_epochs=finetune_config['num_epochs'],
    per_device_train_batch_size=finetune_config['batch_size'],
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=50,
    logging_dir=finetune_config['log_path'],
    fp16=True
)

# EarlyStoppingCallback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=finetune_config['tolerance']
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping]
)

# Train
trainer.train()

# Save model
trainer.save_model(finetune_config['save_path'])

# # Optimizer and loss
# optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config['lr_rate'])
# loss_fn = torch.nn.MSELoss()

# # Training loop
# for epoch in range(finetune_config['num_epochs']):
#     model.train()
#     for step, batch in enumerate(train_dataloader):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         targets = batch["prop"].to(device)
#         optimizer.zero_grad()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         loss = loss_fn(outputs.logits.squeeze(), targets.squeeze())
#         loss.backward()
#         optimizer.step()
#         if step % 100 == 0:
#             print(f"Epoch {epoch+1}, Step {step}: loss = {loss.item():.4f}")
#     print(f"Epoch {epoch+1} complete")

# # Save model
# torch.save(model.state_dict(), finetune_config['save_path'])