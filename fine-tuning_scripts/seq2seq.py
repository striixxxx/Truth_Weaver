import os
os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
from google.colab import files
seq2seq_file = files.upload()
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

dataset = load_dataset("json", data_files={"train": list(seq2seq_file.keys())[0]})
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(batch):
    model_inputs = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=1024)
    labels = tokenizer(batch["output"], truncation=True, padding="max_length", max_length=768)
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    return model_inputs

encoded = dataset.map(preprocess, batched=True)
train_test = encoded["train"].train_test_split(test_size=0.2)

args = Seq2SeqTrainingArguments(
    output_dir="./seq2seq_model",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, 
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=True,
    save_total_limit=1
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained("./seq2seq_modelD")
tokenizer.save_pretrained("./seq2seq_modelD")

