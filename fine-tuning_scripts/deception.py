import os
os.environ["WANDB_MODE"] = "disabled"
from datasets import Dataset, load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import json
with open("data.json") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)
print(dataset[0])
model_name = "google/flan-t5-small" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
def preprocess(example):
    return tokenizer(
        example["input"],
        text_target=example["output"],  
        truncation=True,
        max_length=512,
        padding="max_length"          
    )

tokenized_dataset = dataset.map(preprocess, batched=True)


split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./truth_weaver",
    eval_strategy="steps",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=20,
    predict_with_generate=True,   
    push_to_hub=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
trainer.train()
trainer.save_model("./truth_weaver_model")
tokenizer.save_pretrained("./truth_weaver_model")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256)
decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_text)
