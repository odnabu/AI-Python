from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import Dataset
from transformers import TrainingArguments

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = {"text": ["I love this movie", "This is bad"], "label": [1, 0]}
dataset = Dataset.from_dict(data)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)


training_args = TrainingArguments(
    output_dir="Lessons/Les_10/results",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_dir="Lessons/Les_10/logs",
    logging_steps=10,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

text = "This movie was fantastic!"

inputs = tokenizer(text, return_tensors="pt")       # __ PROBLEM HER __

output = model(**inputs)
prediction = output.logits.argmax(dim=-1).item()
print("Positive" if prediction == 1 else "Negative")
