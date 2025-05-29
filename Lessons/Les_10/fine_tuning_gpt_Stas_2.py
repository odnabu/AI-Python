from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM
)
from datasets import Dataset
import torch
from datasets import load_dataset

# 1. Загрузка модели и токенизатора
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 2. Подготовка датасета

#dataset = load_dataset("wikitext", "wikitext-103-v1")
#data = {"text": ["I love this movie", "This is bad"], "label": [1, 0]}
examples = {
    "text": [
        "Question: What is the capital of France?",
        "Question: Who wrote '1984'?",
    ],
    "labels": ["Paris", "George Orwell"]
}
dataset = Dataset.from_dict(examples)

# 3. Токенизация
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
tokenized_dataset = dataset.map(tokenize, batched=True)

# 4. Аргументы тренировки
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=10
)

# 5. Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# 6. Предсказание на новом примере
text = "What is the capital city of germany?"
inputs = tokenizer(text, return_tensors="pt")

# перемещаем входные данные на то же устройство, где модель
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# делаем вывод
model.eval()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    #pred = output.logits.argmax(dim=-1).item()
    #print("Positive" if pred == 1 else "Negative")