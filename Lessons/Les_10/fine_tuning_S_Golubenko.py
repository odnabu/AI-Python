from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)
from datasets import Dataset
import torch

# Установить библиотеки:
# pip install datasets
# pip install transformers
# pip install torch             --- Нужна библиотека под МОЮ Видеокарту  |==>

# ==> С сайта установить CUDA 12.6 для моей видеокарты RTX4050 (объем 2,5 Гб).
# _____ Installing previous versions of PyTorch: https://pytorch.org/get-started/previous-versions/
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 1. Загрузка модели и токенизатор:
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Подготовка датасета:
data = {"text": ["I love this movie", "This is bad"], "label": [1, 0]}
dataset = Dataset.from_dict(data)

# 3. Токенизация:
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)

# 4. Аргументы тренировки:
training_args = TrainingArguments(
    output_dir="Lessons/Les_10/results",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_dir="Lessons/Les_10/logs",
    logging_steps=10
)

# 5. Обучение:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# 6. Предсказание на новом примере:
text = "This movie was fantastic!"
inputs = tokenizer(text, return_tensors="pt")

# 7. Перемещаем входные данные на то же устройство, где модель:
inputs = {k: v.to(model.device) for k, v in inputs.items()}     # \\\\\\  NB! - доработка кода //////

# 8. Делаем вывод:
model.eval()
with torch.no_grad():
    output = model(**inputs)
    pred = output.logits.argmax(dim=-1).item()
    print("Positive" if pred == 1 else "Negative")

