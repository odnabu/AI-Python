from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch

# ______  1. Загрузка модели и токенизатора
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

# ______  2. Подготовка датасета
file_path = "dataset_neurotics.txt"
labels = []
texts = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            label, text = line.split(" - ", 1)
            labels.append(int(label.strip()))
            texts.append(text.strip())
        except ValueError:
            print(f"⚠️ Пропущена строка из-за формата: {line}")


data = {
    "text": texts,
    "label": labels}
dataset = Dataset.from_dict(data)

# ______  3. Токенизация
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)

# 🔹 Разделяем на обучающую и тестовую выборки:
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
test_dataset = train_test["test"]


# ______  4. Аргументы тренировки (обучения)
training_args = TrainingArguments(
    output_dir="./results_hw_10",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_dir="./logs_hw_10",
    logging_steps=10
)

# ______  5. Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset     # ⬅️ для оценки на тесте
)
trainer.train()

# ______  6. Предсказание на новом примере
text = "Да, я верю в астрологию"
inputs = tokenizer(text, return_tensors="pt")

# ______  7. Перемещаем входные данные на то же устройство, где модель
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# ______  8. Делаем вывод
model.eval()
with torch.no_grad():
    output = model(**inputs)
    pred = output.logits.argmax(dim=-1).item()
    label_map = {
        0: "Человек с нормальными личными границами.",
        1: "В целом человек способен отслеживать свои личные границы и обозначать их. Есть незначительные отклонения.",
        2: "Есть некритичные отклонения, в незначительной степени снижающие качество жизни самого человека и его окружающих. Готов обратиться к психологу.",
        3: "Человек осознает какие-то проблемы с собственными личными границами, но не понимает как их решать.",
        4: "Невротик с довольно высокой степенью расстройства. Даже не понимает, что у него проблемы."
    }
    print(label_map.get(pred, "Неизвестный класс"))
