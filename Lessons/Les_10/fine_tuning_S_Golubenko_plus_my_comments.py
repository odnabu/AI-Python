
""" __ NB! __ """       # Video 10, 45:40.
# ОПИСАНИЕ задачи: тренировка модели, чтобы она распознавала какой отзыв: позитивный или негативный.
# При этом есть предобученная модель, которую мы дообучаем на специфическом датасете - отзывах к фильмам.


from transformers import (
    AutoModelForSequenceClassification,     # Модель, предобученная для КЛАССИФИКАЦИИ данных.
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,       # Модель, предобученyая для LLM.
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

# ______  1. Загрузка модели и токенизатора  _________________________________________
# Импорт ПРЕДОБУЧЕННОЙ модели:
model_name = "distilbert-base-uncased"      # Очень быстрая и маленькая модель, чтобы можно было запустить и попробовать локально.
# Далее модель прокидыватся в метод  .from_pretrained() для ПОЛУЧЕНИЯ предобученной версии этой модели:
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# Токеназер нужен для создания токенов из ИНПУТА:
tokenizer = AutoTokenizer.from_pretrained(model_name)   # нужен В СВЯЗКЕ с моделью, потому что
# модель имеет свои способы разбиения на токены.


# ______  2. Подготовка датасета  ___________________________________________________________________
data = {
    "text": [
        "I love this movie", "This is bad"
    ],
    "label": [1, 0]}
# label - ответы: 1 - хороший отзыв, 0 - плохой отзыв.
dataset = Dataset.from_dict(data)
# Video 10, 40:50 - как подгрузить другой датасет.

# ______  3. Токенизация  ___________________________________________________________________________
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")
# truncation - очистка от ЛИШНИХ пробелов.

# Команда для применения токенизации, определенной в tokenize(), на ВСЕ инпуты:
tokenized_dataset = dataset.map(tokenize, batched=True)

# ______  4. Аргументы тренировки (обучения)  _______________________________________________________
training_args = TrainingArguments(
    output_dir="Lessons/Les_10/results",    # Выходная папка для результатов.
    per_device_train_batch_size=4,          # Для распределения между процессорами или видеокартами, те каких размеров будет батч для каждого девайса.
    num_train_epochs=2,                     # Эпоха = 1 цикл обучения: ввод инпутов -->
                                            # --> бэкпропогашн - ошибки назад в нейросеть для корректировки ВСЕОВ на всех нейронах.
    logging_dir="Lessons/Les_10/logs",      # Папка для хранения логов от обучения.
    logging_steps=10                        # Как часто буде проходить логирование.
)

# ______  5. Обучение  ______________________________________________________________________________
# Класс, ответственный за обучение:
trainer = Trainer(
    model=model,                        # Ему передаются: - модель
    args=training_args,                 #                 - тренировочные аргументы (параметры).
    train_dataset=tokenized_dataset     # Передача ПОДГОТОВЛЕННОГО тренировочного датасета.
)
trainer.train()

# ______  6. Предсказание на новом примере  __________________________________________________________
# Для проверки ДОобученной модели:
text = "This movie was fantastic!"      # Вводим текст, которым хотим проверить качество ДОобученно модели.
inputs = tokenizer(text, return_tensors="pt")       # Токенизация нового инпута.

# ______  7. Перемещаем входные данные на то же устройство, где модель  _______________________________
inputs = {k: v.to(model.device) for k, v in inputs.items()}     # \\\\\\  NB! - доработка кода //////

# ______  8. Делаем вывод  ____________________________________________________________________________
model.eval()
with torch.no_grad():
    output = model(**inputs)
    pred = output.logits.argmax(dim=-1).item()      # .logits - функция, которая выбрасывает значение от 0 до 1.
    print("Positive" if pred == 1 else "Negative")

