# Путь к файлу, который находится в той же папке, можно указать напрямую:
file_path = "dataset_neurotics.txt"

# Инициализируем списки:
labels = []
texts = []

# Читаем файл построчно
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # Удаляем пробелы и переносы строк
        if not line:
            continue  # Пропускаем пустые строки
        try:
            label, text = line.split(" - ", 1)     # Разделяем по ' - '
            labels.append(int(label.strip()))                   # Преобразуем лейбл в int
            texts.append(text.strip())               # Убираем кавычки у текста .strip('"')
        except ValueError:
            print(f"⚠️ Пропущена строка из-за формата: {line}")

    print(len(labels), labels, sep="\n")
    print(len(texts), texts, sep="\n")
