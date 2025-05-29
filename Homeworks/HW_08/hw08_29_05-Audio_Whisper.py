# Müller Alexader
# \033[0;__;__m \033[m   or   \033[1;__;__m \033[m
# print('#' * 115)      # Для разделения блоков на листе с кодом:
""" ################################################################################################################
 28.05.25
 AI  &  Python HW_08: 8. Работа с мультимодальными моделями: изображения, звук, видео.
 ################################################################################################################### """

# ------------------------ SHORTCUTS ------------------------
# Ctrl + W - выделить текущий блок. если нажимать это сочетание дальше, то будут выделяться родительские блоки.
# Ctrl+Y - Удаление всей строки. Кстати, команда копирования Ctrl+C без выделения также работает для всей строки.
# Ctrl+Akt+L / Alt+Enter - Привести код к принятым стандартам (для Python - PEP8).
# Ctrl+R — Изменить название класса/функции и т. п. по всему проекту.
# Ctrl+Shift + F - Найти по всем файлам.
# Shift + F6 - заменить имя элемента во всех частях во всех файлах.
# -----------------------------------------------------------

print('.' * 60)


""" %%%%%%%%   Task 1   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
# Поэкспериментируйте самостоятельно с Whisper — попробуйте распознавать речь из разных аудиоисточников.


""" __ NB! __ """   # ___ Для запуска файла ввести в командной строке:
# python Homeworks/HW_08/hw08_29_05-Audio_Whisper.py --record 5 --model large --language ru --output Homeworks/HW_08/

l = 60      # Ограничение длины строки на экране.

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import torch                        # Библиотека для работы с нейронными сетями.
import whisper                      # Библиотека от OpenAI для распознавания речи.
import tempfile                     # Для создания временных файлов.
from datetime import datetime       # Для работы с датой и временем.
from pathlib import Path            # Для работы с путями файлов.
import argparse                     # Для разбора аргументов командной строки.
import sys                          # Для работы с системными функциями.
# ___ Модуль rich для красивого вывода в консоли: ____________________________________________
from rich import print              # Description: https://github.com/textualize/rich/blob/master/README.ru.md
                                    # Colors: https://rich.readthedocs.io/en/stable/appendix/colors.html
from rich.console import Console
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Активация библиотеки rich для красивой печати в Терминале:
console = Console()

# ___ ИНСТРУКЦИЯ по печати ___
# Печать цветного оформления в ТЕРМИНАЛЕ с помощью пакета rich:
# ИЛИ так:
# rprint(f'[red]{'*' * 60}[/red]\n{query_1[:-1]}:\n {vector_1}')
# ИЛИ так:
# console.print(f'[red]{'*' * 60}[/red]\n{query_1[:-1]}:\n {vector_1}')




def setup_whisper():
    """
    ___ УСТАНАВЛИВАЕТ и НАСТРАИВАЕТ  Whisper ___ если это необходимо.
    Возвращает доступное устройство (CUDA или CPU).
    CUDA - технология для использования видеокарты (GPU) для вычислений
    CPU - центральный процессор компьютера
    """
    # Проверяем, установлен ли whisper, если нет - устанавливаем его:
    try:
        import whisper                              # Пытаемся импортировать библиотеку.
    except ImportError:                             # Если библиотека не установлена, возникнет ошибка.
        print(f"[green]Installing OpenAI Whisper...[/green]")      # Сообщаем о начале установки.
        os.system("pip install -U openai-whisper")  # Запускаем команду установки через pip.
        try:
            import whisper                          # Пробуем снова импортировать после установки.
        except ImportError:                         # Если установка не удалась.
            print(f"[red]Failed to install whisper.[/red] Please install manually: ")  # Выводим сообщение о неудаче.
            print("pip install -U openai-whisper")  # Предлагаем установить вручную.
            sys.exit(1)                             # Завершаем программу с кодом ошибки 1.

    # Определяем устройство для вычислений:
    device = "cuda" if torch.cuda.is_available() else "cpu"     # Если доступен GPU, используем его, иначе CPU.
    print(f"[blue1]Using device: {device}[/blue1]")             # Выводим информацию об используемом устройстве.

    return device                                               # Возвращаем определенное устройство.


def load_model(model_size="tiny", device=None):
    """
    ___ ЗАГРУЖАЕТ модель Whisper ___
    Аргументы:
        model_size (str): Размер модели. Варианты: "tiny", "base", "small", "medium", "large"
        device (str): Устройство для использования (cuda или cpu)
    Возвращает:
        Загруженную модель
    Примечание: Чем больше модель, тем точнее распознавание, но требуется больше памяти и времени
    """
    print(f"[green4]Loading Whisper[/green4] [green1]{model_size}[/green1] [green4]model...[/green4]")      # Сообщаем о начале загрузки модели.
    model = whisper.load_model(model_size, device=device)               # Загружаем модель указанного размера.
    print(f"[green4]Model loaded successfully.\n{'':*<{l}}[/green4]")   # Сообщаем об успешной загрузке.
    return model                                                        # Возвращаем загруженную модель.


def transcribe_audio_file(model, audio_path, language=None):
    """
    ___ ТРАНСКРИБИРУЕТ ___ (преобразует речь в текст) аудиофайл с помощью Whisper.
    Аргументы:
        model: Модель Whisper
        audio_path (str): Путь к аудиофайлу
        language (str, опционально): Код языка (например, "en", "fr", "ja", "ru" и т.д.)
    Возвращает:
        dict: Результат транскрипции (словарь с текстом и доп. информацией)
    """
    print(f"\n[magenta1]Transcribing: {audio_path}[/magenta1]") # Сообщаем о начале транскрипции.

    # Устанавливаем параметры транскрипции:
    options = {}                                                # Создаем пустой словарь для параметров.
    if language:                                                # Если указан язык.
        options["language"] = language                          # Добавляем его в параметры.

    # Транскрибируем аудиофайл:
    result = model.transcribe(audio_path, **options)            # **options передает все параметры из словаря.
    return result               # Возвращаем результат транскрипции.


# ======  Код доработан  ============================================================================
def save_transcription(result, output_path):
    """
    ___ СОХРАНЯЕТ результат ТРАНСКРИПЦИИ ___  в .txt-файл.
    Аргументы:
        result (dict): Результат транскрипции от Whisper.
        output_file (str, опционально): Путь для сохранения результата.
    Возвращает:
        str: Путь к сохраненному файлу.
    """
    # Генерируем имя выходного файла, если оно не указано в ПУТИ:
    if os.path.isdir(output_path):                              # Если указана папка, формируем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # Создаем метку времени в формате ГодМесяцДень_ЧасМинутаСекунда.
        filename = f"transcription_{timestamp}.txt"             # Формируем имя файла с временной меткой.
        output_file = os.path.join(output_path, filename)
    else:
        output_file = output_path

    # Сохраняем транскрипцию:
    with open(output_file, "w", encoding="utf-8") as f:         # Открываем файл для записи с кодировкой UTF-8.
        f.write(result["text"])                                 # Записываем текст из результата в файл.

    print(f"\n[dark_magenta]Transcription saved to: {output_file}\n{'':*<{l}}[/dark_magenta]")   # Сообщаем о сохранении транскрипции.
    return output_file                                          # Возвращаем путь к сохраненному файлу.
# ===================================================================================================



def record_audio(duration=5, sample_rate=16000, channels=1):
    """
    ___ ЗАПИСЬ ___ аудио с микрофона.
    Аргументы:
        duration (int): Длительность записи в секундах
        sample_rate (int): Частота дискретизации (качество звука)
        channels (int): Количество каналов (1=моно, 2=стерео)
    Возвращает:
        str: Путь к записанному аудиофайлу
    """
    try:
        # Пробуем импортировать библиотеки для записи звука:
        import sounddevice as sd                                # Для записи аудио.
        import soundfile as sf                                  # Для сохранения аудио в файл.
    except ImportError:                                         # Если библиотеки не установлены.
        print("[green]Installing required packages for audio recording...[/green]")  # Сообщаем о начале установки.
        os.system("pip install sounddevice soundfile")          # Устанавливаем нужные пакеты.
        import sounddevice as sd                                # Импортируем после установки.
        import soundfile as sf                                  # Импортируем после установки.

    print(f"\n\t[bright_magenta]RECORDING audio for {duration} seconds...[/bright_magenta]")         # Сообщаем о начале записи.
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)  # Начинаем запись.
    sd.wait()                                                   # Ждем окончания записи.

    # Сохраняем запись во временный файл:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:  # Создаем временный файл с расширением .wav.
        temp_file = f.name                                      # Запоминаем имя временного файла.

    sf.write(temp_file, recording, sample_rate)                 # Записываем аудио во временный файл.
    print(f"[purple4]Audio recorded and saved to temporary file: {temp_file}\n{'':*<{l}}[/purple4]")  # Сообщаем о сохранении аудио.

    return temp_file                                            # Возвращаем путь к временному файлу.


def main():
    """
    ___ ОСНОВНАЯ функция программы ___
    Обрабатывает аргументы командной строки и управляет процессом транскрипции.
    """
    # Создаем парсер аргументов командной строки:
    parser = argparse.ArgumentParser(description="Whisper Speech Recognition Tool")

    # Добавляем аргументы командной строки:
    parser.add_argument("--file", type=str, help="Path to audio file for transcription")  # Путь к аудиофайлу.
    parser.add_argument("--record", type=int, default=0,
                        help="Record audio for specified seconds")              # Запись аудио на указанное количество секунд.
    parser.add_argument("--model", type=str, default="base",       # Размер модели (по умолчанию "base").
                        choices=["tiny", "base", "small", "medium", "large"],   # Доступные размеры моделей.
                        help="Whisper model size")
    parser.add_argument("--language", type=str, help="Language code (e.g., 'en', 'fr')")  # Код языка.
    parser.add_argument("--output", type=str, help="Output file path")  # Путь к выходному audio-файлу.

    args = parser.parse_args()                  # Разбираем аргументы командной строки.

    # Настройка:
    device = setup_whisper()                    # Устанавливаем и настраиваем Whisper.

    # Загружаем модель:
    model = load_model(args.model, device)      # Загружаем модель указанного размера.

    # Получаем путь к аудиофайлу:
    audio_path = None                           # Изначально путь к аудио не задан.
    if args.file:                               # Если указан аудиофайл.
        audio_path = args.file                  # Используем указанный файл.
    elif args.record > 0:                       # Если указано время записи.
        audio_path = record_audio(duration=args.record)             # Записываем аудио с микрофона.
    else:                                       # Если не указан ни файл, ни запись.
        print("No audio input specified. Use --file or --record")   # Сообщаем, что не указан источник аудио.
        print("Example: python whisper_demo.py --file audio.mp3")   # Пример использования с файлом.
        print("Example: python whisper_demo.py --record 10")        # Пример использования с записью.
        return                                  # Завершаем функцию.

    # Транскрибируем аудио:
    result = transcribe_audio_file(model, audio_path, args.language)  # Распознаем речь из аудио.

    # Выводим результат транскрипции:
    print(f"\n[magenta3]Transcription: [/magenta3]")    # Заголовок для транскрипции.
    print(f"\t{result["text"]}")                    # Выводим распознанный текст.

    # Сохраняем транскрипцию, если указан выходной файл:
    if args.output:                                 # Если указан путь для сохранения.
        save_transcription(result, args.output)     # Сохраняем транскрипцию в указанный файл.

    # Удаляем временный файл, если было выполнено аудио:
    if args.record > 0 and audio_path:              # Если была запись.
        os.unlink(audio_path)                       # Удаляем временный файл.
        print(f"\n[deep_sky_blue4]Temporary audio file deleted: {audio_path}[/deep_sky_blue4]")  # Сообщаем об удалении.


# Точка входа в программу:
if __name__ == "__main__":                          # Если этот файл запущен напрямую (не импортирован).
    main()                                          # Вызываем основную функцию.
    print(f"\nProgram completed.\n{'':.<{l}}")

""" %%%%%%%%   Task 2   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
# Подумайте о своих идеях для мультимодальных AI-приложений — какие задачи можно решить,
# комбинируя разные модальности?

# Смотри файл: Homeworks/HW_08/hw08_29_05-Audio_Whisper.md


""" %%%%%%%%   Task 3   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
# Поделитесь своими идеями!





