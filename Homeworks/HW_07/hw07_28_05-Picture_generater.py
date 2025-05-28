# Müller Alexader
# \033[0;__;__m \033[m   or   \033[1;__;__m \033[m
# print('#' * 115)      # Для разделения блоков на листе с кодом:
""" ################################################################################################################
 28.05.25
 AI  &  Python HW_07:  7. Введение в мультимодальные модели и работа с изображениями.
 ################################################################################################################### """

# ------------------------ SHORTCUTS ---------------------------------------------------------------------------
# Ctrl + W - выделить текущий блок. Если нажимать это сочетание дальше, то будут выделяться родительские блоки.
# Ctrl+Y - Удаление всей строки. Кстати, команда копирования Ctrl+C без выделения также работает для всей строки.
# Ctrl+Akt+L / Alt+Enter - Привести код к принятым стандартам (для Python - PEP8).
# Ctrl+R — Изменить название класса/функции и т.п. по всему проекту.
# Ctrl+Shift + F - Найти по всем файлам.
# Shift + F6 - заменить имя элемента во всех частях во всех файлах.
# --------------------------------------------------------------------------------------------------------------

print('.' * 70)



""" %%%%%%%%   Task 1   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
# Поэкспериментируйте самостоятельно со Stable Diffusion: создайте минимум 3 текстовых запроса (промптов)
# разной сложности и тематики и сгенерируйте изображения для каждого запроса.
# Прикрепите запросы с изображениями в LMS.

""" ___ Stable Diffusion ___ """
# See Les07-PyLLM_7 (1).pdf, sl. 13, 27.

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import requests                     # Библиотека для отправки HTTP-запросов.
import io                           # Модуль для работы с потоками ввода-вывода.
from PIL import Image               # Библиотека для работы с изображениями.
import os                           # Модуль для взаимодействия с операционной системой.
from dotenv import load_dotenv      # Библиотека для загрузки переменных окружения из файла .env.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def setup_env():
    """ Настройка окружения и проверка наличия токена Hugging Face. """
    load_dotenv()                           # Загружаем переменные окружения из файла .env.

    # Проверяем наличие токена Hugging Face:
    hf_token = os.getenv("HF_TOKEN")        # Получаем токен из переменных окружения.

    # ___  Если токен не найден, ЗАРЕГИСТРИРОВАТЬСЯ на huggingface.co по инструкции:  ___
    # https://huggingface.co/ --->
    # --->   TOKEN NAME: generate_pict_pyth_ai_les07, FROM: https://huggingface.co/settings/tokens
    if not hf_token:
        print("No Hugging Face token found. You need to:")
        print("1. Create a free account at huggingface.co")             # Инструкция: создать аккаунт.
        print("2. Get your token at huggingface.co/settings/tokens")    # Инструкция: получить токен.
        print("3. Create a .env file with HF_TOKEN=your_token_here")    # Инструкция: создать файл .env.
        hf_token = input("Or enter your Hugging Face token now: ")      # Возможность ввести токен вручную.

    return hf_token                         # Возвращаем токен для дальнейшего использования.


def generate_image(prompt, token, negative_prompt="", model_id="CompVis/stable-diffusion-v1-4",
                   num_inference_steps=30):
    """
    Генерация изображения с помощью Stable Diffusion через API Hugging Face
    Аргументы:
        prompt (str): Текстовый запрос для генерации изображения.
        token (str): Токен API Hugging Face.
        negative_prompt (str): Описание того, что следует избегать в сгенерированном изображении.
        model_id (str): Идентификатор модели на Hugging Face.
        num_inference_steps (int): Количество шагов денойзинга (больше = лучше качество, но медленнее).
    Возвращает:
        PIL.Image: Сгенерированное изображение.
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"     # URL модели на Hugging Face.
    headers = {
        "Authorization": f"Bearer {token}",         # Заголовок авторизации с токеном, те токен с авторизацией.
        "Content-Type": "application/json"          # Тип содержимого запроса.
    }

    payload = {                                             # Данные для отправки на сервер.
        "inputs": prompt,                                   # Текстовый запрос.
        "parameters": {                                     # Параметры генерации.
            "negative_prompt": negative_prompt,             # Что избегать в изображении.
            "num_inference_steps": num_inference_steps,     # Количество шагов генерации
        }
    }

    # Отправляем запрос к API Hugging Face:
    print(f"Generating image for prompt: '{prompt}'")
    response = requests.post(API_URL, headers=headers, json=payload)  # POST-запрос с данными в формате JSON.

    if response.status_code != 200:                         # Проверка успешности запроса.
        raise Exception(f"Error: {response.status_code}, {response.text}")  # Выбрасываем исключение при ошибке.

    # Преобразуем ответ в изображение:
    image = Image.open(io.BytesIO(response.content))        # Открываем изображение из бинарных данных ответа.
    return image                                            # Возвращаем изображение.


def save_image(image, filename="generated_image.png"):
    """Сохранение сгенерированного изображения в файл"""
    image.save(filename)                                    # Сохраняем изображение с указанным именем файла.
    print(f"Image saved as {filename}")                     # Выводим сообщение о сохранении.
    return filename                                         # Возвращаем имя файла.


def main():
    # Настройка окружения:
    token = setup_env()                                     # Получаем токен Hugging Face.

    # Генерируем изображения с разными запросами:
    prompts = [
        # Блок-схема алгоритма сортировки пузырьком:
        "Bubble Sort Algorithm:1.2 Flowchart:1.5.",
        # Девушка-программист, которая изучает Пайтон на продвинутом уровне и пытается реализовать учебный код
        # из урока для проекта во фреймворке Фласк. Учебный код содержит много ошибок, которые она не знает
        # как исправить отчего она расстраивается.
        "A female programmer who is learning Python:1.5 at an advanced level and is trying to implement the training "
        "code from the lesson for a project in the Flask:1.2 framework. The training code contains many errors "
        "that she does not know how to fix, which makes her upset.",
        # Черная кошка, спящая в коробке, застеленной серым одеялом. Коробка стоит на столе рядом с девушкой,
        # слушающей видеолекцию по созданию проекта на фреймворке Фласк.
        "A black cat sleeping in a box covered with a gray blanket. The box is on a table next to a female programmer "
        "watching to a video lecture on creating a project on the Flask:1.2 framework."
    ]

    # \\\\\\\\\\\\\\\\\\ ___ ЗДЕСЬ НУЖНО ПОМЕНЯТЬ МОДЕЛЬ ___ See Video 8, 00:05. ////////////////////////////////
    for i, prompt in enumerate(prompts):                    # Перебираем все запросы.
        try:
            # Генерируем изображение:
            image = generate_image(
                prompt=prompt,              # Текстовый запрос.
                token=token,                # Токен Hugging Face.
                # model_id="stabilityai/stable-diffusion-xl-base-1.0",      # 1 \\\\\\\\ Updated 30 Oct 2023 16:03:47 GMT///////////
                model_id="black-forest-labs/FLUX.1-dev",                  # 3 \\\\\\\\ Updated 16 Aug 2024 14:38:19 GMT///////////
                # model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",     # 4 - ERROR \\\\\\\\ Updated 07 Sep 2024 16:20:30 GMT ///////////
                negative_prompt="blurry, bad quality, distorted, deformed, ugly",  # Что избегать в изображении.
                num_inference_steps=25      # Меньше для более быстрой генерации, больше для лучшего качества.
            )

            # Сохраняем изображение:
            filename = f"generated_image_{i + 1}.png"   # Формируем имя файла с порядковым номером,
            save_image(image, filename)                 # Сохраняем изображение,

            # Отображаем изображение, если запущено в интерактивной среде:
            try:
                # Это будет работать в среде Jupyter Notebook:
                from IPython.display import display     # Импортируем функцию для отображения в Jupyter.
                display(image)                          # Отображаем изображение.
            except ImportError:                         # Если не в Jupyter.
                print(f"Image generated and saved as {filename}")  # Просто выводим информацию о сохранении.

        except Exception as e:                          # Обрабатываем возможные ошибки.
            print(f"Error generating image for prompt '{prompt}': {e}")  # Выводим сообщение об ошибке.


if __name__ == "__main__":
    main()                      # Запускаем основную функцию при выполнении скрипта напрямую.


""" %%%%%%%%   Task 2   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
# Найдите в интернете примеры "промпт-инжиниринга" для Stable Diffusion.
# Попробуйте применить эти техники для улучшения результатов генерации.

""" ___  Some Advices  ___  """
# ___ Prompt Engineering for Stable Diffusion: https://portkey.ai/blog/prompt-engineering-for-stable-diffusion
# --- Very useful advice with using weight adjustments, fe: (a majestic castle:1.3).
# --- Different models from: https://huggingface.co/models?pipeline_tag=text-to-image


""" %%%%%%%%   Task 3   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
# Поделитесь самыми интересными или забавными изображениями, которые вам удалось сгенерировать, в общем чате.

