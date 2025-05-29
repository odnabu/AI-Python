# Müller Alexader
# \033[0;__;__m \033[m   or   \033[1;__;__m \033[m
# print('#' * 115)      # Для разделения блоков на листе с кодом:
""" ################################################################################################################
 28.05.25
 AI  &  Python 7: Работа с мультимодальными моделями: изображения, звук, видео.
 GitHub Course: https://github.com/viacheslav-bandylo/llm-course/tree/main
 ################################################################################################################### """

# ------------------------ SHORTCUTS ------------------------
# Ctrl + W - выделить текущий блок. если нажимать это сочетание дальше, то будут выделяться родительские блоки.
# Ctrl+Y - Удаление всей строки. Кстати, команда копирования Ctrl+C без выделения также работает для всей строки.
# Ctrl+Akt+L / Alt+Enter - Привести код к принятым стандартам (для Python - PEP8).
# Ctrl+R — Изменить название класса/функции и т. п. по всему проекту.
# Ctrl+Shift + F - Найти по всем файлам.
# Shift + F6 - заменить имя элемента во всех частях во всех файлах.
# -----------------------------------------------------------

print('.' * 80)


"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_________     Мультимодальность:    ________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        Текст, Изображения, Звук, Видео, Другое                                  """
# Текст - Слова, предложения, абзацы.
# Изображения -  Картинки, фотографии, рисунки.
# Звук - Речь, музыка, звуки, животных, окружающий шум.
# Видео - Последовательность изображений, динамика, движение, звук.
# Другое - 3D-модели, сенсорные данные итд.

# Пример использования CLIP: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-07/ai7-1.py

# Установка библиотек PIL and clip:
# pip install Pillow
# pip install git+https://github.com/openai/CLIP.git

# Для запуска кода из этого файла в терминале ввести команду:
# python Lessons/Les_07/Les07_28_05-Picture_Audio_Video.py

# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import torch            # Библиотека для работы с тензорами и нейронными сетями
# import clip             # Библиотека CLIP (Contrastive Language-Image Pre-training) от OpenAI
# from PIL import Image   # Библиотека для работы с изображениями
# import requests         # Библиотека для отправки HTTP-запросов
# from io import BytesIO  # Модуль для работы с бинарными данными в памяти
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#
# def setup_clip():
#     """
#     Настройка CLIP: проверка установки необходимых пакетов.
#     Возвращает доступное устройство (CUDA для GPU или CPU).
#     """
#     # Проверяем, установлен ли CLIP, если нет - выводим сообщение:
#     try:
#         import clip
#     except ImportError:
#         print("Installing CLIP...")                 # Сообщение о необходимости установки CLIP.
#
#     # Определяем доступное устройство: GPU (CUDA) или CPU:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")                # Выводим информацию об используемом устройстве.
#
#     return device                                   # Возвращаем устройство для дальнейшего использования.
#
#
# def load_image_from_url(url):
#     """Загрузка изображения по URL-адресу."""
#     response = requests.get(url)                    # Отправляем HTTP-запрос для получения изображения.
#     return Image.open(BytesIO(response.content))    # Открываем изображение из полученных данных.
#
#
# def load_image_from_file(file_path):
#     """Загрузка изображения из локального файла."""
#     return Image.open(file_path)                    # Открываем изображение из файла.
#
#
# def text_to_image_similarity(model, processor, text_queries, images, device):
#     """
#     Вычисление сходства между текстовыми запросами и изображениями.
#     Аргументы:
#         model: Модель CLIP.
#         processor: Препроцессор CLIP для обработки изображений.
#         text_queries: Список текстовых запросов.
#         images: Список изображений в формате PIL.
#         device: Устройство для вычислений (cuda/cpu).
#     Возвращает:
#         Оценки сходства между каждым текстовым запросом и каждым изображением
#     """
#     # Обработка текста с помощью библиотек clip и torch:
#     text_inputs = clip.tokenize(text_queries).to(device)    # Токенизируем текст и переносим (.to()) на устройство.
#     with torch.no_grad():                                   # Отключаем вычисление градиентов для экономии памяти.
#         text_features = model.encode_text(text_inputs)      # Кодируем текст в векторы признаков.
#         text_features /= text_features.norm(dim=-1, keepdim=True)  # Нормализуем векторы (делаем длину = 1).
#
#     # Обработка изображений с помощью библиотеки torch:
#     image_inputs = torch.stack([processor(img).to(device) for img in images])  # Обрабатываем и объединяем изображения
#     with torch.no_grad():                                   # Отключаем вычисление градиентов.
#         image_features = model.encode_image(image_inputs)   # Кодируем изображения в векторы признаков.
#         image_features /= image_features.norm(dim=-1, keepdim=True)  # Нормализуем векторы (a /= 5  |=>  a = a/5).
#
#     # Вычисляем сходство между текстом и изображениями:
#     #       @ - это матричное умножение из библиотеки torch, T - транспонирование.
#     #       softmax превращает сходства в вероятности (в сумме дают 1).
#     similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
#     return similarity.cpu().numpy()                         # Переносим результат на CPU и конвертируем в numpy массив.
#
#
# def main():
#     # Настройка:
#     device = setup_clip()                                   # Определяем устройство для вычислений.
#
#     # Загружаем модель и препроцессор CLIP:
#     print("Loading CLIP model...")
#     model, processor = clip.load("ViT-B/32", device=device)  # ViT-B/32 - вариант модели (Vision Transformer).
#
#     # Пример использования с URL-адресами:
#     print("\nExample 1: Comparing text prompts to online images")
#
#     # URL-адреса изображений:
#     image_urls = [
#         "https://cdn.shopify.com/s/files/1/0086/0795/7054/files/Golden-Retriever.jpg?v=1645179525",     # Собака.
#         "https://miro.medium.com/v2/resize:fit:1400/1*tMKkGydXuiOBOb15srANvg@2x.jpeg"                   # Закат.
#     ]
#
#     # Загружаем изображения по URL:
#     try:
#         images = [load_image_from_url(url) for url in image_urls]   # Загружаем каждое изображение.
#         print(f"Successfully loaded {len(images)} images")
#     except Exception as e:
#         print(f"Error loading images: {e}")                         # Выводим ошибку, если изображения не загрузились.
#         print("Falling back to local images if available...")
#         # Здесь можно добавить запасной вариант загрузки:
#         return
#
#     # Текстовые запросы для сравнения с изображениями:
#     text_queries = ["a dog", "a car", "a sunset", "a person"]
#
#     # Вычисляем сходство между текстом и изображениями:
#     similarities = text_to_image_similarity(model, processor, text_queries, images, device)
#
#     # Выводим результаты:
#     print("\nSimilarity Results (%):")
#     for i, text in enumerate(text_queries):
#         print(f"\nText: '{text}'")
#         for j, url in enumerate(image_urls):
#             print(f"  Image {j + 1}: {similarities[i][j] * 100:.2f}%")      # Выводим процент сходства.
#
#     # Пример с классификацией изображений:
#     print("\nExample 2: Zero-shot image classification")
#
#     # Изображение для классификации:
#     image = images[1]                   # Используем второе изображение из предыдущего примера.
#
#     # Возможные метки для классификации:
#     labels = ["a photo of a dog", "a photo of a cat", "a photo of a car", "a photo of a sunset"]
#
#     # Обрабатываем изображение:
#     image_input = processor(image).unsqueeze(0).to(device)  # Добавляем размерность для батча и переносим на устройство
#
#     # Обрабатываем текст:
#     text_inputs = clip.tokenize(labels).to(device)          # Токенизируем метки.
#
#     # Вычисляем вероятности:
#     with torch.no_grad():                                   # Отключаем вычисление градиентов.
#         image_features = model.encode_image(image_input)    # Кодируем изображение.
#         text_features = model.encode_text(text_inputs)      # Кодируем текст.
#
#         # Нормализуем векторы признаков:
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#
#         # Вычисляем сходство и конвертируем в вероятности:
#         logits_per_image = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#         probs = logits_per_image.cpu().numpy()[0]           # Получаем вероятности как numpy массив.
#
#     # Выводим результаты:
#     print("\nClassification Results:")
#     for i, label in enumerate(labels):
#         print(f"{label}: {probs[i] * 100:.2f}%")            # Выводим вероятность для каждой метки.
#
#
# if __name__ == "__main__":
#     main()          # Запускаем основную функцию при выполнении скрипта напрямую.



"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%___________   Stable Diffusion   __________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

# Пример использования Stable Diffusion:
# https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-07/ai7-2.py

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
        # Вид на океанские волны  в штормовую погоду с пляжа на берегу Атлантического океана в Португалии:
        "View of ocean waves in stormy weather from a beach on the Atlantic Ocean in Portugal.",
        # Межзвездное космическое пространство на окраине галактики с видом на центр галактики.
        "Interstellar span on the outskirts of the galaxy with a view of the center of the galaxy.",
    ]

    # \\\\\\\\\\\\\\\\\\ ___ ЗДЕСЬ НУЖНО ПОМЕНЯТЬ МОДЕЛЬ ___ See Video 8, 00:05. ////////////////////////////////
    for i, prompt in enumerate(prompts):                    # Перебираем все запросы.
        try:
            # Генерируем изображение:
            image = generate_image(
                prompt=prompt,              # Текстовый запрос.
                token=token,                # Токен Hugging Face.
                model_id="stabilityai/stable-diffusion-xl-base-1.0",     # \\\\\\\\ ///////////
                negative_prompt="blurry, bad quality, distorted, ugly",  # Что избегать в изображении.
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




""" ___________________________________  Review of previously covered material  ___________________________________ """
"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%___________   ---   __________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
""" __________ --- __________ """
""" __________ --- __________ """
#       ●
# ___ EXAMPLE __________________________________________________
# ___ END of Example __________________________________________________


""" ______  Task 1  ______________________________________________________________________________________________ """
#


