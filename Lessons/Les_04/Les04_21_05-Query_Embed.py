# Müller Alexader
# \033[0;__;__m \033[m   or   \033[1;__;__m \033[m
# print('#' * 115)      # Для разделения блоков на листе с кодом:
""" ################################################################################################################
 21.05.25
 AI  &  Python 4: Обработка запросов и таймаутов. Эмбеддинги и векторные представления.
 ################################################################################################################### """

# Ссылка на все файлы с кодом для этой лекции:
#       https://github.com/viacheslav-bandylo/llm-course/tree/main/lesson-ai-04

# VIDEO 4: https://player.vimeo.com/video/1086323175?h=35d0be3cc0

# ------------------------ SHORTCUTS ------------------------
# Ctrl + W - выделить текущий блок. если нажимать это сочетание дальше, то будут выделяться родительские блоки.
# Ctrl+Y - Удаление всей строки. Кстати, команда копирования Ctrl+C без выделения также работает для всей строки.
# Ctrl+Akt+L / Alt+Enter - Привести код к принятым стандартам (для Python - PEP8).
# Ctrl+R — Изменить название класса/функции и т. п. по всему проекту.
# Ctrl+Shift + F - Найти по всем файлам.
# Shift + F6 - заменить имя элемента во всех частях во всех файлах.
# -----------------------------------------------------------

print('.' * 80)


# Активация виртуального окружения командой в консоли:
# https://ru.stackoverflow.com/questions/1388073/%D0%9A%D0%B0%D0%BA-%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D1%82%D0%B8%D1%82%D1%8C-%D0%B2%D0%B8%D1%80%D1%82%D1%83%D0%B0%D0%BB%D1%8C%D0%BD%D1%83%D1%8E-%D1%81%D1%80%D0%B5%D0%B4%D1%83-venv-%D1%87%D0%B5%D1%80%D0%B5%D0%B7-%D0%BA%D0%BE%D0%BD%D1%81%D0%BE%D0%BB%D1%8C-pycharm
# . ./.venv/bin/activate

"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%______   Обработка запросов и таймаутов   ______%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """

""" __________ Ограничения скорости отправки запросов или токенов __________ """
# Ограничения скорости как защита от БЛОКИРОВКИ --> Решение: Slide 13, Video 4, 15:00.
# См. также Les04-PyLLM_4_LfS-Query_Embed.pdf, с. 3.
# File on GitHub: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-04/ai4-1.py

# # +++++++++++++++++++++++++++++++++
# import time
# from google import genai
# from dotenv import load_dotenv
# import os
# # +++++++++++++++++++++++++++++++++
#
# # Загружаем переменные окружения из файла .env:
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
#
# # Инициализируем клиент Gemini один раз вне функции для улучшения производительности:
# client = genai.Client(api_key=api_key)
#
#
# def get_gemini_response(prompt):
#     """
#     Отправляет запрос к модели Gemini и возвращает текст ответа.
#     :param prompt: Текст запроса.
#     :return: Текст ответа модели.
#     """
#     # Устанавливаем Небольшую задержку перед отправкой запроса (можно убрать или изменить):
#     time.sleep(0.3)
#
#     # Используем глобальный объект клиента:
#     response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=[prompt],
#     )
#
#     return response.text
#
#
# # Пример использования
# if __name__ == "__main__":
#     response = get_gemini_response("Whats rate limits?")
#     print(response)



""" __________ Что делать, если запрос "зависает"? __________ """
# Что делать, если запрос "зависает"? --> Решение: Slide 14, Video 4, 17:50.
# File on GitHub: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-04/ai4-2.py

# # ++++++++++++++++++++++++++++++++++
# import time
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv
# import os
# from requests import ReadTimeout
# # ++++++++++++++++++++++++++++++++++
#
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
#
# # Инициализируем клиент Gemini с таймаутом (в секундах)
# timeout_seconds = 10  # Установите желаемое значение таймаута
# client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000))
#
#
# def get_gemini_response(prompt):
#     """
#     Отправляет запрос к модели Gemini и возвращает текст ответа.
#     Обрабатывает исключение TimeoutError.
#     :param prompt: Текст запроса.
#     :return: Текст ответа модели или None в случае таймаута.
#     """
#     # Небольшая задержка для предотвращения слишком частых запросов (настройте при необходимости)
#     time.sleep(0.3)
#
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",  # Используемая модель Gemini
#             contents=[prompt]
#         )
#         return response.text  # Возвращаем текст ответа
#     except ReadTimeout:
#         # Выводим значение таймаута в секундах
#         return f"Запрос к Gemini превысил таймаут ({timeout_seconds} секунд)."
#     except Exception as e:
#         # Общая обработка исключений для отлова неожиданных ошибок
#         return f"Произошла ошибка: {str(e)}"
#
#
# if __name__ == "__main__":
#     response = get_gemini_response("Whats request timeout?")
#     if response:
#         print(response)
#     else:
#         print("Не удалось получить ответ от Gemini.")



""" __________ Автоматический повтор запроса __________ """
# Автоматический повтор запроса --> Решение: Slide 16, Video 4, 28:30.
# File on GitHub: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-04/ai4-3.py

# Если, например, сервер не ответит за 10 секунд, программа обработает ошибку и не зависнет.
# Обработка распространенных ошибок API (500, 429, 403):
#   ● 500 (Internal Server Error) – ошибка на стороне сервера API.
#   ● 503 (UNAVAILABLE) – ошибка на стороне сервера API 503: 'The model is overloaded. Please try again later.'
#   ● 403 (Forbidden) – неверный API-ключ.
#   ● 429 (Rate Limit) – превышено количество запросов.
# Библиотека tenacity в Python позволяет автоматически повторять запрос в случае ошибки.
# Это полезно при взаимодействии с ненадёжными ресурсами, например:
#   ● Сетевые запросы (HTTP-запросы, API)
#   ● Запросы к базе данных
#   ● Долговременные вычисления с возможными сбоями

# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import time
# from tenacity import retry, stop_after_attempt, wait_exponential
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv
# import os
# from requests import ReadTimeout
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
#
# # Инициализируем клиент Gemini с таймаутом (в секундах)
# timeout_seconds = 10  # Установите желаемое значение таймаута
# client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_seconds * 1000))
#
# # ___ ДЕКОРАТОР ПОВТОРА ЗАПРОСА: ___
# # Повторяем попытки подключения к серверу при сбое.
# # stop_after_attempt(3): останавливаемся после 3-й попытки.
# # wait_exponential(multiplier=1, min=2, max=10):
# #   - multiplier=1: начальное время ожидания 2 секунды.
# #   - min=2: минимальное время ожидания 2 секунды.
# #   - max=10: максимальное время ожидания 10 секунд.
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
# def get_gemini_response(prompt):
#     """
#     Отправляет запрос к модели Gemini и возвращает текст ответа.
#     Обрабатывает исключение TimeoutError.
#     :param prompt: Текст запроса.
#     :return: Текст ответа модели или None в случае таймаута.
#     """
#
#     # Небольшая задержка для предотвращения слишком частых запросов (настройте при необходимости)
#     time.sleep(0.3)
#
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.0-flash",               # Используемая модель Gemini.
#             contents=[prompt]
#         )
#         return response.text                        # Возвращаем текст ответа.
#     except ReadTimeout:
#         # Исправлено |=> выводим значение таймаута в секундах без преобразования:
#         return f"Запрос к Gemini превысил таймаут ({timeout_seconds} секунд)."
#     except Exception as e:
#         # Общая обработка исключений для отлова неожиданных ошибок:
#         return f"Произошла ошибка: {e}"
#
#
# if __name__ == "__main__":
#     response = get_gemini_response("How can I handle errors when working with the API??")
#     if response:
#         print(response)
#     else:
#         print("Не удалось получить ответ от Gemini.")



"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%___________   Эмбеддинги и       __________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
""""                                         векторные представления                                           """

""" __________ Эмбеддинг __________ """
# Эмбеддинг --> Example: Slide 21, Video 4,  37:15.
# File on GitHub: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-04/ai4-4.py

# Эмбеддинг (embedding) - числовое представление текста, которое позволяет AI понимать смысл слов и предложений.
# Чем ближе числа в векторе, тем ближе слова по смыслу.
# AI находит наиболее похожие вектора и возвращает подходящие отзывы. Это называется семантическим поиском –
# поиск не по ключевым словам, а по смыслу.

# Получение Эмбеддинга - преобразования текста в вектор. Для этого используется другая модель из genai:
# "text-embedding-004"

# # +++++++++++++++++++++++++++++++
# import os
# from dotenv import load_dotenv
# from google import genai
# import numpy as np                  #
#
# from rich.console import Console
# from rich import print as rprint
# # +++++++++++++++++++++++++++++++
#
# console = Console()
#
# # Загрузка переменных окружения из файла .env:
# load_dotenv()
#
# # Получение API-ключа из переменной окружения:
# api_key = os.getenv("GEMINI_API_KEY")
#
# # Инициализация клиента Gemini для работы с API:
# client = genai.Client(api_key=api_key)
#
#
# def get_embedding(text):
#     """
#     Получает embedding (векторное представление) для заданного текста.
#     :param text: Строка текста, для которого требуется получить embedding.
#     :return: Список чисел, представляющих векторное embedding.
#     """
#
#     # Отправка запроса к API для получения векторного представления текста:
#     response = client.models.embed_content(
#         model="text-embedding-004",
#         contents=text)
#
#     return np.array(response.embeddings[0].values)     # Возвращаем embedding как numpy array.
#
#
# # Получение embedding для заданных текстовых значений:
# query_1 = "Python."
# query_2 = "Кодинг – это моё хобби."
# vector_1 = get_embedding(query_1)
# vector_2 = get_embedding(query_2)
#
# # Вывод полученных векторов с поясняющими сообщениями:
# # Описание ЦВЕТОВОГО оформления смотри по ссылке: https://pkg.go.dev/github.com/whitedevops/colors.
# print(f'\033[7;33;40m {'*' * 60} \033[0m\n\n', f'\t\033[1;33m{query_1[:-1]}:\033[0m\n', vector_1)
# print(f'\033[7;33;40m {'*' * 60} \033[0m\n\n', f'\t\033[1;33m{query_2[:-1]}:\033[m\n', vector_2)
#
# # Печать цветного оформления в ТЕРМИНАЛЕ с помощью пакета rich:
# # rprint(f'[red]{'*' * 60}[/red]\n{query_1[:-1]}:\n {vector_1}')
# # console.print(f'[red]{'*' * 60}[/red]\n{query_1[:-1]}:\n {vector_1}')


""" __________ Как AI определяет схожесть текстов __________ """
# Как AI определяет схожесть текстов --> Решение: Slide 29, Video 4, 58:30.
# File on GitHub: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-04/ai4-5.py

# FAISS (Facebook AI Similarity Search) – быстрый поиск похожих векторов.
# INSTALLING faiss: Video 4, 1:00:50 +++ https://myscale.com/blog/simple-steps-to-install-faiss-using-pip/
# pip install faiss-cpu             # Для процессорной обработки.
# # pip install faiss-gpu           # Для обработки на видеокарте.
# Step 5: Verify Your Installation:
# python -c "import faiss; print(faiss.__version__)"

# +++++++++++++++++++++++++++++++++++
import os
import faiss
from dotenv import load_dotenv
from google import genai
import numpy as np
# +++++++++++++++++++++++++++++++++++

# Загрузка переменных окружения из файла .env:
load_dotenv()

# Получение API-ключа из переменной окружения:
api_key = os.getenv("GEMINI_API_KEY")

# Инициализация клиента Gemini для работы с API:
client = genai.Client(api_key=api_key)


def get_embedding(text):
    """
    Получает embedding (векторное представление) для заданного текста.
    :param text: Строка текста, для которого требуется получить embedding.
    :return: Список чисел, представляющих векторное embedding.
    """

    # Отправка запроса к API для получения векторного представления текста
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text)

    return np.array(response.embeddings[0].values)     # Возвращаем embedding как numpy array


# ---  Секция семантического поиска  ----------------------------------------------------------

# 1. Создание набора текстов для поиска
texts_to_index = [
    "Кошка сидит на окне",
    "Собака играет в парке",
    "Ананас растет в тропиках",
    "Кот спит на диване",
    "Пес лает на почтальона",
    "Фрукт ананас очень вкусный",
    "Домашняя кошка любит ласку",
    "Верный пес охраняет дом",
    "Спелый ананас полон витаминов",
    "Кошки любят рыбу и мясо",
    "Основной рацион кошек - это белок",
    "Чем кормить котенка?",
    "Лучший корм для кошек - сбалансированный",
    "Коты едят сухой и влажный корм",
    "Нельзя кормить кошку шоколадом",
    "Молоко не всегда полезно для кошек",
    "Я обожаю программировать.",
    "Программирование – это то, что мне очень нравится.",
    "Меня увлекает разработка программного обеспечения.",
    "Я испытываю страсть к написанию кода.",
    "Программирование приносит мне огромное удовольствие.",
    "Мне интересно заниматься программированием.",
    "Моё хобби – это кодинг.",
    "В свободное время я занимаюсь программированием.",
    "Кодинг – это моё любимое увлечение.",
    "Я увлекаюсь кодингом на досуге.",
    "Программирование – это моё хобби и страсть.",
    "Когда есть свободное время, я кодирую.",
    "Кодинг - это моё хобби, которым я наслаждаюсь."
]

# 2. Получение embedding для каждого текста и сохранение их в списке:
embeddings_list = [get_embedding(text) for text in texts_to_index]
embeddings_array = np.array(embeddings_list)    # Преобразуем в numpy array для FAISS.

# 3. Создание FAISS индекса:
dimension = embeddings_array.shape[1]   # Размерность embedding.
index = faiss.IndexFlatL2(dimension)    # Используем IndexFlatL2 для L2 дистанции (евклидова).
index.add(embeddings_array)             # Добавляем embeddings в индекс.


# 4. Функция для выполнения семантического поиска:
def semantic_search(query, index, texts, k=2):
    """
    Выполняет семантический поиск по индексу FAISS.
    :param query: Поисковый запрос (строка).
    :param index: FAISS индекс.
    :param texts: Список текстов, которые были проиндексированы.
    :param k: Количество ближайших соседей для поиска (по умолчанию 2).
    :return: Список из k наиболее релевантных текстов.
    """
    query_embedding = get_embedding(query).reshape(1, -1)   # Получаем embedding для запроса и меняем размерность.
    D, I = index.search(query_embedding, k)                 # Ищем k ближайших соседа.
    results = [texts[i] for i in I[0]]                      # Получаем тексты по индексам.
    return results


# 5. Пример использования семантического поиска
search_query = "Питание кошек"
search_results = semantic_search(search_query, index, texts_to_index, k=4)

print("\n--- Результаты семантического поиска ---")
print(f"Запрос: '{search_query}'")
print("Найденные соответствия:")
for result in search_results:
    print(f"- {result}")



""" __________ Построение векторных баз данных __________ """
# Построение векторных баз данных --> Решение: Slide 35, Video 4, 1:21:13.
# Сначала установить библиотеку faiss через команду в консоли:
# pip install faiss     или, если первый способ не сработает    pip install faiss-cpu





""" ___________________________________  Review of previously covered material  ___________________________________ """
"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%___________   ---   __________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
""" __________ --- __________ """
""" __________ --- __________ """
#       ●
# ___ EXAMPLE __________________________________________________
# ___ END of Example __________________________________________________


""" ______  Task 1  ______________________________________________________________________________________________ """
#


