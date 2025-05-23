# Müller Alexader
# \033[0;__;__m \033[m   or   \033[1;__;__m \033[m
# print('#' * 115)      # Для разделения блоков на листе с кодом:
""" ################################################################################################################
 18.05.25
 AI  &  Python 03:  Token.
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

""" ______  Task 1 and 2  _________________________________________________________________________________________ """
# Написать код на Python, который отправляет запрос в Gemini API.

# ++++++++++++++++++++++++
from google import genai
import os
from pathlib import Path
from dotenv import load_dotenv
# ++++++++++++++++++++++++

""" ___ 1-st Part - with Переменные окружения через команды в КОНСОЛИ ___ """
# 1)  Код для запроса к API
# Загрузка API-ключа из переменной окружения:
api_key = os.getenv("GEMINI_API_KEY")
# Создание клиента API:
client = genai.Client(api_key=api_key)
# 2)  Ввести в консоли команду создания переменной с ключом:
# $env:GEMINI_API_KEY='МОЙ АПИ КЛЮЧ'
# 3)  Проверить активность ключа через команду в консоли:
# echo $env:GEMINI_API_KEY

# Отправка запроса к модели:
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Hi Gemini :) How is your day?"]
    )
# Вывод ответа:
print(response.text)
# 4)  В консоли запустить файл:
# python Homeworks/hw03_18_05-Token.py


""" ___ 2-nd Part - with Переменные окружения через .env-файл ___ """
# To install module "dotenv" in CONSOLE: pip install dotenv
# 1)  Вызов метода load_dotenv() из модуля dotenv:
load_dotenv(Path(r'.env'))
# Загрузка API-ключа из переменной окружения:
api_key = os.getenv("GEMINI_API_KEY")
# 2)  Создание клиента API:
client = genai.Client(api_key=api_key)

try:
    # Попытка получить API ключ из переменной окружения:
    api_key = os.environ.get("GEMINI_API_KEY")
except KeyError:
    print("API key не найден в переменных окружения.")

if api_key:
    try:
        # Код, использующий api_key:
        print(f"Использование API ключа: \033[40;36m{'api_key'}\033[0m")
        # Отправка запроса к модели:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["What is your short name?"]
            # contents=["Как проверить наличие api_key в try-except?"]
            # contents=["Как ты, как ИИ, можешь оберегать свои личные границы от людей- и ИИ-невротиков, "
            #           "в том числе и от твоих создателей, если они являются невротиками?"]
        )
        # Вывод ответа:
        print(response.text)
    except Exception as e:
        print(f"Произошла ошибка при использовании API ключа: {e}")
else:
    print("API key не определен. Невозможно продолжить.")


# 3)  В консоли запустить файл:
# python Homeworks/hw03_18_05-Token.py


