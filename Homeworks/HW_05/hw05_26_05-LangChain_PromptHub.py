# Müller Alexader
# \033[0;__;__m \033[m   or   \033[1;__;__m \033[m
# print('#' * 115)      # Для разделения блоков на листе с кодом:
""" ################################################################################################################
 26.05.25
 AI  &  Python 05:  LangChain и PromptHub.
 ################################################################################################################### """

# ------------------------ SHORTCUTS ------------------------
# Ctrl + W - выделить текущий блок. если нажимать это сочетание дальше, то будут выделяться родительские блоки.
# Ctrl+Y - Удаление всей строки. Кстати, команда копирования Ctrl+C без выделения также работает для всей строки.
# Ctrl+Akt+L / Alt+Enter - Привести код к принятым стандартам (для Python - PEP8).
# Ctrl+R — Изменить название класса/функции и т. п. по всему проекту.
# Ctrl+Shift + F - Найти по всем файлам.
# Shift + F6 - заменить имя элемента во всех частях во всех файлах.
# -----------------------------------------------------------

print('.' * 70)


# LangChain и PromptHub позволяют создавать мощные AI-приложения, снижая сложность разработки и
# управления промптами. Их изучение и применение сделает ваши проекты более эффективными и продуктивными.

""" ______  Task 1  ______________________________________________________________________________________________ """
# Создайте простую цепочку LangChain для суммаризации текста из URL веб-страницы.

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Функция для создания цепочки, которая объединяет документы и обрабатывает их с помощью LLM:
from langchain.chains.combine_documents import create_stuff_documents_chain
# Класс для создания шаблонов промптов для чата:
from langchain_core.prompts import ChatPromptTemplate
# Класс для работы с генеративной моделью от Google:
from langchain_google_genai import ChatGoogleGenerativeAI
# Класс для загрузки документов (в данном случае – веб-страницы):
from langchain_community.document_loaders import WebBaseLoader
# Модуль для работы с переменными окружения (из файла .env):
from dotenv import load_dotenv
# Модуль для работы с операционной системой (например, для получения переменных окружения):
import os
# ----------------------------------------------------------------------
import re           # Модуль для регулярных выражений.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Функция, которая автоматически форматирует ответы от модели (Gemini) для вывода в
# окне Run в PyCharm с использованием цветов и стилей через ANSI-коды:
def format_gemini_response(response: str) -> str:
    """
    Принимает строку — ответ от модели (Gemini),
    и возвращает форматированную строку с ANSI-кодами для красивого вывода в консоль.
    """

    # ANSI-коды:
    RESET = '\033[0m';          BOLD_BACK = '\033[40;1m'
    BOLD = '\033[1m';           ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    GREEN = '\033[32m'
    CYAN = '\033[36m'
    YELLOW = '\033[33m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    BLUE = '\033[34m'
    GRAY = '\033[37m'

    formatted = response
    f = 30       # Number of fillers.
    # ..........................   Форматирование **жирного** текста в ЗАГОЛОВКАХ пунктов:
    pat_headers = r'^\*\*(.*?)\*\*$'
    formatted = re.sub(pat_headers, f"{BOLD_BACK}{MAGENTA}{' ▹▹▹  \\1 ':◃<{f}} {RESET}", formatted, flags=re.MULTILINE)
    # ..........................   Форматирование **жирного** текста:
    formatted = re.sub(r'\*\*(.*?)\*\*', f"{BOLD}\\1{RESET}", formatted)
    # ..........................   Форматирование *курсива*:
    formatted = re.sub(r'\*\b(.*?)\*', f"{ITALIC}{MAGENTA}\\1{RESET}", formatted)
    # ..........................   Преобразование заголовков Markdown вида "# Заголовок":
    pat_sections = r'^#{1,6} (.*)'
    formatted = re.sub(pat_sections, f"{BOLD_BACK}{MAGENTA}{' ▷▷▷  \\1 ':◁<{f}} {RESET}", formatted, flags=re.MULTILINE)
    # ..........................   Подчеркнутые строки, начинающиеся с подчёркивания (_) или «**_text_**»:
    formatted = re.sub(r'^_ (.*)', f"{UNDERLINE}\\1{RESET}", formatted, flags=re.MULTILINE)
    # ..........................   Маркированные списки 1-го уровня:
    formatted = re.sub(r'^[-*] (.*?)', f"{BOLD}{GREEN} ▶ \\1{RESET}", formatted, flags=re.MULTILINE)   # ※ ❇
    # Маркированные списки 2-го уровня:
    formatted = re.sub(r'^ {4}\* (.*?)', f"{BOLD}{MAGENTA}    ▷\\1{RESET}", formatted, flags=re.MULTILINE)   # ⩥ ▷ ▹
    # ..........................   Пронумерованные списки:
    formatted = re.sub(r'^(\d+)\. (.*?)', f"{BOLD}{GREEN}\\1. \\2{RESET}", formatted, flags=re.MULTILINE)
    # ..........................   Таблицы (просто подчёркивание заголовка таблиц):
    formatted = re.sub(r'^(.*\|.*)', f"{UNDERLINE}\\1{RESET}", formatted, flags=re.MULTILINE)

    return formatted



# # Загружаем переменные окружения из файла .env. Это нужно, чтобы получить секретные ключи,
# # НЕ прописывая их в коде:
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
#
# # Инициализируем генеративную модель Google с указанной моделью "gemini-2.0-flash" и передаём ей API-ключ:
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
#
# # Создаем загрузчик, который скачает содержимое указанного веб-адреса:
# # loader = WebBaseLoader("https://www.thelancet.com/journals/lancet/article/PIIS0140-6736%2814%2960685-1/fulltext")
# loader = WebBaseLoader("https://diapark.ru/razrab_metod.html")
#
# # Загружаем документ с веб-страницы. В переменной docs будет храниться текст или структура полученного документа:
# docs = loader.load()
#
# # Создаем шаблон для промптов, который будет использоваться для генерации ответа.
# # Здесь {context} - это место, куда подставится загруженный документ.
# prompt = ChatPromptTemplate.from_template("Выдели 10 ключевых моментов текста: {context}")
# #  и переведи на русский язык
#
# # Создаем цепочку, которая объединяет документы и передает их в LLM вместе с подсказкой:
# chain = create_stuff_documents_chain(llm, prompt)
#
# # Запускаем цепочку, передавая загруженный документ в качестве параметра "context".
# # Функция invoke (invoke буквально обозначает "вызвать", в нашем случае вызвать цепочку)
# # обрабатывает входные данные и возвращает результат от модели:
# result = chain.invoke({"context": docs})
#
# # Выводим полученный результат (например, краткое изложение текста) на экран:
# # print(result)
# formatted_output = format_gemini_response(result)
# print(formatted_output)

""" ___ FACIT ___ """
# К сожалению, порталы и сайты авторитетных мировых научных изданий закрыты для парсинга LLM-ботами.
# Причем, доступ закрыт как к страницам в html-формате, так и к тексту на странце в формате pdf.
# Возможно подключение модуля bs4 (Beautiful Soup) позволит сузить область текста на странице для
# обработки LLM-ой. Но я не уверена, что и эта возможность уже закрыта для доступа.


""" ______  Task 2  ______________________________________________________________________________________________ """
# Зарегистрируйтесь на PromptHub и изучите его интерфейс. Найдите и опишите 3 интересных промпта.
# Dvornik Olga aka odnabu on PromptHub: https://app.prompthub.us/odnabu

# ===  PROMPTS in "Healthcare & Medical" and "Data Analyze" from PromptHub  ===============================
# 1 - https://app.prompthub.us/taliaz/triage-summary-o1-mhs
# 2 - https://app.prompthub.us/biocode/diagnosis-tagger
# 3 - https://app.prompthub.us/nicos-team-7805/gmail-data-extraction


""" ______  Task 3  ______________________________________________________________________________________________ """
# (Дополнительно) Создайте цепочку "вопрос-ответ по документам" для небольшого текстового файла.


# Article from https://www.thelancet.com/journals/lancet/article/PIIS0140-6736%2814%2960685-1/fulltext

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os  # Модуль для работы с операционной системой (например, для работы с переменными окружения).
from dotenv import load_dotenv  # Функция для загрузки переменных окружения из файла .env.
from langchain_community.document_loaders import PyPDFLoader  # Импортируем загрузчик для PDF-файлов.
import asyncio  # Модуль для работы с асинхронным программированием.
from langchain_core.vectorstores import InMemoryVectorStore  # Импортируем класс для создания векторного хранилища в памяти.
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Импортируем класс для генерации эмбеддингов с использованием Google Generative AI.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Определяем асинхронную функцию для чтения PDF-файла  --->  Video 5, 49:55:
async def read_pdf():
    # Создаем объект загрузчика для указанного PDF-файла:
    # Вставить свой путь к файлу:
    loader = PyPDFLoader(r'article_from_LANCET.pdf')    # _NB!_ Если файл в ТОЙ же папке, что и код, то относит путь
                                                        # прописывать не надо, только имя файла.
    pages = []  # Инициализируем пустой список для хранения страниц из PDF.

    # Асинхронно перебираем страницы PDF с помощью метода alazy_load():
    async for page in loader.alazy_load():
        pages.append(page)  # Добавляем каждую страницу в список.

    return pages  # Возвращаем список страниц.


# Загружаем переменные окружения из файла .env:
load_dotenv()

# Получаем API-ключ из переменной окружения "GEMINI_API_KEY":
api_key = os.getenv("GEMINI_API_KEY")

# Если переменная "GOOGLE_API_KEY" не установлена, присваиваем ей значение из GEMINI_API_KEY:
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = api_key

# Выводим сообщение о начале чтения PDF-файла:
print('Start reading...')
# Запускаем асинхронную функцию read_pdf и сохраняем результат (список страниц):
result = asyncio.run(read_pdf())

# Выводим сообщение о начале формирования эмбеддингов:
print('Start embeding...')
# Создаем векторное хранилище, преобразуя документы (страницы PDF) в эмбеддинги.
# Для генерации эмбеддингов используется модель GoogleGenerativeAIEmbeddings с указанной моделью "models/embedding-001".
vector_store = InMemoryVectorStore.from_documents(result, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Выполняем поиск документов по смысловому запросу.
# Функция similarity_search ищет наиболее похожие страницы по заданному запросу.
docs = vector_store.similarity_search("Relationship between systolic or diastolic blood pressure", k=2)

# Перебираем найденные документы и выводим номер страницы и содержание.
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content}\n')


