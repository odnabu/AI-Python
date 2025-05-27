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
# # loader = WebBaseLoader("https://pdf.sciencedirectassets.com/271074/1-s2.0-S0140673614X60861/1-s2.0-S0140673614606851/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEIP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIHzFBj4rd7fvNJlIYc2oxpStblHK6SZfncAeEX4yvs4EAiAtBEpo1Xppxdk8ajmL6w4kFeBOqrWo%2BwlKtQpn8qPsESqzBQhMEAUaDDA1OTAwMzU0Njg2NSIMjzklz9usOX8PhaNBKpAFwALi31CCxN9syJOHHciDm2lR9iU0f8Ijkslz843RTWb809C2fxv8QHD%2FSg14hCwgtmjbsM2%2BjPCcZRUZpSn%2F3ff7tnsGhzjqaMi59Nkd9Ifl6c2vd7GCy75xqBhhVCDO1PtBVUyjalR%2F7K814M%2BkUb1Hx9u6xXLMPF%2BxwvLFeeWEpJTS3p8YUhuqWeYOIw93x1zTKdUmgf84JLr1OLRFK%2F3LdE4F%2Fv9g89s6x82jbPVWy%2FXGHPyJFxR%2BX%2BntKz1xLIYy50%2BiVfOhwF8wWqXgEqIqDjyM%2BKXK2c9eu42VrQXtWLD9PiMYD%2FBUfG2%2FPjJ%2FgvQ0OULD1st3XWYgp6bQDF%2BJxMqyFcWteGsC8iGYwC%2FaLkZRzzF1cv3ywv1Ncxj%2BFGktDHAe8MqoC5nt89kpbylBemGFoYADO1SLzr6HQ2O1vVnZ9uA2C8wk14XepVEBGvAlIIaHSjzkfSuPgTgoqxQzFRKtKnI0HHmI9y%2FNMl6CslSpHTUi%2FeKu9U86QYhaeyL37r5JVefSwuUGG5m%2FHN6HfLoOn0k3B8vbxvNRCTMSHZ0A9KXGVTw22Nh1vh6ErhA3d4tRJWb0LlKuCcaD7kBynNIEUi6nJ3dkifgOChZVw8UCel0FBYx9p%2BgEBOKIjuB8CsyJAn%2Bxjjm2d0JDsVgLsEpblCl2LK%2BoTVUzGH%2BNsxjtbyYtwLCg%2BLDUOVGbHcA3lr13R5gHdyhPsa8pJT%2BXsXYgjAXeQ%2Bk%2BEUIaLAclUglk8YumLGykQa0Bu4YlLMJrr6XWizdA2GsL1mTkQ9WaeGRKi21Dn50toi7NAjZobUzVX4NHox5r7xnJJj9y%2Bz%2FJLcyIFg0c6RRkhtJ2hwz8hqCXVwPH8pDrlriCo4swyOjSwQY6sgFmI%2FA85JwVTmrh1XE2WO9Psg1A1aIdcTmaO6vF6nfAyEddfoOyinOFjRmn1xKsl07Z%2FVZa6AzSXs71Ol9QSUBDsopJZ52UOm1iKXpmcdgJPZkbrwc9QWta9%2FtYH%2FzE3cEN7dqR4m9ToQtLbxYIUZsNsD%2Bw%2Fx8LBSe6UR0Zo%2BSu4HRwIylPLfKhOF9kzCdacjbaXR%2FhwA8tmgRqEpvfruInK4lAgALt81vIH7ixtXYXuwiu&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250526T193909Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2QRXANAS%2F20250526%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8977d7553dfae1efb58a63753449ce1f962ec571b975c4fbdda0c711cacaf737&hash=52092d80b7369e0b9a622816be0bdc9713065ccb2554d47aa75d6b899a25d8b8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0140673614606851&tid=spdf-249dc7d3-308d-4a05-b630-c7d32ba52ff7&sid=f179e90b9110a445c9785b211335545610fdgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1e035b5157010000025957&rr=945fbca67e3ae51e&cc=de")
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


