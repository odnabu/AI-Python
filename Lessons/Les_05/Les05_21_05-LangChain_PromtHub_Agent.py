# Müller Alexader
# \033[0;__;__m \033[m   or   \033[1;__;__m \033[m
# print('#' * 115)      # Для разделения блоков на листе с кодом:
""" ################################################################################################################
 21.05.25
 AI  &  Python 5: Экосистема LangChain и PromptHub. Агентный подход в разработке с использованием LLM.
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

print('.' * 60)


"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_____________   LangChain:       ____________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                                                Введение и основы                                                  """

# Сначала установить библиотеку LangChain через команду в консоли:
# pip install LangChain
# pip install langchain_google_genai
# pip install langchain_core
# pip install langchain_community

# Python file has "No Module Named bs4.": https://stackoverflow.com/questions/11783875/importerror-no-module-named-bs4-beautifulsoup
# ---> install BeautifulSoup4 (bs4):
# pip install beautifulsoup4

""" __________ Пример цепочки в LangChain __________ """
# Пример цепочки в LangChain --> Решение: Slide 19, Video 5, 12:00.
# Video 5 --- https://player.vimeo.com/video/1086361719?h=0ff4830d57
# File on GitHub: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-05/ai5-1.py


# ЗАДАНИЕ: создать приложение, которое суммаризирует текст с заданной веб-страницы.

# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # Функция для создания цепочки, которая объединяет документы и обрабатывает их с помощью LLM:
# from langchain.chains.combine_documents import create_stuff_documents_chain
# # Класс для создания шаблонов промптов для чата:
# from langchain_core.prompts import ChatPromptTemplate
# # Класс для работы с генеративной моделью от Google:
# from langchain_google_genai import ChatGoogleGenerativeAI
# # Класс для загрузки документов (в данном случае – веб-страницы):
# from langchain_community.document_loaders import WebBaseLoader
# # Модуль для работы с переменными окружения (из файла .env):
# from dotenv import load_dotenv
# # Модуль для работы с операционной системой (например, для получения переменных окружения):
# import os
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# # Загружаем переменные окружения из файла .env. Это нужно, чтобы получить секретные ключи,
# # НЕ прописывая их в коде:
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
#
# # Инициализируем генеративную модель Google с указанной моделью "gemini-2.0-flash" и передаём ей API-ключ:
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
#
# # Создаем загрузчик, который скачает содержимое указанного веб-адреса:
# loader = WebBaseLoader("https://habr.com/ru/articles/883604/")
#
# # Загружаем документ с веб-страницы. В переменной docs будет храниться текст или структура полученного документа:
# docs = loader.load()
#
# # Создаем шаблон для промптов, который будет использоваться для генерации ответа.
# # Здесь {context} - это место, куда подставится загруженный документ.
# prompt = ChatPromptTemplate.from_template("Напишите краткое изложение следующего текста: {context}")
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
# print(result)

""" _NB!_    ___  QUESTION  ___ """
# Что означает это ошибка? ---> See Video 5, 26:45.
# "USER_AGENT environment variable not set, consider setting it to identify your requests."



"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%_____________   LangChain:       ____________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
                                       Интеграция с внешними источниками данных                                    """

""" __________ Примеры использования Document Loaders с векторным хранилищем __________ """

""" _ NB! _ """   #   --->  далее ПРИМЕР семантического поиска по PDF-файлу.

# Установить модуль:
# pip install PyPDF

#  --> Решение: Slide 32, Video 5, 43:55.
# File on GitHub: https://github.com/viacheslav-bandylo/llm-course/blob/main/lesson-ai-05/ai5-2.py

""" _ NB! _ """   #    asyncio - Модуль для работы с асинхронным программированием |=> Video 5, 49:10.
# Асинхронное программирование 49:55 - распараллеливание кода, те выполнение кода по частям и параллельно.

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
    # loader = PyPDFLoader('../files/The-Old-Man-and-The-Sea-by-Ernest-Hemingway.pdf')    # вставить свой путь к файлу
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
print('Start reading..')
# Запускаем асинхронную функцию read_pdf и сохраняем результат (список страниц):
result = asyncio.run(read_pdf())

# Выводим сообщение о начале формирования эмбеддингов:
print('Start embeding..')
# Создаем векторное хранилище, преобразуя документы (страницы PDF) в эмбеддинги.
# Для генерации эмбеддингов используется модель GoogleGenerativeAIEmbeddings с указанной моделью "models/embedding-001".
vector_store = InMemoryVectorStore.from_documents(result, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# Выполняем поиск документов по смысловому запросу.
# Функция similarity_search ищет наиболее похожие страницы по заданному запросу.
# docs = vector_store.similarity_search("Moment when sharks for the first time attack the fish", k=2)
docs = vector_store.similarity_search("Relationship between systolic or diastolic blood pressure", k=2)

# Перебираем найденные документы и выводим номер страницы и содержание.
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content}\n')



""" ___________________________________  Review of previously covered material  ___________________________________ """
"""" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%___________   ---   __________%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% """
""" __________ --- __________ """
""" __________ --- __________ """
#       ●
# ___ EXAMPLE __________________________________________________
# ___ END of Example __________________________________________________


""" ______  Task 1  ______________________________________________________________________________________________ """
#


