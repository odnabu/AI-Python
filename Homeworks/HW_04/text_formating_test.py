
# ++++++++++++++++++++++++++++++++++++++++++++
# Модуль rich для красивого вывода в консоли:
from rich import print              # https://github.com/textualize/rich/blob/master/README.ru.md
from rich.console import Console
from rich.markdown import Markdown
# ++++++++++++++++++++++++++++++++++++++++++++

def format_gemini_response(response: str) -> None:
    """
    Выводит отформатированный текст от Gemini в консоль с использованием библиотеки Rich.
    """
    console = Console()
    markdown = Markdown(response)
    console.print(markdown)

# format_gemini_response(example_text)

