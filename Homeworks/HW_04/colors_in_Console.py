
# Описание ЦВЕТОВОГО оформления смотри по ссылке: https://pkg.go.dev/github.com/whitedevops/colors.
# + How do I print colored text to the terminal?: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal/60252056#60252056

"""
	ResetAll = "\033[0m"

	Bold       = "\033[1m"
	Dim        = "\033[2m"
	Underlined = "\033[4m"
	Blink      = "\033[5m"
	Reverse    = "\033[7m"
	Hidden     = "\033[8m"

	ResetBold       = "\033[21m"
	ResetDim        = "\033[22m"
	ResetUnderlined = "\033[24m"
	ResetBlink      = "\033[25m"
	ResetReverse    = "\033[27m"
	ResetHidden     = "\033[28m"

	Default      = "\033[39m"
	Black        = "\033[30m"
	Red          = "\033[31m"
	Green        = "\033[32m"
	Yellow       = "\033[33m"
	Blue         = "\033[34m"
	Magenta      = "\033[35m"
	Cyan         = "\033[36m"
	LightGray    = "\033[37m"
	DarkGray     = "\033[90m"
	LightRed     = "\033[91m"
	LightGreen   = "\033[92m"
	LightYellow  = "\033[93m"
	LightBlue    = "\033[94m"
	LightMagenta = "\033[95m"
	LightCyan    = "\033[96m"
	White        = "\033[97m"

	BackgroundDefault      = "\033[49m"
	BackgroundBlack        = "\033[40m"
	BackgroundRed          = "\033[41m"
	BackgroundGreen        = "\033[42m"
	BackgroundYellow       = "\033[43m"
	BackgroundBlue         = "\033[44m"
	BackgroundMagenta      = "\033[45m"
	BackgroundCyan         = "\033[46m"
	BackgroundLightGray    = "\033[47m"
	BackgroundDarkGray     = "\033[100m"
	BackgroundLightRed     = "\033[101m"
	BackgroundLightGreen   = "\033[102m"
	BackgroundLightYellow  = "\033[103m"
	BackgroundLightBlue    = "\033[104m"
	BackgroundLightMagenta = "\033[105m"
	BackgroundLightCyan    = "\033[106m"
	BackgroundWhite        = "\033[107m"
"""

# SGR color constants: https://python-forum.io/thread-36511-post-154045.html#pid154045
# rene-d 2018

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    # cancel SGR codes if we don't write to a terminal
    if not __import__("sys").stdout.isatty():
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        # set Windows console in VT mode
        if __import__("platform").system() == "Windows":
            kernel32 = __import__("ctypes").windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            del kernel32


if __name__ == '__main__':
    for i in dir(Colors):
        if i[0:1] != "_" and i != "END":
            print("{:>16} {}".format(i, getattr(Colors, i) + i + Colors.END))


