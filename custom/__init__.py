from custom import windows
from facefusion.common_helper import is_windows


def setup() -> None:
	if is_windows():
		windows.setup()

