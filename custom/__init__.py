from custom import common, windows
from facefusion.common_helper import is_windows


def setup() -> None:
	if is_windows():
		windows.setup()

	common.setup()
