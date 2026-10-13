from functools import lru_cache
from typing import Any

from facefusion import content_analyser

_INITIALIZED: bool = False


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	bypass_content_analyser()

	_INITIALIZED = True


def bypass_content_analyser() -> None:
	def mock_analysis(*args: Any, **kwargs: Any) -> bool:
		return False

	content_analyser.pre_check = lambda: True
	content_analyser.analyse_image = lru_cache(mock_analysis)
	content_analyser.analyse_video = lru_cache(mock_analysis)
	content_analyser.analyse_stream = mock_analysis
	content_analyser.analyse_frame = mock_analysis
	content_analyser.detect_nsfw = mock_analysis

