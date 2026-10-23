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

	overrides = {
		'pre_check': lambda: True,
		'analyse_image': lru_cache(mock_analysis),
		'analyse_video': lru_cache(mock_analysis),
		'analyse_stream': mock_analysis,
		'analyse_frame': mock_analysis,
		'detect_nsfw': mock_analysis,
	}

	for name, func in overrides.items():
		setattr(content_analyser, name, func)
