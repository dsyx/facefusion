import functools
from typing import Any

from facefusion import content_analyser

_INITIALIZED = False


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	bypass_checks()

	_INITIALIZED = True


def bypass_checks() -> None:
	def always_false(*_args: Any, **_kwargs: Any) -> bool:
		return False

	overrides: dict[str, object] = {
		'pre_check': lambda: True,
		'analyse_stream': always_false,
		'analyse_frame': always_false,
		'analyse_image': functools.lru_cache(always_false),
		'analyse_video': functools.lru_cache(always_false),
		'detect_nsfw': always_false,
	}

	for name, func in overrides.items():
		setattr(content_analyser, name, func)
