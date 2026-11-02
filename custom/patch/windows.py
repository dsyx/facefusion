import builtins
import contextlib
import functools
import io
import os
import site
import sys
from pathlib import Path
from typing import Any

_INITIALIZED: bool = False


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	setup_utf8_encoding()
	setup_dll_directories()

	_INITIALIZED = True


def setup_utf8_encoding() -> None:
	os.environ['PYTHONUTF8'] = '1'

	if sys.flags.utf8_mode:
		return

	for stream in (sys.stdout, sys.stderr):
		if isinstance(stream, io.TextIOWrapper):
			with contextlib.suppress(Exception):
				stream.reconfigure(encoding='utf-8')

	@functools.wraps(io.text_encoding)
	def utf8_text_encoding(encoding: str | None, stacklevel: int = 2) -> str:
		return 'utf-8' if encoding is None else encoding

	setattr(io, utf8_text_encoding.__name__, utf8_text_encoding)

	original_open = io.open

	@functools.wraps(original_open)
	def utf8_open(
		file: Any,
		mode: str = 'r',
		buffering: int = -1,
		encoding: str | None = None,
		*args: Any,
		**kwargs: Any,
	) -> Any:
		if 'b' not in mode and encoding is None:
			encoding = 'utf-8'
		return original_open(file, mode, buffering, encoding, *args, **kwargs)

	for module in (builtins, io):
		setattr(module, utf8_open.__name__, utf8_open)


def setup_dll_directories() -> None:
	candidate_paths = sys.path.copy()
	if hasattr(site, 'getsitepackages'):
		candidate_paths.extend(site.getsitepackages())

	site_packages_paths = {
		Path(path).resolve()
		for path in candidate_paths
		if path and Path(path).is_dir() and Path(path).name.lower() == 'site-packages'
	}
	candidate_library_paths: set[Path] = set()

	for site_packages_path in site_packages_paths:
		candidate_library_paths.update(site_packages_path.glob('nvidia/*/bin'))
		candidate_library_paths.add(site_packages_path / 'onnxruntime' / 'capi')
		candidate_library_paths.add(site_packages_path / 'tensorrt_libs')

	library_directory_paths = sorted(
		str(path) for path in candidate_library_paths if path.is_dir()
	)

	if library_directory_paths:
		for library_directory_path in library_directory_paths:
			with contextlib.suppress(Exception):
				os.add_dll_directory(library_directory_path)
		system_path = os.getenv('PATH')
		os.environ['PATH'] = os.pathsep.join(
			library_directory_paths + ([system_path] if system_path else [])
		)
