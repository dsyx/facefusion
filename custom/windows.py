import contextlib
import os
import site
import sys
from pathlib import Path

_INITIALIZED: bool = False


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	setup_terminal_encoding()
	setup_dll_directories()

	_INITIALIZED = True


def setup_terminal_encoding() -> None:
	if hasattr(sys.stdout, 'reconfigure'):
		with contextlib.suppress(Exception):
			sys.stdout.reconfigure(encoding='utf-8')
	if hasattr(sys.stderr, 'reconfigure'):
		with contextlib.suppress(Exception):
			sys.stderr.reconfigure(encoding='utf-8')
	os.environ['PYTHONUTF8'] = '1'


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
