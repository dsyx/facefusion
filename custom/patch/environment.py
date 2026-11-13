import contextlib
import os
import site
import sys
from pathlib import Path

_INITIALIZED = False
_DLL_DIRECTORIES: list[object] = []


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	setup_library_paths()

	_INITIALIZED = True


def setup_library_paths() -> None:
	if sys.platform == 'win32':
		site_packages = list(
			dict.fromkeys(
				Path(p).resolve()
				for p in (*sys.path, *site.getsitepackages())
				if p and Path(p).name.lower() == 'site-packages' and Path(p).is_dir()
			)
		)

		candidates = []
		for site_path in site_packages:
			candidates.extend(sorted({dll.parent for dll in site_path.glob('nvidia/**/*.dll')}))
			candidates.append(site_path / 'onnxruntime' / 'capi')
			candidates.append(site_path / 'tensorrt_libs')

		library_paths = list(dict.fromkeys(str(p) for p in candidates if p.is_dir()))

		if library_paths:
			for library_path in library_paths:
				with contextlib.suppress(OSError):
					_DLL_DIRECTORIES.append(os.add_dll_directory(library_path))
			env_paths = os.getenv('PATH', '').split(os.pathsep)
			os.environ['PATH'] = os.pathsep.join(
				dict.fromkeys(p for p in (*library_paths, *env_paths) if p)
			)
