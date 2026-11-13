import threading

from onnxruntime import InferenceSession

import facefusion.choices
from facefusion import inference_manager
from facefusion.filesystem import get_file_name
from facefusion.types import DownloadSet, InferenceOptionSet, InferencePool, InferenceProvider

_INITIALIZED = False
_INFERENCE_POOL_LOCKS: dict[tuple[str, tuple[str, ...]], threading.Lock] = {}
_INFERENCE_POOL_LOCKS_GUARD = threading.Lock()

TENSORRT_FP16_MODELS: frozenset[str] = frozenset(
	{
		'hyperswap_1a_256',
		'hyperswap_1b_256',
		'hyperswap_1c_256',
	}
)


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	enable_tensorrt_fp16()
	enable_inference_pool_lock()

	_INITIALIZED = True


def adjust_inference_providers(
	model_path: str,
	inference_providers: list[InferenceProvider],
) -> list[InferenceProvider]:
	if not model_path or not inference_providers:
		return inference_providers

	if get_file_name(model_path) not in TENSORRT_FP16_MODELS:
		return inference_providers

	tensorrt_provider = facefusion.choices.execution_provider_set.get('tensorrt')
	adjusted_providers: list[InferenceProvider] = []

	for inference_provider in inference_providers:
		if (
			isinstance(inference_provider, tuple)
			and len(inference_provider) == 2
			and inference_provider[0] == tensorrt_provider
			and isinstance(inference_provider[1], dict)
		):
			# Clone to avoid mutating @lru_cache provider options.
			inference_option_set: InferenceOptionSet = dict(inference_provider[1])
			inference_option_set['trt_fp16_enable'] = True
			adjusted_providers.append((inference_provider[0], inference_option_set))
		else:
			adjusted_providers.append(inference_provider)

	return adjusted_providers


def enable_tensorrt_fp16() -> None:
	original_create_inference_session = inference_manager.create_inference_session

	def create_inference_session(
		model_path: str,
		inference_providers: list[InferenceProvider],
	) -> InferenceSession:
		adjusted_providers = adjust_inference_providers(model_path, inference_providers)
		return original_create_inference_session(model_path, adjusted_providers)

	setattr(inference_manager, 'create_inference_session', create_inference_session)


def resolve_inference_pool_lock(module_name: str, model_names: list[str]) -> threading.Lock:
	pool_key = (module_name, tuple(model_names))
	with _INFERENCE_POOL_LOCKS_GUARD:
		pool_lock = _INFERENCE_POOL_LOCKS.get(pool_key)
		if pool_lock is None:
			pool_lock = threading.Lock()
			_INFERENCE_POOL_LOCKS[pool_key] = pool_lock
		return pool_lock


def enable_inference_pool_lock() -> None:
	original_get_inference_pool = inference_manager.get_inference_pool
	original_clear_inference_pool = inference_manager.clear_inference_pool

	def get_inference_pool(
		module_name: str,
		model_names: list[str],
		model_source_set: DownloadSet,
	) -> InferencePool:
		with resolve_inference_pool_lock(module_name, model_names):
			return original_get_inference_pool(module_name, model_names, model_source_set)

	def clear_inference_pool(module_name: str, model_names: list[str]) -> None:
		with resolve_inference_pool_lock(module_name, model_names):
			original_clear_inference_pool(module_name, model_names)

	setattr(inference_manager, 'get_inference_pool', get_inference_pool)
	setattr(inference_manager, 'clear_inference_pool', clear_inference_pool)
