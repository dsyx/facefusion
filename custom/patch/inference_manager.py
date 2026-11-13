from onnxruntime import InferenceSession

import facefusion.choices
from facefusion import inference_manager
from facefusion.filesystem import get_file_name
from facefusion.types import InferenceOptionSet, InferenceProvider

_INITIALIZED = False

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
