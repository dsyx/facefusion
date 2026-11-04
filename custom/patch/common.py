import functools
from typing import Any

import cv2
from cv2.typing import Size

from facefusion import content_analyser, face_helper, face_landmarker
from facefusion.types import FaceLandmark5, Matrix, WarpTemplate

_INITIALIZED: bool = False


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	bypass_content_analyser()
	fix_face_alignment()

	_INITIALIZED = True


def bypass_content_analyser() -> None:
	def mock_analysis(*args: Any, **kwargs: Any) -> bool:
		return False

	overrides = {
		'pre_check': lambda: True,
		'analyse_image': functools.lru_cache(mock_analysis),
		'analyse_video': functools.lru_cache(mock_analysis),
		'analyse_stream': mock_analysis,
		'analyse_frame': mock_analysis,
		'detect_nsfw': mock_analysis,
	}

	for name, func in overrides.items():
		setattr(content_analyser, name, func)


def fix_face_alignment() -> None:
	@functools.wraps(face_helper.estimate_matrix_by_face_landmark_5)
	def estimate_matrix_by_face_landmark_5(
		face_landmark_5: FaceLandmark5,
		warp_template: WarpTemplate,
		crop_size: Size,
	) -> Matrix:
		warp_template_norm = face_helper.WARP_TEMPLATE_SET[warp_template] * crop_size
		affine_matrix = cv2.estimateAffinePartial2D(
			face_landmark_5,
			warp_template_norm,
			method=cv2.RANSAC,
			ransacReprojThreshold=max(100, crop_size[0]),
		)[0]
		return affine_matrix

	for module in (face_helper, face_landmarker):
		setattr(
			module,
			estimate_matrix_by_face_landmark_5.__name__,
			estimate_matrix_by_face_landmark_5,
		)
