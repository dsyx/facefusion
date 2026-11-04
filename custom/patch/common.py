import functools
from typing import Any

import cv2
import numpy
from cv2.typing import Size

from facefusion import (
	content_analyser,
	face_creator,
	face_helper,
	face_landmarker,
	state_manager,
)
from facefusion.types import Face, FaceLandmark5, Matrix, WarpTemplate

_INITIALIZED: bool = False


def setup() -> None:
	global _INITIALIZED
	if _INITIALIZED:
		return

	bypass_content_analyser()
	fix_face_alignment()
	fix_face_filter()

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


def fix_face_filter() -> None:
	original_create_faces = face_creator.create_faces

	@functools.wraps(original_create_faces)
	def create_faces(*args: Any, **kwargs: Any) -> list[Face]:
		faces = original_create_faces(*args, **kwargs)
		face_landmarker_score = state_manager.get_item('face_landmarker_score')
		if not face_landmarker_score or face_landmarker_score <= 0:
			return faces

		valid_faces: list[Face] = []
		for face in faces:
			landmarker_score = face.score_set.get('landmarker', 0.0)
			if landmarker_score < face_landmarker_score:
				continue

			if landmarker_score == face_landmarker_score and '68' in face.landmark_set:
				face.landmark_set['5/68'] = face_helper.convert_to_face_landmark_5(
					face.landmark_set['68']
				)

			landmark_5_68 = face.landmark_set.get('5/68')
			if landmark_5_68 is not None:
				angle_rad = numpy.radians(face.angle)
				eye_dx = (
					(landmark_5_68[1][0] - landmark_5_68[0][0]) * numpy.cos(angle_rad)
					+ (landmark_5_68[1][1] - landmark_5_68[0][1]) * numpy.sin(angle_rad)
				)
				if eye_dx <= 0:
					continue

			valid_faces.append(face)

		return valid_faces

	setattr(face_creator, create_faces.__name__, create_faces)
