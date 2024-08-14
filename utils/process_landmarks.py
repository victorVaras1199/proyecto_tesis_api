import numpy as np

from .calculate_angle import calculate_angle


def process_landmarks(landmarks, mp_pose, image):
	# Getting values
	left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
	right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

	left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
	left_elbow_visibility = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility

	right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
	right_elbow_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility

	left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
	right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

	# Getting angles and coords
	if (left_elbow_visibility < 0.5 and right_elbow_visibility < 0.5):
		return {
			"elbow": {
				"right": {
					"angle": 0.0,
					"coords": (0.0, 0.0)
				},
				"left": {
					"angle": 0.0,
					"coords": (0.0, 0.0)
				}
			}
		}

	if (left_elbow_visibility < 0.5):
		right_elbow_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
		right_elbow_coords = tuple(np.multiply(right_elbow, [image.shape[1], image.shape[0]]).astype(int))

		return {
			"elbow": {
				"right": {
					"angle": right_elbow_angle,
					"coords": right_elbow_coords
				},
				"left": {
					"angle": 0.0,
					"coords": (0.0, 0.0)
				}
			}
		}

	if (right_elbow_visibility < 0.5):
		leftt_elbow_angle = calculate_angle(leftt_wrist, leftt_elbow, leftt_shoulder)
		leftt_elbow_coords = tuple(np.multiply(leftt_elbow, [image.shape[1], image.shape[0]]).astype(int))

		return {
			"elbow": {
				"right": {
					"angle": 0.0,
					"coords": (0.0, 0.0)
				},
				"left": {
					"angle": left_elbow_angle,
					"coords": left_elbow_coords
				}
			}
		}

	right_elbow_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
	left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

	left_elbow_coords = tuple(np.multiply(left_elbow, [image.shape[1], image.shape[0]]).astype(int))
	right_elbow_coords = tuple(np.multiply(right_elbow, [image.shape[1], image.shape[0]]).astype(int))

	return {
		"elbow": {
			"right": {
				"angle": right_elbow_angle,
				"coords": right_elbow_coords
			},
			"left": {
				"angle": left_elbow_angle,
				"coords": left_elbow_coords
			}
		}
	}
