import io

import base64
import cv2
import imageio
import json
import mediapipe as mp
import numpy as np
import os
import tempfile

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO

from enums.http_methods import HttpMethods
from enums.request_body import RequestBody
from utils.process_landmarks import process_landmarks

app = Flask(__name__)
CORS(app, resources = {
	r"/*": {
		"origins": "*"
	}
})

IMAGES_FOLDER = "assets/images"
VIDEOS_FOLDER = "assets/videos"
MODEL_PATH = "assets/best.pt"

if not os.path.exists(IMAGES_FOLDER):
	os.makedirs(IMAGES_FOLDER)

if not os.path.exists(VIDEOS_FOLDER):
	os.makedirs(VIDEOS_FOLDER)

app.config["IMAGES_FOLDER"] = IMAGES_FOLDER
app.config["VIDEOS_FOLDER"] = VIDEOS_FOLDER
app.config["MODEL_PATH"] = MODEL_PATH

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

@app.route("/estimate-pose-image", methods=[HttpMethods.POST.value])
def estimate_pose_image():
	"""
	Endpoint to estimate the pose in an image sent via a POST request.

	The image should be sent in a form field named "file".
	The image is processed to estimate pose landmarks and the image with the drawn landmarks is returned.

	Returns:
		- If no file is sent: "No file part", 400
		- If no file is selected: "No selected file", 400
		- Processed image with drawn landmarks in JPEG format.
	"""
	if RequestBody.FILE.value not in request.files:
		response = {
			"error": "No file part"
		}

		return jsonify(response), 400

	file = request.files[RequestBody.FILE.value]

	if file.filename == "":
		response = {
			"error": "No selected file"
		}

		return jsonify(response), 400

	file_path = os.path.join(IMAGES_FOLDER, file.filename)
	file.save(file_path)

	# Use the pre-trained model to perform the estimation
	model = YOLO(app.config["MODEL_PATH"])

	results = model(f"{IMAGES_FOLDER}/{file.filename}")
	image_with_poses = results[0].plot()

	yolo_result_path = os.path.join(IMAGES_FOLDER, f"yolo_{file.filename}")
	cv2.imwrite(yolo_result_path, image_with_poses)

	angles = {
		"elbow": {
			"right": 0,
			"left": 0
		}
	}

	# Use MediaPipe to estimate angles.
	with mp_pose.Pose(static_image_mode=True) as pose:
		image = cv2.imread(file_path)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		results = pose.process(image_rgb)

		if results.pose_landmarks is not None:
			coords_and_angles = process_landmarks(results.pose_landmarks.landmark, mp_pose, image)

			image_yolo = cv2.imread(yolo_result_path)

			if (coords_and_angles["elbow"]["right"]["angle"] > 0.0):
				cv2.putText(image_yolo, str(int(coords_and_angles["elbow"]["right"]["angle"])), coords_and_angles["elbow"]["right"]["coords"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

			if (coords_and_angles["elbow"]["left"]["angle"] > 0.0):
				cv2.putText(image_yolo, str(int(coords_and_angles["elbow"]["left"]["angle"])), coords_and_angles["elbow"]["left"]["coords"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
			# if (coords_and_angles["elbow"]["right"]["angle"] > 0.0 and coords_and_angles["elbow"]["left"]["angle"] > 0.0):

			angles["elbow"]["right"] = int(coords_and_angles["elbow"]["right"]["angle"])
			angles["elbow"]["left"] = int(coords_and_angles["elbow"]["left"]["angle"])
		else:
			response = {
				"error": "Person not found"
			}

			return jsonify(response), 400

	_, img_encoded = cv2.imencode(".jpg", image_yolo)
	img_bytes = io.BytesIO(img_encoded.tobytes())

	img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

	response = {
		"file": img_base64,
		"angles": angles
	}

	if os.path.exists(file_path):
		os.remove(file_path)

	if os.path.exists(yolo_result_path):
		os.remove(yolo_result_path)

	return jsonify(response), 200

@app.route("/estimate-pose-video", methods=[HttpMethods.POST.value])
def estimate_pose_video():
	"""
	Endpoint to estimate the pose in an video sent via a POST request.

	The video should be sent in a form field named "file".
	The video is processed to estimate pose landmarks and the video with the drawn landmarks is returned.

	Returns:
		- If no file is sent: "No file part", 400
		- If no file is selected: "No selected file", 400
		- Processed video with drawn landmarks in MP4 format.
	"""
	if RequestBody.FILE.value not in request.files:
		response = {
			"error": "No file part"
		}

		return jsonify(response), 400

	file = request.files[RequestBody.FILE.value]

	if file.filename == "":
		response = {
			"error": "No selected file"
		}

		return jsonify(response), 400

	file_path = os.path.join(VIDEOS_FOLDER, file.filename)
	file.save(file_path)

	model = YOLO(app.config["MODEL_PATH"])

	video_capture = cv2.VideoCapture(file_path)
	frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(video_capture.get(cv2.CAP_PROP_FPS))

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	output_path = os.path.join(VIDEOS_FOLDER, f"yolo_{file.filename}")
	out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

	while True:
		ret, frame = video_capture.read()

		if not ret:
			break

		results = model(frame)  # Assuming model processes frame directly

		processed_frame = results[0].plot()  # Assuming results include a method to plot

		out.write(processed_frame)

	video_capture.release()
	out.release()

	with open(output_path, "rb") as video_file:
		video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

	if os.path.exists(file_path):
		os.remove(file_path)

	if os.path.exists(output_path):
		print("ELIMINA VIDEO RESULTADO")
		os.remove(output_path)

	response = {
		"file": video_base64
	}

	return jsonify(response), 200

if __name__ == "__main__":
	app.run(debug=True)
