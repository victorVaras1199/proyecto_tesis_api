# Flask Pose Estimation API

This project is a Flask-based web API for pose estimation using MediaPipe and OpenCV. It processes images to detect and draw pose landmarks.

## Table of Contents

- [Flask Pose Estimation API](#flask-pose-estimation-api)
	- [Table of Contents](#table-of-contents)
	- [Introduction](#introduction)
	- [Requirements](#requirements)
	- [Installation](#installation)
	- [Usage](#usage)

## Introduction

The Flask Pose Estimation API allows you to upload an image, processes it to detect human pose landmarks, and returns the image with the landmarks drawn.

## Requirements

- Python 3.7+
- Conda (instead of PIP)
- Flask
- Flask-CORS
- OpenCV
- NumPy
- MediaPipe

## Installation

Note: You need to have installed Conda in your system

1. Clone the repository:

```sh
git clone https://github.com/joshcast777/pose-estimation-api.git
```

2. Navigate to the project directory:

```bash
cd flask-pose-estimation-api
```

3. Create and activate a virtual environment:

```sh
conda create -n <venv_name>

conda activate <venv_name>
```

4. Install the required packages:

```sh
conda install pip

pip install -r requirements.txt
```

5. Deactivate the virtual environment (after stop working)

```sh
conda deactivate
```

## Usage

1. Start the Flask server:

```sh
python app.py
```

1. Use any client to test endpoints, like Postman or Insomnia, and make a `POSt` request to the `/estimate-pose` endpoint, with a body with a property name it `file` and selecting an image.
