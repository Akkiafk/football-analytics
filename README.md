# ⚽ Football Analytics using YOLO + OpenCV

<p align="center">

Real-time football player tracking and match analytics using computer vision.

</p>

---

## Overview

This project is an end-to-end football analytics pipeline built using **YOLO**, **OpenCV**, and **Optical Flow**.

The system detects and tracks players, referees, goalkeepers, and the ball from broadcast football footage and performs multiple analytics including:

- Player Tracking
- Ball Tracking
- Ball Possession Detection
- Team Classification
- Camera Motion Compensation
- Perspective Transformation
- Player Speed Estimation
- Distance Covered
- Ball Control Statistics

The project is designed to work directly on football match videos and generate an annotated analytics video.

---

## Features

✅ Player Detection

✅ Ball Detection

✅ Goalkeeper Detection

✅ Referee Detection

✅ Team Color Classification

✅ Ball Possession Tracking

✅ Camera Motion Estimation

✅ View Transformation

✅ Speed Estimation

✅ Distance Covered

✅ Team Ball Control %

---

# Demo

## Sample Input

<p align="center">

<video src="assets/input.mp4"></video>

</p>

---

## Output

<p align="center">

<video src="assets/output.mp4"></video>

</p>

---

## Screenshots

### Player Tracking

![Tracking](assets/tracking.png)

### Ball Possession

![Possession](assets/ball_control.png)

### Speed Estimation

![Speed](assets/speed.png)

---

# Model Used

### Object Detection

- YOLOv11 Nano (Ultralytics)

Used for detecting:

- Players
- Ball
- Referees
- Goalkeepers

### Tracking

- ByteTrack

Used to maintain player identities across frames.

### Optical Flow

Lucas-Kanade Optical Flow is used for estimating camera motion between consecutive frames.

### Team Classification

Dominant jersey colors are extracted using KMeans clustering and assigned to the nearest team centroid.

### Goalkeeper Assignment

Goalkeepers are assigned to teams based on spatial proximity to the average player positions.

---

# Tech Stack

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas
- Scikit-Learn

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/football-analytics.git

cd football-analytics
```

Create virtual environment

```bash
python -m venv ml_env
```

Activate environment

### Windows

```bash
ml_env\Scripts\activate
```

### macOS/Linux

```bash
source ml_env/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the YOLO model

Place the trained model inside

```
models/
    yolo26n.pt
```

---

# Running

Place your input video inside

```
input_videos/
```

Update

```python
VIDEO_NAME = "your_video_name"
```

Run

```bash
python main.py
```

Output will be generated inside

```
output_videos/
```

---

# Project Structure

```
football-analytics/

camera_movement_estimator/

player_ball_assigner/

speed_and_distance_estimator/

team_assigner/

team_differentiation/

tracking/

utils/

view_transformer/

models/

input_videos/

output_videos/

main.py

requirements.txt
```

---

# Future Improvements

- Player Re-identification
- Jersey Number Recognition
- Pass Detection
- Heatmaps
- Expected Goals (xG)
- Formation Detection
- Tactical Analysis

---

# Acknowledgements

- Ultralytics YOLO
- OpenCV
- ByteTrack
