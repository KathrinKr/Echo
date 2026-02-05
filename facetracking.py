import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pythonosc.udp_client import SimpleUDPClient
osc = SimpleUDPClient("127.0.0.1", 8000)

import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from flask import Flask, Response
import threading
import time

# ---------------- MediaPipe ----------------
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

FACE_NOSE_ID = 1
POSE_NOSE_ID = 0

# ---------------- Bewegung ----------------
MOVEMENT_THRESHOLD = 8
HISTORY_LENGTH = 5
STILL_TIME_REQUIRED = 1.0

movement_history = deque(maxlen=HISTORY_LENGTH)
previous_nose = None
still_start = None

# ---------------- Video ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ---------------- Flask ----------------
app = Flask(__name__)
color_frame = None
mask_frame = None
lock = threading.Lock()

# MJPEG generator
def mjpeg_stream(get_frame):
    while True:
        with lock:
            frame = get_frame()
            if frame is None:
                continue

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")
        time.sleep(1/30)

@app.route("/color")
def color():
    return Response(
        mjpeg_stream(get_color_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/mask")
def mask():
    return Response(
        mjpeg_stream(get_mask_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def get_color_frame():
    return color_frame

def get_mask_frame():
    return mask_frame


def start_flask():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

threading.Thread(target=start_flask, daemon=True).start()

# ---------------- Main Loop ----------------
with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh, \
     mp_pose.Pose(model_complexity=1) as pose, \
     mp_selfie.SelfieSegmentation(model_selection=1) as segment:

    last_still = -1

    while True:
        success, frame = cap.read()
        if not success:
            break

        # ---------------- Zoom nur fürs Tracking ----------------
        ZOOM = 2.0  # Zoom-Faktor

        h, w, _ = frame.shape
        new_w = int(w / ZOOM)
        new_h = int(h / ZOOM)

        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2

        cropped = frame[y1:y1+new_h, x1:x1+new_w]
        frame_zoomed = cv2.resize(cropped, (w, h))

        # ---------------- MediaPipe Inputs ----------------
        rgb_zoomed = cv2.cvtColor(frame_zoomed, cv2.COLOR_BGR2RGB)   # Tracking
        rgb_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        # Maske

        face_results = face_mesh.process(rgb_zoomed)
        pose_results = pose.process(rgb_zoomed)
        seg_results = segment.process(rgb_original)

        # ---------------- Maske erzeugen (Originalbild!) ----------------
        raw_mask = seg_results.segmentation_mask
        soft_mask = cv2.GaussianBlur(raw_mask, (35, 35), 0)
        mask_gray = (soft_mask * 255).astype(np.uint8)

        # Originalbild für Flask behalten
        color_full = frame

        # Frames für Flask bereitstellen
        with lock:
            color_frame = color_full
            mask_frame = mask_gray

        # ---------------- OSC Stillness ----------------
        face_nose = None
        pose_nose = None

        if face_results.multi_face_landmarks:
            lm = face_results.multi_face_landmarks[0].landmark[FACE_NOSE_ID]
            face_nose = (int(lm.x * w), int(lm.y * h))

        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark[POSE_NOSE_ID]
            pose_nose = (int(lm.x * w), int(lm.y * h))

        matched = face_nose and pose_nose and math.dist(face_nose, pose_nose) < 80

        still = 0
        now = time.time()

        if matched and face_nose:
            if previous_nose:
                dist = math.dist(face_nose, previous_nose)
                movement_history.append(1 if dist > MOVEMENT_THRESHOLD else 0)

                if len(movement_history) == HISTORY_LENGTH:
                    if sum(movement_history) <= 1:
                        if still_start is None:
                            still_start = now
                        elif now - still_start > STILL_TIME_REQUIRED:
                            still = 1
                    else:
                        still_start = None
            previous_nose = face_nose
        else:
            previous_nose = None
            movement_history.clear()
            still_start = None

        if still != last_still:
            osc.send_message("/still", still)
            last_still = still

cap.release()
