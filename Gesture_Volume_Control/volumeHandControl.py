import cv2
import time
import numpy as np
import HandTrackingModule as htm   # Custom hand tracking module (likely built with Mediapipe)
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --------------------------- Configuration ---------------------------
CAM_WIDTH, CAM_HEIGHT = 640, 480   # Dimensions of the camera feed
cap = cv2.VideoCapture(0)          # Open webcam (device index 0)
cap.set(3, CAM_WIDTH)              # 3 = width property
cap.set(4, CAM_HEIGHT)             # 4 = height property

prev_time = 0  # For calculating FPS later

# Initialize hand detector (from custom HandTrackingModule)
detector = htm.HandDetector(detection_conf=0.7, max_hands=1)


# --------------------------- Audio Setup ---------------------------
# Use Pycaw to control system audio
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))

# Get system volume range (usually in decibels)
vol_range = volume_ctrl.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]  # Min and max volume levels

# Initialize volume control variables
vol = 0
vol_bar = 400   # Y-coordinate for volume bar
vol_percentage = 0
hand_area = 0
volume_color = (255, 0, 0)  # Default color (red) for UI elements

# --------------------------- Main Loop ---------------------------
while True:
    success, img = cap.read()  # Capture frame from webcam
    if not success:
        break

    # Detect hand and landmarks
    img = detector.findHands(img)  # Draw hand landmarks on frame
    lm_list, bbox = detector.findPosition(img, draw=True)  # Get landmark list & bounding box

    if lm_list:
        # Estimate hand size (area of bounding box)
        hand_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

        # Apply filter: Ignore very small or very large detections
        if 250 < hand_area < 1000:
            # Measure distance between thumb tip (id=4) and index finger tip (id=8)
            length, img, line_info = detector.findDistance(4, 8, img)

            # Map hand distance (50–200 px) to volume range (0–100%)
            vol_bar = np.interp(length, [50, 200], [400, 150])
            vol_percentage = np.interp(length, [50, 200], [0, 100])

            # Smoothen volume percentage (round to nearest 10)
            smoothness = 10
            vol_percentage = smoothness * round(vol_percentage / smoothness)

            # Check which fingers are up
            fingers = detector.fingersUp()

            # If pinky is down → set system volume
            if fingers and not fingers[4]:
                volume_ctrl.SetMasterVolumeLevelScalar(vol_percentage / 100, None)
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                volume_color = (0, 255, 0)  # Green = active change
            else:
                volume_color = (255, 0, 0)  # Red = inactive

    # --------------------------- UI Elements ---------------------------
    # Draw volume bar background
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    # Fill volume bar based on hand distance
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
    # Display percentage text
    cv2.putText(img, f'{int(vol_percentage)} %', (40, 450),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display current system volume (cross-check)
    current_vol = int(volume_ctrl.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol Set: {current_vol}', (400, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, volume_color, 3)

    # --------------------------- FPS Calculation ---------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Display window
    cv2.imshow("Hand Volume Control", img)
    cv2.waitKey(1)