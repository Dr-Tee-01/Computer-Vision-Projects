"""
Hand Tracking Module
Author: Adapted from open-source hand tracking implementations (Mediapipe-based)
Purpose: Provides a reusable class for detecting hands, locating landmarks,
         checking finger states (up/down), and measuring distances.
"""

import cv2
import mediapipe as mp
import time
import math


class HandDetector:
    """
    A wrapper class around Mediapipe's Hands solution.
    Provides higher-level functionality for:
    - Detecting and drawing hands
    - Extracting landmark positions
    - Checking which fingers are raised
    - Measuring distance between two landmarks
    """

    def __init__(self, mode=False, max_hands=2, detection_conf=0.5, track_conf=0.5):
        """
        Initialize the hand detector with customizable parameters.

        Args:
            mode (bool): Static mode (True) or dynamic tracking (False).
            max_hands (int): Maximum number of hands to track.
            detection_conf (float): Minimum detection confidence.
            track_conf (float): Minimum tracking confidence.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        # Load Mediapipe hand solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.track_conf
        )
        self.mp_draw = mp.solutions.drawing_utils  # For rendering hand landmarks
        self.tip_ids = [4, 8, 12, 16, 20]  # Landmark IDs for fingertips

        self.results = None
        self.lm_list = []

    def findHands(self, img, draw=True):
        """
        Process an image to detect hands and optionally draw landmarks.

        Args:
            img (ndarray): BGR image from OpenCV.
            draw (bool): Whether to overlay landmarks on the image.

        Returns:
            img (ndarray): Processed image with drawings (if enabled).
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks,
                                                self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, hand_no=0, draw=True):
        """
        Get pixel coordinates of all landmarks for the selected hand.

        Args:
            img (ndarray): Input image.
            hand_no (int): Index of the hand (if multiple detected).
            draw (bool): Whether to draw circles & bounding box.

        Returns:
            lm_list (list): Landmark list with format [[id, x, y], ...].
            bbox (tuple): Bounding box (xmin, ymin, xmax, ymax).
        """
        x_list, y_list = [], []
        bbox = ()
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, _ = img.shape

            for idx, lm in enumerate(my_hand.landmark):
                # Convert normalized coordinates (0–1) to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([idx, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Compute bounding box around the hand
            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = (xmin, ymin, xmax, ymax)

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingersUp(self):
        """
        Determine which fingers are raised.

        Returns:
            fingers (list[int]): List of 5 integers (1=up, 0=down).
                                 Order: [Thumb, Index, Middle, Ring, Pinky]
        """
        fingers = []

        # Thumb: compare x-coordinates (special case due to orientation)
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers: compare y-coordinates of fingertip vs joint
        for idx in range(1, 5):
            if self.lm_list[self.tip_ids[idx]][2] < self.lm_list[self.tip_ids[idx] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        """
        Compute Euclidean distance between two hand landmarks.

        Args:
            p1 (int): Landmark ID of first point.
            p2 (int): Landmark ID of second point.
            img (ndarray): Input image.
            draw (bool): Whether to draw the line & points.

        Returns:
            length (float): Distance between points.
            img (ndarray): Image with drawings (if enabled).
            info (list): [x1, y1, x2, y2, cx, cy] coordinates.
        """
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    """
    Example usage of HandDetector in real-time with webcam.
    Displays detected hand landmarks and prints coordinates of the thumb tip.
    """
    prev_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)  # Detect hands
        lm_list, _ = detector.findPosition(img)  # Get landmarks

        if lm_list:
            print("Thumb Tip:", lm_list[4])  # Print thumb tip coordinates

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # Display FPS on frame
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()