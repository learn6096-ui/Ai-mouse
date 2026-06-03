import cv2
import ctypes
import platform

# --- Windows MediaPipe Fix ---
if platform.system() == 'Windows':
    try:
        msvcrt = ctypes.CDLL('msvcrt')
        original_cdll = ctypes.CDLL
        class MockCDLL(original_cdll):
            def __getattr__(self, name):
                if name == 'free':
                    return msvcrt.free
                return super().__getattr__(name)
        ctypes.CDLL = MockCDLL
    except Exception as e:
        print(f"Failed to apply MediaPipe fix: {e}")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
from screeninfo import get_monitors
import time
import math
import os

# --- PRO CONFIGURATION ---
SENSITIVITY = 1.2
# One-Euro Filter Parameters (Tuning for "Apple Trackpad" feel)
BETA = 0.01   
SAMP_RATE = 60 
MIN_CUTOFF = 1.0
PINCH_THRESHOLD_RATIO = 0.15 
SCROLL_Y_THRESHOLD = 0.2

# Disable PyAutoGUI fail-safe and pause for ultra-responsiveness
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

class OneEuroFilter:
    """Adaptive low-pass filter to remove jitter while maintaining responsiveness."""
    def __init__(self, freq, mincutoff=MIN_CUTOFF, beta=BETA):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.x_prev = None
        self.dx_prev = None

    def _low_pass_filter(self, x, x_prev, alpha):
        return alpha * x + (1 - alpha) * x_prev

    def filter(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            return x
        
        dx = (x - self.x_prev) * self.freq
        edx = self._low_pass_filter(dx, self.dx_prev, self._alpha(self.freq, 1.0))
        self.dx_prev = edx
        
        cutoff = self.mincutoff + self.beta * abs(edx)
        alpha = self._alpha(self.freq, cutoff)
        
        res = self._low_pass_filter(x, self.x_prev, alpha)
        self.x_prev = res
        return res

    def _alpha(self, freq, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

class VirtualCanvas:
    """Manages global multi-monitor coordinate mapping."""
    def __init__(self):
        self.monitors = get_monitors()
        self.min_x = min(m.x for m in self.monitors)
        self.min_y = min(m.y for m in self.monitors)
        self.max_x = max(m.x + m.width for m in self.monitors)
        self.max_y = max(m.y + m.height for m in self.monitors)
        self.total_width = self.max_x - self.min_x
        self.total_height = self.max_y - self.min_y
        print(f"Virtual Workspace Initialized: {self.total_width}x{self.total_height}")

    def map_coords(self, norm_x, norm_y):
        """Maps 0.0-1.0 normalized values to the multi-monitor canvas."""
        abs_x = self.min_x + (norm_x * self.total_width)
        abs_y = self.min_y + (norm_y * self.total_height)
        return abs_x, abs_y

class AirMouseEngine:
    def __init__(self):
        # MediaPipe Tasks Setup
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.canvas = VirtualCanvas()
        
        # Filters for X and Y
        self.filter_x = OneEuroFilter(SAMP_RATE)
        self.filter_y = OneEuroFilter(SAMP_RATE)
        
        self.is_dragging = False
        self.last_time = time.time()

    def get_dist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def draw_landmarks(self, frame, landmarks):
        """Simple landmark drawing using OpenCV."""
        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
        # Draw connections (Simplified)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17) # Palm
        ]
        for start_idx, end_idx in connections:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]
            cv2.line(frame, (int(p1.x * w), int(p1.y * h)), 
                     (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

        print("Engine Running... Press 'q' to stop.")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Use current time in ms for timestamp
            timestamp_ms = int(time.time() * 1000)
            result = self.detector.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                
                # Palm Scale for normalization (distance from wrist to middle finger base)
                palm_size = self.get_dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y))

                # 1. CURSOR MOTION (Index Finger)
                raw_x, raw_y = self.canvas.map_coords(lm[8].x, lm[8].y)
                smooth_x = self.filter_x.filter(raw_x)
                smooth_y = self.filter_y.filter(raw_y)
                
                pyautogui.moveTo(smooth_x, smooth_y)

                # 2. GESTURE ENGINE
                # Left Click / Drag: Index + Middle pinch
                idx_mid_dist = self.get_dist((lm[8].x, lm[8].y), (lm[12].x, lm[12].y))
                if idx_mid_dist < (PINCH_THRESHOLD_RATIO * palm_size):
                    if not self.is_dragging:
                        pyautogui.mouseDown()
                        self.is_dragging = True
                else:
                    if self.is_dragging:
                        pyautogui.mouseUp()
                        self.is_dragging = False

                # Right Click: Index + Ring pinch
                idx_ring_dist = self.get_dist((lm[8].x, lm[8].y), (lm[16].x, lm[16].y))
                if idx_ring_dist < (PINCH_THRESHOLD_RATIO * palm_size):
                    pyautogui.rightClick()
                    time.sleep(0.2)

                # Scroll Up: 3 Fingers extended (Index, Mid, Ring) and moving up
                if lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y < lm[14].y:
                    if lm[8].y < 0.4:
                        pyautogui.scroll(30)

                # Scroll Down: Pinky extension gesture
                if lm[20].y < lm[18].y and lm[16].y > lm[14].y: 
                    pyautogui.scroll(-30)

                # HUD Visualization
                self.draw_landmarks(frame, lm)

            cv2.imshow("AI Air Mouse Pro", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = AirMouseEngine()
    engine.run()
