"""
Virtual Steering Wheel Pro v2.0 – High-Performance Edition
==========================================================
Optimized for:
  1. Real PC games (DirectX / DirectInput – NFS Most Wanted, GTA, Forza, etc.)
  2. Web browser games (WASD & Arrow Keys)

Performance Architecture:
  - Threaded webcam capture (overlaps I/O with inference → nearly doubles FPS)
  - Pre-computed hand connection index table (eliminates per-frame iteration)
  - ROI-based semi-transparent HUD (avoids full-frame numpy copy)
  - Numpy-vectorized landmark extraction

Accuracy Architecture:
  - One-Euro Filter on steering angle (adaptive jitter removal, instant responsiveness)
  - One-Euro Filter on each thumb signal (eliminates false gas/brake triggers)
  - Debounced state machine for gas/brake (requires N consecutive frames to switch)
  - Confidence-gated hand detection (rejects low-quality detections)
  - Hand separation validation (rejects overlapping or impossibly far hands)
  - Steering angle auto-calibration with manual re-zero hotkey
  - Dual thumb metrics (vector alignment + curl distance, combined for robustness)
"""

import os
import sys
import time
import math
import threading
import collections
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

# ───────────────────────────────────────────────
# INPUT SYSTEM
# ───────────────────────────────────────────────

USE_DIRECTINPUT = False
try:
    import pydirectinput
    pydirectinput.PAUSE = 0.0
    pydirectinput.FAILSAFE = False
    USE_DIRECTINPUT = True
except ImportError:
    pass

import pyautogui
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False


class KeyController:
    """Thread-safe key dispatcher with scan-code support."""

    def __init__(self):
        self.active_keys = set()
        self._lock = threading.Lock()

    def key_down(self, key):
        with self._lock:
            if key not in self.active_keys:
                try:
                    if USE_DIRECTINPUT:
                        pydirectinput.keyDown(key)
                    else:
                        pyautogui.keyDown(key)
                except Exception:
                    pass
                self.active_keys.add(key)

    def key_up(self, key):
        with self._lock:
            if key in self.active_keys:
                try:
                    if USE_DIRECTINPUT:
                        pydirectinput.keyUp(key)
                    else:
                        pyautogui.keyUp(key)
                except Exception:
                    pass
                self.active_keys.discard(key)

    def release_all(self):
        with self._lock:
            for key in list(self.active_keys):
                try:
                    if USE_DIRECTINPUT:
                        pydirectinput.keyUp(key)
                    else:
                        pyautogui.keyUp(key)
                except Exception:
                    pass
            self.active_keys.clear()


# ───────────────────────────────────────────────
# THREADED WEBCAM CAPTURE
# ───────────────────────────────────────────────

class WebcamCapture:
    """Reads frames in a background thread so the main loop never blocks on I/O."""

    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame = None
        self._grabbed = False
        self._running = True

        # Pre-read first frame
        self._grabbed, self._frame = self.cap.read()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            grabbed, frame = self.cap.read()
            with self._lock:
                self._grabbed = grabbed
                self._frame = frame

    def read(self):
        with self._lock:
            return self._grabbed, self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


# ───────────────────────────────────────────────
# ONE-EURO FILTER (Adaptive low-pass filter)
# ───────────────────────────────────────────────

class OneEuroFilter:
    """
    Eliminates jitter on steady signals while preserving responsiveness to fast changes.
    Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems", CHI 2012.
    """

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff  # Minimum cutoff frequency (lower → smoother when still)
        self.beta = beta              # Speed coefficient (higher → more responsive to fast moves)
        self.d_cutoff = d_cutoff      # Cutoff for derivative filter
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def _smoothing_factor(self, te, cutoff):
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        te = t - self._t_prev
        if te <= 0:
            return self._x_prev

        # Filtered derivative
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self._x_prev) / te
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filtered signal
        a = self._smoothing_factor(te, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


# ───────────────────────────────────────────────
# DEBOUNCED GESTURE STATE
# ───────────────────────────────────────────────

class DebouncedGesture:
    """
    Requires N consecutive frames of a new state before switching.
    Eliminates single-frame glitches from flipping gas/brake/nitro.
    """

    def __init__(self, initial=False, frames_to_activate=3, frames_to_deactivate=2):
        self.state = initial
        self.frames_to_activate = frames_to_activate
        self.frames_to_deactivate = frames_to_deactivate
        self._counter = 0
        self._pending_state = initial

    def update(self, raw_active: bool) -> bool:
        if raw_active == self._pending_state:
            self._counter += 1
        else:
            self._pending_state = raw_active
            self._counter = 1

        threshold = self.frames_to_activate if raw_active else self.frames_to_deactivate

        if self._pending_state != self.state and self._counter >= threshold:
            self.state = self._pending_state
            self._counter = 0

        return self.state

    def reset(self):
        self.state = False
        self._counter = 0
        self._pending_state = False


# ───────────────────────────────────────────────
# PROFILES
# ───────────────────────────────────────────────

PROFILES = {
    "WASD (Browser / Default)": {
        "left": "a", "right": "d", "accel": "w", "brake": "s", "nitro": "space",
    },
    "ARROWS (NFS Most Wanted)": {
        "left": "left", "right": "right", "accel": "up", "brake": "down", "nitro": "shiftleft",
    },
}

profile_names = list(PROFILES.keys())
current_profile_idx = 0
swap_hands = False

# ───────────────────────────────────────────────
# MEDIAPIPE MODEL
# ───────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found. Downloading hand_landmarker.task from Google Storage...")
    import urllib.request
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to auto-download model: {e}\n"
            f"Please download manually from: {MODEL_URL}\n"
            f"and place it at: {MODEL_PATH}"
        )

# Pre-compute hand connection indices once (avoid iterating the protobuf every frame)
HAND_CONNECTIONS_LIST = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.45,
    min_tracking_confidence=0.45,
)
landmarker = vision.HandLandmarker.create_from_options(options)

# ───────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────

DEADZONE_DEG = 6.0
MAX_STEER_DEG = 38.0
PWM_CYCLE_SEC = 0.06         # 60ms PWM period (faster pulses → smoother in-game steering)

# Thumb detection thresholds (combined metric: alignment + curl)
THUMB_PRESS_THRESHOLD = 0.25   # Combined score below this → pressed
THUMB_RELEASE_THRESHOLD = 0.45 # Combined score above this → released

# Nitro gesture
NITRO_GESTURE_THRESHOLD = 0.85

# Hand validation
MIN_HAND_SEPARATION = 0.08   # Minimum normalized X distance between hands
MAX_HAND_SEPARATION = 0.85   # Maximum normalized X distance between hands

# Calibration
steering_offset = 0.0        # Auto-calibration offset subtracted from raw angle

# ───────────────────────────────────────────────
# FILTERS & STATE
# ───────────────────────────────────────────────

controller = KeyController()

# One-Euro filters for each signal
angle_filter = OneEuroFilter(min_cutoff=1.5, beta=0.9, d_cutoff=1.0)
accel_thumb_filter = OneEuroFilter(min_cutoff=2.5, beta=0.3, d_cutoff=1.0)
brake_thumb_filter = OneEuroFilter(min_cutoff=2.5, beta=0.3, d_cutoff=1.0)

# Debounced state machines for gestures
accel_debounce = DebouncedGesture(initial=False, frames_to_activate=3, frames_to_deactivate=2)
brake_debounce = DebouncedGesture(initial=False, frames_to_activate=3, frames_to_deactivate=2)
nitro_debounce = DebouncedGesture(initial=False, frames_to_activate=3, frames_to_deactivate=2)

steering_mode = "PWM_SMOOTH"
nitro_gesture_enabled = True
pressing_accel = False
pressing_brake = False
pressing_nitro = False

last_hands_seen_time = time.time()
HAND_LOSS_GRACE_SEC = 0.18

# FPS tracking
fps_timer = time.time()
fps_counter = 0
current_fps = 0

# Calibration state
calibration_samples = collections.deque(maxlen=30)
calibrated = False

# ───────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────

def extract_landmarks_np(landmarks, w, h):
    """Vectorized landmark extraction to numpy array → shape (21, 2)."""
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.int32)
    return pts


def get_grip_center(landmarks, w, h):
    """Grip center = 75% Middle MCP + 25% Wrist (the natural pivot of a steering grip)."""
    cx = int((landmarks[9].x * 0.75 + landmarks[0].x * 0.25) * w)
    cy = int((landmarks[9].y * 0.75 + landmarks[0].y * 0.25) * h)
    return cx, cy


def get_thumb_combined_score(landmarks):
    """
    Combined thumb metric using two independent signals:
      1. Vector alignment: cos(angle between hand axis and thumb axis)
         +1 = thumb UP, -1 = thumb DOWN
      2. Curl distance: normalized distance from Thumb Tip to Index MCP
         Small = curled in (pressed), Large = extended out

    The combined score is normalized to [0, 1]:
      0.0 = definitely pressed (thumb down + curled)
      1.0 = definitely released (thumb up + extended)
    """
    # Hand axis: Wrist(0) → Middle MCP(9)
    hx = landmarks[9].x - landmarks[0].x
    hy = landmarks[9].y - landmarks[0].y
    h_len = math.hypot(hx, hy) + 1e-6

    # Thumb axis: Thumb MCP(2) → Thumb Tip(4)
    tx = landmarks[4].x - landmarks[2].x
    ty = landmarks[4].y - landmarks[2].y
    t_len = math.hypot(tx, ty) + 1e-6

    # 1. Alignment metric: cosine similarity → [-1, +1] → remap to [0, 1]
    alignment = (hx * tx + hy * ty) / (h_len * t_len)
    alignment_01 = (alignment + 1.0) * 0.5  # 0=down, 1=up

    # 2. Curl distance: Thumb Tip(4) to Index MCP(5), normalized by palm scale
    dx = landmarks[4].x - landmarks[5].x
    dy = landmarks[4].y - landmarks[5].y
    curl_dist = math.hypot(dx, dy) / h_len
    # Typical range: 0.1 (curled tight) to 0.8 (extended).
    # Remap to [0, 1]: values below 0.2 → 0, values above 0.7 → 1
    curl_01 = max(0.0, min(1.0, (curl_dist - 0.2) / 0.5))

    # Weighted combination: alignment is the primary signal, curl is secondary confirmation
    combined = alignment_01 * 0.65 + curl_01 * 0.35
    return combined


def is_pinky_extended(landmarks):
    """Pinky extension for nitro trigger."""
    hx = landmarks[9].x - landmarks[0].x
    hy = landmarks[9].y - landmarks[0].y
    scale = math.hypot(hx, hy) + 1e-6
    dx = landmarks[20].x - landmarks[17].x
    dy = landmarks[20].y - landmarks[17].y
    return (math.hypot(dx, dy) / scale) > NITRO_GESTURE_THRESHOLD


def validate_hand_pair(left_lm, right_lm):
    """
    Reject invalid hand detections:
    - Hands too close (overlapping detection artifacts)
    - Hands impossibly far apart
    """
    sep = abs(right_lm[9].x - left_lm[9].x)
    return MIN_HAND_SEPARATION < sep < MAX_HAND_SEPARATION


def draw_hand_skeleton(frame, pts, color):
    """Draw hand skeleton from pre-computed numpy points array."""
    for s, e in HAND_CONNECTIONS_LIST:
        cv2.line(frame, tuple(pts[s]), tuple(pts[e]), color, 1, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(frame, tuple(pts[i]), 2, color, -1)


def draw_steering_wheel(frame, center, radius, angle, color=(0, 240, 255)):
    """Racing wheel with rotating spokes."""
    cx, cy = center
    cv2.circle(frame, (cx, cy), radius, (30, 30, 30), 7)
    cv2.circle(frame, (cx, cy), radius, color, 2)
    cv2.circle(frame, (cx, cy), 14, (40, 40, 40), -1)
    cv2.circle(frame, (cx, cy), 14, (0, 200, 255), 2)

    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    r1 = radius - 4

    cv2.line(frame, (cx, cy), (int(cx - r1 * c), int(cy - r1 * s)), color, 3)
    cv2.line(frame, (cx, cy), (int(cx + r1 * c), int(cy + r1 * s)), color, 3)
    cv2.line(frame, (cx, cy), (int(cx - (r1 - 6) * s), int(cy + (r1 - 6) * c)), color, 2)

    # 12-o'clock red notch
    cv2.circle(frame, (int(cx + radius * s), int(cy - radius * c)), 4, (0, 0, 255), -1)


def draw_hud(frame, angle, accel_score, brake_score, nitro_active, profile_name, swap_h):
    """Compact racing telemetry HUD with ROI-based semi-transparency."""
    h, w = frame.shape[:2]

    # ROI semi-transparent top bar (avoids full-frame copy)
    top_roi = frame[0:55, 0:w]
    dark = np.full_like(top_roi, (15, 15, 15), dtype=np.uint8)
    cv2.addWeighted(dark, 0.70, top_roi, 0.30, 0, top_roi)
    frame[0:55, 0:w] = top_roi

    # ROI semi-transparent bottom bar
    bot_roi = frame[h - 30:h, 0:w]
    dark_bot = np.full_like(bot_roi, (15, 15, 15), dtype=np.uint8)
    cv2.addWeighted(dark_bot, 0.70, bot_roi, 0.30, 0, bot_roi)
    frame[h - 30:h, 0:w] = bot_roi

    # Title
    cv2.putText(frame, "VIRTUAL WHEEL PRO v2", (12, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 240, 255), 1)

    hand_str = "R=Gas L=Brk" if swap_h else "L=Gas R=Brk"
    cv2.putText(frame, f"PROFILE: {profile_name}", (12, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    info = f"{steering_mode} | {hand_str} | FPS: {current_fps}"
    cv2.putText(frame, info, (w - 280, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1)

    backend = "DirectInput" if USE_DIRECTINPUT else "PyAutoGUI"
    cv2.putText(frame, f"BACKEND: {backend}", (w - 280, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # Gauges
    bar_w, bar_h = 130, 14
    gas_x = 15 if not swap_h else w - 150
    brk_x = w - 150 if not swap_h else 15
    gy = 72

    # Gas gauge
    cv2.rectangle(frame, (gas_x, gy), (gas_x + bar_w, gy + bar_h), (35, 35, 35), -1)
    g_fill = int(max(0.0, min(1.0, 1.0 - accel_score)) * bar_w)
    gc = (0, 255, 0) if pressing_accel else (0, 120, 0)
    if g_fill > 0:
        cv2.rectangle(frame, (gas_x, gy), (gas_x + g_fill, gy + bar_h), gc, -1)
    cv2.rectangle(frame, (gas_x, gy), (gas_x + bar_w, gy + bar_h), (100, 100, 100), 1)
    gl = "GAS ACTIVE" if pressing_accel else "GAS"
    cv2.putText(frame, gl, (gas_x, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gc, 1)

    # Brake gauge
    cv2.rectangle(frame, (brk_x, gy), (brk_x + bar_w, gy + bar_h), (35, 35, 35), -1)
    b_fill = int(max(0.0, min(1.0, 1.0 - brake_score)) * bar_w)
    bc = (0, 0, 255) if pressing_brake else (0, 0, 120)
    if b_fill > 0:
        cv2.rectangle(frame, (brk_x, gy), (brk_x + b_fill, gy + bar_h), bc, -1)
    cv2.rectangle(frame, (brk_x, gy), (brk_x + bar_w, gy + bar_h), (100, 100, 100), 1)
    bl = "BRAKE ACTIVE" if pressing_brake else "BRAKE"
    cv2.putText(frame, bl, (brk_x, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bc, 1)

    # Steering direction
    ai = int(round(angle))
    if angle < -DEADZONE_DEG:
        sd, sc = f"< LEFT {abs(ai)} deg", (0, 255, 120)
    elif angle > DEADZONE_DEG:
        sd, sc = f"RIGHT {abs(ai)} deg >", (0, 255, 120)
    else:
        sd, sc = "CENTER", (200, 200, 200)
    cv2.putText(frame, sd, (w // 2 - 65, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 2)

    if nitro_active:
        cv2.putText(frame, ">>> NITRO <<<", (w // 2 - 70, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 100, 0), 2)

    help_t = "[M]Profile [H]Swap [T]Steer [N]Nitro [C]Calibrate [ESC]Quit"
    cv2.putText(frame, help_t, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────

print("=" * 62)
print("  VIRTUAL STEERING WHEEL PRO v2.0 (High-Performance Edition)")
print("=" * 62)
print(f"  Backend : {'PyDirectInput (DirectX/NFS Ready)' if USE_DIRECTINPUT else 'PyAutoGUI'}")
print(f"  Filter  : One-Euro (adaptive jitter removal)")
print(f"  Webcam  : Threaded capture (non-blocking)")
print("-" * 62)
print("  Controls:")
print("    Tilt hands LEFT/RIGHT  -> Steer")
print("    Left thumb DOWN        -> Gas (W / Up)")
print("    Right thumb DOWN       -> Brake (S / Down)")
print("    Right pinky EXTEND     -> Nitro (Space / Shift)")
print("-" * 62)
print("  Hotkeys (in window):")
print("    [M] Profile   [H] Swap Hands   [T] Steer Mode")
print("    [N] Nitro      [C] Calibrate     [ESC] Quit")
print("=" * 62)
print("\n  Hold both hands steady for 1 second -> auto-calibration...\n")

webcam = WebcamCapture(src=0, width=640, height=480)
start_time = time.time()

try:
    while True:
        ret, raw_frame = webcam.read()
        if not ret or raw_frame is None:
            print("Webcam frame acquisition failed.")
            break

        now = time.time()

        # FPS
        fps_counter += 1
        if now - fps_timer >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_timer = now

        # Prepare frame
        frame = cv2.resize(raw_frame, (640, 480))
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts_ms = int((now - start_time) * 1000)
        results = landmarker.detect_for_video(mp_image, ts_ms)

        # Profile keys
        ap = PROFILES[profile_names[current_profile_idx]]
        k_left, k_right = ap["left"], ap["right"]
        k_accel, k_brake, k_nitro = ap["accel"], ap["brake"], ap["nitro"]

        # Hand detection
        has_two = (results.hand_landmarks is not None
                   and len(results.hand_landmarks) >= 2)

        if has_two:
            # Sort by screen X to fix handedness
            sorted_h = sorted(results.hand_landmarks[:2], key=lambda lm: lm[9].x)
            left_lm, right_lm = sorted_h[0], sorted_h[1]

            # Validate hand pair
            if not validate_hand_pair(left_lm, right_lm):
                has_two = False

        if has_two:
            last_hands_seen_time = now

            # Extract pixel coordinates (numpy-vectorized)
            left_pts = extract_landmarks_np(left_lm, w, h)
            right_pts = extract_landmarks_np(right_lm, w, h)

            left_cx, left_cy = get_grip_center(left_lm, w, h)
            right_cx, right_cy = get_grip_center(right_lm, w, h)

            # Draw skeletons
            draw_hand_skeleton(frame, left_pts, (0, 170, 255))
            draw_hand_skeleton(frame, right_pts, (0, 255, 170))

            # Steering angle
            cv2.line(frame, (left_cx, left_cy), (right_cx, right_cy), (100, 255, 255), 2)
            dx = right_cx - left_cx
            dy = right_cy - left_cy
            raw_angle = math.degrees(math.atan2(dy, dx))

            # Auto-calibration: collect samples for the first ~1 second
            if not calibrated:
                calibration_samples.append(raw_angle)
                if len(calibration_samples) >= 25:
                    # Use median of calibration samples as zero offset
                    steering_offset = float(np.median(calibration_samples))
                    calibrated = True
                    print(f"  [Auto-Calibrated] Steering zero-offset = {steering_offset:.1f}°")
                    angle_filter.reset()

            # Apply calibration offset
            corrected_angle = raw_angle - steering_offset

            # One-Euro filtered angle
            filtered_angle = angle_filter(corrected_angle, now)

            # Draw wheel
            wheel_r = max(55, min(110, int(math.hypot(dx, dy) * 0.42)))
            draw_steering_wheel(frame, ((left_cx + right_cx) // 2, (left_cy + right_cy) // 2),
                                wheel_r, filtered_angle)

            # ──── STEERING ────
            angle_mag = abs(filtered_angle)

            if angle_mag < DEADZONE_DEG:
                controller.key_up(k_left)
                controller.key_up(k_right)
            else:
                tgt = k_left if filtered_angle < 0 else k_right
                opp = k_right if filtered_angle < 0 else k_left
                controller.key_up(opp)

                if steering_mode == "DIRECT_HOLD":
                    controller.key_down(tgt)
                else:
                    intensity = min(1.0, (angle_mag - DEADZONE_DEG) / (MAX_STEER_DEG - DEADZONE_DEG))
                    if intensity >= 0.80:
                        controller.key_down(tgt)
                    else:
                        t_cyc = now % PWM_CYCLE_SEC
                        if t_cyc < PWM_CYCLE_SEC * intensity:
                            controller.key_down(tgt)
                        else:
                            controller.key_up(tgt)

            # ──── GAS & BRAKE ────
            accel_hand = left_lm if not swap_hands else right_lm
            brake_hand = right_lm if not swap_hands else left_lm

            raw_accel_score = get_thumb_combined_score(accel_hand)
            raw_brake_score = get_thumb_combined_score(brake_hand)

            # One-Euro filter each thumb signal
            accel_score = accel_thumb_filter(raw_accel_score, now)
            brake_score = brake_thumb_filter(raw_brake_score, now)

            # Debounced state transitions
            raw_accel_active = accel_score < THUMB_PRESS_THRESHOLD
            raw_brake_active = brake_score < THUMB_PRESS_THRESHOLD
            raw_accel_inactive = accel_score > THUMB_RELEASE_THRESHOLD
            raw_brake_inactive = brake_score > THUMB_RELEASE_THRESHOLD

            # Update debounced state
            accel_want = accel_debounce.update(raw_accel_active and not raw_accel_inactive)
            brake_want = brake_debounce.update(raw_brake_active and not raw_brake_inactive)

            # Apply gas
            if accel_want and not pressing_accel:
                controller.key_down(k_accel)
                pressing_accel = True
            elif not accel_want and pressing_accel:
                controller.key_up(k_accel)
                pressing_accel = False

            # Apply brake
            if brake_want and not pressing_brake:
                controller.key_down(k_brake)
                pressing_brake = True
            elif not brake_want and pressing_brake:
                controller.key_up(k_brake)
                pressing_brake = False

            # Thumb tip indicators
            accel_pts = extract_landmarks_np(accel_hand, w, h)
            brake_pts = extract_landmarks_np(brake_hand, w, h)
            a_tip = tuple(accel_pts[4])
            b_tip = tuple(brake_pts[4])

            ac = (0, 255, 0) if pressing_accel else (0, 100, 0)
            bc = (0, 0, 255) if pressing_brake else (0, 0, 100)
            cv2.circle(frame, a_tip, 7 if pressing_accel else 4, ac, -1)
            cv2.circle(frame, b_tip, 7 if pressing_brake else 4, bc, -1)

            # Nitro
            nitro_active = False
            if nitro_gesture_enabled:
                raw_nitro = is_pinky_extended(right_lm)
                nitro_want = nitro_debounce.update(raw_nitro)
                if nitro_want:
                    nitro_active = True
                    if not pressing_nitro:
                        controller.key_down(k_nitro)
                        pressing_nitro = True
                else:
                    if pressing_nitro:
                        controller.key_up(k_nitro)
                        pressing_nitro = False

            draw_hud(frame, filtered_angle, accel_score, brake_score,
                     nitro_active, profile_names[current_profile_idx], swap_hands)

        else:
            # No valid hand pair
            if now - last_hands_seen_time > HAND_LOSS_GRACE_SEC:
                controller.release_all()
                pressing_accel = False
                pressing_brake = False
                pressing_nitro = False
                accel_debounce.reset()
                brake_debounce.reset()
                nitro_debounce.reset()

                cv2.rectangle(frame, (w // 2 - 170, h // 2 - 30),
                              (w // 2 + 170, h // 2 + 30), (15, 15, 15), -1)
                cv2.rectangle(frame, (w // 2 - 170, h // 2 - 30),
                              (w // 2 + 170, h // 2 + 30), (0, 80, 255), 2)
                cv2.putText(frame, "HOLD UP BOTH HANDS", (w // 2 - 145, h // 2 + 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 150, 255), 2)

            draw_hud(frame, 0.0, 0.8, 0.8, False,
                     profile_names[current_profile_idx], swap_hands)

        cv2.imshow("Virtual Steering Wheel Pro", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord('m'), ord('M')):
            controller.release_all()
            pressing_accel = pressing_brake = pressing_nitro = False
            accel_debounce.reset()
            brake_debounce.reset()
            nitro_debounce.reset()
            current_profile_idx = (current_profile_idx + 1) % len(profile_names)
            print(f"  [Profile] {profile_names[current_profile_idx]}")
        elif key in (ord('h'), ord('H')):
            controller.release_all()
            pressing_accel = pressing_brake = False
            accel_debounce.reset()
            brake_debounce.reset()
            accel_thumb_filter.reset()
            brake_thumb_filter.reset()
            swap_hands = not swap_hands
            print(f"  [Hands] {'R=Gas L=Brake' if swap_hands else 'L=Gas R=Brake'}")
        elif key in (ord('t'), ord('T')):
            controller.release_all()
            steering_mode = "DIRECT_HOLD" if steering_mode == "PWM_SMOOTH" else "PWM_SMOOTH"
            print(f"  [Steer] {steering_mode}")
        elif key in (ord('n'), ord('N')):
            nitro_gesture_enabled = not nitro_gesture_enabled
            if not nitro_gesture_enabled and pressing_nitro:
                controller.key_up(k_nitro)
                pressing_nitro = False
            nitro_debounce.reset()
            print(f"  [Nitro] {'ON' if nitro_gesture_enabled else 'OFF'}")
        elif key in (ord('c'), ord('C')):
            # Manual re-calibration
            calibrated = False
            calibration_samples.clear()
            angle_filter.reset()
            steering_offset = 0.0
            print("  [Calibrate] Hold hands straight... collecting samples...")

except KeyboardInterrupt:
    print("\n  Interrupt received.")

finally:
    print("\n  Shutting down... Releasing all keys.")
    controller.release_all()
    landmarker.close()
    webcam.release()
    cv2.destroyAllWindows()
    print("  Virtual Steering Wheel Pro v2.0 stopped.")
