# NOTE: This file is unused in the current client-side inference build.
# Contains model URLs/paths, landmark tables, and body-part config for server-side inference.

"""Constants, landmark index tables, body-part configuration, and model paths."""

import os

import cv2

# ── Frame / composite constants ───────────────────────────────────────────────

INFER_MAX_W = 480
BLUR_RADIUS = 21
OUTLINE_STRENGTH = 0.5
MAX_LOOP_FPS = 30  # per-model inference loop cap

ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# ── Model paths ───────────────────────────────────────────────────────────────
# models/ lives two levels above this file (at the project root)

_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
)
os.makedirs(_MODELS_DIR, exist_ok=True)

SEG_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
)
SEG_PATH = os.path.join(_MODELS_DIR, "selfie_segmenter.tflite")

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_PATH = os.path.join(_MODELS_DIR, "pose_landmarker_lite.task")

HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_PATH = os.path.join(_MODELS_DIR, "hand_landmarker.task")

FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_PATH = os.path.join(_MODELS_DIR, "face_landmarker.task")

# ── Landmark index tables ─────────────────────────────────────────────────────

FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ── Body-part landmark groups ─────────────────────────────────────────────────

TORSO_LM = {11, 12, 23, 24}
L_ARM_LM = {11, 13, 15, 17, 19, 21}
R_ARM_LM = {12, 14, 16, 18, 20, 22}
L_LEG_LM = {23, 25, 27}
R_LEG_LM = {24, 26, 28}
L_FOOT_LM = {27, 29, 31}
R_FOOT_LM = {28, 30, 32}

# (name, landmark set, dilation px, new-frame smooth weight)
PART_CONFIG = [
    ("head", set(range(11)), 30, 0.6),
    ("torso", TORSO_LM, 40, 0.5),
    ("left_arm", L_ARM_LM, 25, 0.85),
    ("right_arm", R_ARM_LM, 25, 0.85),
    ("left_leg", L_LEG_LM, 50, 0.5),
    ("right_leg", R_LEG_LM, 50, 0.5),
    ("left_foot", L_FOOT_LM, 80, 0.9),
    ("right_foot", R_FOOT_LM, 80, 0.9),
]

POSE_CONNECTIONS = [
    (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
    (0, 1), (0, 4), (9, 10), (0, 9), (0, 10), (7, 8),
    (0, 11), (0, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (27, 31), (29, 31),
    (28, 30), (28, 32), (30, 32),
]

PART_CONN = {
    "left_arm":  [(a, b) for a, b in POSE_CONNECTIONS if a in L_ARM_LM  and b in L_ARM_LM],
    "right_arm": [(a, b) for a, b in POSE_CONNECTIONS if a in R_ARM_LM  and b in R_ARM_LM],
    "left_leg":  [(a, b) for a, b in POSE_CONNECTIONS if a in L_LEG_LM  and b in L_LEG_LM],
    "right_leg": [(a, b) for a, b in POSE_CONNECTIONS if a in R_LEG_LM  and b in R_LEG_LM],
    "left_foot": [(a, b) for a, b in POSE_CONNECTIONS if a in L_FOOT_LM and b in L_FOOT_LM],
    "right_foot":[(a, b) for a, b in POSE_CONNECTIONS if a in R_FOOT_LM and b in R_FOOT_LM],
}

PART_NAMES = [name for name, *_ in PART_CONFIG]
