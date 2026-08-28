import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

# =============================
# FILES
# =============================
MODEL_PATH = Path(
    r"C:\Users\DELL\Desktop\Master web\play code\play\handsss\hand_landmarker.task"
)
IMAGE_PATH = Path(
    r"inter url of the image "
)

# =============================
# CAMERA SOURCE
# =============================
CAMERA_SOURCE = 1   # Camo Studio

# =============================
# SETTINGS
# =============================
INVERT_HANDEDNESS = False

BORDER_COLOR = (255, 255, 255)   # white
BORDER_THICKNESS = 1
FEATHER_SIZE = 4

THUMB_ID = 4
INDEX_ID = 8

WINDOW_NAME = "LIVE IMAGE REVEAL"

# Detection ek chhoti copy pe chalega -- speed ke liye
DETECTION_SCALE = 0.5   # 0.5 = half resolution pe hand detect hoga

# =============================
# MEDIAPIPE
# =============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# =============================
# LOAD IMAGE
# =============================
overlay = cv2.imread(str(IMAGE_PATH))
if overlay is None:
    print("Image nahi mili!")
    exit()

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("Camera nahi mila!")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("Camera se frame nahi mila!")
    exit()

first_frame = cv2.flip(first_frame, 1)
h, w = first_frame.shape[:2]
image = cv2.resize(overlay, (w, h))

print(f"Camera resolution detected: {w}x{h}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
is_fullscreen = False

start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if frame.shape[1] != w or frame.shape[0] != h:
            h, w = frame.shape[:2]
            image = cv2.resize(overlay, (w, h))

        # ---------------------------------
        # Detection ek downscaled frame pe (fast)
        # ---------------------------------
        small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        sh, sw = small.shape[:2]

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        corners = {}

        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name

                if INVERT_HANDEDNESS:
                    label = "Right" if label == "Left" else "Left"

                thumb = hand_landmarks[THUMB_ID]
                index = hand_landmarks[INDEX_ID]

                # Normalized coords (0-1) -> seedha full-res frame pe scale karo
                thumb_pt = (int(thumb.x * w), int(thumb.y * h))
                index_pt = (int(index.x * w), int(index.y * h))

                if label == "Left":
                    corners["bottom_left"] = thumb_pt
                    corners["top_left"] = index_pt
                else:
                    corners["bottom_right"] = thumb_pt
                    corners["top_right"] = index_pt

        # ---------------------------------
        # Polygon reveal -- sirf bounding box ke andar process
        # ---------------------------------
        if len(corners) == 4:
            poly_points = np.array([
                corners["top_left"],
                corners["top_right"],
                corners["bottom_right"],
                corners["bottom_left"]
            ], dtype=np.int32)

            # Bounding box nikaalo (thoda padding taaki feather cut na jaye)
            pad = FEATHER_SIZE * 3
            x_min = max(0, poly_points[:, 0].min() - pad)
            x_max = min(w, poly_points[:, 0].max() + pad)
            y_min = max(0, poly_points[:, 1].min() - pad)
            y_max = min(h, poly_points[:, 1].max() + pad)

            if x_max > x_min and y_max > y_min:
                roi_w = x_max - x_min
                roi_h = y_max - y_min

                # Polygon points ko ROI ke local coords mein shift karo
                local_poly = poly_points - [x_min, y_min]

                mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
                cv2.fillPoly(mask_roi, [local_poly], 255)

                mask_blurred = cv2.GaussianBlur(
                    mask_roi, (0, 0), sigmaX=FEATHER_SIZE, sigmaY=FEATHER_SIZE
                )
                alpha = (mask_blurred.astype(np.float32) / 255.0)[:, :, None]

                frame_roi = frame[y_min:y_max, x_min:x_max]
                image_roi = image[y_min:y_max, x_min:x_max]

                blended = (frame_roi.astype(np.float32) * (1 - alpha) +
                           image_roi.astype(np.float32) * alpha).astype(np.uint8)

                frame[y_min:y_max, x_min:x_max] = blended

                cv2.polylines(
                    frame, [poly_points], isClosed=True,
                    color=BORDER_COLOR, thickness=BORDER_THICKNESS,
                    lineType=cv2.LINE_AA
                )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            )

cap.release()
cv2.destroyAllWindows()
