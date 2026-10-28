
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "pose_landmarker_lite.task"
)


def create_pose_detector():
    """建立 MediaPipe Tasks 人體姿態偵測器。"""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到人體姿態模型：{MODEL_PATH}"
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.PoseLandmarker.create_from_options(options)


def detect_pose(image, pose, timestamp_ms=0):
    """偵測單張影像中的人體關鍵點。"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image,
    )

    results = pose.detect_for_video(mp_image, timestamp_ms)

    if not results.pose_landmarks:
        return None

    return results.pose_landmarks[0]


def get_landmark_points(landmarks, image_width, image_height):
    """取得人體關鍵點的像素座標。"""
    points = {}

    landmark_names = {
        "right_shoulder": 12,
        "right_hip": 24,
        "right_knee": 26,
        "right_ankle": 28,
        "right_heel": 30,
        "right_foot_index": 32,
        "left_shoulder": 11,
        "left_hip": 23,
        "left_knee": 25,
        "left_ankle": 27,
        "left_heel": 29,
        "left_foot_index": 31,
    }

    for name, index in landmark_names.items():
        landmark = landmarks[index]

        points[name] = {
            "x": landmark.x * image_width,
            "y": landmark.y * image_height,
            "visibility": landmark.visibility,
        }

    return points


import math


def calculate_angle(point_a, point_b, point_c):
    """計算 A-B-C 的幾何夾角，範圍 0～180 度。"""
    vector_ba = (
        point_a["x"] - point_b["x"],
        point_a["y"] - point_b["y"],
    )
    vector_bc = (
        point_c["x"] - point_b["x"],
        point_c["y"] - point_b["y"],
    )

    dot_product = (
        vector_ba[0] * vector_bc[0]
        + vector_ba[1] * vector_bc[1]
    )

    length_ba = math.hypot(*vector_ba)
    length_bc = math.hypot(*vector_bc)

    if length_ba == 0 or length_bc == 0:
        return None

    cosine = dot_product / (length_ba * length_bc)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def calculate_joint_angles(points):
    """計算左右膝關節與髖關節的幾何角度。"""
    angles = {}

    for side in ("right", "left"):
        shoulder = f"{side}_shoulder"
        hip = f"{side}_hip"
        knee = f"{side}_knee"
        ankle = f"{side}_ankle"

        if not all(
            name in points
            for name in (shoulder, hip, knee, ankle)
        ):
            continue

        angles[f"{side}_knee"] = calculate_angle(
            points[hip],
            points[knee],
            points[ankle],
        )

        angles[f"{side}_hip"] = calculate_angle(
            points[shoulder],
            points[hip],
            points[knee],
        )

    return angles

def draw_pose_analysis(frame, landmarks, points, angles):
    """在影片影格上繪製人體骨架、關節點與角度。"""
    height, width = frame.shape[:2]

    # MediaPipe 人體骨架連線：肩、軀幹、腿部與手臂
    connections = [
        (11, 12),  # 左右肩
        (11, 23), (12, 24),  # 肩到髖
        (23, 24),  # 左右髖
        (11, 13), (13, 15),  # 左手臂
        (12, 14), (14, 16),  # 右手臂
        (23, 25), (25, 27),  # 左腿
        (24, 26), (26, 28),  # 右腿
        (27, 29), (29, 31),  # 左腳
        (28, 30), (30, 32),  # 右腳
    ]

    # 繪製骨架連線
    for start_idx, end_idx in connections:
        start = landmarks[start_idx]
        end = landmarks[end_idx]

        if (
            start.visibility < 0.5
            or end.visibility < 0.5
        ):
            continue

        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))

        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    # 繪製已取得的關鍵點
    for point in points.values():
        if point["visibility"] < 0.5:
            continue

        x = int(point["x"])
        y = int(point["y"])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # 在畫面左上角顯示關節角度
    y_text = 30

    for name, angle in angles.items():
        if angle is None:
            continue

        text = f"{name}: {angle:.1f} deg"
        cv2.putText(
            frame,
            text,
            (15, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_text += 28

    return frame
