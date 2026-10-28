from pathlib import Path
import cv2
import subprocess

def get_video_info(video_path):
    """讀取影片的基本資訊。"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("無法開啟影片，請確認檔案路徑與影片格式。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError("無法取得有效的影片 FPS 或影格數。")

    duration = frame_count / fps

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }

import pandas as pd

from pose_analyzer import (
    create_pose_detector,
    detect_pose,
    get_landmark_points,
    calculate_joint_angles,
)


def analyze_video(video_path):
    """逐格分析影片，記錄人體關鍵點與關節角度。"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("無法開啟影片，請確認檔案路徑與格式。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("影片的 FPS 或影格數無效。")

    records = []
    pose = create_pose_detector()

    try:
        frame_index = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_index / fps
            timestamp_ms = int(timestamp * 1000)
            landmarks = detect_pose(frame, pose, timestamp_ms)

            record = {
                "frame": frame_index, 
                "time_seconds": timestamp,
            }

            if landmarks is not None:
                points = get_landmark_points(
                    landmarks, width, height
                )
                angles = calculate_joint_angles(points)

                for name, point in points.items():
                    record[f"{name}_x"] = point["x"]
                    record[f"{name}_y"] = point["y"]
                    record[f"{name}_visibility"] = point["visibility"]

                for name, angle in angles.items():
                    record[f"{name}_angle_deg"] = angle
            else:
                record["pose_detected"] = False

            record.setdefault(
                "pose_detected", landmarks is not None
            )
            records.append(record)
            frame_index += 1

    finally:
        cap.release()
        pose.close()

    if not records:
        raise ValueError("影片中沒有可讀取的影格。")

    results = pd.DataFrame(records)

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps,
        "results": results,
    }

def export_results_csv(analysis, output_path):
    """將影片分析結果匯出為 CSV。"""
    results = analysis["results"]

    if results.empty:
        raise ValueError("沒有分析資料可以匯出。")

    results.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    return output_path

def export_annotated_video(video_path, output_path):
    """輸出疊加人體骨架與關節角度的分析影片。"""
    from pose_analyzer import draw_pose_analysis

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("無法開啟原始影片。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("影片資訊無效，無法輸出分析影片。")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    
    import imageio_ffmpeg

    temp_output_path = output_path.with_name(
        output_path.stem + "_temp.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(temp_output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise ValueError("無法建立暫存影片，請確認輸出路徑。")


    pose = create_pose_detector()
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int(frame_index * 1000 / fps)
            landmarks = detect_pose(frame, pose, timestamp_ms)

            if landmarks is not None:
                points = get_landmark_points(
                    landmarks, width, height
                )
                angles = calculate_joint_angles(points)
                frame = draw_pose_analysis(
                    frame, landmarks, points, angles
                )
            else:
                cv2.putText(
                    frame,
                    "Pose not detected",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            # 顯示影片時間
            elapsed = frame_index / fps
            cv2.putText(
                frame,
                f"Time: {elapsed:.2f} s",
                (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            writer.write(frame)
            frame_index += 1

    finally:
        cap.release()
        writer.release()
        pose.close()

    
    if frame_index == 0:
        raise ValueError("影片沒有可輸出的影格。")

    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i", str(temp_output_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"H.264 影片轉碼失敗：{exc.stderr[-1500:]}"
        ) from exc
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()

    return output_path

