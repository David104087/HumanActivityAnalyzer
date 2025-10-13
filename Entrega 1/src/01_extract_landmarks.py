# src/01_extract_landmarks.py
import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

mp_pose = mp.solutions.pose

def extract_landmarks_from_video(video_path, out_csv_path, fps_sample=1):
    """
    Extrae pose landmarks con MediaPipe y guarda en CSV.
    fps_sample: tomar 1 frame cada N frames (para acelerar); puede ser 1 para todos.
    """
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    columns = []
    # MediaPipe Pose landmarks: 33 puntos (0..32)
    for i in range(33):
        columns += [f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"]
    df_rows = []
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        frame_idx = 0
        pbar = tqdm(total=frames_total, desc=f"Procesando {video_name}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # sample control
            if frame_idx % fps_sample == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                row = {"video": video_name, "frame": frame_idx, "timestamp_s": frame_idx / fps}
                if results.pose_landmarks:
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        row[f"x_{i}"] = lm.x
                        row[f"y_{i}"] = lm.y
                        row[f"z_{i}"] = lm.z
                        row[f"vis_{i}"] = lm.visibility
                else:
                    # llenar NaN si no hay detección
                    for i in range(33):
                        row[f"x_{i}"] = np.nan
                        row[f"y_{i}"] = np.nan
                        row[f"z_{i}"] = np.nan
                        row[f"vis_{i}"] = np.nan
                df_rows.append(row)
            frame_idx += 1
            pbar.update(1)
        pbar.close()
    cap.release()
    df = pd.DataFrame(df_rows)
    df.to_csv(out_csv_path, index=False)
    print(f"Guardado: {out_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", default="../videos")
    parser.add_argument("--out_dir", default="../landmarks")
    parser.add_argument("--fps_sample", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for f in os.listdir(args.videos_dir):
        if f.lower().endswith((".mp4",".avi",".mov")):
            vpath = os.path.join(args.videos_dir, f)
            out_csv = os.path.join(args.out_dir, f"{os.path.splitext(f)[0]}_landmarks.csv")
            extract_landmarks_from_video(vpath, out_csv, fps_sample=args.fps_sample)
