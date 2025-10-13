# src/utils.py
import cv2
import pandas as pd
import numpy as np
import os

def overlay_keypoints_on_frame(frame, landmarks_row, color=(0,255,0)):
    """
    landmarks_row: fila del CSV preprocesado con columnas x_i,y_i (normalizadas)
    Convierte coords normalizadas a pixeles y dibuja círculos/lineas.
    """
    h, w = frame.shape[:2]
    for i in range(33):
        nx = landmarks_row.get(f"x_{i}", None)
        ny = landmarks_row.get(f"y_{i}", None)
        if pd.isna(nx) or pd.isna(ny):
            continue
        px = int(nx * w)
        py = int(ny * h)
        cv2.circle(frame, (px,py), 3, color, -1)
    return frame

def create_skeleton_overlay_video(video_path, landmarks_csv, out_video):
    import pandas as pd
    df = pd.read_csv(landmarks_csv).set_index("frame")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, (w,h))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        row = df.loc[frame_idx] if frame_idx in df.index else None
        if row is not None:
            frame = overlay_keypoints_on_frame(frame, row)
        out.write(frame)
        frame_idx += 1
    cap.release(); out.release()
