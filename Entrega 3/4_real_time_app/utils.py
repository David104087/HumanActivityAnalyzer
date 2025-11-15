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

def angle_between(a, b, c):
    """
    Angle at point b formed by points a-b-c. Inputs are (x,y).
    Returns angle in degrees.
    Copiado de src/03_compute_features.py
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def calculate_angle(landmark_a, landmark_b, landmark_c):
    """
    Calcula el ángulo en el punto b formado por a-b-c.
    Adaptación de angle_between para landmarks de MediaPipe.
    
    Args:
        landmark_a, landmark_b, landmark_c: Objetos landmark de MediaPipe con atributos x, y
    
    Returns:
        float: Ángulo en grados
    """
    a = (landmark_a.x, landmark_a.y)
    b = (landmark_b.x, landmark_b.y)
    c = (landmark_c.x, landmark_c.y)
    return angle_between(a, b, c)


def calculate_trunk_angle(landmarks):
    """
    Calcula el ángulo del tronco (inclinación del torso).
    Usa la aproximación vertical vs línea hombros-cadera.
    Adaptado de src/03_compute_features.py
    
    Args:
        landmarks: Lista de landmarks de MediaPipe
    
    Returns:
        float: Ángulo del tronco en grados
    """
    # Índices de landmarks (MediaPipe Pose)
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # Calcular punto medio de hombros y caderas
    shoulder_mid = (
        (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2,
        (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    )
    hip_mid = (
        (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2,
        (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    )
    
    # Aproximación vertical: punto arriba del hombro
    vertical_point = (shoulder_mid[0], shoulder_mid[1] - 1)
    
    return angle_between(vertical_point, shoulder_mid, hip_mid)


def calculate_motion_energy(current_landmarks, previous_landmarks):
    """
    Calcula la energía de movimiento entre dos frames consecutivos.
    Adaptado de src/03_compute_features.py para usar con MediaPipe landmarks.
    
    Args:
        current_landmarks: Landmarks del frame actual
        previous_landmarks: Landmarks del frame anterior (puede ser None)
    
    Returns:
        float: Energía de movimiento (promedio de diferencias absolutas)
    """
    if previous_landmarks is None:
        return 0.0
    
    # Calcular diferencias absolutas en x e y para todos los landmarks
    differences = []
    for i in range(len(current_landmarks)):
        curr = current_landmarks[i]
        prev = previous_landmarks[i]
        
        diff_x = abs(curr.x - prev.x)
        diff_y = abs(curr.y - prev.y)
        
        differences.extend([diff_x, diff_y])
    
    # Retornar el promedio de todas las diferencias
    return np.mean(differences)
