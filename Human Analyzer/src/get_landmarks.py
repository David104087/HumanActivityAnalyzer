import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Intento de cargar mediapipe
try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe no está instalado. Ejecuta: pip install mediapipe", file=sys.stderr)
    raise

POSE = mp.solutions.pose

# Extensiones válidas de video
VALID_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# ------------------------------------------------------
# --- Utilidades ---------------------------------------
# ------------------------------------------------------

def es_video(path: Path) -> bool:
    """Verifica si el archivo corresponde a un formato de video."""
    return path.suffix.lower() in VALID_VIDEO_EXT


def stats_luminancia(img_gray: np.ndarray) -> Tuple[float, float]:
    """Promedio y desviación estándar de luminancia."""
    return float(img_gray.mean()), float(img_gray.std())


def diferencia_abs_media(prev: Optional[np.ndarray], curr: np.ndarray) -> Optional[float]:
    """Calcula la Mean Absolute Difference entre frames consecutivos."""
    if prev is None:
        return None

    if prev.shape != curr.shape:
        try:
            prev = cv2.resize(prev, (curr.shape[1], curr.shape[0]))
        except Exception:
            return None

    diff = np.abs(curr.astype(np.int16) - prev.astype(np.int16))
    return float(diff.mean())


def obtener_landmarks_pose(result):
    """Extrae landmarks 2D y 3D desde la salida de MediaPipe Pose."""
    lm_2d, lm_3d = None, None

    if result.pose_landmarks:
        lm_2d = [
            (i, lm.x, lm.y, lm.z, getattr(lm, "visibility", np.nan))
            for i, lm in enumerate(result.pose_landmarks.landmark)
        ]

    if hasattr(result, "pose_world_landmarks") and result.pose_world_landmarks:
        lm_3d = [
            (i, lm.x, lm.y, lm.z)
            for i, lm in enumerate(result.pose_world_landmarks.landmark)
        ]

    return lm_2d, lm_3d


# ------------------------------------------------------
# --- Procesamiento de video ---------------------------
# ------------------------------------------------------

def analizar_video(
    video: Path,
    etiqueta: str,
    pose_obj: "POSE.Pose",
    salto: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir el archivo: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duracion = total_frames / fps if fps > 0 else np.nan

    datos_pose = []
    datos_frames = []

    previo_gray = None
    idx = 0

    barra = tqdm(total=total_frames, desc=f"Procesando {video.name}", unit="frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % salto != 0:
            idx += 1
            barra.update(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = pose_obj.process(rgb)
        lm2d, lm3d = obtener_landmarks_pose(res)

        tiempo = idx / fps if fps > 0 else np.nan
        mean_l, std_l = stats_luminancia(gray)
        mad_mov = diferencia_abs_media(previo_gray, gray)

        datos_frames.append({
            "action": etiqueta,
            "video_filename": video.name,
            "frame_idx": idx,
            "timestamp_sec": tiempo,
            "width": ancho,
            "height": alto,
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duracion,
            "luminance_mean": mean_l,
            "luminance_std": std_l,
            "motion_mad": mad_mov
        })

        if lm2d is not None:
            registro = {
                "action": etiqueta,
                "video_filename": video.name,
                "frame_idx": idx,
                "timestamp_sec": tiempo
            }

            for lid, x, y, z, vis in lm2d:
                registro[f"lm{lid}_x"] = x
                registro[f"lm{lid}_y"] = y
                registro[f"lm{lid}_z"] = z
                registro[f"lm{lid}_vis"] = vis

            datos_pose.append(registro)

        previo_gray = gray
        idx += 1
        barra.update(1)

    barra.close()
    cap.release()

    return pd.DataFrame(datos_pose), pd.DataFrame(datos_frames)


# ------------------------------------------------------
# --- Programa principal -------------------------------
# ------------------------------------------------------

def ejecutar():
    parser = argparse.ArgumentParser(description="Extractor de características y landmarks desde videos usando MediaPipe Pose.")
    parser.add_argument("--stride", type=int, default=1, help="Analizar 1 de cada N frames.")
    parser.add_argument("--static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    args = parser.parse_args()

    ruta_videos = Path("./data/raw/videos")
    ruta_salida = Path("./data/processed")
    ruta_salida.mkdir(parents=True, exist_ok=True)

    carpetas = [d for d in ruta_videos.iterdir() if d.is_dir()]
    if not carpetas:
        print("ERROR: No hay carpetas de acciones dentro de data/raw/videos.")
        sys.exit(1)

    print(f"Acciones detectadas: {[c.name for c in carpetas]}")

    acumulado_pose = []
    acumulado_frames = []

    with POSE.Pose(
        static_image_mode=args.static_image_mode,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        smooth_landmarks=True
    ) as pose_detector:

        for carpeta in carpetas:
            accion = carpeta.name
            lista_videos = [p for p in carpeta.glob("**/*") if es_video(p)]

            for v in lista_videos:
                try:
                    df_p, df_f = analizar_video(v, accion, pose_detector, args.stride)
                    acumulado_pose.append(df_p)
                    acumulado_frames.append(df_f)
                except Exception as err:
                    print(f"Advertencia: error procesando {v.name}: {err}", file=sys.stderr)

    archivo_pose = ruta_salida / "datosmediapipe.csv"
    archivo_frames = ruta_salida / "datos_analisis.csv"

    if acumulado_pose:
        pd.concat(acumulado_pose, ignore_index=True).to_csv(archivo_pose, index=False)
    else:
        columnas = ["action", "video_filename", "frame_idx", "timestamp_sec"]
        for i in range(33):
            columnas += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z", f"lm{i}_vis"]
        pd.DataFrame(columns=columnas).to_csv(archivo_pose, index=False)

    if acumulado_frames:
        pd.concat(acumulado_frames, ignore_index=True).to_csv(archivo_frames, index=False)
    else:
        cols = ["action","video_filename","frame_idx","timestamp_sec","width","height",
                "fps","total_frames","duration_sec","luminance_mean","luminance_std","motion_mad"]
        pd.DataFrame(columns=cols).to_csv(archivo_frames, index=False)

    print("\nProceso finalizado.")
    print(f" -> Guardado: {archivo_pose}")
    print(f" -> Guardado: {archivo_frames}")


if __name__ == "__main__":
    ejecutar()
