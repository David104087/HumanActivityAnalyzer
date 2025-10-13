"""
00_basic_video_eda.py
--------------------------------------
Módulo inicial del pipeline de análisis de video.

Objetivo:
    - Extraer información básica de los videos (metadatos técnicos).
    - Analizar color promedio, brillo promedio y nivel de movimiento.
    - Generar estadísticas descriptivas y gráficas exploratorias.

Entradas:
    - Carpeta con archivos de video (.mp4, .avi, .mov).

Salidas:
    - CSV: basic_video_stats.csv (información técnica y visual)
    - Gráficos: histogramas y comparaciones guardadas en ./reports/

Uso:
    python src/00_basic_video_eda.py --videos_dir ./videos --out_dir ./reports
"""

import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Funciones auxiliares
# ================================

def extraer_metadatos(video_path):
    """
    Extrae información técnica básica del video.
    Devuelve: dict con nombre, fps, frames, duración, resolución, tamaño_MB
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ADVERTENCIA] No se pudo abrir: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duracion = frames / fps if fps else 0
    ancho = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    alto = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    tamaño_MB = os.path.getsize(video_path) / (1024 * 1024)
    cap.release()

    return {
        "nombre": os.path.basename(video_path),
        "fps": round(fps, 2),
        "frames": int(frames),
        "duracion_seg": round(duracion, 2),
        "resolucion": f"{int(ancho)}x{int(alto)}",
        "tamaño_MB": round(tamaño_MB, 2)
    }


def color_y_brillo(video_path, sample_rate=30):
    """
    Calcula color y brillo promedio del video tomando una muestra de frames.
    sample_rate: tomar 1 frame cada N frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    colores, brillos = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            color_prom = frame.mean(axis=(0, 1))   # Promedio RGB
            brillo = frame.mean()                  # Intensidad promedio
            colores.append(color_prom)
            brillos.append(brillo)
        frame_idx += 1

    cap.release()
    if len(colores) == 0:
        return (0, 0, 0), 0
    return np.mean(colores, axis=0), np.mean(brillos)


def nivel_movimiento(video_path, step=10):
    """
    Estima el nivel de movimiento del video comparando frames consecutivos.
    step: cuántos frames saltar entre comparaciones.
    """
    cap = cv2.VideoCapture(video_path)
    prev = None
    difs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            dif = np.mean(cv2.absdiff(prev, gray))
            difs.append(dif)
        prev = gray
        for _ in range(step - 1):  # saltar frames
            cap.grab()

    cap.release()
    return np.mean(difs) if len(difs) > 0 else 0


# ================================
# Proceso principal del EDA
# ================================

def basic_video_eda(videos_dir, out_dir, sample_rate=30, step=10):
    """
    Ejecuta el EDA básico sobre todos los videos de una carpeta.
    """
    os.makedirs(out_dir, exist_ok=True)
    datos = []

    print(f"\n[INFO] Iniciando EDA en carpeta: {videos_dir}\n")

    for archivo in os.listdir(videos_dir):
        if archivo.endswith((".mp4", ".avi", ".mov")):
            path = os.path.join(videos_dir, archivo)
            print(f"Procesando: {archivo} ...")

            meta = extraer_metadatos(path)
            if not meta:
                continue

            # Añadir color y brillo
            color, brillo = color_y_brillo(path, sample_rate)
            meta["R"], meta["G"], meta["B"] = color
            meta["Brillo"] = brillo

            # Añadir movimiento
            meta["Movimiento"] = nivel_movimiento(path, step)

            datos.append(meta)

    df = pd.DataFrame(datos)
    csv_path = os.path.join(out_dir, "basic_video_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] CSV generado: {csv_path}\n")

    # ================================
    # Análisis estadístico básico
    # ================================
    print("\n=== Estadísticas básicas ===")
    print(df.describe(include="all"))

    # ================================
    # Visualizaciones
    # ================================
    print("\n[INFO] Generando gráficos...")

    # Distribución de duración
    plt.figure(figsize=(8, 5))
    plt.hist(df["duracion_seg"], bins=10, color="skyblue", edgecolor="black")
    plt.title("Distribución de Duraciones de Videos")
    plt.xlabel("Duración (segundos)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "duraciones.png"))
    plt.close()

    # Relación entre duración y brillo
    plt.figure(figsize=(8, 5))
    plt.scatter(df["duracion_seg"], df["Brillo"], color="orange")
    plt.title("Relación entre duración y brillo promedio")
    plt.xlabel("Duración (s)")
    plt.ylabel("Brillo promedio")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "color_vs_duracion.png"))
    plt.close()

    # Color promedio por video (RGB apilado)
    plt.figure(figsize=(10, 6))
    plt.bar(df["nombre"], df["R"], label="Rojo")
    plt.bar(df["nombre"], df["G"], label="Verde", bottom=df["R"])
    plt.bar(df["nombre"], df["B"], label="Azul", bottom=df["R"] + df["G"])
    plt.legend()
    plt.title("Color promedio por video")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "color_promedio.png"))
    plt.close()

    # Nivel de movimiento
    plt.figure(figsize=(10, 5))
    plt.bar(df["nombre"], df["Movimiento"], color="teal")
    plt.title("Nivel promedio de movimiento por video")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "movimiento.png"))
    plt.close()

    print(f"[INFO] Gráficos guardados en: {out_dir}\n")
    print(df[["nombre", "duracion_seg", "Brillo", "Movimiento", "tamaño_MB"]])
    print("\nEDA básico finalizado.\n")


# ================================
# Ejecución directa desde terminal
# ================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", default="../videos", help="Carpeta con los videos")
    parser.add_argument("--out_dir", default="../reports", help="Carpeta de salida para reportes")
    parser.add_argument("--sample_rate", type=int, default=30, help="Frames a saltar para análisis de color/brillo")
    parser.add_argument("--step", type=int, default=10, help="Frames a saltar para cálculo de movimiento")
    args = parser.parse_args()

    basic_video_eda(args.videos_dir, args.out_dir, args.sample_rate, args.step)
