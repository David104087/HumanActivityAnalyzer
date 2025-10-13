# src/02_preprocess_landmarks.py
import pandas as pd
import numpy as np
import os

def load_landmarks(csv_path):
    return pd.read_csv(csv_path)

def normalize_landmarks(df):
    """
    Normaliza las coordenadas usando distancia entre los hombros o altura del torso.
    Strategy: calcular escala = dist(LEFT_SHOULDER, RIGHT_HOULDER) o
    dist(hip_top, shoulder_top) (fallback).
    Luego, centrar usando punto medio de caderas o hombros.
    """
    # índices de MediaPipe
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    df_norm = df.copy()
    xs = df[[f"x_{i}" for i in range(33)]].values
    ys = df[[f"y_{i}" for i in range(33)]].values

    centers_x = []
    centers_y = []
    scales = []

    for idx, row in df.iterrows():
        try:
            lx = row[f"x_{LEFT_SHOULDER}"]; ly = row[f"y_{LEFT_SHOULDER}"]
            rx = row[f"x_{RIGHT_SHOULDER}"]; ry = row[f"y_{RIGHT_SHOULDER}"]
            hx = row[f"x_{LEFT_HIP}"]; hy = row[f"y_{LEFT_HIP}"]
            hrx = row[f"x_{RIGHT_HIP}"]; hry = row[f"y_{RIGHT_HIP}"]
        except KeyError:
            raise

        if np.isnan([lx,ly,rx,ry,hx,hy,hrx,hry]).any():
            centers_x.append(np.nan); centers_y.append(np.nan); scales.append(np.nan)
            continue

        center_x = (lx + rx + hx + hrx)/4.0
        center_y = (ly + ry + hy + hry)/4.0
        # escala por ancho de hombros
        scale = np.sqrt((lx-rx)**2 + (ly-ry)**2)
        if scale == 0: scale = 1e-6
        centers_x.append(center_x); centers_y.append(center_y); scales.append(scale)

    df_norm["center_x"] = centers_x
    df_norm["center_y"] = centers_y
    df_norm["scale"] = scales

    # aplicar normalización: (coord - center) / scale
    for i in range(33):
        df_norm[f"nx_{i}"] = (df_norm[f"x_{i}"] - df_norm["center_x"]) / df_norm["scale"]
        df_norm[f"ny_{i}"] = (df_norm[f"y_{i}"] - df_norm["center_y"]) / df_norm["scale"]
        df_norm[f"nz_{i}"] = df_norm[f"z_{i}"] / df_norm["scale"]
    return df_norm

def moving_average_smoothing(df, window=3):
    """
    Suavizado simple para cada columna nx_i, ny_i
    """
    smooth_df = df.copy()
    cols = [c for c in df.columns if c.startswith(("nx_","ny_","nz_"))]
    for c in cols:
        smooth_df[c] = df[c].rolling(window, min_periods=1, center=True).mean()
    return smooth_df

def detect_bad_frames(df, vis_threshold=0.3, missing_pct_threshold=0.5):
    """
    Marca frames con demasiados landmarks faltantes o baja visibilidad.
    """
    bad_mask = []
    for idx, row in df.iterrows():
        vis = [row[f"vis_{i}"] for i in range(33)]
        n_missing = sum(pd.isna(vis))
        n_lowvis = sum([1 for v in vis if (not pd.isna(v) and v < vis_threshold)])
        total = 33
        if (n_missing/total) > missing_pct_threshold or (n_lowvis/total) > 0.6:
            bad_mask.append(True)
        else:
            bad_mask.append(False)
    df["bad_frame"] = bad_mask
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark_csv", required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    df = load_landmarks(args.landmark_csv)
    dfn = normalize_landmarks(df)
    dfs = moving_average_smoothing(dfn, window=5)
    df_final = detect_bad_frames(dfs)
    out = args.out_csv or args.landmark_csv.replace(".csv","_preprocessed.csv")
    df_final.to_csv(out, index=False)
    print("Guardado:", out)
