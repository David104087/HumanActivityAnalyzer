# src/03_compute_features.py
import pandas as pd
import numpy as np
import os

# helpers para ángulos
def angle_between(a, b, c):
    """
    Angle at point b formed by points a-b-c. Inputs are (x,y).
    Returns angle in degrees.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def compute_frame_features(df_pre):
    """
    Input: preprocessed df con nx_, ny_ columns.
    Output: df_features por frame (velocidad media, aceleración media, ángulos clave, inclinación).
    """
    N = len(df_pre)
    features = []
    for i in range(N):
        row = df_pre.iloc[i]
        # recoger puntos (x,y) de interés
        def p(idx):
            return (row[f"nx_{idx}"], row[f"ny_{idx}"])
        # landmarks útiles:
        LEFT_HIP = 23; RIGHT_HIP = 24
        LEFT_KNEE = 25; RIGHT_KNEE = 26
        LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
        LEFT_ANKLE = 27; RIGHT_ANKLE = 28
        # ángulos de rodilla (izq y der)
        try:
            knee_left_angle = angle_between(p(23), p(25), p(27))
            knee_right_angle = angle_between(p(24), p(26), p(28))
        except:
            knee_left_angle = np.nan; knee_right_angle = np.nan
        # ángulo cadera (hip-shoulder-knee aproximado)
        try:
            hip_left_angle = angle_between(p(11), p(23), p(25))
            hip_right_angle = angle_between(p(12), p(24), p(26))
        except:
            hip_left_angle = np.nan; hip_right_angle = np.nan

        # inclinación del tronco: línea hombros vs vertical -> usar y difference
        try:
            shoulder_mid = ((row["nx_11"]+row["nx_12"])/2, (row["ny_11"]+row["ny_12"])/2)
            hip_mid = ((row["nx_23"]+row["nx_24"])/2, (row["ny_23"]+row["ny_24"])/2)
            trunk_angle = angle_between((shoulder_mid[0], shoulder_mid[1]-1), shoulder_mid, hip_mid)  # approx vertical
        except:
            trunk_angle = np.nan

        features.append({
            "video": row["video"],
            "frame": row["frame"],
            "knee_left": knee_left_angle,
            "knee_right": knee_right_angle,
            "hip_left": hip_left_angle,
            "hip_right": hip_right_angle,
            "trunk_angle": trunk_angle
        })
    fdf = pd.DataFrame(features)
    # calcular velocidades como diferencia de posición entre frames (media de todas las articulaciones)
    # velocidad media por frame:
    pos_cols = [c for c in df_pre.columns if c.startswith("nx_") or c.startswith("ny_")]
    pos = df_pre[pos_cols].fillna(method="ffill").fillna(0).values
    pos = pos.reshape(len(df_pre), -1)
    vel = np.vstack([np.zeros(pos.shape[1]), np.abs(np.diff(pos, axis=0))])
    fdf["motion_energy"] = vel.mean(axis=1)
    return fdf

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_csv", required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    dfp = pd.read_csv(args.preprocessed_csv)
    fdf = compute_frame_features(dfp)
    out = args.out_csv or args.preprocessed_csv.replace("_preprocessed.csv","_features.csv")
    fdf.to_csv(out, index=False)
    print("Guardado features:", out)
