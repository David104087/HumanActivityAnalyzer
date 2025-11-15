import pandas as pd
import numpy as np
import os
import argparse

def angle_between(a, b, c):
    """
    Calcula el ángulo en el punto b formado por a-b-c.
    Entradas son (x,y). Retorna ángulo en grados.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def compute_frame_features(df_pre, video_name):
    """
    Input: Dataframe preprocesado con columnas nx_, ny_ (normalizadas).
    Output: Dataframe de features por frame (6 features).
    """
    N = len(df_pre)
    features = []
    
    # Columnas de landmarks normalizadas
    pos_cols = [c for c in df_pre.columns if c.startswith("nx_") or c.startswith("ny_")]
    if not pos_cols:
        print(f"ADVERTENCIA: No se encontraron columnas 'nx_' o 'ny_' en {video_name}. Omitiendo.")
        return pd.DataFrame()

    for i in range(N):
        row = df_pre.iloc[i]
        
        # helper para obtener puntos (x,y) normalizados
        def p(idx):
            return (row[f"nx_{idx}"], row[f"ny_{idx}"])

        # Índices de MediaPipe
        LEFT_HIP = 23; RIGHT_HIP = 24
        LEFT_KNEE = 25; RIGHT_KNEE = 26
        LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
        LEFT_ANKLE = 27; RIGHT_ANKLE = 28
        
        try:
            # 1. Ángulo de rodilla izquierda
            knee_left_angle = angle_between(p(LEFT_HIP), p(LEFT_KNEE), p(LEFT_ANKLE))
            # 2. Ángulo de rodilla derecha
            knee_right_angle = angle_between(p(RIGHT_HIP), p(RIGHT_KNEE), p(RIGHT_ANKLE))
            
            # 3. Ángulo de cadera izquierda
            hip_left_angle = angle_between(p(LEFT_SHOULDER), p(LEFT_HIP), p(LEFT_KNEE))
            # 4. Ángulo de cadera derecha
            hip_right_angle = angle_between(p(RIGHT_SHOULDER), p(RIGHT_HIP), p(RIGHT_KNEE))

            # 5. Ángulo de inclinación del tronco
            shoulder_mid = ((row["nx_11"] + row["nx_12"]) / 2, (row["ny_11"] + row["ny_12"]) / 2)
            hip_mid = ((row["nx_23"] + row["nx_24"]) / 2, (row["ny_23"] + row["ny_24"]) / 2)
            vertical_point = (shoulder_mid[0], shoulder_mid[1] - 1) # Punto vertical "arriba"
            trunk_angle = angle_between(vertical_point, shoulder_mid, hip_mid)
            
        except Exception as e:
            # Si faltan landmarks (NaN) en este frame, ponemos NaN en los features
            knee_left_angle = np.nan
            knee_right_angle = np.nan
            hip_left_angle = np.nan
            hip_right_angle = np.nan
            trunk_angle = np.nan

        features.append({
            "video_name": video_name, # Usamos el nombre de video simple (ej. "Video1")
            "frame": row["frame"],      # Pasamos el número de frame
            "knee_left": knee_left_angle,
            "knee_right": knee_right_angle,
            "hip_left": hip_left_angle,
            "hip_right": hip_right_angle,
            "trunk_angle": trunk_angle
        })
        
    fdf = pd.DataFrame(features)
    
    if fdf.empty:
        return fdf
    
    # 6. 'motion_energy' (Energía de movimiento)
    #    Calcula la diferencia de posición de *todos* los landmarks entre frames
    pos = df_pre[pos_cols].ffill().fillna(0).values
    pos = pos.reshape(len(df_pre), -1)
    
    # Calculamos la diferencia absoluta entre frames
    vel = np.vstack([np.zeros(pos.shape[1]), np.abs(np.diff(pos, axis=0))])
    
    # La energía de movimiento es el promedio de estas diferencias
    fdf["motion_energy"] = vel.mean(axis=1)
    
    return fdf

# --- Ejecución del Script ---

def main():
    parser = argparse.ArgumentParser(description="Calcula features (ángulos, movimiento) desde landmarks preprocesados.")
    parser.add_argument("--preprocessed_csv", required=True, help="Ruta al archivo CSV preprocesado (ej. Video1_preprocessed.csv)")
    parser.add_argument("--out_dir", required=True, help="Directorio donde se guardarán los features (ej. data/features_per_frame)")
    args = parser.parse_args()

    # Extraer el nombre simple del video (ej. "Video1")
    # de la ruta "data/preprocessed/Video1_preprocessed.csv"
    base_name = os.path.basename(args.preprocessed_csv)
    video_name = base_name.replace("_preprocessed.csv", "")
    
    print(f"Procesando features para: {video_name}...")
    
    dfp = pd.read_csv(args.preprocessed_csv)
    
    # Verificación de sanidad: la columna 'frame' es VITAL
    if "frame" not in dfp.columns:
        print(f"ERROR: El archivo {args.preprocessed_csv} no tiene columna 'frame'. Omitiendo.")
        return

    fdf = compute_frame_features(dfp, video_name)
    
    if fdf.empty:
        print(f"No se generaron features para {video_name}.")
        return

    # Asegurarse que el directorio de salida exista
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Guardar el archivo de features
    # ej. "data/features_per_frame/Video1_features.csv"
    out_name = f"{video_name}_features.csv"
    out_path = os.path.join(args.out_dir, out_name)
    fdf.to_csv(out_path, index=False)
    print(f"✓ Guardado features en: {out_path}")

if __name__ == "__main__":
    main()