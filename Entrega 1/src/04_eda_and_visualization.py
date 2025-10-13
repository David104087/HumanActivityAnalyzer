# src/04_eda_and_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ============================================
# Funciones auxiliares
# ============================================

def load_annotations(ann_path):
    """
    Espera CSV con: video, start_frame, end_frame, label
    (si LabelStudio/CVAT exportan otro formato, adaptar).
    """
    return pd.read_csv(ann_path)

def join_features_annotations(features_df, ann_df):
    """
    Para cada frame, asignar label (si cae dentro de algun segmento).
    """
    features_df["label"] = "unknown"
    for _, seg in ann_df.iterrows():
        mask = (
            (features_df["video"] == seg["video"]) &
            (features_df["frame"] >= seg["start_frame"]) &
            (features_df["frame"] <= seg["end_frame"])
        )
        features_df.loc[mask, "label"] = seg["label"]
    return features_df

def plot_feature_distribution(df_features, feature, out_dir, label_column=None):
    plt.figure(figsize=(8,5))
    if label_column and label_column in df_features.columns:
        sns.boxplot(x=label_column, y=feature, data=df_features)
        plt.title(f"Distribución de {feature} por {label_column}")
        plt.xticks(rotation=45)
    else:
        sns.histplot(df_features[feature].dropna(), kde=True)
        plt.title(f"Distribución de {feature}")
    fpath = os.path.join(out_dir, f"dist_{feature}.png")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    print("Guardado:", fpath)

def compute_quality_metrics(df_landmarks, df_features):
    total_frames = len(df_landmarks)
    missing_frames = df_landmarks["bad_frame"].sum() if "bad_frame" in df_landmarks.columns else 0
    pct_missing = missing_frames / total_frames if total_frames > 0 else 0
    print(f"Frames totales: {total_frames}, faltantes: {missing_frames} ({pct_missing:.2%})")
    scale_var = df_landmarks["scale"].std() if "scale" in df_landmarks.columns else np.nan
    print("Std scale:", scale_var)
    if "motion_energy" in df_features.columns:
        print("Motion energy stats:\n", df_features["motion_energy"].describe())

# ============================================
# Ejecución principal
# ============================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EDA y visualización de landmarks y features")
    parser.add_argument("--preprocessed_csv", required=True, help="Ruta del archivo preprocesado")
    parser.add_argument("--features_csv", required=True, help="Ruta del archivo de features")
    parser.add_argument("--annotations_csv", required=False, default=None, help="Ruta del archivo de anotaciones (opcional)")
    parser.add_argument("--out_dir", default="../reports", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Cargar datos
    dfl = pd.read_csv(args.preprocessed_csv)
    dff = pd.read_csv(args.features_csv)
    print(f"[INFO] Cargados {len(dfl)} frames preprocesados y {len(dff)} features")

    # Si existen anotaciones
    if args.annotations_csv and os.path.exists(args.annotations_csv):
        print(f"[INFO] Cargando anotaciones desde {args.annotations_csv}")
        ann = load_annotations(args.annotations_csv)
        dff = join_features_annotations(dff, ann)
        label_column = "label"
    else:
        print("[INFO] No se encontraron anotaciones — ejecutando EDA puro.")
        label_column = None

    # Graficar algunas variables clave
    for feature in ["trunk_angle", "motion_energy"]:
        if feature in dff.columns:
            plot_feature_distribution(dff, feature, args.out_dir, label_column)

    # Correlación general (sin importar etiquetas)
    valid_cols = [c for c in ["knee_left","knee_right","hip_left","hip_right","trunk_angle","motion_energy"] if c in dff.columns]
    if len(valid_cols) >= 2:
        plt.figure(figsize=(8,6))
        corr = dff[valid_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
        plt.title("Correlación entre features")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir,"corr_features.png"))
        plt.close()
        print("Guardado: corr_features.png")

    # Métricas de calidad
    compute_quality_metrics(dfl, dff)

    # Exportar resumen
    out_csv = os.path.join(args.out_dir, "features_with_labels.csv")
    dff.to_csv(out_csv, index=False)
    print(f"[OK] EDA finalizado. Guardado resumen en: {out_csv}")
