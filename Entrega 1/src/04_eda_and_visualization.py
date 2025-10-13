# src/04_eda_and_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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
        mask = (features_df["video"]==seg["video"]) & (features_df["frame"]>=seg["start_frame"]) & (features_df["frame"]<=seg["end_frame"])
        features_df.loc[mask, "label"] = seg["label"]
    return features_df

def plot_feature_distribution_by_label(df_features, feature, out_dir):
    plt.figure(figsize=(8,5))
    sns.boxplot(x="label", y=feature, data=df_features)
    plt.title(f"Distribución de {feature} por label")
    plt.xticks(rotation=45)
    fpath = os.path.join(out_dir, f"box_{feature}.png")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    print("Guardado:", fpath)

def compute_quality_metrics(df_landmarks, df_features):
    total_frames = len(df_landmarks)
    missing_frames = df_landmarks["bad_frame"].sum()
    pct_missing = missing_frames / total_frames
    print(f"Frames totales: {total_frames}, faltantes: {missing_frames} ({pct_missing:.2%})")
    # variabilidad de scale (posible cambio de distancia a cámara)
    scale_var = df_landmarks["scale"].std()
    print("Std scale:", scale_var)
    # motion energy distribution
    print("Motion energy stats:", df_features["motion_energy"].describe())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_csv", required=True)
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--annotations_csv", required=True)
    parser.add_argument("--out_dir", default="../reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dfl = pd.read_csv(args.preprocessed_csv)
    dff = pd.read_csv(args.features_csv)
    ann = load_annotations(args.annotations_csv)
    dff_labeled = join_features_annotations(dff, ann)

    # ejemplos de gráficas
    plot_feature_distribution_by_label(dff_labeled, "trunk_angle", args.out_dir)
    plot_feature_distribution_by_label(dff_labeled, "motion_energy", args.out_dir)

    # correlación
    plt.figure(figsize=(8,6))
    corr = dff_labeled[["knee_left","knee_right","hip_left","hip_right","trunk_angle","motion_energy"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
    plt.title("Correlación entre features")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir,"corr_features.png"))
    plt.close()

    compute_quality_metrics(dfl, dff)
    dff_labeled.to_csv(os.path.join(args.out_dir,"features_with_labels.csv"), index=False)
    print("Guardado features con labels.")
