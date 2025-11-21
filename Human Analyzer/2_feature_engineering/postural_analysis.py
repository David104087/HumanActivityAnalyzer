"""
Script 2B: Análisis Postural
Calcula inclinaciones laterales y métricas posturales adicionales.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def calcular_inclinacion_lateral(fila):
    """
    Calcula la inclinación lateral del tronco comparando hombros y caderas.
    Retorna el ángulo de inclinación en grados.
    """
    try:
        # Hombros
        hombro_izq = np.array([fila["lm11_x"], fila["lm11_y"], fila["lm11_z"]])
        hombro_der = np.array([fila["lm12_x"], fila["lm12_y"], fila["lm12_z"]])
        
        # Caderas
        cadera_izq = np.array([fila["lm23_x"], fila["lm23_y"], fila["lm23_z"]])
        cadera_der = np.array([fila["lm24_x"], fila["lm24_y"], fila["lm24_z"]])
        
        # Vector entre hombros (horizontal ideal)
        vec_hombros = hombro_der - hombro_izq
        
        # Vector entre caderas
        vec_caderas = cadera_der - cadera_izq
        
        # Calcular ángulo de inclinación en el plano frontal (eje Y)
        # Usamos solo las coordenadas X y Y (ignoramos Z)
        angulo_hombros = np.degrees(np.arctan2(vec_hombros[1], vec_hombros[0]))
        angulo_caderas = np.degrees(np.arctan2(vec_caderas[1], vec_caderas[0]))
        
        # Inclinación lateral (diferencia entre hombros y caderas)
        inclinacion = abs(angulo_hombros - angulo_caderas)
        
        return inclinacion
        
    except Exception:
        return np.nan


def calcular_alineacion_vertical(fila):
    """
    Calcula qué tan alineado está el cuerpo verticalmente.
    Retorna el ángulo de desviación respecto a la vertical.
    """
    try:
        # Centro de hombros
        hombro_izq = np.array([fila["lm11_x"], fila["lm11_y"]])
        hombro_der = np.array([fila["lm12_x"], fila["lm12_y"]])
        centro_hombros = (hombro_izq + hombro_der) / 2
        
        # Centro de caderas
        cadera_izq = np.array([fila["lm23_x"], fila["lm23_y"]])
        cadera_der = np.array([fila["lm24_x"], fila["lm24_y"]])
        centro_caderas = (cadera_izq + cadera_der) / 2
        
        # Vector de cadera a hombros
        vec_tronco = centro_hombros - centro_caderas
        
        # Ángulo respecto a la vertical (eje Y negativo)
        angulo = np.degrees(np.arctan2(abs(vec_tronco[0]), abs(vec_tronco[1])))
        
        return angulo
        
    except Exception:
        return np.nan


def analizar_simetria_brazos(fila):
    """
    Calcula la simetría entre los brazos izquierdo y derecho.
    """
    try:
        # Ángulos de codos
        ang_codo_izq = fila.get("ang_elb_izq", np.nan)
        ang_codo_der = fila.get("ang_elb_der", np.nan)
        
        # Diferencia entre ambos codos
        if not np.isnan(ang_codo_izq) and not np.isnan(ang_codo_der):
            return abs(ang_codo_izq - ang_codo_der)
        
        return np.nan
        
    except Exception:
        return np.nan


def analizar_simetria_piernas(fila):
    """
    Calcula la simetría entre las piernas izquierda y derecha.
    """
    try:
        # Ángulos de rodillas
        ang_rod_izq = fila.get("ang_rod_izq", np.nan)
        ang_rod_der = fila.get("ang_rod_der", np.nan)
        
        # Diferencia entre ambas rodillas
        if not np.isnan(ang_rod_izq) and not np.isnan(ang_rod_der):
            return abs(ang_rod_izq - ang_rod_der)
        
        return np.nan
        
    except Exception:
        return np.nan


def visualizar_analisis_postural(df, output_dir="./results/visualizations"):
    """Genera visualizaciones del análisis postural."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribución de inclinación lateral por acción
    plt.figure(figsize=(12, 6))
    df_plot = df[df["inclinacion_lateral"].notna()]
    
    sns.violinplot(data=df_plot, x="action", y="inclinacion_lateral")
    plt.title("Distribución de Inclinación Lateral por Acción")
    plt.xlabel("Acción")
    plt.ylabel("Inclinación Lateral (grados)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/inclinacion_lateral.png", dpi=300)
    plt.close()
    
    # 2. Alineación vertical por acción
    plt.figure(figsize=(12, 6))
    df_plot = df[df["alineacion_vertical"].notna()]
    
    sns.boxplot(data=df_plot, x="action", y="alineacion_vertical")
    plt.title("Alineación Vertical por Acción")
    plt.xlabel("Acción")
    plt.ylabel("Desviación de Vertical (grados)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/alineacion_vertical.png", dpi=300)
    plt.close()
    
    # 3. Simetría corporal
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    df_plot = df[df["simetria_brazos"].notna()]
    sns.histplot(data=df_plot, x="simetria_brazos", hue="action", ax=axes[0], bins=30)
    axes[0].set_title("Simetría de Brazos")
    axes[0].set_xlabel("Diferencia angular (grados)")
    
    df_plot = df[df["simetria_piernas"].notna()]
    sns.histplot(data=df_plot, x="simetria_piernas", hue="action", ax=axes[1], bins=30)
    axes[1].set_title("Simetría de Piernas")
    axes[1].set_xlabel("Diferencia angular (grados)")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/simetria_corporal.png", dpi=300)
    plt.close()
    
    print(f"\n✓ Visualizaciones guardadas en: {output_dir}/")


def main():
    """Función principal de análisis postural."""
    print("\n" + "="*70)
    print("PASO 2B: ANÁLISIS POSTURAL")
    print("="*70 + "\n")
    
    # Cargar landmarks
    origen_landmarks = "./data/processed/datosmediapipe.csv"
    origen_features = "./data/processed/model_features.csv"
    
    print(f"Cargando landmarks desde: {origen_landmarks}")
    
    try:
        df_landmarks = pd.read_csv(origen_landmarks)
        print(f"Landmarks cargados: {df_landmarks.shape}")
    except FileNotFoundError:
        print(f"ERROR: No se encuentra {origen_landmarks}")
        return
    
    print("\nCalculando métricas posturales...")
    
    # Calcular inclinación lateral
    tqdm.pandas(desc="Inclinación lateral")
    df_landmarks["inclinacion_lateral"] = df_landmarks.progress_apply(
        calcular_inclinacion_lateral, axis=1
    )
    
    # Calcular alineación vertical
    tqdm.pandas(desc="Alineación vertical")
    df_landmarks["alineacion_vertical"] = df_landmarks.progress_apply(
        calcular_alineacion_vertical, axis=1
    )
    
    # Si existen las características de ángulos, calcular simetrías
    try:
        df_features = pd.read_csv(origen_features)
        
        # Extraer ángulos básicos del primer frame de cada ventana
        # (esto es aproximado, idealmente se haría frame por frame)
        print("\nCalculando simetrías corporales...")
        
        # Aquí simplificamos usando las medias de los ángulos
        df_features["simetria_brazos"] = abs(
            df_features["ang_elb_izq_m"] - df_features["ang_elb_der_m"]
        )
        df_features["simetria_piernas"] = abs(
            df_features["ang_rod_izq_m"] - df_features["ang_rod_der_m"]
        )
        
        # Guardar features enriquecidas
        destino_features = "./data/processed/model_features_postural.csv"
        df_features.to_csv(destino_features, index=False)
        print(f"\n✓ Features con análisis postural: {destino_features}")
        
    except FileNotFoundError:
        print("\n⚠ No se encontraron features. Solo se analizarán landmarks.")
        df_features = df_landmarks.copy()
    
    # Guardar landmarks con análisis postural
    destino_landmarks = "./data/processed/datosmediapipe_postural.csv"
    df_landmarks.to_csv(destino_landmarks, index=False)
    print(f"✓ Landmarks con análisis postural: {destino_landmarks}")
    
    # Estadísticas
    print("\n" + "-"*70)
    print("ESTADÍSTICAS DE ANÁLISIS POSTURAL")
    print("-"*70)
    
    print(f"\nInclinación Lateral:")
    print(f"  Media: {df_landmarks['inclinacion_lateral'].mean():.2f}°")
    print(f"  Std: {df_landmarks['inclinacion_lateral'].std():.2f}°")
    print(f"  Min: {df_landmarks['inclinacion_lateral'].min():.2f}°")
    print(f"  Max: {df_landmarks['inclinacion_lateral'].max():.2f}°")
    
    print(f"\nAlineación Vertical:")
    print(f"  Media: {df_landmarks['alineacion_vertical'].mean():.2f}°")
    print(f"  Std: {df_landmarks['alineacion_vertical'].std():.2f}°")
    
    if "simetria_brazos" in df_features.columns:
        print(f"\nSimetría de Brazos:")
        print(f"  Media: {df_features['simetria_brazos'].mean():.2f}°")
        
        print(f"\nSimetría de Piernas:")
        print(f"  Media: {df_features['simetria_piernas'].mean():.2f}°")
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    visualizar_analisis_postural(df_features)
    
    print("\n" + "="*70)
    print("ANÁLISIS POSTURAL COMPLETADO")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
