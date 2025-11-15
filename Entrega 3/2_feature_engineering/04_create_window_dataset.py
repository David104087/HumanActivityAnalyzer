# 2_feature_engineering/04_create_window_dataset.py
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# --- Configuración ---

# 1. Directorio donde están los 20 archivos de features
FEATURES_DIR = "data/features_per_frame"

# 2. Archivo con las etiquetas procesadas
LABELS_FILE = "data/labels/processed/processed_labels.csv"

# 3. Archivo de salida para el dataset de entrenamiento final
OUTPUT_DIR = "data/processed_windowed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "windowed_dataset.csv")

# 4. TAMAÑO DE LA VENTANA (Memoria)
#    Cuántos frames de contexto usaremos. 5 es un buen punto de partida.
WINDOW_SIZE = 5

# 5. Los 6 features base que calculamos
FEATURE_COLS = [
    'knee_left', 'knee_right', 'hip_left', 'hip_right', 
    'trunk_angle', 'motion_energy'
]

# --- Fin Configuración ---


def load_all_features(features_dir):
    """Carga y concatena todos los ..._features.csv en un solo DataFrame."""
    all_files = glob.glob(os.path.join(features_dir, "*_features.csv"))
    if not all_files:
        print(f"ERROR: No se encontraron archivos *_features.csv en {features_dir}")
        return pd.DataFrame()
        
    df_list = []
    print(f"Cargando {len(all_files)} archivos de features...")
    for f in all_files:
        df_list.append(pd.read_csv(f))
        
    return pd.concat(df_list, ignore_index=True)


def create_windowed_dataset():
    """
    Función principal para cargar, combinar, etiquetar y crear ventanas.
    """
    
    # 1. Cargar Features
    df_features = load_all_features(FEATURES_DIR)
    if df_features.empty:
        return

    # 2. Cargar Etiquetas
    try:
        df_labels = pd.read_csv(LABELS_FILE)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo de etiquetas en {LABELS_FILE}")
        return
        
    print(f"Features cargados: {len(df_features)} frames")
    print(f"Etiquetas cargadas: {len(df_labels)} frames")

    # 3. Unir Features y Etiquetas
    print("Uniendo features y etiquetas...")
    df_full = pd.merge(
        df_features, 
        df_labels, 
        on=["video_name", "frame"],
        how="inner" # "inner" descarta frames que no tengan etiqueta
    )
    
    # Si un frame no tenía etiqueta, se descarta (muy importante)
    print(f"Frames después de unir con etiquetas: {len(df_full)}")

    # 4. Ordenar y Preparar para Ventanas
    df_full = df_full.sort_values(by=["video_name", "frame"])
    
    # Manejar NaNs (si un ángulo no se pudo calcular):
    # Rellenamos hacia adelante (ffill) y luego hacia atrás (bfill)
    # Esto asegura que no haya NaNs en nuestros datos de ventana
    df_full[FEATURE_COLS] = df_full[FEATURE_COLS].ffill().bfill()
    
    all_windowed_data = []
    
    print(f"Creando ventanas de tamaño {WINDOW_SIZE}...")
    
    # 5. Iterar por cada video por separado
    # (¡CRUCIAL! para no crear ventanas que mezclen Video1 y Video2)
    grouped = df_full.groupby('video_name')
    for video_name, group in tqdm(grouped, desc="Procesando videos"):
        
        features_array = group[FEATURE_COLS].values
        labels_array = group['label'].values
        
        # No podemos crear ventanas para videos más cortos que la ventana
        if len(group) < WINDOW_SIZE:
            continue

        # 6. Lógica de Ventana Deslizante
        windowed_features_list = []
        windowed_labels_list = []
        
        # Empezamos desde el frame W-1 para tener un historial completo
        for i in range(WINDOW_SIZE - 1, len(features_array)):
            
            # Obtener la ventana de features (ej. 5 frames, 6 features)
            # El rango es [i - (W-1)] hasta [i+1]
            window_features = features_array[i - (WINDOW_SIZE - 1) : i + 1]
            
            # Aplanar (flatten) la ventana (de (5, 6) -> a un vector de 30)
            window_flat = window_features.flatten()
            
            # La etiqueta es la del frame actual (el último de la ventana)
            label = labels_array[i]
            
            windowed_features_list.append(window_flat)
            windowed_labels_list.append(label)

        # 7. Crear un DataFrame temporal para este video
        if windowed_features_list:
            # Crear los nombres de las 30 columnas
            # (ej. f_0_knee_left, f_0_knee_right, ..., f_4_motion_energy)
            new_cols = [f"f_{t}_{col}" for t in range(WINDOW_SIZE) for col in FEATURE_COLS]
            
            df_video_windowed = pd.DataFrame(
                np.array(windowed_features_list), 
                columns=new_cols
            )
            df_video_windowed['label'] = windowed_labels_list
            all_windowed_data.append(df_video_windowed)

    # 8. Combinar datos de todos los videos y Guardar
    print("Combinando datos finales...")
    final_dataset = pd.concat(all_windowed_data, ignore_index=True)
    
    # Asegurar que el directorio de salida exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    final_dataset.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- ¡Proceso Completado! ---")
    print(f"Dataset final de ventanas guardado en: {OUTPUT_FILE}")
    print(f"Total de muestras (ventanas) creadas: {len(final_dataset)}")
    print(f"Total de features por muestra: {len(final_dataset.columns) - 1}")


if __name__ == "__main__":
    create_windowed_dataset()