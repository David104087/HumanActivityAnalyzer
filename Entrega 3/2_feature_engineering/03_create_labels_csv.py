import pandas as pd
import json
import os

# 1. Lista de TUS archivos CSV de Label Studio
LABEL_STUDIO_FILES = [
    "project-3-at-2025-10-31-15-38-164b9b98.csv",
    "project-3-at-2025-10-31-20-18-44786f60.csv"
]

# 2. Ruta de salida (la he ajustado a la que te salió en el log)
OUTPUT_LABELS_PATH = "data/labels/processed/processed_labels.csv"

# 3. Etiquetas a ignorar
LABELS_TO_IGNORE = ['unknown', 'discard']

# 4. !! TRADUCTOR DE NOMBRES (CORREGIDO) !!
#    La lista ahora usa los nombres sanitizados de Label Studio
#    para los videos 11 al 20.
MASTER_VIDEO_LIST = [
    # Videos 1-10 (Estos funcionaron bien)
    "VID_20251012_2045270692",
    "VID_20251012_2047230572",
    "VID_20251012_2048479882",
    "VID_20251012_2049446052",
    "VID_20251012_2050331212",
    "VID_20251012_2051467552",
    "VID_20251012_2053043673",
    "VID_20251012_2053457622",
    "VID_20251012_2054460522",
    "VID_20251012_2055488053",
    
    # Videos 11-20 (CORREGIDOS con nombres sanitizados del log)
    # El orden se basa en tu lista de archivos original
    "WhatsApp_Video_2025-10-12_at_9.47.33_PM",   # Video 11
    "WhatsApp_Video_2025-10-12_at_9.47.43_PM",   # Video 12
    "WhatsApp_Video_2025-10-12_at_9.49.12_PM_1", # Video 13
    "WhatsApp_Video_2025-10-12_at_9.49.12_PM",   # Video 14
    "WhatsApp_Video_2025-10-12_at_9.49.13_PM_1", # Video 15
    "WhatsApp_Video_2025-10-12_at_9.49.13_PM_2", # Video 16
    "WhatsApp_Video_2025-10-12_at_9.49.13_PM_3", # Video 17
    "WhatsApp_Video_2025-10-12_at_9.49.13_PM",   # Video 18
    "WhatsApp_Video_2025-10-12_at_9.49.14_PM_1", # Video 19
    "WhatsApp_Video_2025-10-12_at_9.49.14_PM",   # Video 20
]

# 5. Creamos el diccionario de mapeo
#    Ej: {"VID_20251012_2045270692": "Video1", ... , "WhatsApp_Video_2025-10-12_at_9.47.33_PM": "Video11"}
video_name_mapper = {name: f"Video{i+1}" for i, name in enumerate(MASTER_VIDEO_LIST)}

# --- Fin Configuración ---


def get_simple_video_name(label_studio_path, mapper):
    """
    Busca en la ruta de Label Studio (ej. /data/.../abc-VIDEO_NAME.mp4)
    y devuelve el nombre simple (ej. "Video1") usando el 'mapper'.
    """
    for original_name, simple_name in mapper.items():
        if original_name in label_studio_path:
            return simple_name
    
    print(f"  -> ADVERTENCIA: No se encontró mapeo para {label_studio_path}")
    return None


def process_label_studio_export(raw_csv_paths, output_csv_path, mapper):
    """
    Lee las exportaciones CSV de Label Studio y las convierte en un
    mapeo de frame-por-frame (video_name, frame, label) usando
    los nombres simples ("Video1", "Video2", etc.).
    """
    all_processed_rows = []
    total_frames = 0
    
    print("Iniciando procesamiento de archivos de Label Studio...")

    for csv_file in raw_csv_paths:
        print(f"\nLeyendo archivo: {csv_file}")
        
        # Ajusta la ruta para que coincida con tu log
        full_path = f"data/labels/unprocessed/{os.path.basename(csv_file)}"
        
        try:
            if not os.path.exists(full_path):
                 print(f"ERROR: No se encontró el archivo en {full_path}")
                 print(f"Buscando en la carpeta actual: {csv_file}")
                 if not os.path.exists(csv_file):
                    print(f"ERROR: Tampoco se encontró en {csv_file}. Omitiendo.")
                    continue
                 else:
                    full_path = csv_file # Usar la ruta de la carpeta actual
            
            df_raw = pd.read_csv(full_path)
        except FileNotFoundError:
            print(f"ERROR: No se pudo leer {full_path}. Omitiendo.")
            continue

        print(f"Procesando {len(df_raw)} filas de anotaciones...")
        for _, row in df_raw.iterrows():
            try:
                raw_video_path = row['video']
                simple_name = get_simple_video_name(raw_video_path, mapper)
                
                if simple_name is None:
                    continue 

                labels_str = row['videoLabels']
                clean_str = labels_str.replace('""', '"')
                if clean_str.startswith('"') and clean_str.endswith('"'):
                    clean_str = clean_str[1:-1]
                    
                labels_json = json.loads(clean_str)

                for segment in labels_json:
                    label_name = segment['timelinelabels'][0]

                    if label_name in LABELS_TO_IGNORE:
                        continue

                    for time_range in segment['ranges']:
                        start_frame = time_range['start']
                        end_frame = time_range['end']

                        for frame_num in range(start_frame, end_frame + 1):
                            all_processed_rows.append({
                                'video_name': simple_name,
                                'frame': frame_num,
                                'label': label_name
                            })
                            total_frames += 1

            except Exception as e:
                print(f"ADVERTENCIA: Omitiendo fila en {csv_file} debido a un error: {e}")
                print(f"Datos problemáticos: video='{row.get('video')}', labels='{row.get('videoLabels')[:50]}...'")

    if not all_processed_rows:
        print("ERROR: No se procesaron filas. ¿Los archivos de entrada son correctos?")
        return

    print(f"\nProcesamiento completado. Total de {total_frames} frames etiquetados.")
    df_final = pd.DataFrame(all_processed_rows)
    
    # Asegurar que el directorio de salida exista
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    df_final.to_csv(output_csv_path, index=False)
    print(f"¡Éxito! Archivo de etiquetas guardado en: {output_csv_path}")
    print("\nResumen de etiquetas generadas:")
    print(df_final['label'].value_counts())
    print("\nVideos encontrados y procesados (Deberían ser 20):")
    print(df_final['video_name'].value_counts().sort_index())


if __name__ == "__main__":
    # Verificamos que los archivos de Label Studio existan en la ubicación esperada
    # Asumimos que están en 'data/labels/unprocessed/' según tu log
    
    base_unprocessed_dir = "data/labels/unprocessed"
    paths_to_check = [os.path.join(base_unprocessed_dir, f) for f in LABEL_STUDIO_FILES]
    
    # Si no están allí, los buscamos en la carpeta actual (donde se ejecuta el script)
    if not all(os.path.exists(p) for p in paths_to_check):
        print("Advertencia: No se encontraron los CSV en 'data/labels/unprocessed/'.")
        print("Buscando en la carpeta actual...")
        paths_to_check = LABEL_STUDIO_FILES # Revertir a los nombres base
    
    # Ejecutamos la función principal
    process_label_studio_export(
        LABEL_STUDIO_FILES, # Pasamos los nombres base
        OUTPUT_LABELS_PATH, 
        video_name_mapper
    )