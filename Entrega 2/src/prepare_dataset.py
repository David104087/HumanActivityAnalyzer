import os
import pandas as pd
import json
from glob import glob

def load_labels(labels_path):
    """
    Carga y procesa el archivo de etiquetas
    """
    labels_df = pd.read_csv(labels_path)
    frame_labels = {}
    
    for _, row in labels_df.iterrows():
        # Normalizar el nombre del video
        original_name = os.path.basename(row['video'])
        video_name = normalize_video_name(original_name)
        
        print(f"\nProcesando etiqueta de video:")
        print(f"  Original: {original_name}")
        print(f"  Normalizado: {video_name}")
        
        # El JSON tiene dobles comillas extras, las eliminamos
        video_labels_str = row['videoLabels'].replace('""', '"')
        if video_labels_str.startswith('"') and video_labels_str.endswith('"'):
            video_labels_str = video_labels_str[1:-1]
            
        video_labels = json.loads(video_labels_str)
        frame_labels[video_name] = {}
        
        # Procesar cada etiqueta y sus rangos
        for label_info in video_labels:
            label = label_info['timelinelabels'][0]
            for range_info in label_info['ranges']:
                start = range_info['start']
                end = range_info['end']
                for frame in range(start, end + 1):
                    frame_labels[video_name][frame] = label
    
    return frame_labels

def normalize_video_name(name):
    """
    Normaliza el nombre del video para que coincida entre features y etiquetas
    """
    # Eliminar la extensión .mp4
    name = name.replace('.mp4', '')
    
    # Si el nombre contiene un guión, tomamos solo la parte después del guión
    if '-' in name:
        name = name.split('-', 1)[1].strip()
    
    # Extraer el número entre paréntesis si existe
    number = None
    if "(" in name and ")" in name:
        number = name[name.find("(")+1:name.find(")")]
        # Eliminar el texto entre paréntesis del nombre
        name = name[:name.find("(")].strip() + name[name.find(")")+1:].strip()
    
    # Manejar nombres de WhatsApp
    if "WhatsApp" in name:
        # Quedarnos con la parte después de "WhatsApp Video "
        if "WhatsApp Video " in name:
            name = name.replace("WhatsApp Video ", "")
        
        # Preservar el formato de fecha y hora
        if "2025" in name:
            parts = name.split(" at ")
            if len(parts) == 2:
                date_part = parts[0].strip()
                time_part = parts[1].strip()
                
                # Asegurarnos de mantener el formato de fecha completo
                if "2025" in date_part:
                    # Mantener el formato "2025-10-12"
                    name = "2025-10-12 at " + time_part
                
                # Agregar el número si existía, al final
                if number:
                    name = name + " " + number
    
    return name.strip()

def load_features(features_dir):
    """
    Carga todos los archivos de features y los combina
    """
    all_features = []
    feature_files = glob(os.path.join(features_dir, '*_landmarks_features.csv'))
    print(f"\nEncontrados {len(feature_files)} archivos de features")
    
    for file_path in feature_files:
        file_name = os.path.basename(file_path)
        video_name = file_name.replace('_landmarks_features.csv', '')
        print(f"\nProcesando archivo de features:")
        print(f"  Original: {file_name}")
        
        features_df = pd.read_csv(file_path)
        
        # Normalizar el nombre del video
        normalized_name = normalize_video_name(video_name)
        print(f"  Video original: {video_name}")
        print(f"  Video normalizado: {normalized_name}")
        
        features_df['video_name'] = normalized_name
        all_features.append(features_df)
    
    if not all_features:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal de frames cargados: {len(combined_df)}")
    return combined_df

def combine_features_and_labels(features_df, frame_labels):
    """
    Asigna etiquetas a cada frame en el dataset de features
    """
    def get_label(row):
        video_labels = frame_labels.get(row['video_name'], {})
        if not video_labels:
            print(f"No se encontraron etiquetas para el video: {row['video_name']}")
            return 'unknown'
        label = video_labels.get(row['frame'], 'unknown')
        return label
    
    features_df['label'] = features_df.apply(get_label, axis=1)
    return features_df

def main():
    # Directorios
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(base_dir, 'data', 'raw')
    labels_path = os.path.join(base_dir, 'data', 'labels', 'labels.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    
    print(f"\nDirectorios a utilizar:")
    print(f"- Base dir: {base_dir}")
    print(f"- Features dir: {features_dir}")
    print(f"- Labels path: {labels_path}")
    print(f"- Output dir: {output_dir}\n")
    
    # Asegurar que existe el directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar que existen los archivos necesarios
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"No se encontró el archivo de etiquetas en: {labels_path}")
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"No se encontró el directorio de features en: {features_dir}")
    
    print("Cargando etiquetas...")
    frame_labels = load_labels(labels_path)
    
    print("Cargando features...")
    features_df = load_features(features_dir)
    
    if features_df.empty:
        raise ValueError(f"No se encontraron archivos de features en: {features_dir}")
    
    print("Combinando features con etiquetas...")
    dataset = combine_features_and_labels(features_df, frame_labels)
    
    # Eliminar filas sin etiquetas y la columna temporal video_name
    dataset = dataset[dataset['label'] != 'unknown'].drop('video_name', axis=1)
    
    # Guardar dataset procesado
    output_path = os.path.join(output_dir, 'combined_dataset.csv')
    dataset.to_csv(output_path, index=False)
    print(f"\nDataset guardado en: {output_path}")
    print(f"Número total de muestras: {len(dataset)}")
    print("\nDistribución de etiquetas:")
    print(dataset['label'].value_counts())
    print("\nColumnas en el dataset final:")
    print(dataset.columns.tolist())

if __name__ == "__main__":
    main()