# Ejecución de los scripts

Primero, ubíquese fuera de la carpeta src, a la altura del Entregable 1, y ejecute los siguientes scripts en este orden:

## 0. análisis inicial
python src/00_basic_video_eda.py --videos_dir ./videos --out_dir ./reports

## 1. extraer landmarks
python src/01_extract_landmarks.py --videos_dir ./videos --out_dir ./landmarks --fps_sample 1

## 2. preprocesar cada CSV producido
python src/02_preprocess_landmarks.py --landmark_csv ./landmarks/video1_landmarks.csv

## 3. computar features
python src/03_compute_features.py --preprocessed_csv ./landmarks/video1_landmarks_preprocessed.csv

## 4. EDA y unir con anotaciones
python src/04_eda_and_visualization.py --preprocessed_csv ./landmarks/video1_landmarks_preprocessed.csv --features_csv ./landmarks/video1_features.csv --annotations_csv ./annotations/video1_annotations.csv --out_dir ./reports
