# Ejecución de los scripts

Primero, ubíquese fuera de la carpeta src, a la altura del Entregable 1, ubique todos los videos dentro de la carpeta videos/ a la misma altura de la carpeta src/, y ejecute los siguientes scripts en este orden:

## 0. análisis inicial
```bash
python src/00_basic_video_eda.py --videos_dir ./videos --out_dir ./reports
```

## 1. extraer landmarks
```bash
python src/01_extract_landmarks.py --videos_dir ./videos --out_dir ./landmarks --fps_sample 1
```

## 2. preprocesar cada CSV producido

Para un video en específico, puede ejecutar este comando:
```bash
python src/02_preprocess_landmarks.py --landmark_csv ./landmarks/video1_landmarks.csv
```

Sin embargo, para automatizar este proceso, use este comando de bash:
```bash
#!/bin/bash
LANDMARKS_DIR="./landmarks"

echo "Preprocesando archivos..."
for f in "$LANDMARKS_DIR"/*_landmarks.csv; do
    [ -e "$f" ] || continue
    python src/02_preprocess_landmarks.py --landmark_csv "$f"
done
```

## 3. computar features

Para un video en específico, puede ejecutar este comando:
```bash
python src/03_compute_features.py --preprocessed_csv ./landmarks/video1_landmarks_preprocessed.csv
```

Sin embargo, para automatizar este proceso, use este comando de bash:
```bash
#!/bin/bash
LANDMARKS_DIR="./landmarks"

echo "Generando features..."
for f in "$LANDMARKS_DIR"/*_preprocessed.csv; do
    [ -e "$f" ] || continue
    python src/03_compute_features.py --preprocessed_csv "$f"
done
```

## 4. EDA y unir con anotaciones

Para automatizar este proceso, use este comando de bash:
```bash
#!/bin/bash
LANDMARKS_DIR="./landmarks"
REPORTS_DIR="./reports"

echo "Ejecutando análisis EDA y visualización..."
mkdir -p "$REPORTS_DIR"

for pre_csv in "$LANDMARKS_DIR"/*_landmarks_preprocessed.csv; do
    [ -e "$pre_csv" ] || continue

    # Nombre base del video (sin sufijo)
    base=$(basename "$pre_csv" "_landmarks_preprocessed.csv")

    # Archivo de features correspondiente
    features_csv="$LANDMARKS_DIR/${base}_landmarks_features.csv"

    # Verificar existencia de ambos archivos
    if [[ -f "$pre_csv" && -f "$features_csv" ]]; then
        echo "→ Ejecutando EDA para: $base"
        python src/04_eda_and_visualization.py \
            --preprocessed_csv "$pre_csv" \
            --features_csv "$features_csv" \
            --out_dir "$REPORTS_DIR"
    else
        echo "Faltan archivos para $base — se omite."
        echo "   preprocessed: $([[ -f $pre_csv ]] && echo OK || echo NO)"
        echo "   features: $([[ -f $features_csv ]] && echo OK || echo NO)"
    fi
done

echo "EDA completado para todos los videos disponibles."
```