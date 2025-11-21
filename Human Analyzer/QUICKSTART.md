# Inicio Rápido - Sistema de Reconocimiento de Actividad Humana

## Configuración Inicial (Una sola vez)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Organizar videos
# Crear estructura:
#   data/raw/videos/
#     ├── caminar/
#     ├── caminar-atras/
#     ├── girar/
#     ├── pararse/
#     └── sentarse/
```

## Opción A: Pipeline Automático (Recomendado)

```bash
python pipeline.py
```

Esto ejecutará automáticamente:
1. Extracción de landmarks
2. Ingeniería de características
3. Análisis postural
4. Entrenamiento de modelos
5. Evaluación

## Opción B: Ejecución Manual

```bash
# Paso 1: Extraer landmarks (toma varios minutos)
python 1_data_extraction/extract_landmarks.py --stride 1

# Paso 2: Generar características
python 2_feature_engineering/compute_features.py

# Paso 2B (Opcional): Análisis postural
python 2_feature_engineering/postural_analysis.py

# Paso 3: Entrenar modelos (toma varios minutos)
python 3_model_training/train_models.py

# Paso 3B (Opcional): Evaluación detallada
python 3_model_training/evaluate_models.py
```

## Sistema en Tiempo Real

```bash
python 4_real_time_app/real_time_system.py
```

**Controles:**
- `q`: Salir

## Verificación Rápida

Después de cada paso, verifica:

1. **Paso 1**: 
   - ✓ `data/processed/datosmediapipe.csv`
   - ✓ `data/processed/datos_analisis.csv`

2. **Paso 2**: 
   - ✓ `data/processed/model_features.csv`

3. **Paso 2B** (Opcional):
   - ✓ `results/visualizations/inclinacion_lateral.png`
   - ✓ `results/visualizations/alineacion_vertical.png`
   - ✓ `results/visualizations/simetria_corporal.png`

4. **Paso 3**: 
   - ✓ `assets/best_random_forest_model.joblib`
   - ✓ `assets/best_xgboost_model.joblib`
   - ✓ `assets/label_encoder.joblib`
   - ✓ `results/evaluations/random_forest_confusion_matrix.png`
   - ✓ `results/evaluations/xgboost_confusion_matrix.png`

5. **Paso 3B** (Opcional):
   - ✓ `results/evaluations/models_comparison.png`

## Estructura Generada

```
Human Analyzer/
├── data/processed/              # CSVs generados
├── assets/                      # Modelos entrenados
└── results/                     # Análisis y visualizaciones
    ├── evaluations/
    └── visualizations/
```

## Comandos Útiles

### Procesar solo 1 de cada 5 frames (más rápido)

```bash
python 1_data_extraction/extract_landmarks.py --stride 5
```

### Pipeline sin pasos opcionales

```bash
python pipeline.py --skip-optional
```

### Ver ayuda de cualquier script

```bash
python [script].py --help
```

## Problemas Comunes

### No detecta la cámara

```bash
# Cierra otras apps que usen la cámara
# Verifica permisos en Windows
```

### Error al cargar modelos

```bash
# Asegúrate de entrenar primero
python 3_model_training/train_models.py
```

### Predicciones erráticas

- Mejora la iluminación
- Mantente a 2-3 metros de la cámara
- Asegúrate de que el cuerpo completo sea visible

## Métricas Esperadas

- **Accuracy**: >90%
- **FPS**: 15-30 fps
- **Latencia**: 50-100ms

## Próximos Pasos

1. Revisa métricas en `results/evaluations/`
2. Explora visualizaciones en `results/visualizations/`
3. Ejecuta el sistema en tiempo real
4. Experimenta con diferentes configuraciones

Para más información, consulta `README_NUEVO.md`
