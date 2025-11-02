# Entrega 2: Análisis y Clasificación de Actividades Humanas

## Descripción General
Esta entrega se centra en el procesamiento, análisis y clasificación de datos de actividades humanas utilizando técnicas de aprendizaje automático. Se implementó un pipeline completo que incluye preparación de datos, preprocesamiento, optimización de hiperparámetros, entrenamiento de modelos y visualización de resultados.

## Estructura del Proyecto

```
Entrega 2/
├── data/
│   ├── raw/                    # Datos sin procesar (landmarks)
│   ├── processed/              # Datos procesados
│   │   ├── combined_dataset.csv    # Dataset combinado con etiquetas
│   │   ├── train_dataset.csv       # Conjunto de entrenamiento
│   │   ├── test_dataset.csv        # Conjunto de prueba
│   │   └── scaler.pkl             # Objeto StandardScaler guardado
│   └── labels/                 # Archivos de etiquetas
├── results/
│   ├── models/                 # Modelos entrenados
│   │   ├── svm.pkl
│   │   ├── randomforest.pkl
│   │   ├── xgboost.pkl
│   │   └── label_encoder.pkl
│   ├── models_optimized/       # Modelos con hiperparámetros optimizados
│   │   ├── svm_optimized.pkl
│   │   ├── randomforest_optimized.pkl
│   │   └── xgboost_optimized.pkl
│   ├── metrics/                # Métricas y resultados
│   │   ├── model_comparison.csv
│   │   └── optimized_model_comparison.csv
│   └── visualizations/         # Gráficas y visualizaciones
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       ├── roc_curves.png
│       └── class_metrics.png
└── src/                       # Código fuente
    ├── prepare_dataset.py
    ├── preprocess_dataset.py
    ├── train_models.py
    ├── optimize_models.py
    ├── visualize_results.py
    └── pipeline.py
```

## Scripts y su Funcionalidad

### 1. prepare_dataset.py
- **Función**: Preparación inicial del dataset
- **Operaciones**:
  - Carga y procesa el archivo de etiquetas
  - Normaliza nombres de videos
  - Combina características con etiquetas
  - Genera el dataset combinado inicial
- **Salida**: `combined_dataset.csv`

### 2. preprocess_dataset.py
- **Función**: Preprocesamiento y balance de datos
- **Operaciones**:
  - Manejo de valores faltantes
  - Normalización de características usando StandardScaler
  - Balanceo de clases usando SMOTE
  - División en conjuntos de entrenamiento y prueba
- **Salidas**:
  - `train_dataset.csv`
  - `test_dataset.csv`
  - `scaler.pkl`

### 3. optimize_models.py
- **Función**: Optimización de hiperparámetros
- **Operaciones**:
  - Búsqueda en cuadrícula (GridSearchCV) para cada modelo
  - Optimización de SVM, Random Forest y XGBoost
  - Validación cruzada con 5 folds
- **Salidas**: Modelos optimizados en `models_optimized/`

### 4. train_models.py
- **Función**: Entrenamiento de modelos
- **Operaciones**:
  - Carga los hiperparámetros optimizados
  - Entrena SVM, Random Forest y XGBoost
  - Realiza validación cruzada
  - Evalúa en conjunto de prueba
- **Salidas**: 
  - Modelos entrenados en `models/`
  - Métricas en `metrics/model_comparison.csv`

### 5. visualize_results.py
- **Función**: Generación de visualizaciones
- **Operaciones**:
  - Comparación de rendimiento entre modelos
  - Matriz de confusión
  - Importancia de características
  - Curvas ROC
  - Métricas por clase
- **Salidas**: Visualizaciones en `visualizations/`

### 6. pipeline.py
- **Función**: Orquestación del proceso completo
- **Operaciones**:
  - Ejecuta todos los scripts en orden
  - Maneja errores y excepciones
  - Registra tiempos de ejecución
  - Proporciona resumen del proceso

## Resultados Principales

### Modelos Implementados
1. **Support Vector Machine (SVM)**
   - Optimización de kernel, C y gamma
   - Adecuado para problemas no lineales

2. **Random Forest**
   - Optimización de n_estimators, max_depth, min_samples
   - Mejor rendimiento general en el conjunto de datos

3. **XGBoost**
   - Optimización de max_depth, learning_rate, n_estimators
   - Buen balance entre velocidad y rendimiento

### Métricas y Evaluación
- Accuracy como métrica principal
- Validación cruzada con 5 folds
- Matrices de confusión para análisis detallado
- Curvas ROC para cada clase
- Análisis de importancia de características

## Instrucciones de Uso

1. Asegurarse de tener todas las dependencias instaladas:
   ```
   pip install sklearn xgboost pandas numpy matplotlib seaborn imblearn
   ```

2. Ejecutar el pipeline completo:
   ```
   python pipeline.py
   ```

3. Los resultados se generarán en la carpeta `results/`

## Conclusiones

- El Random Forest mostró el mejor rendimiento general
- La optimización de hiperparámetros mejoró significativamente los resultados
- Las actividades estáticas (Sit, Squat) son más fáciles de clasificar que las dinámicas
- El balanceo de clases con SMOTE ayudó a mejorar la clasificación de clases minoritarias