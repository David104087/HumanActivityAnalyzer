# Sistema de Reconocimiento de Actividad Humana en Tiempo Real

Sistema completo de análisis y clasificación de actividades humanas usando MediaPipe, Machine Learning y Computer Vision. Clasifica 5 acciones en tiempo real: **caminar**, **caminar-atras**, **girar**, **pararse** y **sentarse**.

## Estructura del Proyecto

```
Human Analyzer/
│
├── 1_data_extraction/              # Paso 1: Extracción de datos
│   └── extract_landmarks.py        # Extrae landmarks con MediaPipe
│
├── 2_feature_engineering/          # Paso 2: Ingeniería de características
│   ├── compute_features.py         # Calcula ángulos y características temporales
│   └── postural_analysis.py        # Análisis de inclinaciones y posturas
│
├── 3_model_training/               # Paso 3: Entrenamiento y evaluación
│   ├── train_models.py             # Entrena Random Forest y XGBoost
│   └── evaluate_models.py          # Evaluación detallada de modelos
│
├── 4_real_time_app/                # Paso 4: Aplicación en tiempo real
│   ├── real_time_system.py         # Sistema de inferencia en vivo
│   └── utils.py                    # Funciones auxiliares
│
├── data/                           # Datos del proyecto
│   ├── raw/videos/                 # Videos originales (por acción)
│   └── processed/                  # CSVs generados
│
├── assets/                         # Modelos entrenados
│   ├── best_random_forest_model.joblib
│   ├── best_xgboost_model.joblib
│   └── label_encoder.joblib
│
├── results/                        # Resultados y análisis
│   ├── evaluations/                # Métricas y reportes
│   └── visualizations/             # Gráficas y visualizaciones
│
├── pipeline.py                     # Pipeline completo automatizado
├── requirements.txt                # Dependencias
├── .gitignore                      # Archivos ignorados por git
└── README.md                       # Documentación
```

## Características Principales

### Análisis Técnico
- **Extracción de Landmarks**: 33 puntos de MediaPipe con normalización robusta
- **Ángulos Biomecánicos**: 8 ángulos clave (codos, hombros, caderas, rodillas)
- **Características Temporales**: 48 features por ventana de 15 frames
- **Análisis Postural**: Inclinaciones laterales, alineación vertical, simetría corporal

### Machine Learning
- **Modelos**: Random Forest y XGBoost con GridSearchCV
- **Pipeline**: StandardScaler → PCA (95% varianza) → Clasificador
- **Validación**: Split estratificado 70/30 con cross-validation
- **Métricas**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Sistema en Tiempo Real
- **Inferencia**: Detección y clasificación en vivo desde webcam
- **Suavizado**: Ventana de 8 frames con votación mayoritaria
- **Gate de Quietud**: Detección automática de estados sin movimiento
- **UI Moderna**: Panel lateral con círculo de progreso y métricas en tiempo real

## Instalación

### 1. Clonar y configurar entorno

```bash
cd "Human Analyzer"

# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso del Sistema

### Opción A: Pipeline Completo (Recomendado)

Ejecuta todo el proceso automáticamente:

```bash
python pipeline.py
```

Para saltar pasos opcionales:

```bash
python pipeline.py --skip-optional
```

### Opción B: Ejecución Manual por Pasos

#### Paso 1: Extraer Landmarks

```bash
python 1_data_extraction/extract_landmarks.py --stride 1
```

**Salidas:**
- `data/processed/datosmediapipe.csv`: Landmarks 3D
- `data/processed/datos_analisis.csv`: Metadatos de video

#### Paso 2: Ingeniería de Características

```bash
python 2_feature_engineering/compute_features.py
```

**Salida:**
- `data/processed/model_features.csv`: 48 características por ventana

#### Paso 2B: Análisis Postural (Opcional)

```bash
python 2_feature_engineering/postural_analysis.py
```

**Salidas:**
- `data/processed/model_features_postural.csv`
- Visualizaciones en `results/visualizations/`

#### Paso 3: Entrenar Modelos

```bash
python 3_model_training/train_models.py
```

**Salidas:**
- `assets/best_random_forest_model.joblib`
- `assets/best_xgboost_model.joblib`
- `assets/label_encoder.joblib`
- Matrices de confusión en `results/evaluations/`

#### Paso 3B: Evaluar Modelos (Opcional)

```bash
python 3_model_training/evaluate_models.py
```

**Salidas:**
- Métricas detalladas por clase
- Gráfica comparativa de modelos
- Análisis de errores

#### Paso 4: Sistema en Tiempo Real

```bash
python 4_real_time_app/real_time_system.py
```

**Controles:**
- `q`: Salir del sistema

## Preparación de Datos

### Estructura de Videos

Organiza tus videos en la siguiente estructura:

```
data/raw/videos/
├── caminar/
│   ├── video1.mp4
│   └── video2.mp4
├── caminar-atras/
│   ├── video1.mp4
│   └── video2.mp4
├── girar/
│   ├── video1.mp4
│   └── video2.mp4
├── pararse/
│   ├── video1.mp4
│   └── video2.mp4
└── sentarse/
    ├── video1.mp4
    └── video2.mp4
```

### Formatos Soportados

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)
- M4V (.m4v)

## Metodología CRISP-DM

El proyecto sigue la metodología CRISP-DM:

1. **Comprensión del Negocio**: Sistema de análisis de actividad humana para detección automática
2. **Comprensión de los Datos**: Videos de 5 acciones con MediaPipe para landmarks
3. **Preparación de Datos**: Normalización, extracción de ángulos, ventanas temporales
4. **Modelado**: Random Forest y XGBoost con GridSearchCV
5. **Evaluación**: Métricas múltiples, análisis de errores, validación cruzada
6. **Despliegue**: Sistema en tiempo real con interfaz gráfica

## Especificaciones Técnicas

### Ángulos Biomecánicos (8)

1. Codo izquierdo: Hombro-Codo-Muñeca (izq)
2. Codo derecho: Hombro-Codo-Muñeca (der)
3. Hombro izquierdo: Codo-Hombro-Cadera (izq)
4. Hombro derecho: Codo-Hombro-Cadera (der)
5. Cadera izquierda: Hombro-Cadera-Rodilla (izq)
6. Cadera derecha: Hombro-Cadera-Rodilla (der)
7. Rodilla izquierda: Cadera-Rodilla-Tobillo (izq)
8. Rodilla derecha: Cadera-Rodilla-Tobillo (der)

### Características (48 totales)

| Grupo | Tipo | Cantidad |
|-------|------|----------|
| Posición | Mean | 8 |
| Posición | Std | 8 |
| Posición | Min | 8 |
| Posición | Max | 8 |
| Velocidad | Mean | 8 |
| Velocidad | Std | 8 |
| **TOTAL** | | **48** |

### Normalización

1. **Traslación**: Restar centro de cadera (promedio landmarks 23 y 24)
2. **Escalado**: Dividir por distancia euclidiana entre hombros (11 y 12)
3. **Validación**: Verificar distancia > 1e-6

### Análisis Postural Adicional

- **Inclinación Lateral**: Ángulo entre hombros respecto a horizontal
- **Alineación Vertical**: Desviación del tronco respecto a vertical
- **Simetría de Brazos**: Diferencia entre ángulos de codos
- **Simetría de Piernas**: Diferencia entre ángulos de rodillas

## Interfaz en Tiempo Real

### Características de la UI

- **Panel Lateral Derecho**: 
  - Nombre de actividad detectada
  - Círculo de progreso con porcentaje de confianza
  - Estado del detector (Persona detectada / Sin persona)
  - Contador de frames procesados
  
- **Banner Superior**: Título del sistema

- **Esqueleto Personalizado**:
  - Landmarks: Cyan vibrante
  - Conexiones: Naranja dorado

### Código de Colores

- **Verde**: Confianza > 80%
- **Azul**: Confianza 50-80%
- **Rojo**: Confianza < 50%

## Métricas Esperadas

- **Accuracy**: >90%
- **FPS**: 15-30 fps
- **Latencia**: 50-100ms por predicción

## Resolución de Problemas

### Error: No se encuentra model_features.csv

```bash
python 1_data_extraction/extract_landmarks.py --stride 1
python 2_feature_engineering/compute_features.py
```

### Error: No se puede acceder a la cámara

1. Cierra otras aplicaciones que usen la cámara
2. Verifica permisos de cámara en Windows
3. Intenta cambiar índice de cámara en el código

### Rendimiento bajo (FPS bajos)

1. Usa `model_complexity=0` en MediaPipe (ya configurado)
2. Aumenta `--stride` en extracción de landmarks
3. Cierra aplicaciones en segundo plano

## Aspectos Éticos

### Consideraciones de Privacidad

- Los landmarks solo capturan posiciones articulares, no identidad facial
- Procesamiento en tiempo real sin almacenamiento de video
- No se recolecta información personal identificable

### Uso Responsable

- Sistema diseñado para análisis de movimiento, no vigilancia
- Obtener consentimiento antes de grabar personas
- Datos deben ser anónimos y seguros

## Autores

Desarrollado como proyecto final para Inteligencia Artificial I  
Semestre 2025-2  
Facultad de Ingeniería, Diseño y Ciencias Aplicadas

## Licencia

Este proyecto es para uso educativo y de investigación.

## Referencias

- MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/guide
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/
- OpenCV: https://opencv.org/
