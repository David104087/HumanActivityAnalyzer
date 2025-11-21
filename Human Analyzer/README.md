# Sistema de Reconocimiento de Actividad Humana en Tiempo Real

Sistema completo de clasificación de actividades humanas usando MediaPipe y Machine Learning. Detecta 5 acciones en tiempo real: **caminar-adelante**, **caminar-atras**, **girar**, **parase** y **sentarse**.

## Características Principales

- Detección de pose en tiempo real con MediaPipe
- Extracción de 8 ángulos biomecánicos normalizados
- Ventana temporal de 15 frames para estabilidad
- 48 características engineered (posición + velocidad)
- Modelos entrenados: Random Forest y XGBoost
- Pipeline con PCA (95% varianza)
- Sistema de suavizado de predicciones
- Interfaz visual con confianza en tiempo real
- Gate de quietud para evitar predicciones erráticas

## Estructura del Proyecto

```
proyecto/
│
├── data/
│   ├── raw/
│   │   └── videos/              # Videos originales organizados por acción
│   │       ├── caminar-adelante/
│   │       ├── caminar-atras/
│   │       ├── girar/
│   │       ├── parase/
│   │       └── sentarse/
│   │
│   └── processed/               # CSVs generados automáticamente
│       ├── datos_analisis.csv   # Metadatos de frames
│       ├── datosmediapipe.csv   # Landmarks 3D crudos
│       └── model_features.csv   # 48 características temporales
│
├── models/                      # Modelos entrenados
│   ├── best_random_forest_model.joblib
│   ├── best_xgboost_model.joblib
│   └── label_encoder.joblib
│
├── src/
│   ├── get_landmarks.py         # Extracción de landmarks con MediaPipe
│   ├── feature_engineering.py   # Cálculo de características temporales
│   ├── model_training.py        # Entrenamiento de modelos
│   └── main.py                  # Inferencia en tiempo real
│
├── requirements.txt
└── README.md
```

## Requisitos del Sistema

- Python 3.10+
- Webcam funcional
- Sistema operativo: Windows, macOS o Linux

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd Human\ Analyzer
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Preparación de Datos

### 1. Organizar videos

Coloca tus videos en la estructura correcta:

```
data/raw/videos/
├── caminar-adelante/
│   ├── video1.mp4
│   └── video2.mp4
├── caminar-atras/
│   ├── video1.mp4
│   └── video2.mp4
├── girar/
│   ├── video1.mp4
│   └── video2.mp4
├── parase/
│   ├── video1.mp4
│   └── video2.mp4
└── sentarse/
    ├── video1.mp4
    └── video2.mp4
```

**Importante:** Los nombres de las carpetas deben ser exactamente: `caminar-adelante`, `caminar-atras`, `girar`, `parase`, `sentarse`

### 2. Formatos de video soportados

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)
- M4V (.m4v)

## Flujo de Ejecución

### Paso 1: Extraer Landmarks

Procesa los videos y extrae landmarks 3D de MediaPipe:

```bash
python src/get_landmarks.py --stride 1
```

**Parámetros opcionales:**
- `--stride N`: Procesar 1 de cada N frames (default: 1)
- `--min_detection_confidence X`: Umbral de detección (default: 0.5)
- `--min_tracking_confidence X`: Umbral de tracking (default: 0.5)
- `--static_image_mode`: Modo de imagen estática

**Salidas:**
- `data/processed/datosmediapipe.csv`: 33 landmarks (x, y, z, visibility) por frame
- `data/processed/datos_analisis.csv`: Metadatos de video (fps, resolución, luminancia, etc.)

### Paso 2: Generar Características

Calcula características temporales desde los landmarks:

```bash
python src/feature_engineering.py
```

**Proceso:**
1. Normaliza landmarks por centro de cadera
2. Escala por distancia entre hombros
3. Calcula 8 ángulos biomecánicos:
   - Codo izquierdo y derecho
   - Hombro izquierdo y derecho
   - Cadera izquierda y derecha
   - Rodilla izquierda y derecha
4. Aplica ventana temporal de 15 frames
5. Genera 48 características:
   - Posición: mean, std, min, max (32 features)
   - Velocidad: mean, std (16 features)

**Salida:**
- `data/processed/model_features.csv`: Dataset listo para entrenar

### Paso 3: Entrenar Modelos

Entrena Random Forest y XGBoost con GridSearchCV:

```bash
python src/model_training.py
```

**Configuración:**
- Split: 70% train, 30% test (estratificado)
- Pipeline: StandardScaler → PCA(0.95) → Clasificador
- Random Forest GridSearch:
  - `n_estimators`: [100, 200]
  - `max_depth`: [10, 20, None]
  - `min_samples_split`: [2, 5]
- XGBoost GridSearch:
  - `n_estimators`: [100, 200]
  - `max_depth`: [5, 10]
  - `learning_rate`: [0.01, 0.1]

**Salidas:**
- `models/best_random_forest_model.joblib`
- `models/best_xgboost_model.joblib`
- `models/label_encoder.joblib`
- Matrices de confusión (PNG)
- Classification reports en consola

### Paso 4: Ejecutar Sistema en Tiempo Real

Inicia la detección en vivo desde la webcam:

```bash
python src/main.py
```

**Controles:**
- Presiona `q` para salir

## Arquitectura del Sistema en Tiempo Real

### Pipeline de Procesamiento

```
Frame → MediaPipe → Landmarks → Normalización → Ángulos (8)
                                                      ↓
                                            Búfer (15 frames)
                                                      ↓
                                          Características (48)
                                                      ↓
                                              Gate de Quietud
                                                   /    \
                                                  /      \
                                      suma_std < 40    suma_std >= 40
                                            /              \
                                           /                \
                                    "quieto" (98%)      Modelo Random Forest
                                                              ↓
                                                        Predicción
                                                              ↓
                                                    Historial (8 frames)
                                                              ↓
                                                    Votación Mayoritaria
                                                              ↓
                                                      Filtro Lerp (suavizado)
                                                              ↓
                                                        Visualización
```

### Componentes Clave

#### 1. Búfer de Ángulos (`angle_deque`)
- Tamaño: 15 frames
- Almacena los últimos 8 ángulos por frame
- Permite cálculo de características temporales

#### 2. Gate de Quietud
- Umbral: `suma(std_angulos) < 40.0`
- Detecta cuando la persona está inmóvil
- Fuerza predicción "quieto" con 98% confianza
- Evita predicciones erráticas en reposo

#### 3. Historial de Predicciones (`prediction_history`)
- Tamaño: 8 frames
- Almacena tuplas (acción, confianza)
- Suavizado por votación mayoritaria

#### 4. Filtro Lerp
- `displayed_confidence = 0.8 * displayed_confidence + 0.2 * avg_confidence`
- Transiciones suaves entre predicciones
- Evita parpadeo en la interfaz

### Visualización

**Barra Superior (UI):**
- Fondo oscuro (30, 30, 30)
- Texto "ACCION:" y acción detectada en MAYÚSCULAS
- Texto "CONF:" y porcentaje de confianza
- Barra de progreso proporcional a la confianza

**Código de Colores:**
- Verde (0, 255, 0): Confianza > 80%
- Amarillo (0, 255, 255): Confianza 50-80%
- Rojo (0, 0, 255): Confianza < 50%

**Esqueleto:**
- Dibujado con MediaPipe drawing utilities
- Conexiones estándar de pose (33 landmarks)

## Especificaciones Técnicas

### Ángulos Biomecánicos

Los 8 ángulos calculados son:

1. **Codo Izquierdo**: Hombro izq → Codo izq → Muñeca izq
2. **Codo Derecho**: Hombro der → Codo der → Muñeca der
3. **Hombro Izquierdo**: Codo izq → Hombro izq → Cadera izq
4. **Hombro Derecho**: Codo der → Hombro der → Cadera der
5. **Cadera Izquierda**: Hombro izq → Cadera izq → Rodilla izq
6. **Cadera Derecha**: Hombro der → Cadera der → Rodilla der
7. **Rodilla Izquierda**: Cadera izq → Rodilla izq → Tobillo izq
8. **Rodilla Derecha**: Cadera der → Rodilla der → Tobillo der

### Normalización

1. **Traslación**: Restar centro de cadera (promedio de landmarks 23 y 24)
2. **Escalado**: Dividir por distancia euclidiana entre hombros (landmarks 11 y 12)
3. **Robustez**: Verificar `dist_hombros > 1e-6` para evitar división por cero

### Características (48 totales)

| Grupo | Características | Cantidad |
|-------|-----------------|----------|
| Posición Mean | Media de 8 ángulos en 15 frames | 8 |
| Posición Std | Desv. estándar de 8 ángulos | 8 |
| Posición Min | Mínimo de 8 ángulos | 8 |
| Posición Max | Máximo de 8 ángulos | 8 |
| Velocidad Mean | Media de velocidad angular | 8 |
| Velocidad Std | Desv. estándar de velocidad | 8 |
| **TOTAL** | | **48** |

## Optimizaciones y Mejoras

### vs. Proyecto Original

1. **Normalización Robusta**
   - Verificación de división por cero
   - Manejo de casos extremos en distancia entre hombros

2. **Gate de Quietud**
   - Evita predicciones erráticas cuando la persona está inmóvil
   - Umbral calibrado: `suma(std) < 40.0`

3. **Suavizado Adaptativo**
   - Ventana de 8 frames para votación mayoritaria
   - Filtro Lerp con factor 0.2 para transiciones suaves

4. **Gestión de Errores**
   - Limpieza automática de búferes si no se detecta cuerpo
   - Estado "BUSCANDO..." visible
   - Validación de landmarks antes de calcular ángulos

5. **Interfaz Mejorada**
   - Colores dinámicos según confianza
   - Barra de progreso visual
   - Información clara y legible

## Resolución de Problemas

### Error: "No se encuentra el archivo model_features.csv"

**Solución:** Ejecuta primero los pasos 1 y 2:
```bash
python src/get_landmarks.py --stride 1
python src/feature_engineering.py
```

### Error: "No se puede acceder a la cámara"

**Soluciones:**
1. Verifica que la webcam esté conectada
2. Cierra otras aplicaciones que usen la cámara
3. En Windows, verifica permisos de cámara en Configuración
4. Prueba cambiar `cv2.VideoCapture(0)` a `cv2.VideoCapture(1)` en `main.py`

### Advertencia: "No hay carpetas de acciones"

**Solución:** Verifica la estructura de carpetas:
```
data/raw/videos/
├── caminar-adelante/
├── caminar-atras/
├── girar/
├── parase/
└── sentarse/
```

### Modelo con baja precisión (<90%)

**Soluciones:**
1. Agrega más videos de entrenamiento
2. Asegúrate de que los videos tengan buena iluminación
3. Verifica que las acciones sean claramente diferenciables
4. Aumenta `--stride` en get_landmarks.py para más frames
5. Ajusta hiperparámetros en `model_training.py`

### Predicciones erráticas en tiempo real

**Soluciones:**
1. Aumenta `HISTORIAL_PREDICCION` en `main.py` (default: 8)
2. Ajusta `UMBRAL_QUIETUD` (default: 40.0)
3. Modifica factor Lerp (default: 0.2 → 0.1 para más suavizado)
4. Mejora iluminación del entorno

## Personalización

### Cambiar Ventana Temporal

En `feature_engineering.py` y `main.py`:
```python
VENTANA_TEMPORAL = 20  # Cambiar de 15 a 20
```

### Agregar Nueva Acción

1. Crear carpeta en `data/raw/videos/nueva-accion/`
2. Colocar videos de la nueva acción
3. Re-ejecutar todo el pipeline (pasos 1-4)

### Usar Modelo XGBoost en Tiempo Real

En `main.py`, línea 16:
```python
MODEL_PATH = Path("./models/best_xgboost_model.joblib")
```

### Ajustar Confianza de MediaPipe

En `main.py`:
```python
MIN_DETECTION_CONFIDENCE = 0.7  # Más estricto
MIN_TRACKING_CONFIDENCE = 0.7   # Más estricto
```

## Performance

### Métricas Esperadas

- **Accuracy**: >90% en dataset de test
- **FPS en tiempo real**: 15-30 (depende del hardware)
- **Latencia**: ~50-100ms por predicción

### Requerimientos de Hardware

**Mínimos:**
- CPU: Intel i5 o equivalente
- RAM: 4GB
- Webcam: 720p

**Recomendados:**
- CPU: Intel i7 o equivalente
- RAM: 8GB+
- Webcam: 1080p

## Créditos y Referencias

- **MediaPipe**: Google AI (https://mediapipe.dev/)
- **Scikit-learn**: Machine Learning en Python
- **XGBoost**: Gradient Boosting optimizado
- **OpenCV**: Computer Vision

## Licencia

Este proyecto es para uso educativo y de investigación.

## Soporte

Para reportar bugs o sugerencias, abre un issue en el repositorio del proyecto.

---

Desarrollado como parte del curso de Inteligencia Artificial, Semestre 7.
