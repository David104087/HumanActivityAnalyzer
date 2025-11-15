# 🚀 README-STARTER - Guía Rápida del Proyecto

## 📋 Resumen del Proyecto

Sistema de **clasificación de actividades humanas en tiempo real** que usa:
- **MediaPipe Pose** para detectar landmarks corporales (33 puntos)
- **Ventanas Deslizantes (Sliding Windows)** para contexto temporal
- **Machine Learning** (Random Forest, SVM, XGBoost) para clasificar actividades

---

## � Glosario: ¿Qué es cada cosa y por qué existe?

### �🗂️ Carpetas Principales

#### `1_data_extraction/`
**Rol:** Punto de entrada del pipeline. Extrae información bruta de los videos.  
**Por qué existe:** Los videos MP4 no se pueden usar directamente en ML. Necesitamos convertirlos a coordenadas numéricas (landmarks).  
**Qué contiene:**
- `01_extract_landmarks.py`: Lee videos frame por frame, detecta la pose humana con MediaPipe y guarda las coordenadas (x, y) de 33 puntos del cuerpo.

#### `2_feature_engineering/`
**Rol:** Transforma landmarks en características significativas para ML.  
**Por qué existe:** Las coordenadas crudas (x, y) no son buenas features porque dependen de la posición de la cámara. Necesitamos medidas invariantes como ángulos y movimiento.  
**Qué contiene:**
- `02_compute_features.py`: Convierte 66 coordenadas → 6 features interpretables (ángulos de articulaciones, inclinación, movimiento)
- `03_create_labels_csv.py`: Herramienta auxiliar para convertir anotaciones manuales a formato estándar
- `04_create_window_dataset.py`: ⭐ **Implementa sliding windows**. Combina features de múltiples frames para dar contexto temporal

#### `3_model_training/`
**Rol:** Prepara datos finales y entrena modelos de clasificación.  
**Por qué existe:** Los datos crudos necesitan normalización y división. Luego entrenamos varios algoritmos para encontrar el mejor.  
**Qué contiene:**
- `05_preprocess_train_split.py`: Normaliza features (StandardScaler), divide train/test, balancea clases
- `06_train_models.py`: Entrena Random Forest, SVM y XGBoost; compara sus resultados

#### `4_real_time_app/`
**Rol:** Aplicación de producción que usa los modelos entrenados.  
**Por qué existe:** El objetivo final es clasificar actividades EN VIVO desde una webcam.  
**Qué contiene:**
- `run_realtime.py`: App principal que captura video, calcula features, aplica sliding windows y predice
- `utils.py`: Funciones reutilizables para calcular ángulos y movimiento (compartidas con entrenamiento)

#### `data/`
**Rol:** Almacén centralizado de todos los datos del pipeline.  
**Por qué existe:** Separar datos de código mantiene el proyecto organizado y facilita el versionado.  
**Subcarpetas:**
- `raw_videos/`: Videos originales MP4 (opcional si ya tienes landmarks)
- `preprocessed/`: Landmarks extraídos (66 columnas por frame) - Salida del paso 1
- `features_per_frame/`: Features calculados (6 columnas por frame) - Salida del paso 2
- `labels/`: Anotaciones manuales (qué actividad ocurre en cada frame)
- `processed_windowed/`: Dataset final con ventanas (30 columnas) + scaler y encoder - Salida de pasos 4-5

#### `assets/`
**Rol:** Modelos y archivos necesarios para la app en tiempo real.  
**Por qué existe:** Separar los modelos de producción de los experimentales. Solo copiamos aquí el mejor modelo.  
**Qué contiene:**
- `randomforest.pkl`: Modelo entrenado (copiado desde `results/models/`)
- `scaler.pkl`: StandardScaler entrenado con 30 features (⚠️ CRÍTICO: debe ser el mismo del entrenamiento)
- `label_encoder.pkl`: Mapeo entre números (0,1,2...) y nombres de actividades ("Walk", "Sit"...)

#### `results/`
**Rol:** Almacena todos los modelos entrenados y sus métricas.  
**Por qué existe:** Permite comparar múltiples experimentos sin sobrescribir resultados previos.  
**Subcarpetas:**
- `models/`: Todos los modelos entrenados (.pkl)
- `metrics/`: CSVs con accuracy, precision, recall, F1-score

---

### 📄 Archivos en la Raíz

#### `run_full_pipeline.py`
**Rol:** Script maestro que ejecuta TODO el pipeline automáticamente.  
**Por qué existe:** En lugar de ejecutar 6 scripts manualmente, este automatiza todo el proceso.  
**Cuándo usarlo:** Cuando tienes nuevos datos y quieres re-entrenar desde cero.  
**Qué hace:**
1. Calcula features (paso 2)
2. Crea dataset con ventanas (paso 4)
3. Preprocesa y divide datos (paso 5)
4. Entrena modelos (paso 6)
5. Copia el mejor modelo a `assets/`

#### `run_realtime.py` (también en `4_real_time_app/`)
**Rol:** Aplicación final para clasificación en tiempo real.  
**Por qué existe:** Es el producto entregable del proyecto.  
**Cuándo usarlo:** Después de entrenar, para demostrar el sistema funcionando.  
**Cómo funciona:**
- Abre webcam
- Detecta pose frame por frame
- Mantiene cola de 5 frames (sliding window)
- Predice actividad cada frame
- Muestra resultado en pantalla

#### `requirements.txt`
**Rol:** Lista de todas las dependencias de Python.  
**Por qué existe:** Permite replicar el entorno exacto en cualquier máquina.  
**Cuándo usarlo:** Primera vez que configuras el proyecto (`pip install -r requirements.txt`)

#### `utils.py` y `utils_check.py`
**Rol:** Funciones auxiliares reutilizables.  
**Por qué existen:** Evitar duplicar código entre entrenamiento y tiempo real.  
**Qué contienen:**
- `utils.py`: Cálculo de ángulos (`calculate_angle`), inclinación del tronco, energía de movimiento
- `utils_check.py`: Verificaciones de sanidad (revisar formato de CSVs, etc.)

#### `README.md`
**Rol:** Documentación completa y detallada del proyecto.  
**Por qué existe:** Guía técnica exhaustiva para usuarios avanzados.  
**Diferencia con README-STARTER.md:** README.md es más técnico; README-STARTER.md es más didáctico y visual.

---

### 📊 Archivos Clave de Datos

#### `*_preprocessed.csv` (en `data/preprocessed/`)
**Formato:** `video, frame, nx_0, nx_1, ..., nx_32, ny_0, ny_1, ..., ny_32`  
**Rol:** Landmarks crudos extraídos de MediaPipe.  
**Por qué este formato:** MediaPipe detecta 33 puntos (0=nariz, 11=hombro izq, 23=cadera izq, etc.). Cada punto tiene coordenadas normalizadas (x, y) entre 0 y 1.  
**Columnas:** 2 (video, frame) + 66 (33 puntos × 2 coords) = 68 columnas

#### `*_features.csv` (en `data/features_per_frame/`)
**Formato:** `video_name, frame, knee_left, knee_right, hip_left, hip_right, trunk_angle, motion_energy`  
**Rol:** Features calculados por frame (características geométricas).  
**Por qué este formato:** Reducimos 66 números a 6 features significativas e invariantes a la posición de la cámara.  
**Columnas:** 2 (identificadores) + 6 (features) = 8 columnas

#### `processed_labels.csv` (en `data/labels/`)
**Formato:** `video_name, frame, label`  
**Rol:** Anotaciones manuales de qué actividad ocurre en cada frame.  
**Por qué este formato:** Necesitamos supervisión (ground truth) para entrenar modelos.  
**Ejemplo:**
```csv
video1,0,Walk
video1,1,Walk
video1,50,Sit
video1,51,Sit
```

#### `windowed_dataset.csv` (en `data/processed_windowed/`)
**Formato:** `f_0_knee_left, f_0_knee_right, ..., f_4_motion_energy, label`  
**Rol:** Dataset final para entrenar modelos (con contexto temporal).  
**Por qué este formato:** Cada fila representa una ventana de 5 frames (30 features) + su etiqueta.  
**Columnas:** 30 (5 frames × 6 features) + 1 (label) = 31 columnas  
**Por qué 30 features:** El modelo necesita ver "historia" para distinguir actividades temporales.

#### `scaler.pkl`
**Rol:** Objeto StandardScaler entrenado que normaliza las 30 features.  
**Por qué existe:** Los modelos de ML funcionan mejor cuando todas las features tienen media=0 y desviación estándar=1.  
**⚠️ CRÍTICO:** El scaler de entrenamiento DEBE ser el mismo usado en tiempo real. Si no, las predicciones serán incorrectas.  
**Cómo se usa:**
```python
# Entrenamiento
scaler.fit(X_train)  # Aprende media y std de train
X_train_scaled = scaler.transform(X_train)

# Tiempo real (usar el MISMO scaler)
X_new_scaled = scaler.transform(X_new)  # Usa media y std aprendidas
```

#### `label_encoder.pkl`
**Rol:** Mapeo bidireccional entre nombres de actividades y números.  
**Por qué existe:** Los modelos trabajan con números (0, 1, 2...), pero los humanos necesitamos nombres ("Walk", "Sit"...).  
**Cómo funciona:**
```python
# Entrenamiento: texto → número
["Walk", "Sit", "Walk"] → [0, 1, 0]

# Tiempo real: número → texto
model.predict([...]) → [0] → label_encoder.inverse_transform([0]) → ["Walk"]
```

---

### 🎯 Archivos de Modelos

#### `randomforest.pkl`, `svm.pkl`, `xgboost.pkl`
**Rol:** Modelos entrenados listos para hacer predicciones.  
**Por qué varios:** Comparamos múltiples algoritmos y elegimos el mejor (usualmente Random Forest).  
**Diferencias:**
- **Random Forest**: Rápido, robusto, interpretable. Ideal para este proyecto.
- **SVM**: Preciso pero lento de entrenar. Bueno para datasets pequeños.
- **XGBoost**: Muy preciso pero complejo. Mejor para competencias.

---

## 🗂️ Estructura de Carpetas (Vista Rápida)

```
Entrega 3/
│
├── 1_data_extraction/          # Paso 1: Extracción de landmarks
│   └── 01_extract_landmarks.py
│
├── 2_feature_engineering/      # Paso 2-4: Features + Ventanas
│   ├── 02_compute_features.py
│   ├── 03_create_labels_csv.py
│   └── 04_create_window_dataset.py  ⭐ SLIDING WINDOWS
│
├── 3_model_training/           # Paso 5-6: Preprocesado + Entrenamiento
│   ├── 05_preprocess_train_split.py
│   └── 06_train_models.py
│
├── 4_real_time_app/            # App en tiempo real
│   ├── run_realtime.py         ⭐ SLIDING WINDOWS en producción
│   └── utils.py
│
├── data/                       # Datos del pipeline
│   ├── raw_videos/            # Videos originales (opcional)
│   ├── preprocessed/          # Landmarks extraídos (CSV)
│   ├── features_per_frame/    # Features calculados por frame
│   ├── labels/                # Etiquetas manuales
│   └── processed_windowed/    # Dataset final con ventanas ⭐
│
├── assets/                     # Modelos para producción
│   ├── randomforest.pkl       # Modelo entrenado
│   ├── scaler.pkl             # Escalador (30 features)
│   └── label_encoder.pkl      # Codificador de labels
│
├── results/                    # Resultados del entrenamiento
│   ├── models/                # Modelos entrenados
│   └── metrics/               # Métricas y reportes
│
├── run_full_pipeline.py        # ⚡ Script maestro (ejecuta todo)
└── requirements.txt
```

---

## 🔢 Orden de Ejecución de Scripts

### Pipeline Completo (Automático)
```bash
python run_full_pipeline.py
```

### Pipeline Manual (Paso a Paso)
```bash
# Paso 1: Extraer landmarks (33 puntos x,y por frame)
python 1_data_extraction/01_extract_landmarks.py

# Paso 2: Calcular 6 features por frame
python 2_feature_engineering/02_compute_features.py --batch

# Paso 3: (Opcional) Crear archivo de labels
python 2_feature_engineering/03_create_labels_csv.py

# Paso 4: Crear dataset con ventanas deslizantes ⭐
python 2_feature_engineering/04_create_window_dataset.py

# Paso 5: Escalar y dividir train/test
python 3_model_training/05_preprocess_train_split.py

# Paso 6: Entrenar modelos
python 3_model_training/06_train_models.py

# Paso 7: Ejecutar aplicación en tiempo real
python 4_real_time_app/run_realtime.py
```

---

## ⭐ Concepto Clave: SLIDING WINDOWS (Ventanas Deslizantes)

### ¿Por qué usar ventanas deslizantes?

Una actividad humana **NO** se puede clasificar con un solo frame. Necesitamos **contexto temporal**:
- **Caminar**: requiere ver movimiento de piernas en varios frames
- **Sentarse**: es una transición gradual, no instantánea
- **Estar de pie**: necesita confirmar que no hay movimiento significativo

### 🎯 Estrategia Implementada

#### 1️⃣ Features por Frame (6 features)
Cada frame individual tiene estas características:

| Feature | Descripción | Rango |
|---------|-------------|-------|
| `knee_left` | Ángulo rodilla izquierda (cadera-rodilla-tobillo) | 0°-180° |
| `knee_right` | Ángulo rodilla derecha | 0°-180° |
| `hip_left` | Ángulo cadera izquierda (hombro-cadera-rodilla) | 0°-180° |
| `hip_right` | Ángulo cadera derecha | 0°-180° |
| `trunk_angle` | Inclinación del tronco (vertical-hombros-cadera) | 0°-180° |
| `motion_energy` | Energía de movimiento vs frame anterior | 0.0-1.0 |

#### 2️⃣ Ventana Deslizante (WINDOW_SIZE = 5)

En lugar de clasificar con 6 features, usamos **5 frames × 6 features = 30 features**:

```python
# Ejemplo visual de ventana deslizante:

Frame 0: [knee_L=150°, knee_R=145°, hip_L=170°, hip_R=168°, trunk=85°, motion=0.02]
Frame 1: [knee_L=148°, knee_R=143°, hip_L=169°, hip_R=167°, trunk=84°, motion=0.05]
Frame 2: [knee_L=146°, knee_R=141°, hip_L=168°, hip_R=166°, trunk=83°, motion=0.08]
Frame 3: [knee_L=144°, knee_R=139°, hip_L=167°, hip_R=165°, trunk=82°, motion=0.12]
Frame 4: [knee_L=142°, knee_R=137°, hip_L=166°, hip_R=164°, trunk=81°, motion=0.15]
         ↑
         Clasificación: "Walk" (etiqueta del frame 4)

# La ventana se "desliza" un frame hacia adelante:

Frame 1: [knee_L=148°, knee_R=143°, hip_L=169°, hip_R=167°, trunk=84°, motion=0.05]
Frame 2: [knee_L=146°, knee_R=141°, hip_L=168°, hip_R=166°, trunk=83°, motion=0.08]
Frame 3: [knee_L=144°, knee_R=139°, hip_L=167°, hip_R=165°, trunk=82°, motion=0.12]
Frame 4: [knee_L=142°, knee_R=137°, hip_L=166°, hip_R=164°, trunk=81°, motion=0.15]
Frame 5: [knee_L=140°, knee_R=135°, hip_L=165°, hip_R=163°, trunk=80°, motion=0.18]
         ↑
         Clasificación: "Walk" (etiqueta del frame 5)
```

#### 3️⃣ Formato del Dataset Final

**Archivo:** `data/processed_windowed/windowed_dataset.csv`

**Estructura:**
```
f_0_knee_left, f_0_knee_right, ..., f_0_motion_energy,  ← Frame más antiguo (t-4)
f_1_knee_left, f_1_knee_right, ..., f_1_motion_energy,  ← Frame t-3
f_2_knee_left, f_2_knee_right, ..., f_2_motion_energy,  ← Frame t-2
f_3_knee_left, f_3_knee_right, ..., f_3_motion_energy,  ← Frame t-1
f_4_knee_left, f_4_knee_right, ..., f_4_motion_energy,  ← Frame actual (t)
label                                                     ← Etiqueta del frame actual
```

**Total:** 30 columnas de features + 1 columna de label = 31 columnas

---

## 📊 Flujo de Datos Completo

```
┌─────────────────┐
│  Video (MP4)    │
│  30 fps, 2 min  │
└────────┬────────┘
         │ 01_extract_landmarks.py
         ↓
┌─────────────────────────────────┐
│  Landmarks CSV                  │
│  video, frame, nx_0...nx_32,   │
│                ny_0...ny_32     │
│  (33 puntos × 2 coords = 66)   │
└────────┬────────────────────────┘
         │ 02_compute_features.py
         ↓
┌─────────────────────────────────┐
│  Features CSV (por frame)       │
│  video, frame,                  │
│  knee_L, knee_R, hip_L, hip_R, │
│  trunk_angle, motion_energy     │
│  (6 features)                   │
└────────┬────────────────────────┘
         │ + labels (manual)
         │ 04_create_window_dataset.py ⭐
         ↓
┌─────────────────────────────────┐
│  Windowed Dataset               │
│  f_0_knee_left, ..., f_4_motion │
│  (5 frames × 6 feat = 30 feat)  │
│  + label                        │
└────────┬────────────────────────┘
         │ 05_preprocess_train_split.py
         ↓
┌─────────────────────────────────┐
│  Train/Test Escalados           │
│  + scaler.pkl (30 features)     │
│  + label_encoder.pkl            │
└────────┬────────────────────────┘
         │ 06_train_models.py
         ↓
┌─────────────────────────────────┐
│  Modelos Entrenados             │
│  randomforest.pkl               │
│  svm.pkl                        │
│  xgboost.pkl                    │
└────────┬────────────────────────┘
         │ Copy to assets/
         ↓
┌─────────────────────────────────┐
│  Producción (Tiempo Real)       │
│  run_realtime.py                │
│  - Detecta pose (MediaPipe)     │
│  - Calcula 6 features           │
│  - Guarda últimos 5 frames      │
│  - Aplana ventana (30 feat)     │
│  - Escala con scaler.pkl        │
│  - Predice con modelo           │
└─────────────────────────────────┘
```

---

## 🎬 Implementación de Sliding Windows

### En Entrenamiento (04_create_window_dataset.py)

```python
WINDOW_SIZE = 5  # Número de frames de contexto

# Para cada video (NO mezclar videos diferentes):
for video in videos:
    features = video.features  # Array (N_frames, 6)
    labels = video.labels      # Array (N_frames,)
    
    # Iterar desde frame 4 (índice donde ya hay 5 frames de historia)
    for i in range(WINDOW_SIZE - 1, len(features)):
        # Extraer ventana: frames [i-4, i-3, i-2, i-1, i]
        window = features[i - (WINDOW_SIZE - 1) : i + 1]
        
        # Aplanar: (5, 6) → (30,)
        window_flat = window.flatten()
        
        # La etiqueta es la del frame ACTUAL (i)
        label = labels[i]
        
        # Guardar muestra: [f0_feat0, f0_feat1, ..., f4_feat5, label]
        dataset.append(window_flat + [label])
```

**Resultado:**
- Frame 0-3: Descartados (no hay suficiente historia)
- Frame 4: Primera muestra (historia completa)
- Frame N: Última muestra

### En Tiempo Real (run_realtime.py)

```python
from collections import deque

WINDOW_SIZE = 5
features_buffer = deque(maxlen=WINDOW_SIZE)  # Cola FIFO

while True:
    frame = camera.read()
    landmarks = mediapipe.detect(frame)
    
    # Calcular 6 features del frame actual
    current_features = calculate_features(landmarks)
    
    # Agregar a buffer (automáticamente elimina el más viejo si lleno)
    features_buffer.append(current_features)
    
    # Solo predecir cuando tenemos 5 frames completos
    if len(features_buffer) == WINDOW_SIZE:
        # Aplanar ventana: (5, 6) → (30,)
        window_flat = np.array(features_buffer).flatten()
        
        # Crear DataFrame para mantener nombres de features
        window_df = pd.DataFrame([window_flat], columns=FEATURE_COLUMNS)
        
        # Escalar
        window_scaled = scaler.transform(window_df)
        
        # Predecir
        prediction = model.predict(window_scaled)
        
        show_on_screen(prediction)
    else:
        show_on_screen("Cargando contexto...")
```

---

## 🔧 Scripts Detallados

### 01_extract_landmarks.py
**Entrada:** Videos MP4 en `data/raw_videos/`  
**Salida:** CSVs en `data/preprocessed/` con 66 columnas (33 landmarks × 2 coords)  
**Qué hace:** Usa MediaPipe Pose para detectar 33 puntos del cuerpo por frame

---

### 02_compute_features.py
**Entrada:** CSVs de landmarks (`data/preprocessed/*_preprocessed.csv`)  
**Salida:** CSVs de features (`data/features_per_frame/*_features.csv`)  
**Qué hace:**
- Calcula ángulos de rodillas y caderas
- Calcula inclinación del tronco
- Calcula energía de movimiento (diferencia entre frames)
- **Resultado:** 6 features por frame

---

### 03_create_labels_csv.py
**Entrada:** Export de LabelStudio o creación manual  
**Salida:** `data/labels/processed/processed_labels.csv`  
**Formato:** `video_name, frame, label`  
**Qué hace:** Convierte anotaciones manuales a formato frame-by-frame

---

### 04_create_window_dataset.py ⭐
**Entrada:**
- Features: `data/features_per_frame/*_features.csv`
- Labels: `data/labels/processed/processed_labels.csv`

**Salida:** `data/processed_windowed/windowed_dataset.csv`

**Qué hace:**
1. Carga todos los CSVs de features
2. Une features con labels por `(video_name, frame)`
3. Ordena por video y frame
4. **Aplica ventana deslizante:**
   - Por cada video (separadamente)
   - Crea ventanas de 5 frames consecutivos
   - Aplana cada ventana: (5, 6) → (30,)
   - Asigna la etiqueta del frame actual
5. Concatena todas las ventanas en un solo dataset

**Parámetros clave:**
```python
WINDOW_SIZE = 5  # Modificar aquí para cambiar contexto temporal
```

---

### 05_preprocess_train_split.py
**Entrada:** `data/processed_windowed/windowed_dataset.csv`  
**Salida:**
- `data/processed_windowed/train_dataset.csv`
- `data/processed_windowed/test_dataset.csv`
- `data/processed_windowed/scaler.pkl` ⭐ (entrena con 30 features)
- `data/processed_windowed/label_encoder.pkl`

**Qué hace:**
1. Separa features (30 cols) y labels (1 col)
2. Split train/test (80/20, estratificado)
3. **Balancea clases** con SMOTE (solo train)
4. **Entrena StandardScaler** con train (30 features)
5. Escala train y test con el mismo scaler
6. Codifica labels (texto → números)

---

### 06_train_models.py
**Entrada:**
- `data/processed_windowed/train_dataset.csv`
- `data/processed_windowed/test_dataset.csv`

**Salida:**
- `results/models/randomforest.pkl`
- `results/models/svm.pkl`
- `results/models/xgboost.pkl`
- `results/metrics/model_comparison.csv`

**Qué hace:**
1. Entrena 3 modelos con validación cruzada (5-fold)
2. Evalúa en conjunto de test
3. Guarda métricas y reportes

---

### run_realtime.py ⭐
**Entrada:**
- Webcam en vivo
- `assets/randomforest.pkl`
- `assets/scaler.pkl` (30 features)
- `assets/label_encoder.pkl`

**Salida:** Predicción en tiempo real en pantalla

**Qué hace:**
1. Captura frames de webcam
2. Detecta pose con MediaPipe
3. Calcula 6 features por frame
4. **Mantiene buffer de 5 frames** (deque)
5. Cuando buffer está lleno:
   - Aplana ventana (30 features)
   - Escala con scaler
   - Predice con modelo
   - Muestra actividad en pantalla

---

## ⚙️ Parámetros Configurables

### WINDOW_SIZE (Tamaño de Ventana)

**Ubicaciones:**
- `04_create_window_dataset.py` línea ~15
- `run_realtime.py` línea ~28

**Valores comunes:**
- `WINDOW_SIZE = 3`: Menos contexto, más rápido, menos preciso
- `WINDOW_SIZE = 5`: ✅ **Recomendado** (balance)
- `WINDOW_SIZE = 7`: Más contexto, más preciso, menos muestras
- `WINDOW_SIZE = 10`: Mucho contexto, lento en tiempo real

**⚠️ IMPORTANTE:** Si cambias WINDOW_SIZE, debes:
1. Volver a ejecutar `04_create_window_dataset.py`
2. Volver a ejecutar `05_preprocess_train_split.py` (nuevo scaler con N×6 features)
3. Volver a ejecutar `06_train_models.py`
4. Actualizar `run_realtime.py` con el mismo valor

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar repositorio
cd "Entrega 3"

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar datos:
#    - Landmarks CSV en: data/preprocessed/
#    - Labels CSV en: data/labels/processed/processed_labels.csv

# 4. Ejecutar pipeline completo
python run_full_pipeline.py

# 5. Verificar resultados
cat results/metrics/model_comparison.csv

# 6. Ejecutar tiempo real
python 4_real_time_app/run_realtime.py
```

---

## 🎓 Conceptos Clave para Nuevos Colaboradores

### 1. ¿Por qué 30 features?
```
6 features/frame × 5 frames = 30 features totales
```

### 2. ¿Por qué no mezclar videos en las ventanas?
```python
# ❌ MAL: Video1_frame99, Video2_frame0, ... → ventana inválida
# ✅ BIEN: Video1_frame4-8 → ventana válida
```
Cada video se procesa independientemente para evitar discontinuidades.

### 3. ¿Qué hace el scaler?
Normaliza las 30 features para que todas estén en la misma escala (media=0, std=1).
**Crucial:** El scaler de entrenamiento DEBE ser el mismo en tiempo real.

### 4. ¿Cómo funciona la cola (deque)?
```python
deque(maxlen=5)  # FIFO (First In, First Out)
[A, B, C, D, E]  # Lleno
append(F)
[B, C, D, E, F]  # A se eliminó automáticamente
```

---

## 📚 Recursos Adicionales

- **MediaPipe Pose:** https://google.github.io/mediapipe/solutions/pose
- **Sliding Windows en Time Series:** https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
- **Documentación scikit-learn:** https://scikit-learn.org/

---

**✅ ¡Listo para empezar!**
