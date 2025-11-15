# Entrega 3 - Sistema de Clasificación de Actividades Humanas en Tiempo Real

Este proyecto implementa un sistema completo de clasificación de actividades humanas usando MediaPipe Pose y Machine Learning, con soporte para ventanas deslizantes (temporal context).

## 📋 Tabla de Contenidos

- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Preparación de Datos](#preparación-de-datos)
- [Pipeline Completo](#pipeline-completo)
- [Uso Individual de Scripts](#uso-individual-de-scripts)
- [Ejecución en Tiempo Real](#ejecución-en-tiempo-real)
- [Troubleshooting](#troubleshooting)

---

## 🗂 Estructura del Proyecto

```
Entrega 3/
├── 1_data_extraction/
│   └── 01_extract_landmarks.py          # Extrae landmarks con MediaPipe
│
├── 2_feature_engineering/
│   ├── 02_compute_features.py           # Calcula 6 features por frame
│   ├── 03_create_labels_csv.py          # Auxiliar para crear labels
│   └── 04_create_window_dataset.py      # Crea dataset con ventanas
│
├── 3_model_training/
│   ├── 05_preprocess_train_split.py     # Escala y hace split
│   └── 06_train_models.py               # Entrena modelos (RF, SVM, XGB)
│
├── 4_real_time_app/
│   └── (recursos opcionales)
│
├── assets/                               # Modelos entrenados para producción
│   ├── randomforest.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── data/
│   ├── preprocessed/                    # ← AQUÍ van tus landmarks CSV
│   ├── features_per_frame/              # Features calculados
│   ├── labels/                          # (opcional)
│   ├── labels_nuevos.csv               # ← AQUÍ va tu CSV de labels
│   └── processed_windowed/              # Datasets con ventanas
│
├── results/
│   ├── models/                          # Modelos entrenados
│   └── reports/                         # Métricas y reportes
│
├── run_realtime.py                      # 🎥 Aplicación en tiempo real
├── run_full_pipeline.py                 # 🚀 Script maestro (todo automático)
├── utils.py                             # Funciones de cálculo de features
└── requirements.txt
```

---

## 📦 Requisitos

### Instalación de dependencias

```bash
cd "Entrega 3"
pip install -r requirements.txt
```

### Paquetes principales:
- `opencv-python` - Procesamiento de video
- `mediapipe` - Detección de pose
- `pandas`, `numpy` - Manipulación de datos
- `scikit-learn` - Machine Learning
- `joblib` - Serialización de modelos
- `xgboost` (opcional) - Modelo adicional

---

## 📁 Preparación de Datos

### Paso 0: Organizar tus archivos

#### A) Landmarks (YA LOS TIENES)

Coloca tus archivos de landmarks en: **`data/preprocessed/`**

**Formato esperado:** Archivos CSV con el patrón `*_preprocessed.csv` o `*_landmarks.csv`

**Columnas requeridas:**
```
video, frame, nx_0, nx_1, ..., nx_32, ny_0, ny_1, ..., ny_32
```

**Ejemplo de organización:**
```bash
data/preprocessed/
├── video1_preprocessed.csv
├── video2_preprocessed.csv
├── video3_preprocessed.csv
├── ...
└── video20_preprocessed.csv
```

**⚠️ IMPORTANTE:** 
- Si tus archivos se llaman `*_landmarks.csv` en lugar de `*_preprocessed.csv`, renómbralos:
  ```bash
  cd data/preprocessed
  for f in *_landmarks.csv; do mv "$f" "${f/_landmarks.csv/_preprocessed.csv}"; done
  ```

#### B) Labels (Etiquetas)

Crea el archivo: **`data/labels_nuevos.csv`**

**Formato requerido:**
```csv
video,frame,label
video1,0,Walk to front
video1,1,Walk to front
video1,2,Walk to front
video1,150,Sit
video1,151,Sit
video2,0,Stand
...
```

**Etiquetas disponibles (según tu enunciado):**
1. Walk to front
2. Walk to back
3. Sit
4. Turn 180
5. Stand
6. Lean Right
7. Lean Left
8. Squat

**Cómo crear este archivo:**

**Opción 1: Si tienes export de LabelStudio**
```bash
python 2_feature_engineering/03_create_labels_csv.py \
  --labelstudio_json tu_export.json \
  --out data/labels_nuevos.csv
```

**Opción 2: Crear template y completar manualmente**
```bash
python 2_feature_engineering/03_create_labels_csv.py \
  --create_template \
  --out data/labels_template.csv

# Luego edita el archivo con tus labels reales
```

**Opción 3: Crear manualmente en Excel/LibreOffice**
- Columnas: `video`, `frame`, `label`
- Una fila por cada frame etiquetado
- Guarda como CSV en `data/labels_nuevos.csv`

---

## 🚀 Pipeline Completo

### Opción A: Script Maestro (Recomendado - TODO AUTOMÁTICO)

```bash
cd "Entrega 3"
python run_full_pipeline.py
```

Esto ejecuta automáticamente:
1. ✅ Calcula features por frame
2. ✅ Crea dataset con ventanas deslizantes (window_size=5)
3. ✅ Preprocesa y hace split train/test
4. ✅ Entrena modelos (RandomForest, SVM, XGBoost)
5. ✅ Copia el mejor modelo a `assets/`

**Opciones adicionales:**
```bash
# Con ventana diferente (ej. 7 frames)
python run_full_pipeline.py --window_size 7

# Si ya calculaste features antes
python run_full_pipeline.py --skip_features

# Solo preparar datos, no entrenar
python run_full_pipeline.py --skip_training
```

---

### Opción B: Paso a Paso Manual

#### Paso 1: Calcular Features por Frame

```bash
cd "Entrega 3"

# Procesar todos los archivos de una vez (RECOMENDADO)
python 2_feature_engineering/02_compute_features.py --batch

# O procesar archivo individual
python 2_feature_engineering/02_compute_features.py \
  --preprocessed_csv data/preprocessed/video1_preprocessed.csv \
  --out_csv data/features_per_frame/video1_features.csv
```

**Salida:** `data/features_per_frame/*_features.csv` con columnas:
- `video`, `frame`
- `knee_left`, `knee_right`, `hip_left`, `hip_right`, `trunk_angle`, `motion_energy`

---

#### Paso 2: Crear Dataset con Ventanas Deslizantes

```bash
python 2_feature_engineering/04_create_window_dataset.py \
  --data_dir data \
  --window_size 5
```

**¿Qué hace?**
- Lee todos los `*_features.csv` en `data/features_per_frame/`
- Lee `data/labels_nuevos.csv`
- Une features + labels por `(video, frame)`
- Crea ventanas deslizantes de tamaño 5
- Cada fila del dataset resultante = 5 frames × 6 features = 30 columnas + 1 label

**Salida:** `data/processed_windowed/windowed_dataset.csv`

**⚠️ Ventana deslizante explicada:**
```
Frame 0: [f0_knee_left, f0_knee_right, ..., f0_motion] → descartado (no hay historia)
Frame 1: [f0_feat..., f1_feat...] → descartado
Frame 2: [f0_feat..., f1_feat..., f2_feat...] → descartado
Frame 3: [f0_feat..., f1_feat..., f2_feat..., f3_feat...] → descartado
Frame 4: [f0_feat..., f1_feat..., f2_feat..., f3_feat..., f4_feat...] + label_4 → ✅ primera fila
Frame 5: [f1_feat..., f2_feat..., f3_feat..., f4_feat..., f5_feat...] + label_5 → ✅ segunda fila
...
```

---

#### Paso 3: Preprocesar y Dividir (Train/Test)

```bash
python 3_model_training/05_preprocess_train_split.py \
  --input data/processed_windowed/windowed_dataset.csv \
  --out_dir data/processed_windowed
```

**¿Qué hace?**
- Carga el dataset windowed
- Entrena un `StandardScaler` con train data
- Divide en train/test (80/20, estratificado)
- Escala ambos conjuntos
- Codifica labels con `LabelEncoder`

**Salida:**
- `data/processed_windowed/scaler.pkl` ← ⚠️ CRÍTICO para tiempo real
- `data/processed_windowed/label_encoder.pkl`
- `data/processed_windowed/train_windowed.csv`
- `data/processed_windowed/test_windowed.csv`

---

#### Paso 4: Entrenar Modelos

```bash
python 3_model_training/06_train_models.py
```

**Modelos entrenados:**
1. **Random Forest** (n_estimators=200)
2. **SVM** (kernel=rbf, C=10)
3. **XGBoost** (si está instalado)

**Salida:**
- `results/models/randomforest.pkl`
- `results/models/svm.pkl`
- `results/models/xgboost.pkl` (opcional)
- `results/reports/training_report.txt` ← Ver métricas aquí

**Ver resultados:**
```bash
cat results/reports/training_report.txt
```

---

#### Paso 5: Copiar Modelo a Assets (para tiempo real)

```bash
# Copiar scaler y label encoder
cp data/processed_windowed/scaler.pkl assets/
cp data/processed_windowed/label_encoder.pkl assets/

# Copiar el mejor modelo (revisar training_report.txt)
cp results/models/randomforest.pkl assets/

# O si SVM fue mejor:
# cp results/models/svm.pkl assets/randomforest.pkl
```

---

## 🎥 Ejecución en Tiempo Real

Una vez que tienes los modelos entrenados y copiados a `assets/`:

```bash
cd "Entrega 3"
python run_realtime.py
```

**Controles:**
- Presiona `Q` para salir

**¿Qué hace la app?**
1. Abre tu webcam
2. Detecta pose con MediaPipe
3. Calcula los 6 features por frame
4. Guarda los últimos 5 frames en una cola (deque)
5. Cuando tiene 5 frames completos:
   - Aplana la ventana (5×6 = 30 features)
   - Escala con el scaler entrenado
   - Predice actividad con el modelo
6. Muestra la predicción en pantalla

**Mensajes esperados:**
- `"Cargando contexto..."` → Los primeros 4 frames (llenando ventana)
- `"Actividad: Walk to front"` → Predicción con contexto completo
- `"No se detecta persona"` → MediaPipe no ve tu cuerpo

**⚠️ RECOMENDACIONES CRÍTICAS:**

1. **Distancia de la cámara:**
   - Colócate a 2-3 metros de la cámara
   - Asegúrate de que tu cuerpo COMPLETO esté visible (cabeza a pies)
   - Si la cámara no ve tus rodillas/tobillos, los features serán inválidos

2. **Iluminación:**
   - Buena iluminación frontal
   - Evita contraluz
   - Similar a las condiciones de los videos de entrenamiento

3. **Ropa:**
   - Evita ropa muy holgada o del mismo color que el fondo
   - MediaPipe funciona mejor con contraste

4. **Movimiento:**
   - Haz movimientos claros y completos
   - Recuerda: el modelo necesita 5 frames (ventana) para predecir
   - Si cambias de actividad, espera ~5 frames para que actualice

---

## 🔧 Troubleshooting

### Problema 1: "No se encontraron archivos *_preprocessed.csv"

**Solución:**
```bash
# Verifica que tus archivos están en la carpeta correcta
ls data/preprocessed/

# Si están con otro nombre, renombra:
cd data/preprocessed
for f in *_landmarks.csv; do 
  mv "$f" "${f/_landmarks.csv/_preprocessed.csv}"
done
```

---

### Problema 2: "Faltan columnas de features: ['knee_left', ...]"

**Causa:** Los CSV de features no tienen las columnas esperadas

**Solución:**
```bash
# Verifica un CSV de features
head -n 2 data/features_per_frame/video1_features.csv

# Debe tener estas columnas:
# video,frame,knee_left,knee_right,hip_left,hip_right,trunk_angle,motion_energy

# Si no las tiene, vuelve a calcular features:
python 2_feature_engineering/02_compute_features.py --batch
```

---

### Problema 3: "KeyError: 'label'" o "No se pueden unir features y labels"

**Causa:** El archivo `labels_nuevos.csv` no existe o tiene formato incorrecto

**Solución:**
```bash
# Verifica que existe
cat data/labels_nuevos.csv | head -n 5

# Debe tener EXACTAMENTE estas columnas (primera línea):
# video,frame,label

# Verifica que los nombres de video coinciden con tus CSV:
cut -d',' -f1 data/labels_nuevos.csv | sort | uniq
# Debe mostrar: video1, video2, ... (los mismos nombres que tus CSV sin extensión)
```

---

### Problema 4: "Scaler mismatch" en tiempo real

**Causa:** El scaler fue entrenado con diferente número de features

**Solución:**
```bash
# Verifica que el scaler en assets es el correcto
ls -lh assets/scaler.pkl

# Si entrenaste con WINDOW_SIZE=5, debe haber sido creado después de 05_preprocess
# Asegúrate de copiar el scaler correcto:
cp data/processed_windowed/scaler.pkl assets/scaler.pkl --force
```

---

### Problema 5: Predicciones muy malas en tiempo real

**Causas posibles:**

1. **Desajuste de cámara:**
   - La webcam no ve tu cuerpo completo
   - **Solución:** Aléjate de la cámara, usa una cámara externa o webcam con más campo de visión

2. **Datos de entrenamiento diferentes:**
   - Tus videos de entrenamiento tienen diferente encuadre/iluminación que tu webcam
   - **Solución:** Graba nuevos videos de entrenamiento con tu webcam en las mismas condiciones

3. **Window size incorrecto:**
   - **Solución:** Verifica que `run_realtime.py` tiene `WINDOW_SIZE = 5` (el mismo que usaste en `04_create_window_dataset.py`)

4. **Features calculados incorrectamente:**
   - **Solución:** Vuelve a calcular todo desde el principio:
     ```bash
     rm -rf data/features_per_frame/* data/processed_windowed/*
     python run_full_pipeline.py
     ```

---

### Problema 6: XGBoost no se instala

**Solución:**
```bash
# XGBoost es opcional, puedes usar solo RF y SVM
# Si quieres instalarlo:
pip install xgboost

# Si falla, omite XGBoost (el script lo detecta automáticamente)
```

---

### Problema 7: "method='ffill' is deprecated" (Pandas warning)

**Solución:** Edita `02_compute_features.py` línea ~85:
```python
# Cambiar:
pos = df_pre[pos_cols].fillna(method="ffill").fillna(0).values

# Por:
pos = df_pre[pos_cols].ffill().fillna(0).values
```

---

## 📊 Verificación Rápida (Checklist)

Antes de entrenar, verifica:

- [ ] Tengo archivos CSV en `data/preprocessed/` con nombres `*_preprocessed.csv`
- [ ] Tengo el archivo `data/labels_nuevos.csv` con columnas: `video,frame,label`
- [ ] Los nombres de video en `labels_nuevos.csv` coinciden con los nombres de los CSV (sin extensión)
- [ ] Instalé todas las dependencias: `pip install -r requirements.txt`
- [ ] Estoy en el directorio `Entrega 3/`

Después de entrenar, verifica:

- [ ] Existe `assets/randomforest.pkl` (o el modelo que elegiste)
- [ ] Existe `assets/scaler.pkl` (copiado desde `data/processed_windowed/`)
- [ ] Existe `assets/label_encoder.pkl`
- [ ] Revisé `results/reports/training_report.txt` y la accuracy es > 0.7

---

## 🎯 Resumen de Comandos (Copy-Paste)

```bash
# 1. Ir al directorio
cd "Entrega 3"

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Verificar que tienes los datos
ls data/preprocessed/        # Deben aparecer tus CSV
cat data/labels_nuevos.csv | head -n 5  # Verificar formato

# 4. Ejecutar pipeline completo
python run_full_pipeline.py

# 5. Revisar resultados
cat results/reports/training_report.txt

# 6. Ejecutar en tiempo real
python run_realtime.py
```

---

## 📚 Información Adicional

### Ajustar Window Size

Si quieres cambiar el tamaño de la ventana (ej. 7 frames en lugar de 5):

1. Modificar `run_full_pipeline.py`: `--window_size 7`
2. Modificar `run_realtime.py` línea ~35: `WINDOW_SIZE = 7`
3. Re-entrenar todo el pipeline

**Nota:** Ventanas más grandes = más contexto temporal pero menos muestras de entrenamiento

---

### Balanceo de Clases

Si tienes clases desbalanceadas, puedes:

1. **Opción 1:** Modificar `05_preprocess_train_split.py` para incluir SMOTE
2. **Opción 2:** Ajustar `class_weight='balanced'` en los modelos
3. **Opción 3:** Grabar más videos de las clases minoritarias

---

### Performance

**Tiempos esperados (con 20 videos de 2:30 min):**
- Calcular features: ~5-10 min
- Crear windowed dataset: ~30 seg
- Entrenar Random Forest: ~2-5 min
- Entrenar SVM: ~10-30 min
- Tiempo real: 30 FPS (depende de tu CPU)

---

## 📞 Soporte

Si algo no funciona:

1. Revisa la sección [Troubleshooting](#troubleshooting)
2. Verifica el checklist de verificación
3. Ejecuta comandos de diagnóstico en la sección correspondiente

---

**¡Buena suerte con tu proyecto! 🚀**