# Inicio Rapido - Sistema de Reconocimiento de Actividad Humana

## Pasos para poner en marcha el sistema

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Colocar Videos

Organiza tus videos en la siguiente estructura:

```
data/raw/videos/
├── caminar-adelante/    [COLOCAR VIDEOS AQUI]
├── caminar-atras/       [COLOCAR VIDEOS AQUI]
├── girar/               [COLOCAR VIDEOS AQUI]
├── parase/              [COLOCAR VIDEOS AQUI]
└── sentarse/            [COLOCAR VIDEOS AQUI]
```

### 3. Ejecutar Pipeline Completo

```bash
# Paso 1: Extraer landmarks (toma varios minutos)
python src/get_landmarks.py --stride 1

# Paso 2: Generar caracteristicas (rapido)
python src/feature_engineering.py

# Paso 3: Entrenar modelos (toma varios minutos)
python src/model_training.py

# Paso 4: Ejecutar sistema en tiempo real
python src/main.py
```

## Verificacion Rapida

Despues de cada paso, verifica:

1. **Paso 1**: Se crearon `data/processed/datosmediapipe.csv` y `data/processed/datos_analisis.csv`
2. **Paso 2**: Se creo `data/processed/model_features.csv`
3. **Paso 3**: Se crearon archivos en la carpeta `models/`
4. **Paso 4**: Se abre ventana con webcam y deteccion en tiempo real

## Controles en Tiempo Real

- **q**: Salir del sistema
- La pantalla muestra:
  - Accion detectada en tiempo real
  - Porcentaje de confianza
  - Barra de progreso
  - Esqueleto de la persona

## Colores de Confianza

- **Verde**: Alta confianza (>80%)
- **Amarillo**: Media confianza (50-80%)
- **Rojo**: Baja confianza (<50%)

## Acciones Detectables

1. caminar-adelante
2. caminar-atras
3. girar
4. parase
5. sentarse

## Problemas Comunes

### No detecta la camara

```bash
# Verifica que la camara funciona
# Cierra otras apps que usen la camara
```

### Error al cargar modelos

```bash
# Asegurate de ejecutar primero model_training.py
python src/model_training.py
```

### Predicciones erraticas

- Mejora la iluminacion del ambiente
- Mantente a 2-3 metros de la camara
- Asegurate de que el cuerpo completo sea visible

## Estructura de Archivos Generados

```
proyecto/
├── data/
│   └── processed/
│       ├── datos_analisis.csv      (metadatos)
│       ├── datosmediapipe.csv      (landmarks crudos)
│       └── model_features.csv      (caracteristicas finales)
│
└── models/
    ├── best_random_forest_model.joblib    (modelo RF)
    ├── best_xgboost_model.joblib          (modelo XGB)
    ├── label_encoder.joblib               (codificador)
    ├── random_forest_confusion_matrix.png (evaluacion)
    └── xgboost_confusion_matrix.png       (evaluacion)
```

## Metricas Esperadas

- **Accuracy**: >90%
- **FPS**: 15-30 fps
- **Latencia**: 50-100ms

Para mas informacion detallada, consulta `README.md`
