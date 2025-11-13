"""
Script de clasificación de actividades humanas en tiempo real usando cámara web.

Este script utiliza MediaPipe Pose para detectar landmarks corporales y un modelo
Random Forest pre-entrenado para clasificar actividades (Sit, Stand, Walk, etc.).

Requisitos:
    - Modelo entrenado (randomforest.pkl)
    - Escalador (scaler.pkl)
    - Codificador de etiquetas (label_encoder.pkl)
    - Funciones de extracción de características (utils.py)
"""

import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd
from utils import calculate_angle, calculate_trunk_angle, calculate_motion_energy


# ============================================================================
# CONSTANTES
# ============================================================================

# Orden exacto de características usado durante el entrenamiento
FEATURE_COLUMNS = [
    'knee_left', 
    'knee_right', 
    'hip_left', 
    'hip_right', 
    'trunk_angle', 
    'motion_energy'
]

# Rutas a los assets
MODEL_PATH = 'assets/randomforest.pkl'
SCALER_PATH = 'assets/scaler.pkl'
LABEL_ENCODER_PATH = 'assets/label_encoder.pkl'


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta el sistema de clasificación en tiempo real.
    """
    
    # ------------------------------------------------------------------------
    # 1. CARGAR ASSETS
    # ------------------------------------------------------------------------
    print("Cargando modelo y assets...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        print("✓ Assets cargados correctamente")
    except FileNotFoundError as e:
        print(f"✗ Error: No se encontró el archivo {e.filename}")
        print("  Asegúrate de que los archivos .pkl están en la carpeta 'assets/'")
        return
    except Exception as e:
        print(f"✗ Error al cargar assets: {e}")
        return
    
    # ------------------------------------------------------------------------
    # 2. INICIALIZAR CÁMARA Y MEDIAPIPE
    # ------------------------------------------------------------------------
    print("Inicializando cámara y MediaPipe Pose...")
    
    # Inicializar cámara web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Error: No se pudo acceder a la cámara web")
        return
    
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Variables de estado
    prev_landmarks = None
    current_pose_landmarks = None
    
    print("✓ Sistema iniciado correctamente")
    print("\nPresiona 'q' para salir\n")
    
    # ------------------------------------------------------------------------
    # 3. BUCLE PRINCIPAL
    # ------------------------------------------------------------------------
    while True:
        # Leer frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("✗ Error: No se pudo leer el frame de la cámara")
            break
        
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Convertir BGR a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame con MediaPipe Pose
        results = pose.process(rgb_frame)
        
        # Actualizar landmarks actuales si se detecta una persona
        if results.pose_landmarks is not None:
            current_pose_landmarks = results.pose_landmarks.landmark
        else:
            current_pose_landmarks = None
        
        # --------------------------------------------------------------------
        # 4. EXTRACCIÓN DE CARACTERÍSTICAS Y PREDICCIÓN
        # --------------------------------------------------------------------
        if current_pose_landmarks is not None:
            try:
                # Calcular las 6 características usando utils.py
                
                # Ángulos de rodillas (knee angles)
                # LEFT: HIP(23) - KNEE(25) - ANKLE(27)
                knee_left = calculate_angle(
                    current_pose_landmarks[23],  # LEFT_HIP
                    current_pose_landmarks[25],  # LEFT_KNEE
                    current_pose_landmarks[27]   # LEFT_ANKLE
                )
                
                # RIGHT: HIP(24) - KNEE(26) - ANKLE(28)
                knee_right = calculate_angle(
                    current_pose_landmarks[24],  # RIGHT_HIP
                    current_pose_landmarks[26],  # RIGHT_KNEE
                    current_pose_landmarks[28]   # RIGHT_ANKLE
                )
                
                # Ángulos de caderas (hip angles)
                # LEFT: SHOULDER(11) - HIP(23) - KNEE(25)
                hip_left = calculate_angle(
                    current_pose_landmarks[11],  # LEFT_SHOULDER
                    current_pose_landmarks[23],  # LEFT_HIP
                    current_pose_landmarks[25]   # LEFT_KNEE
                )
                
                # RIGHT: SHOULDER(12) - HIP(24) - KNEE(26)
                hip_right = calculate_angle(
                    current_pose_landmarks[12],  # RIGHT_SHOULDER
                    current_pose_landmarks[24],  # RIGHT_HIP
                    current_pose_landmarks[26]   # RIGHT_KNEE
                )
                
                # Ángulo del tronco
                trunk_angle = calculate_trunk_angle(current_pose_landmarks)
                
                # Energía de movimiento (0 en el primer frame)
                if prev_landmarks is None:
                    motion_energy = 0.0
                else:
                    motion_energy = calculate_motion_energy(
                        current_pose_landmarks, 
                        prev_landmarks
                    )
                
                # Actualizar landmarks previos para el siguiente frame
                prev_landmarks = current_pose_landmarks
                
                # --------------------------------------------------------
                # 5. CREAR VECTOR DE CARACTERÍSTICAS
                # --------------------------------------------------------
                features_dict = {
                    'knee_left': knee_left,
                    'knee_right': knee_right,
                    'hip_left': hip_left,
                    'hip_right': hip_right,
                    'trunk_angle': trunk_angle,
                    'motion_energy': motion_energy
                }
                
                # Crear DataFrame con una sola fila
                features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
                
                # --------------------------------------------------------
                # 6. ESCALAR Y PREDECIR
                # --------------------------------------------------------
                # Escalar características
                features_scaled = scaler.transform(features_df)
                
                # Predecir actividad
                prediction_numeric = model.predict(features_scaled)
                
                # Decodificar etiqueta
                activity = label_encoder.inverse_transform(prediction_numeric)[0]
                
                # --------------------------------------------------------
                # 7. VISUALIZAR PREDICCIÓN
                # --------------------------------------------------------
                # Dibujar el nombre de la actividad en el frame
                cv2.putText(
                    frame,
                    f'Actividad: {activity}',
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA
                )
                
                # Opcional: Mostrar valores de características (para debugging)
                y_offset = 80
                cv2.putText(
                    frame,
                    f'Motion: {motion_energy:.3f}',
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                
            except Exception as e:
                # Manejar errores en el cálculo de características
                cv2.putText(
                    frame,
                    f'Error: {str(e)[:30]}',
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Dibujar landmarks de la pose
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
        
        else:
            # No se detecta persona
            prev_landmarks = None
            
            cv2.putText(
                frame,
                'No se detecta persona',
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
        
        # --------------------------------------------------------------------
        # 8. MOSTRAR VIDEO
        # --------------------------------------------------------------------
        cv2.imshow('Clasificacion de Actividades en Tiempo Real', frame)
        
        # Esperar tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nCerrando aplicación...")
            break
    
    # ------------------------------------------------------------------------
    # 9. LIMPIEZA
    # ------------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("✓ Aplicación cerrada correctamente")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
