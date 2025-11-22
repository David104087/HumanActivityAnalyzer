import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import joblib
import mediapipe as mp


# ==============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ==============================================================================

# Rutas de los modelos
MODEL_PATH = Path("./assets/best_random_forest_model.joblib")
ENCODER_PATH = Path("./assets/label_encoder.joblib")

# Parámetros del sistema
VENTANA_TEMPORAL = 15  # Frames para calcular características
HISTORIAL_PREDICCION = 8  # Frames para suavizado de predicciones
UMBRAL_QUIETUD = 40.0  # Suma de std de ángulos para detectar quietud

# Configuración de MediaPipe
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Configuración visual
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
UI_BAR_HEIGHT = 80


# ==============================================================================
# FUNCIONES DE CÁLCULO GEOMÉTRICO
# ==============================================================================

def calcular_angulo(p1, p2, p3):
    """
    Calcula el ángulo formado por tres puntos usando producto punto.
    p1, p2, p3: numpy arrays de shape (3,)
    Retorna el ángulo en grados.
    """
    p1, p2, p3 = map(np.array, (p1, p2, p3))
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return np.nan
    
    cos_theta = np.dot(v1, v2) / denom
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_theta))


def extraer_angulos_frame(landmarks):
    """
    Extrae 8 ángulos biomecánicos de los landmarks de MediaPipe.
    Normaliza por centro de cadera y distancia entre hombros.
    
    landmarks: mediapipe pose landmarks
    Retorna: lista de 8 ángulos [codo_izq, codo_der, hombro_izq, hombro_der,
                                   cadera_izq, cadera_der, rodilla_izq, rodilla_der]
    """
    try:
        # Extraer coordenadas 3D de los 33 landmarks
        puntos = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ])
        
        # Normalización por centro de cadera
        cadera_izq = puntos[23]
        cadera_der = puntos[24]
        centro_cadera = (cadera_izq + cadera_der) / 2
        
        pts_norm = puntos - centro_cadera
        
        # Normalización por distancia entre hombros
        hombro_izq = pts_norm[11]
        hombro_der = pts_norm[12]
        dist_hombros = np.linalg.norm(hombro_izq - hombro_der)
        
        if dist_hombros < 1e-6:  # Evitar división por cero
            return None
        
        pts_norm = pts_norm / dist_hombros
        
        # Puntos relevantes normalizados
        si, sd = pts_norm[11], pts_norm[12]  # Hombros
        ci, cd = pts_norm[23], pts_norm[24]  # Caderas
        ei, ed = pts_norm[13], pts_norm[14]  # Codos
        mi, md = pts_norm[15], pts_norm[16]  # Muñecas
        ri, rd = pts_norm[25], pts_norm[26]  # Rodillas
        ti, td = pts_norm[27], pts_norm[28]  # Tobillos
        
        # Calcular 8 ángulos
        angulos = [
            calcular_angulo(si, ei, mi),  # Codo izquierdo
            calcular_angulo(sd, ed, md),  # Codo derecho
            calcular_angulo(ei, si, ci),  # Hombro izquierdo
            calcular_angulo(ed, sd, cd),  # Hombro derecho
            calcular_angulo(si, ci, ri),  # Cadera izquierda
            calcular_angulo(sd, cd, rd),  # Cadera derecha
            calcular_angulo(ci, ri, ti),  # Rodilla izquierda
            calcular_angulo(cd, rd, td),  # Rodilla derecha
        ]
        
        # Verificar que todos los ángulos sean válidos
        if any(np.isnan(a) for a in angulos):
            return None
        
        return angulos
        
    except Exception as e:
        return None


def calcular_caracteristicas(ventana_angulos):
    """
    Calcula 48 características desde una ventana de 15 frames de ángulos.
    
    ventana_angulos: deque con 15 elementos, cada uno es una lista de 8 ángulos
    Retorna: numpy array de 48 características
    """
    # Convertir deque a numpy array (15, 8)
    angulos_array = np.array(list(ventana_angulos))
    
    if angulos_array.shape[0] != 15:
        return None
    
    # Calcular velocidades angulares (diferencias entre frames)
    velocidades = np.diff(angulos_array, axis=0)  # (14, 8)
    velocidades = np.vstack([np.zeros(8), velocidades])  # (15, 8) - agregar 0 al inicio
    
    # Estadísticas de posición (32 features)
    angle_mean = np.mean(angulos_array, axis=0)  # 8
    angle_std = np.std(angulos_array, axis=0)    # 8
    angle_min = np.min(angulos_array, axis=0)    # 8
    angle_max = np.max(angulos_array, axis=0)    # 8
    
    # Estadísticas de velocidad (16 features)
    velocity_mean = np.mean(velocidades, axis=0)  # 8
    velocity_std = np.std(velocidades, axis=0)    # 8
    
    # Concatenar todas las características (48 total)
    features = np.concatenate([
        angle_mean, angle_std, angle_min, angle_max,
        velocity_mean, velocity_std
    ])
    
    return features


# ==============================================================================
# CLASE PRINCIPAL DEL SISTEMA
# ==============================================================================

class SistemaReconocimientoTiempoReal:
    """Sistema de reconocimiento de actividad humana en tiempo real."""
    
    def __init__(self):
        """Inicializa el sistema cargando modelos y configurando componentes."""
        print("\n" + "="*60)
        print("SISTEMA DE RECONOCIMIENTO DE ACTIVIDAD HUMANA")
        print("="*60 + "\n")
        
        # Cargar modelo y encoder
        self.cargar_modelos()
        
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=0,  # 0 es más rápido, 1 es balanceado, 2 es más preciso
            smooth_landmarks=True
        )
        
        # Búferes
        self.angle_deque = deque(maxlen=VENTANA_TEMPORAL)
        self.prediction_history = deque(maxlen=HISTORIAL_PREDICCION)
        
        # Estado de visualización
        self.displayed_action = "INICIANDO..."
        self.displayed_confidence = 0.0
        
        # Contador de frames y estado del detector
        self.frame_count = 0
        self.detector_status = "Iniciando..."
        
        # Estado de pantalla completa
        self.fullscreen = False
        
        print("Sistema inicializado correctamente.\n")
    
    def cargar_modelos(self):
        """Carga el modelo entrenado y el label encoder."""
        print("Cargando modelos...")
        
        if not MODEL_PATH.exists():
            print(f"ERROR: No se encuentra el modelo en {MODEL_PATH}")
            print("Ejecuta primero: python src/model_training.py")
            sys.exit(1)
        
        if not ENCODER_PATH.exists():
            print(f"ERROR: No se encuentra el encoder en {ENCODER_PATH}")
            print("Ejecuta primero: python src/model_training.py")
            sys.exit(1)
        
        self.model = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(ENCODER_PATH)
        
        print(f"Modelo cargado: {MODEL_PATH.name}")
        print(f"Clases detectables: {list(self.label_encoder.classes_)}")
    
    def procesar_frame(self, frame):
        """
        Procesa un frame individual.
        
        frame: imagen BGR de OpenCV
        Retorna: frame procesado con visualización
        """
        # Incrementar contador de frames
        self.frame_count += 1
        
        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Detectar pose
        results = self.pose.process(rgb_frame)
        
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Procesar landmarks
        if results.pose_landmarks:
            self.detector_status = "Persona detectada"
            # Dibujar esqueleto con estilo personalizado
            landmark_style = self.mp_drawing.DrawingSpec(
                color=(0, 255, 200),  # Cyan vibrante
                thickness=3,
                circle_radius=4
            )
            connection_style = self.mp_drawing.DrawingSpec(
                color=(255, 180, 0),  # Naranja dorado
                thickness=2,
                circle_radius=2
            )
            
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style
            )
            
            # Extraer ángulos
            angulos = extraer_angulos_frame(results.pose_landmarks)
            
            if angulos is not None:
                # Agregar ángulos al búfer
                self.angle_deque.append(angulos)
                
                # Si tenemos suficientes frames, predecir
                if len(self.angle_deque) == VENTANA_TEMPORAL:
                    self.predecir()
        else:
            # Si no detecta cuerpo, limpiar búferes
            self.detector_status = "Sin persona"
            self.angle_deque.clear()
            self.prediction_history.clear()
            self.displayed_action = "BUSCANDO..."
            self.displayed_confidence = 0.0
        
        # Dibujar interfaz y obtener canvas extendido
        frame_con_panel = self.dibujar_interfaz(frame)
        
        return frame_con_panel
    
    def predecir(self):
        """Realiza predicción basada en la ventana de ángulos."""
        # Calcular características
        features = calcular_caracteristicas(self.angle_deque)
        
        if features is None:
            return
        
        # Gate de quietud
        angulos_array = np.array(list(self.angle_deque))
        angle_std = np.std(angulos_array, axis=0)
        suma_std = np.sum(angle_std)
        
        if suma_std < UMBRAL_QUIETUD:
            # Persona quieta
            accion = "quieto"
            confianza = 0.98
        else:
            # Predecir con modelo
            features_reshaped = features.reshape(1, -1)
            
            try:
                pred_encoded = self.model.predict(features_reshaped)[0]
                accion = self.label_encoder.inverse_transform([pred_encoded])[0]
                
                # Obtener probabilidades
                if hasattr(self.model, 'predict_proba'):
                    probas = self.model.predict_proba(features_reshaped)[0]
                    confianza = probas[pred_encoded]
                else:
                    confianza = 0.85
            except Exception as e:
                accion = "ERROR"
                confianza = 0.0
        
        # Agregar a historial
        self.prediction_history.append((accion, confianza))
        
        # Suavizado por votación mayoritaria
        self.suavizar_prediccion()
    
    def suavizar_prediccion(self):
        """Aplica suavizado por votación mayoritaria en el historial."""
        if len(self.prediction_history) == 0:
            return
        
        # Contar votos
        votos = {}
        confianzas = {}
        
        for accion, conf in self.prediction_history:
            votos[accion] = votos.get(accion, 0) + 1
            if accion not in confianzas:
                confianzas[accion] = []
            confianzas[accion].append(conf)
        
        # Acción ganadora
        accion_ganadora = max(votos, key=votos.get)
        
        # Promedio de confianza de la acción ganadora
        avg_confidence = np.mean(confianzas[accion_ganadora])
        
        # Filtro Lerp para transición suave
        self.displayed_confidence = 0.8 * self.displayed_confidence + 0.2 * avg_confidence
        self.displayed_action = accion_ganadora.upper()
    
    def dibujar_interfaz(self, frame):
        """
        Dibuja la interfaz de usuario sobre el frame.
        Incluye información de la acción detectada, confianza y estado.
        """
        h, w = frame.shape[:2]
        
        # Determinar colores según confianza
        if self.displayed_confidence > 0.8:
            color_principal = (50, 205, 50)  # Verde Lima
            color_secundario = (34, 139, 34)  # Verde Oscuro
        elif self.displayed_confidence >= 0.5:
            color_principal = (0, 191, 255)  # Azul Cielo
            color_secundario = (0, 120, 180)  # Azul Medio
        else:
            color_principal = (255, 99, 71)  # Rojo Tomate
            color_secundario = (178, 34, 34)  # Rojo Oscuro
        
        # Crear canvas más grande: video original + panel lateral
        panel_width = 250
        canvas_width = w + panel_width
        canvas = np.zeros((h, canvas_width, 3), dtype=np.uint8)
        
        # Colocar frame original en el lado izquierdo
        canvas[0:h, 0:w] = frame
        
        # Coordenadas del panel (ahora en el canvas extendido)
        panel_x = w
        
        # Fondo del panel sólido
        cv2.rectangle(canvas, (panel_x, 0), (canvas_width, h), (25, 25, 35), -1)
        
        # Línea decorativa vertical
        cv2.line(canvas, (panel_x, 0), (panel_x, h), color_principal, 3)
        
        # Título del panel
        cv2.putText(
            canvas,
            "ACTIVIDAD",
            (panel_x + 15, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )
        
        # Acción detectada (más grande y centrada)
        action_text = self.displayed_action
        font_scale = 0.9 if len(action_text) > 10 else 1.1
        text_size = cv2.getTextSize(action_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 3)[0]
        text_x = panel_x + (panel_width - text_size[0]) // 2
        
        cv2.putText(
            canvas,
            action_text,
            (text_x, 85),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            color_principal,
            3,
            cv2.LINE_AA
        )
        
        # Línea separadora
        cv2.line(canvas, (panel_x + 15, 110), (canvas_width - 15, 110), (80, 80, 90), 1)
        
        # Confianza con círculo de progreso
        conf_y = 160
        circle_center = (panel_x + 60, conf_y)
        circle_radius = 40
        
        # Círculo de fondo
        cv2.circle(canvas, circle_center, circle_radius, (50, 50, 60), 5)
        
        # Arco de progreso (solo si hay confianza significativa)
        angle = int(360 * self.displayed_confidence)
        if angle > 5:  # Solo dibujar si hay progreso visible
            axes = (circle_radius, circle_radius)
            cv2.ellipse(canvas, circle_center, axes, -90, 0, angle, color_principal, 5)
        
        # Porcentaje en el centro del círculo
        conf_text = f"{int(self.displayed_confidence * 100)}"
        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 3)[0]
        text_x = circle_center[0] - text_size[0] // 2
        text_y = circle_center[1] + text_size[1] // 2
        
        cv2.putText(
            canvas,
            conf_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )
        
        # Símbolo de porcentaje
        cv2.putText(
            canvas,
            "%",
            (text_x + text_size[0] + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        
        # Etiqueta "Confianza"
        cv2.putText(
            canvas,
            "CONFIANZA",
            (panel_x + 125, conf_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )
        
        # Indicador de nivel con barras
        bar_y_start = conf_y + 15
        bar_spacing = 8
        bar_height = 5
        bar_width = 75
        
        levels = ["BAJA", "MEDIA", "ALTA"]
        thresholds = [0.5, 0.8, 1.0]
        
        for i, (level, threshold) in enumerate(zip(levels, thresholds)):
            y_pos = bar_y_start + (i * bar_spacing)
            
            # Determinar si esta barra está activa
            if self.displayed_confidence >= (threshold - 0.3):
                bar_color = color_principal
                text_color = (255, 255, 255)
            else:
                bar_color = (60, 60, 70)
                text_color = (120, 120, 130)
            
            # Dibujar mini barra
            cv2.rectangle(
                canvas,
                (panel_x + 130, y_pos),
                (panel_x + 130 + bar_width, y_pos + bar_height),
                bar_color,
                -1
            )
        
        # Línea separadora inferior
        cv2.line(canvas, (panel_x + 15, 240), (canvas_width - 15, 240), (80, 80, 90), 1)
        
        # Información adicional en la parte inferior del panel
        info_y = 270
        
        # Determinar color del estado según detección
        if self.detector_status == "Persona detectada":
            status_color = (100, 220, 100)  # Verde
        else:
            status_color = (180, 100, 100)  # Rojo
        
        # Estado del detector
        cv2.putText(
            canvas,
            f"Estado: {self.detector_status}",
            (panel_x + 15, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
            cv2.LINE_AA
        )
        
        # Contador de frames
        cv2.putText(
            canvas,
            f"Frames: {self.frame_count}",
            (panel_x + 15, info_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 160),
            1,
            cv2.LINE_AA
        )
        
        # Instrucciones
        instructions = [
            "Q: Salir",
            "F: Pantalla",
            "   completa"
        ]
        
        inst_y = h - 75
        for i, inst in enumerate(instructions):
            cv2.putText(
                canvas,
                inst,
                (panel_x + 15, inst_y + (i * 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (120, 120, 130),
                1,
                cv2.LINE_AA
            )
        
        # Banner superior sobre el video (lado izquierdo)
        banner_height = 45
        banner_overlay = canvas[0:banner_height, 0:w].copy()
        cv2.rectangle(banner_overlay, (0, 0), (w, banner_height), (20, 20, 28), -1)
        cv2.addWeighted(banner_overlay, 0.7, canvas[0:banner_height, 0:w], 0.3, 0, canvas[0:banner_height, 0:w])
        
        # Título de la aplicación
        cv2.putText(
            canvas,
            "Human Activity Recognition",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 210),
            1,
            cv2.LINE_AA
        )
        
        # Línea decorativa horizontal
        cv2.line(canvas, (0, banner_height), (w, banner_height), color_secundario, 2)
        
        # Retornar canvas con video + panel
        return canvas
    
    def toggle_fullscreen(self, window_name):
        """Alterna entre modo pantalla completa y modo ventana."""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            # Modo pantalla completa
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
            print("Pantalla completa activada (presiona F nuevamente para salir)")
        else:
            # Modo ventana normal
            cv2.setWindowProperty(
                window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )
            print("Modo ventana restaurado")
    
    def ejecutar(self):
        """Ejecuta el loop principal del sistema."""
        print("\nIniciando captura de video...")
        print("Presiona 'q' para salir.")
        print("Presiona 'f' para pantalla completa.\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: No se puede acceder a la cámara.")
            print("Verifica que la cámara esté conectada y no esté siendo usada por otra aplicación.")
            sys.exit(1)
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        window_name = 'Human Activity Recognition System'
        
        # Crear ventana redimensionable
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("="*60)
        print("SISTEMA EN EJECUCIÓN")
        print("="*60)
        print("Acciones detectables:")
        for accion in self.label_encoder.classes_:
            print(f"  - {accion}")
        print("\nPresiona 'q' para detener el sistema.")
        print("Presiona 'f' para pantalla completa.")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("ERROR: No se puede leer el frame de la cámara.")
                    break
                
                # Procesar frame
                frame_procesado = self.procesar_frame(frame)
                
                # Mostrar resultado
                cv2.imshow(window_name, frame_procesado)
                
                # Detectar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nSaliendo del sistema...")
                    break
                elif key == ord('f') or key == ord('F'):
                    self.toggle_fullscreen(window_name)
        
        except KeyboardInterrupt:
            print("\n\nInterrumpido por el usuario.")
        
        finally:
            # Liberar recursos
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print("Recursos liberados. Sistema detenido.")


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

def main():
    """Función principal."""
    sistema = SistemaReconocimientoTiempoReal()
    sistema.ejecutar()


if __name__ == "__main__":
    main()
