"""
Utilidades para el sistema en tiempo real.
Funciones auxiliares reutilizables.
"""
import numpy as np


def calcular_angulo(p1, p2, p3):
    """
    Calcula el ángulo formado por tres puntos.
    
    Args:
        p1: Punto 1 (numpy array)
        p2: Punto 2 - vértice del ángulo (numpy array)
        p3: Punto 3 (numpy array)
    
    Returns:
        float: Ángulo en grados
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


def normalizar_landmarks(landmarks):
    """
    Normaliza landmarks por centro de cadera y distancia entre hombros.
    
    Args:
        landmarks: MediaPipe pose landmarks
    
    Returns:
        numpy array: Landmarks normalizados, o None si falla
    """
    try:
        # Extraer puntos
        puntos = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ])
        
        # Centro de cadera
        cadera_izq = puntos[23]
        cadera_der = puntos[24]
        centro_cadera = (cadera_izq + cadera_der) / 2
        
        pts_norm = puntos - centro_cadera
        
        # Distancia entre hombros
        hombro_izq = pts_norm[11]
        hombro_der = pts_norm[12]
        dist_hombros = np.linalg.norm(hombro_izq - hombro_der)
        
        if dist_hombros < 1e-6:
            return None
        
        pts_norm = pts_norm / dist_hombros
        
        return pts_norm
        
    except Exception:
        return None


def extraer_angulos_biomecánicos(pts_norm):
    """
    Extrae 8 ángulos biomecánicos desde landmarks normalizados.
    
    Args:
        pts_norm: Landmarks normalizados (33, 3)
    
    Returns:
        list: Lista de 8 ángulos, o None si falla
    """
    try:
        # Puntos relevantes
        si, sd = pts_norm[11], pts_norm[12]  # Hombros
        ci, cd = pts_norm[23], pts_norm[24]  # Caderas
        ei, ed = pts_norm[13], pts_norm[14]  # Codos
        mi, md = pts_norm[15], pts_norm[16]  # Muñecas
        ri, rd = pts_norm[25], pts_norm[26]  # Rodillas
        ti, td = pts_norm[27], pts_norm[28]  # Tobillos
        
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
        
        # Verificar validez
        if any(np.isnan(a) for a in angulos):
            return None
        
        return angulos
        
    except Exception:
        return None


def calcular_inclinacion_lateral(landmarks):
    """
    Calcula la inclinación lateral del tronco.
    
    Args:
        landmarks: MediaPipe pose landmarks
    
    Returns:
        float: Inclinación en grados, o None si falla
    """
    try:
        puntos = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks.landmark
        ])
        
        hombro_izq = puntos[11][:2]  # Solo X, Y
        hombro_der = puntos[12][:2]
        
        vec_hombros = hombro_der - hombro_izq
        
        # Ángulo respecto a la horizontal
        angulo = np.degrees(np.arctan2(vec_hombros[1], vec_hombros[0]))
        
        return abs(angulo)
        
    except Exception:
        return None


def calcular_caracteristicas_ventana(ventana_angulos):
    """
    Calcula 48 características desde una ventana de ángulos.
    
    Args:
        ventana_angulos: Deque o lista con 15 frames de 8 ángulos
    
    Returns:
        numpy array: 48 características, o None si falla
    """
    try:
        # Convertir a numpy array
        angulos_array = np.array(list(ventana_angulos))
        
        if angulos_array.shape[0] != 15:
            return None
        
        # Velocidades
        velocidades = np.diff(angulos_array, axis=0)
        velocidades = np.vstack([np.zeros(8), velocidades])
        
        # Estadísticas de posición
        angle_mean = np.mean(angulos_array, axis=0)
        angle_std = np.std(angulos_array, axis=0)
        angle_min = np.min(angulos_array, axis=0)
        angle_max = np.max(angulos_array, axis=0)
        
        # Estadísticas de velocidad
        velocity_mean = np.mean(velocidades, axis=0)
        velocity_std = np.std(velocidades, axis=0)
        
        # Concatenar
        features = np.concatenate([
            angle_mean, angle_std, angle_min, angle_max,
            velocity_mean, velocity_std
        ])
        
        return features
        
    except Exception:
        return None


def determinar_color_confianza(confianza):
    """
    Determina el color según el nivel de confianza.
    
    Args:
        confianza: Valor de confianza (0-1)
    
    Returns:
        tuple: Color BGR (principal, secundario)
    """
    if confianza > 0.8:
        return (50, 205, 50), (34, 139, 34)  # Verde
    elif confianza >= 0.5:
        return (0, 191, 255), (0, 120, 180)  # Azul
    else:
        return (255, 99, 71), (178, 34, 34)  # Rojo


def formato_tiempo(segundos):
    """
    Formatea segundos a MM:SS.
    
    Args:
        segundos: Tiempo en segundos
    
    Returns:
        str: Tiempo formateado
    """
    minutos = int(segundos // 60)
    segs = int(segundos % 60)
    return f"{minutos:02d}:{segs:02d}"


class FPSCounter:
    """Contador de FPS para monitorear rendimiento."""
    
    def __init__(self, ventana=30):
        """
        Args:
            ventana: Número de frames para promediar FPS
        """
        self.ventana = ventana
        self.tiempos = []
        import time
        self.ultimo_tiempo = time.time()
    
    def actualizar(self):
        """Actualiza el contador con el tiempo actual."""
        import time
        tiempo_actual = time.time()
        delta = tiempo_actual - self.ultimo_tiempo
        self.ultimo_tiempo = tiempo_actual
        
        if delta > 0:
            fps = 1.0 / delta
            self.tiempos.append(fps)
            
            if len(self.tiempos) > self.ventana:
                self.tiempos.pop(0)
    
    def obtener_fps(self):
        """
        Retorna el FPS promedio.
        
        Returns:
            float: FPS promedio
        """
        if not self.tiempos:
            return 0.0
        return sum(self.tiempos) / len(self.tiempos)


def validar_landmarks(landmarks, umbral_visibilidad=0.5):
    """
    Valida que los landmarks tengan buena visibilidad.
    
    Args:
        landmarks: MediaPipe pose landmarks
        umbral_visibilidad: Visibilidad mínima requerida
    
    Returns:
        bool: True si los landmarks son válidos
    """
    try:
        # Verificar landmarks críticos
        indices_criticos = [11, 12, 23, 24, 25, 26]  # Hombros, caderas, rodillas
        
        for idx in indices_criticos:
            lm = landmarks.landmark[idx]
            if hasattr(lm, 'visibility') and lm.visibility < umbral_visibilidad:
                return False
        
        return True
        
    except Exception:
        return False
