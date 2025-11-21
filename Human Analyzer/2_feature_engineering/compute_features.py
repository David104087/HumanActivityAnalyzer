"""
Script 2: Ingeniería de Características
Calcula ángulos biomecánicos y características temporales.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.options.mode.chained_assignment = None


def calcular_angulo(p1, p2, p3):
    """
    Calcula el ángulo formado por los puntos p1 – p2 – p3.
    p2 es el vértice del ángulo.
    """
    p1, p2, p3 = map(np.array, (p1, p2, p3))
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return np.nan
    
    cos_theta = np.dot(v1, v2) / denom
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    
    return np.degrees(np.arccos(cos_theta))


def generar_angulos(fila):
    """
    Extrae landmarks, los normaliza y calcula ángulos biomecánicos.
    Retorna un diccionario con 8 ángulos.
    """
    try:
        puntos = np.array([
            [fila[f"lm{i}_x"], fila[f"lm{i}_y"], fila[f"lm{i}_z"]]
            for i in range(33)
        ])
        
        # Normalización por centro de cadera
        cadera_izq, cadera_der = puntos[23], puntos[24]
        centro_cadera = (cadera_izq + cadera_der) / 2
        pts_norm = puntos - centro_cadera
        
        # Normalización por distancia entre hombros
        hombro_i, hombro_d = pts_norm[11], pts_norm[12]
        dist_hombros = np.linalg.norm(hombro_i - hombro_d)
        
        if dist_hombros < 1e-6:
            return None
        
        pts_norm /= dist_hombros
        
        # Puntos relevantes
        si, sd = pts_norm[11], pts_norm[12]  # Hombros
        ci, cd = pts_norm[23], pts_norm[24]  # Caderas
        ei, ed = pts_norm[13], pts_norm[14]  # Codos
        mi, md = pts_norm[15], pts_norm[16]  # Muñecas
        ri, rd = pts_norm[25], pts_norm[26]  # Rodillas
        ti, td = pts_norm[27], pts_norm[28]  # Tobillos
        
        res = {}
        res["ang_elb_izq"] = calcular_angulo(si, ei, mi)
        res["ang_elb_der"] = calcular_angulo(sd, ed, md)
        res["ang_sho_izq"] = calcular_angulo(ei, si, ci)
        res["ang_sho_der"] = calcular_angulo(ed, sd, cd)
        res["ang_cad_izq"] = calcular_angulo(si, ci, ri)
        res["ang_cad_der"] = calcular_angulo(sd, cd, rd)
        res["ang_rod_izq"] = calcular_angulo(ci, ri, ti)
        res["ang_rod_der"] = calcular_angulo(cd, rd, td)
        
        return res
        
    except Exception:
        return None


def ventana_temporal(df_vid, tam_ventana=15):
    """
    Calcula características temporales con ventana deslizante.
    Genera estadísticas de posición y velocidad angular.
    """
    nombre_video = df_vid["video_filename"].iloc[0]
    tqdm.pandas(desc=f"Extrayendo ángulos: {nombre_video}", leave=False)
    
    lista_ang = df_vid.progress_apply(generar_angulos, axis=1)
    df_ang = pd.DataFrame(lista_ang.tolist()).dropna()
    
    if df_ang.empty:
        return pd.DataFrame()
    
    # Rolling window para estadísticas
    roll = df_ang.rolling(window=tam_ventana, min_periods=tam_ventana)
    
    # Estadísticas de posición
    a_mean = roll.mean().add_suffix("_m")
    a_std = roll.std().add_suffix("_s")
    a_min = roll.min().add_suffix("_lo")
    a_max = roll.max().add_suffix("_hi")
    
    # Velocidad angular
    vel = df_ang.diff().fillna(0)
    roll_v = vel.rolling(window=tam_ventana, min_periods=tam_ventana)
    
    v_mean = roll_v.mean().add_suffix("_vm")
    v_std = roll_v.std().add_suffix("_vs")
    
    # Unir características
    df_res = pd.concat([a_mean, a_std, a_min, a_max, v_mean, v_std], axis=1)
    df_res.dropna(inplace=True)
    
    if df_res.empty:
        return pd.DataFrame()
    
    df_res["action"] = df_vid["action"].iloc[0]
    df_res["video_filename"] = nombre_video
    df_res["frame_idx"] = df_res.index
    
    return df_res


def main():
    """Función principal de ingeniería de características."""
    print("\n" + "="*70)
    print("PASO 2: INGENIERÍA DE CARACTERÍSTICAS")
    print("="*70 + "\n")
    
    origen = "./data/processed/datosmediapipe.csv"
    destino = "./data/processed/model_features.csv"
    
    print(f"Cargando landmarks desde: {origen}")
    
    try:
        df = pd.read_csv(origen)
        print(f"Dataset cargado: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: No se encuentra el archivo {origen}")
        print("Ejecuta primero: python 1_data_extraction/extract_landmarks.py")
        return
    
    print("\nCalculando características temporales...")
    print("- Ventana temporal: 15 frames")
    print("- Ángulos biomecánicos: 8")
    print("- Características por ventana: 48\n")
    
    grupos = df.groupby("video_filename")
    resultados = []
    
    for nombre, grupo in tqdm(grupos, desc="Procesando videos"):
        feats = ventana_temporal(grupo, tam_ventana=15)
        resultados.append(feats)
    
    df_final = pd.concat(resultados, ignore_index=True)
    df_final.to_csv(destino, index=False)
    
    print(f"\n✓ Características guardadas: {destino}")
    print(f"  Shape: {df_final.shape}")
    print(f"  Acciones: {df_final['action'].unique().tolist()}")
    
    print("\n" + "="*70)
    print("INGENIERÍA DE CARACTERÍSTICAS COMPLETADA")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
