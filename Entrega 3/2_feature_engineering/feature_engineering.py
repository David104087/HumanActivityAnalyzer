import pandas as pd
import numpy as np
from tqdm import tqdm

# Evita warnings de pandas innecesarios
pd.options.mode.chained_assignment = None


# -------------------------------------------------------------
#       Cálculo geométrico: Ángulo entre tres puntos
# -------------------------------------------------------------
def obtener_angulo(p1, p2, p3):
    """
    Calcula el ángulo formado por los puntos p1 – p2 – p3 usando producto punto.
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


# -------------------------------------------------------------
#   Extrae landmarks 3D y calcula los 8 ángulos de interés
# -------------------------------------------------------------
def generar_angulos(fila):
    """
    Extrae 33 landmarks, los normaliza por cadera y distancia entre hombros,
    y calcula 8 ángulos biomecánicos.
    """

    try:
        puntos = np.array([
            [fila[f"lm{i}_x"], fila[f"lm{i}_y"], fila[f"lm{i}_z"]]
            for i in range(33)
        ])



        cadera_izq, cadera_der = puntos[23], puntos[24]
        centro_cadera = (cadera_izq + cadera_der) / 2

        pts_norm = puntos - centro_cadera

        hombro_i, hombro_d = pts_norm[11], pts_norm[12]
        dist_hombros = np.linalg.norm(hombro_i - hombro_d)

        if dist_hombros == 0:
            return None

        pts_norm /= dist_hombros

        res = {}
        # puntos relevantes
        si, sd = pts_norm[11], pts_norm[12]
        ci, cd = pts_norm[23], pts_norm[24]
        ei, ed = pts_norm[13], pts_norm[14]
        mi, md = pts_norm[15], pts_norm[16]
        ri, rd = pts_norm[25], pts_norm[26]
        ti, td = pts_norm[27], pts_norm[28]

        # cálculos
        res["ang_elb_izq"] = obtener_angulo(si, ei, mi)
        res["ang_elb_der"] = obtener_angulo(sd, ed, md)
        res["ang_sho_izq"] = obtener_angulo(ei, si, ci)
        res["ang_sho_der"] = obtener_angulo(ed, sd, cd)
        res["ang_cad_izq"] = obtener_angulo(si, ci, ri)
        res["ang_cad_der"] = obtener_angulo(sd, cd, rd)
        res["ang_rod_izq"] = obtener_angulo(ci, ri, ti)
        res["ang_rod_der"] = obtener_angulo(cd, rd, td)

        return res

    except Exception:
        return None


# -------------------------------------------------------------
#   Cálculo de características temporales mediante ventanas
# -------------------------------------------------------------
def ventana_temporal(df_vid, tam_ventana=15):
    """
    Calcula estadísticas temporales (posición + velocidad)
    para un grupo de frames pertenecientes a un solo video.
    """

    nombre_video = df_vid["video_filename"].iloc[0]
    tqdm.pandas(desc=f"Extrayendo ángulos: {nombre_video}", leave=False)

    lista_ang = df_vid.progress_apply(generar_angulos, axis=1)
    df_ang = pd.DataFrame(lista_ang.tolist()).dropna()

    if df_ang.empty:
        return pd.DataFrame()

    cols_ang = df_ang.columns

    roll = df_ang.rolling(window=tam_ventana, min_periods=tam_ventana)

    # estadísticas sobre ángulos
    a_mean = roll.mean().add_suffix("_m")
    a_std = roll.std().add_suffix("_s")
    a_min = roll.min().add_suffix("_lo")
    a_max = roll.max().add_suffix("_hi")

    # velocidad angular (diferencias entre frames)
    vel = df_ang.diff().fillna(0)
    roll_v = vel.rolling(window=tam_ventana, min_periods=tam_ventana)

    v_mean = roll_v.mean().add_suffix("_vm")
    v_std = roll_v.std().add_suffix("_vs")

    # unir todo
    df_res = pd.concat([a_mean, a_std, a_min, a_max, v_mean, v_std], axis=1)
    df_res.dropna(inplace=True)

    if df_res.empty:
        return pd.DataFrame()

    df_res["action"] = df_vid["action"].iloc[0]
    df_res["video_filename"] = nombre_video
    df_res["frame_idx"] = df_res.index

    return df_res


# -------------------------------------------------------------
#   Script principal
# -------------------------------------------------------------
def ejecutar():
    origen = "./data/processed/datosmediapipe.csv"
    destino = "./data/processed/model_features.csv"

    print(f"Cargando dataset desde {origen} ...")
    try:
        df = pd.read_csv(origen)
    except FileNotFoundError:
        print(f"ERROR: archivo no encontrado: {origen}")
        return

    print("Agrupando por video...")

    grupos = df.groupby("video_filename")
    resultados = []

    for nombre, grupo in tqdm(grupos, desc="Procesando videos"):
        feats = ventana_temporal(grupo, tam_ventana=15)
        resultados.append(feats)

    df_final = pd.concat(resultados, ignore_index=True)
    df_final.to_csv(destino, index=False)

    print("\nProceso completado.")
    print(f"Características generadas y guardadas en: {destino}")
    print(f"Shape final: {df_final.shape}")


if __name__ == "__main__":
    ejecutar()
