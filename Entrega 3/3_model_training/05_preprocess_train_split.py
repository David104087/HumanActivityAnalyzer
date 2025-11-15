import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Configuración ---
INPUT_FILE = "data/processed_windowed/windowed_dataset.csv"
OUTPUT_DIR = "data/processed_windowed"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_dataset.csv")
SCALER_FILE = os.path.join(OUTPUT_DIR, "scaler.pkl") # ¡El nuevo scaler de 30 features!
# --- Fin Configuración ---

def load_dataset(dataset_path):
    """Carga el dataset de ventanas"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {dataset_path}")
    
    return pd.read_csv(dataset_path)

def normalize_features(X_train, X_test):
    """
    Normaliza las características numéricas (30 features)
    """
    # Usamos StandardScaler, como en tu pipeline original
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def handle_missing_values(X, y):
    """
    Maneja los valores faltantes (aunque el script 04 ya debería haberlos quitado)
    """
    if X.isnull().values.any():
        print("\nValores faltantes encontrados. Rellenando con la media...")
        X = X.fillna(X.mean())
    else:
        print("\nNo se encontraron valores faltantes.")
    
    return X

def balance_classes(X, y):
    """
    Balancea las clases usando SMOTE
    """
    print("\nDistribución de clases original:")
    print(y.value_counts())
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print("\nDistribución de clases después de SMOTE:")
    print(pd.Series(y_balanced).value_counts())
    
    return X_balanced, y_balanced

def split_dataset(X, y, test_size=0.2):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba
    """
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def main():
    # Cargar dataset de ventanas
    print(f"Cargando dataset desde: {INPUT_FILE}...")
    df = load_dataset(INPUT_FILE)
    print(f"Dataset cargado: {len(df)} muestras")
    
    # Separar características (X) y etiquetas (y)
    # Automáticamente toma todas las columnas (30) excepto 'label'
    X = df.drop('label', axis=1)
    y = df['label']
    
    print(f"Separando {len(X.columns)} features y 1 etiqueta.")
    
    # Manejar valores faltantes
    X = handle_missing_values(X, y)
    
    # Dividir en train/test (ANTES de balancear)
    print("\nDividiendo en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Balancear clases (SÓLO en el conjunto de entrenamiento)
    print("\nBalanceando clases (SMOTE) en el conjunto de entrenamiento...")
    X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)
    
    # Normalizar características (después de dividir y balancear)
    print("\nNormalizando características (StandardScaler)...")
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train_balanced, X_test)
    
    # Guardar conjuntos procesados
    print("\nGuardando datasets procesados y el scaler...")
    
    # Recrear DataFrames para guardar en CSV
    feature_columns = X.columns
    train_data = pd.DataFrame(X_train_scaled, columns=feature_columns)
    train_data['label'] = y_train_balanced
    
    test_data = pd.DataFrame(X_test_scaled, columns=feature_columns)
    test_data['label'] = y_test.values # y_test no fue balanceado
    
    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)
    
    # Guardar el scaler para uso en tiempo real
    joblib.dump(scaler, SCALER_FILE)
    
    print("\n--- ¡Preprocesamiento Completado! ---")
    print(f"Datasets guardados en:")
    print(f"- Train: {TRAIN_FILE} ({len(train_data)} muestras)")
    print(f"- Test: {TEST_FILE} ({len(test_data)} muestras)")
    print(f"- Scaler: {SCALER_FILE}")

if __name__ == "__main__":
    main()