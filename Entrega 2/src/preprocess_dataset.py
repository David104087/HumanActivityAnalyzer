import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_dataset(dataset_path):
    """
    Carga el dataset combinado
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {dataset_path}")
    
    return pd.read_csv(dataset_path)

def normalize_features(df, feature_columns):
    """
    Normaliza las características numéricas usando StandardScaler
    """
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

def handle_missing_values(df):
    """
    Maneja los valores faltantes en el dataset
    """
    # Verificar si hay valores faltantes
    missing_values = df.isnull().sum()
    
    if missing_values.any():
        print("\nValores faltantes encontrados:")
        print(missing_values[missing_values > 0])
        
        # Para características numéricas, llenar con la media
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    else:
        print("\nNo se encontraron valores faltantes")
    
    return df

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
    # Directorios
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'processed', 'combined_dataset.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Cargar dataset
    print("Cargando dataset...")
    df = load_dataset(input_path)
    print(f"Dataset cargado: {len(df)} muestras")
    
    # Separar características y etiquetas
    feature_columns = ['knee_left', 'knee_right', 'hip_left', 'hip_right', 
                      'trunk_angle', 'motion_energy']
    X = df[feature_columns]
    y = df['label']
    
    # Manejar valores faltantes
    print("\nVerificando valores faltantes...")
    X = handle_missing_values(X)
    
    # Normalizar características
    print("\nNormalizando características...")
    X_normalized, scaler = normalize_features(X, feature_columns)
    
    # Balancear clases
    print("\nBalanceando clases...")
    X_balanced, y_balanced = balance_classes(X_normalized, y)
    
    # Dividir en train/test
    print("\nDividiendo en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = split_dataset(X_balanced, y_balanced)
    
    # Guardar conjuntos procesados
    train_data = pd.DataFrame(X_train, columns=feature_columns)
    train_data['label'] = y_train
    test_data = pd.DataFrame(X_test, columns=feature_columns)
    test_data['label'] = y_test
    
    train_path = os.path.join(output_dir, 'train_dataset.csv')
    test_path = os.path.join(output_dir, 'test_dataset.csv')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    print("\nGuardando datasets procesados...")
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    # Guardar el scaler para uso futuro
    import joblib
    joblib.dump(scaler, scaler_path)
    
    print(f"\nDatasets guardados en:")
    print(f"- Train: {train_path}")
    print(f"- Test: {test_path}")
    print(f"- Scaler: {scaler_path}")
    
    print("\nEstadísticas finales:")
    print(f"- Muestras de entrenamiento: {len(train_data)}")
    print(f"- Muestras de prueba: {len(test_data)}")
    print("\nDistribución de clases en conjunto de entrenamiento:")
    print(train_data['label'].value_counts())
    print("\nDistribución de clases en conjunto de prueba:")
    print(test_data['label'].value_counts())

if __name__ == "__main__":
    main()