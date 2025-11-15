# 3_model_training/06_train_models.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- Configuración de Rutas ---
DATA_DIR = "data/processed_windowed"
TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_dataset.csv")

RESULTS_DIR = "results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
COMPARISON_PATH = os.path.join(METRICS_DIR, "model_comparison.csv")
# --- Fin Configuración ---

def load_data():
    """
    Carga los datasets de entrenamiento y prueba (con 30 features)
    """
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)
    
    # Separar características y etiquetas
    X_train = train_data.drop('label', axis=1)
    y_train_raw = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test_raw = test_data['label']
    
    # Codificar etiquetas (ej. 'Sit' -> 0)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    # Guardar el codificador para uso en tiempo real
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"LabelEncoder guardado en: {LABEL_ENCODER_PATH}")
    
    return X_train, X_test, y_train, y_test, le.classes_

def train_model(model, X_train, y_train, model_name):
    """
    Entrena un modelo usando validación cruzada (CV)
    """
    # Usamos 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Iniciando validación cruzada (5-fold) para {model_name}...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    
    print(f"Resultados de CV para {model_name}:")
    print(f"Accuracy promedio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Entrenar el modelo final con TODOS los datos de entrenamiento
    print(f"Entrenando modelo final de {model_name}...")
    model.fit(X_train, y_train)
    
    return model, cv_scores

def evaluate_model(model, X_test, y_test, class_names, model_name):
    """
    Evalúa el modelo en el conjunto de prueba
    """
    print(f"Evaluando {model_name} en el conjunto de prueba...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    
    print(f"\nResultados de Evaluación para {model_name}:")
    print(f"Accuracy en Test: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(class_report)
    
    return accuracy, class_report

def save_model(model, model_name):
    """
    Guarda el modelo entrenado
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"\nModelo guardado en: {model_path}")

def main():
    print("Cargando datos (train/test)...")
    X_train, X_test, y_train, y_test, class_names = load_data()
    print(f"Datos cargados: {len(X_train)} muestras de entrenamiento, {len(X_test)} muestras de prueba")
    
    # Definir los modelos con parámetros por defecto
    # (El entrenamiento con SVM puede ser lento)
    models = {
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    
    # Entrenar y evaluar cada modelo
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Procesando Modelo: {model_name}")
        
        # 1. Entrenar
        trained_model, cv_scores = train_model(model, X_train, y_train, model_name)
        
        # 2. Evaluar
        test_accuracy, class_report = evaluate_model(trained_model, X_test, y_test, class_names, model_name)
        
        # 3. Guardar
        save_model(trained_model, model_name.lower())
        
        results[model_name] = {
            'cv_scores': cv_scores,
            'test_accuracy': test_accuracy,
            'classification_report': class_report
        }
    
    # Guardar resultados comparativos
    print("\nGuardando resultados de comparación...")
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    comparison_data = {
        'Model': [],
        'CV_Accuracy_Mean': [],
        'CV_Accuracy_Std': [],
        'Test_Accuracy': []
    }
    
    for model_name, result in results.items():
        comparison_data['Model'].append(model_name)
        comparison_data['CV_Accuracy_Mean'].append(result['cv_scores'].mean())
        comparison_data['CV_Accuracy_Std'].append(result['cv_scores'].std())
        comparison_data['Test_Accuracy'].append(result['test_accuracy'])
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    
    print("\n--- ¡Entrenamiento Completado! ---")
    print("Comparación final de modelos:")
    print(comparison_df.to_string(index=False))
    print(f"\nReporte de comparación guardado en: {COMPARISON_PATH}")
    print(f"Modelos guardados en: {MODELS_DIR}")

if __name__ == "__main__":
    main()