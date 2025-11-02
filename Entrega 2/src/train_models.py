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
warnings.filterwarnings('ignore')

def load_data():
    """
    Carga los datasets de entrenamiento y prueba
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data', 'processed', 'train_dataset.csv')
    test_path = os.path.join(base_dir, 'data', 'processed', 'test_dataset.csv')
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separar características y etiquetas
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Guardar el codificador para uso futuro
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'models')
    os.makedirs(models_dir, exist_ok=True)
    label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    joblib.dump(le, label_encoder_path)
    
    return X_train, X_test, y_train_encoded, y_test_encoded, le.classes_

def train_model(model, X_train, y_train, model_name):
    """
    Entrena un modelo usando validación cruzada y retorna las métricas
    """
    # Configurar validación cruzada con 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Realizar validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    
    print(f"\nResultados de validación cruzada para {model_name}:")
    print(f"Accuracy promedio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Entrenar el modelo final con todos los datos de entrenamiento
    model.fit(X_train, y_train)
    
    return model, cv_scores

def evaluate_model(model, X_test, y_test, class_names, model_name):
    """
    Evalúa el modelo en el conjunto de prueba
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    
    print(f"\nResultados de evaluación para {model_name}:")
    print(f"Accuracy en test: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(class_report)
    
    return accuracy, class_report

def save_model(model, model_name):
    """
    Guarda el modelo entrenado
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"\nModelo guardado en: {model_path}")

def main():
    print("Cargando datos...")
    X_train, X_test, y_train, y_test, class_names = load_data()
    print(f"Datos cargados: {len(X_train)} muestras de entrenamiento, {len(X_test)} muestras de prueba")
    
    # Definir modelos
    models = {
        'SVM': SVC(kernel='rbf', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    # Entrenar y evaluar cada modelo
    results = {}
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Entrenando {model_name}...")
        
        # Entrenar modelo con validación cruzada
        trained_model, cv_scores = train_model(model, X_train, y_train, model_name)
        
        # Evaluar en conjunto de prueba
        test_accuracy, class_report = evaluate_model(trained_model, X_test, y_test, class_names, model_name)
        
        # Guardar modelo
        save_model(trained_model, model_name.lower())
        
        # Almacenar resultados
        results[model_name] = {
            'cv_scores': cv_scores,
            'test_accuracy': test_accuracy,
            'classification_report': class_report
        }
    
    # Guardar resultados en archivo
    print("\nGuardando resultados...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Crear DataFrame con resultados comparativos
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
    comparison_path = os.path.join(metrics_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    print("\nComparación final de modelos:")
    print(comparison_df.to_string(index=False))
    print(f"\nResultados guardados en: {comparison_path}")

if __name__ == "__main__":
    main()