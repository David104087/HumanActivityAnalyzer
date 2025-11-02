import os
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
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
    
    return X_train, X_test, y_train_encoded, y_test_encoded, le.classes_

def optimize_svm(X_train, y_train):
    """
    Optimiza hiperparámetros para SVM
    """
    print("\nOptimizando SVM...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    
    svm = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor accuracy en CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_random_forest(X_train, y_train):
    """
    Optimiza hiperparámetros para Random Forest
    """
    print("\nOptimizando Random Forest...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor accuracy en CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_xgboost(X_train, y_train):
    """
    Optimiza hiperparámetros para XGBoost
    """
    print("\nOptimizando XGBoost...")
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor accuracy en CV: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, class_names, model_name):
    """
    Evalúa el modelo optimizado en el conjunto de prueba
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResultados de evaluación para {model_name}:")
    print(f"Accuracy en test: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return accuracy, classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

def save_model(model, model_name):
    """
    Guarda el modelo optimizado
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'results', 'models_optimized')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    joblib.dump(model, model_path)
    print(f"\nModelo guardado en: {model_path}")

def main():
    print("Cargando datos...")
    X_train, X_test, y_train, y_test, class_names = load_data()
    print(f"Datos cargados: {len(X_train)} muestras de entrenamiento, {len(X_test)} muestras de prueba")
    
    # Optimizar y evaluar cada modelo
    results = {}
    
    # SVM
    print("\n" + "="*50)
    svm_model = optimize_svm(X_train, y_train)
    accuracy, report = evaluate_model(svm_model, X_test, y_test, class_names, "SVM")
    save_model(svm_model, "svm_optimized")
    results['SVM'] = {'accuracy': accuracy, 'report': report}
    
    # Random Forest
    print("\n" + "="*50)
    rf_model = optimize_random_forest(X_train, y_train)
    accuracy, report = evaluate_model(rf_model, X_test, y_test, class_names, "RandomForest")
    save_model(rf_model, "randomforest_optimized")
    results['RandomForest'] = {'accuracy': accuracy, 'report': report}
    
    # XGBoost
    print("\n" + "="*50)
    xgb_model = optimize_xgboost(X_train, y_train)
    accuracy, report = evaluate_model(xgb_model, X_test, y_test, class_names, "XGBoost")
    save_model(xgb_model, "xgboost_optimized")
    results['XGBoost'] = {'accuracy': accuracy, 'report': report}
    
    # Guardar resultados en archivo
    print("\nGuardando resultados...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Crear DataFrame con resultados comparativos
    comparison_data = {
        'Model': [],
        'Best_Parameters': [],
        'Test_Accuracy': [],
        'Macro_Avg_F1': []
    }
    
    for model_name, result in results.items():
        comparison_data['Model'].append(model_name)
        comparison_data['Test_Accuracy'].append(result['accuracy'])
        comparison_data['Macro_Avg_F1'].append(result['report']['macro avg']['f1-score'])
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(metrics_dir, 'optimized_model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    print("\nComparación final de modelos optimizados:")
    print(comparison_df.to_string(index=False))
    print(f"\nResultados guardados en: {comparison_path}")

if __name__ == "__main__":
    main()