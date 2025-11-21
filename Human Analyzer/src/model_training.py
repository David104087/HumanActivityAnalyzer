import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def cargar_datos(ruta_csv):
    """Carga el dataset de características desde CSV."""
    print(f"Cargando datos desde {ruta_csv}...")
    try:
        df = pd.read_csv(ruta_csv)
        print(f"Dataset cargado: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: No se encuentra el archivo {ruta_csv}")
        print("Asegúrate de ejecutar primero get_landmarks.py y feature_engineering.py")
        sys.exit(1)


def preparar_datos(df):
    """Separa características y etiquetas, codifica las etiquetas."""
    # Eliminar columnas no numéricas
    cols_excluir = ['action', 'video_filename', 'frame_idx']
    X = df.drop(columns=cols_excluir, errors='ignore')
    y = df['action']
    
    print(f"\nDimensiones de características: {X.shape}")
    print(f"Distribución de clases:")
    print(y.value_counts())
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nClases codificadas: {dict(enumerate(le.classes_))}")
    
    return X, y_encoded, le


def entrenar_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    """Entrena Random Forest con GridSearchCV."""
    print("\n" + "="*60)
    print("ENTRENANDO RANDOM FOREST")
    print("="*60)
    
    # Pipeline: Scaler -> PCA -> Clasificador
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=random_state)),
        ('clf', RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])
    
    # GridSearch
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 5]
    }
    
    print("\nBuscando mejores hiperparámetros...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score CV: {grid_search.best_score_:.4f}")
    
    # Evaluar en test
    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"ACCURACY EN TEST: {acc:.4f}")
    print(f"{'='*60}")
    
    return grid_search.best_estimator_, y_pred


def entrenar_xgboost(X_train, y_train, X_test, y_test, random_state=42):
    """Entrena XGBoost con GridSearchCV."""
    print("\n" + "="*60)
    print("ENTRENANDO XGBOOST")
    print("="*60)
    
    # Pipeline: Scaler -> PCA -> Clasificador
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=random_state)),
        ('clf', xgb.XGBClassifier(random_state=random_state, eval_metric='mlogloss'))
    ])
    
    # GridSearch
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [5, 10],
        'clf__learning_rate': [0.01, 0.1]
    }
    
    print("\nBuscando mejores hiperparámetros...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score CV: {grid_search.best_score_:.4f}")
    
    # Evaluar en test
    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"ACCURACY EN TEST: {acc:.4f}")
    print(f"{'='*60}")
    
    return grid_search.best_estimator_, y_pred


def mostrar_evaluacion(y_test, y_pred, label_encoder, titulo):
    """Muestra métricas de evaluación y matriz de confusión."""
    print(f"\n{'='*60}")
    print(f"{titulo}")
    print(f"{'='*60}\n")
    
    # Classification report
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Matriz de Confusión - {titulo}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    # Guardar figura
    nombre_archivo = titulo.lower().replace(' ', '_') + '_confusion_matrix.png'
    plt.savefig(f'./models/{nombre_archivo}')
    print(f"\nMatriz de confusión guardada en: ./models/{nombre_archivo}")
    plt.close()


def main():
    """Función principal de entrenamiento."""
    print("\n" + "="*60)
    print("SISTEMA DE ENTRENAMIENTO DE MODELOS")
    print("Reconocimiento de Actividad Humana")
    print("="*60 + "\n")
    
    # Configuración
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    
    # Crear directorio de modelos
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # 1. Cargar datos
    ruta_csv = "./data/processed/model_features.csv"
    df = cargar_datos(ruta_csv)
    
    # 2. Preparar datos
    X, y, label_encoder = preparar_datos(df)
    
    # 3. Split train/test estratificado
    print(f"\nDividiendo datos: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 4. Guardar LabelEncoder
    le_path = models_dir / "label_encoder.joblib"
    joblib.dump(label_encoder, le_path)
    print(f"\nLabelEncoder guardado en: {le_path}")
    
    # 5. Entrenar Random Forest
    rf_model, rf_pred = entrenar_random_forest(X_train, y_train, X_test, y_test, RANDOM_STATE)
    mostrar_evaluacion(y_test, rf_pred, label_encoder, "RANDOM FOREST")
    
    # Guardar modelo Random Forest
    rf_path = models_dir / "best_random_forest_model.joblib"
    joblib.dump(rf_model, rf_path)
    print(f"\nModelo Random Forest guardado en: {rf_path}")
    
    # 6. Entrenar XGBoost
    xgb_model, xgb_pred = entrenar_xgboost(X_train, y_train, X_test, y_test, RANDOM_STATE)
    mostrar_evaluacion(y_test, xgb_pred, label_encoder, "XGBOOST")
    
    # Guardar modelo XGBoost
    xgb_path = models_dir / "best_xgboost_model.joblib"
    joblib.dump(xgb_model, xgb_path)
    print(f"\nModelo XGBoost guardado en: {xgb_path}")
    
    # 7. Comparación final
    rf_acc = accuracy_score(y_test, rf_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"XGBoost Accuracy:       {xgb_acc:.4f}")
    print(f"\nMejor modelo: {'Random Forest' if rf_acc > xgb_acc else 'XGBoost'}")
    print("="*60 + "\n")
    
    print("Entrenamiento completado exitosamente.")
    print("Los modelos están listos para usar en main.py")


if __name__ == "__main__":
    main()
