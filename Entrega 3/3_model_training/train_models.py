"""
Script 3: Entrenamiento de Modelos
Entrena múltiples modelos de clasificación con GridSearchCV.
"""
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
    """Carga el dataset de características."""
    print(f"Cargando datos desde: {ruta_csv}")
    
    try:
        df = pd.read_csv(ruta_csv)
        print(f"Dataset cargado: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: No se encuentra el archivo {ruta_csv}")
        print("Ejecuta primero: python 2_feature_engineering/compute_features.py")
        sys.exit(1)


def preparar_datos(df):
    """Separa características y etiquetas, codifica las etiquetas."""
    cols_excluir = ['action', 'video_filename', 'frame_idx']
    X = df.drop(columns=cols_excluir, errors='ignore')
    y = df['action']
    
    print(f"\nDimensiones de características: {X.shape}")
    print(f"\nDistribución de clases:")
    print(y.value_counts())
    print(f"\nTotal de muestras: {len(y)}")
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nClases codificadas:")
    for idx, clase in enumerate(le.classes_):
        print(f"  {idx}: {clase}")
    
    return X, y_encoded, le


def entrenar_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    """Entrena Random Forest con GridSearchCV."""
    print("\n" + "="*70)
    print("ENTRENANDO RANDOM FOREST")
    print("="*70 + "\n")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=random_state)),
        ('clf', RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])
    
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 5]
    }
    
    print("Configuración GridSearchCV:")
    print(f"  Parámetros a probar: {len(param_grid['clf__n_estimators']) * len(param_grid['clf__max_depth']) * len(param_grid['clf__min_samples_split'])} combinaciones")
    print(f"  CV folds: 3")
    print(f"  Scoring: accuracy\n")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor score CV: {grid_search.best_score_:.4f}")
    
    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"ACCURACY EN TEST: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*70}")
    
    return grid_search.best_estimator_, y_pred, grid_search.best_params_


def entrenar_xgboost(X_train, y_train, X_test, y_test, random_state=42):
    """Entrena XGBoost con GridSearchCV."""
    print("\n" + "="*70)
    print("ENTRENANDO XGBOOST")
    print("="*70 + "\n")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=random_state)),
        ('clf', xgb.XGBClassifier(random_state=random_state, eval_metric='mlogloss'))
    ])
    
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [5, 10],
        'clf__learning_rate': [0.01, 0.1]
    }
    
    print("Configuración GridSearchCV:")
    print(f"  Parámetros a probar: {len(param_grid['clf__n_estimators']) * len(param_grid['clf__max_depth']) * len(param_grid['clf__learning_rate'])} combinaciones")
    print(f"  CV folds: 3")
    print(f"  Scoring: accuracy\n")
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor score CV: {grid_search.best_score_:.4f}")
    
    y_pred = grid_search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"ACCURACY EN TEST: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*70}")
    
    return grid_search.best_estimator_, y_pred, grid_search.best_params_


def guardar_modelo(modelo, nombre, directorio="./assets"):
    """Guarda un modelo entrenado."""
    Path(directorio).mkdir(exist_ok=True)
    ruta = Path(directorio) / nombre
    joblib.dump(modelo, ruta)
    print(f"✓ Modelo guardado: {ruta}")
    return ruta


def guardar_confusion_matrix(y_test, y_pred, label_encoder, nombre, directorio="./results/evaluations"):
    """Guarda la matriz de confusión como imagen."""
    Path(directorio).mkdir(parents=True, exist_ok=True)
    
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
    plt.title(f'Matriz de Confusión - {nombre}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    ruta = Path(directorio) / f"{nombre.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(ruta, dpi=300)
    plt.close()
    
    print(f"✓ Matriz de confusión guardada: {ruta}")


def main():
    """Función principal de entrenamiento."""
    print("\n" + "="*70)
    print("PASO 3: ENTRENAMIENTO DE MODELOS")
    print("="*70 + "\n")
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    
    # Cargar datos
    ruta_csv = "./data/processed/model_features.csv"
    df = cargar_datos(ruta_csv)
    
    # Preparar datos
    X, y, label_encoder = preparar_datos(df)
    
    # Split train/test
    print(f"\nDividiendo datos: {int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Guardar LabelEncoder
    le_path = guardar_modelo(label_encoder, "label_encoder.joblib")
    
    # Entrenar Random Forest
    rf_model, rf_pred, rf_params = entrenar_random_forest(
        X_train, y_train, X_test, y_test, RANDOM_STATE
    )
    
    print(f"\nClassification Report - Random Forest:")
    print(classification_report(y_test, rf_pred, target_names=label_encoder.classes_))
    
    rf_path = guardar_modelo(rf_model, "best_random_forest_model.joblib")
    guardar_confusion_matrix(y_test, rf_pred, label_encoder, "Random Forest")
    
    # Entrenar XGBoost
    xgb_model, xgb_pred, xgb_params = entrenar_xgboost(
        X_train, y_train, X_test, y_test, RANDOM_STATE
    )
    
    print(f"\nClassification Report - XGBoost:")
    print(classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))
    
    xgb_path = guardar_modelo(xgb_model, "best_xgboost_model.joblib")
    guardar_confusion_matrix(y_test, xgb_pred, label_encoder, "XGBoost")
    
    # Comparación final
    rf_acc = accuracy_score(y_test, rf_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    print(f"\nRandom Forest:")
    print(f"  Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"  Parámetros: {rf_params}")
    
    print(f"\nXGBoost:")
    print(f"  Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
    print(f"  Parámetros: {xgb_params}")
    
    print(f"\nMejor modelo: {'Random Forest' if rf_acc > xgb_acc else 'XGBoost'}")
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
