"""
Script 3B: Evaluación Detallada de Modelos
Genera métricas adicionales y análisis de rendimiento.
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def cargar_modelo_y_datos():
    """Carga modelo entrenado y datos de prueba."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Cargar datos
    df = pd.read_csv("./data/processed/model_features.csv")
    
    cols_excluir = ['action', 'video_filename', 'frame_idx']
    X = df.drop(columns=cols_excluir, errors='ignore')
    y = df['action']
    
    # Codificar etiquetas
    le = joblib.load("./assets/label_encoder.joblib")
    y_encoded = le.transform(y)
    
    # Split (mismo que en entrenamiento)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )
    
    return X_test, y_test, le


def evaluar_modelo(nombre_modelo, X_test, y_test, label_encoder):
    """Evalúa un modelo específico."""
    print(f"\n{'='*70}")
    print(f"EVALUANDO: {nombre_modelo}")
    print(f"{'='*70}\n")
    
    # Cargar modelo
    modelo = joblib.load(f"./assets/{nombre_modelo}")
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Métricas globales
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Métricas Globales:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report detallado
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatriz de Confusión:")
    print(cm)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred
    }


def comparar_modelos(resultados):
    """Compara múltiples modelos y genera visualización."""
    print(f"\n{'='*70}")
    print("COMPARACIÓN DE MODELOS")
    print(f"{'='*70}\n")
    
    # Crear DataFrame de comparación
    data = []
    for nombre, metricas in resultados.items():
        data.append({
            'Modelo': nombre.replace('_', ' ').title(),
            'Accuracy': metricas['accuracy'],
            'Precision': metricas['precision'],
            'Recall': metricas['recall'],
            'F1-Score': metricas['f1_score']
        })
    
    df_comp = pd.DataFrame(data)
    print(df_comp.to_string(index=False))
    
    # Visualización
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_comp))
    width = 0.2
    
    ax.bar(x - 1.5*width, df_comp['Accuracy'], width, label='Accuracy')
    ax.bar(x - 0.5*width, df_comp['Precision'], width, label='Precision')
    ax.bar(x + 0.5*width, df_comp['Recall'], width, label='Recall')
    ax.bar(x + 1.5*width, df_comp['F1-Score'], width, label='F1-Score')
    
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Score')
    ax.set_title('Comparación de Modelos')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp['Modelo'])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./results/evaluations/models_comparison.png", dpi=300)
    plt.close()
    
    print(f"\n✓ Gráfica de comparación guardada: ./results/evaluations/models_comparison.png")


def analizar_errores(y_test, y_pred, label_encoder):
    """Analiza los errores de clasificación."""
    print(f"\n{'='*70}")
    print("ANÁLISIS DE ERRORES")
    print(f"{'='*70}\n")
    
    errores = y_test != y_pred
    num_errores = errores.sum()
    
    print(f"Total de errores: {num_errores} de {len(y_test)} ({num_errores/len(y_test)*100:.2f}%)")
    
    # Errores por clase
    print(f"\nErrores por clase:")
    for i, clase in enumerate(label_encoder.classes_):
        mascara_clase = y_test == i
        errores_clase = errores[mascara_clase].sum()
        total_clase = mascara_clase.sum()
        
        if total_clase > 0:
            tasa_error = errores_clase / total_clase * 100
            print(f"  {clase}: {errores_clase}/{total_clase} ({tasa_error:.2f}%)")
    
    # Confusiones más comunes
    cm = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm, 0)  # Ignorar diagonal (aciertos)
    
    print(f"\nTop 5 confusiones más comunes:")
    confusiones = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                confusiones.append((
                    label_encoder.classes_[i],
                    label_encoder.classes_[j],
                    cm[i, j]
                ))
    
    confusiones.sort(key=lambda x: x[2], reverse=True)
    
    for i, (real, pred, count) in enumerate(confusiones[:5], 1):
        print(f"  {i}. {real} → {pred}: {count} veces")


def main():
    """Función principal de evaluación."""
    print("\n" + "="*70)
    print("PASO 3B: EVALUACIÓN DETALLADA DE MODELOS")
    print("="*70)
    
    # Cargar datos de prueba
    print("\nCargando datos de prueba...")
    X_test, y_test, label_encoder = cargar_modelo_y_datos()
    print(f"Muestras de prueba: {len(y_test)}")
    
    # Evaluar modelos
    modelos = {
        'random_forest': 'best_random_forest_model.joblib',
        'xgboost': 'best_xgboost_model.joblib'
    }
    
    resultados = {}
    
    for nombre, archivo in modelos.items():
        resultados[nombre] = evaluar_modelo(archivo, X_test, y_test, label_encoder)
    
    # Comparar modelos
    comparar_modelos(resultados)
    
    # Análisis de errores del mejor modelo
    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DETALLADO DEL MEJOR MODELO: {mejor_modelo[0].upper()}")
    print(f"{'='*70}")
    
    analizar_errores(y_test, mejor_modelo[1]['y_pred'], label_encoder)
    
    print("\n" + "="*70)
    print("EVALUACIÓN COMPLETADA")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
