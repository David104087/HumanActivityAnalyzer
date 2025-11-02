import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

plt.style.use('seaborn')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_data_and_models():
    """
    Carga los datos de prueba y los modelos entrenados
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Cargar datos de prueba
    test_path = os.path.join(base_dir, 'data', 'processed', 'test_dataset.csv')
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Cargar modelos y label encoder
    models_dir = os.path.join(base_dir, 'results', 'models')
    
    # Cargar label encoder primero
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
    # Codificar etiquetas
    y_test = le.transform(y_test)
    
    # Cargar modelos
    models = {}
    for model_name in ['svm', 'randomforest', 'xgboost']:
        model_path = os.path.join(models_dir, f'{model_name}.pkl')
        models[model_name] = joblib.load(model_path)
    
    # Cargar label encoder
    le = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
    return X_test, y_test, models, le

def plot_model_comparison(save_dir):
    """
    Genera gráfica comparativa del rendimiento de los modelos
    """
    results_path = os.path.join(os.path.dirname(save_dir), 'metrics', 'model_comparison.csv')
    results = pd.read_csv(results_path)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results['Model']))
    width = 0.35
    
    plt.bar(x - width/2, results['CV_Accuracy_Mean'], width, label='CV Accuracy', yerr=results['CV_Accuracy_Std']*2)
    plt.bar(x + width/2, results['Test_Accuracy'], width, label='Test Accuracy')
    
    plt.xlabel('Modelos')
    plt.ylabel('Accuracy')
    plt.title('Comparación de Rendimiento entre Modelos')
    plt.xticks(x, results['Model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    """
    Genera matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión Normalizada - Random Forest')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_feature_importance(model, feature_names, save_dir):
    """
    Genera gráfica de importancia de características
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importances = importances.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importances)), importances['importance'])
    plt.yticks(range(len(importances)), importances['feature'])
    plt.xlabel('Importancia')
    plt.title('Importancia de Características - Random Forest')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def plot_roc_curves(X_test, y_test, model, classes, save_dir):
    """
    Genera curvas ROC para cada clase
    """
    # Crear una figura más grande para acomodar mejor las curvas
    plt.figure(figsize=(15, 10))
    
    # Convertir las etiquetas numéricas a one-hot encoding
    n_classes = len(classes)
    y_test_bin = np.eye(n_classes)[y_test]
    
    # Obtener las probabilidades de predicción
    y_score = model.predict_proba(X_test)
    
    # Calcular ROC para cada clase
    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    # Añadir la línea diagonal de referencia
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Configurar el gráfico
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC para cada clase - Random Forest', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Ajustar los límites y márgenes
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_class_metrics(y_true, y_pred, classes, save_dir):
    """
    Genera gráfica comparativa de métricas por clase
    """
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }, index=classes)
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', width=0.8)
    plt.title('Métricas por Clase - Random Forest')
    plt.xlabel('Clase')
    plt.ylabel('Puntaje')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_metrics.png'))
    plt.close()

def main():
    # Crear directorio para visualizaciones
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    viz_dir = os.path.join(base_dir, 'results', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Cargando datos y modelos...")
    X_test, y_test, models, le = load_data_and_models()
    
    # Generar predicciones con Random Forest (mejor modelo)
    rf_model = models['randomforest']
    y_pred = rf_model.predict(X_test)
    classes = le.classes_
    
    print("\nGenerando visualizaciones...")
    
    # 1. Comparación de modelos
    plot_model_comparison(viz_dir)
    print("- Comparación de modelos generada")
    
    # 2. Matriz de confusión
    plot_confusion_matrix(y_test, y_pred, classes, viz_dir)
    print("- Matriz de confusión generada")
    
    # 3. Importancia de características
    plot_feature_importance(rf_model, X_test.columns, viz_dir)
    print("- Importancia de características generada")
    
    # 4. Curvas ROC
    plot_roc_curves(X_test, y_test, rf_model, classes, viz_dir)
    print("- Curvas ROC generadas")
    
    # 5. Métricas por clase
    plot_class_metrics(y_test, y_pred, classes, viz_dir)
    print("- Métricas por clase generadas")
    
    print(f"\nVisualizaciones guardadas en: {viz_dir}")

if __name__ == "__main__":
    main()