import os
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ==============================
#   RUTAS
# ==============================
DATA_DIR = "data/processed_windowed"
TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_dataset.csv")

RESULTS_DIR = "results_tuning"
MODELS_DIR = os.path.join(RESULTS_DIR, "best_models")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

COMPARISON_CSV = os.path.join(METRICS_DIR, "tuning_comparison.csv")


# ==============================
#   CARGA DE DATOS
# ==============================
def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Codificar etiquetas a números
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Clases: {list(le.classes_)}")  # Para verificar codificación

    return X_train, X_test, y_train_enc, y_test_enc, le


# ==============================
#   DEFINICIÓN DE MODELOS Y GRIDS
# ==============================
def get_models_and_grids():

    models = {
        "SVM": (
            SVC(probability=True),
            {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        ),

        "RandomForest": (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        ),

        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        ),

        "KNN": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "p": [1, 2]   # Manhattan vs Euclidiana
            }
        )
    }

    return models


# ==============================
#   TUNING AUTOMÁTICO
# ==============================
def tune_and_train_models(X_train, y_train, X_test, y_test):
    models = get_models_and_grids()
    results = []

    for model_name, (model, grid) in models.items():
        print("\n" + "="*60)
        print(f"TUNING: {model_name}")
        print("="*60)

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"\n🏆 Mejor modelo para {model_name}:")
        print(best_params)

        # ---- Evaluar con cross-validation en TRAIN para métricas CV ----
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # ---- Evaluar en TEST ----
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy Test {model_name}: {test_accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Guardar modelo
        model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pkl")
        joblib.dump(best_model, model_path)
        print(f"Modelo guardado en: {model_path}")

        # Guardar resultados
        results.append({
            "Model": model_name,
            "CV_Accuracy_Mean": cv_mean,
            "CV_Accuracy_Std": cv_std,
            "Test_Accuracy": test_accuracy
        })

    return results


# ==============================
#   GUARDAR RESULTADOS COMPARATIVOS
# ==============================
def save_comparison(results):
    df = pd.DataFrame(results)
    df = df[["Model", "CV_Accuracy_Mean", "CV_Accuracy_Std", "Test_Accuracy"]]
    df.to_csv(COMPARISON_CSV, index=False)
    print("\nResultados comparativos guardados en:")
    print(COMPARISON_CSV)
    print(df)


# ==============================
#   MAIN
# ==============================
def main():
    print("Cargando datos...")
    X_train, X_test, y_train, y_test, le = load_data()

    print("\nIniciando model tuning...")
    results = tune_and_train_models(X_train, y_train, X_test, y_test)

    print("\nGuardando comparación final...")
    save_comparison(results)

    print("\nTUNING COMPLETO")


if __name__ == "__main__":
    main()
