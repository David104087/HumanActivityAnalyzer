"""
run_full_pipeline.py

Script maestro que ejecuta todo el pipeline desde features hasta modelo entrenado.
Útil para re-entrenar rápidamente cuando tienes nuevos datos.

Pasos que ejecuta:
1. Compute features (si no existen)
2. Create windowed dataset
3. Preprocess and split
4. Train models
5. Copy best model to assets/

Uso:
  python run_full_pipeline.py
  
  # O con parámetros personalizados:
  python run_full_pipeline.py --window_size 7 --skip_features
"""

import os
import sys
import argparse
import subprocess


def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado"""
    print("\n" + "="*80)
    print(f"PASO: {description}")
    print("="*80)
    print(f"Comando: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error ejecutando: {description}")
        sys.exit(1)
    
    print(f"\n✓ Completado: {description}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=5, help='Tamaño de ventana deslizante')
    parser.add_argument('--skip_features', action='store_true', help='Saltar cálculo de features (si ya existen)')
    parser.add_argument('--skip_training', action='store_true', help='Solo preparar datos, no entrenar')
    
    args = parser.parse_args()
    
    # Cambiar al directorio de Entrega 3
    os.chdir('Entrega 3')
    
    # Paso 1: Compute features (opcional)
    if not args.skip_features:
        run_command(
            'python "2_feature_engineering/02_compute_features.py" --batch',
            "Cálculo de features por frame"
        )
    else:
        print("\n⊗ Saltando cálculo de features (--skip_features)")
    
    # Paso 2: Create windowed dataset
    run_command(
        f'python "2_feature_engineering/04_create_window_dataset.py" --window_size {args.window_size}',
        "Creación de dataset con ventanas deslizantes"
    )
    
    # Paso 3: Preprocess and split
    run_command(
        'python "3_model_training/05_preprocess_train_split.py"',
        "Preprocesamiento y split de datos"
    )
    
    # Paso 4: Train models (opcional)
    if not args.skip_training:
        run_command(
            'python "3_model_training/06_train_models.py"',
            "Entrenamiento de modelos"
        )
        
        # Paso 5: Copy best model to assets
        print("\n" + "="*80)
        print("COPIANDO MODELOS A ASSETS")
        print("="*80)
        
        os.makedirs('assets', exist_ok=True)
        
        # Copiar scaler y label encoder
        subprocess.run('cp data/processed_windowed/scaler.pkl assets/', shell=True)
        subprocess.run('cp data/processed_windowed/label_encoder.pkl assets/', shell=True)
        
        # Copiar el mejor modelo (por defecto randomforest)
        subprocess.run('cp results/models/randomforest.pkl assets/', shell=True)
        
        print("\n✓ Modelos copiados a assets/")
        print("  - assets/randomforest.pkl")
        print("  - assets/scaler.pkl")
        print("  - assets/label_encoder.pkl")
    else:
        print("\n⊗ Saltando entrenamiento (--skip_training)")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETADO")
    print("="*80)
    print("\nPróximos pasos:")
    print("  1. Revisa results/reports/training_report.txt para ver métricas")
    print("  2. Ejecuta: python run_realtime.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
