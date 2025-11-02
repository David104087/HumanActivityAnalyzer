import os
import subprocess
import time
from datetime import datetime

def run_script(script_path, description):
    """
    Ejecuta un script de Python y maneja los errores
    """
    print(f"\n{'='*80}")
    print(f"Ejecutando: {description}")
    print(f"Script: {script_path}")
    print(f"Tiempo de inicio: {datetime.now().strftime('%H:%M:%S')}")
    print('='*80)
    
    try:
        result = subprocess.run(['python', script_path], check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Advertencias:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar {script_path}:")
        print(e.stdout)
        print("Error:", e.stderr)
        return False

def main():
    # Obtener el directorio base
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'src')
    
    # Lista de scripts a ejecutar en orden
    scripts = [
        {
            'name': 'prepare_dataset.py',
            'description': 'Preparación inicial del dataset'
        },
        {
            'name': 'preprocess_dataset.py',
            'description': 'Preprocesamiento y balance de datos'
        },
        {
            'name': 'optimize_models.py',
            'description': 'Optimización de hiperparámetros'
        },
        {
            'name': 'train_models.py',
            'description': 'Entrenamiento de modelos'
        },
        {
            'name': 'visualize_results.py',
            'description': 'Generación de visualizaciones'
        }
    ]
    
    # Tiempo de inicio
    start_time = time.time()
    print("\nIniciando pipeline de entrenamiento")
    print(f"Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ejecutar cada script en orden
    for script in scripts:
        script_path = os.path.join(src_dir, script['name'])
        
        if not os.path.exists(script_path):
            print(f"\nError: No se encontró el script {script['name']}")
            continue
        
        success = run_script(script_path, script['description'])
        
        if not success:
            print(f"\nError en la ejecución de {script['name']}. Deteniendo el pipeline.")
            break
    
    # Tiempo total
    end_time = time.time()
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print("\n" + "="*80)
    print("Resumen del pipeline:")
    print(f"Tiempo total de ejecución: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Fecha y hora de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()