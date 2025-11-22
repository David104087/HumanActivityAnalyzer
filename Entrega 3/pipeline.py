"""
Pipeline Completo - Sistema de Reconocimiento de Actividad Humana
Ejecuta todos los pasos del proyecto de forma secuencial.
"""
import sys
import subprocess
from pathlib import Path
import time


class Pipeline:
    """Clase para gestionar el pipeline completo del proyecto."""
    
    def __init__(self):
        """Inicializa el pipeline."""
        self.pasos = [
            {
                "numero": 1,
                "nombre": "Extracción de Landmarks",
                "script": "1_data_extraction/extract_landmarks.py",
                "args": ["--stride", "1"],
                "descripcion": "Procesa videos y extrae landmarks con MediaPipe"
            },
            {
                "numero": 2,
                "nombre": "Ingeniería de Características",
                "script": "2_feature_engineering/compute_features.py",
                "args": [],
                "descripcion": "Calcula ángulos biomecánicos y características temporales"
            },
            {
                "numero": 3,
                "nombre": "Análisis Postural",
                "script": "2_feature_engineering/postural_analysis.py",
                "args": [],
                "descripcion": "Analiza inclinaciones y métricas posturales",
                "opcional": True
            },
            {
                "numero": 4,
                "nombre": "Entrenamiento de Modelos",
                "script": "3_model_training/train_models.py",
                "args": [],
                "descripcion": "Entrena Random Forest y XGBoost con GridSearchCV"
            },
            {
                "numero": 5,
                "nombre": "Evaluación de Modelos",
                "script": "3_model_training/evaluate_models.py",
                "args": [],
                "descripcion": "Genera métricas y análisis de rendimiento",
                "opcional": True
            }
        ]
        
        self.inicio = None
        self.resultados = []
    
    def mostrar_header(self):
        """Muestra el encabezado del pipeline."""
        print("\n" + "="*80)
        print(" "*20 + "PIPELINE COMPLETO")
        print(" "*10 + "Sistema de Reconocimiento de Actividad Humana")
        print("="*80)
        print("\nEste pipeline ejecutará los siguientes pasos:")
        
        for paso in self.pasos:
            opcional = " (opcional)" if paso.get("opcional", False) else ""
            print(f"\n{paso['numero']}. {paso['nombre']}{opcional}")
            print(f"   {paso['descripcion']}")
        
        print("\n" + "="*80)
    
    def verificar_prerrequisitos(self):
        """Verifica que existan los scripts necesarios."""
        print("\nVerificando prerrequisitos...")
        
        for paso in self.pasos:
            script_path = Path(paso["script"])
            if not script_path.exists():
                print(f"✗ ERROR: No se encuentra {paso['script']}")
                return False
            print(f"✓ {paso['script']}")
        
        # Verificar carpeta de videos
        videos_path = Path("./data/raw/videos")
        if not videos_path.exists():
            print(f"\n✗ ERROR: No existe la carpeta {videos_path}")
            print("  Crea la carpeta y coloca los videos organizados por acción.")
            return False
        
        carpetas = [d for d in videos_path.iterdir() if d.is_dir()]
        if not carpetas:
            print(f"\n✗ ERROR: No hay carpetas de acciones en {videos_path}")
            return False
        
        print(f"✓ Carpeta de videos encontrada con {len(carpetas)} acciones")
        
        return True
    
    def ejecutar_paso(self, paso):
        """Ejecuta un paso del pipeline."""
        print("\n" + "="*80)
        print(f"PASO {paso['numero']}: {paso['nombre'].upper()}")
        print("="*80)
        
        inicio = time.time()
        
        try:
            # Construir comando
            cmd = [sys.executable, paso["script"]] + paso.get("args", [])
            
            # Ejecutar
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True
            )
            
            duracion = time.time() - inicio
            
            print(f"\n✓ Paso {paso['numero']} completado en {duracion:.2f} segundos")
            
            return {
                "paso": paso['numero'],
                "nombre": paso['nombre'],
                "exito": True,
                "duracion": duracion
            }
            
        except subprocess.CalledProcessError as e:
            duracion = time.time() - inicio
            print(f"\n✗ Error en paso {paso['numero']}")
            print(f"   Código de salida: {e.returncode}")
            
            return {
                "paso": paso['numero'],
                "nombre": paso['nombre'],
                "exito": False,
                "duracion": duracion,
                "error": str(e)
            }
    
    def ejecutar(self, saltar_opcionales=False):
        """Ejecuta el pipeline completo."""
        self.inicio = time.time()
        self.mostrar_header()
        
        if not self.verificar_prerrequisitos():
            print("\n✗ Prerrequisitos no cumplidos. Abortando pipeline.")
            return False
        
        print("\n" + "="*80)
        input("Presiona ENTER para iniciar el pipeline...")
        
        for paso in self.pasos:
            # Saltar pasos opcionales si se solicita
            if saltar_opcionales and paso.get("opcional", False):
                print(f"\n⊘ Saltando paso opcional: {paso['nombre']}")
                continue
            
            resultado = self.ejecutar_paso(paso)
            self.resultados.append(resultado)
            
            if not resultado["exito"]:
                print(f"\n✗ Pipeline detenido por error en paso {resultado['paso']}")
                self.mostrar_resumen()
                return False
        
        self.mostrar_resumen()
        return True
    
    def mostrar_resumen(self):
        """Muestra el resumen de ejecución del pipeline."""
        duracion_total = time.time() - self.inicio
        
        print("\n" + "="*80)
        print(" "*30 + "RESUMEN DEL PIPELINE")
        print("="*80)
        
        print(f"\nTiempo total: {duracion_total:.2f} segundos ({duracion_total/60:.2f} minutos)")
        print(f"\nPasos ejecutados: {len(self.resultados)}")
        
        print("\nDetalle por paso:")
        for res in self.resultados:
            estado = "✓" if res["exito"] else "✗"
            print(f"  {estado} Paso {res['paso']}: {res['nombre']} - {res['duracion']:.2f}s")
        
        # Archivos generados
        print("\nArchivos generados:")
        archivos_esperados = [
            "./data/processed/datosmediapipe.csv",
            "./data/processed/datos_analisis.csv",
            "./data/processed/model_features.csv",
            "./assets/best_random_forest_model.joblib",
            "./assets/best_xgboost_model.joblib",
            "./assets/label_encoder.joblib"
        ]
        
        for archivo in archivos_esperados:
            path = Path(archivo)
            if path.exists():
                tamaño = path.stat().st_size / 1024 / 1024  # MB
                print(f"  ✓ {archivo} ({tamaño:.2f} MB)")
            else:
                print(f"  ✗ {archivo} (no encontrado)")
        
        print("\n" + "="*80)
        
        if all(r["exito"] for r in self.resultados):
            print("\n✓ PIPELINE COMPLETADO EXITOSAMENTE")
            print("\nPróximos pasos:")
            print("  1. Revisar métricas en: ./results/evaluations/")
            print("  2. Ver visualizaciones en: ./results/visualizations/")
            print("  3. Ejecutar sistema en tiempo real:")
            print("     python 4_real_time_app/real_time_system.py")
        else:
            print("\n✗ PIPELINE COMPLETADO CON ERRORES")
            print("Revisa los mensajes de error anteriores.")
        
        print("\n" + "="*80 + "\n")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline completo de Reconocimiento de Actividad Humana"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Saltar pasos opcionales (análisis postural y evaluación)"
    )
    
    args = parser.parse_args()
    
    pipeline = Pipeline()
    exito = pipeline.ejecutar(saltar_opcionales=args.skip_optional)
    
    sys.exit(0 if exito else 1)


if __name__ == "__main__":
    main()
