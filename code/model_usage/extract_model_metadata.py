import torch
import json
from ultralytics import YOLO
from pathlib import Path
import yaml
import os

# Ruta del modelo
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'pebble_seg_model.pt')

def extract_model_metadata():
    """
    Extrae toda la metadata del modelo YOLO entrenado
    """
    print("🔍 Extrayendo metadata del modelo YOLO...")
    print(f"📁 Modelo: {model_path}")
    print("-" * 80)
    
    try:
        # Cargar el checkpoint directamente con torch
        print("⚡ Cargando checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("\n📊 INFORMACIÓN GENERAL DEL MODELO:")
        print("=" * 50)
        
        # Información básica del modelo
        if 'model' in checkpoint:
            print(f"✅ Modelo cargado exitosamente")
            
        # Epoch información
        if 'epoch' in checkpoint:
            print(f"🎯 Epoch final: {checkpoint['epoch']}")
            print(f"🏆 Este es el MEJOR modelo obtenido en el epoch: {checkpoint['epoch']}")
        
        # Métricas del entrenamiento
        if 'best_fitness' in checkpoint:
            best_fitness = checkpoint['best_fitness']
            if best_fitness is not None:
                print(f"🏅 Mejor fitness: {best_fitness:.6f}")
            else:
                print(f"🏅 Mejor fitness: No disponible (None)")
        
        # Información del entrenador
        if 'train_args' in checkpoint:
            print(f"\n🔧 ARGUMENTOS DE ENTRENAMIENTO:")
            print("-" * 30)
            train_args = checkpoint['train_args']
            if train_args is not None:
                for key, value in train_args.items():
                    print(f"  {key}: {value}")
            else:
                print("  No disponible (None)")
        
        # Información del optimizador
        if 'optimizer' in checkpoint:
            print(f"\n⚙️ OPTIMIZADOR:")
            print("-" * 20)
            optimizer_info = checkpoint['optimizer']
            if optimizer_info is not None and 'param_groups' in optimizer_info:
                for i, group in enumerate(optimizer_info['param_groups']):
                    print(f"  Grupo {i}:")
                    for key, value in group.items():
                        if key != 'params':  # Excluir parámetros (muy largos)
                            print(f"    {key}: {value}")
            else:
                print("  No disponible")
        
        # Métricas detalladas si están disponibles
        if 'model' in checkpoint:
            try:
                # Cargar usando ultralytics para obtener más info
                model = YOLO(model_path)
                
                print(f"\n🏗️ ARQUITECTURA DEL MODELO:")
                print("-" * 30)
                print(f"  Nombre del modelo: {model.model_name if hasattr(model, 'model_name') else 'YOLOv8'}")
                print(f"  Número de clases: {model.model.nc if hasattr(model.model, 'nc') else 'N/A'}")
                print(f"  Nombres de clases: {model.names}")
                
                # Información del dispositivo de entrenamiento
                if hasattr(model.model, 'device'):
                    print(f"  Dispositivo de entrenamiento: {model.model.device}")
                    
            except Exception as e:
                print(f"⚠️ No se pudo cargar información adicional del modelo: {e}")
        
        # Fecha de entrenamiento si está disponible
        if 'date' in checkpoint:
            date_val = checkpoint['date']
            print(f"\n📅 INFORMACIÓN TEMPORAL:")
            print("-" * 25)
            if date_val is not None:
                print(f"  Fecha de entrenamiento: {date_val}")
            else:
                print(f"  Fecha de entrenamiento: No disponible")
        
        # Versión de ultralytics
        if 'version' in checkpoint:
            version_val = checkpoint['version']
            if version_val is not None:
                print(f"  Versión de Ultralytics: {version_val}")
            else:
                print(f"  Versión de Ultralytics: No disponible")
        
        # Información de updates/época
        if 'updates' in checkpoint:
            updates_val = checkpoint['updates']
            if updates_val is not None:
                print(f"  Total de updates: {updates_val}")
            else:
                print(f"  Total de updates: No disponible")
            
        # EMA (Exponential Moving Average) info
        if 'ema' in checkpoint:
            print(f"\n🎭 INFORMACIÓN EMA:")
            print("-" * 20)
            ema_info = checkpoint['ema']
            if ema_info is not None:
                if hasattr(ema_info, 'updates'):
                    print(f"  EMA updates: {ema_info.updates}")
                if hasattr(ema_info, 'decay'):
                    print(f"  EMA decay: {ema_info.decay}")
            else:
                print("  EMA info no disponible")
        
        # Mostrar todas las claves disponibles en el checkpoint
        print(f"\n🔑 CLAVES DISPONIBLES EN EL CHECKPOINT:")
        print("-" * 45)
        for key in checkpoint.keys():
            value = checkpoint[key]
            if value is not None:
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: <{type(value).__name__}>")
            else:
                print(f"  {key}: None")
        
        # Buscar archivo de resultados de entrenamiento
        model_dir = Path(model_path).parent.parent
        results_file = model_dir / 'results.csv'
        args_file = model_dir / 'args.yaml'
        
        if results_file.exists():
            print(f"\n📈 ARCHIVO DE RESULTADOS ENCONTRADO:")
            print("-" * 40)
            print(f"  📄 {results_file}")
            
            # Leer últimas líneas del CSV para ver progreso
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    print(f"  📊 Total de epochs registrados: {len(lines) - 1}")  # -1 por header
                    if len(lines) > 1:
                        print(f"  🔚 Última línea: {lines[-1].strip()}")
            except Exception as e:
                print(f"  ⚠️ Error leyendo CSV: {e}")
        
        if args_file.exists():
            print(f"\n⚙️ ARCHIVO DE CONFIGURACIÓN ENCONTRADO:")
            print("-" * 45)
            print(f"  📄 {args_file}")
            
            try:
                with open(args_file, 'r') as f:
                    args_data = yaml.safe_load(f)
                    print(f"\n🔧 CONFIGURACIÓN COMPLETA DEL ENTRENAMIENTO:")
                    print("-" * 50)
                    for key, value in args_data.items():
                        print(f"  {key}: {value}")
            except Exception as e:
                print(f"  ⚠️ Error leyendo YAML: {e}")
        
        # Guardar metadata en JSON
        output_file = Path(model_path).parent / 'model_metadata.json'
        
        # Preparar diccionario con metadata para guardar
        metadata = {}
        for key in ['epoch', 'best_fitness', 'train_args', 'date', 'version', 'updates']:
            if key in checkpoint:
                metadata[key] = checkpoint[key]
        
        # Convertir valores no serializables
        def convert_for_json(obj):
            if hasattr(obj, 'item'):  # Para tensors de pytorch
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        metadata = convert_for_json(metadata)
        
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n💾 METADATA GUARDADA EN:")
        print("-" * 30)
        print(f"  📄 {output_file}")
        
        print(f"\n" + "="*80)
        print(f"🎯 RESUMEN PRINCIPAL:")
        print(f"="*80)
        
        # Analizar el archivo CSV para encontrar el epoch real
        epoch_from_model = checkpoint.get('epoch', 'N/A')
        print(f"🏆 EPOCH EN EL MODELO: {epoch_from_model} (no confiable)")
        
        # Buscar el epoch real en el CSV
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        # La última línea válida es el último epoch entrenado
                        last_line = lines[-1].strip()
                        if last_line:
                            epoch_real = last_line.split(',')[0]
                            print(f"🎯 EPOCH REAL (del CSV): {epoch_real}")
                            print(f"📊 TOTAL DE EPOCHS ENTRENADOS: {len(lines) - 1}")
                        
                        # Analizar para encontrar el mejor epoch
                        print(f"\n🔍 ANALIZANDO EL MEJOR EPOCH...")
                        header = lines[0].strip().split(',')
                        
                        # Buscar columnas de métricas importantes
                        fitness_cols = []
                        for i, col in enumerate(header):
                            if 'fitness' in col.lower() or 'map' in col.lower() or 'precision' in col.lower():
                                fitness_cols.append((i, col))
                        
                        if fitness_cols:
                            print(f"📈 Métricas encontradas: {[col[1] for col in fitness_cols]}")
                            
                            # Analizar líneas para encontrar mejores valores
                            best_epochs = {}
                            for line_num, line in enumerate(lines[1:], 1):  # Skip header
                                values = line.strip().split(',')
                                if len(values) > max(col[0] for col in fitness_cols):
                                    for col_idx, col_name in fitness_cols:
                                        try:
                                            value = float(values[col_idx])
                                            if col_name not in best_epochs or value > best_epochs[col_name][1]:
                                                best_epochs[col_name] = (line_num, value)
                                        except:
                                            continue
                            
                            for metric, (best_epoch, best_value) in best_epochs.items():
                                print(f"  🏅 Mejor {metric}: Epoch {best_epoch} (valor: {best_value:.4f})")
                        
            except Exception as e:
                print(f"⚠️ Error analizando CSV: {e}")
        
        # Mostrar fitness del modelo (con manejo seguro de None)
        best_fitness = checkpoint.get('best_fitness')
        if best_fitness is not None:
            print(f"🏅 FITNESS DEL MODELO: {best_fitness:.6f}")
        else:
            print(f"🏅 FITNESS DEL MODELO: No disponible (None)")
            
        print(f"📁 MODELO UBICADO EN: {model_path}")
        print(f"✅ EXTRACCIÓN DE METADATA COMPLETADA")
        
    except Exception as e:
        print(f"❌ Error extrayendo metadata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_model_metadata()
