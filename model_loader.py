# model_loader.py
import os
import numpy as np
from ultralytics import YOLO
import insightface # <-- Add insightface import

def load_models(config):
    """Loads YOLO models and the InsightFace FaceAnalysis app."""
    # --- Corrected Configuration Access ---
    person_model_path = config.get('person_model_path', 'yolov8n.pt')
    id_card_model_path = config.get('id_card_model_path', 'id_card_detector.pt')
    arcface_model_name = config.get('model_name', 'buffalo_l')
    providers_str = config.get('providers', 'CPU')
    # --- End Correction ---

    person_model = None
    id_card_model = None
    face_app = None # <-- Variable for insightface app

    print("--- Loading Detection & Recognition Models ---")
    models_loaded_successfully = True


    # --- Load Person YOLO Model ---
    try:
        # ... (Keep existing person model loading code) ...
        if not os.path.exists(person_model_path):
            print(f"Info: Person model '{person_model_path}' not found locally. YOLO might attempt download.")
        person_model = YOLO(person_model_path)
        _ = person_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        print(f"[ OK ] Person detection model loaded: '{person_model_path}'")
    except Exception as e:
        print(f"[FAIL] ERROR loading person model '{person_model_path}': {e}")
        models_loaded_successfully = False

    # --- Load ID Card YOLO Model ---
    try:
        # ... (Keep existing ID card model loading code) ...
        if not os.path.exists(id_card_model_path):
            print(f"[FAIL] ERROR: ID Card model file not found at '{id_card_model_path}'.")
            print("       Check the 'id_card_model' path in config.ini.")
            models_loaded_successfully = False
        else:
            id_card_model = YOLO(id_card_model_path)
            _ = id_card_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
            print(f"[ OK ] ID Card detection model loaded: '{id_card_model_path}'")
    except Exception as e:
        print(f"[FAIL] ERROR loading ID Card model '{id_card_model_path}': {e}")
        models_loaded_successfully = False

    # --- Load InsightFace FaceAnalysis App ---
    print(f"Loading ArcFace model '{arcface_model_name}' via InsightFace...")
    try:
        providers = [p.strip() + 'ExecutionProvider' for p in providers_str.split(',')]
        print(f"Attempting to use Execution Providers: {providers}")

        face_app = insightface.app.FaceAnalysis(name=arcface_model_name,
                                                allowed_modules=['detection', 'recognition'],
                                                providers=providers)
        face_app.prepare(ctx_id=0, det_size=(640, 640)) # det_size can be adjusted
        # Perform a dummy analysis to ensure loading
        _ = face_app.get(np.zeros((100, 100, 3), dtype=np.uint8))
        print(f"[ OK ] InsightFace FaceAnalysis app loaded (Detector+Recognizer: '{arcface_model_name}').")
    except Exception as e:
        print(f"[FAIL] ERROR loading InsightFace model '{arcface_model_name}'. Error: {e}")
        print("       Check model name in config.ini, dependencies, and providers (CPU/CUDA).")
        models_loaded_successfully = False


    print("--- Model Loading Complete ---")
    if not models_loaded_successfully:
        print("[WARNING] One or more models failed to load.")

    # Return all loaded models/apps
    return person_model, id_card_model, face_app, models_loaded_successfully