# generate_embeddings.py
import insightface
import cv2
import numpy as np
import pandas as pd
import os
import sys
from config_loader import load_config # Assuming config_loader.py is in the same dir

def generate_known_embeddings(config):
    """
    Processes student images from the database CSV, extracts ArcFace embeddings,
    and saves them to the specified file.
    """
    db_csv_path = config.get('csv_file', 'students_db.csv') # Direct key access with default
    embeddings_output_file = config.get('embeddings_file', 'known_embeddings.npy') # Direct key access with default
    arcface_model_name = config.get('model_name', 'buffalo_l') # Direct key access with default
    providers_str = config.get('providers', 'CPU') # Direct key access with default

    # Parse providers string into a list
    providers = [p.strip() + 'ExecutionProvider' for p in providers_str.split(',')]
    print(f"Using Execution Providers: {providers}")

    print("--- Generating Known Embeddings ---")
    print(f"Loading ArcFace model '{arcface_model_name}'...")

    try:
        # Initialize FaceAnalysis - this loads detector and recognizer
        # allowed_modules=['detection', 'recognition'] ensures both are loaded
        # You might need to run this once with internet to download models
        face_app = insightface.app.FaceAnalysis(name=arcface_model_name,
                                                allowed_modules=['detection', 'recognition'],
                                                providers=providers)
        face_app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 for CPU or first GPU
        print("ArcFace model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load InsightFace model '{arcface_model_name}'. Error: {e}")
        print("Ensure 'insightface' and 'onnxruntime' are installed.")
        print("If using GPU, check CUDA/cuDNN setup and provider setting in config.ini.")
        sys.exit(1)

    # Load the student database CSV
    if not os.path.exists(db_csv_path):
        print(f"ERROR: Database CSV file not found at '{db_csv_path}'")
        sys.exit(1)

    try:
        db = pd.read_csv(db_csv_path)
        required_cols = ["student_id", "name", "image_path"]
        if not all(col in db.columns for col in required_cols):
             print(f"ERROR: DB CSV '{db_csv_path}' must have columns: {', '.join(required_cols)}")
             sys.exit(1)
        db['student_id'] = db['student_id'].astype(str).str.strip()
        db['image_path'] = db['image_path'].astype(str).str.strip()
        print(f"Loaded {len(db)} student records from '{db_csv_path}'.")
    except Exception as e:
        print(f"ERROR: Failed to load or parse database CSV '{db_csv_path}': {e}")
        sys.exit(1)

    known_embeddings = {}
    processed_count = 0
    error_count = 0
    no_face_count = 0
    multiple_faces_count = 0

    csv_dir = os.path.dirname(os.path.abspath(db_csv_path))

    print("Processing student images...")
    for index, row in db.iterrows():
        student_id = row['student_id']
        image_path_rel = row['image_path']
        name = row['name']

        if not image_path_rel or pd.isna(image_path_rel):
            print(f"  [Skip] No image path for student {student_id} ({name}).")
            error_count += 1
            continue

        abs_path = image_path_rel if os.path.isabs(image_path_rel) else os.path.join(csv_dir, image_path_rel)
        abs_path = os.path.normpath(abs_path)

        if not os.path.exists(abs_path):
            print(f"  [Error] Image file not found for student {student_id} ({name}): {abs_path}")
            error_count += 1
            continue

        try:
            img = cv2.imread(abs_path)
            if img is None:
                print(f"  [Error] Failed to read image for student {student_id} ({name}): {abs_path}")
                error_count += 1
                continue

            # Use insightface to get faces (includes detection and embedding)
            # IMPORTANT: Ensure image is BGR format (cv2.imread usually is)
            faces = face_app.get(img) # Returns a list of Face objects

            if len(faces) == 0:
                print(f"  [Warn] No face detected for student {student_id} ({name}) in image: {abs_path}")
                no_face_count += 1
                continue # Skip if no face found

            if len(faces) > 1:
                print(f"  [Warn] Multiple faces ({len(faces)}) detected for student {student_id} ({name}) in image: {abs_path}. Using the largest face.")
                multiple_faces_count += 1
                # Select the face with the largest bounding box area
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

            # Get the embedding from the first (or largest) face
            embedding = faces[0].normed_embedding # Use the normalized embedding

            # Store the embedding (ensure it's a numpy array)
            known_embeddings[student_id] = np.array(embedding)
            processed_count += 1
            # print(f"  [OK] Processed embedding for student {student_id} ({name}).")

        except Exception as e:
            print(f"  [Error] Exception processing image for student {student_id} ({name}) at {abs_path}: {e}")
            error_count += 1

    print("\n--- Embedding Generation Summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Images not found/unreadable/no path: {error_count}")
    print(f"Images with no face detected: {no_face_count}")
    print(f"Images with multiple faces: {multiple_faces_count}")

    if processed_count == 0:
        print("\nERROR: No embeddings were generated. Cannot save file. Check image paths and face detection.")
    else:
        try:
            print(f"\nSaving {len(known_embeddings)} embeddings to '{embeddings_output_file}'...")
            # Save as a dictionary {student_id: numpy_array}
            np.save(embeddings_output_file, known_embeddings)
            print("Embeddings saved successfully.")
        except Exception as e:
            print(f"ERROR: Failed to save embeddings to '{embeddings_output_file}': {e}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        config = load_config()
        generate_known_embeddings(config)
    except FileNotFoundError as e:
        print(f"ERROR: Configuration file not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()