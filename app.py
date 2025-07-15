# app.py
import sys
import os
from flask import Flask, request, jsonify, send_file, render_template, current_app
import threading
import traceback # Import traceback for detailed error logging

# --- Local Module Imports ---
# Ensure these files exist in the same directory or are accessible via PYTHONPATH
try:
    from config_loader import load_config
    from model_loader import load_models # Loads YOLO models and InsightFace app
    from database_manager import DatabaseManager # Handles DB, Embeddings, and Email triggering
    from image_processor import process_frame_logic # Performs actual frame analysis
    from utils import decode_image, encode_image # Image encoding/decoding helpers
    from fined_log_manager import FinedLogManager
except ImportError as e:
    print(f"FATAL: Failed to import necessary modules: {e}")
    print("Ensure config_loader.py, model_loader.py, database_manager.py, image_processor.py, email_notifier.py, fined_log_manager.py, and utils.py are present.")
    sys.exit(1)

# --- Global Variables ---
CONFIG = None
person_model = None
id_card_model = None
face_app = None # Global for InsightFace FaceAnalysis application
db_manager = None # Manages database, embeddings, and email logic
fined_log_manager = None
models_loaded_ok = False # Flag to track if all models loaded successfully

# --- Flask App Initialization ---
# Looks for templates in a 'templates' subfolder by default
app = Flask(__name__)

# --- Initialization Function ---
def initialize_app():
    """Loads configuration, models, and initializes the database manager."""
    global CONFIG, person_model, id_card_model, face_app, db_manager, models_loaded_ok

    print("\n" + "="*60 + "\n      Starting ID Card Compliance Monitoring System\n" + "="*60 + "\n")

    # 1. Load Configuration
    try:
        CONFIG = load_config() # Returns a flat dictionary
        if not isinstance(CONFIG, dict):
             raise TypeError("load_config did not return a dictionary.")
    except Exception as e:
        print(f"FATAL: Failed to load configuration. Exiting. Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 2. Load Models (YOLO Person, YOLO ID, InsightFace App)
    try:
        person_model, id_card_model, face_app, models_loaded_ok = load_models(CONFIG)
        if not models_loaded_ok:
            print("\n[WARNING] One or more models failed to load. Processing might be impaired or fail.")
        # Consider exiting if models are critical, e.g.:
        # if person_model is None or id_card_model is None or face_app is None:
        #    print("[FATAL] Critical model loading failure. Exiting.")
        #    sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Error during model loading: {e}")
        traceback.print_exc()
        sys.exit(1)


    # 3. Initialize Database Manager (Handles DB, Embeddings, Emails)
    try:
        db_manager = DatabaseManager(CONFIG)
        if not db_manager.is_loaded:
             # is_loaded should reflect if DB CSV load was okay
             print("\n[WARNING] Database CSV loading failed or encountered issues. Fining might be disabled or inaccurate.")
        # Check specifically if embeddings needed for recognition are loaded
        _, known_embeddings = db_manager.get_recognition_data()
        if not known_embeddings and models_loaded_ok and face_app is not None: # Only warn if face app itself loaded
            print("[WARNING] No known embeddings loaded. Face recognition will be disabled.")

    except Exception as e:
         print(f"\n[CRITICAL WARNING] Failed to initialize DatabaseManager: {e}. Recognition/fining/emails disabled.")
         traceback.print_exc()
         db_manager = None # Ensure db_manager is None if init fails


    
    # --- 4. Initialize Fined Log Manager --- ADD THIS ---
    try:
        log_csv_path = CONFIG.get('fined_log_csv', 'fined_log.csv') # Get path from config
        fined_log_manager = FinedLogManager(log_csv_path) # Create instance
        app.fined_log_manager = fined_log_manager
        
        print(f"[ OK ] Fined Log Manager initialized (File: {log_csv_path}).")
        print(f"[DEBUG initialize_app] Type of app.fined_log_manager: {type(app.fined_log_manager)}")
        print(f"[DEBUG initialize_app] Type of global fined_log_manager: {type(fined_log_manager)}, Is None: {fined_log_manager is None}")
    except Exception as e:
        print(f"\n[CRITICAL WARNING] Failed to initialize FinedLogManager: {e}. Fined event logging disabled.")
        traceback.print_exc()
        app.fined_log_manager = None
        fined_log_manager = None # Ensure it's None on failure
    # ------------------------------------
    print(f"[DEBUG initialize_app] Type of fined_log_manager: {type(fined_log_manager)}, Is None: {fined_log_manager is None}")



    # 5. Print Configuration Summary (using direct key access)
    print("\n--- Backend Ready ---")
    print(f"Configured Settings Summary:")
    print(f"  - Camera Preference: Index {CONFIG.get('camera_index', 'N/A')}")
    print(f"  - Person Model:      {CONFIG.get('person_model_path', 'N/A')}")
    print(f"  - ID Card Model:     {CONFIG.get('id_card_model_path', 'N/A')}")
    print(f"  - ArcFace Model:     {CONFIG.get('model_name', 'N/A')} (via InsightFace)")
    print(f"  - Student Database:  {CONFIG.get('csv_file', 'N/A')}")
    print(f"  - Embeddings File:   {CONFIG.get('embeddings_file', 'N/A')}")
    print(f"  - ArcFace Threshold: {CONFIG.get('similarity_threshold', 'N/A')}")
    print(f"  - Fine Amount:       ${CONFIG.get('fine_amount', 0.0):.2f}")
    email_status = "Enabled" if CONFIG.get('email_enabled', False) else "Disabled"
    sender = CONFIG.get('sender_email', 'N/A')
    print(f"  - Email Notifications: {email_status} (Sender: {sender})")
    img_dir = CONFIG.get('fined_images_dir', 'N/A')
    log_csv = CONFIG.get('fined_log_csv', 'N/A')
    print(f"  - Fined Image Dir:   {img_dir}")
    print(f"  - Fined Log CSV:     {log_csv}")
    print("--------------------------")


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page using Flask's template rendering."""
    try:
        if CONFIG is None:
             print("[ERROR in app.py / route]: Global CONFIG is None!")
             return "Error: Application configuration not loaded correctly.", 500

        # Access 'camera_index' directly from the flat CONFIG dictionary
        camera_index_from_config = CONFIG.get('camera_index', 2) # Default to 0 if not found

        # Optional debug print
        # print(f"[DEBUG app.py / route] Value from CONFIG: {camera_index_from_config}")

        camera_index_str_to_pass = str(camera_index_from_config)
        # Pass the string value to the template
        return render_template('index.html', preferred_camera_index_str=camera_index_str_to_pass)

    except Exception as e:
        print(f"Error rendering index template or preparing context: {e}")
        traceback.print_exc() # Print full traceback to console
        return f"Error loading page. Jinja/Context Error: <pre>{e}</pre>", 500


@app.route('/process', methods=['POST'])
def process_image_endpoint():
    """Receives image data, processes it using imported logic, and returns results."""
    log_manager = current_app.fined_log_manager
    print(f"[DEBUG /process] Checking log_manager from current_app. Type: {type(log_manager)}, Is None: {log_manager is None}")
    # Check if essential components are loaded and ready
    if not models_loaded_ok or person_model is None or id_card_model is None or face_app is None:
         return jsonify({"error": "Core models/apps not loaded", "processed_image": None, "detections": []}), 503 # Service Unavailable
    if db_manager is None or not db_manager.is_loaded:
        return jsonify({"error":"Database unavailable"}), 503 # Service Unavailable if DB is essential
    
    if log_manager is None: # Check the local variable accessed via current_app
         print("[Warning /process] FinedLogManager not available via current_app. Fines will not be logged to CSV.")
                # Allow processing to continue, but logging won't happen

    # Check if embeddings are needed and available (allow processing even without them if desired)
    # _, known_embeddings = db_manager.get_recognition_data()
    # if not known_embeddings:
    #      print("Info: Processing request, but no embeddings loaded (recognition disabled).")
    #      pass # Allow detection/ID check to proceed

    try:
        # Get image data from JSON payload
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        frame = decode_image(data['image'])
        if frame is None:
            return jsonify({"error": "Failed to decode image data"}), 400
        
        
        # --- Call the main processing logic from image_processor ---
        # Pass all necessary components
        processed_frame, detected_info = process_frame_logic(
            frame, person_model, id_card_model, face_app, db_manager, log_manager, CONFIG
        )
        # ---

        # Handle potential errors from processing logic itself
        if processed_frame is None:
             error_msg = "Unknown processing error occurred"
             # Try to get more specific error if provided
             if detected_info and isinstance(detected_info, list) and len(detected_info) > 0 and 'error' in detected_info[0]:
                 error_msg = detected_info[0]['error']
             print(f"Error reported by process_frame_logic: {error_msg}")
             return jsonify({"error": error_msg}), 500 # Internal Server Error

        # Encode the processed frame for sending back to the browser
        encoded_frame = encode_image(processed_frame)
        if encoded_frame is None:
            print("Error encoding processed frame to base64.")
            return jsonify({"error": "Failed to encode processed image"}), 500

        # Return successful results
        return jsonify({
            "processed_image": encoded_frame,
            "detections": detected_info
        })

    except Exception as e:
        # Catch-all for unexpected errors within the endpoint
        print(f"!! Unexpected Error in /process endpoint: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": "An internal server error occurred during processing"}), 500


@app.route('/get_totals', methods=['GET'])
def get_totals_endpoint():
    """Returns the current violation count and total fine amount."""
    if db_manager is None:
         # Return defaults if DB manager failed to initialize
         return jsonify({"violations": 0, "fine": 0.0})
    try:
        violations, fine = db_manager.get_totals()
        return jsonify({"violations": violations, "fine": float(fine)}) # Ensure fine is float
    except Exception as e:
        print(f"Error in /get_totals: {e}")
        traceback.print_exc()
        return jsonify({"error": "Failed to calculate totals"}), 500


@app.route('/export_violations', methods=['GET'])
def export_violations_endpoint():
    """Exports the current student database (with fines) as a CSV file."""
    if db_manager is None or not db_manager.is_loaded:
        return "Error: Database is not available for export.", 503 # Service Unavailable
    try:
        # Delegate CSV generation to database manager
        csv_buffer, filename, mimetype = db_manager.export_database_csv()
        # Use Flask's send_file to stream the buffer as a download
        return send_file(
            csv_buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename # Use filename generated by db_manager
        )
    except ValueError as ve: # Catch specific error from export_database_csv if DB not loaded
         print(f"Export failed: {ve}")
         return str(ve), 503
    except Exception as e:
        print(f"Error during CSV export: {e}")
        traceback.print_exc()
        return "Error generating export file.", 500


# --- Main Execution Block ---
if __name__ == '__main__':
    # Run the initialization sequence
    initialize_app()

    # Start the Flask web server
    print("\n--- Starting Flask Server ---")
    print(f"Access UI via: http://127.0.0.1:5000 (or your server's IP)")
    print("Use CTRL+C in the terminal to stop the server.")
    try:
        # Run on all available interfaces (0.0.0.0)
        # Set debug=False for production/general use
        # threaded=True allows handling multiple requests concurrently, needed for background email sending
        # Use a production WSGI server (like Gunicorn or Waitress) for deployment instead of Flask's built-in server
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
         # Catch errors related to starting the server itself
         print(f"\n--- Flask Server Could Not Start: {e} ---")
         traceback.print_exc()
    finally:
        # This runs when the server is shut down (e.g., by Ctrl+C)
        print("\n--- Server Shutdown ---")