# image_processor.py
import os
import datetime
import cv2
import numpy as np
from scipy.spatial.distance import cosine # <-- For cosine distance/similarity

#print("[DEBUG image_processor.py] 'os' module imported successfully.")

# Import helpers and constants from utils
from utils import (draw_text_with_background,
                   COLOR_PERSON_WITH_ID, COLOR_RECOGNIZED_NO_ID,
                   COLOR_UNKNOWN_NO_ID, COLOR_ID_CARD, COLOR_TEXT)

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two embeddings."""
    # Cosine distance = 1 - cosine similarity
    # Cosine similarity = 1 - cosine distance
    try:
        # Ensure inputs are numpy arrays
        emb1 = np.asarray(embedding1)
        emb2 = np.asarray(embedding2)
        # Handle potential shape mismatches if necessary, though normed_embedding should be consistent
        if emb1.shape != emb2.shape:
             # print(f"Warning: Embedding shape mismatch: {emb1.shape} vs {emb2.shape}")
             return 0.0 # Or raise error
        similarity = 1 - cosine(emb1, emb2)
        return similarity
    except Exception as e:
        print(f"Error calculating cosine similarity: {e}")
        return 0.0


def process_frame_logic(frame, person_model, id_card_model, face_app, db_manager, fined_log_manager,config): # <-- Added face_app
    """
    Processes frame: detects persons (YOLO), detects IDs (YOLO),
    detects faces and extracts embeddings within person ROIs (InsightFace),
    compares embeddings, applies fines, draws results.
    """
    
    person_conf = config.get('person_conf_threshold', 0.6) # Use direct key + default
    id_card_conf = config.get('id_card_conf_threshold', 0.5) # Use direct key + default
    arcface_thresh = config.get('similarity_threshold', 0.5) # Use direct key + default
    fined_images_dir = config.get('fined_images_dir', 'fined_student_images') # <-- Get image save directory
    
    if frame is None:
        print("Error: process_frame_logic received None frame.")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Input Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return error_frame, [{"error": "Input frame was None"}]

    # Check if models/apps are loaded
    if person_model is None or id_card_model is None or face_app is None: # <-- Check face_app
        error_frame = frame.copy()
        draw_text_with_background(error_frame, "ERROR: Models/FaceApp not loaded", (10, 30),
                                  fontScale=0.7, color=(255,255,255), bg_color=(200,0,0), alpha=0.8)
        return error_frame, [{"error": "Detection models or FaceAnalysis app not loaded"}]

    processed_frame = frame.copy()
    detected_info = []

    # Get current known face data from the database manager
    known_names_map, known_embeddings_map = db_manager.get_recognition_data()
    recognition_possible = db_manager.is_loaded and bool(known_embeddings_map) # Check if embeddings were loaded

    if not recognition_possible:
         draw_text_with_background(processed_frame, "WARN: Embeddings N/A", (10, 60),
                                   fontScale=0.7, color=(0,0,0), bg_color=(255,200,0), alpha=0.8)

    # --- Person Detection (YOLO) ---
    try:
        person_results = person_model(processed_frame, stream=False, classes=[0], conf=person_conf, verbose=False)
        person_boxes = person_results[0].boxes if person_results and len(person_results) > 0 and person_results[0].boxes is not None else []
    except Exception as e:
        print(f"Error during person detection: {e}")
        draw_text_with_background(processed_frame, "Person Detection Error", (10, 90),
                                  fontScale=0.6, color=(255,255,255), bg_color=(200,0,0), alpha=0.7)
        person_boxes = []

    # --- ID Card Detection (YOLO) ---
    id_card_centers = []
    try:
        # ... (Keep existing ID card detection logic) ...
        id_card_results = id_card_model(processed_frame, stream=False, conf=id_card_conf, verbose=False)
        id_card_boxes = id_card_results[0].boxes if id_card_results and len(id_card_results) > 0 and id_card_results[0].boxes is not None else []
        for id_box in id_card_boxes:
            ix1, iy1, ix2, iy2 = map(int, id_box.xyxy[0])
            cv2.rectangle(processed_frame, (ix1, iy1), (ix2, iy2), COLOR_ID_CARD, 2)
            id_card_centers.append(((ix1 + ix2) / 2, (iy1 + iy2) / 2))
            draw_text_with_background(processed_frame, "ID", (ix1, iy1 - 5),
                                      fontScale=0.4, color=COLOR_TEXT, bg_color=COLOR_ID_CARD[:3], alpha=0.7)
    except Exception as e:
        print(f"Warning: ID card detection failed: {e}")

    # --- Process Each Detected Person ---
    for person_box in person_boxes:
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])

        # Clamp Coordinates & Basic Check
        h, w = processed_frame.shape[:2]
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        if y1 >= y2 or x1 >= x2 or (y2 - y1) < 30 or (x2 - x1) < 20: # Adjust minimum size if needed
            continue

        person_status = "unknown_no_id"
        box_color = COLOR_UNKNOWN_NO_ID
        display_name = "Unknown"
        similarity_score = 0.0
        matched_student_id = None
        matched_student_name = "Unknown"
        face_detected_in_roi = False # Flag

        # Check if an ID card center falls within this person's bounding box
        id_found_for_person = any(x1 < icx < x2 and y1 < icy < y2 for icx, icy in id_card_centers)

        if id_found_for_person:
            person_status = "id_detected"
            box_color = COLOR_PERSON_WITH_ID
            display_name = "ID Verified"
            # Optionally: Could still try face detection/rec here if desired
        else:
            # No ID found, attempt face detection and recognition within the person ROI
            person_roi = processed_frame[y1:y2, x1:x2]

            if person_roi.shape[0] > 0 and person_roi.shape[1] > 0 and recognition_possible:
                try:
                    # Use insightface app.get() on the person ROI
                    faces = face_app.get(person_roi) # Detect faces and get embeddings

                    if len(faces) > 0:
                        face_detected_in_roi = True
                        # If multiple faces, optionally pick the largest/most central
                        if len(faces) > 1:
                            # print(f"Debug: Multiple faces ({len(faces)}) in ROI, using largest.")
                            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

                        detected_embedding = faces[0].normed_embedding

                        # --- Compare with known embeddings ---
                        best_match_id = None
                        max_similarity = 0.0

                        for known_id, known_embedding in known_embeddings_map.items():
                            sim = calculate_cosine_similarity(detected_embedding, known_embedding)
                            if sim > max_similarity:
                                max_similarity = sim
                                best_match_id = known_id

                        similarity_score = max_similarity

                        if best_match_id is not None and max_similarity >= arcface_thresh:
                            # Match found!
                            person_status = "recognized_no_id"
                            box_color = COLOR_RECOGNIZED_NO_ID
                            matched_student_id = best_match_id
                            matched_student_name = known_names_map.get(matched_student_id, "Name Error") # Get name from map
                            display_name = f"Fine: {matched_student_name} ({max_similarity:.2f})"

                            # Apply fine using DatabaseManager
                            fine_applied = db_manager.apply_fine(matched_student_id, matched_student_name)
                            
                            # --- >> CAPTURE IMAGE & LOG FINE (if fine was applied) << ---
                            if fine_applied and fined_log_manager: # Check if fine was new and logger exists
                                #print(f"  [DEBUG] Entered image capture/log block for {matched_student_id}")
                                now = datetime.datetime.now()
                                timestamp_str = now.strftime('%Y%m%d_%H%M%S')
                                image_filename = f"{matched_student_id}_{timestamp_str}.jpg"
                                save_path = os.path.join(fined_images_dir, image_filename)
                                #print(f"  [DEBUG] Attempting to save image to: {save_path}")

                                try:
                                    abs_fined_images_dir = os.path.abspath(fined_images_dir)
                                    print(f"  [DEBUG] Absolute image directory: {abs_fined_images_dir}")
                                    # Ensure directory exists
                                    os.makedirs(fined_images_dir, exist_ok=True)
                                    #print(f"  [DEBUG] ROI shape: {person_roi.shape}, path: {save_path}")
                                    # Save the ROI image
                                    success = cv2.imwrite(save_path, person_roi)
                                    if success:
                                        print(f"  [Capture] Saved image: {save_path}")
                                        #print(f"  [DEBUG] Calling log_fine with: id={matched_student_id}, name={matched_student_name}, ts={now}, img={image_filename}")
                                        # Log the fine details including the relative filename
                                        fined_log_manager.log_fine(
                                            student_id=matched_student_id,
                                            name=matched_student_name,
                                            timestamp=now, # Pass datetime object
                                            image_filename=image_filename # Just the filename
                                        )
                                    else:
                                         print(f"  [Capture FAIL] Failed to save image: {save_path}")
                                         # Log anyway, but maybe with empty filename?
                                         fined_log_manager.log_fine(matched_student_id, matched_student_name, now, "SAVE_FAILED")

                                except Exception as capture_e:
                                    print(f"  [Capture ERROR] Exception saving image or logging fine: {capture_e}")
                                    import traceback
                                    traceback.print_exc()
                                    # Log anyway?
                                    fined_log_manager.log_fine(matched_student_id, matched_student_name, now, "CAPTURE_ERROR")
                            # --- >> END CAPTURE & LOG << ---

                            #elif not fine_applied:
                                #print(f"  [DEBUG] Image/Log skipped: Fine was not applied this time (already fined today?).")
                            #elif not fined_log_manager:
                                 #print(f"  [DEBUG] Image/Log skipped: fined_log_manager is None.")

                        else:
                            # Face detected, but not recognized (below threshold)
                            person_status = "unknown_no_id" # Or maybe "unrecognized_face"?
                            box_color = COLOR_UNKNOWN_NO_ID
                            display_name = f"Unknown Face ({max_similarity:.2f})"
                    else:
                        # No face detected within the person ROI by insightface
                        person_status = "unknown_no_id"
                        box_color = COLOR_UNKNOWN_NO_ID
                        display_name = "Unknown (No Face)"

                except Exception as face_e:
                    print(f"Error during face detection/embedding in ROI: {face_e}")
                    person_status = "error"
                    display_name = "Face Detection Error"
            elif not recognition_possible:
                 # Embeddings not loaded
                 person_status = "unknown_no_id"
                 box_color = COLOR_UNKNOWN_NO_ID
                 display_name = "Unknown (Rec N/A)"
            # Else: ROI was invalid (should be caught earlier)


        # --- Draw Bounding Box and Label ---
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)
        label_y = y1 - 7 if y1 > 20 else y2 + 15
        draw_text_with_background(processed_frame, display_name, (x1 + 2, label_y),
                                  fontScale=0.45, color=COLOR_TEXT, thickness=1,
                                  bg_color=box_color[:3], alpha=0.75)

        # --- Store Detection Info ---
        detected_info.append({
            "status": person_status,
            "student_id": matched_student_id,
            "name": matched_student_name,
            # Show similarity only if a face was detected and compared
            "similarity": f"{similarity_score:.2f}" if face_detected_in_roi else "N/A",
            "bbox": [x1, y1, x2, y2]
        })

    return processed_frame, detected_info