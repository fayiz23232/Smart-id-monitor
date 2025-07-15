# database_manager.py
import pandas as pd
import os
import numpy as np
import threading # Required for email thread
import datetime
import io

# --- Import the email sending function from the separate module ---
try:
    from email_notifier import send_fine_notification
except ImportError:
    print("[ERROR] Could not import 'send_fine_notification' from 'email_notifier.py'. Ensure the file exists.")
    # Define a dummy function so the rest of the code doesn't crash immediately
    def send_fine_notification(*args, **kwargs):
        print("[WARN] Dummy email function called because import failed.")

# --- DatabaseManager Class ---
class DatabaseManager:
    def __init__(self, config):
        # --- Load Basic Config ---
        self.csv_file_path = config.get('csv_file', 'students_db.csv')
        self.embeddings_file_path = config.get('embeddings_file', 'known_embeddings.npy')
        self.fine_amount = config.get('fine_amount', 50.0) # Default based on logs

        # --- Store Email Config ---
        self.email_config = {
            'enabled': config.get('email_enabled', False),
            'smtp_server': config.get('smtp_server'),
            'smtp_port': config.get('smtp_port'),
            'sender_email': config.get('sender_email'),
            'sender_password': config.get('sender_password'),
            'use_tls': config.get('use_tls', True),
            'email_subject': config.get('email_subject')
        }
        # --------------------------

        self.students_db = None
        self.known_ids = []
        self.known_names = {} # {id: name}
        self.known_embeddings = {} # {id: embedding_array}
        self.known_emails = {} # {id: email} <-- Store emails

        self.fined_students_today = set()
        self.current_day = datetime.date.today()
        self.db_lock = threading.Lock()
        self.is_loaded = self._load_database_and_embeddings()

    def _load_database_and_embeddings(self):
        """Loads student info (incl. email) from CSV and embeddings."""
        print(f"--- Loading Student Database ('{self.csv_file_path}') & Embeddings ('{self.embeddings_file_path}') ---")
        db_loaded = False
        embeddings_loaded = False

        # --- Load CSV Data ---
        try:
            if not os.path.exists(self.csv_file_path):
                print(f"[WARN] Database CSV file '{self.csv_file_path}' not found.")
                self.students_db = pd.DataFrame(columns=["student_id", "name", "image_path", "fine_amount", "email"]) # Added email col
                self.known_ids, self.known_names, self.known_emails = [], {}, {}
                db_loaded = True
            else:
                db = pd.read_csv(self.csv_file_path)
                required_cols = ["student_id", "name", "image_path", "fine_amount", "email"] # Added 'email'
                if not all(col in db.columns for col in required_cols):
                    print(f"[FAIL] ERROR: DB CSV '{self.csv_file_path}' must have columns: {', '.join(required_cols)}")
                    return False # Fail if columns missing

                # Clean data
                db['student_id'] = db['student_id'].astype(str).str.strip()
                db['fine_amount'] = pd.to_numeric(db['fine_amount'], errors='coerce').fillna(0).astype(float)
                db['name'] = db['name'].astype(str).str.strip().fillna('Unknown')
                db['email'] = db['email'].astype(str).str.strip().replace('', np.nan) # Handle empty strings

                self.students_db = db
                self.known_ids = db["student_id"].tolist()
                self.known_names = pd.Series(db.name.values, index=db.student_id).to_dict()
                # Create {id: email} map, excluding rows where email is NaN/empty
                self.known_emails = db.dropna(subset=['email']).set_index('student_id')['email'].to_dict()

                print(f"[ OK ] Database CSV loaded: {len(self.known_ids)} students ({len(self.known_emails)} with emails).")
                db_loaded = True

        except pd.errors.EmptyDataError:
             print(f"[WARN] Database file '{self.csv_file_path}' is empty.")
             self.students_db = pd.DataFrame(columns=["student_id", "name", "image_path", "fine_amount", "email"])
             self.known_ids, self.known_names, self.known_emails = [], {}, {}
             db_loaded = True
        except Exception as e:
            print(f"[FAIL] ERROR loading database CSV '{self.csv_file_path}': {e}")
            return False # Hard fail on other CSV errors

        # --- Load Embeddings ---
        if not os.path.exists(self.embeddings_file_path):
             print(f"[FAIL] ERROR: Embeddings file '{self.embeddings_file_path}' not found.")
             embeddings_loaded = False
        else:
             try:
                 loaded_data = np.load(self.embeddings_file_path, allow_pickle=True)
                 if isinstance(loaded_data, np.ndarray) and loaded_data.size == 1 and isinstance(loaded_data.item(), dict):
                     self.known_embeddings = loaded_data.item()
                 elif isinstance(loaded_data, dict):
                      self.known_embeddings = loaded_data
                 else:
                     raise TypeError("Loaded embeddings file is not in the expected dictionary format.")
                 print(f"[ OK ] Embeddings loaded for {len(self.known_embeddings)} students from '{self.embeddings_file_path}'.")
                 # Optional verification checks...
                 embeddings_loaded = True
             except Exception as e:
                 print(f"[FAIL] ERROR loading or parsing embeddings file '{self.embeddings_file_path}': {e}")
                 embeddings_loaded = False
        # ---

        # --- Final Status ---
        if not db_loaded:
             print("[FAIL] Database CSV failed to load.")
             return False
        if not embeddings_loaded:
            print("[WARN] Embeddings failed to load. Face recognition may be impaired.")
            # Decide if embeddings are mandatory

        print("--- Database & Embeddings Loading Complete ---")
        return db_loaded

    def _reset_daily_fines_if_needed(self):
        """Resets the set of fined students if the day has changed. Must be called within db_lock."""
        today = datetime.date.today()
        if today != self.current_day:
            # This check should ideally be inside the lock for thread-safety if multiple threads could call it
            # but since apply_fine holds the lock when calling this, it's currently safe.
            print(f"--- New Day ({today}) --- Resetting daily fined list ---")
            self.fined_students_today = set()
            self.current_day = today

    def apply_fine(self, student_id, student_name):
        """Applies fine, saves DB, and triggers email notification in a new thread."""
        if not self.is_loaded or self.students_db is None:
            print(f"Error: Cannot apply fine. Database not loaded.")
            return False # Exit early if DB not ready

        fine_applied_successfully = False
        new_total_fine_amount = 0.0 # Store the student's new total fine

        # Use lock for modifying shared resources (students_db, fined_students_today)
        with self.db_lock:
            self._reset_daily_fines_if_needed()

            if student_id in self.fined_students_today:
                # Fine was not applied *this time*, so return False immediately
                return False # Exit the method here

            print(f"-> Violation: Applying ${self.fine_amount:.2f} fine to {student_name} (ID: {student_id})")
            student_indices = self.students_db.index[self.students_db['student_id'] == student_id].tolist()

            if student_indices:
                db_idx = student_indices[0]
                current_fine = self.students_db.loc[db_idx, 'fine_amount']
                new_fine = (current_fine if pd.notna(current_fine) else 0) + self.fine_amount
                self.students_db.loc[db_idx, 'fine_amount'] = new_fine
                new_total_fine_amount = new_fine # Assign the student's total for the email

                # Attempt to save DB
                try:
                    self.students_db.to_csv(self.csv_file_path, index=False, float_format='%.2f') # Save with format
                    # Only if save succeeds: update state and set success flag
                    self.fined_students_today.add(student_id)
                    fine_applied_successfully = True # Mark success
                except Exception as e:
                    print(f"   ERROR: Failed to save updated DB to '{self.csv_file_path}': {e}")
                    # Revert the change in memory if save failed
                    self.students_db.loc[db_idx, 'fine_amount'] = current_fine
                    fine_applied_successfully = False # Mark failure
            else:
                print(f"   ERROR: Student ID {student_id} not found in DB for applying fine.")
                fine_applied_successfully = False # Mark failure

        # --- Correctly Indented Email Trigger Block ---
        # This block now runs AFTER the 'with' block finishes
        if fine_applied_successfully and self.email_config.get('enabled', False):
            # Indented one level (relative to the start of the method)
            recipient_email = self.known_emails.get(student_id)
            if recipient_email:
                # Indented two levels
                print(f"  [Info] Preparing email notification for {student_name}...")
                email_thread = threading.Thread(
                    target=send_fine_notification, # Target the imported function
                    args=(recipient_email, student_name, self.fine_amount, new_total_fine_amount, self.email_config),
                    daemon=True
                )
                email_thread.start()
            else:
                # Indented two levels
                print(f"  [Info] Fine applied to {student_name}, but no email address found in database.")
        elif not self.email_config.get('enabled', False) and fine_applied_successfully:
             # Indented one level
             print(f"  [Info] Fine applied to {student_name}, but email notifications are disabled.")
        # --- End of Email Trigger Block ---

        # Return the actual success status of applying the fine (and saving)
        return fine_applied_successfully

    def get_totals(self):
        """Calculates total outstanding fine and violations today. Thread-safe."""
        if not self.is_loaded or self.students_db is None:
            return 0, 0.0

        with self.db_lock:
            self._reset_daily_fines_if_needed()
            try:
                fine_column = pd.to_numeric(self.students_db['fine_amount'], errors='coerce').fillna(0)
                total_fine = fine_column.sum()
                violations_today = len(self.fined_students_today)
                # print(f"[DEBUG get_totals] Calculated total_fine: {total_fine}") # Keep if needed
                return violations_today, float(total_fine)
            except Exception as e:
                print(f"Error calculating totals: {e}")
                return 0, 0.0

    def get_recognition_data(self):
        """Returns known names map and embeddings map for face recognition."""
        if not self.is_loaded or not self.known_embeddings:
            return {}, {}
        return self.known_names, self.known_embeddings

    def export_database_csv(self):
        """Exports the current state of the database to a CSV buffer. Thread-safe."""
        if not self.is_loaded or self.students_db is None:
            raise ValueError("Database not loaded, cannot export.")

        with self.db_lock:
            try:
                buffer = io.BytesIO()
                self.students_db.to_csv(buffer, index=False, encoding='utf-8', float_format='%.2f')
                buffer.seek(0)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"student_fines_export_{timestamp}.csv"
                return buffer, filename, 'text/csv'
            except Exception as e:
                print(f"Error exporting database to CSV buffer: {e}")
                raise