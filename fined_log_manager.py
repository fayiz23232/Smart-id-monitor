# fined_log_manager.py
import csv
import os
import threading
import datetime

class FinedLogManager:
    """Handles writing fine event records to a dedicated CSV file."""

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.file_lock = threading.Lock() # Lock for thread-safe file writing
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Creates the log file and writes the header if it doesn't exist."""
        with self.file_lock:
            # Ensure directory exists
            log_dir = os.path.dirname(self.log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    print(f"[Info] Created directory for fined log: {log_dir}")
                except OSError as e:
                    print(f"[ERROR] Failed to create directory for fined log '{log_dir}': {e}")
                    # Decide how to handle this - maybe disable logging?

            # Check if file exists or is empty to write header
            header_needed = not os.path.exists(self.log_file_path) or os.path.getsize(self.log_file_path) == 0
            if header_needed:
                try:
                    with open(self.log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        header = ["student_id", "name", "timestamp", "image_filename"]
                        writer.writerow(header)
                    print(f"[Info] Initialized fined log file: {self.log_file_path}")
                except IOError as e:
                    print(f"[ERROR] Failed to initialize fined log file '{self.log_file_path}': {e}")

    def log_fine(self, student_id, name, timestamp, image_filename):
        """Appends a fine record to the CSV file."""
        with self.file_lock:
            try:
                # Format timestamp for consistency
                ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(timestamp, datetime.datetime) else str(timestamp)

                # Prepare row data
                row = [student_id, name, ts_str, image_filename]

                # Append to the CSV file
                with open(self.log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(row)
                # print(f"[Log OK] Logged fine for {student_id} to {self.log_file_path}") # Optional success log

            except IOError as e:
                print(f"[ERROR] Failed to write to fined log file '{self.log_file_path}': {e}")
            except Exception as e:
                 print(f"[ERROR] Unexpected error writing to fined log: {e}")