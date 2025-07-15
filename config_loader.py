# config_loader.py
import configparser
import os

def load_config(config_file='config.ini'):
    """Loads configuration from an INI file."""
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    try:
        config.read(config_file)
        settings = {}
        # [SETTINGS]
        settings['camera_index'] = config.getint('SETTINGS', 'camera_index', fallback=0)
        # ---- ADD THIS LINE ----
        print(f"[DEBUG] Config Loader: Read camera_index = {settings['camera_index']} (Type: {type(settings['camera_index'])})")
        settings['face_match_threshold'] = config.getfloat('SETTINGS', 'face_match_threshold', fallback=0.4)
        settings['person_conf_threshold'] = config.getfloat('SETTINGS', 'person_conf_threshold', fallback=0.6)
        settings['id_card_conf_threshold'] = config.getfloat('SETTINGS', 'id_card_conf_threshold', fallback=0.5)
        # Ignored source settings for web UI mode
        settings['source'] = config.get('SETTINGS', 'source', fallback='camera') # Keep for potential future use or info
        settings['video_path'] = config.get('SETTINGS', 'video_path', fallback='')
        settings['image_path'] = config.get('SETTINGS', 'image_path', fallback='')

        # [MODELS]
        settings['person_model_path'] = config.get('MODELS', 'person_model', fallback='yolov8n.pt')
        settings['id_card_model_path'] = config.get('MODELS', 'id_card_model', fallback='id_card_detector.pt')
        # settings['face_recognition_method'] = config.get('MODELS', 'face_recognition', fallback='template_matching') # Keep if needed later

        # [DATABASE]
        settings['csv_file'] = config.get('DATABASE', 'csv_file', fallback='students_db.csv')

        # [FINE]
        settings['fine_amount'] = config.getfloat('FINE', 'fine_amount', fallback=10.0)

        # [ARCFACE]
        settings['model_name'] = config.get('ARCFACE', 'model_name', fallback='buffalo_l') # MAKE SURE THIS IS PRESENT
        settings['similarity_threshold'] = config.getfloat('ARCFACE', 'similarity_threshold', fallback=0.5) # MAKE SURE THIS IS PRESENT
        settings['providers'] = config.get('ARCFACE', 'providers', fallback='CPU') # MAKE SURE THIS IS PRESENT

        # [EMAIL]
        settings['email_enabled'] = config.getboolean('EMAIL', 'enabled', fallback=False)
        settings['smtp_server'] = config.get('EMAIL', 'smtp_server', fallback=None)
        settings['smtp_port'] = config.getint('EMAIL', 'smtp_port', fallback=587)
        settings['sender_email'] = config.get('EMAIL', 'sender_email', fallback=None)
        settings['sender_password'] = config.get('EMAIL', 'sender_password', fallback=None)
        settings['use_tls'] = config.getboolean('EMAIL', 'use_tls', fallback=True)
        settings['email_subject'] = config.get('EMAIL', 'subject', fallback='Fine Notification')
        # --------------

        # [LOGGING]
        settings['fined_images_dir'] = config.get('LOGGING', 'fined_images_dir', fallback='fined_student_images')
        settings['fined_log_csv'] = config.get('LOGGING', 'fined_log_csv', fallback='fined_log.csv')
        # --------------

        print("--- Configuration Loaded ---")
        # ... (printing logic - maybe add email sender?) ...
        print("--------------------------")

        print("--- Configuration Loaded ---")
        for section in config.sections():
            print(f"  [{section}]")
            for key, value in config.items(section):
                # Clarify camera_index usage for web mode
                if section == 'SETTINGS' and key == 'camera_index':
                    print(f"    {key} = {value} (Note: Used as *preference* for camera selection in browser)")
                elif section == 'SETTINGS' and key in ('source', 'video_path', 'image_path'):
                    print(f"    {key} = {value} (Note: Not directly used in web UI mode)")
                else:
                    print(f"    {key} = {value}")
        print("--------------------------")
        # --- ADD THIS ---
        print("\n[DEBUG] Final structure of 'settings' dictionary being returned:")
        import json
        print(json.dumps(settings, indent=4))
        print("---------------------------------------\n")
        # --------------
        return settings

    except configparser.Error as e:
        print(f"Error reading configuration file '{config_file}': {e}")
        raise
    except ValueError as e:
        print(f"Error converting configuration value in '{config_file}': {e}")
        raise

# Example usage (optional, for testing this module directly)
# if __name__ == '__main__':
#     try:
#         config = load_config()
#         print("\nLoaded Config Dictionary:")
#         print(config)
#     except Exception as e:
#         print(f"Failed to load config: {e}")