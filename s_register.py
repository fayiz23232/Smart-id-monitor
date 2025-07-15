import cv2
import numpy as np
import pandas as pd
import configparser
import os

# Load config settings
config = configparser.ConfigParser()
config.read("config.ini")

# Load student details
csv_file = "s_details.csv"
if not os.path.exists(csv_file):
    print(f"❌ Error: {csv_file} not found.")
    exit()

s_details = pd.read_csv(csv_file)

# Check if students_db.csv exists
students_db_file = "students_db.csv"
if os.path.exists(students_db_file):
    students_db = pd.read_csv(students_db_file)
else:
    students_db = pd.DataFrame(columns=["student_id", "name", "branch", "year", "image_path", "fine_amount"]) # Modified column

for _, row in s_details.iterrows():
    try:
        student_id, name, branch, year, image_path = row
    except ValueError:
        print("❌ Error: CSV format incorrect. Ensure 5 columns: ID, Name, Branch, Year, ImagePath")
        continue

    # Read student image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error processing {name}: Unable to read image.")
        continue

    # Check if student already exists
    if student_id in students_db["student_id"].values:
        print(f"✅ Student {name} already in database. Skipping...")
    else:
        new_entry = pd.DataFrame([{
            "student_id": student_id,
            "name": name,
            "branch": branch,
            "year": year,
            "image_path": image_path,  # Save image path
            "fine_amount": 0
        }])

        students_db = pd.concat([students_db, new_entry], ignore_index=True)
        print(f"✅ {name} registered successfully.")

# Save updated database
students_db.to_csv(students_db_file, index=False)
print("✅ Image paths stored in students_db.csv with initial fine 0.")