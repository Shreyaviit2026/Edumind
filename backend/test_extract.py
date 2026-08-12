import sys
import os

# Add backend directory to sys.path
sys.path.append(os.path.abspath('.'))

from utils.text_extractor import extract_text

file_path = r"C:\Edumind\AI-Powered-Student-Learning-Assistant\backend\storage\subjects\Cloud\Unit 1\docs\CC_Unit_I.pdf"
try:
    print(f"Extracting from: {file_path}")
    text = extract_text(file_path)
    print(f"Extracted {len(text)} characters.")
except Exception as e:
    import traceback
    traceback.print_exc()
