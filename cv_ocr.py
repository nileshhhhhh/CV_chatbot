import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Optional: If Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folder paths
pdf_folder = "data"
output_folder = "txt_output"
os.makedirs(output_folder, exist_ok=True)

# Process each PDF in the folder
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_text_path = os.path.join(output_folder, base_name + ".txt")

        print(f"Processing {filename}...")

        try:
            # Step 1: Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=300)

            # Step 2: OCR each image page
            all_text = ""
            for i, image in enumerate(images):
                text = pytesseract.image_to_string(image)
                all_text += f"\n--- Page {i+1} ---\n{text}\n"

            # Step 3: Save to .txt file
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(all_text)

            print(f"Saved output to {output_text_path}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
