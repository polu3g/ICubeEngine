from PIL import Image
import pytesseract

# Path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update this path based on your Tesseract installation

# Load the image
image_path = 'prescription_sample.png'
image = Image.open(image_path)

# Perform OCR on the image
extracted_text = pytesseract.image_to_string(image)

# Print the extracted text
print("Extracted Text:\n")
print(extracted_text)
