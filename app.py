from flask import Flask, render_template, request
import numpy as np
import pytesseract
from PIL import Image
import os
import cv2
import easyocr

app = Flask(__name__)
reader = easyocr.Reader(['en'])

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print("Predict route hit")
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the form."

        file = request.files['file']
        
        if file.filename == '':
            return "No selected file."

        # Open image directly from uploaded file stream
        img = Image.open(file.stream)

        # Convert PIL image to OpenCV format (BGR)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply Otsu's thresholding (binary inverse + Otsu)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply pytesseract OCR with config --psm 6
        extracted_text = pytesseract.image_to_string(thresh, config='--psm 6')
        extracted_text=clean_label(extracted_text)
        if not extracted_text.strip():
            extracted_text = "No text detected. Please try a clearer image."

        print(extracted_text)

        return render_template('index.html', text=extracted_text)

    return "Only POST method is supported."


def clean_label(label):
    lines = label.splitlines()
    cleaned_lines = [line.lstrip(": ").rstrip() for line in lines]
    return " ".join(cleaned_lines)

if __name__ == '__main__':
    app.run(debug=True)