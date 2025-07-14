from flask import Flask, render_template, request, abort, jsonify
from docx import Document
from reportlab.pdfgen import canvas
import numpy as np
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

        # Apply EasyOCR OCR
        result = reader.readtext(img_cv)
        extracted_text = ' '.join([line[1] for line in result])
        extracted_text=clean_label(extracted_text)
        if not extracted_text.strip():
            extracted_text = "No text detected. Please try a clearer image."

        print(extracted_text)

        # Save DOCX
        doc = Document()
        doc.add_paragraph(extracted_text)
        os.makedirs(os.path.join("static", "outputs"), exist_ok=True)
        docx_path = os.path.join("static", "outputs", "output.docx")
        doc.save(docx_path)

        # Save PDF
        pdf_path = os.path.join("static", "outputs", "output.pdf")
        c = canvas.Canvas(pdf_path)
        textobject = c.beginText(40, 800)
        for line in extracted_text.splitlines():
            textobject.textLine(line)
        c.drawText(textobject)
        c.save()

        # Save TXT
        txt_path = os.path.join("static", "outputs", "output.txt")
        with open(txt_path, "w") as f:
            f.write(extracted_text)

        return jsonify({
            'text': extracted_text,
            'docx_link': '/' + docx_path,
            'pdf_link': '/' + pdf_path,
            'txt_link': '/' + txt_path
        })

    abort(405, description="Method Not Allowed: Only POST is supported.")


def clean_label(label):
    lines = label.splitlines()
    cleaned_lines = [line.lstrip(": ").rstrip() for line in lines]
    return " ".join(cleaned_lines)

if __name__ == '__main__':
    app.run(debug=True)