# ✍️ Handwritten-to-Digital Text Conversion

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-yellow?logo=javascript)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
![HTML5](https://img.shields.io/badge/HTML5-Markup-red?logo=html5)
![CSS3](https://img.shields.io/badge/CSS3-Styles-blue?logo=css3)
![GSAP](https://img.shields.io/badge/GSAP-Animations-green?logo=greensock)

> A **deep learning pipeline** that extracts and digitizes text from handwritten images using **CNN** and **Bidirectional GRU**, deployed on a custom-built frontend for an **interactive handwritten-to-digital conversion experience**.

---

## 🎥 Demo Video
▶ **[Watch on YouTube](https://youtu.be/-mKvonGXq2Y)**

---

## 📸 Demo Preview

### 🎥 Live Demo GIF  
*(Replace `demo.gif` with your actual GIF demo)*  
![Demo GIF](assets/demo.gif)

### 🖼️ Screenshots
| Upload Handwritten Image | Digital Text Output |
|--------------------------|--------------------|
| ![Upload Screenshot](assets/upload.png) | ![Output Screenshot](assets/output.png) |

---

## 🚀 Features

- 🧠 **Deep Learning Model** – Combines **CNN** for visual feature extraction and **Bidirectional GRU** for sequence modeling.
- 📜 **Handwriting Recognition** – Converts handwritten notes into editable digital text.
- 🌐 **Custom Frontend** – Built with HTML, CSS, JavaScript, and GSAP for animations and smooth UI.
- 📂 **Interactive Upload & Processing** – Users can upload images and get instant digitized results.
- 📊 **Visualization** – Displays bounding boxes and recognition overlays for transparency.

---

## 🛠️ Tech Stack

- **Backend / Model**
  - TensorFlow / Keras
  - CNN (Convolutional Neural Network)
  - Bidirectional GRU (Gated Recurrent Unit)
  - Python, NumPy, Pandas
- **Frontend**
  - HTML, CSS, JavaScript
  - GSAP for animations
- **Deployment**
  - Flask / FastAPI (for serving model predictions)
  - Docker (optional for containerization)

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/handwritten-text-conversion.git
cd handwritten-text-conversion
