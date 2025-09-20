# 🎭 LFW Face Recognition Web App

A complete **face recognition web application** using deep learning with PyTorch and Flask. Features a professional web interface with 99% accuracy on the LFW dataset using ResNet18 transfer learning.

## � Live Demo

![Face Recognition Demo](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)
![Flask](https://img.shields.io/badge/Flask-2.x-green)

**🚀 Quick Start:** `python app.py` → Open `http://localhost:5000`

## ✨ Features

### 🎯 **Core Capabilities**
- **99% Accuracy** - ResNet18 transfer learning model
- **62 Famous People** - Recognizes celebrities, politicians, athletes
- **Real-time Predictions** - Instant face recognition with confidence scores
- **Professional Web UI** - Modern, responsive design with dark mode

### �️ **Web Application**
- **Drag & Drop Upload** - Easy image upload interface
- **Live Confidence Meter** - Visual confidence scoring
- **Mobile Responsive** - Works on all devices
- **REST API** - JSON endpoints for integration

### 🧠 **Machine Learning**
- **Transfer Learning** - Pre-trained ResNet18 backbone
- **Data Augmentation** - Robust training pipeline
- **Model Comparison** - Basic CNN vs Transfer Learning analysis
- **Performance Metrics** - Comprehensive evaluation tools

## 🏗️ Project Structure

```
FaceRecognition/
├── 📊 FaceRecognition.ipynb          # Training notebook & analysis
├── 🌐 app.py                         # Flask web server
├── 🧠 model_utils.py                 # Model utilities & classes
├── 🎨 face_recognition_app.html      # Web interface
├── 🏆 best_improved_transfer_model.pth  # Trained model (99% accuracy)
├── 📋 requirements.txt               # Dependencies
├── 📄 README.md                      # This file
├── 📜 LICENSE                        # MIT License
└── 🔧 .gitignore                     # Git ignore rules
```

## � Quick Start

### 1️⃣ **Clone Repository**
```bash
git clone https://github.com/Chanu716/face-recognition-transfer-learning.git
cd face-recognition-transfer-learning
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Launch Web App**
```bash
python app.py
```

### 4️⃣ **Open Browser**
Navigate to `http://localhost:5000`

## 💻 Usage Guide

### 🌐 **Web Interface**
1. **Upload Image** - Drag & drop or click to select
2. **Get Prediction** - Instant face recognition results
3. **View Confidence** - Color-coded confidence meter
4. **Try Another** - Reset and test more images

### 🔧 **API Endpoints**

#### **Predict Face**
```bash
POST /predict
Content-Type: multipart/form-data

# Response
{
  "success": true,
  "prediction": "George_W_Bush",
  "confidence": 0.987,
  "top_predictions": [
    {"name": "George_W_Bush", "confidence": 0.987},
    {"name": "Tony_Blair", "confidence": 0.008}
  ]
}
```

#### **Health Check**
```bash
GET /health
# Response: {"status": "healthy", "model_loaded": true}
```

### 🐍 **Python API**
```python
from model_utils import load_model, predict_face, get_lfw_class_names

# Load model
model = load_model('best_improved_transfer_model.pth')
class_names = get_lfw_class_names()

# Predict
prediction, confidence = predict_face(model, 'image.jpg', class_names)
print(f"Prediction: {prediction} (Confidence: {confidence:.3f})")
```

## � Recognized People (62 Total)

### 👑 **Politicians**
- George W Bush, Tony Blair, Colin Powell, Jacques Chirac
- Vladimir Putin, Hugo Chavez, Junichiro Koizumi, Donald Rumsfeld

### 🎬 **Celebrities** 
- Angelina Jolie, Jennifer Aniston, Winona Ryder, Jennifer Lopez
- Brad Pitt, Tom Cruise, Johnny Depp, Leonardo DiCaprio

### ⚽ **Athletes**
- Serena Williams, David Beckham, Tiger Woods, Andre Agassi

### 📺 **Media & Others**
- Condoleezza Rice, Kofi Annan, John Ashcroft, Laura Bush

**📋 [View Complete List](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)**

## 🏆 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **99.0%** |
| **Model** | ResNet18 Transfer Learning |
| **Dataset** | LFW (Labeled Faces in the Wild) |
| **Classes** | 62 Famous People |
| **Input Size** | 224x224 RGB |
| **Parameters** | ~11M (30% trainable) |

### 📊 **Architecture Comparison**

| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| Basic CNN | ~75% | ~15 min | ~1.2M |
| **ResNet18 Transfer** | **99%** | **36 min** | **~11M** |

## �️ Technical Details

### 🧠 **Model Architecture**
- **Backbone:** Pre-trained ResNet18 (ImageNet)
- **Transfer Learning:** Frozen early layers
- **Custom Head:** Fully connected classifier
- **Optimization:** Adam with learning rate scheduling
- **Regularization:** Dropout, BatchNorm, Weight decay

### 🌐 **Web Stack**
- **Backend:** Flask 2.x with CORS support
- **Frontend:** HTML5, TailwindCSS, Vanilla JavaScript
- **Features:** Drag-drop upload, real-time predictions
- **Responsive:** Mobile-first design with dark mode

### 📊 **Data Pipeline**
- **Dataset:** LFW via scikit-learn
- **Preprocessing:** Resize, normalize, tensor conversion
- **Augmentation:** Random transforms for robustness
- **Validation:** Stratified train/test split

## 🔧 Development

### 📓 **Jupyter Notebook**
Explore the complete training process:
```bash
jupyter notebook FaceRecognition.ipynb
```

### 🔨 **Local Development**
```bash
# Install in development mode
pip install -e .

# Run with hot reload
export FLASK_ENV=development
python app.py
```

### 🧪 **Testing**
```python
# Test model loading
from model_utils import load_model
model = load_model('best_improved_transfer_model.pth')

# Test prediction
prediction, confidence = predict_face(model, 'test_image.jpg', class_names)
```

## ⚠️ Limitations

- **Closed-Set Recognition:** Only recognizes the 62 trained people
- **Single Face:** Designed for one face per image
- **Domain Specific:** Optimized for LFW dataset individuals
- **No Unknown Detection:** Will classify any face as one of the 62

## 🚀 Future Enhancements

- [ ] **Open-set recognition** for unknown faces
- [ ] **Multi-face detection** in single image
- [ ] **Real-time video** processing
- [ ] **Model optimization** for mobile deployment
- [ ] **Ensemble methods** for improved accuracy

## � Deployment

### 🐳 **Docker** (Recommended)
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

### ☁️ **Cloud Platforms**
- **Heroku:** Ready with `requirements.txt`
- **AWS/GCP:** Deploy with gunicorn
- **Vercel/Netlify:** Static frontend + API

## 📊 Dataset Information

**LFW (Labeled Faces in the Wild)**
- **Source:** [Kaggle LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- **Size:** 13,000+ images
- **Classes:** 62 famous individuals
- **Format:** 224x224 RGB images
- **Split:** 80% train, 20% test

## 👨‍💻 Developer Information

**Name:** Chanikya  
**Email:** [karrichanikya@gmail.com](mailto:karrichanikya@gmail.com)  
**Phone:** [+91 9182789929](tel:+919182789929)

### 🎓 **About This Project**
This project is developed for **educational and learning purposes** to demonstrate:
- Deep learning with PyTorch
- Transfer learning techniques
- Web application development with Flask
- Machine learning model deployment
- Full-stack development skills

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## ⭐ Support

If you found this project helpful:

1. **⭐ Star this repository**
2. **🍴 Fork for your own projects**
3. **📢 Share with others**
4. **🐛 Report issues**
5. **💡 Suggest improvements**

---

<div align="center">

### 🚀 **Ready to recognize faces? Start now!**

```bash
git clone https://github.com/Chanu716/face-recognition-transfer-learning.git
cd face-recognition-transfer-learning
pip install -r requirements.txt
python app.py
```

**[🌐 Open http://localhost:5000](http://localhost:5000)**

</div>