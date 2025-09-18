# Face Recognition with Transfer Learning

A deep learning project implementing face recognition using CNN and Transfer Learning with ResNet18, achieving **93% accuracy** on the LFW dataset.

## 🎯 Overview

This project demonstrates the power of transfer learning for face recognition tasks. It compares a basic CNN implementation with an advanced ResNet18-based transfer learning approach, showing significant performance improvements.

## 📊 Results

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Basic CNN | ~75% | ~15 minutes |
| **Transfer Learning (ResNet18)** | **93.06%** | **36 minutes** |

## 🚀 Features

- **Dual Architecture Comparison**: Basic CNN vs Transfer Learning
- **High Performance**: 93% accuracy on face recognition
- **Comprehensive Evaluation**: Classification reports, confusion matrices, sample predictions
- **Clean Implementation**: Well-documented, production-ready code
- **GPU Optimized**: CUDA support for faster training

## 📁 Dataset

Uses the **LFW (Labeled Faces in the Wild)** dataset:
- **62 famous people** (politicians, athletes, celebrities)
- **20+ images per person** for robust training
- **Automatic download** via scikit-learn

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/face-recognition-transfer-learning.git
cd face-recognition-transfer-learning

# Install dependencies
pip install torch torchvision scikit-learn matplotlib seaborn tqdm
```

## 💻 Usage

1. **Open the Jupyter Notebook:**
```bash
jupyter notebook FaceRecognition.ipynb
```

2. **Run all cells** to:
   - Load and preprocess the LFW dataset
   - Train the basic CNN model
   - Train the transfer learning model  
   - Compare results and visualize predictions

3. **Use the trained model:**
```python
# Predict a new face (must be one of the 62 people from LFW)
predicted_name, confidence = predict_new_face(
    improved_transfer_model, 
    'path/to/image.jpg', 
    lfw.target_names
)
print(f"Predicted: {predicted_name} (Confidence: {confidence:.3f})")
```

## 🏗️ Architecture

### Basic CNN
- 3 Convolutional layers with BatchNorm
- Adaptive pooling and dropout
- ~1.2M parameters

### Transfer Learning (ResNet18)  
- Pre-trained ResNet18 backbone
- Frozen early layers (transfer learning)
- Custom classifier head
- ImageNet normalization
- ~11M parameters (only ~30% trainable)

## 📈 Performance Analysis

The transfer learning approach shows dramatic improvements:

- **+18% accuracy gain** over basic CNN
- **Better generalization** with pre-trained features
- **Faster convergence** due to transfer learning
- **Higher confidence scores** in predictions

## 🎯 Recognized People

The model can identify 62 famous individuals including:
- **Politicians**: George W Bush, Tony Blair, Colin Powell, Jacques Chirac
- **Athletes**: Serena Williams, David Beckham, Tiger Woods
- **Celebrities**: Angelina Jolie, Jennifer Aniston, Winona Ryder
- **World Leaders**: Vladimir Putin, Hugo Chavez, Junichiro Koizumi

## 🔧 Technical Details

- **Framework**: PyTorch + torchvision
- **Image Size**: 224x224 (ResNet18), 64x64 (Basic CNN)
- **Batch Size**: 32 (Transfer Learning), 64 (Basic CNN)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, BatchNorm, Weight decay
- **Early Stopping**: Prevents overfitting

## 📊 Sample Results

The transfer learning model achieves excellent performance:
- **Batch Accuracy**: 96.9%
- **Average Confidence**: 0.923
- **Precision/Recall**: 0.90+ for most classes

## ⚠️ Limitations

- **Closed-set recognition**: Only works for the 62 trained people
- **No unknown person detection**: Will classify any face as one of the 62
- **Single face per image**: Doesn't handle multiple faces
- **Domain specific**: Optimized for these particular individuals

## 🚀 Future Improvements

- **Open-set recognition**: Detect unknown faces
- **Face detection**: Handle multiple faces in images
- **Real-time inference**: Optimize for production deployment
- **Data augmentation**: Improve robustness
- **Ensemble methods**: Combine multiple models

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Created by [Your Name] - feel free to contact me!

---

⭐ **Star this repo** if you found it helpful!