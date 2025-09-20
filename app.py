from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import torch
import os
import logging
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from model_utils import (
    load_model, preprocess_image, predict_face, 
    get_lfw_class_names, get_top_predictions, validate_image
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Global variables for model and class names
model = None
class_names = None
device = None

def initialize_model():
    """Initialize the face recognition model"""
    global model, class_names, device
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load class names from LFW dataset
        logger.info("Loading LFW class names...")
        try:
            class_names = get_lfw_class_names()
            num_classes = len(class_names)
            logger.info(f"Loaded {num_classes} classes")
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            # Fallback to a default set if there are issues
            num_classes = 62  # Default for LFW with min_faces_per_person=20
            class_names = np.array([f"Person_{i}" for i in range(num_classes)])
            logger.warning(f"Using fallback class names with {num_classes} classes")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Determine which model file to use (prioritize improved transfer model)
        model_files = [
            'best_improved_transfer_model.pth',
            'best_transfer_face_model.pth', 
            'best_face_model.pth'
        ]
        
        model_path = None
        for model_file in model_files:
            full_path = os.path.join(script_dir, model_file)
            if os.path.exists(full_path):
                model_path = full_path
                break
        
        if model_path is None:
            raise FileNotFoundError("No trained model file found!")
        
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path, num_classes, device)
        logger.info("Model loaded successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False

@app.route('/')
def index():
    """Serve the main HTML interface"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(script_dir, 'face_recognition_app.html')
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <h1>Face Recognition API</h1>
        <p>The HTML interface file is not found. Please ensure 'face_recognition_app.html' is in the same directory.</p>
        <p>API Endpoints:</p>
        <ul>
            <li>POST /predict - Upload image for prediction</li>
            <li>GET /health - Check API health</li>
            <li>GET /model-info - Get model information</li>
        </ul>
        """

@app.route('/health')
def health_check():
    """Health check endpoint"""
    global model, class_names
    
    status = {
        'status': 'healthy' if model is not None else 'error',
        'model_loaded': model is not None,
        'num_classes': len(class_names) if class_names is not None else 0,
        'device': str(device) if device else 'unknown'
    }
    
    return jsonify(status)

@app.route('/model-info')
def model_info():
    """Get model information"""
    global model, class_names, device
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    info = {
        'model_type': 'Face Recognition CNN',
        'architecture': 'Transfer Learning with custom CNN',
        'num_classes': len(class_names),
        'input_size': '224x224',
        'device': str(device),
        'classes': class_names.tolist() if class_names is not None else []
    }
    
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    global model, class_names, device
    
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Validate image
        image_file.seek(0)  # Reset file pointer
        is_valid, validation_message = validate_image(image_file)
        
        if not is_valid:
            return jsonify({'error': validation_message}), 400
        
        # Reset file pointer for processing
        image_file.seek(0)
        
        # Preprocess image
        logger.info("Preprocessing image...")
        img_tensor = preprocess_image(image_file)
        
        # Get prediction
        logger.info("Running prediction...")
        predicted_name, confidence_score = predict_face(model, img_tensor, class_names, device)
        
        # Get top 5 predictions for additional context
        top_predictions = get_top_predictions(model, img_tensor, class_names, device, top_k=5)
        
        # Prepare response
        response = {
            'success': True,
            'predicted_name': predicted_name,
            'confidence_score': confidence_score,
            'confidence_percentage': f"{confidence_score * 100:.1f}%",
            'top_predictions': top_predictions,
            'model_info': {
                'num_classes': len(class_names),
                'device': str(device)
            }
        }
        
        logger.info(f"Prediction successful: {predicted_name} ({confidence_score:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict-base64', methods=['POST'])
def predict_base64():
    """Alternative prediction endpoint for base64 encoded images"""
    global model, class_names, device
    
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            # Remove data URL prefix
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_file = BytesIO(image_bytes)
        
        # Validate image
        is_valid, validation_message = validate_image(image_file)
        
        if not is_valid:
            return jsonify({'error': validation_message}), 400
        
        # Reset file pointer
        image_file.seek(0)
        
        # Preprocess and predict
        img_tensor = preprocess_image(image_file)
        predicted_name, confidence_score = predict_face(model, img_tensor, class_names, device)
        top_predictions = get_top_predictions(model, img_tensor, class_names, device, top_k=5)
        
        response = {
            'success': True,
            'predicted_name': predicted_name,
            'confidence_score': confidence_score,
            'confidence_percentage': f"{confidence_score * 100:.1f}%",
            'top_predictions': top_predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Base64 prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize model
    logger.info("Initializing Face Recognition API...")
    if initialize_model():
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to initialize model. Exiting...")
        exit(1)