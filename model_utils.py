import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

class FaceCNN(nn.Module):
    """
    Optimized CNN Architecture for face recognition
    Matches the simple CNN architecture from FaceRecognition.ipynb
    """
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()
        # Simplified architecture for faster training
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # Larger kernel, fewer layers
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(4, 4)  # Larger pooling for speed
        
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(4, 4)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling
        
        # Smaller fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Efficient forward pass
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class ImprovedTransferFaceCNN(nn.Module):
    """
    Transfer Learning model with ResNet18 backbone
    Matches the ImprovedTransferFaceCNN from FaceRecognition.ipynb
    """
    def __init__(self, num_classes=62):
        super(ImprovedTransferFaceCNN, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for i, (name, param) in enumerate(self.backbone.named_parameters()):
            if i < 30:
                param.requires_grad = False
        
        # Replace classifier with custom head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, num_classes, device):
    """
    Load the trained face recognition model
    Automatically detects the model type based on the saved state dict
    """
    # Load the state dict to inspect its structure
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if it's a transfer learning model (has 'backbone' in keys)
    if any('backbone' in key for key in state_dict.keys()):
        print(f"Detected transfer learning model (ResNet18-based)")
        model = ImprovedTransferFaceCNN(num_classes)
    else:
        print(f"Detected simple CNN model")
        model = FaceCNN(num_classes)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_image_transform():
    """
    Get the image transformation pipeline used for prediction
    Matches the transform from predict_new_face function
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_image(image_file):
    """
    Preprocess uploaded image for model prediction
    """
    # Open image and convert to RGB
    img = Image.open(image_file).convert('RGB')
    
    # Apply transformation
    transform = get_image_transform()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def predict_face(model, img_tensor, class_names, device):
    """
    Predict face from preprocessed image tensor
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_name = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_name, confidence_score


def get_lfw_class_names():
    """
    Get the class names from LFW dataset
    These are the 62 people with at least 20 images each from the LFW dataset
    """
    # LFW target names for people with min_faces_per_person=20
    lfw_names = [
        'Abdullah Gul', 'Adrien Brody', 'Al Sharpton', 'Alan Greenspan', 
        'Alejandro Toledo', 'Alvaro Uribe', 'Angelina Jolie', 'Arnold Schwarzenegger',
        'Ariel Sharon', 'Ben Affleck', 'Bill Clinton', 'Blair Underwood',
        'Carlos Menem', 'Colin Powell', 'David Beckham', 'Denzel Washington',
        'Donald Rumsfeld', 'George W Bush', 'Gerhard Schroeder', 'Glenn Close',
        'Gray Davis', 'Hamid Karzai', 'Hans Blix', 'Hugo Chavez',
        'Jacques Chirac', 'Jean Chretien', 'Jennifer Aniston', 'Jennifer Capriati',
        'Jennifer Lopez', 'Jeremy Greenstock', 'John Ashcroft', 'John Negroponte',
        'Jose Bono', 'Junichiro Koizumi', 'Kofi Annan', 'Laura Bush',
        'Lleyton Hewitt', 'Luiz Inacio Lula da Silva', 'Mahmoud Abbas', 'Megawati Sukarnoputri',
        'Michael Bloomberg', 'Nestor Kirchner', 'Paul Bremer', 'Pete Sampras',
        'Recep Tayyip Erdogan', 'Ricardo Lagos', 'Rudolph Giuliani', 'Saddam Hussein',
        'Serena Williams', 'Silvio Berlusconi', 'Tiger Woods', 'Tom Daschle',
        'Tom Ridge', 'Tony Blair', 'Vicente Fox', 'Vladimir Putin',
        'Winona Ryder', 'Yasser Arafat', 'Yoriko Kawaguchi', 'Zhang Ziyi',
        'Andre Agassi', 'Dick Cheney'
    ]
    return np.array(lfw_names)


def get_top_predictions(model, img_tensor, class_names, device, top_k=5):
    """
    Get top K predictions with confidence scores
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    predictions = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        name = class_names[idx]
        predictions.append({
            'name': name,
            'confidence': prob,
            'percentage': f"{prob * 100:.1f}%"
        })
    
    return predictions


def validate_image(image_file):
    """
    Validate uploaded image file
    """
    try:
        img = Image.open(image_file)
        
        # Check image format
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            return False, "Unsupported image format. Please upload JPEG or PNG images."
        
        # Check image size
        if img.size[0] < 32 or img.size[1] < 32:
            return False, "Image too small. Please upload images larger than 32x32 pixels."
        
        if img.size[0] > 4096 or img.size[1] > 4096:
            return False, "Image too large. Please upload images smaller than 4096x4096 pixels."
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"