import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from torch import nn

trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']


# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

            # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

def predict(imagepath):
    image = Image.open(imagepath).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze_(0) #adds one more dimention = [32,3,224,224]

    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(
            torch.load("/Users/apple/Desktop/Car Damage DL/streamlit app/model/saved_model.pth",map_location=torch.device('cpu')))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)#[[12,15,,9,5,,8,20]]
        _, predicted = torch.max(output, 1)#[20,5]
        return class_names[predicted.item()]





    return "Front Breakage"