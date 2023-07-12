import torch
import urllib
from PIL import Image
from torchvision import transforms
from torchvision.models import *

#sanity check data preparation
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")

try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# filename = 'data/oboe-scaled.jpg'
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# lenet_model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
class_file = 'imagenet_classes.txt'
class_list = []
with open(class_file, 'r') as file:
    class_list = file.readlines()
    
model_dict = {'googlenet':  GoogLeNet_Weights, 
              'resnet18':   ResNet18_Weights, 
              'resnet50':   ResNet50_Weights, 
              'resnet101':  ResNet101_Weights, 
              'vgg16':      VGG16_Weights}

for modelname in model_dict.keys():
    model = torch.hub.load('pytorch/vision:v0.10.0', modelname, weights=model_dict[modelname].IMAGENET1K_V1)
    model.eval()

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    assert len(class_list) == output.shape[-1]

    print(f"model {modelname} pred: \t class {class_list[torch.argmax(probabilities).item()]} \t\t\t with {torch.max(probabilities)*100:.4f}%")