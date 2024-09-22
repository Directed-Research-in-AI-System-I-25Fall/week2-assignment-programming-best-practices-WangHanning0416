import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification
import torch.nn as nn

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_test = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')

#print(model)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        outputs = model(images)
        _, predicted = torch.max(outputs.logits, 1)
        predicted //= 100
        #print("#predicted:", predicted, "  labels:", labels)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #print("total:", total, "  correct:", correct)

accuracy = correct / total * 100
print(f'Accuracy of the model on the MNIST test images: {accuracy:.2f}%')
