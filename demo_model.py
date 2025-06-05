import torch
from src.models.pointnet2 import PointNet2SemSeg
import numpy as np

# Create model
model = PointNet2SemSeg(num_classes=8, input_channels=6)
print("✅ PointNet++ model loaded successfully!")

# Test with random data
dummy_points = torch.randn(1, 1000, 6)  # 1000 points with XYZ + RGB
predictions = model(dummy_points)
print(f"✅ Model inference successful! Output shape: {predictions.shape}")

# Show class predictions
predicted_classes = predictions.argmax(dim=2)
class_names = ['man-made terrain', 'natural terrain', 'high vegetation',
               'low vegetation', 'buildings', 'hard scape', 
               'scanning artifacts', 'cars']

print("✅ Sample predictions:")
for i in range(min(10, predictions.shape[1])):
    pred_class = predicted_classes[0, i].item()
    print(f"Point {i}: {class_names[pred_class]}")