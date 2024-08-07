import os

import cv2
import torch


for stroke_i in range(3):
    image_path = f"test_images/strokes/{stroke_i}.png"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).float()

    mask = torch.zeros_like(img)

    torch.save([mask, img], f"./colab_demo/strokes_{stroke_i}.pth")
