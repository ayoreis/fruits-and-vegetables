import cv2
import numpy as np
from pathlib import Path

base_dir = Path(__file__).resolve().parent
image_path = base_dir / "data" / "areas" / "6385" / "1751001178" / "6385_1.jpg"
output_dir = base_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read input image: {image_path}")

colors = [[17,44],[89,255],[0,255]]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

hsv = cv2.cvtColor(image, cv2.cvtColor.COLOR_BGR2HSV)
for i in colors[1::2]:
    
