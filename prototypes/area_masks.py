import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
import json
import random
import textwrap
from pathlib import Path

plano = None

base_dir = Path(__file__).resolve().parent.parent

with open(base_dir / "input" / "planogram.json") as file:
    d = json.load(file)
    plano = d


name = "1751013242"
camname = "6462"

# print([i["image_coordinates"] for i in plano["Frutas e Legumes"]["cameras"][camname]["products"]])

image_path = base_dir / "input" / "cameras" / camname / f"{name}.jpg"
output_dir = base_dir / "output" / "area_masks"
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"Could not read input image: {image_path}")


index = 0
for i in plano["Frutas e Legumes"]["cameras"][camname]["products"]:
    index += 1
    points = np.array([v if min(v) > 0 else 0 for v in i["image_coordinates"].values()])
    
    rect = cv2.boundingRect(points)
    x,y,w,h = rect
    
    cropped = image[y:y+h, x:x+w].copy()
    
    points = points - points.min(axis=0)
    
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA) 
    
    dst = cv2.bitwise_and(cropped,cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8)*255
    
    dst2 = bg+ dst
    
    output_images = {
        "cropped": cropped,
        "dst": dst
    }
    for suffix, output_image in output_images.items():
        output_path = output_dir / f"{index}_{suffix}.png"
        if not cv2.imwrite(str(output_path), output_image):
            raise OSError(f"Could not save image: {output_path}")
    print(i)

    