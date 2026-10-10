import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
import json
import random
import textwrap

plano = None

with open("./data/planogram.json") as file:
    d = json.load(file)
    plano = d


name = "1751013242"
camname = "6462"

print([i["image_coordinates"] for i in plano["Frutas e Legumes"]["cameras"][camname]["products"]])

image = Image.open(f"./data/cameras/{camname}/{name}.jpg")

draw = ImageDraw.Draw(image)

print()

for i in plano["Frutas e Legumes"]["cameras"][camname]["products"]:
    points = [tuple(v) for v in i["image_coordinates"].values()]

    draw.line(
        points + [points[0]],
        fill=[k for k in ImageColor.colormap][random.randint(0, len(ImageColor.colormap) - 1)],
        width=3,
    )

    label = "\n".join(
        [
            f"Area: {i['area_id']}",
            f"Product: {i['product_id']}",
            *textwrap.wrap(i["product_name"], width=24),
        ]
    )
    center_x = sum(point[0] for point in points) // len(points)
    center_y = sum(point[1] for point in points) // len(points)
    font = ImageFont.load_default()
    text_box = draw.multiline_textbbox((center_x, center_y), label, font=font, anchor="mm", spacing=1)
    padding = 2
    draw.rectangle(
        (
            text_box[0] - padding,
            text_box[1] - padding,
            text_box[2] + padding,
            text_box[3] + padding,
        ),
        fill="black",
    )
    draw.multiline_text(
        (center_x, center_y),
        label,
        font=font,
        fill="white",
        anchor="mm",
        spacing=1,
    )

image.show()