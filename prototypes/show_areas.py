from PIL import Image

from shelf.outlines import draw_areas
from shelf.planogram import load_cameras

name = "1751013242"
camname = "6462"

products = load_cameras("./input/planogram.json")[camname]["products"]

image = Image.open(f"./input/cameras/{camname}/{name}.jpg")
draw_areas(image, products).show()
