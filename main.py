from PIL import Image

from outline_areas import draw_areas
from planogram import load_cameras

name = "1751013242"
camname = "6462"

products = load_cameras("./data/planogram.json")[camname]["products"]

image = Image.open(f"./data/cameras/{camname}/{name}.jpg")
draw_areas(image, products).show()
