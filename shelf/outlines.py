"""Outline every planogram area on the full camera photos, labelled with its ids and product name.

Output layout: <output>/<camera>/<timestamp>.jpg
"""

import argparse
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from planogram import area_quad, load_cameras

COLORS = list(ImageColor.colormap)


def draw_areas(image, products):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for product in products:
        points = [tuple(point) for point in area_quad(product)]

        # Seed by area so each area keeps the same color across images.
        color = random.Random(product["area_id"]).choice(COLORS)
        draw.line(points + [points[0]], fill=color, width=3)

        label = "\n".join(
            [
                f"Area: {product['area_id']}",
                f"Product: {product['product_id']}",
                *textwrap.wrap(product["product_name"], width=24),
            ]
        )
        center_x = sum(point[0] for point in points) // len(points)
        center_y = sum(point[1] for point in points) // len(points)
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
        draw.multiline_text((center_x, center_y), label, font=font, fill="white", anchor="mm", spacing=1)

    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planogram", type=Path, default=Path("input/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("input/cameras"))
    parser.add_argument("--output", type=Path, default=Path("output/outlined"))
    args = parser.parse_args()

    for camera, camera_data in sorted(load_cameras(args.planogram).items()):
        output_dir = args.output / camera
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted((args.cameras / camera).glob("*.jpg")):
            with Image.open(image_path) as image:
                draw_areas(image.convert("RGB"), camera_data["products"]).save(output_dir / image_path.name)
            print(f"{camera}/{image_path.name}")


if __name__ == "__main__":
    main()
