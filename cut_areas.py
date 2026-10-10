"""Cut every planogram area out of the camera photos and stretch it into a rectangle.

Output layout: <output>/<camera>/<timestamp>/<area_id>.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from planogram import area_quad, load_cameras


def rectangle_size(quad):
    top_left, top_right, bottom_right, bottom_left = quad
    width = max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    height = max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    return round(width), round(height)


def cut_area(image, quad):
    width, height = rectangle_size(quad)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    # Parts of the area outside the image (negative coordinates) come out black.
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_CONSTANT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planogram", type=Path, default=Path("data/planogram.json"))
    parser.add_argument("--cameras", type=Path, default=Path("data/cameras"))
    parser.add_argument("--output", type=Path, default=Path("data/areas"))
    args = parser.parse_args()

    for camera, camera_data in sorted(load_cameras(args.planogram).items()):
        areas = [(area["area_id"], np.array(area_quad(area), dtype=np.float32)) for area in camera_data["products"]]

        for image_path in sorted((args.cameras / camera).glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image {image_path}")
                continue

            output_dir = args.output / camera / image_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            for area_id, quad in areas:
                cv2.imwrite(str(output_dir / f"{area_id}.jpg"), cut_area(image, quad))

            print(f"{camera}/{image_path.name}: {len(areas)} areas")


if __name__ == "__main__":
    main()
