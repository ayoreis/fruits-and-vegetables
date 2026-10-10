"""Reading data/planogram.json."""

import json

CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")


def load_cameras(planogram_path):
    with open(planogram_path) as file:
        return json.load(file)["Frutas e Legumes"]["cameras"]


def area_quad(area):
    """The area's corners in clockwise order from the top left.

    The corner keys are not always in the same order in the JSON, so they are looked up by name.
    """
    return [area["image_coordinates"][corner] for corner in CORNERS]
