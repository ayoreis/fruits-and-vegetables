from collections import Counter

from PIL import Image


def top_3_colors(image_path):
	"""Return the three most common colors in an image as HSV tuples."""
	with Image.open(image_path) as image:
		colors = Counter(image.convert("HSV").getdata())

	return [color for color, _ in colors.most_common(3)]


