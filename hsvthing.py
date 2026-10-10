# Source - https://stackoverflow.com/a/71625050
# Posted by nathancy, modified by community. See post 'Timeline' for change history
# Retrieved 2026-10-10, License - CC BY-SA 4.0

import cv2
import numpy as np
from collections import Counter

from PIL import Image


def nothing(x):
    pass


imgpath = 'data/areas/6215/1751001268/6215_10.jpg'

image = cv2.imread(imgpath)

# Create a window
cv2.namedWindow('image')


def top_colors(image_path):
    """Return the three most common, sufficiently distinct OpenCV hue values."""
    with Image.open(image_path) as image:
        pixels = image.convert("HSV").getdata()

    # Pillow stores hue in 0..255, while OpenCV uses 0..179.
    hue_counts = Counter((pixel[0] * 180) // 256 for pixel in pixels)
    if not hue_counts:
        return [0, 0, 0]

    # Do not merge adjacent occupied bins: photographs commonly contain a
    # continuous hue range, which otherwise produces just one result.
    selected = []
    for hue, _count in hue_counts.most_common():
        if all(min((hue - other) % 180, (other - hue) % 180) >= 12 and hue
               for other in selected):
            selected.append(hue)
            print("f",hue, selected)
        if len(selected) == 2:
            break
    
    
    return selected



topcolors = top_colors(imgpath)
print("a", topcolors)



"""
# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)
"""
# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0


hWiggle = 10
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

for i in topcolors:
    lower = np.array([i - hWiggle, 80, 0])
    upper = np.array([i + hWiggle, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    combined_mask = cv2.bitwise_or(combined_mask, mask)


def fill_small_enclosed_areas(mask, max_area=300):
    """Fill small background regions fully enclosed by the color mask."""
    background = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        background, connectivity=8
    )
    height, width = mask.shape
    filled_mask = mask.copy()

    for label in range(1, count):
        x, y, region_width, region_height, area = stats[label]
        touches_border = (
            x == 0
            or y == 0
            or x + region_width == width
            or y + region_height == height
        )
        if not touches_border and area <= max_area:
            filled_mask[labels == label] = 255

    return filled_mask


# Include a one-pixel border, then fill only small holes enclosed by found colors.
combined_mask = cv2.dilate(
    combined_mask,
    np.ones((3, 3), dtype=np.uint8),
    iterations=1,
)
combined_mask = fill_small_enclosed_areas(combined_mask, max_area=100)

result = cv2.bitwise_and(image, image, mask=combined_mask)


total_pixels = image.shape[0] * image.shape[1]
black_pixels = np.count_nonzero(np.all(result == 0, axis=2))
print(f"Total pixels: {total_pixels}")
print(f"Black pixels: {black_pixels}")
print("percent", 1 - black_pixels/total_pixels)

cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
cv2.setTrackbarPos('HMin', 'image', hMin)
cv2.setTrackbarPos('SMin', 'image', sMin)
cv2.setTrackbarPos('VMin', 'image', vMin)
cv2.setTrackbarPos('HMax', 'image', hMax)
cv2.setTrackbarPos('SMax', 'image', sMax)
cv2.setTrackbarPos('VMax', 'image', vMax)

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin', 'image')
    sMin = cv2.getTrackbarPos('SMin', 'image')
    vMin = cv2.getTrackbarPos('VMin', 'image')
    hMax = cv2.getTrackbarPos('HMax', 'image')
    sMax = cv2.getTrackbarPos('SMax', 'image')
    vMax = cv2.getTrackbarPos('VMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
"""