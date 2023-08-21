import numpy as np
import pylab as plt

# Load the image
from PIL import Image
from PIL import ImageDraw

from variables import WIDTH, HEIGHT

img = Image.open("test_data/94/100.jpg")

# Grid lines at these intervals (in pixels)
# dx and dy can be different
dx, dy = 8, 8

image = img.resize((WIDTH, HEIGHT))

# Draw some lines
draw = ImageDraw.Draw(image)
y_start = 0
y_end = image.height
step_size = 8
for x in range(0, image.width, step_size):
    line = ((x, y_start), (x, y_end))
    draw.line(line, fill=128)
x_start = 0
x_end = image.width
for y in range(0, image.height, step_size):
    line = ((x_start, y), (x_end, y))
    draw.line(line, fill=128)
del draw

# Show the result
plt.imshow(image)
plt.show()