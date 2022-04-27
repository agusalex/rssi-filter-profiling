# Evan Russenberger-Rosica
# Create a Grid/Matrix of Images
import glob
import os
from math import ceil

from PIL import Image

if __name__ == '__main__':
    PATH = r"."

    frame_width = 1920
    images_per_row = 3
    padding = 2

    os.chdir(PATH)

    images = glob.glob("*.png")

    img_width, img_height = Image.open(images[0]).size
    scaled_img_width = ceil(img_width)
    scaled_img_height = ceil(img_height)

    number_of_rows = ceil(len(images) / images_per_row)
    frame_height = ceil(img_height * number_of_rows)

    new_im = Image.new('RGB', (frame_width, frame_height))

    i, j = 0, 0
    for num, im in enumerate(images):
        if num % images_per_row == 0:
            i = 0
        im = Image.open(im)
        # Here I resize my opened image, so it is no bigger than 100,100
        im.thumbnail((scaled_img_width, scaled_img_height))
        # Iterate through a 4 by 4 grid with 100 spacing, to place my image
        y_cord = (j // images_per_row) * scaled_img_height
        new_im.paste(im, (i, y_cord))
        print(i, y_cord)
        i = (i + scaled_img_width) + padding
        j += 1

    new_im.show()
    new_im.save("out.jpg", "JPEG", quality=80, optimize=True, progressive=True)
