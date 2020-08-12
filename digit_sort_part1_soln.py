'''
digit_sort_part1.py
Math 266
Summer 2020
Project #2

This program  will read in an image of a string of digits 0-9 that are
in reverse order.  The program will sort these digits and return an image of the sorted digits.
To do this the program will disassemble the image into 10 parts, each part containing a digit.
Next we reassemble those parts to create an image of the digits in order.

'''

# Libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio  # pip install imageio if necessary


######################  Part 1. ######################

#  Read in the first image, - pip install imageio if necessary , find the images on canvas

image_1 = imageio.imread('image_1.jpg')

# check to see if the shape of the first image is correct 28 x 280

print(f' The shape of the first image is {image_1.shape} .')

# partition this image into 10 separate images and store these images in a list called
# "image_parts_1".  Use a list comprehension to do this.
# hint:  images_parts_1[:,k:k+28] will return the kth part of  the image image_1

image_parts_1 = [image_1[:, k:k+28] for k in range(0, 280, 28)]

# now put the image back together reversing the order of the parts
# hint: use a for loop and the np.hstack method

image_1_sorted = image_parts_1[9]

for k in range(8, -1, -1):
    image_1_sorted = np.hstack((image_1_sorted, image_parts_1[k]))

# put the two images in a list "img"
image_list = [image_1, image_1_sorted]

# plot the results
fig, ax = plt.subplots(nrows=2, ncols=1)
for k in range(2):
    ax[k].imshow(image_list[k], cmap='gray')
    ax[k].set_xticks([])
    ax[k].set_yticks([])
    ax[k].set_xlabel(f"{['Input image_1', 'Output image_1'][k]}")

plt.suptitle('Figure 1.')

# save the plot
plt.savefig("part1.png")
plt.show()
# to save the new image as jpg

imageio.imwrite('image_1_sorted.jpg', image_1_sorted)
