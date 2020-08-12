'''
digit_sort.py
Math 266
Summer 2020
Project #2

This program  has two parts. In part 1. we read in an image of a string of digits 0-9 that are
in reverse order.  The program will sort these digits and return an image of the sorted digits.
To do this the program will disassemble the image into 10 parts, then reassemble those parts
to create an image of the digits in order. In part 2. the program will read in  an image showing a
string of random digits it random order. The program will sort these digits as wel 
and return an image of the sorted digits.  Part 2. will require that we create a function that will
recognize the digit value in an image.
'''
# Libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio  # pip install imageio if necessary


######################  Part 1. ######################

#  Read in the first image, - pip install imageio if necessary , find the images on canvas

image_1 = imageio.imread('../data/image_1.jpg')

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
fig, ax = plt.subplots(nrows=4, ncols=1)
for k in range(2):
    ax[k].imshow(image_list[k], cmap='gray')
    ax[k].set_xticks([])
    ax[k].set_yticks([])
    ax[k].set_xlabel(f"{['Input image_1', 'Output image_1'][k]}")



# *************************************  Part 2. ***************************************



# load in the parameter matrix and name it beta_matrix, find the matrix on canvas
beta_matrix = np.load('../data/beta_matrix.npy')


# confirm the size of the parameter matrix is 285x10
print(f' The shape of the beta_matrix is {beta_matrix.shape} .')

# the function below is requied for part  2.
def digit_recognizer(img, parameter_matrix):
    '''
    This function takes in an image array and parameter matrix and returns the digit value
    in the image
    '''
# first we need to flatten the image into a 1d array.  Some call this "unrolling " the image
# hint reseach the method "flatten"
    img = img.flatten()

# Now create the "feature vector" for the input image by appending a 1 to the flattened array
    img = np.append(img, 1)

# Determine the output "label" or y value for the feature vector input. This is done using the model
# beta_matrix.  Call the  output y. y will be a 10-vector, with the position with the largest entry
# representing the digit value.

#  for example if y = [0.05, 0.9 , 0.83, 2.2 , 1.07, 0.33, 0.49, 0.23, 0.46, 0.73] then the digit is "3"

    y = img @ parameter_matrix

    # now return the positon in y that has the maximum value.  Look at numpy's argmax
    return np.argmax(y)

# Read in image 2 and "sort" it

image_2 = imageio.imread('../data/image_2.jpg')

# as for part 1. partition this image into 10 parts

image_parts_2 = [image_2[:, k:k+28] for k in range(0, 280, 28)]

# Now we need to create a list that gives the 10 digit values for 10 image parts

digit_values = []  # initialize the list, we will use the append method to add digit values

# Call the funcion "digit_recognizer" 10 times and append the return value to the list digit_values

for k in range(len(image_parts_2)):
    digit_values.append(digit_recognizer(image_parts_2[k], beta_matrix))

# Whew! we are almost there.   We need to sort the list of image parts in the right order before we reassemble.
# We will use some "fancy indexing" to do this. For example the command:
# image_parts[[1,3,5,7,9,2,4,6,8,0]] will return the list with the components in the order of the indices, that is,
# item #1 then item #3 the item #5 etc. We just need to determine the correct list of indices.
# to determine how to find the correct list of indices look the the numpy method "np.argsort"

sorted_indices = np.argsort(digit_values)

# Now rearange the list of images in the correct order using the sorted_indices and fancy indexing 
image_parts_2 = np.array(image_parts_2)
sorted_parts = image_parts_2[sorted_indices]

# now reassemble the sorted image from the parts like you did in part 1.

image_2_sorted = sorted_parts[0]

for k in range(1,9):
    image_2_sorted = np.hstack((image_2_sorted, sorted_parts[k]))



# put the two images in a list "img"
image_list = [image_2, image_2_sorted]

# plot the results we are adding on to the subplot created earlier

for k in range(2,4):
    ax[k].imshow(image_list[k-2], cmap='gray')
    ax[k].set_xticks([])
    ax[k].set_yticks([])
    ax[k].set_xlabel(f"{['Input image_2', 'Output image_2'][k-2]}")

plt.suptitle(f" Figure 2.")   

# save your results
plt.savefig("output.jpg")

