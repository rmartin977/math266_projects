'''
Project #4
Math 266 Linear Algebra
Enter your name here 
Summer 2020
'''

#import libraries

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt 
import os
plt.ion()

# ========== Part 1: Load the movie ratings dataset  ============

ratings_matrix = np.load("ratings_matrix.npy")
os.system('clear')

# confirm the shape of the matrix is (1682,943)

print(f'\nThe size of the ratings_matrix is {ratings_matrix.shape}\n')  

# ratings_matrix is a 1682x943 matrix, containing ratings(1-5) of 1682 movies by
# 943 users.  A 0 rating indicates that the user did not rate that movie.

# load the python dictionary that we will use to map movie id's to movie titles
# note: to determine the title of a movie given its id we type in dictionary[id]
# where id the the movie id number
# for example: dictionary[0] will return "Toy Story (1995)""

dictionary = np.load("dictionary.npy",allow_pickle=True).item()

# let us display the first 5 entries in the dictionary

print('\nThe first five movies in the dictionary.\n'.title())

for k in range(5):
    print(f'{k}.  {dictionary[k]}')

# Your output should appear as below. Notice again the movie Toy Story has movie id 0. 
# So the first row in the ratings_matrix corresponds to Toy  Story.

# 0.  Toy Story (1995)
# 1.  GoldenEye (1995)
# 2.  Four Rooms (1995)
# 3.  Get Shorty (1995)
# 4.  Copycat (1995)

input('press a key to continue')

# Let us visualize the ratings_matrix

plt.close()
plt.imshow(ratings_matrix,cmap='gray')
plt.xlabel('Users')
plt.ylabel('Movies')
plt.show()

input('press return to continue')

# ========== Part 2: Compute some statistics on the ratings_matrix  ============

# To find the average rating for the movie Toy Story we need to ignore the entries in the row with a 
# zero.

# first find column indices with non zero entries on row 0
indices = ratings_matrix[0,:]>0

average = ratings_matrix[0,indices].mean()
print(f'\n\nThe average rating for Toy Story is {average.round(2)}')

# We can find the total points for a movie by just summing the row for that movie
# Let us find the total score for Toy Story

total_score = ratings_matrix[0,:].sum()
print(f'\nThe total score for Toy Story is {total_score}')

# Your turn -- Find the five top movies in terms of total score -- 
# print out the five move titles
# once you have a list of the id's for the top five movies use a for loop
# like above to print out the titles.

print('\n\nThe five movies with largest totals scores:\n'.title())

######  Your code here  #######

scores = ratings_matrix.sum(1)
max_5 = np.argsort(scores)[-5:][::-1]
for id in max_5:
    print(dictionary[id])

# ========== Part 3: Recommender System  ============

# In this part of the exercise we will implement a simple movie recommender system using
# cosine similarity.

# If two vectors are normalized (norm=1) then the cosine similarity between the two vectors
# is computed by simply computing the dot product of the vectors.  This dot product returns
# a scalar that will give us a measure of how similar the two vectors are. We thus can determine
# how similar two movies are by forming the dot product of the normed feature vectors for each 
# movie. The feature vectors are the rows of the ratings_matrix.


# Let us compare the movies Toy_Story and Full_Metal_Jacket.  These movies should not be similar.

# first normalize the vector for each movie.  Note Full Metal Jacke has movie id #187
toy_story_norm = ratings_matrix[0,:]/norm(ratings_matrix[0,:])
Full_Metal_Jacket_norm = ratings_matrix[187,:]/norm(ratings_matrix[187,:])

# now compute the similarity between the two movies and display the result.
similarity_1 = np.dot(toy_story_norm,Full_Metal_Jacket_norm)

print(f'The similarity measure between Toy Story and Full Metal Jacket is {similarity_1}')

#  Now we compare the movies Star Wars #49  and Return of the Jedi #180.  These movies should be similar.

Star_Wars_norm = ratings_matrix[49,:]/norm(ratings_matrix[49,:])
Return_of_the_Jedi_norm = ratings_matrix[180,:]/norm(ratings_matrix[180,:])

# now compute the similarity between the two movies and display the result.
similarity_2 = np.dot(Star_Wars_norm,Return_of_the_Jedi_norm)

print(f'\nThe similarity measure between Star Wars and Return of the Jedi is {similarity_2}')
print('\n\n')
input('press return to continue\n\n\n')

# The results show that this cosine similarity stuff works. Cool!
# Now your turn

# Write a python function called top_five that will take as an input the id# 
# for a movie a user likes and return a list of the 5 movies we recommend to the user.
# These will be the five movies most "similar" to the input.

# For example when the function is executed with with Titanic #312 as input the output
# generate is below. Of course the first movie in the list should be the movie itself.

# Titanic (1997)
# Good Will Hunting (1997)
# Contact (1997)
# Apt Pupil (1998)
# Tomorrow Never Dies (1997)

def top_5(m,i):

 # This program will determine the 5 rows in matrix m
 # that are most similar to the target row i.
 #
 # Cosine similartiy is used in determining similarity
 # normalize the rows of matrix m
    m = m/np.linalg.norm(m,axis=1).reshape(-1,1)

    similarities = np.dot(m,m[i])

    l =np.argsort(similarities)[::-1][:5]
    for k in l:
        print(dictionary[k])


# Test top_five with the input Toy Story #0

# uncomment the line below to test it
print('The top five recommendations for Toy Story are:\n'.title())
top_5(ratings_matrix,63)

# Test top_five with the input Room with a View #212

# uncomment the line below to test it
print('\n\nThe top five recommendations for Room with a View are:\n'.title())
top_5(ratings_matrix,212)


# ========== Part 4: Dimensionality reduction using SVD  ============

# This part is completely optional and not for the faint of heart.

# Use the procedure outlined in chapter 7 in the text to reduce the data matrix
# from size 1682x943  to size 1682x100.
# We  thus are reducing the dimension of our feature vectors from 1682 to 100.

# After reducing your matrix redo the last two exercises above and compare your 
# results.  They should be the same!
