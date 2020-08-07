'''
Project #3 part 1.
Math 266 Linear Algebra
Solution 
Summer 2020
'''
 
# import libraries

import simpleaudio as sa
import matplotlib.pyplot as pyplot
import numpy as np
from numpy import sin, pi
from scipy.io import wavfile
import os

os.system('clear')
print('\n\n\tProject #3 Audio')

sample_rate = 44100
mid_C = 262  # frequency for middle C on the piano


def tone(freq=440, duration=2):
    ''' 
    This function generates a sine wave tone with frequency = freq
    and duration = duration. It will return a numpy array containin
    the audio sample values.

    Usage:

    audio = tone(262,3)

    Will return a python array with sample values
    for the "middle C note" duration 3 sec 

     Use the play_sound function to play the note.

    '''

    dt = 1/sample_rate  # separation between samples

    # discretize the time domain
    #  for the sigal
    t = np.arange(0, duration, dt)

    # need a trig review?  Go to
    # https://www.youtube.com/watch?v=OCfAEZphAzE
    note = np.sin(2*pi*freq*t)

    # normalize to 16-bit range
    note *= 32767/np.max(np.abs(note))

    # convert to 16-bit data
    note = note.astype(np.int16)

    return note


def play_sound(audio):
    '''
    function will play the sound stored in array audio.

    '''

    play_obj = sa.play_buffer(audio, 1, 2, sample_rate=44100)
    while play_obj.is_playing():
        pass
    play_obj.wait_done
    return

# generate and save the frequecies for the chromatic scale in a list called freqs

freqs = np.zeros(13)  # initialize a list of zeros
freqs[0] = 262  # start with middle c
for k in range(12):
    freqs[k+1] = freqs[k]*2**(1/12)  # next frequencey is 12th root of 2 times previous frequecy

# generate list of notes for chromatic scale

notes = []
for k in range(13):
    notes.append(tone(freqs[k],.4))
notes = np.array(notes)

chromatic_scale = np.hstack(notes)

print('\n Playing chromatic scale')
play_sound(chromatic_scale)

input("\n\n\t please press a key to continue:\n")

print('\n Playing C major scale')

scale = notes[[0,2,4,5,7,9,11,12]]

C_scale = np.hstack(scale)
C_scale = np.hstack((C_scale,C_scale[::-1]))
play_sound(C_scale)
print('\n\ndone......')

wavfile.write("scale.wav",44100,C_scale)


