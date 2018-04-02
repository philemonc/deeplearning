# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:33:26 2017

@author: csuwe
"""

import math
import numpy as np
#import h5py
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

def linear_function():
    """
    Implements a linear function:
    Initializes W to be a random tensor of shape (4,3)
    Initializes X to be a random tensor of shape (3,1)
    Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """
    np.random.seed(1)
    ### START CODE HERE ### (4 lines of code)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W,X), b)
    ### END CODE HERE ###
    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    ### START CODE HERE ###
    sess = tf.Session()
    result = sess.run(Y)
    ### END CODE HERE ###
    # close the session
    sess.close()
    return result
np.random.seed(1)

y_hat = tf.constant(36, name='y_hat') # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y') # Define y. Set to 39
loss = tf.Variable((y - y_hat)**2, name='loss') # Create a variable for the loss
init = tf.global_variables_initializer() # When init is run later (session.run

# the loss variable will be initialize

with tf.Session() as session: # Create a session and print the outpu

    session.run(init) # Initializes the variables
    print(session.run(loss))

    a = tf.constant(2)
    b = tf.constant(10)
    c = tf.multiply(a,b)
    print(c)

sess = tf.Session()
print(sess.run(c))
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 4}))
sess.close()

X = tf.constant(np.random.randn(3,1), name = "X")

