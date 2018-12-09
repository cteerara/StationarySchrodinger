import math
import tensorflow as tf
tf.enable_eager_execution();


#-------------------------------------------#
#---- Fourier polynomial -------------------#
def FPoly(x):
    # INPUT: tensorflow array of positions. x
    # OUTPUT: an array fpoly. Consist of tensorflow arrays of the fourier basis evaluated in the domain x.
    #         len(fpoly) = 2*len(x)+1
    #         fpoly[0] = constant tensorflow array of 1's of length len(x)
    #         fpoly[i] = sin(i*x) for odd i
    #         fpoly[i] = cos((i-1)*x) for even i
    #         An example of an output from input x of length 2 is [ <1>, <sin(x)>, cos<x>, <sin(2x)>, <cos(2x)> ] where <> denotes a tensorflow array 

    np = x.get_shape()
    np = np[0];
    fpoly = [];
    ones = []
    for i in range(0,np):
      ones.append(1.)
    
    fpoly.append(tf.constant(ones,dtype=tf.float64))

    for n in range(1,np+1):
      fpoly.append(tf.math.sin(n*x))
      fpoly.append(tf.math.cos(n*x))

    return fpoly
    

pi = 4*math.atan(1.0)
x = tf.constant([0,pi/2],dtype=tf.float64)
F = FPoly(x)
for i in range(0,len(F)):
  print(F[i])
