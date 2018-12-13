import math
import tensorflow as tf
tf.enable_eager_execution();
import tfAPI


#-------------------------------------------#
#---- Fourier polynomial -------------------#
def FPoly(x,n):
    # INPUT: tensorflow array of positions. x
    # OUTPUT: an array fpoly. Consist of tensorflow arrays of the fourier basis evaluated in the domain x.
    #         len(fpoly) = 2*len(x)+1
    #         fpoly[0] = constant tensorflow array of 1's of length len(x)
    #         fpoly[i] = sin(i*x) for odd i
    #         fpoly[i] = cos((i-1)*x) for even i
    #         An example of an output from input x of length 2 is [ <1>, <sin(x)>, cos<x>, <sin(2x)>, <cos(2x)> ] where <> denotes a tensorflow array 

    np = n+1; # Number of modes.
    fpoly = [];
    ones = []
    for i in range(0,np-1):
      ones.append(1.)
    
    fpoly.append(tf.constant(ones,dtype=tf.float64))

    for n in range(1,np+1):
      fpoly.append(tf.math.sin(n*x))
      fpoly.append(tf.math.cos(n*x))

    return fpoly
    
def project(F,b):
    # INPUT: tensorflow array F
    #        array of tensorflow arrays b where b[i] is the i-th basis of the basis set
    # OUTPUT: Tensorflow array A where A[i] is the coefficient of basis b[i] 
    #         and \Sum_0^n(A[i]B[i]) is the projection of F onto basis b
    np = len(b)
    nmax = (np-1)//2+1 # maximum n-modes
#    for i in range(1,nmax+2,2):
#        n = i//2+1
#        print('**** currently on mode n =',n)
#        print(b[i])
#        print(b[i+1])
    c = []
    for i in range(0,np):
        c.append(tf.reshape(tfAPI.integrate(F,b[i]),[]))
    A = tf.Variable(c,dtype=tf.float64)
    return A



pi = 4*math.atan(1.0)
x = tf.constant([0,pi/2],dtype=tf.float64)
n = tf.shape(x)
n = n[0]
b = FPoly(x,2)
F = tf.constant([1,1],dtype=tf.float64)
for i in range(0,len(b)):
    print(b[i])
print('*****************')
print(project(F,b))

#x1 = tf.constant(1)
#x2 = tf.constant(2)
#x3 = tf.Variable([1,2])
#x4 = tf.Variable([x1,x2])
#print(x3)
#print('------------------')
#print(x4)
#
#
#
