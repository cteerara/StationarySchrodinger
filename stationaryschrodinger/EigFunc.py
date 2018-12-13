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
    for i in range(0,tfAPI.tflen(x)):
      ones.append(1.)
    
    fpoly.append(tf.constant(ones,dtype=tf.float64))

    for n in range(1,np+1):
      fpoly.append(tf.math.sin(n*x))
      fpoly.append(tf.math.cos(n*x))

    return fpoly
    
def project(F,b,x):
    # INPUT: tensorflow array F
    #        array of tensorflow arrays b where b[i] is the i-th basis of the basis set
    #        x is a 1D tensorflow array of evenly spaced positions used as the bounds of integration. len(x) == len(F) == len(b[i])
    # OUTPUT: Tensorflow array A where A[i] is the coefficient of basis b[i] 
    #         and \Sum_0^n(A[i]B[i]) is the projection of F onto basis b
    np = len(b)
    nmax = (np-1)//2+1 # maximum n-modes
#    for i in range(1,nmax+2,2):
#        n = i//2+1
#        print('**** currently on mode n =',n)
#        print(b[i])
#        print(b[i+1])
    tmparr = []
    dx = x[1]-x[0]
    for i in range(0,np):
        tmparr.append( tfAPI.integrate(F,b[i],dx) )
    FdotB = tf.Variable(tmparr,dtype=tf.float64)
    FPdotB_mat = []
    for i in range(0,np):
        FPdotB_mat.append([])
    for i in range(0,np):
        for j in range(0,np):
            FPdotB_mat[i].append(0)
    for i in range(0,np):
        for j in range(0,np):
            FPdotB_mat[i][j] = tfAPI.integrate(b[i],b[j],dx)
    FPdotB_mat = tf.Variable(FPdotB_mat,dtype=tf.float64)
    
    return FPdotB_mat



pi = 4*math.atan(1.0)
x = tf.constant([0,pi/2],dtype=tf.float64)
n = tf.shape(x)
n = n[0]
b = FPoly(x,0)
F = tf.constant([1,1],dtype=tf.float64)
for i in range(0,len(b)):
    print(b[i])
print('*****************')
print(project(F,b,x))

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
