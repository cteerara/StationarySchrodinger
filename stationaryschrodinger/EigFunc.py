import math
import tensorflow as tf
tf.enable_eager_execution();
import tfAPI

#-------------------------------------------#
#---- Fourier polynomial -------------------#
def FPoly(x,n):
    # INPUT: tensorflow array of positions. x
    #        size of basis n
    # OUTPUT: a list fpoly of length n consists of tensorflow arrays of the fourier basis evaluated in the domain x.
    #         len(fpoly) = 2*len(x)+1
    #         fpoly[0] = constant tensorflow array of 1's of length len(x)
    #         fpoly[i] = sin(nmode*x) for odd i
    #         fpoly[i] = cos(nmode*x) for even i
    #         An example of an output from input x of length 2 is [ <<1>>, <<sin(x)>>, <<cos(x)>>, <<sin(2x)>>, <<cos(2x)>> ] where <<>> denotes a tensorflow array 

    np = n; # Number of modes.
    fpoly = [];
    ones = []
    #> Fill up the 0th mode
    for i in range(0,tfAPI.tflen(x)):
      ones.append(1.)
    
    fpoly.append(tf.constant(ones,dtype=tf.float64))
    #> Fill up modes n > 0
    nmode = 1
    for i in range(1,np):
        print(nmode)
        if i%2 == 1: # Sin mode
            fpoly.append(tf.math.sin(nmode*x))
        else:
            fpoly.append(tf.math.cos(nmode*x))
            # after calculating the cosine mode, increase nmode
            nmode = nmode+1

    return fpoly
    
def project(F,b,x):
    # INPUT: tensorflow array F
    #        array of tensorflow arrays b where b[i] is the i-th basis of the basis set
    #        x is a 1D tensorflow array of evenly spaced positions used as the bounds of integration. len(x) == len(F) == len(b[i])
    # OUTPUT: Tensorflow array A where A[i] is the coefficient of basis b[i] 
    #         and \Sum_0^n(A[i]B[i]) is the projection of F onto basis b
    #         A = inv( <b[i],b[j]> ) * <F,b[i]> where <> indicate an inner product
    np = len(b)
    nmax = (np-1)//2+1 # maximum n-modes
    FdotB = []
    dx = x[1]-x[0]
    # Create a list of 1D tensorflow arrays <F,b[i]> 
    for i in range(0,np):
        FdotB.append( tfAPI.integrate(F,b[i],dx) )
    FdotB = tf.Variable(FdotB,dtype=tf.float64)

    
    LHS = []
    for i in range(0,np):
        LHS.append([])
    for i in range(0,np):
        for j in range(0,np):
            LHS[i].append(0)
    for i in range(0,np):
        for j in range(0,np):
            LHS[i][j] = tfAPI.integrate(b[i],b[j],dx)
    LHS = tf.Variable(LHS,dtype=tf.float64)

    RHS = []
    for i in range(0,np):
        RHS.append(tfAPI.integrate(b[i],F,dx))
    RHS = tf.Variable(RHS,dtype=tf.float64)
    
#    return tf.linalg.matmul(tf.linalg.inv(FPdotB_mat),tf.reshape(FdotB_vec,[len(b),1]))
    return tf.linalg.solve(LHS,rhs=tf.reshape(RHS,[len(b),1]))



#pi = 4*math.atan(1.0)
##x = tf.constant([0,pi/2,pi,3*pi/2,2*pi],dtype=tf.float64)
#x = []
#n = 4
#for i in range(0,n+1):
#    x.append(i*pi/n)
#x = tf.constant(x,dtype=tf.float64)
#
#n = tf.shape(x)
#n = n[0]
#b = FPoly(x,5)
#F = tf.math.cos(2*x)
#for i in range(0,len(b)):
#    print(b[i])
#print('*****************')
#print(project(F,b,x))
#
