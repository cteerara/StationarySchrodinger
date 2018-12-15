import tensorflow as tf
tf.enable_eager_execution();

def zeroes(dimvec):
    # Generate tensorflow array of zeroes of size dimvec
    # INPUT: [int a, int b]
    # OUTPUT: a-by-b Tensorflow array of zeroes of type float64
    return (tf.Variable(tf.zeros(dimvec, dtype=tf.float64)));

def compare(t1,t2):
    # Compare two 2D tensorflow arrays
    # INPUT: tensorflow arrays t1,t2
    # OUTPUT: true if t1==t2, else false
    t1size = t1.get_shape();
    t2size = t2.get_shape();
    if (len(t1size) != len(t2size)):
      return False;

    isequal = True;
    for i in range(0,len(t1size)):
      isequal = isequal and t1size[i] == t2size[i]

    if (not isequal):
      return False;
    else: 
        tol = tf.constant(1e-6,shape=t1size,dtype=tf.float64)
        EqualTensor = tf.math.less( tf.math.abs(t1-t2), tol)
        isequal = tf.math.reduce_all(EqualTensor)
        if (isequal):
          return True;
        else:
          return False;        

def tfdim(t):
    # INPUT: tensorflow array t
    # OUTPUT: n where n is the number of dimensions of t 
    #         e.g., [1,1,1] has n==1 (1D array) and [[1,1],[2,2]] has n==2 (2D array)
    return tf.reshape(tf.shape(tf.shape(t)),[])

def tflen(t):
    # INPUT: 1D tensorflow array t
    # OUTPUT: n where n is the length of t
    #         e.g., [1,1,1] has n==3
    scalar1 = tf.constant(1)
    if not compare(scalar1,tfdim(t)):
        raise ValueError('input array is not 1D')
    return tf.reshape(tf.shape(t),[])

def integrate(t1,t2,x):
    # INPUT: 1D tensorflow arrays of the same length t1,t2
    #        t1 and t2 are vector representing a function defined on the same domain at the same evenly spaced grid points
    #        dx is the spacings between gridpoints
    # OUTPUT: tout where tout = \Sum_i(t1[i]*t2[i])
    #         tout is the result of a numerical integration of t1*t2

    #> Handle inappopriate size/dim array
    dx = x[1]-x[0]
    for i in range(0,tflen(x)-1):
      if (dx-(x[i+1]-x[i])) > 1e-6:
        raise ValueError('Gridpoint are not evenly spaced. The difference of the spacings exceeds 1e-6')
    scalar1 = tf.constant(1)
    if not compare(tflen(t1),tflen(t2)):  
        raise ValueError('Input arrays are not the same shape')
    n = tflen(t1)
    tout = tf.reshape(tf.reduce_sum(t1[0:n-1]*t2[0:n-1]),[])
    return tout*dx

#t1 = tf.constant(2,shape=[2,2],dtype=tf.float64)
#print(t1)
#
#t1 = tf.constant([3,4])
#t2 = tf.constant([1,2])
#print(integrate(t1,t2))
#x = integrate(t1,t2)
#print(x)
