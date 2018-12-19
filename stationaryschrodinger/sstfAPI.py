import tensorflow as tf
tf.enable_eager_execution();


def compare(t1,t2,tol):
    # Compare two 2D tensorflow arrays
    # INPUT: tensorflow arrays t1,t2
    #        tolerence tol
    # OUTPUT: true if (t1-t2) < tol, else false
    t1size = t1.get_shape();
    t2size = t2.get_shape();
    if (len(t1size) != len(t2size)):
      print('Array dimension mismatch')
      return False;
    if (t1.dtype != t2.dtype):
      print('Datatype mismatch')
      return False

    isequal = True;
    for i in range(0,len(t1size)):
      isequal = isequal and t1size[i] == t2size[i]

    if (not isequal):
      print('Length in each dimension mismatch')
      return False;
    else: 
        tol = tf.constant(tol,shape=t1size,dtype=tf.float32)
        EqualTensor = tf.math.less( tf.cast(tf.math.abs(t1-t2),tf.float32), tol) 
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
    tol = 1e-6
    scalar1 = tf.constant(1,dtype=tf.int32)
    tdim = tfdim(t)
    if not compare(scalar1,tf.cast(tdim,scalar1.dtype),tol):
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
    tol = 1e-6
    if not compare(tflen(t1),tflen(t2),tol):  
        raise ValueError('Input arrays are not the same shape')
    n = tflen(t1)
    tout = tf.reshape(tf.reduce_sum(t1[0:n-1]*t2[0:n-1]),[])
    return tout*dx

