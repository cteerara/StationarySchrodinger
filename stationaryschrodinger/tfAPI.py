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
    isequal = True;
    if (t1size[0] != t2size[0]) or (t1size[1] != t2size[1]):
        isequal = False;
    else: 
        for i in range(0,t1size[0]):
            for j in range(0,t1size[1]):
                t1elem = t1[i,j];
                t2elem = t2[i,j];
                isequal = ( isequal and tf.Variable(tf.math.equal(t1elem,t2elem)) )
    return isequal;

