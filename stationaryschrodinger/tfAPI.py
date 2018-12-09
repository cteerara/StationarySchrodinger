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
        EqualTensor = tf.math.equal(t1,t2)
        isequal = tf.math.reduce_all(EqualTensor)
        if (isequal):
          return True;
        else:
          return False;        

#t1 = tf.constant([0,0])
#t2 = tf.constant([0,0])
#x = compare(t1,t2)
#print(x)
