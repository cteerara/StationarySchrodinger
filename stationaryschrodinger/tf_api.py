import tensorflow as tf
tf.enable_eager_execution();

def zeroes(dimvec):
    return (tf.zeros(dimvec, tf.float64));

def compare(t1,t2):
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

t1 = zeroes([2,2])
t2 = tf.constant(0,shape=[],dtype=tf.float64)
print(t2)
tf.assign(t1[0,0],t2)
print(t1)
