import tensorflow as tf
tf.enable_eager_execution();
import tfAPI

def readPotentialEnergy(filepath):
    # Function reads potential potential energy from a text file
    # INPUT: path to the potential energy file fomr cwd
    # OUTPUT: output array of size 2. 
    #         output[0] is a tensorflow array of positions
    #         output[1] is a tensorflow array of potential energy value
    fid = open(filepath,'r')
    line = 'String is not empty'
    pos = [];
    en = [];
    while line: 
        line = fid.readline()
        if not line: break; # Break when I see an empty line
        if line[0] != '#':
            data = line.split();
            pos.append(float(data[0]));
            en.append(float(data[1]));
    postf = tf.constant(pos,shape=[len(pos)],dtype=tf.float64)
    entf = tf.constant(en,shape=[len(en)],dtype=tf.float64)
    return [postf,entf]

#x = readPotentialEnergy('../tests/IOtest.dat')
#x0 = tf.constant([0.,1.,2.,3.],dtype=tf.float64)
#y = x0.get_shape();
#for i in range(0,y[0]):
#  print(i)
#
#print(x[0])
#print(x[1])
#print(tfAPI.compare(x0,x[0]))
#pos = x[0];
#print(pos[1]-pos[0])
