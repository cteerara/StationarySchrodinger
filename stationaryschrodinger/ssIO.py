import tensorflow as tf
tf.enable_eager_execution();
import sstfAPI

def readPotentialEnergy(filepath):
    # Function reads potential potential energy from a text file
    # INPUT: path to the potential energy file from cwd
    #        input file MUST NOT contain an empty line
    # OUTPUT: output list of size 2. 
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

def readConsts(filepath):
    # Function reads the basis size and constant from a text file
    # INPUT: path from cwd to an input file which contains 2 lines
    #        Each line indicates the constant name followed by white space then the constant value
    # OUTPUT: A list of constants. Each entry corresponds to each line of constant from top to bottom of the file
    # All the output will be of type float. If you intend to use it as an integer, you must cast it as such.
     
    fid = open(filepath,'r')
    line = 'String is not empty'
    const = [];
    while line: 
        line = fid.readline()
        if not line: break; # Break when I see an empty line
        if line[0] != '#':
            data = line.split();
            const.append(float(data[1]));
    return const
