import tensorflow as tf
tf.enable_eager_execution();

def readPotentialEnergy(filepath):
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

x = readPotentialEnergy('potential_energy.dat')
print(x[0])
print(x[1])

