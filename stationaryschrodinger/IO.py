#def readPotentialEnergy(filepath):
#    fid = open(filepath,'r')

class PE:
    def __init__(self):
        self.en = [];
        self.pos = [];

def readPotentialEnergy(filepath):
    fid = open(filepath,'r')
    line = 'Hi, I am not empty'
    while line:
        line = fid.readline()
        if not line: break; # Break when I see an empty line
        if line[0] != '#':
            data = line.split();
            PE.pos.append(float(data[0]));
            PE.en.append(float(data[1]));
    print(PE.en)

readPotentialEnergy('potential_energy.dat')

