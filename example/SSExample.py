import math
import sys
import tensorflow as tf
tf.enable_eager_execution();
sys.path.append("../")
from stationaryschrodinger import sstfAPI
from stationaryschrodinger import ssIO
from stationaryschrodinger import EigFunc

def SSExample():
  Position,PotEn = ssIO.readPotentialEnergy('./Example_pot.dat')
  c,BasisSize = ssIO.readConsts('./Example_consts.dat')
  BasisSize = int(BasisSize)
  Basis = EigFunc.FPoly(Position,BasisSize)
  Hij = EigFunc.hamil(Position,Basis,c,PotEn)
  EigVecVal = EigFunc.Eig(Hij)

SSExample()

