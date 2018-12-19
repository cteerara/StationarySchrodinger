import math
import sys
import tensorflow as tf
tf.enable_eager_execution();
sys.path.append("../")
from stationaryschrodinger import tfAPI
from stationaryschrodinger import ssIO
from stationaryschrodinger import EigFunc

def SSExample():
  # Read position and Potential Energy
  Position,PotEn = ssIO.readPotentialEnergy('./Example_pot.dat')

  # Readd constant c and Basis Size
  c,BasisSize = ssIO.readConsts('./Example_consts.dat')

  # Recast Basis Size as an integer
  BasisSize = int(BasisSize)

  # Create a list of Fourier Polynomial basis of size 'BasisSize'
  Basis = EigFunc.FPoly(Position,BasisSize)

  # Generate the Hamiltonian based on the input parameters
  Hij = EigFunc.hamil(Position,Basis,c,PotEn)

  # Get Eigenvalues and EigenVectors and also print out the lowest energy state and the corresponding wavefunction.
  LES = EigFunc.LowestEnergyState(Hij)

  
  print('')
  print('The domain is defined by the Example_pot.dat')
  print('For this example the domain is [0,2*pi]')
  print('Basis size: 3') 
  print('-------------')
  print('The basis is:')
  for i in range(0,len(Basis)):
    print(Basis[i])
  print('-------------')
  print('The Hamiltonian is:')
  print(Hij)
  print('-------------')
  print('Lowest energy is:')
  print(LES[0])
  print('-------------')
  print('The corresponding Eigen Vector is:')
  print(LES[1])


SSExample()
 
