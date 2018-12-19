import math
import tensorflow as tf
tf.enable_eager_execution();
import tfAPI

#-------------------------------------------#
#---- Fourier polynomial -------------------#
def FPoly(x,n):
    # INPUT: tensorflow array of positions. x
    #        size of basis n
    # OUTPUT: a list fpoly of length n consists of tensorflow arrays of the fourier basis evaluated in the domain x.
    #         len(fpoly) = 2*len(x)+1
    #         fpoly[0] = constant tensorflow array of 1's of length len(x)
    #         fpoly[i] = sin(nmode*x) for odd i
    #         fpoly[i] = cos(nmode*x) for even i
    #         An example of an output from input x of length 2 is [ <<1>>, <<sin(x)>>, <<cos(x)>>, <<sin(2x)>>, <<cos(2x)>> ] where <<>> denotes a tensorflow array 

    np = n; # Number of modes.
    fpoly = [];
    ones = []
    #> Fill up the 0th mode
    for i in range(0,tfAPI.tflen(x)):
      ones.append(1.)
    
    fpoly.append(tf.constant(ones,dtype=x.dtype))
    #> Fill up modes n > 0
    nmode = 1
    for i in range(1,np):
        if i%2 == 1: # Sin mode
            fpoly.append(tf.math.sin(nmode*x))
        else:
            fpoly.append(tf.math.cos(nmode*x))
            # after calculating the cosine mode, increase nmode
            nmode = nmode+1

    return fpoly
    
def project(F,b,x):
    # INPUT: tensorflow array F
    #        array of tensorflow arrays b where b[i] is the i-th basis of the basis set
    #        x is a 1D tensorflow array of evenly spaced positions used as the bounds of integration. len(x) == len(F) == len(b[i])
    # OUTPUT: Tensorflow array A where A[i] is the coefficient of basis b[i] 
    #         and \Sum_0^n(A[i]B[i]) is the projection of F onto basis b
    #         A = inv( <b[i],b[j]> ) * <F,b[i]> where <> indicate an inner product
    np = len(b)
    nmax = (np-1)//2+1 # maximum n-modes
    FdotB = []
    dx = x[1]-x[0]
    
    LHS = []
    for i in range(0,np):
        LHS.append([])
    for i in range(0,np):
        for j in range(0,np):
            LHS[i].append(0)
    for i in range(0,np):
        for j in range(0,np):
            LHS[i][j] = tfAPI.integrate(b[i],b[j],x)
    LHS = tf.Variable(LHS,dtype=x.dtype)

    RHS = []
    for i in range(0,np):
        RHS.append(tfAPI.integrate(b[i],F,x))
    RHS = tf.Variable(RHS,dtype=x.dtype)
    
    return tf.reshape( tf.linalg.solve(LHS,rhs=tf.reshape(RHS,[len(b),1])), [len(b)] )

def hamil(x,b,c,V):
  # INPUT: tensroflow array x, b, V
  #        scalar value c
  # OUTPUT: the Hamiltinan H
  #         where H[i][j] = c*n^2 * delta_{ij} + <b[i]|V|b[j]>
  #         delta_{ij} = 1 if i == j and 0 otherwise
  #         <> is the bra-ket notation
  n = len(b)
  Hij = []
  for i in range(0,n):
    Hij.append([])


  nmode = 0
  for i in range(0,n):
    for j in range(0,n):
      if i==j:
        Hij[i].append(nmode**2*c)
        if j%2 == 0:
          nmode = nmode+1
      else:
        Hij[i].append(0.0)

    Hij[0][0] = 0.0

  for i in range(0,n):
    for j in range(0,i+1):
      # <b[i]|V|b[j]>
      Bra_bi_V_bj_Ket = tfAPI.integrate(b[i]*V,b[j],x)
      Hij[i][j] = Hij[i][j] + Bra_bi_V_bj_Ket
      if i != j:
        # Symmetry
        Hij[j][i] = Hij[j][i]+Bra_bi_V_bj_Ket
        
  return tf.Variable(Hij,dtype=x.dtype)

def LowestEnergyState(Hij):
  # INPUT: n-by-n matrix of the hamiltonial
  # OUTPUT: Lowest Energy State and the corresponding eigenvector
  Val,Vec = tf.linalg.eigh(Hij,name=None)
  
  return [Val[0],Vec[:,0]]
