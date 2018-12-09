import math
import tensorflow as tf
tf.enable_eager_execution();


#-------------------------------------------#
#---- Smaller API to make things easier ----#
#-------------------------------------------#
def cos(x):
    return math.cos(x)
def sin(x):
    return math.sin(x)

#-------------------------------------------#
#---- Fourier polynomial -------------------#
def FPoly(x,np):
    # INPUT: tensorflow array of positions. x
    #        number of gridpoints np
    # OUTPUT: tensorflow array of fourier polynomials

    Fpoly = [];
    for i in range(0,np):
      Fpoly.append([]);
    for i in range(0,np):
      Fpoly[0].append(1)
    for n in range(1,nmax+1):
      Fpoly.append(sin(n*x));
      Fpoly.append(cos(n*x));

    


x = [];
for i in range(0,3):
  x.append([]);

for i in range(0,3):
  for j in range(0,3):
    x[i].append(j)

print(x)
