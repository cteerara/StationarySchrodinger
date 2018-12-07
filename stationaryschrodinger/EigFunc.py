import math

#-------------------------------------------#
#---- Smaller API to make things easier ----#
#-------------------------------------------#
def cos(x):
    return math.cos(x)
def sin(x):
    return math.sin(x)

#-------------------------------------------#
#---- Fourier polynomial -------------------#
def FPoly(n,a,b):
    # Defines the orthonormal, on bounds [a,b], of fourier polynomial at degree n
    # INPUT: integer n indicating wave number
    # OUTPUT: 
