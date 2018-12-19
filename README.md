
> Stationary Schrodinger equation

## Usage
  * Download the repository.

  * This package require the tensorflow library. Make sure that the library is installed

  * Assuming PATH is the path from your current working directory to where StationarySchrodinger directory is, add the followings to the top of your script.

```sh
import sys
import tensorflow as tf
tf.enable_eager_execution();
sys.path.append("PATH/StationarySchrodinger")
from stationaryschrodinger import sstfAPI
from stationaryschrodinger import ssIO
from stationaryschrodinger import EigFunc
```
## Coverage
  To perform coverage test. 

  * go to the package directory '/StationarySchrodinger/'
  * Run the following script:
  ```sh
  coverage run --source=stationaryschrodinger setup.py test
  coverage report -m
  ```
Then call the functions in the modules as explained below.  
See the script in StationarySchrodinger/example/SSExample.py for an example of usage.  

## Modules
* ### ssIO
    This module handles IO operations for the package. 
    #### Functions:
    * **readPotentialEnergy(filepath)**

         Function reads potential potential energy from a text file

         INPUT: 

          path from the current working directory to the potential energy file 

          input file MUST NOT contain an empty line

         OUTPUT: 

          output list of length 2. 

          output[0] is a tensorflow array of positions

          output[1] is a tensorflow array of potential energy value

    * **readConsts(filepath)**

         Function reads the basis size and constant from a text file

         INPUT:

           path from the current working directory to an input file. Each line indicates the constant name followed by white space then the constant value

         OUTPUT: 

            A list of constants. Each entry corresponds to each constants from the text file from top to bottom of the file


         All the output will be of type float. If you intend to use them as an integer, you must cast them to int separately. 

        
* ### tfAPI
    This module provides functions that is used to handle tensorflow arrays

    #### Functions:

    * **compare(t1,t2,tol)**
        Compare two 2D tensorflow arrays

        INPUT: 

          tensorflow arrays t1,t2

          tolerence tol

        OUTPUT: 

          if (t1[i][j]-t2[i][j]) < tol for all (i,j) return true, else false


    * **tfdim(t)**  

        INPUT: 

          tensorflow array t

        OUTPUT: 

          n where n is the number of dimensions of t 

          e.g., [1,1,1] has n == 1 (1D array) and [[1,1],[2,2]] has n==2 (2D array)
          
    * **tflen(t)**

	     INPUT: 
	
	      1D tensorflow array t
	
	     OUTPUT: 
	
	      n where n is the length of t
	
	      e.g., [1,1,1] has n==3

          
    * **integrate(t1,t2,x)** 

	     INPUT: 
	
	      1D tensorflow arrays of the same length t1,t2,x
	
	      t1 and t2 are vector representing a function defined on the domain x where x is an evenly spaced grid points
	
	     OUTPUT: 

	      tout where tout = dx * \Sum_i(t1[i] * t2[i])
	
	      dx = x[1]-x[0]
	
	      tout is the result of a numerical integration of t1*t2


          
* ### EigFunc 

    This module contains functions that handle creating basis sets, Hamiltonian, and calculating the Eigenvalues/Vectors

    #### Functions:  

    * **FPoly(x,n)**  

       INPUT: 

	        tensorflow array of positions. x
	
	        size of basis n

       OUTPUT: 

	         a list *fpoly* of length n consists of tensorflow arrays of the fourier basis evaluated in the domain x.
	
	         len(*fpoly*) = 2*len(x)+1
	
	         fpoly[0] = constant tensorflow array of 1's of length len(x)
	
	         fpoly[i] = sin(nmode*x) for odd i
	
	         fpoly[i] = cos(nmode*x) for even i
	
	         An example of an output from input x of length 2 is [ <<1>>, <<sin(x)>>, <<cos(x)>>, <<sin(2x)>>, <<cos(2x)>> ] 
	
	          where << >> denotes a tensorflow array 

  
    *   **project(F,b,x)** 

          INPUT: 

            tensorflow array F

            A list of tensorflow arrays b where b[i] is the i-th basis of the basis set

            tensorflow array x where x is the domain of the problem

            len(x) == len(F) == len(b[i])

         OUTPUT: 

          Tensorflow array A where A[i] is the coefficient of basis b[i] 

          and \Sum_0^n(A[i]B[i]) is the projection of F onto basis b

          A = inv( <b[i],b[j]> ) * <F,b[i]> where <> indicate an inner product

  

    *   **hamil(x,b,c,V)**  

         INPUT: 

          tensroflow array x, b, V

          x defines the domain of the problem

          b is a list of tensorflow array where b[i] is the i-th basis set

          V is a tensorflow array defining the potential energy on the domain x

         OUTPUT: 

          the Hamiltinan H

          where H[i][j] = c*n^2 * delta_{ij} + <b[i] | V | b[j]>

          n is the wavenumber

          delta_{ij} = 1 if i == j and 0 otherwise

          <| |> is the bra-ket notation



    * **Eig(Hij)**
            Function prints out the lowest energy state and the corresponding wavefunction's amplitudes on each basis. The EigenVector[i] represents the amplitude of basis b[i], and the wavefunction corresponding to the lowest energy state is \Sum_0^n (EigenVector[i]b[i]) 

          INPUT: 

            n-by-n tensorflow array represneting the the hamiltonial

          OUTPUT: 

            a list of eigen value and eigen vectors of the hamiltonian

            EigVecVal[0] = tensorflow array of eigen values

            EigVecVal[1] = tensorflow array of eigenvectors 


          



```


