Python code for solving the Schrodinger Equation for some
quantum physics problems.

The method used for solving the second order differential 
equation is the variational Rayleigh-Ritz method with the 
B-splines functions as a basis set.

A detailed description of the B-splines functions 
and their numerical implementation in quantum problems are shown in 
the paper

Applications of B-splines in atomic and molecular physics
H. Bachau, E. Cormier, P. Decleva, J. E. Hansen and F. Martin.
Rep. Prog. Phys. 64 (2001) 1815-1942

The code use a Scypy package for python and in particular
the bspline.py code taken from 

http://johntfoster.github.io/posts/pythonnumpy-implementation-of-bspline-basis-functions.html

