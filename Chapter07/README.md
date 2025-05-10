# Solving Poisson's equation / gravity solvers
Poisson solvers are typically key ingredients of astrophysical simulation codes involving self-gravity. In addition to the simpler methods in the lecture notes, you find here the most important methods that are actually used in production codes. A short and didactic implementation of the fast multipole method (FMM) is unfortunately still missing from this list.

The first two methods work on structured grid data, the third method works on unstructured point data. 

This directory contains sample implementations of three key methods to solve Poisson's equation:
* A matrix-free conjugate gradient solver `poisson2d_conjugate_gradient.ipynb' that reaches an optimal complexity of $O(N^{d+1})$ in $d$ dimensions
* A multigrid solver that accelerates a Gauss-Seidel relaxation solver `poisson2d_multigrid.ipynb' reaching an optimal complexity $O(N^d)$ in d dimensions, the implementation is for 2D square grids
* A Barnes & Hut tree method to solve the 'free' Poisson equation for a number of $N$ points using a multipole expansion of the Green's function along with an Octree-based decomposition `poisson3d_barnes_hut_tree.ipynb', reaching an optimal complexity of $O(N\log N)$.

