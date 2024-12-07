# Inspired by https://medium.com/@natsunoyuki/quantum-mechanics-with-python-de2a7f8edd1f

from scipy import sparse   
from scipy.sparse import linalg as sla
import numpy as np

def discretized_one_dimensional_time_independent_Schrödinger_equation(V, dx=1, n_solutions=20):
    """Solves the time independent 1D Schrodinger Equation for the given discretized potential

    Note: assumes Dirichlet boundary conditions (Psi=0 and dPsi=0 at boundaries)

    Args:
        V: discretized potential
        dx: size of points in V (assumes hbar=m=1)
        n_solutions: number of eigenvalues to solve for
    Returns:
        Es, Psis: energies and wavefunctions
    """
    Nx = len(V)

    H = sparse.eye(Nx, Nx, format='lil') * 2

    for i in range(Nx - 1):
        H[i, i + 1] = -1
        H[i + 1, i] = -1
    H = H / (dx ** 2)
    # Add in the potential energy V
    for i in range(Nx):
        H[i, i] = H[i, i] + V[i]
    # convert to csc sparse matrix format:
    H = H.tocsc()
    # obtain neigs solutions from the sparse matrix:
    [Es, Psis] = sla.eigs(H, k=n_solutions, which='SM')
    for i in range(n_solutions):
        # normalize the eigenvectors:
        Psis[:, i] = Psis[:, i] / np.sqrt(
                                np.trapz(np.conj(
                                Psis[:,i])*Psis[:,i]))
        # eigen values MUST be real:
        Es = np.real(Es)
        
    return Es, Psis