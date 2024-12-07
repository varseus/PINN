from simulation import discretized_one_dimensional_time_independent_Schrödinger_equation
from torch.utils.data import Dataset
import numpy as np
import random
import sys
import pickle

class Dataset_Random_Potentials(Dataset):
    """Dataset for a set of randomly generated 1D potentials and their solutions.
    """
    def __init__(self, Nx=500, n_polynomial=0, n_sho=0, n_square_well=0, n_double_well=0, seed=1, cache=None, load=None):
        """Randomly generates 1D potentials

        Args:
            Nx (int, optional): Size of potentials. Defaults to 500.
            n_polynomial (int, optional): Number of randomly generated polynomial potentials. Defaults to 0.
            n_sho (int, optional): Number of randomly generated simple hamonic oscillator potentials. Defaults to 0.
            n_square_well (int, optional): Number of randomly generated square well potentials. Defaults to 0.
            n_double_well (int, optional): Number of randomly generated double well potentials. Defaults to 0.
            seed (int, optional): Random seed. Defaults to 1.
            cache (file, optional): Location to save results to. Defaults to None.
            load (file, optional): Location to load results from. Defaults to None.
        """
        self.Nx = Nx
        self.n_solutions = 3
        self.Vs = []
        self.Es = []
        self.Psis = []

        random.seed(seed)

        if load:
            print(f"Loading dataset from {load}")
            with open(load, 'rb') as handle:
                self.Vs, self.Es, self.Psis = pickle.load(handle)
            print('\n' + self.__str__())
            return

        def generate(n, func):
            """Generates n potentials/solutions, created by func

            Args:
                n (int): number of potentials to generate
                func (func): function to generate potentials
            """
            for i in range(n):
                sys.stdout.flush()
                sys.stdout.write(f"\r{str(func.__name__).replace('_', ' ')}... {100*(i+1)/n}%")
                while True:
                    try:
                        V, Es, Psis = func()
                        break
                    except:
                        print('Failed to converge on a solution... skipping') # failed to find solution
                self.Vs.append(V)
                self.Es.append(Es)
                self.Psis.append(Psis)

        generate(n_double_well, self.generate_random_double_well)
        generate(n_square_well, self.generate_random_well)
        generate(n_sho, self.generate_random_sho)
        generate(n_polynomial, self.generate_random_polynomial)

        if cache:
            with open(cache, 'wb') as handle:
                pickle.dump((self.Vs, self.Es, self.Psis), handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('\n' + self.__str__())

    def generate_random_double_well(self):
        """Generates a random double well potential and its first 3 solutions.
        """
        V = np.ones(self.Nx)*1e10*(random.random()**10)
        start = int(random.random()*self.Nx)
        end = int(random.random()*(self.Nx-start))+start
        start2 = int(random.random()*(self.Nx-end))+end
        end2 = int(random.random()*(self.Nx-start2))+start2

        V[start:end] = 0
        V[start2:end2] = 0
        Es, Psis = discretized_one_dimensional_time_independent_Schrödinger_equation(V, n_solutions=self.n_solutions, dx=1) # pay attention to scale

        return V, Es, Psis

    def generate_random_well(self):
        """Generates a random square well potential and its first 3 solutions.
        """
        V = np.ones(self.Nx)*1e3*(random.random()**10)
        start = int(random.random()*self.Nx)
        end = int(random.random()*(self.Nx-start))+start

        V[start:end] = 0
        Es, Psis = discretized_one_dimensional_time_independent_Schrödinger_equation(V, n_solutions=self.n_solutions, dx=1) # pay attention to scale

        return V, Es, Psis

    def generate_random_sho(self):
        """Generates a random square simple harmonic oscillator potential and its first 3 solutions.
        """
        xs = np.linspace(-1,1, self.Nx)
        def f(x, params=[1,1]):
            return params[1] * (x+params[0])**2
        params = [random.random()-0.5, random.random()*5]
        V = [f(x, params) for x in xs]
        V -= np.min(V)
        Es, Psis = discretized_one_dimensional_time_independent_Schrödinger_equation(V, n_solutions=self.n_solutions, dx=1) # pay attention to scale

        return V, Es, Psis
    
    def generate_random_polynomial(self):
        """Generates a random 5th degree polynomial potential and its first 3 solutions.
        """
        xs = np.linspace(-1,1, self.Nx)
        def f(x, params=[]):
            return np.sum([(x**i)*param for i, param in enumerate(params)])
        params = [(random.random()-0.5)*1 for i in range(5)]
        V = [f(x, params) for x in xs]
        V -= np.min(V)
        Es, Psis = discretized_one_dimensional_time_independent_Schrödinger_equation(V, n_solutions=self.n_solutions, dx=1) # pay attention to scale

        return V, Es, Psis

    def __str__(self):
        return f'Random Potendial dataset with [{len(self.Vs)}, {len(self.Vs[0])}] potentials and [{len(self.Es)}, {len(self.Es[0])}] eigenvalues'
    
    def __len__(self):
        return len(self.Vs)

    def __getitem__(self, idx):
        V, E, Psi = self.Vs[idx], self.Es[idx], self.Psis[idx]
        
        return V, E, Psi

if __name__ == '__main__':
    data = Dataset_Random_Potentials(n_polynomial=20000, n_double_well=20000, n_square_well=20000, n_sho=20000, cache='data/randomly_generated_dataset.pkl')