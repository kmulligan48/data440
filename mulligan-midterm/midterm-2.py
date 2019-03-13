import numpy as np
import matplotlib.pyplot as plt

def main():
    dVC = 10 # value of d_{VC} given
    epsilon = 0.05 # value of epsilon given
    delta = 0.05 # value of delta given
    Nlst = []
    N = 1000 # initial value of N
    Nlst.append(N)
    for i in range(1,20):
        N = (8 / (np.power(epsilon, 2))) * (np.log((4 * (np.power((2 * N), 
            (dVC)) + 1)) / delta))
        Nlst.append(N)      
    plt.xlabel("Iteration")
    plt.ylabel("Value of N")
    plt.title("Convergence of N")
    iter = np.arange(20)
    plt.plot(iter,Nlst,'b.')
    print(N)

main()