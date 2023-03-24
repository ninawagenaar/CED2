import numpy as np
import matplotlib.pyplot as plt
import time

# fixed variables
b = 0.6
c = 0.5
rho = 0.03
sigma = 0.05
a_min = 0.001
a_max = 1

# Initializing variables
x_min = 1
x_max = 4
n = 100
m = 100
h = 0.1
K = 400
state_space = np.linspace(x_min, x_max, n+1)
delta = state_space[1] - x_min

action_space = np.linspace(a_min, a_max, m+1)
eta = action_space[1] - a_min

# define the functions
def loss(x, a, c=c):
    return np.log(a) - c * x**2

def f(x, a, b=b):
    return a - b*x + (x**2/x**2+1)

# returns a vector (dimension n) with for every state the optimal action in the action space
# This function only works for the entire state space, if for a specific i Phi(x[i]) is needed,
# call the function with x and then take Phi[i].
def Phi(state_space, action_space=action_space, b=b, n=n):
    phi_vector = np.zeros(n+1)
    for i in range(n+1):
        approx = b*state_space[i] - (state_space[i]**2 / (state_space[i]**2 + 1))
        if approx in action_space:
            phi_vector[i] = approx
        else:
            differences = np.absolute(action_space-approx)
            phi_vector[i] = action_space[np.where(differences == differences.min())[0]]
    return phi_vector

def V_hat(V, y, mu=0.5, state_space=state_space):

    if y >= state_space[-1]:
        return V[-1]
    elif y <= state_space[0]:
        return V[0]
    
    else:
        for i in range(len(V)):
            if state_space[i] <= y and state_space[i+1] > y:
                return (1-mu) * V[i] + mu * V[i+1]

def g(xi, aj, V, h=h):
    return h * loss(xi, aj) + (np.exp(-rho * h)/2) * ( V_hat(V, xi + h * f(xi, aj) + np.sqrt(h) * sigma * xi) - V_hat(V, xi + h * f(xi, aj) + np.sqrt(h) * sigma * xi) )

def update_V(V, state_space=state_space, action_space=action_space):
    V_new = np.zeros(n+1)
    
    for i in range(n+1):

        maximizer = np.zeros(m+1)
        for j in range(m+1):
            
            maximizer[j] = g(state_space[i], action_space[j], V)
        V_new[i] = maximizer.max()

    return V_new

        

if __name__ == "__main__":
    V0 = 1/rho * loss(state_space, Phi(state_space))

    start_time = time.time()
    V = update_V(V0)
    norm = np.absolute(V - V0).max()

    for i in range(K):
        new_V = update_V(V)
        new_norm = np.absolute(new_V - V).max()
        print("k: ", i, "   runtime: ", (round(time.time()-start_time, 5)), "   max norm dk: ", new_norm, " quantity_log: ", np.log(norm/new_norm)/h)
        V = new_V
        norm = new_norm
        
    plt.plot(V, state_space)
    plt.xlabel("Vk")
    plt.ylabel("state space x")
    plt.show()


