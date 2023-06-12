from program import iteration, generer_vecteurs, create_matrix
import numpy as np
import math
#Paramètres :
d=2
T=15 #choisir T en fonction des résultats qu'on a après
n=100
D=2*(1+4*T/math.pi)**d

vect_init=vect_init=np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]), 100)
vectors=generer_vecteurs(T, vect_init)
n=len(vectors) #longueur initiale
print(vectors)


#create_matrix(*vectors)

iteration(*vectors, iter= 0, lambfin = [1 for i in range(n)])