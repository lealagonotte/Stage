from program import iteration, generer_vecteurs, create_matrix
import numpy as np
import math
#Paramètres :
d=2
T=1 #choisir T en fonction des résultats qu'on a après
n=100
D=2*(1+4*T/math.pi)**d

vect_init=np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]), 100)
vectors=generer_vecteurs(T, vect_init)
n=len(vectors)
print(len(vect_init)) #longueur initiale
print(n)
print(D)
#print(vectors)
#create_matrix(*vectors)

res=iteration(*vectors, ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)])
print(res)
somme = 0
for element in res:
    somme += element

print("somme",somme)