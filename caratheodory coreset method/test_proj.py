from proj import iteration, generer_vecteurs, create_matrix
import numpy as np
import math
import matplotlib.pyplot as plt
#Paramètres :
N=100
d=2
n=100
import cmath

#construit l'ensemble A



# on part d'une loi multinomiale
vect_init=np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]), 100)

#on construit les vj
vectoors=generer_vecteurs( vect_init)





#on itère
res=iteration(*vectoors, ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)], lambd_old=[1/n for i in range (n)])
print(res)
print(sum(res))

#test :
def test(vectors, liste_coeff) :
    bool=True
    somme1=0
    somme2=0
    for i in range(n) :
        v = np.array(vectors[i], dtype=np.float64)
        somme1+=v*(1/n)

        somme2+=v*liste_coeff[i]
    for j in range(len(vectors[0])) :
        if abs(somme1[j]-somme2[j] )>10**(-2):
            bool=False
    return bool, (somme1-somme2)

print(sum(res))
print(test(vectoors, res))
