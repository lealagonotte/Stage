from program import iteration, generer_vecteurs, create_matrix
import numpy as np
import math
#Paramètres :
d=2
T=2 #choisir T en fonction des résultats qu'on a après
n=100
#D=2*(1+4*T/math.pi)**d
#print(D)


#construit l'ensemble A
A= []
    #Parcours l'ensemble des coordonnées possibles pour vérifier la condition sur la norme
for z in range(-int(T*2/3), int(T*2/3)+1):
    for y in range(-int(T*2/3), int(T*2/3)+1): 
        vecteur = (np.pi / 2) * np.array([z, y])
            #Vérifie la condition sur la norme infinie et la fréquence T
        if np.linalg.norm(vecteur, np.inf) <= T:
            A.append(vecteur)
#print(A)

D=len(A)


# on part d'une loi multinomiale
vect_init=np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]), 100)

#on construit les vj
vectors=generer_vecteurs(T, vect_init, A)
n=len(vectors)
print(len(vect_init)) #longueur initiale
print(n)

print(vectors)
#print(vect_init)
#create_matrix(*vectors)



#on itère
res=iteration(*vectors, ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)])
print(res)

#test :
def test(vectors, liste_coeff) :
    bool=True
    somme1=0
    somme2=0
    for i in range(n) :
        somme1+=np.multiply((1/n),vectors[i])

        somme2+=np.multiply(liste_coeff[i],vectors[i])
    for j in range(len(vectors[0])) :
        if abs(somme1[j]-somme2[j] )>10**(-2):
            bool=False
    return bool, (somme1-somme2)


print(test(vectors, res))

def reconstruction_noyau(noyau, Xj, A, coeff,x ) :
    """Reconstruit les deux estimateurs et les trace pour les comparer"""
    X=np.linspace(-n,n,4*n)
    Y=
