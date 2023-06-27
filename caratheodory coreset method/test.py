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

D=2*len(A)


# on part d'une loi multinomiale
vect_init=np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]), 100)

#on construit les vj
vectoors=generer_vecteurs(T, vect_init, A)

print("M est de taille", len(create_matrix(*vectoors)[0]))



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

"""def reconstruction_noyau(noyau, Xj, A, coeff,x ) :
    #Reconstruit les deux estimateurs et les trace pour les comparer
    x = np.linspace(min(Xj[0]), max(Xj[0]), 1000)
    y=0
    inter=noyau(x)
    for w in A :
        index = int(w / (x[1] - x[0])) #on trouve l'indice le plus proche
        fft=inter[index]
        somme=0
        for i in range(n):
            
        y += np.fft.fft()/(4**d)*

    # Affiche le graphique de la densité de probabilité
    plt.plot(x, y)
    plt.xlabel('Valeur')
    plt.ylabel('Densité de probabilité')
    plt.title('Estimation de densité de probabilité')
    plt.hist(data, density=True)
    plt.show()
"""
