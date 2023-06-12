import math
import numpy as np

#Paramètres :
d=2
T=15 #choisir T en fonction des résultats qu'on a après
n=100
D=2*(1+4*T/math.pi)**d

def generer_vecteurs(T, vectors):
    """Crée et renvoie l'ensemble des points qu'on considère (de la forme (e^(i(X_j,w)), w in A)
    vectors est l'ensemble des X_j"""
    vecteurs = []
    #Parcours l'ensemble des coordonnées possibles pour vérifier la condition sur la norme
    for z in range(-int(T*2/3), int(T*2/3)):
        for y in range(-int(T*2/3), int(T*2/3)): 
            vecteur = (np.pi / 2) * np.array([z, y])
            #Vérifie la condition sur la norme infinie et la fréquence T
            if np.linalg.norm(vecteur, np.inf) < T:
                vecteurs.append(vecteur)
# Crée les exponentielles complexes
    prodsca = [[math.exp(1j * np.dot(v1, v2)) for v1 in vectors] for v2 in vecteurs]
    liste = [[(x.real, x.imag) for x in prodsca[i]] for i in range (len(prodsca))]
    return  liste

