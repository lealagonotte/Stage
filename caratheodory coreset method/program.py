import math
import numpy as np
import cmath
import scipy.stats as ss

#La méthode de Carathéodory pour les estimateurs à noyau

#Paramètres
d=2 #dimension
T=2 #limite de la fenêtre
n=100
#T=n**(1/(2+d)*math.log(n) #idéalement, pour avoir la meilleure estimation, cf la théorie

#D=int(2*(1+4*T/math.pi)**d) #majoration de a


#on construit A
A= []
    #Parcours l'ensemble des coordonnées possibles pour vérifier la condition sur la norme
for z in range(-int(T*2/3), int(T*2/3)+1):
    for y in range(-int(T*2/3), int(T*2/3)+1): 
        vecteur = (np.pi / 2) * np.array([z, y])
            #Vérifie la condition sur la norme infinie et la fréquence T
        if np.linalg.norm(vecteur, np.inf) <= T:
            A.append(vecteur)
#print(A)

D=2*len(A) #Vrai A


#Des exemples de lois pour teste

# Paramètres de la distribution exponentielle
lambda_1 = 0.5  # Taux de décroissance pour la première dimension
lambda_2 = 0.8  # Taux de décroissance pour la deuxième dimension

# Taille de l'échantillon
n = 100

# Génération des données
data = np.zeros((n, 2))  # Tableau pour stocker les données

for i in range(n):
    # Génération des valeurs pour chaque dimension
    x1 = np.random.exponential(scale=1/lambda_1)
    x2 = np.random.exponential(scale=1/lambda_2)
    
    data[i] = [x1, x2]  # Stockage des valeurs dans le tableau

print(data)




#début des fonctions
def generer_vecteurs(T, vectors, A):
    """Crée et renvoie l'ensemble des points qu'on considère (de la forme (e^(i(X_j,w)), w in A)
    vectors est l'ensemble des X_j donc une liste de liste"""
   
# Crée les exponentielles complexes
    prodsca = [[cmath.exp(1j * np.dot(v1, v2)) for v1 in A] for v2 in vectors]
    print(len(prodsca))
    #print(prodsca)
    liste =[]
    for i in range(len(prodsca)) :
        ss_liste=[]
        for x in prodsca[i] :
            ss_liste.append(x.real)
            ss_liste.append(x.imag)
            
        liste.append(ss_liste)
    return  liste


"""vect_init=np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]), 100)
vectors=generer_vecteurs(T, vect_init)
n=len(vectors) #longueur initiale"""


def create_matrix(*vectors):
    """
    Crée et renvoie la matrice M définie dans l'article

    Paramètres : vectors, ce sont les v_j qu'on met en paramètre, l'étoile permet de mettre un nombre d'arguments variables
    (le nombre de vecteurs diminuent au moins de 1 ) chaque étape

  
    """

    num_vectors = len(vectors)
    vector_size = len(vectors[0]) # Tous les vecteurs ont la meme taille

    # Création de la matrice
    matrix = np.zeros((vector_size +1, num_vectors))
    for i, vector in enumerate(vectors):
        matrix[:vector_size, i] = np.transpose(vector)
    #La dernirère ligne est composée de 1
    matrix[-1, :] = 1
    return matrix

def vecteur_propre(matrix):
    """
    Trouve et renvoie un vecteur non nul du noyau d'une matrice donnée en utilisant la décomposition en valeurs singulières (SVD).

    Arguments :
    - matrix : La matrice pour laquelle nous voulons trouver un vecteur non nul du noyau.

    """

    _, _, vh = np.linalg.svd(matrix)
    kernel_vector = vh[-1, :]

    return kernel_vector


def find_alpha( vector, lambd):
    """Trouve et renvoie un vecteur alpha qui annule une des composantes du vecteur appelé lambda dans l'article*
    il faut trouver alpha de telle sorte que les autres coefficients ne soient pas négatifs donc annuler le minimum en fait doit marche
    Paramètres : 
    
    -vector : vector est un vecteur du noyau calculé grâce à la fonction précédente"""   
   
    old=(- lambd[0] / vector[0])
    i=0
    j=0
    while i<len(vector) :
        if vector[i]!=0 :
            a = (-lambd[i]  /vector[i]) 
        
            if abs(a)<abs(old)  :
                old=a
                j=i 
            i+=1
    
    v=old * vector
    print(len(v))
    res = lambd+ v
    return res,j #renvoie le lambda_i et l'indice de la composante qu'on a annulé

##Itère la procédure tant que il reste plus de D+1 termes dans la somme
def iteration(*vectors, ite, lambfin, indice, lambd_old) :
    """ Itère les deux étapes jusqu'à ce qu'on ait plus que D+1 vecteurs
    Paramètrees :
    -iter : indique à quelle itération on est 
    -lambdfin -> fait le produit des lambda i au fur et à mesure, ce sont ces coefficients que l'on cherche à avoir
    -indice : tableau de 0 ou de 1. Si indice[j]=0 alors on a annulé le coeff devant la composante j sinon elle est encore là"""
    M=create_matrix(*vectors) #on crée la matrice
    
    nb=len(vectors) #nb de vecteurs qu'il nous reste 
    ite+=1 # pour savoir combien d'itération on a déjà fait
    
    #Trouve le vecteur du noyau non nul de M
    v=vecteur_propre(M) 

    #Donne lambda_j et la composante qu'on annule
    (lambd,j)=find_alpha(v, lambd_old) 
    
    # Met à jour le tableau de vecteur (pour les histoires de moyenne, on doit multiplier les nvx vecteurs par n-iter)
    print(sum(lambd))
    vect = []
    lambd_new=[]
    for i in range(len(vectors)):
        v = np.array(vectors[i], dtype=np.float64)
        
        if i != j:  # Exclude the element at index j
            vect.append(v)
            lambd_new.append(lambd[i])
    # Enlève la composante j car c'est celle qu'on a annulé, on réduit ainsi le nb de vecteurs à l'étape suivante
    print(indice)
    lambd_new=np.array(lambd_new, dtype=np.float64)
    rg=0
    for k in range(len(lambd)) :
        if indice[rg]==0 :
            while indice[rg]== 0  :
                rg+=1
        lambfin[rg]=lambd[k]
        if j==k :
            lambfin[rg]=0
            indice[rg]=0
        #print(rg)
        rg+=1
   
    print(ite)
    #print(lambd)
    #Met à jour le tableau des lambdfin car on voit qu'en itérant les poids finaux seront le produit des lambda_i à chaque itération i
    if nb>D+1 : #tant que le nb est supérieur à D+1 on peut encore en supprimer par le théorème de Carathéodory  donc on itére      

        iteration(*vect, ite=ite, lambfin=lambfin, indice=indice, lambd_old=lambd_new)
    return lambfin #renvoie les poids associés au coreset


