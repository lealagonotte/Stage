import math
import numpy as np
import cmath
import scipy.stats as ss


d=2
N=10

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





def generer_vecteurs( vectors):
    """Crée et renvoie l'ensemble des points qu'on considère (de la forme (e^(i(X_j,w)), w in A)
    vectors est l'ensemble des X_j donc une liste de liste"""
    base_totale=[]
    for k in range(len(vectors)) :
        base=[]
        base.append(1)
        for i in range (N):
            base.append(math.sqrt(2)*math.cos(2*math.pi*i*vectors[k]))
            base.append(math.sqrt(2)*math.sin(2*math.pi*i*vectors[k]))
        base_totale.append(base)
    return base_totale



#on a 2N+1 termes dans chaque vecteur


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
    if vector[0]!=0 :
        old=(- lambd[0] / vector[0])
    else :
        old=(- lambd[1] / vector[1])
    i=0
    j=0
    while i<len(vector) :
        if vector[i]!=0 :
            print(i)
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
    if nb>2*N+1 : #tant que le nb est supérieur à D+1 on peut encore en supprimer par le théorème de Carathéodory  donc on itére      

        iteration(*vect, ite=ite, lambfin=lambfin, indice=indice, lambd_old=lambd_new)
    return lambfin
