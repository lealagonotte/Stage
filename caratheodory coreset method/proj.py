import math
import numpy as np
import cmath
import scipy.stats as ss
import math
import matplotlib.pyplot as plt
import cmath

#Paramètres :
N=100
d=1
n=1000
vect_init = [i/(n-1) for i in range(n)]


def generer_vecteurs(vectors):
    """Crée et renvoie l'ensemble des points qu'on considère (de la forme (phi_j),j in [1, N] i in [1, n])
    vectors est l'ensemble des X_j donc une liste de liste"""
    base_final=[]
    for k in range(len(vectors)) :
      vect=[]
      for i in range(1,N+1) :
        if i==1 :
          vect.append(1)
        elif i%2==1 :
          vect.append(math.sqrt(2)*math.cos(2*math.pi*(i//2)*vectors[k]))
        elif i%2==0 :
          vect.append(math.sqrt(2)*math.sin(2*math.pi*(i//2)*vectors[k]))
      base_final.append(vect)
      #pour chaque X_i on crée le vecteur associé avec tous les phi_j
    return base_final



def generer_vecteurs2(vectors):
    """Crée et renvoie l'ensemble des points qu'on considère (de la forme (e^(i(X_j,w)), w in A)
    vectors est l'ensemble des X_j donc une liste de liste"""
    base_totale=[]
    for k in range(len(vectors)) :
      base=[]
      for i in range(N) :
        if i==0 :
          base.append(1)
        elif i%2==1 :
          base.append(math.sqrt(2)*math.cos(2*math.pi*(i//2)*vectors[k]))
        elif i%2==0 :
          base.append(math.sqrt(2)*math.sin(2*math.pi*(i//2)*vectors[k]))
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
    old=math.inf
    i=0
    j=0
    while i<len(vector) :
        if vector[i]!=0 :

            a = (-lambd[i]  /vector[i])

            if abs(a)<abs(old)  :
                old=a
                j=i
        i+=1
    if old==math.inf :
      old=np.min(lambd)
    v=old * vector
    print(len(v))
    res = lambd+ v
    return res,j #renvoie le lambda_i et l'indice de la composante qu'on a annulé


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
    print(sum(indice))
    print(ite)
    #print(lambd)
    #Met à jour le tableau des lambdfin car on voit qu'en itérant les poids finaux seront le produit des lambda_i à chaque itération i
    if nb>N+1 : #tant que le nb est supérieur à D+1 on peut encore en supprimer par le théorème de Carathéodory  donc on itére
        iteration(*vect, ite=ite, lambfin=lambfin, indice=indice, lambd_old=lambd_new)
    return lambfin
res=iteration(*generer_vecteurs(vect_init), ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)], lambd_old=[1/n for i in range (n)])
print(res)
print(sum(res))

def estimateur(Xj):
    """ représent l'estimateur"""
    # Définir les limites de la grille

    alpha_est=[]
    alpha=[]

    vecteurs=generer_vecteurs(Xj)

    coeff=iteration(*vecteurs, ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)], lambd_old=[1/n for i in range (n)])
    for i in range(N) :
      sum2=0
      sum=0
      for j in range(len(vecteurs)):
          sum+=vecteurs[j][i]*(1/n)
          sum2+=vecteurs[j][i]*coeff[j]
      alpha.append(sum)
      alpha_est.append(sum2)


    Xj = np.array(Xj)

    x = np.linspace(0, 1, 1000)
    y = np.random.uniform(0,1, size=1000)
    # Générer la grille de points

    # Calculer les densités de probabilité pour chaque point de la grille
    Z = []
    Z2=[]

    Y=generer_vecteurs2(x)

    for k in range(len(Y)) :
      z = 0
      z2=0
      for i in range(len(alpha)):
        z +=  alpha[i]*Y[k][i]
        z2+=alpha_est[i]*Y[k][i]
      Z.append(z)
      Z2.append(z2)
    # Reshape Z back to the shape of X and Y


    # Tracer la représentation de la densité de probabilité
    plt.figure(figsize=(8, 6))
    plt.plot(x,Z2, label="caratheodory")
    plt.plot(x, Z, label="estimation")
    plt.plot(x,y, label="vrai")
    plt.xlabel('x')
    plt.ylabel('estimateur')
    plt.title('Représentation de l estimateur en dimension  1')
    plt.legend()
    plt.show()

estimateur(vect_init )


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

vectoors= generer_vecteurs(vect_init)
print(test(vectoors, iteration(*vectoors, ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)], lambd_old=[1/n for i in range (n)])))



##dimension 2


import math
import numpy as np
import cmath
import scipy.stats as ss
import math
import matplotlib.pyplot as plt
import cmath

#Paramètres :
N=150
d=2
n=200
x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)

# Regrouper les coordonnées x et y dans un tableau numpy de dimension 2
vect_init = np.column_stack((x, y))
print(vect_init)

def generer_vecteurs(vectors):
    """Crée et renvoie l'ensemble des points qu'on considère (de la forme (phi_j),j in [1, N] i in [1, n])
    vectors est l'ensemble des X_j donc une liste de liste pour la coordonnée c sachant qu'on est en dimension 2 donc i vaut soit 1 soit 0"""
    base_final=[]
    for k in range(len(vectors)) :
      vect=[]
      for i in range(1,N+1) :
        for j in range(1,N+1):
            if i==1 and j==1:
              vect.append(1)
            elif i==1 and j%2==0 :
              vect.append(math.sqrt(2)*math.sin(2*math.pi*(j//2)*vectors[k][1]))
            elif i==1 and j%2==1 :
              vect.append(math.sqrt(2)*math.cos(2*math.pi*(j//2)*vectors[k][1]))
            elif j==1 and i%2==0 :
              vect.append(math.sqrt(2)*math.sin(2*math.pi*(i//2)*vectors[k][0]))
            elif j==1 and i%2==1 :
              vect.append(math.sqrt(2)*math.cos(2*math.pi*(i//2)*vectors[k][0]))
            elif i%2==0 and j%2==0 :
              vect.append(2*math.sin(2*math.pi*(i//2)*vectors[k][0])*math.sin(2*math.pi*(j//2)*vectors[k][1]))
            elif i%2==0 and j%2==1 :
              vect.append(2*math.sin(2*math.pi*(i//2)*vectors[k][0])*math.cos(2*math.pi*(j//2)*vectors[k][1]))
            elif i%2==1 and j%2==0 :
              vect.append(2*math.sin(2*math.pi*(j//2)*vectors[k][1])*math.cos(2*math.pi*(i//2)*vectors[k][0]))
            elif i%2==1 and j%2==1 :
              vect.append(2*math.cos(2*math.pi*(j//2)*vectors[k][0])*math.cos(2*math.pi*(i//2)*vectors[k][1]))
      base_final.append(vect)
      #pour chaque X_i on crée le vecteur associé avec tous les phi_j
    return base_final




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
    old=math.inf
    i=0
    j=0
    while i<len(vector) :
        if vector[i]!=0 :

            a = (-lambd[i]  /vector[i])

            if abs(a)<abs(old)  :
                old=a
                j=i
        i+=1
    if old==math.inf :
      old=np.min(lambd)
    v=old * vector
    print(len(v))
    res = lambd+ v
    return res,j #renvoie le lambda_i et l'indice de la composante qu'on a annulé

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
    print(sum(indice))
    print(ite)
    #print(lambd)
    #Met à jour le tableau des lambdfin car on voit qu'en itérant les poids finaux seront le produit des lambda_i à chaque itération i
    if nb>N**2+1 : #tant que le nb est supérieur à D+1 on peut encore en supprimer par le théorème de Carathéodory  donc on itére
        iteration(*vect, ite=ite, lambfin=lambfin, indice=indice, lambd_old=lambd_new)
    return lambfin
res=iteration(*generer_vecteurs(vect_init), ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)], lambd_old=[1/n for i in range (n)])
print(res)
print(sum(res))



def estimateur(Xj):
    """ représente l'estimateur"""
    # on initialise les listes alpha à 0. Pour des raisons pratiques j'ai choisi de les représenter par une liste et non pas par une matrice

    alpha_est=[0 for j in range(N**2)] #celle ou on a réduit le nombre de données
    alpha=[0 for j in range(N**2)] #alpha total

   #on crée les vecteurs qu'on considère (les (phi_j(x_i))j, i)
    vecteurs=generer_vecteurs(Xj)
  #on applique l'algorithme de Carathéodory à nos vecteurs (pour cette partie c 'est vraiment similaire à ce qu'on a fait avec les estimateurs à noyaux)
    coeff=iteration(*vecteurs, ite= 0, lambfin = [1 for i in range(n)], indice=[1 for i in range(n)], lambd_old=[1/n for i in range (n)])
    for i in range(N**2) :
    #on calcule les alpha (alpha_i,j = 1/n * somme(v) et les alpha_i,j = somme(nouveaux coeff * v))
        sum2=0
        sum=0
        for j in range(len(vecteurs)):
            sum+=vecteurs[j][i]*(1/n)
            sum2+=vecteurs[j][i]*coeff[j]
        alpha[i]=sum
        alpha_est[i]=sum2


    Xj = np.array(Xj)

    x = np.linspace(0,1, 100)
    y = np.linspace(0,1, 100)

    # Générer la grille de points
    X, Y = np.meshgrid(x, y)
    pos = np.column_stack((X.ravel(), Y.ravel()))  # Flatten X and Y to generate pos
  # Calculer les densités de probabilité pour chaque point de la grille
    Z = []
    Z2=[]
    #print(pos)
    P=generer_vecteurs(pos)
    print("P=",len(P))
    print("pos=", (len(pos)), len(pos[0]))
    #on reconstruit l'estimateur à partir des alpha
    moy=0
    for k in range(len(P)) :
        z = 0
        z2=0
        for i in range(len(alpha)):
          z +=  alpha[i]*P[k][i]
          z2+=alpha_est[i]*P[k][i]
        moy+=(z2/len(P))
        Z.append(z)
        Z2.append(z2)
    print("Z=",len(Z))

    print("moyenne de Z2 est", moy)
    #print("moyenne de Z2 est", sum(Z2)/len(Z2))

    # Tracer la représentation de la densité de probabilité
    Z = np.reshape(Z, X.shape)
    plt.pcolormesh(X, Y,Z, shading='auto', cmap='viridis')

    plt.colorbar(label='Densité de probabilité')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Représentation d\'une loi multivariée en dimension 2')
    plt.show()


estimateur(vect_init )
