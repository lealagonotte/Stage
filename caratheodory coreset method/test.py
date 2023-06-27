from program import iteration, generer_vecteurs, create_matrix
import numpy as np
import math
#Paramètres :
d=2
T=2 #choisir T en fonction des résultats qu'on a après
n=100
#D=2*(1+4*T/math.pi)**d
#print(D)
import cmath

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






def gaussian_kernel(x, sigma=1):
    return np.exp(-np.linalg.norm(x) ** 2 / (2 * (sigma ** 2)))

def reconstruction_noyau(noyau, Xj, A, coeff):
    n = len(Xj)
    d = len(Xj[0])
    Xj = np.array(Xj)

    x1 = np.linspace(min(Xj[:, 0]), max(Xj[:, 0]), 100)
    x2=np.linspace(min(Xj[:, 1]), max(Xj[:, 1]), 100)
    
    X, Y = np.meshgrid(x1, x2)
    pos = np.dstack((X, Y))
    print(pos)
    y = np.zeros_like(pos)
    inter=np.array([])
    for elmt in pos :
        inter=np.append(inter, noyau(elmt))
    

    fft = np.fft.fft(inter)

    for w in A:
        index1 = int(w[0] / (x1[1] - x1[0]))
        index2 = int(w[1] / (x2[1] - x2[0]))  # On trouve l'indice le plus proche
        
        pos_w=[index1, index2]
        fft_w = fft[pos_w]
        somme = 0
        for i in range(n):
            somme += coeff[i] * cmath.exp(1j * np.dot(w, Xj[i]))
        
        y += fft_w / (4 ** d) * somme * cmath.exp(-1j * np.dot(w, pos))

    z = np.zeros_like(x)
    for i in range(n):
        z += 1 / n * noyau(Xj[i] - pos)

    # Affiche le graphique de la densité de probabilité
    plt.plot(X, Y, np.real(y))
    plt.plot(X,Y, np.real(z))
    plt.xlabel('Valeur')
    plt.ylabel('Densité de probabilité')
    plt.title('Estimation de densité de probabilité')

    plt.show()

reconstruction_noyau(gaussian_kernel, vect_init,A, res )
