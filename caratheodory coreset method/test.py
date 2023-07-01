from program import iteration, generer_vecteurs, create_matrix
import numpy as np
import math
import matplotlib.pyplot as plt
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
    return np.exp(-(x[0]**2+ x[1]**2)/ (2 * (sigma ** 2)))







def estimateur(noyau, Xj, A, coeff):
    """ représent l'estimateur"""
    # Définir les limites de la grille
    n = len(Xj)
    d = len(Xj[0])
    Xj = np.array(Xj)

    x = np.linspace(min(Xj[:, 0]), max(Xj[:, 0]), 100)
    y = np.linspace(min(Xj[:, 1]), max(Xj[:, 1]), 100)
    # Générer la grille de points
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    pos = np.reshape(pos, (-1, 2))  # Flatten pos to a 2-dimensional array

    # Calculer les densités de probabilité pour chaque point de la grille
    Z = []
    for e in pos:
        z = 0
        for i in range(n):
            z += (1 / n) * noyau(Xj[i] - e)
        Z.append(z)

    # Reshape Z back to the shape of X and Y
    Z = np.reshape(Z, X.shape)

    # Tracer la représentation de la densité de probabilité
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.colorbar(label='Densité de probabilité')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Représentation de l estimateur en dimension 2')
    plt.show()

reconstruction_noyau2(gaussian_kernel, vect_init,A, res )




def reconstruire_coreset_estimator_f(noyau, Xj, A, coeff):
    """reconstruit l'estimateur construit grâce à la méthode de carathéodory en utilisant la transformée de Fourier inverse"""
    # Définir les limites de la grille
    n = len(Xj)
    d = len(Xj[0])
    Xj = np.array(Xj)

    x = np.linspace(min(Xj[:, 0]), max(Xj[:, 0]), 100)
    y = np.linspace(min(Xj[:, 1]), max(Xj[:, 1]), 50)
    print(x)
    # Générer la grille de points
    X, Y = np.meshgrid(x, y)
    pos = np.column_stack((X.ravel(), Y.ravel()))  # Flatten X and Y to generate pos

    # Calculer les densités de probabilité pour chaque point de la grille
    Z = []
    for e in pos:
        Z.append(noyau(e))

    inter = np.array(Z, dtype=np.complex128)
    fft = np.fft.fft(inter)
    print(fft)
    res = np.zeros_like(X, dtype=np.complex128)  # Initialize Y with zeros


    for i, e in enumerate(pos):
        for w in A:
            #print(w)
            index1 = abs(int((w[0]-x[0]) / (x[1] - x[0])))
            index2 = abs(int((w[1]-y[0]) / (y[1] - y[0])))
            pos_w = np.array([index1, index2])  # Construct pos_w as a 1D array
            #print(pos_w)
            index = np.where((pos == np.array([x[index1], y[index2]])))[0][0]  # Find index in pos array
            #print(index)
            fft_w = fft[index]
            somme = 0
            for j in range(n):
                somme += coeff[j] * np.exp(1j * np.dot(w, Xj[j]))

            res.ravel()[i] += (fft_w / (4 ** d)) * somme * np.exp(-1j * np.dot(w, e))


    if len(res.shape) == 1:
        numRows = 1
        numCols = res.shape[0]
    else:
        numRows, numCols = res.shape
    # Tracer la représentation de la densité de probabilité
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y,np.abs(res), shading='auto', cmap='viridis')

    plt.colorbar(label='Densité de probabilité')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Représentation d\'une loi multivariée en dimension 2')
    plt.show()



def reconstruction_noyau(noyau, Xj, A, coeff):
    """reconstruit l'estimateur basé sur la méthode de carathéodory"""
    # Définir les limites de la grille
    n = len(Xj)
    d = len(Xj[0])
    Xj = np.array(Xj)

    x = np.linspace(min(Xj[:, 0]), max(Xj[:, 0]), 100)
    y = np.linspace(min(Xj[:, 1]), max(Xj[:, 1]), 100)
    # Générer la grille de points
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    pos = np.reshape(pos, (-1, 2))  # Flatten pos to a 2-dimensional array

    # Calculer les densités de probabilité pour chaque point de la grille
    Z = []
    for e in pos:
        z = 0
        for i in range(n):
            if coeff[i]!=0 :
                
                z += coeff[i] * noyau(Xj[i] - e)
        Z.append(z)

    # Reshape Z back to the shape of X and Y
    Z = np.reshape(Z, X.shape)

    # Tracer la représentation de la densité de probabilité
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.colorbar(label='Densité de probabilité')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Représentation d\'une loi multivariée en dimension 2')
    plt.show()


reconstruction_noyau(gaussian_kernel, vect_init,A, res )




