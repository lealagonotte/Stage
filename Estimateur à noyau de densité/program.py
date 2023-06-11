##en 1D
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

def estimation_densite(data):
    # Crée une estimation de densité de probabilité à partir des données (noyau gaussien)
    kde = ss.gaussian_kde(data)

    # Génére un échantillon de points pour représenter la courbe de densité
    x = np.linspace(min(data), max(data), 1000)

    # Calcule la densité de probabilité pour chaque point de l'échantillon
    y = kde(x)

    # Affiche le graphique de la densité de probabilité
    plt.plot(x, y)
    plt.xlabel('Valeur')
    plt.ylabel('Densité de probabilité')
    plt.title('Estimation de densité de probabilité')
    plt.hist(data, density=True)
    plt.show()




##en multi dimensionnel

"""def estimation_densite_md(data, pas):
    # Crée une estimation de densité de probabilité à partir des données
    kde = ss.gaussian_kde(data.T)

    # Crée une grille de points pour le calcul de la densité (np.linspace en 2D)
    x_min, x_max = min(data[:, 0]), max(data[:, 0])
    y_min, y_max = min(data[:, 1]), max(data[:, 1])

    x, y = np.mgrid[x_min:x_max:pas, y_min:y_max:pas] #Gé,ère une grille et Espace régulièrement tous les points, h est le pas 
    positions = np.vstack([x.ravel(), y.ravel()]) #Donne une liste 1D de coordonnées (x,y)

    # Calcule la densité de probabilité pour chaque point de la grille
    z = np.reshape(kde(positions).T, x.shape) 
    #Remodule en une matrice 2D qui correspond aux dimensions de la grille

    # Affiche le graphique de la densité de probabilité
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(x, y, z, shading='auto', cmap='viridis')
    plt.colorbar(label='Densité de probabilité')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Estimation de densité de probabilité en dimension 2')
    plt.show()
"""
# Exemple d'utilisation en dimension 2
data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000)
pas = 0.1

#estimation_densite_md(data, pas)



def plot_multivariate_distribution(mean, cov):
    # Créer une instance de la distribution multivariée
    mvn = ss.multivariate_normal(mean=mean, cov=cov)

       # Définir les limites de la grille
    x_min, x_max = mean[0] - 3 * np.sqrt(cov[0, 0]), mean[0] + 3 * np.sqrt(cov[0, 0])
    y_min, y_max = mean[1] - 3 * np.sqrt(cov[1, 1]), mean[1] + 3 * np.sqrt(cov[1, 1])

    # Générer la grille de points
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Calculer les densités de probabilité pour chaque point de la grille
    density = mvn.pdf(pos)

    # Tracer la représentation de la densité de probabilité
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, density, shading='auto', cmap='viridis')
    plt.colorbar(label='Densité de probabilité')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.title('Représentation d\'une loi multivariée en dimension 2')
    plt.show()   # Définir les limites de la grille
    
plot_multivariate_distribution(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]))
 #on observe que ça colle plutôt bien même si moins bien que en dimension 1

 #dans ss.gaussian_kde h est calcule automatiquement
 #quelle influence a h sur l'estimateur?
 

