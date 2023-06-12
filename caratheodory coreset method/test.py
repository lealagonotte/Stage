from program.py import iteration, generer_vecteurs



vect_init=[np.random.normal(loc=0, scale=1, size=1000)]
vectors=generer_vecteurs(T, vect_init)
n=len(vectors) #longueur initiale

#Paramètres :
d=2
T=15 #choisir T en fonction des résultats qu'on a après
n=100
D=2*(1+4*T/math.pi)**d



iteration(vectors, 0, [1 for i in range(n)])