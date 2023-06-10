# Applications

"""#Loi normale centrée réduite
data = np.random.normal(loc=0, scale=1, size=1000)
estimation_densite_probabilite(data)

#Loi exponentielle de paramètre 1
data=np.random.exponential(scale=1, size= 1000)

#Loi uniforme
data =np.random.rand(1000000)

# En combinant 2 lois normales
N = 10000
data = np.concatenate((np.random.normal(0, 1, int(0.3 * N)),
                    np.random.normal(5, 1, int(0.7 * N))))"""

#Test
estimation_densite_probabilite(data)