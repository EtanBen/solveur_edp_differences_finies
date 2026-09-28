import numpy as np
import matplotlib.pyplot as plt

# Paramètres

d = 0.005# Coefficient de diffusion
c = 0   # Coefficient de convection
ajustement=0.95
# Pas définis par l'utilisateur
hx = 0.009 # Pas en espace
ht = 0.5*(hx**2/d)*ajustement  # Pas en temps

# Définition des bornes
Nx = 1  # Longueur de l'intervalle spatial
Nt = 4# Durée de l'intervalle temporel

# Calcul du nombre de points à partir des pas
nbrx = int(Nx / hx) + 1  # Nombre de points en espace
nbrt = int(Nt / ht) + 1  # Nombre de points en temps

# Discrétisation des axes
x = np.linspace(0, Nx, nbrx)  # Vecteur spatial
t = np.linspace(0, Nt, nbrt)  # Vecteur temporel

# Initialisation des solutions
solutionsApprochées = np.zeros((nbrx, nbrt))
solutionExacte = np.zeros((nbrx, nbrt))

# Fonction solution exacte
def u(x, t):
    return np.sin(np.pi * x) * (1 + t)

def f(x,t) :
    return np.pi**2*(t+1)*d*np.sin(np.pi*x)+np.pi*(t+1)*c*np.cos(np.pi*x)+np.sin(np.pi*x)

# Conditions initiales
for i in range(nbrx):
    solutionsApprochées[i, 0] = u(x[i], 0)  # Condition initiale en t=0

for j in range(nbrt):
    solutionsApprochées[0, j] = u(0, t[j])  # Condition aux bords en x=0
for j in range(nbrt):
    solutionsApprochées[nbrx-1, j] = u(1, t[j])   # Condition aux bords en x=1

# Résolution avec différences finies
for n in range(nbrt - 1):  # Parcours temporel
    for i in range(1, nbrx - 1):  # Parcours spatial
            solutionsApprochées[i, n + 1] =( solutionsApprochées[i, n] + ht * (d * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n]) / hx**2
                - c * (solutionsApprochées[i + 1, n] - solutionsApprochées[i, n]) / hx + f(x[i], t[n])))

# Calcul de la solution exacte
for n in range(nbrt):
    for i in range(nbrx):
        solutionExacte[i, n] = u(x[i], t[n])

# Calcul de l'écart
erreur = np.abs(solutionsApprochées - solutionExacte)

# Fonction pour tracer les solutions pour tous les instants t
def tracer_solutions_tous_instants():
    plt.figure(figsize=(12, 8))

    for n in range(nbrt):
        plt.clf()  # Efface la figure précédente
        plt.plot(x, solutionExacte[:, n], label="Solution exacte", linewidth=2)
        plt.plot(x, solutionsApprochées[:, n], label="Solution approchée", linestyle="--", linewidth=2)
        plt.title(f"Solutions exacte et approchée à t={t[n]:.2f} ")
        plt.xlabel("Espace (x)")
        plt.ylabel("Valeur de la solution")
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)  # Pause pour visualisation pas à pas
    plt.show()

    

# Tracé des solutions exacte et approchée pour tous les instants
tracer_solutions_tous_instants()

# Tracé de l'écart
plt.figure(figsize=(10, 6))
plt.imshow(erreur, extent=[0, Nt, 0, Nx], origin='lower', aspect='auto', cmap='hot')
plt.colorbar(label='Écart absolu')
plt.title("Écart entre la solution approchée et la solution exacte")
plt.xlabel("Temps (t)")
plt.ylabel("Espace (x)")
plt.show()
