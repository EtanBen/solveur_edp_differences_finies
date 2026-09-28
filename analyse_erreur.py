import numpy as np
import matplotlib.pyplot as plt
d=0.005
c=0.005
Nx=1
Nt=1
hx=0.009
def simulate_diffusion(ht, d=d, c=c, Nx=Nx, Nt=Nt, hx=hx):

    # Calcul du nombre de points
    nbrx = int(Nx / hx) + 1
    nbrt = int(Nt / ht) + 1

    # Discrétisation des axes
    x = np.linspace(0, Nx, nbrx)
    t = np.linspace(0, Nt, nbrt)

    # Initialisation des solutions
    solutionsApprochées = np.zeros((nbrx, nbrt))
    solutionExacte = np.zeros((nbrx, nbrt))

    # Fonction solution exacte
    def u(x, t):
        return np.sin(np.pi * x) * (1 + t)

    # Terme source
    def f(x, t):
        return (
            np.pi**2 * (t + 1) * d * np.sin(np.pi * x)
            + np.pi * (t + 1) * c * np.cos(np.pi * x)
            + np.sin(np.pi * x)
        )

    # Conditions initiales
    for i in range(nbrx):
        solutionsApprochées[i, 0] = u(x[i], 0)
    # Conditions aux bords 
    for j in range(nbrt):
        solutionsApprochées[0, j] = u(0, t[j])     # Au début 
        solutionsApprochées[-1, j] = u(1, t[j])      # A la fin 

    # Résolution avec différences finies
    for n in range(nbrt - 1):
        for i in range(1, nbrx - 1):
            if c >= 0:
                solutionsApprochées[i, n + 1] = (
                    solutionsApprochées[i, n]
                    + ht
                    * (
                        d
                        * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n])
                        / hx**2
                        - c * (solutionsApprochées[i, n] - solutionsApprochées[i - 1, n]) / hx
                        + f(x[i], t[n])
                    )
                )
            else:
                solutionsApprochées[i, n + 1] = (
                    solutionsApprochées[i, n]
                    + ht
                    * (
                        d
                        * (solutionsApprochées[i + 1, n] - 2 * solutionsApprochées[i, n] + solutionsApprochées[i - 1, n])
                        / hx**2
                        - c * (solutionsApprochées[i + 1, n] - solutionsApprochées[i, n]) / hx
                        + f(x[i], t[n])
                    )
                )

    # Calcul de la solution exacte
    for n in range(nbrt):
        for i in range(nbrx):
            solutionExacte[i, n] = u(x[i], t[n])

    # Calcul de l'écart relatif avec la norme l2
    erreur = np.linalg.norm(solutionsApprochées - solutionExacte) / np.linalg.norm(solutionExacte)
    return erreur
"""
    #Permet de tracer l'erreur au cours du temps 
    erreurs = []
    for n in range(nbrt):
        erreur = np.linalg.norm(solutionsApprochées[:, n] - solutionExacte[:, n]) / np.linalg.norm(solutionExacte[:, n])
        erreurs.append(erreur)

    plt.figure(figsize=(10, 6))
    plt.plot(t, erreurs, label="Erreur au cours du temps")
    plt.xlabel("Temps")
    plt.ylabel("Erreur (norme L2)")
    plt.title("Évolution de l'erreur au cours du temps")
    plt.legend()
    plt.grid()
    plt.show()

    return erreur 
    """














ht = 0.5 * (hx**2 / d) * 0.95  
err = []
tracer = []
ht_values=[ht,ht/2,ht/4,ht/8,ht/16]
# Calcul de l'erreur pour différents ajustements
for htBoucle in ht_values:  
    
    tracer.append(htBoucle)
    err.append(simulate_diffusion(htBoucle))

# Conversion en logarithmes
log_tracer = np.log(tracer)
log_err = np.log(err)

# Régression linéaire pour trouver la pente
coeffs = np.polyfit(log_tracer, log_err, 1)
pente = coeffs[0]  # La pente est le premier coefficient

# Affichage des résultats
plt.loglog(tracer, err, marker='o', label=f"Erreur (pente = {pente:.2f})")
plt.xlabel("Pas de temps ht (échelle log)")
plt.ylabel("Erreur relative (norme L2) (échelle log)")
plt.title("Évolution de l'erreur en fonction du pas de temps (log-log)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# Retourner la pente
print(f"La pente de la courbe log-log est : {pente:.2f}")


# Choix d'un ht global
ht_global = 0.5 * (hx**2 / d) * 0.95  # Basé sur la plus petite valeur possible de hx

# Calcul de l'erreur pour différents hx avec ht global
hx_values = [hx,hx*2, hx*4,hx*8,hx*16]  # Plage de valeurs pour hx
err_hx = []
tracer_hx = []

for hxBoucle in hx_values:
    erreur = simulate_diffusion(ht_global, hx=hxBoucle,)  # Appel de la fonction avec ht global
    err_hx.append(erreur)
    tracer_hx.append(hxBoucle)

# Calcul de la pente
log_hx = np.log(tracer_hx)
log_err = np.log(err_hx)

# Ajout de la courbe pour hx au graphique
plt.loglog(tracer_hx, err_hx, marker='+', label="Erreur vs hx (avec ht global)")

# Ajout du titre, de la légende et des annotations
plt.title(f"Évolution de l'erreur en fonction du pas d'espace (log-log)")
plt.xlabel("hx (log scale)")
plt.ylabel("Erreur (log scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
