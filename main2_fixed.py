# main.py – Cosmologie de la cohérence + Protocole Ariane intégré
# Version finale 17 nov 2025 – 21h30 CET
# Lance avec : python main.py --demo

import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio, os, argparse, json, time

# ====================== PARAMÈTRES v5 CORRIGÉS ======================
N = 256; dx = 1.0; dt = 0.015; steps = 8000
alpha = 1.0; a = -0.2; b = 1.0; gamma = 0.8; l = 2.0; sigma = 0.12

# Grille Fourier
x = np.linspace(0, N*dx, N, endpoint=False)
kx = 2*np.pi * np.fft.fftfreq(N, dx); ky = kx.copy()
KX, KY = np.meshgrid(kx, ky); k2 = KX**2 + KY**2
S_k = np.exp(-0.5 * l**2 * k2)

Phi = np.random.randn(N, N) * 0.03

# ====================== MÉTRIQUES DU PROTOCOLE ARIANE ======================
def compute_metrics(Phi, t):
    F = np.fft.fft2(Phi)
    power = np.abs(F)**2 / (N*N)
    k_values = np.sqrt(k2.flatten())
    power_flat = power.flatten()
    
    # C_loc : cohérence locale (moyenne des gradients normalisés)
    grad_x = np.gradient(Phi, axis=0); grad_y = np.gradient(Phi, axis=1)
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    C_loc = 1 / (1 + np.mean(grad_norm / (np.abs(Phi) + 1e-8)))
    
    # C_spec : cohérence spectrale (pic dans la bande instable)
    k_in_band = (k_values > 0.5) & (k_values < 2.5)
    C_spec = np.max(power_flat[k_in_band]) if np.any(k_in_band) else 0
    
    # C_ord : cohérence d’ordre (entropie de von Neumann du spectre)
    p = power_flat / (power_flat.sum() + 1e-12)
    p = p[p > 0]
    C_ord = -np.sum(p * np.log(p + 1e-12))
    C_ord = 1 / (1 + C_ord)  # normalisé entre 0 et 1
    
    # Entropie H et Fisher discret
    H = -np.sum(p * np.log(p + 1e-12))
    # Fisher discret : énergie de gradient (approximation simple mais stable)
    Fisher = np.sum(grad_x**2 + grad_y**2)
    
    return {
        "t": t*dt,
        "C_loc": float(C_loc),
        "C_spec": float(C_spec),
        "C_ord": float(C_ord),
        "H": float(H),
        "Fisher": float(Fisher),
        "max_Phi": float(Phi.max()),
        "min_Phi": float(Phi.min()),
        "badge": "—⟡—" if (C_loc > 0.7 and C_spec > 1e-4 and C_ord > 0.6) else ""
    }

# ====================== DYNAMIQUE v5 CORRIGÉE ======================
def step(Phi):
    F = np.fft.fft2(Phi)
    lap = np.real(np.fft.ifft2(-k2 * F))
    IS_F = F - S_k * F
    magnitude_sq = np.abs(IS_F)**2 / (N*N)
    nonlocal_term = gamma * np.real(np.fft.ifft2(magnitude_sq * F))
    drift = alpha*lap - a*Phi - b*Phi**3 - nonlocal_term
    noise = sigma * np.random.randn(N, N)
    return Phi + dt*drift + np.sqrt(dt)*noise

# ====================== DÉMO + PROTOCOLE ARIANE ======================
def demo():
    os.makedirs("frames", exist_ok=True)
    os.makedirs("ariane_logs", exist_ok=True)
    
    metrics_log = []
    print("Démarrage – Protocole Ariane actif —⟡—")
    
    global Phi
    for t in range(steps):
        Phi = step(Phi)
        
        # Sauvegarde image tous les 300 steps
        if t % 300 == 0 or t == steps-1:
            plt.figure(figsize=(7,7), facecolor='black')
            plt.imshow(Phi, cmap='twilight_shifted', vmin=-2.2, vmax=2.2)
            plt.axis('off')
            plt.savefig(f"frames/{t:05d}.png", dpi=220, bbox_inches='tight')
            plt.close()
        
        # Protocole Ariane : calcul des métriques tous les 100 steps
        if t % 100 == 0:
            metrics = compute_metrics(Phi, t)
            metrics_log.append(metrics)
            badge = metrics["badge"]
            print(f"t={t:4d} | C_loc={metrics['C_loc']:.3f} | C_spec={metrics['C_spec']:.2e} | C_ord={metrics['C_ord']:.3f} {badge}")
    
    # Sauvegarde du log Ariane
    with open("ariane_logs/run_17nov2025.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    
    # GIF final
    imgs = [imageio.imread(f"frames/{t:05d}.png") for t in range(0, steps, 300)]
    imageio.mimsave("vortex_coherence_ariane.gif", imgs, fps=14)
    
    print("\n—⟡— FIN DU PROTOCOLE ARIANE —⟡—")
    print("→ vortex_coherence_ariane.gif : ton vortex d'ouverture vivant")
    print("→ ariane_logs/run_17nov2025.json : toutes les métriques (C_loc, C_spec, C_ord, H, Fisher)")
    print("→ badge —⟡— affiché quand une vraie condensation de cohérence est détectée")

if __name__ == "__main__":
    demo()