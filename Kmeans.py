from PIL import Image
import numpy as np


def k_means(data, k):
    H, W, _ = data.shape
    X = data.reshape(-1, 3).astype(np.float32)
    N, D = X.shape
    # Inicialização dos centros
    centros = np.zeros((k, D), dtype=X.dtype)
    # Primeiro centro aleatório
    centros[0] = X[np.random.randint(N)]
    min_dists_sq = np.full(N, np.inf, dtype=X.dtype)
    for i in range(1, k):
        # Calcular distância do último centro adicionado para todos os pontos
        diff = X - centros[i - 1]  # (N, 3)
        new_dists_sq = np.sum(diff ** 2, axis=1)  # (N,)
        # Atualizar distâncias mínimas
        min_dists_sq = np.minimum(min_dists_sq, new_dists_sq)
        # Seleção probabilística
        if min_dists_sq.sum() == 0:
            idx = np.random.randint(N)
        else:
            idx = np.random.choice(N, p=min_dists_sq / min_dists_sq.sum())
        centros[i] = X[idx]
    for iteracao in range(100):
        x2 = np.sum(X ** 2, axis=1, keepdims=True)
        c2 = np.sum(centros ** 2, axis=1, keepdims=True).T
        xc = X @ centros.T
        d2 = x2 - 2 * xc + c2
        labels = np.argmin(d2, axis=1)
        novos = np.empty_like(centros)
        delta = 0.0
        for j in range(k):
            m = (labels == j)
            c = X[m].mean(axis=0)
            delta = max(delta, np.linalg.norm(c - centros[j]))
            novos[j] = c
        centros = novos
        # Critério de parada
        if delta < 1e-4:
            break
    centros_u8 = np.clip(np.rint(centros), 0, 255).astype(np.uint8)  # (k,3)
    quant = centros_u8[labels].reshape(H, W, 3)  # (H,W,3)
    labels_hw = labels.reshape(H, W)
    mse = np.mean((data - quant) ** 2)
    psnr = 10*np.log10(255**2/mse)
    print(f"Relação Sinal-Ruído com K = {k}: {psnr:.3f}")
    return quant, labels_hw, centros_u8
img = Image.open("imagem_quantizar.webp")
dados = np.array(img)
clusters = [4, 8, 16, 32, 64]
for j in clusters:
    quant, labels, centros = k_means(dados, j)
    Image.fromarray(quant, mode="RGB").save(
        "imagem_quant" + str(j) + "bits.webp",
        format="WEBP",
        lossless=True,
        method=6
    )

