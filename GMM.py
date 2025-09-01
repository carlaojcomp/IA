import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def gerar_pontos(media, covariancia, amostras):
    #Escalonamento (fator de escala baseado na raiz quadrada dos eigenvalores da covariância)

    # Decomposição espectral (autovalores e autovetores)
    autovalores, autovetores = np.linalg.eigh(covariancia)
    autovalores = np.maximum(autovalores, 0)
    #Gerar amostras unitárias com distribuição normal padrão (média 0 e variância 1)
    unitarias = np.random.randn(amostras, len(media))

    #Aplicar a transformação(Rotação e Escalonamento) para obter as amostras com a distribuição desejada
    transformados = unitarias.dot(np.sqrt(np.diag(autovalores))).dot(autovetores.T)

    # Passo 4: Translação (deslocar para o vetor de média)
    final = transformados + media

    return final
def sample_gmm(pesos, medias, covariancias, n_samples):
    n_components = len(pesos)
    samples = []
    # Para cada amostra, sorteia qual componente usar
    for _ in range(n_samples):
        # Sorteia componente baseado nos pesos
        component = np.random.choice(n_components, p=pesos)
        # Gera uma amostra desse componente
        sample = gerar_pontos(medias[component], covariancias[component], 1)[0]
        samples.append(sample)

    return np.array(samples)


def expectation_step(X, pesos, medias, covariancias):
    n_samples, n_features = X.shape
    n_components = len(pesos)

    # Matriz de responsabilidades
    responsibilities = np.zeros((n_samples, n_components))
    # Para cada componente
    for k in range(n_components):
        # Calcular probabilidade para cada componente
        diff = X - medias[k]
        cov_inv = np.linalg.inv(covariancias[k])
        cov_det = np.linalg.det(covariancias[k])

        # Densidade gaussiana multivariada
        mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
        normalization = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
        responsibilities[:, k] = pesos[k] * normalization * np.exp(-0.5 * mahalanobis)

        # Normalizar responsabilidades
    row_sums = responsibilities.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10  # Evitar divisão por zero
    responsibilities = responsibilities / row_sums
    return responsibilities


def maximization_step(X, responsibilities):
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]

    novos_pesos = np.zeros(n_components)
    novas_medias = np.zeros((n_components, n_features))
    novas_covs = []

    for k in range(n_components):
        # Soma efetiva de responsabilidades para componente k
        Nk = np.sum(responsibilities[:, k])

        # Atualiza peso
        novos_pesos[k] = Nk / n_samples

        # Atualiza média
        novas_medias[k] = np.sum(responsibilities[:, k:k + 1] * X, axis=0) / Nk

        # Atualiza covariância
        dif = X - novas_medias[k]
        peso_diff = responsibilities[:, k:k + 1] * dif
        nova_cov = (peso_diff.T @ dif) / Nk

        # Regularização para evitar matriz singular
        nova_cov += 1e-6 * np.eye(n_features)
        novas_covs.append(nova_cov)

    return novos_pesos, novas_medias, novas_covs


def calcular_log_likelihood(X, pesos, medias, covariancias):
    n_samples, n_features = X.shape
    n_components = len(pesos)
    log_likelihood = 0

    for i in range(n_samples):
        likelihood_sample = 0
        for k in range(n_components):
            diff = X[i] - medias[k]
            cov_inv = np.linalg.inv(covariancias[k])
            cov_det = np.linalg.det(covariancias[k])
            mahalanobis = diff @ cov_inv @ diff
            normalizar = 1.0 / np.sqrt((2 * np.pi) ** n_features * cov_det)
            likelihood_sample += pesos[k] * normalizar * np.exp(-0.5 * mahalanobis)

        log_likelihood += np.log(likelihood_sample + 1e-10)

    return log_likelihood


def k_means(X, k, max_iter=200):
    N, D = X.shape
    centros = np.zeros((k, D), dtype=X.dtype)

    # K-means++ para inicialização inteligente dos centros
    centros[0] = X[np.random.randint(N)]
    min_dists_sq = np.full(N, np.inf, dtype=X.dtype)

    for i in range(1, k):
        # Calcular distância do último centro adicionado para todos os pontos
        diff = X - centros[i - 1]  # (N, D)
        new_dists_sq = np.sum(diff ** 2, axis=1)  # (N,)

        # Atualizar distâncias mínimas
        min_dists_sq = np.minimum(min_dists_sq, new_dists_sq)

        # Seleção probabilística otimizada
        if min_dists_sq.sum() == 0:
            idx = np.random.randint(N)
        else:
            idx = np.random.choice(N, p=min_dists_sq / min_dists_sq.sum())
        centros[i] = X[idx]

    # Algoritmo K-means
    for iteracao in range(max_iter):
        # Calcular distâncias de forma otimizada
        x2 = np.sum(X ** 2, axis=1, keepdims=True)  # (N,1)
        c2 = np.sum(centros ** 2, axis=1, keepdims=True).T  # (1,k)
        xc = X @ centros.T  # (N,k)
        d2 = x2 - 2 * xc + c2

        # Atribuir cada ponto ao centro mais próximo
        labels = np.argmin(d2, axis=1)

        # Atualizar centros
        novos_centros = np.empty_like(centros)
        delta = 0.0

        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                novo_centro = X[mask].mean(axis=0)
            else:
                # Se cluster vazio, manter centro atual
                novo_centro = centros[j]

            delta = max(delta, np.linalg.norm(novo_centro - centros[j]))
            novos_centros[j] = novo_centro

        centros = novos_centros

        # Verificar convergência
        if delta < 1e-4:
            break

    return centros, labels
def inicializar_parametros(X, nucleos):
    n_samples, n_features = X.shape
    # Usar K-means para inicializar centros
    medias, labels = k_means(X, nucleos)

    # Calcular pesos baseados na proporção de pontos em cada cluster
    pesos = np.zeros(nucleos)
    covariancias = []
    for k in range(nucleos):
        mask = (labels == k)
        n_points = np.sum(mask)

        if n_points > 0:
            # Peso proporcional ao número de pontos no cluster
            pesos[k] = n_points / n_samples

            # Calcular covariância do cluster
            cluster_points = X[mask]
            if n_points > 1:
                # Covariância empírica do cluster
                diff = cluster_points - medias[k]
                cov = np.dot(diff.T, diff) / n_points

                # Adicionar regularização para evitar singularidade
                cov += 1e-6 * np.eye(n_features)
            else:
                # Se apenas um ponto, usar covariância baseada na variância global
                cov = np.var(X, axis=0).mean() * np.eye(n_features)
        else:
            # Se cluster vazio, usar valores padrão
            pesos[k] = 1.0 / nucleos
            cov = np.var(X, axis=0).mean() * np.eye(n_features)

        covariancias.append(cov)
    pesos = pesos / pesos.sum()
    return pesos, medias, covariancias


def treinar_gmm_em(X, nucleos):
    # Inicializa parâmetros
    pesos, medias, covariancias = inicializar_parametros(X, nucleos)
    log_likelihood_prev = -np.inf

    for iteracao in range(100):
        # Passo E
        responsibilities = expectation_step(X, pesos, medias, covariancias)

        # Passo M
        pesos, medias, covariancias = maximization_step(X, responsibilities)

        # Calcula log-likelihood
        log_likelihood = calcular_log_likelihood(X, pesos, medias, covariancias)
        # Verifica convergência
        if abs(log_likelihood - log_likelihood_prev) < 1e-6:
            break

    return pesos, medias, covariancias

def gerar_pontos_cluster(centro, largura_desejada, altura_desejada, n_pontos):
    # Desvios padrão baseados na extensão desejada (regra dos 3-sigma)
    std_x = largura_desejada / 6
    std_y = altura_desejada / 6
    x_pontos = np.random.normal(centro[0], std_x, n_pontos)
    y_pontos = np.random.normal(centro[1], std_y, n_pontos)
    return x_pontos, y_pontos

def calcular_covariancia_empirica(x_pontos, y_pontos):
    n = len(x_pontos)
    media_x = np.mean(x_pontos)
    media_y = np.mean(y_pontos)
    cov_x = 0.0  # Variância de X
    cov_y = 0.0  # Variância de Y
    cov_xy = 0.0  # Covariância entre X e Y
    for i in range(n):
        deltaX = x_pontos[i] - media_x  # Desvio de X
        deltaY = y_pontos[i] - media_y  # Desvio de Y
        cov_x += deltaX * deltaX
        cov_y += deltaY * deltaY
        cov_xy += deltaX * deltaY
    cov_x /= (n - 1)
    cov_y /= (n - 1)
    cov_xy /= (n - 1)
    matriz_cov = np.array([[cov_x, cov_xy],[cov_xy, cov_y]])
    return matriz_cov
dados = pd.read_csv("GMM_dataset.csv")
dados = dados.T
dados0 = dados[0].to_numpy()
dados1 = dados[1].to_numpy()
plt.subplot(1, 3, 1)
plt.scatter(dados0, dados1)
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.title("Nuvem de pontos")

ground_truth_pesos = np.array([0.25, 0.25, 0.25, 0.25])

# Médias dos clusters (centros das regiões)
medias_clusters = {
    'olho_esq': [1.35, 5.0],
    'olho_dir': [4.0, 4.85],
    'nariz': [2.9, 3.0],
    'boca': [3.0, 1.35]
}
ground_truth_medias = np.array([
    medias_clusters['olho_esq'],
    medias_clusters['olho_dir'],
    medias_clusters['nariz'],
    medias_clusters['boca']
])
n_pontos = 1000
# Definir extensões desejadas para cada cluster
extensoes = {
    'olho_esq': {'largura': 1.0, 'altura': 0.6},  # Horizontal
    'olho_dir': {'largura': 1.0, 'altura': 0.6},  # Horizontal
    'nariz': {'largura': 0.4, 'altura': 1.2},  # Vertical
    'boca': {'largura': 1.2, 'altura': 0.4}  # Horizontal
}

# Gerar pontos para cada cluster
pontos_clusters = {}
for nome, centro in medias_clusters.items():
    ext = extensoes[nome]
    x_pts, y_pts = gerar_pontos_cluster(centro, ext['largura'], ext['altura'], n_pontos)
    pontos_clusters[nome] = (x_pts, y_pts)


# Calcular covariâncias empíricas
ground_truth_covariancias = []
nomes_ordem = ['olho_esq', 'olho_dir', 'nariz', 'boca']

for nome in nomes_ordem:
    x_pts, y_pts = pontos_clusters[nome]
    centro = medias_clusters[nome]
    cov = calcular_covariancia_empirica(x_pts, y_pts)
    ground_truth_covariancias.append(cov)

sinteticas = sample_gmm(ground_truth_pesos, ground_truth_medias, ground_truth_covariancias, 1000)
plt.subplot(1, 3, 2)
plt.scatter(sinteticas[:, 0], sinteticas[:, 1])
plt.title('Amostras GMM Geradas\n(Ground Truth)')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid(True)

recovered_pesos, recovered_medias, recovered_covariancias = treinar_gmm_em(sinteticas, 4)
# Gerar amostras com parâmetros recuperados
recovered_samples = sample_gmm(recovered_pesos, recovered_medias, recovered_covariancias,1000)
plt.subplot(1, 3, 3)
plt.scatter(recovered_samples[:, 0], recovered_samples[:, 1], alpha=0.6, s=15)
plt.title('Amostras com Parâmetros\nRecuperados (EM)')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.grid(True)
plt.tight_layout()
plt.show()
