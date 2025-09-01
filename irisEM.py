import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

def concatenar(a, b, c, d):
    n = len(a)
    if not (len(b) == len(c) == len(d) == n):
        raise ValueError("Todos os arrays devem ter o mesmo tamanho")

    matriz = np.column_stack((a, b, c, d))
    return matriz

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

    for iteracao in range(200):
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


def atribuir_clusters(X, pesos, medias, covariancias):
    # Calcular probabilidades a posteriori usando o passo E
    probabilidades = expectation_step(X, pesos, medias, covariancias)
    # Atribuir cada amostra ao cluster com maior probabilidade (MAP)
    labels = np.argmax(probabilidades, axis=1)
    return labels


def plot_histograma(arr, i):
    # Criando subgráficos (1 linha, 3 colunas)
    plt.subplot(1, 3, i + 1)
    # Plotando o histograma com numpy
    plt.hist(arr, bins=np.arange(min(arr), max(arr) + 2), edgecolor='black', align='mid')
    # Adicionando título e rótulos aos eixos
    plt.title(f"Histograma do Cluster {i}")
    plt.xlabel("Valores")
    plt.ylabel("Frequência")
    plt.grid(True)


def gerar_histogramas(labels, Y):
    # Filtrando os clusters de acordo com as labels
    cluster0 = Y[labels == 0]
    cluster1 = Y[labels == 1]
    cluster2 = Y[labels == 2]
    print(f"Classes presentes no Cluster 0: {cluster0}")
    valores, contagens = np.unique(cluster0, return_counts=True)
    frequencia = dict(zip(valores, contagens))
    print(f"Classe e Frequencia das classes no Cluster 0: {frequencia}")
    print(f"Classes presentes no Cluster 1: {cluster1}")
    valores, contagens = np.unique(cluster1, return_counts=True)
    frequencia = dict(zip(valores, contagens))
    print(f"Classe e Frequencia das classes no Cluster 1: {frequencia}")
    print(f"Classes presentes no Cluster 2: {cluster2}")
    valores, contagens = np.unique(cluster2, return_counts=True)
    frequencia = dict(zip(valores, contagens))
    print(f"Classe e Frequencia das classes no Cluster 2: {frequencia}")

    # Gerando os histogramas para cada cluster
    plot_histograma(cluster0, 0)
    plot_histograma(cluster1, 1)
    plot_histograma(cluster2, 2)

    # Ajustando layout para evitar sobreposição
    plt.tight_layout()
    # Exibindo o gráfico
    plt.show()



iris =  pd.read_csv('iris.txt')
iris.rename(columns={'   5.1000000e+00   3.5000000e+00   1.4000000e+00   2.0000000e-01   0.0000000e+00': "A"}, inplace=True)
iris.loc[iris.size] = '   5.1000000e+00   3.5000000e+00   1.4000000e+00   2.0000000e-01   0.0000000e+00'
iris = iris['A'].str.split(expand=True)
iris = iris.rename(columns={
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'Classe'
})
iris['A'] = iris['A'].astype('float64')
iris['B'] = iris['B'].astype('float64')
iris['C'] = iris['C'].astype('float64')
iris['D'] = iris['D'].astype('float64')
iris['Classe'] = iris['Classe'].astype('float64')

a = iris['A'].to_numpy()
b = iris['B'].to_numpy()
c = iris['C'].to_numpy()
d = iris['D'].to_numpy()
classe = iris['Classe'].to_numpy()
X = concatenar(a,b,c,d)
pesos, medias, covariancias = treinar_gmm_em(X, 3)
labels = atribuir_clusters(X, pesos, medias, covariancias)
gerar_histogramas(labels,classe)

