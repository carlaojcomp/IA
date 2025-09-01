import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def one_hot_encoder(arr):
    arr_int = arr.astype(int)
    # Aplica a codificação One Hot usando np.eye
    return np.eye(3)[arr_int]


def substituir_valores(arr):
    # Cria um mapeamento para os valores desejados
    mapa = {5: 0, 6: 1, 7: 2}
    resultado = arr.copy()
    for antigo, novo in mapa.items():
        resultado[resultado == antigo] = novo

    return resultado
def normalizar(array):
    return array / array.max()
def dividir(X,Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    tamanho = int(len(X) * 0.7)
    indices_treino = indices[:tamanho]
    indices_teste = indices[tamanho:]
    return X[indices_treino], Y[indices_treino], X[indices_teste], Y[indices_teste]


def estimar_parametros_verossimilhanca(X, y):
    # Converte One-Hot -> rótulos inteiros (0,1,2)
    if y.ndim == 2:
        labels = np.argmax(y, axis=1)
        n_classes = y.shape[1]
    else:
        # Se já vier como vetor de rótulos
        labels = y
        n_classes = len(np.unique(labels))
    classes = np.arange(n_classes)
    n_features = X.shape[1]
    medias = {}
    covariancias = {}
    priors = {}
    for c in classes:
        X_c = X[labels == c]
        n_c = X_c.shape[0]
        # Prior uniforme
        priors[c] = 1.0 / n_classes
        # Média da classe
        mu = X_c.mean(axis=0)
        medias[c] = mu
        # Covariância MLE (divisão por N_c)
        diferenca = X_c - mu
        sigma = (diferenca.T @ diferenca) / n_c
        # Regularização
        sigma += np.eye(n_features) * 1e-6
        covariancias[c] = sigma
    return {
        'classes': classes,
        'medias': medias,
        'covariancias': covariancias,
        'priors': priors
    }
def gerar_pontos(media, covariancia, x):
    jitter = 1e-9
    x = np.atleast_2d(np.asarray(x, dtype=float))
    mu = np.asarray(media, dtype=float).reshape(-1)
    sigma = np.asarray(covariancia, dtype=float)
    dimensoes = mu.shape[0]
    # Garantir simetria e adicionar jitter para estabilidade numérica
    sigma = 0.5 * (sigma + sigma.T) + jitter * np.eye(dimensoes)
    # Inversa e log-determinante
    inv_cov = np.linalg.inv(sigma)
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise ValueError("Matriz de covariância não é positiva definida.")
    diferenca = x - mu
    # Quadrático: (x - μ)^T Σ^{-1} (x - μ)
    quadratico = np.einsum('ni,ij,nj->n', diferenca, inv_cov, diferenca)
    # Constante normalizadora
    const = dimensoes * np.log(2 * np.pi)
    logpdf = -0.5 * (quadratico + logdet + const)
    return logpdf

def calcular_log(X, parametros):
    X = np.atleast_2d(X)
    classes = parametros['classes']
    N = X.shape[0]
    K = len(classes)
    log = np.empty((N, K), dtype=float)
    for j, c in enumerate(classes):
        log[:, j] = gerar_pontos(parametros['medias'][c],
                                      parametros['covariancias'][c],
                                      X)
    return log

def predicao(X, parametros):
    score = calcular_log(X, parametros)
    return np.argmax(score, axis=1)

def calcular_acuracia(y_pred, y_verdadeiro):
    if y_pred.ndim > 1:
        pred_classes = np.argmax(y_pred, axis=1)
    else:
        pred_classes = y_pred

    if y_verdadeiro.ndim > 1:
        true_classes = np.argmax(y_verdadeiro, axis=1)
    else:
        true_classes = y_verdadeiro
    acuracia = np.mean(pred_classes == true_classes) * 100
    return acuracia
base = pd.read_csv('WineQT.csv')
# Reduz as classes para as mais frequentes na base (5,6 e 7)
base = base[base['quality'].isin([5, 6, 7])]
base = base.drop('Id', axis=1)
# Troca os rótulos (5,6 e 7) por (0,1 e 2) respectivamente
classe = base['quality'].to_numpy()
classe = substituir_valores(classe)
base = base.drop('quality', axis=1)
X = base.to_numpy()
# Aplica o feature scaling, codificação One-Hot e divide a base em 70% e 30%
X = normalizar(X)
Y = one_hot_encoder(classe)
X_treino, Y_treino, X_teste, Y_teste = dividir(X,Y)
# Estima os parâmetros, faz a predição e calcula a acurácia
parametros = estimar_parametros_verossimilhanca(X_treino, Y_treino)
y_pred = predicao(X_teste, parametros)
acuracia = calcular_acuracia(y_pred, Y_teste)
print(f"Acurácia do Classificador Bayesiano para dados de Teste: {acuracia:.2f}%")
y_pred = predicao(X_treino, parametros)
acuracia = calcular_acuracia(y_pred, Y_treino)
print(f"Acurácia do Classificador Bayesiano para dados de Treino: {acuracia:.2f}%")



