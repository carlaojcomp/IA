import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')


def dividir_indices(n):
    cont = np.random.default_rng(42)
    index = np.arange(n)
    cont.shuffle(index)
    t = int(n * 0.7)
    return index[:t], index[t:]

def zscore_fit(X):
    media = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return media, std

def zscore_transform(X, media, std):
    return (X - media) / std


def estimar_parametros_Naive_Bayes(X_cont, X_cat, y, cat_cardinalidades):
    y = np.asarray(y, dtype=int).ravel()
    classes = np.unique(y)
    # Priors por frequência
    priors = {}
    for c in classes:
        priors[c] = np.mean(y == c)

    # Gaussiana(independente por feature): média e variância por classe
    gauss_media = {}
    gauss_var = {}
    cat_logprob = {}
    for c in classes:
        mask = (y == c)
        Xc_cont = X_cont[mask]
        Xc_cat = X_cat[mask]
        mu = Xc_cont.mean(axis=0)
        var = Xc_cont.var(axis=0, ddof=0)
        var[var <= 1e-12] = 1e-12  # estabilidade
        gauss_media[c] = mu
        gauss_var[c] = var
        cat_logprob[c] = []
        for j, Kj in enumerate(cat_cardinalidades):
            counts = np.bincount(Xc_cat[:, j], minlength=Kj).astype(float)
            probs = (counts + 1) / (counts.sum() + 1 * Kj)
            cat_logprob[c].append(np.log(probs))

    return {
        "classes": classes,
        "priors": priors,
        "gauss_media": gauss_media,
        "gauss_var": gauss_var,
        "cat_logprob": cat_logprob,
        "cat_cardinalidades": cat_cardinalidades
    }

def calcular_log(x, media, variancia):
    const = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variancia)
    quad = -0.5 * ((x - media) ** 2) / variancia
    return np.sum(const + quad)

def predicao(X_cont, X_cat, parametros):
    classes = parametros["classes"]
    priors = parametros["priors"]
    mu_dict = parametros["gauss_media"]
    var_dict = parametros["gauss_var"]
    cat_lp  = parametros["cat_logprob"]
    Kj_list = parametros["cat_cardinalidades"]
    N = 0
    N = max(N, X_cont.shape[0])
    N = max(N, X_cat.shape[0])

    y_pred = np.empty(N, dtype=int)
    for i in range(N):
        melhor_c, melhor_score = None, -np.inf
        for c in classes:
            score = np.log(priors[c] + 1e-15)
            # parte Gaussiana
            score += calcular_log(X_cont[i], mu_dict[c], var_dict[c])
            # parte Categórica
            for j, Kj in enumerate(Kj_list):
                v = int(X_cat[i, j])
                if 0 <= v < Kj:
                    score += cat_lp[c][j][v]
                else:
                    # penaliza valor fora do domínio
                    score += -1e9
            if score > melhor_score:
                melhor_score = score
                melhor_c = c
        y_pred[i] = melhor_c
    return y_pred

def calcular_acuracia(y_pred, y_true):
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    return (y_pred == y_true).mean() * 100.0

base = pd.read_csv("stroke_prediction.csv")
# Substitui os NaNs de bmi pela mediana
base["bmi"] = base["bmi"].fillna(base["bmi"].median())

# Define a classe e remove ID
y = base["stroke"].to_numpy(dtype=int)
base = base.drop(columns=["id", "stroke"])

# Define quais são contínuas e categóricas
cont_cols = ["age", "avg_glucose_level", "bmi"]
cat_cols = [c for c in base.columns if c not in cont_cols]

# Garante que categóricas virem códigos 0..K-1
X_cat_df = base[cat_cols].copy()
for c in X_cat_df.columns:
    X_cat_df[c] = X_cat_df[c].astype("category").cat.codes

# Extrai arrays
X_cont = base[cont_cols].to_numpy(dtype=float)
X_cat = X_cat_df.to_numpy(dtype=int)

# Cardinalidades por coluna categórica
cat_cardinalidades = []
for c in X_cat_df.columns:
    Kj = int(X_cat_df[c].max() + 1) if len(X_cat_df) else 0
    cat_cardinalidades.append(Kj)

# Divide a Base
index_treino, index_teste = dividir_indices(len(base))
Xc_treino, Xc_teste = X_cont[index_treino], X_cont[index_teste]
Xa_treino, Xa_teste = X_cat[index_treino],  X_cat[index_teste]
y_treino, y_teste = y[index_treino], y[index_teste]

# Padroniza as contínuas
mu_c, sd_c = zscore_fit(Xc_treino)
Xc_treino = zscore_transform(Xc_treino, mu_c, sd_c)
Xc_teste = zscore_transform(Xc_teste, mu_c, sd_c)

# Estima parâmetros e faz a predição
parametros = estimar_parametros_Naive_Bayes(Xc_treino, Xa_treino, y_treino, cat_cardinalidades)
y_pred = predicao(Xc_teste, Xa_teste, parametros)
acuracia = calcular_acuracia(y_pred, y_teste)
print(f"Acurácia (Naive Bayes teste: Gaussiano e Contagem): {acuracia:.2f}%")
y_pred = predicao(Xc_treino, Xa_treino, parametros)
acuracia = calcular_acuracia(y_pred, y_treino)
print(f"Acurácia (Naive Bayes treino: Gaussiano e Contagem): {acuracia:.2f}%")

