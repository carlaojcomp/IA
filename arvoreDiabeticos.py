import pandas as pd
import numpy as np


def H(arr):
    arr = np.array(arr)
    valores, contagens = np.unique(arr, return_counts=True)
    probs = contagens / contagens.sum()
    entropia = -np.sum(probs * np.log2(probs))
    return entropia



def ganho_informacao(X, y, limiar):
    e = X <= limiar
    d = X > limiar

    H_total = H(y)

    He = H(y[e])
    Hd = H(y[d])

    pe = np.sum(e) / len(X)
    pd = np.sum(d) / len(X)

    Imp = H_total - (pe * He + pd * Hd)
    return Imp


def limiar_otimo(x, y):
    valores = np.unique(x)
    valores.sort()
    limiares = (x[:-1] + x[1:]) / 2

    ganhos = []
    for lim in limiares:
        ganhos.append(ganho_informacao(x, y, lim))

    index_max = np.argmax(ganhos)
    return limiares[index_max], ganhos[index_max]
def acuracia(classe1, classe2):
    zeros = 0
    uns = 0
    for i in (classe1):
        if (i == 0):
            zeros += 1
    for i in (classe2):
        if (i == 1):
            uns += 1
    return (zeros+uns)/(len(classe1) + len(classe2))


glicemia = pd.read_csv('diabeticos.txt')
glicemia.rename(columns={"148\t1": "glicemia"}, inplace=True)
glicemia.loc[glicemia.size] = "148\t1"
glicemia[['glicemia', 'diabetico']] = glicemia['glicemia'].str.split('\t', expand=True)
glicemia['glicemia'] = glicemia['glicemia'].astype('int64')
glicemia['diabetico'] = glicemia['diabetico'].astype('int64')
ordenado = glicemia.sort_values(by='glicemia', ascending=True)
x = ordenado['glicemia'].to_numpy()
y = ordenado['diabetico'].to_numpy()

limiar, ganho = limiar_otimo(x, y)
print(f'Limiar: {limiar} Ganho: {ganho}')

#Limiar Ótimo com um nó: 127

classe1 = glicemia.loc[glicemia['glicemia'] <= limiar, 'diabetico'].to_numpy()
classe2 = glicemia.loc[glicemia['glicemia'] > limiar, 'diabetico'].to_numpy()

print(f"Acuracia: {acuracia(classe1, classe2)}")


