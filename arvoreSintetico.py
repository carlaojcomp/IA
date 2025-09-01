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

sintetico = pd.read_csv('sinteticos1.txt')
sintetico.rename(columns={'0.9018017331627175\t5.974004133046255\t1.0': "A"}, inplace=True)
sintetico.loc[sintetico.size] = '0.9018017331627175\t5.974004133046255\t1.0'
sintetico[['A', 'B', 'Classe']] = sintetico['A'].str.split('\t', expand=True)
sintetico['A'] = sintetico['A'].astype('float64')
sintetico['B'] = sintetico['B'].astype('float64')
sintetico['Classe'] = sintetico['Classe'].astype('float64')
a = sintetico['A'].to_numpy()
b = sintetico['B'].to_numpy()
classe = sintetico['Classe'].to_numpy()
limiarA, ganhoA = limiar_otimo(a, classe)
limiarB, ganhoB = limiar_otimo(b, classe)
print(f'Limiar A: {limiarA} Ganho A: {ganhoA}')
print(f'Limiar B: {limiarB} Ganho B: {ganhoB}')
if(ganhoB > ganhoA):
    print(f'A: {ganhoA} < B: {ganhoB}')
    print("Limiar B foi escolhido")
# GanhoB > GanhoA, Limiar B será escolhido

classe1 = sintetico.loc[sintetico['B'] <= limiarB, 'Classe'].to_numpy()
sintetico = sintetico[sintetico['B'] > limiarB] # Remove do dataframe o que já é classe
base2_Classe = sintetico.loc[sintetico['B'] > limiarB, 'Classe'].to_numpy()
base2_A = sintetico.loc[sintetico['B'] > limiarB, 'A'].to_numpy()
base2_B = sintetico.loc[sintetico['B'] > limiarB, 'B'].to_numpy()
# Como os valores menores ou iguais a limiarB são praticamente todos 1 (98% do total), então eles serão uma classe (classe1)
limiar2A, ganho2A = limiar_otimo(base2_A, base2_Classe)
limiar2B, ganho2B = limiar_otimo(base2_B, base2_Classe)
print(f'Limiar 2A: {limiar2A} Ganho 2A: {ganho2A}')
print(f'Limiar 2B: {limiar2B} Ganho 2B: {ganho2B}')
if(ganho2B > ganho2A):
    print(f'A: {ganho2A} < B: {ganho2B}')
    print("Limiar 2B foi escolhido")
# Ganho2B > Ganho2A, Limiar 2B será escolhido
classe2 = sintetico.loc[sintetico['B'] <= limiar2B, 'Classe'].to_numpy()
classe3 = sintetico.loc[sintetico['B'] > limiar2B, 'Classe'].to_numpy()
classe1 = np.concatenate((classe1, classe3), axis=0)
# Como os valores maiores a limiar2B são praticamente todos 1 , então eles serão uma classe (classe3)
# Como classe1 e classe3 possuem predominantemente valores 1s, então eles serão concatenados e dormaraão uma só classe
print(f"Acuracia: {acuracia(classe2, classe1)}")





