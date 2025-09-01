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
def acuracia(classe1, classe2, classe3):
    zeros = 0
    uns = 0
    dois = 0
    for i in (classe1):
        if (i == 0):
            zeros += 1
    for i in (classe2):
        if (i == 1):
            uns += 1
    for i in (classe3):
        if (i == 2):
            dois += 1
    return (zeros+uns+dois)/(len(classe1) + len(classe2) + len(classe3))

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
limiarA, ganhoA = limiar_otimo(a, classe)
limiarB, ganhoB = limiar_otimo(b, classe)
limiarC, ganhoC = limiar_otimo(c, classe)
limiarD, ganhoD = limiar_otimo(d, classe)
print(f'Limiar A: {limiarA} Ganho A: {ganhoA}')
print(f'Limiar B: {limiarB} Ganho B: {ganhoB}')
print(f'Limiar C: {limiarC} Ganho C: {ganhoC}')
print(f'Limiar D: {limiarD} Ganho D: {ganhoD}')

#GanhoD foi o maior dentre os 4, logo será usado o LimiarD
classe1 = iris.loc[iris['D'] <= limiarD, 'Classe'].to_numpy()
iris = iris[iris['D'] > limiarD] # Remove do dataframe o que já é classe
base2_Classe = iris.loc[iris['D'] > limiarD, 'Classe'].to_numpy()
base2_A = iris.loc[iris['D'] > limiarD, 'A'].to_numpy()
base2_B = iris.loc[iris['D'] > limiarD, 'B'].to_numpy()
base2_C = iris.loc[iris['D'] > limiarD, 'C'].to_numpy()
base2_D = iris.loc[iris['D'] > limiarD, 'D'].to_numpy()

#Como os valores menores ou iguais a limiarD são todos 0, então eles serão uma classe (classe1)
print("Limiar D foi escolhido")

limiar2A, ganho2A = limiar_otimo(base2_A, base2_Classe)
limiar2B, ganho2B = limiar_otimo(base2_B, base2_Classe)
limiar2C, ganho2C = limiar_otimo(base2_C, base2_Classe)
limiar2D, ganho2D = limiar_otimo(base2_D, base2_Classe)
print(f'Limiar 2A: {limiar2A} Ganho 2A: {ganho2A}')
print(f'Limiar 2B: {limiar2B} Ganho 2B: {ganho2B}')
print(f'Limiar 2C: {limiar2C} Ganho 2C: {ganho2C}')
print(f'Limiar 2D: {limiar2D} Ganho 2D: {ganho2D}')

print("Limiar 2D foi escolhido")

classe2 = iris.loc[iris['D'] <= limiar2D, 'Classe'].to_numpy()
classe3 = iris.loc[iris['D'] > limiar2D, 'Classe'].to_numpy()

print(f"Acuracia: {acuracia(classe1,classe2, classe3)}")
