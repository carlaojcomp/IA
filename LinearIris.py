import pandas as pd
import numpy as np
def one_hot_encoder(arr):
    arr_int = arr.astype(int)
    # Aplica a codificação One Hot usando np.eye
    return np.eye(3)[arr_int]


def concatenar(a, b, c, d):
    n = len(a)
    if not (len(b) == len(c) == len(d) == n):
        raise ValueError("Todos os arrays devem ter o mesmo tamanho")

    matriz = np.column_stack((a, b, c, d))
    return matriz

def transpor(X):
    return X.transpose()
def pesos_iniciais(linhas, colunas):
    return np.random.rand(linhas, colunas)


def multiplicar(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Número de colunas da matriz A deve ser igual ao número de linhas da matriz B.")
    return np.dot(A, B)

def inverter(X):
    det = np.linalg.det(X)
    if det == 0:
        raise ValueError("Matriz não é invertível (determinante zero).")
    return np.linalg.inv(X)

def pseudo_inversa(X, Y):
    m = len(X)
    # Adiciona Bias
    X_bias = np.c_[np.ones((m, 1)), X]
    M1 = inverter(multiplicar(transpor(X_bias), X_bias))
    return multiplicar(multiplicar(M1,transpor(X_bias)), Y)
def classificar_linear(X, W):
    m = len(X)
    X_bias = np.c_[np.ones((m, 1)), X]
    z = np.dot(X_bias, W)
    y_pred = np.argmax(z, axis=1)
    return y_pred

def acuracia(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho")

    acertos = np.sum(y_true == y_pred)
    acuracia = acertos / len(y_true)
    return acuracia
def gradiente_descendente(X, Y, alfa, rep):
    m = len(X)
    # Adiciona Bias
    X_bias = np.c_[np.ones((m, 1)), X]
    m = len(X_bias)
    linhas_X, colunas_X = X_bias.shape
    linhas_Y, colunas_Y = Y.shape
    W = pesos_iniciais(colunas_X, colunas_Y)
    for i in range(rep):
        y_pred = multiplicar(X_bias, W)
        erro = y_pred - Y
        gradiente = (1/m) * multiplicar(transpor(X_bias), erro)
        W -= alfa * gradiente
    return W
def gradiente_descendente_estocastico(X, Y, alfa, rep):
    m = len(X)
    # Adiciona Bias ao modelo
    X_bias = np.c_[np.ones((m, 1)), X]
    m = len(X_bias)
    linhas_X, colunas_X = X_bias.shape
    linhas_Y, colunas_Y = Y.shape
    W = pesos_iniciais(colunas_X, colunas_Y)
    for i in range(rep):
        indices_embaralhados = np.random.permutation(m)
        X_embaralhado = X_bias[indices_embaralhados]
        Y_embaralhado = Y[indices_embaralhados]
        for j in range(m):
            y_pred = multiplicar(X_embaralhado, W)
            erro = y_pred - Y_embaralhado
            gradiente = (1 / m) * multiplicar(transpor(X_embaralhado), erro)
            W -= alfa * gradiente
    return W
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


Y = one_hot_encoder(classe)
X = concatenar(a,b,c,d)
pesos = pseudo_inversa(X, Y)
classificacao = classificar_linear(X,pesos)
print(f"Acurácia do modelo da PseudoInversa: {acuracia(classe, classificacao)}")
pesado = gradiente_descendente(X,Y,0.01, 10000)
classificar = classificar_linear(X, pesado)
print(f"Acurácia do modelo da Descida do Gradiente: {acuracia(classe, classificar)}")
pesinho = gradiente_descendente_estocastico(X,Y,0.001, 1000)
estocastico = classificar_linear(X,pesinho)
print(f"Acurácia do modelo da Descida do Gradiente Estocástico: {acuracia(classe, estocastico)}")