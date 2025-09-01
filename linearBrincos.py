import numpy as np
import pandas as pd
def one_hot_encoder(arr):
    arr_int = arr.astype(int)
    # Aplica a codificação One Hot usando np.eye
    return np.eye(2)[arr_int]


def concatenar(a, b):
    n = len(a)
    if not (len(b) == n):
        raise ValueError("Todos os arrays devem ter o mesmo tamanho")

    matriz = np.column_stack((a, b))
    return matriz

def transpor(X):
    return X.transpose()



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


def nao_linear_expoente(X, coluna, expoente):
    coluna_quadrada = np.power(X[:, coluna], expoente)
    X = np.column_stack((X, coluna_quadrada))
    return X


def nao_linear_produto(X, coluna1, coluna2):
    produto = X[:, coluna1] * X[:, coluna2]
    X = np.column_stack((X, produto))
    return X

def nao_linear_exponencial(X, coluna):
    coluna_exponencial = np.exp(X[:, coluna])
    X = np.column_stack((X, coluna_exponencial))
    return X
def acuracia(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho")

    acertos = np.sum(y_true == y_pred)
    acuracia = acertos / len(y_true)
    return acuracia

brincos = pd.read_csv('brincos.txt')
brincos.rename(columns={'8.091906 -1.730349 1.000000 ': "A"}, inplace=True)
brincos.loc[brincos.size] = '  8.091906 -1.730349 1.000000  '
brincos = brincos['A'].str.split(expand=True)
brincos = brincos.rename(columns={
    0: 'A',
    1: 'B',
    2: 'Classe'
})

brincos['A'] = brincos['A'].astype('float64')
brincos['B'] = brincos['B'].astype('float64')
brincos['Classe'] = brincos['Classe'].astype('float64')
brincos.loc[brincos['Classe'] == -1.0, 'Classe'] = 0
a = brincos['A'].to_numpy()
b = brincos['B'].to_numpy()
classe = brincos['Classe'].to_numpy()
X = concatenar(a,b)
Y = one_hot_encoder(classe)
pesos = pseudo_inversa(X, Y)
classificacao = classificar_linear(X,pesos)
print(f"Acurácia do modelo da PseudoInversa sem não-linearidade: {acuracia(classe, classificacao)}")
X1 = nao_linear_exponencial(X, 1)
pesos1 = pseudo_inversa(X1, Y)
classificacao1 = classificar_linear(X1, pesos1)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (exponencial): {acuracia(classe, classificacao1)}")
X2 = nao_linear_exponencial(X1, 0)
pesos2 = pseudo_inversa(X2, Y)
classificacao2 = classificar_linear(X2, pesos2)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (exponencial): {acuracia(classe, classificacao2)}")
X3 = nao_linear_expoente(X2, 1,3)
pesos3 = pseudo_inversa(X3, Y)
classificacao3 = classificar_linear(X3, pesos3)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (expoente): {acuracia(classe, classificacao3)}")
X4 = nao_linear_produto(X3, 2,3)
pesos4 = pseudo_inversa(X3, Y)
classificacao4 = classificar_linear(X3, pesos4)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (produto): {acuracia(classe, classificacao4)}")
X5 = nao_linear_expoente(X4, 4,5)
pesos5 = pseudo_inversa(X5, Y)
classificacao5 = classificar_linear(X5, pesos5)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (expoente): {acuracia(classe, classificacao5)}")
X6 = nao_linear_expoente(X5, 1,5)
pesos6 = pseudo_inversa(X6, Y)
classificacao6 = classificar_linear(X6, pesos6)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (expoente): {acuracia(classe, classificacao6)}")
X7 = nao_linear_produto(X6, 4,1)
pesos7 = pseudo_inversa(X7, Y)
classificacao7 = classificar_linear(X7, pesos7)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (produto): {acuracia(classe, classificacao7)}")
X8 = nao_linear_produto(X7,3,5)
pesos8 = pseudo_inversa(X8, Y)
classificacao8 = classificar_linear(X8, pesos8)
print(f"Acurácia do modelo da PseudoInversa com não-linearidade (produto): {acuracia(classe, classificacao8)}")






