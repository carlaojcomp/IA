import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def transpor(X):
    return X.transpose()
def inverter(X):
    det = np.linalg.det(X)
    if det == 0:
        raise ValueError("Matriz não é invertível (determinante zero).")
    return np.linalg.inv(X)
def pesos_iniciais(entrada, saida):
    return np.random.randn(entrada, saida) * (1/np.sqrt(entrada))
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
def converter_para_float(data):
    for coluna in data.columns:
        if data[coluna].dtype == 'object':
            data[coluna] = pd.to_numeric(data[coluna], errors='coerce')
    return data
def acuracia(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho")
    y_true = np.argmax(y_true, axis=1)
    acertos = np.sum(y_true == y_pred)
    acuracia = acertos / len(y_true)
    return acuracia
def gradiente_descendente(X_train, Y_train,X_test , Y_test,alfa, rep):
    m = len(X_train)
    # Adiciona Bias
    X_bias = np.c_[np.ones((m, 1)), X_train]
    m = len(X_bias)
    linhas_X, colunas_X = X_bias.shape
    linhas_Y, colunas_Y = Y_train.shape
    W = pesos_iniciais(colunas_X, colunas_Y)
    historico_acuracia_treino = []
    historico_acuracia_teste = []
    for i in range(rep):
        y_pred = multiplicar(X_bias, W)
        erro = y_pred - Y_train
        gradiente = (1/m) * multiplicar(transpor(X_bias), erro)
        W -= alfa * gradiente
        historico_acuracia_treino.append(acuracia(Y_train, classificar_linear(X_train, W)))
        historico_acuracia_teste.append(acuracia(Y_test, classificar_linear(X_test, W)))
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rep + 1), historico_acuracia_treino, label='Acurácia Treino', marker='o')
    plt.plot(range(1, rep + 1), historico_acuracia_teste, label='Acurácia Teste', marker='s')
    plt.xlabel('Reps')
    plt.ylabel('Acurácia Class Linear (%)')
    plt.title('Evolução da Acurácia por Rep')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return W
def converter_para_float(data):
    for coluna in data.columns:
        if data[coluna].dtype == 'object':
            data[coluna] = pd.to_numeric(data[coluna], errors='coerce')
    return data


def concatenar_dataframes(df1, df2, df3):
    # Concatenar os três DataFrames ao longo das linhas (axis=0)
    df_concatenado = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    # Retornar o DataFrame resultante
    return df_concatenado
def one_hot_encoder(arr_int):
    # Verificar se os valores de arr_int são inteiros positivos
    if np.any(arr_int < 1):
        raise ValueError("Os valores devem ser inteiros positivos.")

    # Encontrar o número máximo de classes
    num_classes = np.max(arr_int)

    # Gerar a codificação one-hot
    return np.eye(num_classes)[arr_int - 1]
def embaralhar(X,Y):
    m, n = X.shape
    indices = np.random.permutation(m)
    return X[indices], Y[indices]

#Função Identidade para a última camada Oculta
def identidade(x):
    return x

#Derivada da função Identidade para a última camada Oculta
def identidade_derivada(x):
    return np.ones_like(x)


#Função para multiplicar duas matrizes
def multiplicar(A, B):
    return np.dot(A, B)

#Função para inverter uma matriz
def inverter(X):
    det = np.linalg.det(X)
    if det == 0:
        raise ValueError("Matriz não é invertível (determinante zero).")
    return np.linalg.inv(X)
#Aplica a softmax após última camada oculta
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Para estabilidade numérica
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#Camada de entrada da Rede
def camada_entrada(X):
    X_normalizado = X.reshape(-1, 9)
    return X_normalizado


#Camada oculta da Rede: adiciona o Bias na base, multiplica pelos pesos e aplica a função de ativação (identidade)
def terceira_camada(A2, W3):
    A2_bias = np.c_[np.ones((A2.shape[0], 1)), A2]
    Z3 = multiplicar(A2_bias, W3)
    A3 = identidade(Z3)
    return A3, Z3, A2_bias
#Gera a predição da rede com base nos pesos de cada camada
def prever(X, W3):
    # Camada de entrada
    X = camada_entrada(X)

    # Terceira camada oculta
    A3, Z3, A2_bias = terceira_camada(X, W3)
    #Softmax
    y_pred = softmax(A3)

    return y_pred


#Calcula a acurácia da predição
def calcular_acuracia(y_pred, y_verdadeiro):
    # Converte para classe prevista (índice do maior valor da saída)
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_verdadeiro, axis=1)

    acuracia = np.mean(pred_classes == true_classes) * 100
    return acuracia

#Treinamento da Rede utilizando o SGD
def treinar_rede(X_treino, Y_treino,X_teste, Y_teste, alfa, epocas):
    m, n = X_treino.shape
    if n != 9:
        raise ValueError(f"Esperado input com {9+1} features, mas recebeu {n}")
    # Inicializa pesos
    W3 = pesos_iniciais(9 + 1, 3)
    #Listas para guardar a acurácia em cada época
    historico_acuracia_treino = []
    historico_acuracia_teste = []
    for epoca in range(epocas):
        #Embaralha os índices
        indices = np.random.permutation(m)
        X_emb = X_treino[indices]
        Y_emb = Y_treino[indices]

        for i in range(m):
            #Redimensionamento da base
            x_i = X_emb[i].reshape(1, -1)
            y_i = Y_emb[i].reshape(1, -1)
            #Passa o conjunto de treinamento pela rede
            X1 = camada_entrada(x_i)
            A1, Z1, X_bias = terceira_camada(X1, W3)

            #Aplica a Softmax após a última camada oculta
            y_pred = softmax(A1)

            #Backpropagation do erro pelas camadas
            erro_saida = y_pred - y_i
            #Pedágio
            grad_W3 = multiplicar(X_bias.T, erro_saida)
            #Pedágio
            W3 -= alfa * grad_W3
        #Calula a acurácia do conjunto de treinamento e de teste após cada época
        y_pred_teste = prever(X_teste, W3)
        acuracia_teste = calcular_acuracia(y_pred_teste, Y_teste)
        y_pred_treino = prever(X_treino, W3)
        acuracia_treino = calcular_acuracia(y_pred_treino, Y_treino)
        #Adiciona as acurácias em uma lista para plotagem
        historico_acuracia_treino.append(acuracia_treino)
        historico_acuracia_teste.append(acuracia_teste)
        #print(f"Acurácia no conjunto de treinamento: {acuracia_treino:.2f}% , EPOCA: {epoca}")
        #print(f"Acurácia no conjunto de teste: {acuracia_teste:.2f}% , EPOCA: {epoca}")
    #Plota a acurácia de cada conjunto de acordo com as épocas
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epocas + 1), historico_acuracia_treino, label='Acurácia Treino', marker='o')
    plt.plot(range(1, epocas + 1), historico_acuracia_teste, label='Acurácia Teste', marker='s')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia MLP (%)')
    plt.title('Evolução da Acurácia por Época')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #Retorna os pesos finais ajustados de cada camada oculta
    return W3


def kfold_separar(X_treino, Y_treino, k):

    # Embaralhar os dados de treino para garantir aleatoriedade
    indices = np.random.permutation(len(X_treino))
    X_treino, Y_treino = X_treino[indices], Y_treino[indices]

    # Determina o tamanho de cada fold
    tamanho = len(X_treino) // k
    pastas = []

    for i in range(k):
        # Índices para o conjunto de validação
        val_indices = range(i * tamanho, (i + 1) * tamanho) if i != k - 1 else range(i * tamanho, len(X_treino))

        # Os dados de treino são os dados que não estão no conjunto de validação
        train_indices = np.setdiff1d(range(len(X_treino)), val_indices)

        # Cria os subconjuntos de treino e validação
        X_train, X_val = X_treino[train_indices], X_treino[val_indices]
        y_train, y_val = Y_treino[train_indices], Y_treino[val_indices]

        pastas.append((X_train, y_train, X_val, y_val))

        # Embaralha os folds antes de retornar
    np.random.shuffle(pastas)
    return pastas


def validar_classificador(X_treino, Y_treino, k):
    pastas = kfold_separar(X_treino, Y_treino, k)
    acuracias_linear_treino = []
    acuracias_linear_teste = []
    acuracias_mlp_treino = []
    acuracias_mlp_teste = []

    # Para cada fold
    for X_train, y_train, X_val, y_val in pastas:
        # Treinar o modelo com gradiente descendente
        pesos = gradiente_descendente(X_train, y_train, X_val, y_val ,0.01, 1000)

        # Classificar usando os pesos obtidos
        y_pred_val = classificar_linear(X_val, pesos)
        y_pred_train = classificar_linear(X_train, pesos)

        # Calcular a acurácia
        acuraccy_val = acuracia(y_val, y_pred_val)
        acuraccy_train = acuracia(y_train, y_pred_train)
        W1_final = treinar_rede(X_train, y_train, X_val, y_val, 0.01, 200)
        # Gera predição após a finalização do treinamento da Rede
        y_pred_teste = prever(X_val, W1_final)
        y_pred_treino = prever(X_train, W1_final)
        # Calcula acurácia da predição
        acuracia_teste = calcular_acuracia(y_pred_teste, y_val)
        acuracia_treino = calcular_acuracia(y_pred_treino, y_train)
        # Printa a acurácia final
        print(f"Acurácia do MLP para este fold (Val): {acuracia_teste:.2f}%")
        print(f"Acurácia do MLP para este fold (Train): {acuracia_treino:.2f}%")
        print(f"Acurácia do Linear para este fold (Val): {acuraccy_val:.2f}")
        print(f"Acurácia do Linear para este fold (Train): {acuraccy_train:.2f}")
        acuracias_linear_teste.append(acuraccy_val)
        acuracias_linear_treino.append(acuraccy_train)
        acuracias_mlp_teste.append(acuracia_teste)
        acuracias_mlp_treino.append(acuracia_treino)

    # Retornar a acurácia média
    print(f"Acurácia média do MLP para os folds (Val): {np.mean(acuracias_mlp_teste):.2f}%")
    print(f"Acurácia média do MLP para os folds (Train): {np.mean(acuracias_mlp_treino):.2f}%")
    print(f"Acurácia média do Linear para os folds (Val): {np.mean(acuracias_linear_teste):.2f}")
    print(f"Acurácia média do Linear para os folds (Train): {np.mean(acuracias_linear_treino):.2f}")
    #Plot a acurácia de cada fold
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, k + 1), acuracias_mlp_treino, label='Acurácia Treino', marker='o')
    plt.plot(range(1, k + 1), acuracias_mlp_teste, label='Acurácia Teste', marker='s')
    plt.xlabel('Pasta')
    plt.ylabel('Acurácia MLP (%)')
    plt.title('Evolução da Acurácia por Pasta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.plot(range(1, k + 1), acuracias_linear_treino, label='Acurácia Treino', marker='o')
    plt.plot(range(1, k + 1), acuracias_linear_teste, label='Acurácia Teste', marker='s')
    plt.xlabel('Pasta')
    plt.ylabel('Acurácia Class Linear (%)')
    plt.title('Evolução da Acurácia por Pasta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




tr1 = pd.read_csv('TreinamentoC1.csv')
tr1.loc[len(tr1)] = tr1.columns
tr1.columns = ['MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135', 'TEMP', 'UMIDADE', 'CLASSE']
tr1 = converter_para_float(tr1)
tr2 = pd.read_csv('TreinamentoC2.csv')
tr2.loc[len(tr2)] = tr2.columns
tr2.columns = ['MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135', 'TEMP', 'UMIDADE', 'CLASSE']
tr2 = converter_para_float(tr2)
tr3 = pd.read_csv('TreinamentoC3.csv')
tr3.loc[len(tr3)] = tr3.columns
tr3.columns = ['MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135', 'TEMP', 'UMIDADE', 'CLASSE']
tr3 = converter_para_float(tr3)
te1 = pd.read_csv('TesteC1.csv')
te1.loc[len(te1)] = te1.columns
te1.columns = ['MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135', 'TEMP', 'UMIDADE', 'CLASSE']
te1 = converter_para_float(te1)
te2 = pd.read_csv('TestesC2.csv')
te2.loc[len(te2)] = te2.columns
te2.columns = ['MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135', 'TEMP', 'UMIDADE', 'CLASSE']
te2 = converter_para_float(te2)
te3 = pd.read_csv('TesteC3.csv')
te3.loc[len(te3)] = te3.columns
te3.columns = ['MQ2', 'MQ3', 'MQ4', 'MQ5', 'MQ6', 'MQ7', 'MQ8', 'MQ9', 'MQ135', 'TEMP', 'UMIDADE', 'CLASSE']
te3 = converter_para_float(te3)
treino = concatenar_dataframes(tr1, tr2, tr3)
teste = concatenar_dataframes(te1, te2, te3)
classe_treino = treino['CLASSE'].to_numpy()
#Dropa os rótulos
treino = treino.drop('CLASSE', axis=1)
treino = treino.drop('TEMP', axis=1)
treino = treino.drop('MQ8', axis=1)
X_treino = treino.to_numpy()
#Aplica a codificação one-hot
Y_treino = one_hot_encoder(classe_treino)
classe_teste = teste['CLASSE'].to_numpy()
#Dropa os rótulos
teste = teste.drop('CLASSE', axis=1)
teste = teste.drop('TEMP', axis=1)
teste = teste.drop('MQ8', axis=1)
X_teste = teste.to_numpy()
#Aplica a codificação one-hot
Y_teste = one_hot_encoder(classe_teste)
X_treino, Y_treino = embaralhar(X_treino, Y_treino)
X_teste, Y_teste = embaralhar(X_teste, Y_teste)
validar_classificador(X_treino,Y_treino,5)
W3_final = treinar_rede(X_treino, Y_treino,X_teste,Y_teste, 0.01, 200)
#Gera predição após a finalização do treinamento da Rede
y_pred_teste = prever(X_teste, W3_final)
y_pred_treino = prever(X_treino, W3_final)
# Calcula acurácia da predição
acuracia_teste = calcular_acuracia(y_pred_teste, Y_teste)
acuracia_treino = calcular_acuracia(y_pred_treino, Y_treino)
#Printa a acurácia final
print(f"Acurácia no conjunto de teste do MLP FINAL: {acuracia_teste:.2f}%")
print(f"Acurácia no conjunto de treinamento do MLP FINAL: {acuracia_treino:.2f}%")
# Treina os pesos, classifica e tira a acurácia dos conjuntos de Treino e de Teste para o Class Linear
pesos = gradiente_descendente(X_treino,Y_treino,X_teste, Y_teste,0.01, 1000)
classificar = classificar_linear(X_teste, pesos)
print(f"Acurácia do modelo Class Linear Teste: {acuracia(Y_teste, classificar)}")
print(f"Acurácia do modelo Class Linear Treino: {acuracia(Y_treino, classificar_linear(X_treino, pesos))}")




