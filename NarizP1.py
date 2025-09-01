import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
# Função para codificação one-hot
def one_hot_encoder(arr):
    arr_int = arr.astype(int)
    # Aplica a codificação One Hot usando np.eye
    return np.eye(10)[arr_int]
def concatenar_dataframes(df1, df2, df3):
    # Concatenar os três DataFrames ao longo das linhas (axis=0)
    df_concatenado = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    # Retornar o DataFrame resultante
    return df_concatenado
#Função sigmoide para camadas as duas primeiras camadas ocultas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Derivada da função sigmoide para camadas as duas primeiras camadas ocultas
def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1 - s)
#Função Identidade para a última camada Oculta
def identidade(x):
    return x

#Derivada da função Identidade para a última camada Oculta
def identidade_derivada(x):
    return np.ones_like(x)

#Função para converter os dados da base para float (real)
def converter_para_float(data):
    for coluna in data.columns:
        if data[coluna].dtype == 'object':
            data[coluna] = pd.to_numeric(data[coluna], errors='coerce')
    return data

#Função para normalizar os dados
def normalizar(array):
    return array / 255.0
#Função para inicialização dos pesos com normalização pelo número de neurônios de entrada
def pesos_iniciais(entrada, saida):
    return np.random.randn(entrada, saida) * (1/np.sqrt(entrada))

#Função para multiplicar duas matrizes
def multiplicar(A, B):
    return np.dot(A, B)

#Função para abaralhar as features e os rótulos e dividí-los em treino e testes na proporcção 70% e 30% respectivamente
def dividir(X,Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    tamanho = int(len(X) * 0.7)

    indices_treino = indices[:tamanho]
    indices_teste = indices[tamanho:]

    return X[indices_treino], Y[indices_treino], X[indices_teste], Y[indices_teste]

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
#Primeira camada oculta da Rede: adiciona o Bias na base, multiplica pelos pesos e aplica a função de ativação(Sigmoide)
def primeira_camada(X, W1):
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    Z1 = multiplicar(X_bias, W1)
    A1 = sigmoid(Z1)
    return A1, Z1, X_bias

#Segunda camada oculta da Rede: adiciona o Bias na base, multiplica pelos pesos e aplica a função de ativação(Sigmoide)
def segunda_camada(A1, W2):
    A1_bias = np.c_[np.ones((A1.shape[0], 1)), A1]
    Z2 = multiplicar(A1_bias, W2)
    A2 = sigmoid(Z2)
    return A2, Z2, A1_bias

#Terceira camada oculta da Rede: adiciona o Bias na base, multiplica pelos pesos e aplica a função de ativação (identidade)
def terceira_camada(A2, W3):
    A2_bias = np.c_[np.ones((A2.shape[0], 1)), A2]
    Z3 = multiplicar(A2_bias, W3)
    A3 = identidade(Z3)
    return A3, Z3, A2_bias
#Gera a predição da rede com base nos pesos de cada camada
def prever(X, W1, W2, W3):
    # Camada de entrada
    X = camada_entrada(X)

    # Primeira camada oculta
    A1, _, A1_bias = primeira_camada(X, W1)

    # Segunda camada oculta
    A2, _, A2_bias = segunda_camada(A1, W2)

    # Terceira camada oculta
    A3, Z3, A2_bias = terceira_camada(A2, W3)
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
        raise ValueError(f"Esperado input com {784} features, mas recebeu {n}")

    # Inicializa pesos
    W1 = pesos_iniciais(9 + 1, 5)
    W2 = pesos_iniciais(5 + 1, 3)
    W3 = pesos_iniciais(3 + 1, 10)
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
            A1, Z1, X_bias = primeira_camada(X1, W1)
            A2, Z2, A1_bias = segunda_camada(A1, W2)
            A3, Z3, A2_bias = terceira_camada(A2, W3)
            #Aplica a Softmax após a última camada oculta
            y_pred = softmax(A3)

            #Backpropagation do erro pelas camadas
            erro_saida = y_pred - y_i
            #Pedágio
            grad_W3 = multiplicar(A2_bias.T, erro_saida)
            delta2 = multiplicar(erro_saida, W3[1:].T) * sigmoid_derivada(Z2)
            #Pedágio
            grad_W2 = multiplicar(A1_bias.T, delta2)
            delta1 = multiplicar(delta2, W2[1:].T) * sigmoid_derivada(Z1)
            # Pedágio
            grad_W1 = multiplicar(X_bias.T, delta1)

            #Ajusta os pesos com base no erro retropropagado
            W3 -= alfa * grad_W3
            W2 -= alfa * grad_W2
            W1 -= alfa * grad_W1
        #Calula a acurácia do conjunto de treinamento e de teste após cada época
        y_pred_teste = prever(X_teste, W1, W2, W3)
        acuracia_teste = calcular_acuracia(y_pred_teste, Y_teste)
        y_pred_treino = prever(X_treino, W1, W2, W3)
        acuracia_treino = calcular_acuracia(y_pred_treino, Y_treino)
        #Adiciona as acurácias em uma lista para plotagem
        historico_acuracia_treino.append(acuracia_treino)
        historico_acuracia_teste.append(acuracia_teste)
        print(f"Acurácia no conjunto de treinamento: {acuracia_treino:.2f}% , EPOCA: {epoca}")
        print(f"Acurácia no conjunto de teste: {acuracia_teste:.2f}% , EPOCA: {epoca}")
    #Plota a acurácia de cada conjunto de acordo com as épocas
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epocas + 1), historico_acuracia_treino, label='Acurácia Treino', marker='o')
    plt.plot(range(1, epocas + 1), historico_acuracia_teste, label='Acurácia Teste', marker='s')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia (%)')
    plt.title('Evolução da Acurácia por Época')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #Retorna os pesos finais ajustados de cada camada oculta
    return W1, W2, W3




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
W1_final, W2_final, W3_final = treinar_rede(X_treino, Y_treino,X_teste,Y_teste, 0.01, 100)
#Gera predição após a finalização do treinamento da Rede
y_pred_teste = prever(X_teste, W1_final, W2_final, W3_final)
y_pred_treino = prever(X_treino,W1_final, W2_final, W3_final)
# Calcula acurácia da predição
acuracia_teste = calcular_acuracia(y_pred_teste, Y_teste)
acuracia_treino = calcular_acuracia(y_pred_treino, Y_treino)
#Printa a acurácia final
print(f"Acurácia no conjunto de teste FINAL: {acuracia_teste:.2f}%")
print(f"Acurácia no conjunto de treinamento FINAL: {acuracia_treino:.2f}%")


