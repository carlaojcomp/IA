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
    X_normalizado = X.reshape(-1, 784)
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

    if n != 784:
        raise ValueError(f"Esperado input com {784} features, mas recebeu {n}")

    # Inicializa pesos
    W1 = pesos_iniciais(784 + 1, 120)
    W2 = pesos_iniciais(120 + 1, 84)
    W3 = pesos_iniciais(84 + 1, 10)
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
            A1, Z1, X_bias = primeira_camada(X1, W1)  # A1: (1,120)
            A2, Z2, A1_bias = segunda_camada(A1, W2)  # A2: (1,84)
            A3, Z3, A2_bias = terceira_camada(A2, W3)  # A3: (1,10)
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


#Carregamento, tratamento e formatação da Base Mnist
mnist = pd.read_csv('mnist_3000.txt')
mnist.rename(columns={'0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t60.0\t141.0\t255.0\t178.0\t141.0\t141.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t63.0\t234.0\t252.0\t253.0\t252.0\t252.0\t252.0\t207.0\t56.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t169.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t253.0\t215.0\t81.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t13.0\t206.0\t252.0\t252.0\t253.0\t227.0\t190.0\t139.0\t253.0\t252.0\t243.0\t125.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t154.0\t253.0\t244.0\t175.0\t176.0\t125.0\t0.0\t0.0\t176.0\t244.0\t253.0\t253.0\t63.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t253.0\t252.0\t142.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t82.0\t240.0\t252.0\t194.0\t19.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t89.0\t253.0\t233.0\t37.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t110.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t213.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t10.0\t178.0\t253.0\t133.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t253.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t197.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t65.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t147.0\t253.0\t226.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t122.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t138.0\t209.0\t28.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t66.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t67.0\t246.0\t76.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t26.0\t159.0\t202.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t76.0\t250.0\t254.0\t228.0\t44.0\t0.0\t0.0\t0.0\t0.0\t13.0\t41.0\t216.0\t253.0\t78.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t200.0\t253.0\t252.0\t224.0\t69.0\t57.0\t57.0\t144.0\t206.0\t253.0\t252.0\t170.0\t9.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t25.0\t216.0\t252.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t206.0\t93.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t28.0\t139.0\t228.0\t252.0\t253.0\t252.0\t214.0\t139.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0': "A"}, inplace=True)
mnist.loc[mnist.size] = '0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t60.0\t141.0\t255.0\t178.0\t141.0\t141.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t63.0\t234.0\t252.0\t253.0\t252.0\t252.0\t252.0\t207.0\t56.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t169.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t253.0\t215.0\t81.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t13.0\t206.0\t252.0\t252.0\t253.0\t227.0\t190.0\t139.0\t253.0\t252.0\t243.0\t125.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t154.0\t253.0\t244.0\t175.0\t176.0\t125.0\t0.0\t0.0\t176.0\t244.0\t253.0\t253.0\t63.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t253.0\t252.0\t142.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t82.0\t240.0\t252.0\t194.0\t19.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t89.0\t253.0\t233.0\t37.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t110.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t213.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t10.0\t178.0\t253.0\t133.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t253.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t197.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t65.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t147.0\t253.0\t226.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t122.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t138.0\t209.0\t28.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t66.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t67.0\t246.0\t76.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t26.0\t159.0\t202.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t76.0\t250.0\t254.0\t228.0\t44.0\t0.0\t0.0\t0.0\t0.0\t13.0\t41.0\t216.0\t253.0\t78.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t200.0\t253.0\t252.0\t224.0\t69.0\t57.0\t57.0\t144.0\t206.0\t253.0\t252.0\t170.0\t9.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t25.0\t216.0\t252.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t206.0\t93.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t28.0\t139.0\t228.0\t252.0\t253.0\t252.0\t214.0\t139.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0'
mnist = mnist['A'].str.split(expand=True)
mnist = converter_para_float(mnist)
classe = mnist[0].to_numpy()
#Dropa os rótulos
mnist = mnist.drop(0, axis=1)
X = mnist.to_numpy()
#Aplica a codificação one-hot
Y = one_hot_encoder(classe)
#Normalização das features
X = normalizar(X)
#Divisão dos conjuntos em treino e teste
X_treino,Y_treino, X_teste, Y_teste = dividir(X,Y)
#Treinamento da Rede
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


#Primeira Execução
'''Acurácia no conjunto de treinamento FINAL: 100.00%
Acurácia no conjunto de teste FINAL: 90.67%'''



#Segunda Execução
'''Acurácia no conjunto de treinamento FINAL: 100.00%
Acurácia no conjunto de teste FINAL: 88.67%
'''


#Terceira Execução
'''Acurácia no conjunto de teste FINAL: 83.67%
Acurácia no conjunto de treinamento FINAL: 99.71%'''




