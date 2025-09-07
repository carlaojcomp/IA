import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def one_hot_encoder(arr):
    arr_int = arr.astype(int)
    # Aplica a codificação One Hot usando np.eye
    return np.eye(10)[arr_int]
def converter_para_float(data):
    for coluna in data.columns:
        if data[coluna].dtype == 'object':
            data[coluna] = pd.to_numeric(data[coluna], errors='coerce')
    return data

#Função para normalizar os dados
def normalizar(array):
    return array / 255.0
def dividir(X,Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    tamanho = int(len(X) * 0.7)

    indices_treino = indices[:tamanho]
    indices_teste = indices[tamanho:]

    return X[indices_treino], Y[indices_treino], X[indices_teste], Y[indices_teste]
def identidade(x):
    return np.maximum(0, x)

#Derivada da função Identidade para a última camada Oculta
def identidade_derivada(x):
    return (x > 0).astype(float)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Para estabilidade numérica
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def conv2d(X, kernel):
    # Dimensões da entrada e do kernel
    h, w = X.shape
    k_h, k_w = kernel.shape

    # Tamanho da saída
    saida_h = (h - k_h) + 1
    saida_w = (w - k_w) + 1

    # Resultado da convolução
    saida = np.zeros((saida_h, saida_w))
    # Convolução 2D
    for i in range(saida_h):
        for j in range(saida_w):
            saida[i, j] = np.sum(X[i:i+k_h, j:j+k_w] * kernel)

    return saida

def conv2d_back(X, kernel, grad_out):
    k_h, k_w = kernel.shape
    grad_kernel = np.zeros_like(kernel)
    for i in range(grad_out.shape[0]):
        for j in range(grad_out.shape[1]):
            grad_kernel += X[i:i + k_h, j:j + k_w] * grad_out[i, j]

    # Gradiente em relação à entrada
    grad_in = np.zeros_like(X)
    for i in range(grad_out.shape[0]):
        for j in range(grad_out.shape[1]):
            grad_in[i:i + k_h, j:j + k_w] += kernel * grad_out[i, j]

    return grad_kernel


def max_pooling(X,size):
    h, w = X.shape
    saida_h = h // size
    saida_w = w // size
    saida = np.zeros((saida_h, saida_w))
    max_pos = np.zeros((h, w), dtype=bool)

    for i in range(saida_h):
        for j in range(saida_w):
            patch = X[i * size:(i + 1) * size, j * size:(j + 1) * size]
            max_val = np.max(patch)
            saida[i, j] = max_val

            # Encontrar a primeira posição do máximo
            max_indices = np.where(patch == max_val)
            if len(max_indices[0]) > 0:
                pi, pj = max_indices[0][0], max_indices[1][0]
                max_pos[i * size + pi, j * size + pj] = True

    return saida, max_pos
def max_pooling_back(grad_out, max_pos, size):
    saida_h, saida_w = grad_out.shape
    input_h, input_w = max_pos.shape
    grad_in = np.zeros((input_h, input_w))
    for i in range(saida_h):
        for j in range(saida_w):
            # Distribuir gradiente apenas para as posições que eram máximas
            patch = max_pos[i * size:(i + 1) * size, j * size:(j + 1) * size]
            grad_in[i * size:(i + 1) * size, j * size:(j + 1) * size] += grad_out[i, j] * patch

    return grad_in
def entropia_cruzada(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))

# Função de derivada da perda para backpropagation
def entropia_cruzada_derivada(y_true, y_pred):
    return y_pred - y_true
class CNN:
        def __init__(self, tam_input, num_filtros, tam_filtros, tam_maxpool, classes):
            self.num_filtros = num_filtros
            self.tam_filtros = tam_filtros
            self.tam_maxpool = tam_maxpool

            # Melhor inicialização
            fan_in = tam_filtros * tam_filtros
            self.filtros = np.random.randn(num_filtros, tam_filtros, tam_filtros) * np.sqrt(2.0 / fan_in)

            # Adicionar bias para os filtros
            self.filtros_bias = np.zeros(num_filtros)

            # Calcular dimensões
            tam_conv = 28 - tam_filtros + 1
            tam_pool = tam_conv // tam_maxpool
            fc_input_size = num_filtros * tam_pool * tam_pool

            # Duas camadas totalmente conectadas
            self.pesos1 = np.random.randn(fc_input_size, 128) * np.sqrt(2.0 / fc_input_size)
            self.bias1 = np.zeros((1, 128))

            self.pesos2 = np.random.randn(128, classes) * np.sqrt(2.0 / 128)
            self.bias2 = np.zeros((1, classes))

            self.cache = {}

        def forward(self, X):
            batch_size = X.shape[0]
            X_reshaped = X.reshape(batch_size, 28, 28)

            conv_out = []
            conv_pre_relu = []
            max_pos_list = []

            for i in range(batch_size):
                img_conv = []
                img_pre_relu = []
                img_positions = []

                for f in range(self.num_filtros):
                    # Convolução com bias
                    conv_result = conv2d(X_reshaped[i], self.filtros[f]) + self.filtros_bias[f]
                    img_pre_relu.append(conv_result.copy())

                    # ReLU
                    conv_result = identidade(conv_result)

                    # Max pooling
                    pooled_result, max_pos = max_pooling(conv_result, self.tam_maxpool)
                    img_conv.append(pooled_result)
                    img_positions.append(max_pos)

                conv_out.append(np.array(img_conv))
                conv_pre_relu.append(img_pre_relu)
                max_pos_list.append(img_positions)

            conv_outputs = np.array(conv_out)
            achatado = conv_outputs.reshape(batch_size, -1)

            # Primeira camada FC com ReLU
            fc1_out = np.dot(achatado, self.pesos1) + self.bias1
            fc1_relu = identidade(fc1_out)

            # Segunda camada FC
            fc2_out = np.dot(fc1_relu, self.pesos2) + self.bias2
            saida = softmax(fc2_out)

            # Cache
            self.cache = {
                'X_reshaped': X_reshaped,
                'conv_pre_relu': conv_pre_relu,
                'conv_outputs': conv_outputs,
                'max_pos': max_pos_list,
                'achatado': achatado,
                'fc1_out': fc1_out,
                'fc1_relu': fc1_relu,
                'fc2_out': fc2_out
            }

            return saida

        def backpropagation(self, X, y, saida, alpha):
            batch_size = X.shape[0]

            # Gradiente da perda
            grad_erro = entropia_cruzada_derivada(y, saida)

            # Gradientes da segunda camada FC
            grad_pesos2 = np.dot(self.cache['fc1_relu'].T, grad_erro) / batch_size
            grad_bias2 = np.mean(grad_erro, axis=0, keepdims=True)

            # Backprop para primeira camada FC
            grad_fc1_relu = np.dot(grad_erro, self.pesos2.T)
            grad_fc1_out = grad_fc1_relu * identidade_derivada(self.cache['fc1_out'])

            grad_pesos1 = np.dot(self.cache['achatado'].T, grad_fc1_out) / batch_size
            grad_bias1 = np.mean(grad_fc1_out, axis=0, keepdims=True)

            # Propagar para conv
            grad_achatado = np.dot(grad_fc1_out, self.pesos1.T)
            grad_conv = grad_achatado.reshape(self.cache['conv_outputs'].shape)

            # Gradientes dos filtros
            grad_filtros = np.zeros_like(self.filtros)
            grad_filtros_bias = np.zeros_like(self.filtros_bias)

            for i in range(batch_size):
                for f in range(self.num_filtros):
                    grad_before_pool = max_pooling_back(
                        grad_conv[i, f],
                        self.cache['max_pos'][i][f],
                        self.tam_maxpool
                    )

                    grad_before_relu = grad_before_pool * identidade_derivada(
                        self.cache['conv_pre_relu'][i][f]
                    )

                    grad_kernel = conv2d_back(
                        self.cache['X_reshaped'][i],
                        self.filtros[f],
                        grad_before_relu
                    )
                    grad_filtros[f] += grad_kernel / batch_size
                    grad_filtros_bias[f] += np.sum(grad_before_relu) / batch_size

            # Atualizar parâmetros
            self.pesos2 -= alpha * grad_pesos2
            self.bias2 -= alpha * grad_bias2
            self.pesos1 -= alpha * grad_pesos1
            self.bias1 -= alpha * grad_bias1
            self.filtros -= alpha * grad_filtros
            self.filtros_bias -= alpha * grad_filtros_bias

        def predict(self, X):
            batch_size = X.shape[0]
            X_reshaped = X.reshape(batch_size, 28, 28)

            conv_outputs = []
            for i in range(batch_size):
                img_conv = []
                for f in range(self.num_filtros):
                    conv_result = conv2d(X_reshaped[i], self.filtros[f]) + self.filtros_bias[f]
                    conv_result = identidade(conv_result)
                    pooled, _ = max_pooling(conv_result, self.tam_maxpool)
                    img_conv.append(pooled)
                conv_outputs.append(np.array(img_conv))

            conv_outputs = np.array(conv_outputs)
            achatado = conv_outputs.reshape(batch_size, -1)

            fc1_out = np.dot(achatado, self.pesos1) + self.bias1
            fc1_relu = identidade(fc1_out)
            fc2_out = np.dot(fc1_relu, self.pesos2) + self.bias2

            return softmax(fc2_out)

        def predict_classes(self, X):
            probabilidades = self.predict(X)
            return np.argmax(probabilidades, axis=1)

        def avaliar(self, X_test, y_test):
            """
            Avalia o modelo em dados de teste

            Args:
                X_test: Dados de teste (N, 784)
                y_test: Labels de teste em one-hot (N, classes)

            Returns:
                dict: Métricas de avaliação
            """
            # Predições
            y_pred_prob = self.predict(X_test)
            y_pred_classes = np.argmax(y_pred_prob, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)

            # Calcular métricas
            acuracia = np.mean(y_pred_classes == y_true_classes)
            perda = entropia_cruzada(y_test, y_pred_prob)

            # Matriz de confusão básica
            num_classes = y_test.shape[1]
            matriz_confusao = np.zeros((num_classes, num_classes))
            for true, pred in zip(y_true_classes, y_pred_classes):
                matriz_confusao[true, pred] += 1

            return {
                'acuracia': acuracia,
                'perda': perda,
                'matriz_confusao': matriz_confusao,
                'predicoes_prob': y_pred_prob,
                'predicoes_classes': y_pred_classes
            }

def treinar_rede(modelo, X_treino, y_treino, alpha):
    batch_size = 64
    epochs = 100
    for epoca in range(epochs):
        total_batches = X_treino.shape[0] // batch_size
        erro_total = 0

        # Dividir os dados em mini-batches
        for i in range(0, len(X_treino), batch_size):
            X_batch = X_treino[i:i + batch_size]
            y_batch = y_treino[i:i + batch_size]

            # Forward pass
            saida = modelo.forward(X_batch)

            # Calcular a perda
            erro = entropia_cruzada(y_batch, saida)
            erro_total += erro

            # Backward pass
            modelo.backpropagation(X_batch, y_batch, saida, alpha)

        # Média do erro para a época
        erro_medio = erro_total / total_batches
        print(f"Época {epoca + 1}, Erro Médio: {erro_medio}")



def calcular_acuracia(y_pred, y_verdadeiro):
    # Converte para classe prevista (índice do maior valor da saída)
    if y_verdadeiro.ndim == 2:
        true_classes = np.argmax(y_verdadeiro, axis=1)
    else:
        true_classes = y_verdadeiro

    # Converter y_pred para classes se estiver em probabilidades
    if y_pred.ndim == 2:
        pred_classes = np.argmax(y_pred, axis=1)
    else:
        pred_classes = y_pred

    return np.mean(pred_classes == true_classes)
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
def validar_classificador(modelo,X_treino, Y_treino, k):
    pastas = kfold_separar(X_treino, Y_treino, k)
    acuracias_treino = []
    acuracias_val = []


    # Para cada fold
    for X_train, y_train, X_val, y_val in pastas:
        # Treinar o modelo com gradiente descendente
        treinar_rede(modelo, X_train, y_train,0.01)

        # Classificar usando os pesos obtidos
        y_pred_val = modelo.predict_classes(X_val)
        y_pred_train = modelo.predict_classes(X_train)

        # Calcular a acurácia
        acuraccy_val = calcular_acuracia(y_pred_val, y_val)
        acuraccy_train = calcular_acuracia(y_pred_train, y_train)

        print(f"Acurácia do Linear para este fold (Val): {acuraccy_val:.2f}")
        print(f"Acurácia do Linear para este fold (Train): {acuraccy_train:.2f}")
        acuracias_val.append(acuraccy_val)
        acuracias_treino.append(acuraccy_train)

    # Retornar a acurácia média
    print(f"Acurácia média do MLP para os folds (Val): {np.mean(acuracias_val):.2f}%")
    print(f"Acurácia média do MLP para os folds (Train): {np.mean(acuracias_treino):.2f}%")
    #Plot a acurácia de cada fold
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, k + 1), acuracias_treino, label='Acurácia Treino', marker='o')
    plt.plot(range(1, k + 1), acuracias_val, label='Acurácia Validação', marker='s')
    plt.xlabel('Pasta')
    plt.ylabel('Acurácia Class Linear (%)')
    plt.title('Evolução da Acurácia por Pasta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
imagens = np.array([X[i].reshape(28, 28) for i in range(3000)])
print(imagens.shape)
'''plt.imshow(X, cmap='gray')
plt.title("Imagem MNIST Redimensionada")
plt.axis('off')  # Desativa os eixos
plt.show()'''
tam_input = 784
num_filtros = 6  # Aumentado de 2 para 6
tam_filtros = 5  # Aumentado de 3 para 5
tam_maxpool = 2
classes = 10

# Criar modelo para treinamento
modelo = CNN(tam_input, num_filtros, tam_filtros, tam_maxpool, classes)
validar_classificador(modelo, X_treino, Y_treino,3)

pred = modelo.predict_classes(X_teste)
print(calcular_acuracia(pred,Y_teste))