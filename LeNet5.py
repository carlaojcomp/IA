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


# Função para normalizar os dados
def normalizar(array):
    return array / 255.0


def dividir(X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    tamanho = int(len(X) * 0.7)

    indices_treino = indices[:tamanho]
    indices_teste = indices[tamanho:]

    return X[indices_treino], Y[indices_treino], X[indices_teste], Y[indices_teste]


# Função de ativação tanh (usada na LeNet-5 original)
def tanh(x):
    return np.tanh(x)


def tanh_derivada(x):
    return 1 - np.tanh(x) ** 2


# Função ReLU (alternativa moderna)
def relu(x):
    return np.maximum(0, x)


def relu_derivada(x):
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

    return grad_kernel, grad_in


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


class LeNet:
    def __init__(self,num_classes):

        self.input_shape = (28, 28)
        self.num_classes = num_classes
        self.activation = relu
        self.activation_deriv = relu_derivada

        # C1: Camada Convolucional 1
        # 6 feature maps, kernel 5x5
        # Input: 28x28 -> Output: 24x24 (28-5+1=24)
        self.num_filters_c1 = 6
        self.kernel_size_c1 = 5
        fan_in_c1 = self.kernel_size_c1 * self.kernel_size_c1
        self.filters_c1 = np.random.randn(self.num_filters_c1, self.kernel_size_c1, self.kernel_size_c1) * np.sqrt(2.0 / fan_in_c1)
        self.bias_c1 = np.zeros(self.num_filters_c1)

        # S2: Subsampling 1 (Max Pooling)
        # Kernel 2x2, stride 2
        # Input: 24x24 -> Output: 12x12
        self.pool_size_s2 = 2

        # C3: Camada Convolucional 2
        # 16 feature maps, kernel 5x5
        # Input: 12x12 -> Output: 8x8 (12-5+1=8)
        self.num_filters_c3 = 16
        self.kernel_size_c3 = 5
        fan_in_c3 = self.kernel_size_c3 * self.kernel_size_c3 * self.num_filters_c1
        self.filters_c3 = np.random.randn(self.num_filters_c3, self.num_filters_c1, self.kernel_size_c3, self.kernel_size_c3) * np.sqrt(2.0 / fan_in_c3)
        self.bias_c3 = np.zeros(self.num_filters_c3)

        # S4: Subsampling 2 (Max Pooling)
        # Kernel 2x2, stride 2
        # Input: 8x8 -> Output: 4x4
        self.pool_size_s4 = 2

        # C5: Camada Totalmente Conectada (convolucional com kernel 4x4)
        # 120 neurônios
        # Input: 4x4x16 = 256 -> Output: 120
        self.fc_input_size = self.num_filters_c3 * 4 * 4  # 16 * 4 * 4 = 256
        self.fc1_size = 120
        fan_in_fc1 = self.fc_input_size
        self.weights_fc1 = np.random.randn(self.fc_input_size, self.fc1_size) * np.sqrt(2.0 / fan_in_fc1)
        self.bias_fc1 = np.zeros((1, self.fc1_size))

        # F6: Camada Totalmente Conectada
        # 84 neurônios
        self.fc2_size = 84
        fan_in_fc2 = self.fc1_size
        self.weights_fc2 = np.random.randn(self.fc1_size, self.fc2_size) * np.sqrt(2.0 / fan_in_fc2)
        self.bias_fc2 = np.zeros((1, self.fc2_size))

        # Output: Camada de Saída
        # 10 neurônios (para MNIST)
        fan_in_out = self.fc2_size
        self.weights_out = np.random.randn(self.fc2_size, num_classes) * np.sqrt(2.0 / fan_in_out)
        self.bias_out = np.zeros((1, num_classes))

        self.cache = {}

    def forward(self, X):
        batch_size = X.shape[0]
        X_reshaped = X.reshape(batch_size, 28, 28)

        # C1: Primeira camada convolucional
        c1_outputs = []
        c1_pre_activation = []

        for i in range(batch_size):
            img_c1 = []
            img_pre_act = []
            for f in range(self.num_filters_c1):
                conv_result = conv2d(X_reshaped[i], self.filters_c1[f]) + self.bias_c1[f]
                img_pre_act.append(conv_result.copy())
                activated = self.activation(conv_result)
                img_c1.append(activated)
            c1_outputs.append(np.array(img_c1))
            c1_pre_activation.append(img_pre_act)

        c1_outputs = np.array(c1_outputs)  # Shape: (batch, 6, 24, 24)

        # S2: Max pooling
        s2_outputs = []
        s2_max_pos = []
        for i in range(batch_size):
            img_s2 = []
            max_pos_s2 = []
            for f in range(self.num_filters_c1):
                pooled, max_pos = max_pooling(c1_outputs[i, f], self.pool_size_s2)
                img_s2.append(pooled)
                max_pos_s2.append(max_pos)
            s2_outputs.append(np.array(img_s2))
            s2_max_pos.append(np.array(max_pos_s2))

        s2_outputs = np.array(s2_outputs)  # Shape: (batch, 6, 12, 12)

        # C3: Segunda camada convolucional
        c3_outputs = []
        c3_pre_activation = []

        for i in range(batch_size):
            img_c3 = []
            img_pre_act = []
            for f in range(self.num_filters_c3):
                conv_result = np.zeros((8, 8))  # 12-5+1 = 8

                # Somar contribuições de todos os feature maps de entrada
                for c in range(self.num_filters_c1):
                    conv_result += conv2d(s2_outputs[i, c], self.filters_c3[f, c])

                conv_result += self.bias_c3[f]
                img_pre_act.append(conv_result.copy())
                activated = self.activation(conv_result)
                img_c3.append(activated)

            c3_outputs.append(np.array(img_c3))
            c3_pre_activation.append(img_pre_act)

        c3_outputs = np.array(c3_outputs)  # Shape: (batch, 16, 8, 8)

        # S4: Max pooling
        s4_outputs = []
        s4_max_pos = []
        for i in range(batch_size):
            img_s4 = []
            max_pos_s4 = []
            for f in range(self.num_filters_c3):
                pooled, max_pos = max_pooling(c3_outputs[i, f], self.pool_size_s4)
                img_s4.append(pooled)
                max_pos_s4.append(max_pos)
            s4_outputs.append(np.array(img_s4))
            s4_max_pos.append(np.array(max_pos_s4))

        s4_outputs = np.array(s4_outputs)  # Shape: (batch, 16, 4, 4)

        # Achatar para camadas totalmente conectadas
        flattened = s4_outputs.reshape(batch_size, -1)  # Shape: (batch, 256)

        # C5/FC1: Primeira camada totalmente conectada
        fc1_out = np.dot(flattened, self.weights_fc1) + self.bias_fc1
        fc1_activated = self.activation(fc1_out)

        # F6/FC2: Segunda camada totalmente conectada
        fc2_out = np.dot(fc1_activated, self.weights_fc2) + self.bias_fc2
        fc2_activated = self.activation(fc2_out)

        # Output: Camada de saída com softmax
        output = np.dot(fc2_activated, self.weights_out) + self.bias_out
        output_softmax = softmax(output)

        # Cache para backpropagation
        self.cache = {
            'X_reshaped': X_reshaped,
            'c1_pre_activation': c1_pre_activation,
            'c1_outputs': c1_outputs,
            's2_outputs': s2_outputs,
            'c3_pre_activation': c3_pre_activation,
            'c3_outputs': c3_outputs,
            's4_outputs': s4_outputs,
            's2_max_pos': s2_max_pos,
            'flattened': flattened,
            's4_max_pos': s4_max_pos,
            'fc1_out': fc1_out,
            'fc1_activated': fc1_activated,
            'fc2_out': fc2_out,
            'fc2_activated': fc2_activated,
            'output': output
        }

        return output_softmax

    def backpropagation(self, X, y, output, learning_rate):
        batch_size = X.shape[0]

        # Gradiente da perda
        grad_output = entropia_cruzada_derivada(y, output)

        # Gradientes da camada de saída
        grad_weights_out = np.dot(self.cache['fc2_activated'].T, grad_output) / batch_size
        grad_bias_out = np.mean(grad_output, axis=0, keepdims=True)

        # Backprop para FC2
        grad_fc2_activated = np.dot(grad_output, self.weights_out.T)
        grad_fc2_out = grad_fc2_activated * self.activation_deriv(self.cache['fc2_out'])

        grad_weights_fc2 = np.dot(self.cache['fc1_activated'].T, grad_fc2_out) / batch_size
        grad_bias_fc2 = np.mean(grad_fc2_out, axis=0, keepdims=True)

        # Backprop para FC1
        grad_fc1_activated = np.dot(grad_fc2_out, self.weights_fc2.T)
        grad_fc1_out = grad_fc1_activated * self.activation_deriv(self.cache['fc1_out'])

        grad_weights_fc1 = np.dot(self.cache['flattened'].T, grad_fc1_out) / batch_size
        grad_bias_fc1 = np.mean(grad_fc1_out, axis=0, keepdims=True)

        # Propagar gradiente para S4
        grad_flattened = np.dot(grad_fc1_out, self.weights_fc1.T)
        grad_s4 = grad_flattened.reshape(self.cache['s4_outputs'].shape)

        # Backprop através de S4 (average pooling)
        grad_c3_from_s4 = np.zeros_like(self.cache['c3_outputs'])
        for i in range(batch_size):
            for f in range(self.num_filters_c3):
                grad_c3_from_s4[i, f] = max_pooling_back(grad_s4[i, f],self.cache['s4_max_pos'][i][f], self.pool_size_s4)

        # Backprop através de C3
        grad_filters_c3 = np.zeros_like(self.filters_c3)
        grad_bias_c3 = np.zeros_like(self.bias_c3)
        grad_s2_from_c3 = np.zeros_like(self.cache['s2_outputs'])

        for i in range(batch_size):
            for f in range(self.num_filters_c3):
                # Gradiente através da ativação
                grad_c3_pre_act = grad_c3_from_s4[i, f] * self.activation_deriv(self.cache['c3_pre_activation'][i][f])

                # Gradientes dos filtros e bias
                for c in range(self.num_filters_c1):
                    # CORREÇÃO: Usar a função corrigida conv2d_back
                    grad_kernel, grad_input = conv2d_back(
                        self.cache['s2_outputs'][i, c],
                        self.filters_c3[f, c],
                        grad_c3_pre_act
                    )
                    grad_filters_c3[f, c] += grad_kernel / batch_size
                    grad_s2_from_c3[i, c] += grad_input

                grad_bias_c3[f] += np.sum(grad_c3_pre_act) / batch_size

        # Backprop através de S2 (Max pooling)
        grad_c1_from_s2 = np.zeros_like(self.cache['c1_outputs'])
        for i in range(batch_size):
            for f in range(self.num_filters_c1):
                grad_c1_from_s2[i, f] = max_pooling_back(
                    grad_s2_from_c3[i, f],
                    self.cache['s2_max_pos'][i][f],
                    self.pool_size_s2
                )

        # Backprop através de C1
        grad_filters_c1 = np.zeros_like(self.filters_c1)
        grad_bias_c1 = np.zeros_like(self.bias_c1)

        for i in range(batch_size):
            for f in range(self.num_filters_c1):
                # Gradiente através da ativação
                grad_c1_pre_act = grad_c1_from_s2[i, f] * self.activation_deriv(self.cache['c1_pre_activation'][i][f])

                # Gradientes dos filtros e bias
                grad_kernel, _ = conv2d_back(
                    self.cache['X_reshaped'][i],
                    self.filters_c1[f],
                    grad_c1_pre_act
                )
                grad_filters_c1[f] += grad_kernel / batch_size
                grad_bias_c1[f] += np.sum(grad_c1_pre_act) / batch_size

        # Atualizar parâmetros
        self.weights_out -= learning_rate * grad_weights_out
        self.bias_out -= learning_rate * grad_bias_out
        self.weights_fc2 -= learning_rate * grad_weights_fc2
        self.bias_fc2 -= learning_rate * grad_bias_fc2
        self.weights_fc1 -= learning_rate * grad_weights_fc1
        self.bias_fc1 -= learning_rate * grad_bias_fc1
        self.filters_c3 -= learning_rate * grad_filters_c3
        self.bias_c3 -= learning_rate * grad_bias_c3
        self.filters_c1 -= learning_rate * grad_filters_c1
        self.bias_c1 -= learning_rate * grad_bias_c1

    def predict(self, X):
        return self.forward(X)

    def predict_classes(self, X):
        probabilidades = self.predict(X)
        return np.argmax(probabilidades, axis=1)

    def avaliar(self, X_test, y_test):
        """
        Avalia o modelo em dados de teste
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
    epochs = 10
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

    # Avaliação final
    #metricas_finais = modelo.avaliar(X_teste, y_teste)
    #print(f"Acurácia final no teste: {metricas_finais['acuracia']:.4f}")

    #return modelo, metricas_finais



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


# Criar modelo para treinamento
modelo = LeNet(10)
treinar_rede(modelo, X_treino,Y_treino,0.01)

pred = modelo.predict_classes(X_teste)
print(calcular_acuracia(pred,Y_teste))