import numba
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')



# Otimizações com Numba JIT
@jit(nopython=True, parallel=True)
def conv2d(X, kernel):
    """Convolução 2D"""
    h, w = X.shape
    k_h, k_w = kernel.shape
    saida_h = h - k_h + 1
    saida_w = w - k_w + 1
    saida = np.zeros((saida_h, saida_w))

    for i in prange(saida_h):
        for j in range(saida_w):
            sum_val = 0.0
            for ki in range(k_h):
                for kj in range(k_w):
                    sum_val += X[i + ki, j + kj] * kernel[ki, kj]
            saida[i, j] = sum_val

    return saida


@jit(nopython=True, parallel=True)
def conv2d_back(X, kernel, grad_out):
    """Backpropagation de convolução"""
    k_h, k_w = kernel.shape
    grad_kernel = np.zeros_like(kernel)
    grad_in = np.zeros_like(X)

    # Gradiente do kernel
    for i in prange(grad_out.shape[0]):
        for j in range(grad_out.shape[1]):
            for ki in range(k_h):
                for kj in range(k_w):
                    grad_kernel[ki, kj] += X[i + ki, j + kj] * grad_out[i, j]

    # Gradiente da entrada
    for i in prange(grad_out.shape[0]):
        for j in range(grad_out.shape[1]):
            for ki in range(k_h):
                for kj in range(k_w):
                    grad_in[i + ki, j + kj] += kernel[ki, kj] * grad_out[i, j]

    return grad_kernel, grad_in


@jit(nopython=True, parallel=True)
def max_pooling(X, size):
    """Max-Pooling"""
    h, w = X.shape
    saida_h = h // size
    saida_w = w // size
    saida = np.zeros((saida_h, saida_w))
    max_pos = np.zeros((h, w), dtype=numba.boolean)

    for i in prange(saida_h):
        for j in range(saida_w):
            max_val = -np.inf
            max_i, max_j = 0, 0
            # Encontrar o máximo no patch
            for pi in range(size):
                for pj in range(size):
                    val = X[i * size + pi, j * size + pj]
                    if val > max_val:
                        max_val = val
                        max_i, max_j = pi, pj

            saida[i, j] = max_val
            max_pos[i * size + max_i, j * size + max_j] = True

    return saida, max_pos
@jit(nopython=True)
def max_pooling_back(grad_out, max_pos, size):
    """Backpropagation de Max-Pooling"""
    saida_h, saida_w = grad_out.shape
    input_h, input_w = max_pos.shape
    grad_in = np.zeros((input_h, input_w))

    for i in range(saida_h):
        for j in range(saida_w):
            for pi in range(size):
                for pj in range(size):
                    if max_pos[i * size + pi, j * size + pj]:
                        grad_in[i * size + pi, j * size + pj] += grad_out[i, j]

    return grad_in

# Função ReLU (Identidade)
@jit(nopython=True)
def relu(x):
    return np.maximum(0, x)

# Derivada da ReLU
@jit(nopython=True)
def relu_derivada(x):
    return (x > 0).astype(numba.float64)


# Função Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)





def one_hot_encoder(arr):
    arr_int = arr.astype(int)
    return np.eye(2)[arr_int]


def converter_para_float(data):
    for coluna in data.columns:
        if data[coluna].dtype == 'object':
            data[coluna] = pd.to_numeric(data[coluna], errors='coerce')
    return data

# Normaliza os Pixels
def normalizar(array):
    return array / 255.0

# Divide a base entre treino e teste
def dividir(X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    tamanho = int(len(X) * 0.7)
    indices_treino = indices[:tamanho]
    indices_teste = indices[tamanho:]
    return X[indices_treino], Y[indices_treino], X[indices_teste], Y[indices_teste]

# Função Entropia Cruzada (Cross-Entr0py)
def entropia_cruzada(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=1))


# Derivada da Função Entropia Cruzada
def entropia_cruzada_derivada(y_true, y_pred):
    return y_pred - y_true

# Classe da Rede Convolutiva: Arquitetura CascNet
class CascNet:
    def __init__(self, num_classes):
        """
            num_classes: Número de classes para classificação
            n_threads: Número de threads
        """
        self.input_shape = (128, 128)
        self.num_classes = num_classes
        self.n_threads = cpu_count()

        # Inicialização das camadas
        self.num_filters_c1 = 12
        self.kernel_size_c1 = 6
        fan_in_c1 = self.kernel_size_c1 * self.kernel_size_c1
        self.filters_c1 = np.random.randn(self.num_filters_c1, self.kernel_size_c1, self.kernel_size_c1) * np.sqrt(
            2.0 / fan_in_c1)
        self.bias_c1 = np.zeros(self.num_filters_c1)

        self.pool_size_s2 = 4

        self.num_filters_c3 = 20
        self.kernel_size_c3 = 10
        fan_in_c3 = self.kernel_size_c3 * self.kernel_size_c3 * self.num_filters_c1
        self.filters_c3 = np.random.randn(self.num_filters_c3, self.num_filters_c1, self.kernel_size_c3,
                                          self.kernel_size_c3) * np.sqrt(2.0 / fan_in_c3)
        self.bias_c3 = np.zeros(self.num_filters_c3)

        self.pool_size_s4 = 4

        # Entrada: 128x128
        # Após C1 (kernel 6x6): 123x123
        # Após S2 (pool 4x4): 30x30
        # Após C3 (kernel 10x10): 21x21
        # Após S4 (pool 4x4): 5x5
        # FC1: 120 -> 84
        # FC2 : 84 -> 2
        # Softmax
        self.fc_input_size = self.num_filters_c3 * 5 * 5
        self.fc1_size = 120
        fan_in_fc1 = self.fc_input_size
        self.weights_fc1 = np.random.randn(self.fc_input_size, self.fc1_size) * np.sqrt(2.0 / fan_in_fc1)
        self.bias_fc1 = np.zeros((1, self.fc1_size))

        self.fc2_size = 84
        fan_in_fc2 = self.fc1_size
        self.weights_fc2 = np.random.randn(self.fc1_size, self.fc2_size) * np.sqrt(2.0 / fan_in_fc2)
        self.bias_fc2 = np.zeros((1, self.fc2_size))

        fan_in_out = self.fc2_size
        self.weights_out = np.random.randn(self.fc2_size, num_classes) * np.sqrt(2.0 / fan_in_out)
        self.bias_out = np.zeros((1, num_classes))

        self.cache = {}

    def c1_filter(self, args):
        """Processa um filtro da camada C1 para uma imagem específica"""
        img, filter_idx = args
        conv_result = conv2d(img, self.filters_c1[filter_idx]) + self.bias_c1[filter_idx]
        return conv_result, relu(conv_result)

    def c1_batch(self, X_reshaped):
        """Processa toda a camada C1 em paralelo"""
        batch_size = X_reshaped.shape[0]
        c1_outputs = np.zeros((batch_size, self.num_filters_c1, 123, 123))
        c1_pre_activation = np.zeros((batch_size, self.num_filters_c1, 123, 123))

        # Paralelizar por imagem e filtro usando ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                # Criar argumentos para cada filtro
                args = [(X_reshaped[i], f) for f in range(self.num_filters_c1)]
                # Processar filtros em paralelo
                results = list(executor.map(self.c1_filter, args))
                for f, (pre_act, activated) in enumerate(results):
                    c1_pre_activation[i, f] = pre_act
                    c1_outputs[i, f] = activated

        return c1_outputs, c1_pre_activation

    def s2(self, c1_outputs):
        """Processa camada S2 (max pooling) em paralelo"""
        batch_size, num_filters = c1_outputs.shape[0], c1_outputs.shape[1]
        s2_outputs = np.zeros((batch_size, num_filters, 30, 30))
        s2_max_pos = []

        def pooling(args):
            i, f = args
            return max_pooling(c1_outputs[i, f], self.pool_size_s2)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                args = [(i, f) for f in range(num_filters)]
                results = list(executor.map(pooling, args))

                max_pos_batch = []
                for f, (pooled, max_pos) in enumerate(results):
                    s2_outputs[i, f] = pooled
                    max_pos_batch.append(max_pos)
                s2_max_pos.append(max_pos_batch)

        return s2_outputs, s2_max_pos

    def s4(self, c3_outputs):
        """Processa camada S4 (max pooling) em paralelo"""
        batch_size, num_filters = c3_outputs.shape[0], c3_outputs.shape[1]
        s4_outputs = np.zeros((batch_size, num_filters, 5, 5))
        s4_max_pos = []

        def pooling(args):
            i, f = args
            return max_pooling(c3_outputs[i, f], self.pool_size_s4)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                args = [(i, f) for f in range(num_filters)]
                results = list(executor.map(pooling, args))

                max_pos_batch = []
                for f, (pooled, max_pos) in enumerate(results):
                    s4_outputs[i, f] = pooled
                    max_pos_batch.append(max_pos)
                s4_max_pos.append(max_pos_batch)

        return s4_outputs, s4_max_pos

    def c3_filter(self, args):
        """Processa um filtro da camada C3"""
        s2_output, filter_idx = args
        # Calcular dimensões corretas: 30 - 10 + 1 = 21
        conv_result = np.zeros((21, 21))

        for c in range(self.num_filters_c1):
            conv_result += conv2d(s2_output[c], self.filters_c3[filter_idx, c])

        conv_result += self.bias_c3[filter_idx]
        return conv_result, relu(conv_result)

    def c3_batch(self, s2_outputs):
        """Processa camada C3 em paralelo"""
        batch_size = s2_outputs.shape[0]
        c3_outputs = np.zeros((batch_size, self.num_filters_c3, 21, 21))  # Corrigido: 30-10+1 = 21
        c3_pre_activation = np.zeros((batch_size, self.num_filters_c3, 21, 21))

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                args = [(s2_outputs[i], f) for f in range(self.num_filters_c3)]
                results = list(executor.map(self.c3_filter, args))
                for f, (pre_act, activated) in enumerate(results):
                    c3_pre_activation[i, f] = pre_act
                    c3_outputs[i, f] = activated

        return c3_outputs, c3_pre_activation

    def forward(self, X):
        """Forward com paralelização"""
        batch_size = X.shape[0]
        X_reshaped = X.reshape(batch_size, 128, 128)

        # C1: Primeira camada convolucional (paralela)
        c1_outputs, c1_pre_activation = self.c1_batch(X_reshaped)

        # S2: Max pooling (paralelo)
        s2_outputs, s2_max_pos = self.s2(c1_outputs)

        # C3: Segunda camada convolucional (paralela)
        c3_outputs, c3_pre_activation = self.c3_batch(s2_outputs)

        # S4: Max pooling (paralelo) - usando método específico para S4
        s4_outputs, s4_max_pos = self.s4(c3_outputs)

        # Achatar para camadas totalmente conectadas
        flattened = s4_outputs.reshape(batch_size, -1)

        # Camadas totalmente conectadas
        fc1_out = np.dot(flattened, self.weights_fc1) + self.bias_fc1
        fc1_activated = relu(fc1_out)

        fc2_out = np.dot(fc1_activated, self.weights_fc2) + self.bias_fc2
        fc2_activated = relu(fc2_out)

        # Output com softmax
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

    def backprop_c3(self, grad_c3_from_s4, batch_size):
        """Backpropagation da camada C3 paralela"""
        grad_filters_c3 = np.zeros_like(self.filters_c3)
        grad_bias_c3 = np.zeros_like(self.bias_c3)
        grad_s2_from_c3 = np.zeros_like(self.cache['s2_outputs'])

        def process_backprop_c3(args):
            i, f = args
            grad_c3_pre_act = grad_c3_from_s4[i, f] * relu_derivada(self.cache['c3_pre_activation'][i][f])

            local_grad_filters = np.zeros((self.num_filters_c1, self.kernel_size_c3, self.kernel_size_c3))
            # Usar as dimensões reais da saída S2
            s2_shape = self.cache['s2_outputs'].shape
            local_grad_s2 = np.zeros((self.num_filters_c1, s2_shape[2], s2_shape[3]))

            for c in range(self.num_filters_c1):
                grad_kernel, grad_input = conv2d_back(
                    self.cache['s2_outputs'][i, c],
                    self.filters_c3[f, c],
                    grad_c3_pre_act
                )
                local_grad_filters[c] = grad_kernel
                local_grad_s2[c] = grad_input

            local_grad_bias = np.sum(grad_c3_pre_act)
            return f, local_grad_filters, local_grad_s2, local_grad_bias

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                args = [(i, f) for f in range(self.num_filters_c3)]
                results = list(executor.map(process_backprop_c3, args))

                for f, local_grad_filters, local_grad_s2, local_grad_bias in results:
                    grad_filters_c3[f] += local_grad_filters / batch_size
                    grad_s2_from_c3[i] += local_grad_s2
                    grad_bias_c3[f] += local_grad_bias / batch_size

        return grad_filters_c3, grad_bias_c3, grad_s2_from_c3

    def backpropagation(self, X, y, output, alpha):
        """Backpropagation com paralelização otimizada"""
        batch_size = X.shape[0]

        # Gradiente da perda
        grad_output = entropia_cruzada_derivada(y, output)

        # Gradientes da camada de saída
        grad_weights_out = np.dot(self.cache['fc2_activated'].T, grad_output) / batch_size
        grad_bias_out = np.mean(grad_output, axis=0, keepdims=True)

        # Backprop para FC2
        grad_fc2_activated = np.dot(grad_output, self.weights_out.T)
        grad_fc2_out = grad_fc2_activated * relu_derivada(self.cache['fc2_out'])

        grad_weights_fc2 = np.dot(self.cache['fc1_activated'].T, grad_fc2_out) / batch_size
        grad_bias_fc2 = np.mean(grad_fc2_out, axis=0, keepdims=True)

        # Backprop para FC1
        grad_fc1_activated = np.dot(grad_fc2_out, self.weights_fc2.T)
        grad_fc1_out = grad_fc1_activated * relu_derivada(self.cache['fc1_out'])

        grad_weights_fc1 = np.dot(self.cache['flattened'].T, grad_fc1_out) / batch_size
        grad_bias_fc1 = np.mean(grad_fc1_out, axis=0, keepdims=True)

        # Propagar gradiente para S4
        grad_flattened = np.dot(grad_fc1_out, self.weights_fc1.T)
        grad_s4 = grad_flattened.reshape(self.cache['s4_outputs'].shape)

        # Backprop através de S4 (paralelo)
        grad_c3_from_s4 = np.zeros_like(self.cache['c3_outputs'])

        def process_s4_backprop(args):
            i, f = args
            return max_pooling_back(grad_s4[i, f], self.cache['s4_max_pos'][i][f], self.pool_size_s4)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                args = [(i, f) for f in range(self.num_filters_c3)]
                results = list(executor.map(process_s4_backprop, args))
                for f, result in enumerate(results):
                    grad_c3_from_s4[i, f] = result

        # Backprop através de C3 (paralelo)
        grad_filters_c3, grad_bias_c3, grad_s2_from_c3 = self.backprop_c3(grad_c3_from_s4, batch_size)

        # Backprop através de S2 e C1 (similar ao C3, mas simplificado aqui)
        grad_c1_from_s2 = np.zeros_like(self.cache['c1_outputs'])

        def process_backprop_s2(args):
            i, f = args
            return max_pooling_back(grad_s2_from_c3[i, f], self.cache['s2_max_pos'][i][f], self.pool_size_s2)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                args = [(i, f) for f in range(self.num_filters_c1)]
                results = list(executor.map(process_backprop_s2, args))
                for f, result in enumerate(results):
                    grad_c1_from_s2[i, f] = result

        # Backprop através de C1 (paralelo)
        grad_filters_c1 = np.zeros_like(self.filters_c1)
        grad_bias_c1 = np.zeros_like(self.bias_c1)

        def process_backprop_c1(args):
            i, f = args
            grad_c1_pre_act = grad_c1_from_s2[i, f] * relu_derivada(self.cache['c1_pre_activation'][i][f])
            grad_kernel, _ = conv2d_back(
                self.cache['X_reshaped'][i],
                self.filters_c1[f],
                grad_c1_pre_act
            )
            return f, grad_kernel, np.sum(grad_c1_pre_act)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(batch_size):
                args = [(i, f) for f in range(self.num_filters_c1)]
                results = list(executor.map(process_backprop_c1, args))
                for f, grad_kernel, grad_bias in results:
                    grad_filters_c1[f] += grad_kernel / batch_size
                    grad_bias_c1[f] += grad_bias / batch_size

        # Atualizar parâmetros
        self.weights_out -= alpha * grad_weights_out
        self.bias_out -= alpha * grad_bias_out
        self.weights_fc2 -= alpha * grad_weights_fc2
        self.bias_fc2 -= alpha * grad_bias_fc2
        self.weights_fc1 -= alpha * grad_weights_fc1
        self.bias_fc1 -= alpha * grad_bias_fc1
        self.filters_c3 -= alpha * grad_filters_c3
        self.bias_c3 -= alpha * grad_bias_c3
        self.filters_c1 -= alpha * grad_filters_c1
        self.bias_c1 -= alpha * grad_bias_c1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def avaliar(self, X_test, y_test):
        """Avalia o modelo em dados de teste"""
        y_pred_prob = self.forward(X_test)
        y_pred_classes = np.argmax(self.forward(X_test), axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        acuracia = np.mean(y_pred_classes == y_true_classes)
        perda = entropia_cruzada(y_test, y_pred_prob)

        num_classes = y_test.shape[1]
        matriz_confusao = np.zeros((num_classes, num_classes))
        for true, pred in zip(y_true_classes, y_pred_classes):
            matriz_confusao[true, pred] += 1
        precisions, recalls, f1s = [], [], []
        for i in range(num_classes):
            VP = matriz_confusao[i, i]
            FP = matriz_confusao[:, i].sum() - VP
            FN = matriz_confusao[i, :].sum() - VP

            precision = VP / (VP + FP) if (VP + FP) > 0 else 0
            recall = VP / (VP + FN) if (VP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            'acuraccy': 100*acuracia,
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s),
            'loss': perda,
            'matriz_confusao': matriz_confusao,
        }


def treinar_rede(modelo, X_treino, y_treino, alpha):
    """Função de treinamento com processamento paralelo de batches"""
    print(f"Iniciando treinamento com {modelo.n_threads} threads...")
    batch_size = 16
    epochs = 5
    for epoca in range(epochs):
        total_batches = X_treino.shape[0] // batch_size
        erro_total = 0

        # Processar mini-batches
        for i in range(0, len(X_treino), batch_size):
            X_batch = X_treino[i:i + batch_size]
            y_batch = y_treino[i:i + batch_size]

            # Forward paralelo
            saida = modelo.forward(X_batch)

            # Calcular a perda
            erro = entropia_cruzada(y_batch, saida)
            erro_total += erro

            # Backpropagation paralelo
            modelo.backpropagation(X_batch, y_batch, saida, alpha)

        # Média do erro para a época
        erro_medio = erro_total / total_batches
        print(f"Época {epoca + 1}, Erro Médio: {erro_medio}")

    print("Treinamento concluído!")


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
def validar_classificador(num_classes,X_treino, Y_treino, k,alpha):
    pastas = kfold_separar(X_treino, Y_treino, k)
    acuracias_treino = []
    acuracias_val = []


    # Para cada fold
    for X_train, y_train, X_val, y_val in pastas:
        modelo = CascNet(num_classes)
        # Treinar o modelo com gradiente descendente
        treinar_rede(modelo, X_train, y_train,alpha)

        # Classificar usando os pesos obtidos
        y_pred_val = modelo.predict(X_val)
        y_pred_train = modelo.predict(X_train)

        # Calcular a acurácia
        acuraccy_val = calcular_acuracia(y_pred_val, y_val)
        acuraccy_train = calcular_acuracia(y_pred_train, y_train)

        print(f"Acurácia para este fold (Val): {100*acuraccy_val:.2f}%")
        print(f"Acurácia para este fold (Train): {100*acuraccy_train:.2f}%")
        acuracias_val.append(acuraccy_val)
        acuracias_treino.append(acuraccy_train)

    # Retornar a acurácia média
    print(f"Acurácia média para os folds (Val): {100*np.mean(acuracias_val):.2f}%")
    print(f"Acurácia média para os folds (Train): {100*np.mean(acuracias_treino):.2f}%")
    #Plot a acurácia de cada fold
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, k + 1), acuracias_treino, label='Acurácia Treino', marker='o')
    plt.plot(range(1, k + 1), acuracias_val, label='Acurácia Validação', marker='s')
    plt.xlabel('Pasta')
    plt.ylabel('Acurácia (%)')
    plt.title('Evolução da Acurácia por Pasta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



mnist = pd.read_csv('mnist_3000.txt')
fei128 = pd.read_csv('FEI128.csv')
mnist.rename(columns={'0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t60.0\t141.0\t255.0\t178.0\t141.0\t141.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t63.0\t234.0\t252.0\t253.0\t252.0\t252.0\t252.0\t207.0\t56.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t169.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t253.0\t215.0\t81.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t13.0\t206.0\t252.0\t252.0\t253.0\t227.0\t190.0\t139.0\t253.0\t252.0\t243.0\t125.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t154.0\t253.0\t244.0\t175.0\t176.0\t125.0\t0.0\t0.0\t176.0\t244.0\t253.0\t253.0\t63.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t253.0\t252.0\t142.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t82.0\t240.0\t252.0\t194.0\t19.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t89.0\t253.0\t233.0\t37.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t110.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t213.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t10.0\t178.0\t253.0\t133.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t253.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t197.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t65.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t147.0\t253.0\t226.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t122.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t138.0\t209.0\t28.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t66.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t67.0\t246.0\t76.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t26.0\t159.0\t202.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t76.0\t250.0\t254.0\t228.0\t44.0\t0.0\t0.0\t0.0\t0.0\t13.0\t41.0\t216.0\t253.0\t78.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t200.0\t253.0\t252.0\t224.0\t69.0\t57.0\t57.0\t144.0\t206.0\t253.0\t252.0\t170.0\t9.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t25.0\t216.0\t252.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t206.0\t93.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t28.0\t139.0\t228.0\t252.0\t253.0\t252.0\t214.0\t139.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0': "A"}, inplace=True)
mnist.loc[mnist.size] = '0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t60.0\t141.0\t255.0\t178.0\t141.0\t141.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t63.0\t234.0\t252.0\t253.0\t252.0\t252.0\t252.0\t207.0\t56.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t169.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t253.0\t215.0\t81.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t13.0\t206.0\t252.0\t252.0\t253.0\t227.0\t190.0\t139.0\t253.0\t252.0\t243.0\t125.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t154.0\t253.0\t244.0\t175.0\t176.0\t125.0\t0.0\t0.0\t176.0\t244.0\t253.0\t253.0\t63.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t253.0\t252.0\t142.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t82.0\t240.0\t252.0\t194.0\t19.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t89.0\t253.0\t233.0\t37.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t110.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t213.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t10.0\t178.0\t253.0\t133.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t253.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t197.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t65.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t196.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t113.0\t253.0\t145.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t147.0\t253.0\t226.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t114.0\t254.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t122.0\t252.0\t225.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t138.0\t209.0\t28.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t247.0\t66.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t67.0\t246.0\t76.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t85.0\t252.0\t253.0\t84.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t26.0\t159.0\t202.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t76.0\t250.0\t254.0\t228.0\t44.0\t0.0\t0.0\t0.0\t0.0\t13.0\t41.0\t216.0\t253.0\t78.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t200.0\t253.0\t252.0\t224.0\t69.0\t57.0\t57.0\t144.0\t206.0\t253.0\t252.0\t170.0\t9.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t25.0\t216.0\t252.0\t252.0\t252.0\t253.0\t252.0\t252.0\t252.0\t206.0\t93.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t28.0\t139.0\t228.0\t252.0\t253.0\t252.0\t214.0\t139.0\t13.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0'
mnist = mnist['A'].str.split(expand=True)
mnist = converter_para_float(mnist)
fei128 = converter_para_float(fei128)
classe = mnist[0].to_numpy()
class128 = fei128['classe'].to_numpy()

# Dropa os rótulos
mnist = mnist.drop(0, axis=1)
fei128 = fei128.drop('classe', axis=1)

X = fei128.to_numpy()
# Aplica a codificação one-hot na Mnist
Y = one_hot_encoder(class128)
# Normalização das features
X = normalizar(X)
# Divisão dos conjuntos em treino e teste
X_treino,Y_treino, X_teste, Y_teste = dividir(X,Y)

# Criar modelo para treinamento
modelo = CascNet(2)
# Treinamento da rede
treinar_rede(modelo, X_treino, Y_treino, 0.01)
# Validar modelo
validar_classificador(2, X_treino,Y_treino,5,0.01)
# Avaliar o modelo treinado
metricas = modelo.avaliar(X_teste, Y_teste)
print(f"Acurácia do modelo: {metricas['acuraccy']:.2f}%")
print(f"Precisão do modelo: {100*metricas['precision']:.2f}%")
print(f"Recall do modelo: {100*metricas['recall']:.2f}%")
print(f"F1-Score do modelo: {100*metricas['f1']:.2f}%")
print(f"Perda do modelo: {100*metricas['loss']:.2f}%")
print(f"Matriz de Confusão do modelo: {metricas['matriz_confusao']}")