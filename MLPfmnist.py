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

fmnist = pd.read_csv('fmnist_3000.txt')
fmnist.rename(columns={'0\t0\t0\t0\t1\t2\t0\t0\t0\t0\t0\t114\t183\t112\t55\t23\t72\t102\t165\t160\t28\t0\t0\t0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t24\t188\t163\t93\t136\t153\t168\t252\t174\t136\t166\t130\t123\t131\t66\t0\t0\t1\t0\t0\t0\t0\t0\t0\t2\t0\t10\t157\t216\t226\t208\t142\t66\t115\t149\t230\t190\t196\t198\t172\t222\t107\t165\t211\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t118\t214\t174\t168\t109\t200\t124\t150\t143\t58\t63\t89\t137\t97\t168\t138\t142\t195\t156\t0\t0\t0\t0\t0\t0\t0\t0\t25\t140\t70\t80\t43\t71\t96\t93\t151\t121\t197\t143\t107\t82\t101\t111\t80\t137\t193\t208\t6\t0\t0\t0\t0\t0\t0\t0\t74\t194\t107\t146\t178\t185\t182\t77\t185\t218\t210\t175\t174\t235\t217\t217\t129\t180\t210\t208\t89\t0\t0\t0\t0\t0\t0\t0\t179\t213\t203\t177\t228\t192\t193\t162\t143\t172\t196\t205\t181\t180\t140\t134\t176\t194\t171\t170\t65\t0\t0\t0\t0\t0\t0\t0\t184\t194\t229\t209\t176\t198\t129\t227\t225\t140\t196\t130\t179\t145\t109\t79\t182\t223\t164\t195\t233\t0\t0\t0\t0\t0\t0\t38\t180\t177\t213\t202\t159\t129\t98\t179\t149\t90\t187\t211\t61\t134\t91\t57\t118\t212\t220\t218\t207\t0\t0\t0\t0\t0\t0\t114\t154\t142\t182\t219\t130\t88\t81\t52\t54\t106\t93\t110\t159\t222\t227\t83\t117\t253\t218\t210\t206\t48\t0\t0\t0\t0\t0\t18\t127\t208\t228\t185\t172\t240\t91\t126\t208\t165\t154\t213\t214\t229\t215\t175\t222\t204\t153\t130\t125\t39\t0\t0\t0\t0\t0\t0\t0\t0\t28\t0\t212\t228\t170\t221\t205\t225\t228\t210\t178\t214\t89\t117\t213\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t196\t141\t56\t176\t210\t164\t162\t91\t135\t196\t158\t69\t181\t0\t0\t3\t2\t3\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t150\t190\t88\t50\t145\t194\t159\t120\t136\t207\t230\t144\t171\t4\t0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t187\t220\t182\t72\t139\t199\t192\t232\t255\t244\t198\t170\t189\t2\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t186\t175\t181\t93\t164\t230\t134\t153\t142\t137\t79\t143\t183\t0\t0\t2\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t93\t72\t144\t138\t164\t113\t124\t98\t80\t57\t97\t138\t124\t4\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t149\t171\t98\t70\t180\t152\t227\t173\t143\t180\t183\t127\t206\t38\t0\t5\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t0\t195\t210\t226\t113\t187\t224\t210\t191\t181\t224\t212\t198\t172\t36\t0\t6\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t0\t153\t197\t171\t175\t161\t171\t199\t224\t187\t206\t192\t176\t179\t48\t0\t6\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t180\t195\t136\t230\t240\t148\t204\t140\t173\t199\t193\t156\t213\t37\t0\t5\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t150\t101\t72\t167\t158\t95\t177\t234\t113\t142\t112\t59\t152\t22\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t197\t178\t82\t47\t64\t106\t112\t121\t110\t189\t225\t121\t98\t58\t0\t4\t0\t0\t0\t0\t0\t0\t0\t0\t0\t2\t0\t0\t202\t219\t161\t135\t205\t200\t156\t195\t231\t234\t218\t182\t223\t99\t0\t6\t0\t0\t0\t0\t0\t0\t0\t0\t0\t4\t0\t0\t188\t152\t118\t222\t214\t203\t233\t226\t193\t200\t173\t53\t166\t97\t0\t6\t0\t0\t0\t0\t0\t0\t0\t0\t0\t3\t0\t2\t182\t152\t51\t89\t174\t183\t168\t112\t109\t181\t170\t136\t108\t60\t0\t4\t0\t0\t0\t0\t0\t0\t0\t0\t0\t2\t0\t5\t194\t193\t204\t104\t116\t241\t217\t196\t171\t249\t207\t197\t202\t45\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t0\t22\t21\t25\t69\t52\t45\t74\t39\t3\t0\t0\t0\t0\t1\t0\t0\t0\t0': "A"}, inplace=True)
fmnist.loc[fmnist.size] = '0\t0\t0\t0\t1\t2\t0\t0\t0\t0\t0\t114\t183\t112\t55\t23\t72\t102\t165\t160\t28\t0\t0\t0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t24\t188\t163\t93\t136\t153\t168\t252\t174\t136\t166\t130\t123\t131\t66\t0\t0\t1\t0\t0\t0\t0\t0\t0\t2\t0\t10\t157\t216\t226\t208\t142\t66\t115\t149\t230\t190\t196\t198\t172\t222\t107\t165\t211\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t118\t214\t174\t168\t109\t200\t124\t150\t143\t58\t63\t89\t137\t97\t168\t138\t142\t195\t156\t0\t0\t0\t0\t0\t0\t0\t0\t25\t140\t70\t80\t43\t71\t96\t93\t151\t121\t197\t143\t107\t82\t101\t111\t80\t137\t193\t208\t6\t0\t0\t0\t0\t0\t0\t0\t74\t194\t107\t146\t178\t185\t182\t77\t185\t218\t210\t175\t174\t235\t217\t217\t129\t180\t210\t208\t89\t0\t0\t0\t0\t0\t0\t0\t179\t213\t203\t177\t228\t192\t193\t162\t143\t172\t196\t205\t181\t180\t140\t134\t176\t194\t171\t170\t65\t0\t0\t0\t0\t0\t0\t0\t184\t194\t229\t209\t176\t198\t129\t227\t225\t140\t196\t130\t179\t145\t109\t79\t182\t223\t164\t195\t233\t0\t0\t0\t0\t0\t0\t38\t180\t177\t213\t202\t159\t129\t98\t179\t149\t90\t187\t211\t61\t134\t91\t57\t118\t212\t220\t218\t207\t0\t0\t0\t0\t0\t0\t114\t154\t142\t182\t219\t130\t88\t81\t52\t54\t106\t93\t110\t159\t222\t227\t83\t117\t253\t218\t210\t206\t48\t0\t0\t0\t0\t0\t18\t127\t208\t228\t185\t172\t240\t91\t126\t208\t165\t154\t213\t214\t229\t215\t175\t222\t204\t153\t130\t125\t39\t0\t0\t0\t0\t0\t0\t0\t0\t28\t0\t212\t228\t170\t221\t205\t225\t228\t210\t178\t214\t89\t117\t213\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t196\t141\t56\t176\t210\t164\t162\t91\t135\t196\t158\t69\t181\t0\t0\t3\t2\t3\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t150\t190\t88\t50\t145\t194\t159\t120\t136\t207\t230\t144\t171\t4\t0\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t187\t220\t182\t72\t139\t199\t192\t232\t255\t244\t198\t170\t189\t2\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t186\t175\t181\t93\t164\t230\t134\t153\t142\t137\t79\t143\t183\t0\t0\t2\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t93\t72\t144\t138\t164\t113\t124\t98\t80\t57\t97\t138\t124\t4\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t149\t171\t98\t70\t180\t152\t227\t173\t143\t180\t183\t127\t206\t38\t0\t5\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t0\t195\t210\t226\t113\t187\t224\t210\t191\t181\t224\t212\t198\t172\t36\t0\t6\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t0\t153\t197\t171\t175\t161\t171\t199\t224\t187\t206\t192\t176\t179\t48\t0\t6\t1\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t180\t195\t136\t230\t240\t148\t204\t140\t173\t199\t193\t156\t213\t37\t0\t5\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t150\t101\t72\t167\t158\t95\t177\t234\t113\t142\t112\t59\t152\t22\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t197\t178\t82\t47\t64\t106\t112\t121\t110\t189\t225\t121\t98\t58\t0\t4\t0\t0\t0\t0\t0\t0\t0\t0\t0\t2\t0\t0\t202\t219\t161\t135\t205\t200\t156\t195\t231\t234\t218\t182\t223\t99\t0\t6\t0\t0\t0\t0\t0\t0\t0\t0\t0\t4\t0\t0\t188\t152\t118\t222\t214\t203\t233\t226\t193\t200\t173\t53\t166\t97\t0\t6\t0\t0\t0\t0\t0\t0\t0\t0\t0\t3\t0\t2\t182\t152\t51\t89\t174\t183\t168\t112\t109\t181\t170\t136\t108\t60\t0\t4\t0\t0\t0\t0\t0\t0\t0\t0\t0\t2\t0\t5\t194\t193\t204\t104\t116\t241\t217\t196\t171\t249\t207\t197\t202\t45\t0\t3\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t0\t0\t22\t21\t25\t69\t52\t45\t74\t39\t3\t0\t0\t0\t0\t1\t0\t0\t0\t0'
fmnist = fmnist['A'].str.split(expand=True)
fmnist = converter_para_float(fmnist)
classe = fmnist[0].to_numpy()
#Dropa os rótulos
fmnist = fmnist.drop(0, axis=1)
X = fmnist.to_numpy()
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
'''Acurácia no conjunto de teste FINAL: 81.00%
Acurácia no conjunto de treinamento FINAL: 99.71%'''



#Segunda Execução
'''Acurácia no conjunto de teste FINAL: 82.22%
Acurácia no conjunto de treinamento FINAL: 99.14%
'''


#Terceira Execução
'''Acurácia no conjunto de treinamento FINAL: 100.00%
Acurácia no conjunto de teste FINAL: 89.33%'''