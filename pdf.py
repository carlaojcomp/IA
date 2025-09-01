import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def gaussiana(n):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * n ** 2)
def parzen(x, amostras, h):
    N = len(amostras)
    densidade = np.zeros_like(x, dtype=float)
    for i, val in enumerate(x):
        kernels = np.sum(gaussiana((val - amostras) / h))
        densidade[i] = (1 / (N * h)) * kernels
    return densidade

def knn(x, amostras, k):
    N = len(amostras)
    densidade = np.zeros_like(x, dtype=float)
    for i, val in enumerate(x):
            distancias = np.abs(amostras - val)
            ordenado = np.sort(distancias)
            dk = ordenado[k - 1]
            vk = 2 * dk
            densidade[i] = k / (N * vk)
    return densidade
media = 2.0
std = 0.307
total = 2000
# Gerar as amostras da distribuição normal
dados = np.random.randn(total)
# Aplicar a transformação para obter a média e o desvio padrão desejados
amostras = media + std * dados
x = np.linspace(amostras.min() - 0.5, amostras.max() + 0.5, 1000)
h = 0.1
k = 400
densidade_parzen = parzen(x, amostras, h)
densidade_knn = knn(x, amostras, k)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plotar o Estimador de Parzen no primeiro subplot
ax1.plot(x, densidade_parzen, label=f'Estimador de Parzen (h={h})')
ax1.set_title('Estimador de Densidade via Janela de Parzen')
ax1.set_xlabel('x')
ax1.set_ylabel('p(x)')
ax1.grid(True)
ax1.legend()

# Plotar o Estimador de KNN no segundo subplot
ax2.plot(x, densidade_knn, label=f'Estimador de KNN (k={k})')
ax2.set_title('Estimador de Densidade KNN')
ax2.set_xlabel('x')
ax2.set_ylabel('p(x)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
