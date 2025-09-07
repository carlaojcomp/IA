import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
base = pd.read_csv('FEI64.csv')
print(base.head())


def gerar_pontos(media, covariancia, amostras):
    #Escalonamento (fator de escala baseado na raiz quadrada dos eigenvalores da covariância)

    # Decomposição espectral (autovalores e autovetores)
    autovalores, autovetores = np.linalg.eigh(covariancia)
    autovalores = np.maximum(autovalores, 0)
    #Gerar amostras unitárias com distribuição normal padrão (média 0 e variância 1)
    unitarias = np.random.randn(amostras, len(media))

    #Aplicar a transformação(Rotação e Escalonamento) para obter as amostras com a distribuição desejada
    transformados = unitarias.dot(np.sqrt(np.diag(autovalores))).dot(autovetores.T)

    # Passo 4: Translação (deslocar para o vetor de média)
    final = transformados + media

    return final
# Gerar o primeiro array com valores entre 2 e 6 (inclusive)
# Usando np.linspace para garantir valores distintos
arrayn1_X = np.linspace(2, 6, 1000, dtype=float)
arrayn1_Y = np.linspace(0, 4, 1000, dtype=float)
arrayn2_X = np.linspace(-4, 0, 1000, dtype=float)
arrayn2_Y = np.linspace(-2, -6, 1000, dtype=float)
soman1_X = np.sum((arrayn1_X - 4)**2)
soman1_Y = np.sum((arrayn1_Y - 2.5)**2)
soman1_XY = np.sum((arrayn1_X - 4) * (arrayn1_Y - 2.5))

covn1_x = soman1_X / 1000
covn1_y = soman1_Y / 1000
covn1_XY = soman1_XY / 1000

# Cálculo da covariância para o segundo conjunto
soman2_X = np.sum((arrayn2_X + 2.5)**2)
soman2_Y = np.sum((arrayn2_Y + 4.2)**2)
soman2_XY = np.sum((arrayn2_X + 2.5) * (arrayn2_Y + 4.2))

covn2_x = soman2_X / 1000
covn2_y = soman2_Y / 1000
covn2_XY = soman2_XY / 1000

# Matrizes de covariância
matrizn1_cov = [[covn1_x, covn1_XY], [covn1_XY, covn1_y]]
matrizn2_cov = [[covn2_x, covn2_XY], [covn2_XY, covn2_y]]


# Médias dos pontos
mediasn1 = [4, 2.5]
mediasn2 = [-2.5, -4.2]
samples = gerar_pontos(mediasn1, matrizn1_cov, 1000)
amostras = gerar_pontos(mediasn2, matrizn2_cov, 1000)
'''
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plotar 'samples' no primeiro subgráfico
axes[0].scatter(samples[:, 0], samples[:, 1], color='blue', alpha=0.5)
axes[0].set_title('Nuvem de pontos - N1')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True)
axes[0].set_xlim(-8, 8)
axes[0].set_ylim(-8, 8)

# Plotar 'amostras' no segundo subgráfico
axes[1].scatter(amostras[:, 0], amostras[:, 1], color='blue', alpha=0.5)
axes[1].set_title('Nuvem de pontos - N2')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(True)
axes[1].set_xlim(-8, 8)
axes[1].set_ylim(-8, 8)

# Exibir o gráfico com os dois plots
plt.tight_layout()
plt.show()'''
