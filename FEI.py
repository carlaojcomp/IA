import zipfile
import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

extracao_dir = 'imagens_extraidas/'


def salvar_csv(array1, array2, colunas_array1, colunas_array2, nome_arquivo):


    # Criar um DataFrame com os arrays como colunas
    df = pd.DataFrame({
        colunas_array1[0]: array1,  # A primeira coluna
        colunas_array2[0]: array2  # A segunda coluna
    })

    # Salvar o DataFrame em um arquivo CSV
    df.to_csv(nome_arquivo, index=False)


# Listar as imagens extraídas
imagens = [f for f in os.listdir(extracao_dir) if f.endswith('.jpg')]
# Inicializar lista de arrays e classe
imagens_array = []
imagens_array32 = []
imagens_array64 = []
imagens_pixels = []
im64_pixels = []
im32_pixels = []
classe = []

# Converter e redimensionar as imagens para 128x128
for imagem_nome in imagens:
    # Caminho completo da imagem
    imagem_path = os.path.join(extracao_dir, imagem_nome)

    # Abrir a imagem
    imagem = Image.open(imagem_path)

    # Fazer downsampling para 128x128
    imagem_redimensionada = imagem.resize((128, 128))
    imagem64 = imagem.resize((64, 64))
    imagem32 = imagem.resize((32, 32))

    # Converter para array NumPy
    imagem_array = np.array(imagem_redimensionada)
    imagem_flat = imagem_array.flatten()

    # Adicionar à lista de pixels das imagens
    imagens_pixels.append(imagem_flat)
    im64 = np.array(imagem64)
    im64_flat = im64.flatten()
    im64_pixels.append(im64_flat)
    im32 = np.array(imagem32)
    im32_flat = im32.flatten()
    im32_pixels.append(im32_flat)

    # Adicionar o array à lista de imagens
    imagens_array.append(imagem_array)
    imagens_array64.append(im64)
    imagens_array32.append(im32)

    # Verificar a classe com base no nome da imagem
    if 'a' in imagem_nome:
        classe.append(0)  # Se 'a' no nome, classe 0
    elif 'b' in imagem_nome:
        classe.append(1)  # Se 'b' no nome, classe 1

# Converter a lista de classes para um array NumPy

df_imagens = pd.DataFrame(im32_pixels)

df_imagens['classe'] = classe
df_imagens.to_csv('FEI32.csv', index=False)

# Exibir a classe das imagens e o array da primeira imagem
print(classe)
plt.imshow(imagens_array32[0], cmap='gray')
plt.title("Imagem MNIST Redimensionada")
plt.axis('off')  # Desativa os eixos
plt.show()

