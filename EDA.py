from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
import kagglehub
from PIL import Image

path = kagglehub.dataset_download("misrakahmed/vegetable-image-dataset")    #faz o download do dataset e devolve o caminho dele no sistema

treino_pasta = os.path.join(path, "Vegetable Images", "train")  #caminho da pasta TRAIN
validacao_pasta = os.path.join(path, "Vegetable Images", "validation")  #caminho da pasta VALIDATION
teste_pasta = os.path.join(path, "Vegetable Images", "test")    #caminho da pasta TEST

treino_dataset = ImageFolder(root=treino_pasta)     #cria pastas para facilitar a manipulacao dos dados (tipo usar .classes pra achar as classes)
validacao_dataset = ImageFolder(root=validacao_pasta)
teste_dataset = ImageFolder(root=teste_pasta)

classes = treino_dataset.classes    #retorna todas as classes do dataset de treino

figura, vetor_de_subplot = plt.subplots(3, 5, figsize=(15, 10))    #numero de linhas, numero de colunas, tamanho da figura
                                                                   #retorna a figura e um array de objetos

vetor_de_subplot = vetor_de_subplot.ravel() #transforma um array multidimensional (igual o caso de "vetor_de_subplot") p/ um unidimensional

for i, classe in enumerate(classes):
    classe_pasta = os.path.join(treino_pasta, classe)   #pega o caminho de cada classe (por exemplo: o caminho p/ a classe "Bean" ou "Carrot")

    nome_primeira_imagem = os.listdir(classe_pasta)[0]   #pega o NOME da primeira imagem de cada classe
                                                    #por exemplo: a primeira imagem da classe "Bean" tem o nome de "0026.jpg"

    caminho_primeira_imagem = os.path.join(classe_pasta, nome_primeira_imagem)  #pega o caminho da primeira imagem

    primeira_imagem = Image.open(caminho_primeira_imagem)   #carrega a primeira imagem de cada classe
    vetor_de_subplot[i].imshow(primeira_imagem) # carrega as imagens nos subplots correspondentes
    vetor_de_subplot[i].set_title(classe)

plt.show()  # ploto o grafico