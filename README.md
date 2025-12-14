# CNN_vegetais

## Descrição

Projeto de **classificação de imagens de vegetais** utilizando Redes Neurais Convolucionais (CNN) com PyTorch. O Projeto implementa tanto uma CNN desenvolvida do zero quanto uma abordagem de transfer learning usando ResNet-18 pré-treinada, comparando desempenho entre os modelos.

## Estrutura do Projeto

O projeto segue um pipeline completo de Deep Learning:

- **Aquisição dos dados** via API do Kaggle. Que já vinham organizados em treino, validação e teste.  
- **Pré-processamento e Data Augmentation** para filtrar qualquer tipo de imagem corrompida, tratamento de ruído e afins para o conjunto de treino.  
- **Carregamento dos dados tratados**
- **Modelagem**:
  - CNN construída do zero com 5 camadas convolucionais e 1 linear, e a função RELU como ativação para a CNN.
  - ResNet-18 pré-treinada.
- **Treinamento e avaliação** com métricas de Loss, Acurácia e F1-Score.  
- **Visualizações Gráficas** das curvas de treino e matriz de confusão (tanto do modelo construído quanto da ResNet-18.  
- **Salvamento dos modelos treinados** na pasta "modelos"

## Como rodar Código no Google Colab

1. Baixe o .zip do código no github:
   - `Code` -> `Download zip`
2. Acesse o **Google Colab**: https://colab.research.google.com e selecione "upload"
4. Vá para onde baixou o projeto e faça upload do arquivo "testes.ipynb"
5. Já dentro do notebook "testes" vá para a pasta de arquivos do Google Colab e adicione os dois arquivos .pth que estão dentro da pasta "modelos" de onde baixou o projeto.
6. Adicione novas fotos ao Notebook e altere o nome delas em "imagem_para_testar" e rode o notebook novamente.

## Observações 

Para visualizar a comparação entre a CNN desenvolvida e a ResNet-18 é necessário abrir o Notebook chamado "Modelling.ipynb". Lá estão todas as métricas, comparações e Matriz de Confusão da CNN desenvolvida. 
