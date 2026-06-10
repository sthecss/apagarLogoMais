# Classificação de Imagens Histológicas Orais usando EfficientNetV2 & Grad-CAM

Este projeto visa a classificação e análise de imagens de histopatologia oral utilizando a arquitetura de Deep Learning **EfficientNetV2**, acompanhada de mapas de ativação clássicos (**Grad-CAM**) para explicabilidade do modelo (interpretação das regiões visuais que motivaram a classificação tecidual).

Para garantir a reprodutibilidade, o projeto está preparado para rodar nativamente em duas plataformas: [**Kaggle**](https://www.kaggle.com/code/sthephannysantos/efficientnetv2-grad-cam-oral-histopathology) e [**Google Colab**](https://colab.research.google.com/drive/1N39yI880RAvK7c_z1Fi1FFBHAhcJn_24?usp=sharing).

---

## Autores

<div align="center">

| | |
|:---:|:---:|
| [![Sthephanny](https://github.com/sthecss.png?size=80)](https://github.com/sthecss) | [![Victor](https://github.com/VictorBertolini.png?size=80)](https://github.com/VictorBertolini) |
| [Sthephanny](https://github.com/sthecss) | [Victor](https://github.com/VictorBertolini) |

</div>

---

## 📌 Sumário

- [Descrição dos Datasets](#-descrição-dos-datasets)
- [Arquiteturas Utilizadas](#-arquiteturas-utilizadas)
- [Modos de Treinamento](#️-modos-de-treinamento)
- [Estratégia de Data Augmentation](#-estratégia-de-data-augmentation)
- [Desenho Experimental](#-desenho-experimental)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Links e Downloads (Releases)](#-links-e-downloads-releases)
- [Como Executar o Projeto](#-como-executar-o-projeto)
- [Guia de Execução Passo a Passo](#-guia-de-execução-passo-a-passo)
- [Análise e Interpretação dos Resultados](#-análise-e-interpretação-dos-resultados)
- [Limitações Conhecidas](#️-limitações-conhecidas)

---

## Descrição dos Datasets

### Dataset 1 — OralEpitheliumDB
Desenvolvido pela Universidade Federal de Uberlândia (UFU). Focado em Displasia Epitelial Oral (OED), com imagens microscópicas de tecido bucal com potencial de transformação maligna. As imagens foram coletadas de 30 línguas de camundongo, com núcleos celulares marcados manualmente por especialista e rótulos avaliados e validados por patologista.

- **Total de imagens:** 456
- **Tarefa:** Classificação binária
- **Subconjunto utilizado:** `Original ROI Images`
- **Fonte:** [LIPAI-Org/OralEpitheliumDB_Dataset](https://github.com/LIPAI-Org/OralEpitheliumDB_Dataset)

**Classes e divisão dos splits:**

| Split | healthy | severe | Total |
| :--- | :---: | :---: | :---: |
| Treino | 72 | 72 | 144 |
| Validação | 19 | 19 | 38 |
| Teste | 23 | 23 | 46 |
| **Total** | **114** | **114** | **228** |

<br>

### Dataset 2 — NDB-UFES
Desenvolvido pela Universidade Federal do Espírito Santo (UFES). Contém imagens histopatológicas de biópsias de pacientes atendidos pelo projeto de Diagnóstico Oral (NDB) entre 2010 e 2021. Além das imagens, o dataset inclui dados sociodemográficos (gênero, idade, cor da pele) e clínicos (tabagismo, consumo de álcool, exposição solar) — utilizados apenas como contexto, não no treinamento.

- **Total de imagens:** 237
- **Tarefa:** Classificação multiclasse (3 classes)
- **Fonte:** [Mendeley Data - NDB-UFES](https://data.mendeley.com/datasets/bbmmm4wgr8/4)

**Classes e divisão dos splits:**

| Split | Leukoplakia w/ dysplasia | Leukoplakia w/o dysplasia | OSCC | Total |
| :--- | :---: | :---: | :---: | :---: |
| Treino | 59 | 39 | 61 | 159 |
| Validação | 15 | 9 | 15 | 39 |
| Teste | 15 | 9 | 15 | 39 |
| **Total** | **89** | **57** | **91** | **237** |

> [!NOTE]
> Os splits de ambos os datasets são fixos e compartilhados para permitir a comparação direta dos resultados entre diferentes execuções.

---

## Arquiteturas Utilizadas

Ambas as arquiteturas são carregadas via biblioteca [TIMM (Pytorch Image Models)](https://github.com/huggingface/pytorch-image-models):

- [EfficientNetV2-B0](https://huggingface.co/timm/tf_efficientnetv2_b0.in1k) (`tf_efficientnetv2_b0.in1k`)
- [EfficientNetV2-B1](https://huggingface.co/timm/tf_efficientnetv2_b1.in1k) (`tf_efficientnetv2_b1.in1k`)

Cada modelo é composto por um **backbone** (responsável pela extração automática de características visuais) e uma **camada classificadora fully connected** (responsável pela predição das classes). Os pesos pré-treinados na ImageNet são baixados automaticamente pelo `timm` na primeira execução.

---

## Modos de Treinamento

Para cada arquitetura e dataset, três abordagens de transferência de aprendizado foram avaliadas:

| Modo | Nome | Descrição |
| :---: | :--- | :--- |
| `FS` | From Scratch | Todos os pesos inicializados aleatoriamente; treinamento do zero absoluto. |
| `PT-FC` | Pré-treinado + Backbone Congelado | Pesos pré-treinados carregados; backbone congelado; apenas a camada classificadora final é treinada. |
| `PT-ALL` | Pré-treinado + Fine-tuning Completo | Pesos pré-treinados carregados; todas as camadas (backbone e classificador) são otimizadas. |

---

## Estratégia de Data Augmentation

As transformações de aumento de dados são aplicadas **exclusivamente no conjunto de treino**. Validação e teste utilizam apenas o redimensionamento padrão e a normalização.

* **Sem Augmentation:** `Resize(224, 224)` + Normalização ImageNet.
* **Com Augmentation:**
  - `Resize(256)` + `RandomCrop(224)`
  - `RandomHorizontalFlip` (probabilidade de 50%)
  - `RandomRotation` sorteando estritamente entre 0°, 90°, 180° ou 270° (25% de chance cada)
  - Normalização padrão ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

---

## Desenho Experimental

```text
2 arquiteturas × 3 modos × 2 condições de augmentation × 2 datasets × 3 seeds = 72 execuções

```

* **Seeds utilizadas:** `42`, `123`, `2025`

Para mitigar o efeito da aleatoriedade, cada configuração é executada 3 vezes (uma por seed). As métricas finais reportadas consistem na **média** e no **desvio padrão** obtidos nas repetições.

**Protocolo de Hiperparâmetros:**

| Parâmetro | Valor |
| --- | --- |
| Dimensão de Entrada | 224 × 224 px |
| Épocas Máximas | 50 |
| Otimizador | Adam |
| Taxa de Aprendizado (LR) | 1e-4 |
| Tamanho do Batch | 32 |
| Função de Perda | Cross Entropy |
| Early Stopping | Patience = 10 |
| Critério de Seleção | Maior Acurácia de Validação |


> [!IMPORTANT]
> **Tempo estimado de execução** Utilizando GPU T4 x2 no Kaggle 2h40, Utilizando GPU T4 no Google 4h50.

---

## Estrutura do Repositório

O código fonte e os recursos visuais estão organizados da seguinte forma:

```text
├── img/
│   ├── help/             # Imagens do passo a passo para execução
│   └── analitic/         # Gráficos e imagens para interpretação dos resultados
├── notebooks/
│   ├── analysis_oral_histopathology.ipynb               # Versão otimizada para Google Colab
│   └── efficientnetv2-grad-cam-oral-histopathology.ipynb # Versão otimizada para Kaggle
└── README.md

```

> [!WARNING]
> **Nota sobre arquivos pesados:** Arquivos de dados e resultados volumosos não são armazenados diretamente no histórico do Git. Eles estão hospedados de forma segura na aba de **[Releases](pipipipopopoRELEASES)** deste repositório.

---

## Links e Downloads (Releases)

Caso opte por rodar o projeto localmente ou inspecionar as saídas brutas, faça o download direto pelos assets da Release:

* **[datasets.zip](githublinkzimlogomais):** O conjunto de dados unificado utilizado em ambas as plataformas.
* **[resultados.zip](googleresultados):** Checkpoints, pesos salvos e logs gerados a partir do ambiente do **Google Colab**.
* **[results.zip](kaggleresultados):** Outputs, logs de treino e matrizes geradas a partir do ambiente do **Kaggle**.

---

## Como Executar o Projeto

Escolha uma das plataformas abaixo para reproduzir ou estender a análise:

### Opção 1: Ambiente Kaggle (Recomendado)

A execução no Kaggle é direta, pois o ambiente e o dataset já estão integrados e hospedados na própria plataforma:

* 🔗 **Notebook Completo:** [EfficientNetV2 + Grad-CAM Oral Histopathology](https://www.kaggle.com/code/sthephannysantos/efficientnetv2-grad-cam-oral-histopathology)
* 🔗 **Dataset Original:** [Oral Dataset no Kaggle](https://www.kaggle.com/datasets/sthephannysantos/oral-dataset)

### Opção 2: Google Colab

Para rodar no Colab, siga as etapas:

1. Acesse a pasta `notebooks/` deste repositório e baixe o arquivo `analysis_oral_histopathology.ipynb`.
2. Faça o upload do arquivo no [Google Colab](https://colab.research.google.com/).
3. Baixe o arquivo `datasets.zip` na nossa aba de **Releases** e faça o upload no ambiente do Colab.
4. *(Opcional)*: Se quiser apenas analisar as plotagens sem precisar realizar todo o treinamento do modelo (72 execuções), baixe e extraia o arquivo `resultados.zip` diretamente no ambiente.

---

## Guia de Execução Passo a Passo

Siga o guia visual caso tenha dúvidas sobre como configurar o ambiente em cada site:

### Ambiente Google Colab (10 Passos)

| Passo | Descrição | Visualização |
| --- | --- | --- |
| **1** | [Descrição do passo 1 do Colab] |  |
| **2** | [Descrição do passo 2 do Colab] |  |
| **3** | [Descrição do passo 3 do Colab] |  |
| **4** | [Descrição do passo 4 do Colab] |  |
| **5** | [Descrição do passo 5 do Colab] |  |
| **6** | [Descrição do passo 6 do Colab] |  |
| **7** | [Descrição do passo 7 do Colab] |  |
| **8** | [Descrição do passo 8 do Colab] |  |
| **9** | [Descrição do passo 9 do Colab] |  |
| **10** | [Descrição do passo 10 do Colab] |  |

### Ambiente Kaggle (3 Passos)

| Passo | Descrição | Visualização |
| --- | --- | --- |
| **1** | [Descrição do passo 1 do Kaggle] |  |
| **2** | [Descrição do passo 2 do Kaggle] |  |
| **3** | [Descrição do passo 3 do Kaggle] |  |

---

## Análise e Interpretação dos Resultados

Nesta seção são consolidadas as principais métricas obtidas e a interpretação visual gerada pelo Grad-CAM. As imagens analíticas completas podem ser consultadas na pasta `img/analitic/`.

### Matriz de Confusão e Curvas de Aprendizado

Os gráficos de perda (loss) e acurácia demonstram o comportamento do modelo ao longo das 50 épocas. A convergência e a capacidade de generalização variaram sensivelmente entre os modos `PT-ALL` e `FS`.

### Visualização de Regiões de Interesse (Grad-CAM)

O uso do Grad-CAM permitiu mapear quais estruturas celulares e texturas teciduais foram determinantes para que a EfficientNetV2 categorizasse as patologias orais. Áreas hipercoradas e regiões com alterações morfológicas severas no epitélio tendem a registrar maior peso nos mapas de calor.

---

## Limitações Conhecidas

* O dataset **NDB-UFES** apresenta um desbalanceamento acentuado de classes (especialmente na classe `Leukoplakia without dysplasia`), o que pode impactar negativamente a métrica de F1-score macro.
* O limite estrito de 50 épocas pode ser insuficiente para a convergência plena nos testes realizados sob o modo `From Scratch (FS)`, principalmente no cenário multiclasse.
* Risco iminente de sobreajuste (overfitting) nos cenários de ajuste fino (`PT-ALL`) aplicados ao **OralEpitheliumDB**, decorrente do tamanho amostral reduzido do dataset (228 imagens totais compartilhadas).

```
