---
title: Document AI Models
date: 2023-12-12
tags: ["DocumentAI"]
math: true
author: ["Camille Barboule"]
categories: ["my_researches"]
---

---
<h2 style="font-weight: bold; color: orange; font-family: 'VotrePolice'; border-bottom: 1px solid;">Multimodal Document Information Retrieval (Focus on documents with texts, graphs, diagrams and tables)</h2>

:rocket: The subject handled here is called "Document Information Extraction" (IE), a sub-category of "Document Understanding" (DU).

:orange_book: Keywords: Document AI, Information Extraction, Intelligent Document Processing, Multimodal Document Processing

Documents are a core part of many businesses in many fields such as law, finance, and technology among others. Automatic understanding of documents such as invoices, contracts, and resumes is lucrative, opening up many new avenues of business. The fields of natural language processing and computer vision have seen tremendous progress through the development of deep learning such that these methods have started to become infused in contemporary document understanding systems ([A Survey of Deep Learning Approaches for OCR and Document Understanding, Subramani et al, 2021](https://arxiv.org/pdf/2011.13534.pdf)).

[TOC]

# 1. What does Document Understanding mean?
End-to-end document understanding systems present in the literature integrate multiple deep neural network architectures for both reading and comprehending a document’s content:
- A computer-vision based document layout analysis module, which partitions each document page into distinct content regions. This model not only delineates between relevant and irrelevant regions, but also serves to categorize the type of content it identifies.
<img src="doc_layout_analysis.PNG" width="500"/>

- An optical character recognition (OCR) model, whose purpose is to locate and faithfully transcribe all written text present in the document. Straddling the boundary between CV and NLP, OCR models may either use document layout analysis directly or solve the problem in an independent fashion.
<img src="ocr_process.PNG" width="500"/>

- Information extraction models that use the output of OCR or document layout analysis to comprehend and identify relationships between the information that is being conveyed in the document. Usually specialized to a particular domain and task, these models provide the structure necessary to make a document machine readable, providing utility in document understanding. Document extraction of information by humans goes beyond simply reading text on a page as it is often necessary to learn page layouts for complete understanding. As such, recent enhancements have extended text encoding strategies for documents by additionally encoding structural and visual information of text in a variety of ways.
    - 2D Positional Embeddings: Multiple sequence tagging approaches have been proposed by embedding attributes of 2D bounding boxes and merging them with text embeddings to create models which are simultaneously aware of both context and spatial positioning when extracting information. While these strategies have seen success, relying solely on the line number or bounding box coordinates can be misleading when the document has been scanned on an uneven surface, leading to curved text. Additionally, bounding box based embeddings still miss critical visual information such as typographical emphases (bold, italics) and images such as logos. **à creuser**
    - Image Embeddings: Information extraction for documents can also be framed as a computer vision challenge wherein the goal of the model is to semantically segment information or regress bounding boxes over the areas of interest. This strategy helps preserve the 2D layout of the document and allows models to take advantage of 2D correlations.  In these cases, an encoding function is applied onto a proposed textual level (i.e. character, token, word) to create individual embedding vectors. These vectors are transposed into each pixel that comprises the bounding box corresponding to the embedded text, ultimately creating an image of W × H × D where W is the width, H is the height, and D is the embedding dimension. **à creuser**
    - Documents as Graphs: Unstructured text on documents can also be represented as graph networks, where the nodes in a graph represent different textual segments. Two nodes are connected with an edge if they are cardinally adjacent to each other, allowing the relationship between words to be modeled directly. An encoder such as a BiLSTM encodes text segments into nodes [GraphIE: A Graph-Based Framework for Information Extraction](https://arxiv.org/pdf/1810.13083.pdf). Edges can be represented as a binary adjacency matrix or a richer matrix, encoding additional visual information such as the distance between segments or shape of the source and target nodes [Graph Convolution for Multimodal Information Extraction from Visually Rich Documents](https://arxiv.org/pdf/1903.11279.pdf). A graph convolutional network is then applied at different receptive fields in a similar fashion to dilated convolutions to ensure that both local and global information can be learned. After this, the representation is passed to a sequence tagging decoder. The most recent model in this field is [Doc2Graph: A Task Agnostic Document Understanding Framework Based on Graph Neural Networks](https://link.springer.com/chapter/10.1007/978-3-031-25069-9_22)

# 2. Focus on Extracting informations from documents
Extraction of information from documents includes many tasks and problems from basic OCR [PP-OCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/pdf/2009.09941.pdf),[A Detailed Analysis of Optical Character Recognition Technology](https://www.researchgate.net/publication/311851325_A_Detailed_Analysis_of_Optical_Character_Recognition_Technology),[A Survey on Optical Character Recognition System](https://arxiv.org/ftp/arxiv/papers/1710/1710.05703.pdf),[Handwritten Optical Character Recognition (OCR): A Comprehensive Systematic Literature Review (SLR)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9151144), [Text Detection Forgot About Document OCR](https://arxiv.org/pdf/2210.07903.pdf),[An Overview of the Tesseract OCR Engine](https://www.researchgate.net/publication/4288303_An_Overview_of_the_Tesseract_OCR_Engine) up to visual question answering (VQA) [InfographicVQA](https://arxiv.org/pdf/2104.12756.pdf), [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/pdf/2007.00398.pdf):
- Key Information Extraction (KIE) [LAMBERT: Layout-Aware Language Modeling for Information Extraction](https://arxiv.org/pdf/2002.08087.pdf) [ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction](https://arxiv.org/pdf/2103.10213.pdf),[Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts](https://arxiv.org/pdf/2105.05796.pdf) aims to extract pre-defined key information (categories of ”fields” – name, email, the amount due, etc.) from a document. A number of datasets for KIE are publicly available [ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction](https://arxiv.org/pdf/2103.10213.pdf), [Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts](https://arxiv.org/pdf/2105.05796.pdf), [Deepform: Extract information from documents (2020)](https://wandb.ai/deepform/political-ad-extraction),[Spatial Dual-Modality Graph Reasoning for Key Information Extraction](https://arxiv.org/pdf/2103.14470.pdf), [Towards Robust Visual Information Extraction in Real World: New Dataset and Novel Solution](https://arxiv.org/pdf/2102.06732.pdf). However, as noted by [Business Document Information Extraction: Towards Practical Benchmarks](https://arxiv.org/pdf/2206.11229.pdf), most of them are relatively small and contain only a few annotated field categories.
- Key Information Localization and Extraction (KILE) [Business Document Information Extraction: Towards Practical Benchmarks](https://arxiv.org/pdf/2206.11229.pdf) additionally requires precise localization of the extracted information in the input image or PDF, which is crucial for human-in-the-loop interactions, auditing, and other processing of the documents. However, many of the existing KIE datasets miss the localization annotations. Due to the lack of large-scale datasets for KILE from business documents, noted by several authors, many research publications use private datasets .
- Line Item Recognition (LIR) is a part of table extraction that aims at finding Line Items (LI), localizing and extracting key information for each item.  The task is related to Table Structure Recognition, which typically aims at detecting table rows, columns and cells. However, sole table structure recognition is not sufficient for LIR: an enumerated item may span several rows in a table; and columns are often not sufficient to distinguish all semantic information. There are several  atasets for Table Detection and/or Structure Recognition, PubTables-1M being the largest with a million tables from scientific articles. The domain of scientific articles is prevailing among the datasets, due to easily obtainable annotations from the LATEX source codes. However, there is a non-trivial domain shift introduced by the difference in the Tables from scientific papers and business documents.
<img src="datasets_KILE_LIR.PNG" width="500"/>

- Named Entity Recognition (NER) is the task of assigning one of the pre-defined categories to entities (usually words or word-pieces in the document) which makes it strongly related to KILE and LIR, especially when these entities have a known location. Note that the task of NER is less general as it only operates on word/token level, and using it to solve KILE is not straightforward,
as the classified tokens have to be correctly aggregated into fields and fields do not necessarily have to contain whole word-tokens.

## 2.1. Datasets for Information Extraction from documents

### 2.1.1. taset for Key Information Localization and Extraction (KILE) and Line Item Recognition (LIR)
[DocILE: Document Information Localization and Extraction Benchmark](https://github.com/rossumai/docile) is a large-scale research benchmark for cross-evaluation of machine learning methods for Key Information Localization and Extraction (KILE) and Line Item Recognition (LIR) from semi-structured business documents such as invoices, orders etc. Such large-scale benchmark was previously missing (Skalický et al., 2022), hindering comparative evaluation. DocILE is the largest dataset of business documents with KILE and LIR labels for 106680 documents (6680 annotated and 100k synthetic) and almost 1M unlabeled documents for unsupervised pre-training.
<img src="docice_LIR.PNG" width="500"/>
<img src="docile_KILE.PNG" width="500"/>

### 2.1.2. Datasets for QA on documents
Here is a summary of the datasets for QA on documents:
<img src="docqa.PNG" width="500"/>
<img src="docqa2.PNG" width="500"/>
<img src="docqa3.PNG" width="500"/>


#### [DocVQA](https://paperswithcode.com/dataset/docvqa) 
DocVQA est un ensemble de données pour le Visual Question Answering (VQA) sur les images de documents, comprenant 50 000 questions définies sur plus de 12 000 images de documents. Il présente un défi majeur pour les modèles en termes de compréhension de la structure du document. Les modèles comme LayoutLM se sont révélés performants sur ce benchmark, offrant des perspectives pratiques prometteuses dans la compréhension des documents.

#### [InfographicVQA](https://arxiv.org/pdf/2104.12756.pdf)
The dataset comprises 30 035 questions over 5 485 images. Questions in the dataset include questions grounded on tables, figures and visualizations as well as questions that require combining multiple cues. Since most infographics contain numerical data, they collected questions that require elementary reasoning skills such as counting, sorting and arithmetic operations. 
<img src="infographicvqa.PNG" width="500"/>
Here are the results on this dataset:
<img src="res_infographicvqa.PNG" width="500"/>
<img src="res2_infographicvqa.PNG" width="500"/>

#### [Document Understanding Dataset and Evaluation (DUDE)](https://arxiv.org/pdf/2305.08455.pdf)
The dataset DUDE was released in the scope of The ICDAR 2023 competition on Document UnderstanDing of Everything took place from February to May of 2023. In this competition, given an input consisting of a PDF with multiple pages and a natural language question, the objective is to provide a natural language answer together with an assessment of the answer confidence (a float value scaled between 0 and 1). Each unique document is annotated with multiple questions of different types, including extractive, abstractive, list, and non-answerable. Annotated QA pairs are not restricted to the answer being explicitly present in the document. Instead, any question on aspect, form, or visual/layout appearance relative to the document under review is allowed.
The dataset [jordyvl/DUDE_loader](https://huggingface.co/datasets/jordyvl/DUDE_loader) is composed of:
<img src="dude.PNG" width="500"/>

- A training-validation set with 30 K QA annotations on 3.7 K documents was given to participants at the beginning of February.
- The 11.4K questions on 12.1 K documents for the test set were only made accessible for a window between March and May.
- For each document of the dataset, there are 3 different OCR versions of the document: one using Tesseract, another using Amazon OCR tool, and another using Azure OCR tool. 
<img src="dude_compo.PNG" width="500"/>

- Analysis of the results of the competition:
<img src="competition_docvqa.PNG" width="500"/>
<img src="competition_docvqa2.PNG" width="500"/>
<img src="competition_docvqa3.PNG" width="500"/>

    - **When answers at the end of the document**: Who is the president and vice-chancellor? Despite the question’s relatively straightforward nature, some systems struggle with providing the appropriate answer. One can hypothesize it is the result of limited context (the answer is located at the end of the document), i.e., models either hallucinate a value or provide a name found earlier within the document.
    - **When answer requires graphical comprehension**: Which is the basis for jurisdiction? To provide a valid answer, the model needs to comprehend the meaning of the form field and recognize the selected checkbox. **None of the participating systems was able to spot the answer correctly.**
    - **When answer requires comparison**: In which year does the Net Requirement exceed 25,000? The question requires comprehending a multi-page table and spotting if any values fulfill the posed condition. Some of the models resort to plausible answers (one of the three dates that the document covers), whereas others correctly decide there is no value exceeding the provided amount.
    - **When answer requires arithmetic**: What is the difference between how much Operator II and Operator III make per hour? The question requires table comprehension, determining rele- vant values, and dividing extracted integers. None of the participating models was able to fulfill this requirement. 
    - **When answer requires counting and list output**: What are the first two behavioral and intellectual disabilities of people with FASDs? It seems most of the models correctly recognized that this type of question requires a list answer but either failed to comprehend the question or provided a list with incorrect length (incomplete or with too many values). 
- Results above the competition on DUDE:
<img src="res_dude.PNG" width="500"/>

#### [MPDocVQA: multipage DocVQA](https://arxiv.org/pdf/2212.05935.pdf)
MPDocVQA is a DocVQA dataset where questions are posed over multi-page documents instead of single pages.
<img src="mp-docvqa.PNG" width="500"/>
MP-DocVQA is an extension of the DocVQA dataset where the questions are posed on documents with between 1 and 20 pages. Here is a more detailed comparison of MPDocVQA vs DocVQA
<img src="mpdocvqa.PNG" width="500"/>
Here is the github page of the project: https://github.com/rubenpt91/MP-DocVQA-Framework

### 2.1.3. Datasets for QA on Charts
[PlotQA](https://github.com/NiteshMethani/PlotQA) est conçu pour relever les défis spécifiques liés au raisonnement sur des graphiques scientifiques, un domaine essentiel de la Visual Question Answering (VQA). PlotQA se distingue par ses 28,9 millions de paires de questions-réponses, basées sur 224 377 graphiques issus de sources réelles et accompagnées de questions élaborées à partir de modèles de questions crowdsourcées.
Ce dataset s'adresse aux défis non couverts par des ensembles de données synthétiques antérieurs comme FigureQA et DVQA. Les datasets existants ne présentent pas de variabilité dans les étiquettes de données, ni de données à valeurs réelles, ni de questions de raisonnement complexes. Ainsi, les modèles conçus pour ces datasets ne répondent pas pleinement aux défis de raisonnement sur les graphiques, en particulier parce qu'ils se limitent à des réponses provenant d'un vocabulaire fixe de petite taille ou d'une zone de sélection dans l'image. En pratique, de nombreuses questions nécessitent un raisonnement et, par conséquent, des réponses à valeurs réelles qui n'apparaissent ni dans un vocabulaire fixe ni dans l'image. PlotQA vise à combler ce fossé en offrant un environnement de test plus réaliste pour les modèles de VQA.
Environ 80,76 % des questions hors vocabulaire (OOV) dans PlotQA ont des réponses qui ne font pas partie d'un vocabulaire fixe, ce qui représente un défi significatif pour les modèles existants. Ces derniers montrent une précision globale en un seul chiffre sur PlotQA, ce qui n'est pas surprenant étant donné qu'ils n'ont pas été conçus pour de telles questions. En réponse, PlotQA propose une approche hybride : certaines questions sont traitées en sélectionnant une réponse à partir d'un vocabulaire fixe ou en extrayant la réponse d'une boîte de sélection prédite dans le graphique, tandis que d'autres questions sont traitées avec un moteur de question-réponse sur tableau, alimenté par un tableau structuré généré en détectant des éléments visuels dans l'image.
PlotQA se positionne donc comme un dataset crucial pour faire progresser la recherche et le développement de modèles de traitement automatique des langues et de vision par ordinateur, notamment dans le domaine complexe du raisonnement sur les données visuelles.

### 2.1.4. Datasets for QA on tables
[TabFact](https://tabfact.github.io/) et [WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions) sont spécifiquement conçus pour tester la capacité des modèles à répondre à des questions basées sur des tableaux et des données structurées.


## 2.2. Models for information extraction in documents
### 2.2.1. Models dedicated to information extraction in documents
#### LayoutLM : Document Foundation Model
Les modèles de type [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) sont des modèles de pointe développés par Microsoft pour la compréhension de documents visuellement riches, intégrant à la fois des éléments textuels et des informations de mise en page.

LayoutLM est une méthode de pré-entraînement multi-modale simple mais efficace pour la compréhension de documents et l'extraction d'informations. Il combine des techniques de traitement du langage naturel (NLP) et de vision par ordinateur (CV) pour modéliser conjointement les interactions entre le texte et les informations de mise en page dans les images de documents numérisés. Cela le rend particulièrement utile pour des tâches telles que la compréhension de formulaires et la compréhension de reçus.

[LayoutLMv3](https://github.com/microsoft/unilm/blob/master/layoutlmv3/README.md), quant à lui, est une évolution de LayoutLM qui vise à pré-entraîner des Transformateurs multimodaux pour l'IA documentaire avec un masquage unifié du texte et de l'image. Ce modèle est pré-entraîné avec un objectif d'alignement mot-patch pour apprendre l'alignement cross-modal en prédisant si le patch d'image correspondant à un mot de texte est masqué. La simplicité de son architecture unifiée et de ses objectifs de formation font de LayoutLMv3 un modèle pré-entraîné polyvalent pour les tâches centrées sur le texte et l'image dans l'IA documentaire. LayoutLMv3 a démontré des performances de pointe non seulement dans les tâches centrées sur le texte, telles que la compréhension de formulaires et de reçus, ainsi que la réponse visuelle aux questions sur les documents, mais aussi dans les tâches centrées sur l'image comme la classification des images de documents et l'analyse de la mise en page des documents.

LayoutLMv3 extrait le texte des documents (chaque page du document étant convertie en image) à l'aide d'un outil d'OCR, ici Tesseract. Le layout de la page du document est quant à lui analysé à l'aide d'un Visual Transformer, qui utilise une projection linéaire de "patches". Le texte issu de l'OCR est alors converti en tokens, et les patches de l'image sont également considérés comme des tokens. Ces éléments sont ensuite transformés en représentations vectorielles contextuelles. Cette représentation vectorielle est ensuite donnée à un Transformer qui est adaptée pour traiter à la fois du texte et des images dans les documents. Le modèle peut prendre une image de document ainsi que des informations de texte et de positionnement (layout) correspondantes. Ce-dernier est entraîné avec les objectifs de pré-entraînements suivants:
- Modélisation de Langue Masquée (MLM) : Cela implique de masquer certains mots du texte et d'entraîner le modèle à les prédire. C'est une méthode courante pour le pré-entraînement des modèles de langage.
- Modélisation d'Image Masquée (MIM) : De manière similaire à MLM, mais appliquée aux images, où certaines parties de l'image sont masquées et le modèle est entraîné pour les reconstruire.
- Alignement Mot-Patch (Word-Patch Alignment, WPA) : Un objectif supplémentaire pour apprendre l'alignement entre les modalités de texte et d'image. Le modèle tente de prédire si un patch d'image correspondant à un mot de texte donné est masqué.

<img src="layoutlmv3.PNG" width="500"/>

LayoutLMv3 peut être comparé aux modèles plus anciens de Document AI tels que DocFormer et SelfDoc:

<img src="other_models.PNG" width="500"/>

LayoutLMv3 utilise des "patches" linéaires pour l'intégration d'images. Cette approche vise à réduire le "goulot d'étranglement" computationnel associé aux réseaux de neurones convolutionnels (CNN). En d'autres termes, plutôt que de s'appuyer sur des CNN complexes pour traiter les images, LayoutLMv3 utilise une méthode plus simple et moins gourmande en ressources. De plus, cette méthode élimine la nécessité de superviser des détecteurs d'objets régionaux pendant l'entraînement, ce qui simplifie le processus d'apprentissage du modèle.
En ce qui concerne les objectifs de pré-entraînement pour la modalité image, LayoutLMv3 est conçu pour apprendre à reconstruire des tokens d'image discrets à partir de patches masqués, plutôt que de se concentrer sur les pixels bruts ou les caractéristiques régionales. Cela permet au modèle de capturer les structures de mise en page de haut niveau au lieu de se concentrer sur les détails bruyants. En d'autres termes, plutôt que de se perdre dans les détails minutieux de l'image, LayoutLMv3 se concentre sur la compréhension globale de la disposition et de la structure des éléments dans les documents.

#### DiT : Document Image Transformer
[DiT: Self-supervised Pre-training for Document Image Transformer](https://arxiv.org/pdf/2203.02378.pdf) released a Transformer dedicated for Document AI tasks. It is a self-supervised pre-training of Vision Transformers (ViTs) trained on Document images.
The Transformer works this way : Following ViT, they use the vanilla Transformer architecture as the backbone of DiT. they divide a document image into nonoverlapping patches and obtain a sequence of patch embeddings. After adding the 1d position embedding, these image patches are passed into a stack of Transformer blocks with multi-head attention. Finally, they take the output of the Transformer encoder as the representation of image patches. 
<img src="dit.PNG" width="500"/>
<img src="dit2.PNG" width="500"/>

In detail:
- An input text image is first resized into 224 × 224 and then the image is split into a sequence of 16 × 16 patches which are used as the input to the image Transformer
- Distinct from the [BEiT](https://arxiv.org/pdf/2106.08254.pdf) model where visual tokens are from the discrete VAE in DALL-E [Zero-Shot Text-to-Image Generation](https://arxiv.org/pdf/2102.12092.pdf), they re-train the discrete VAE (dVAE) model with large-scale document images, so that the generated visual tokens are more domain relevant to the Document AI tasks. 
- The pre-training objective is to recover visual tokens from dVAE based on the corrupted input document images using the Masked Image Modeling (MIM) in [BEiT](https://arxiv.org/pdf/2106.08254.pdf). In this way, the DiT model does not rely on any human-labeled document images, but only leverages large-scale unlabeled data to learn the global patch relationship within each document image.
- They fine-tune our model on four Document AI benchmarks, including the [RVL-CDIP dataset](https://arxiv.org/pdf/1502.07058.pdf) for document image classification, the [PubLayNet dataset](https://arxiv.org/pdf/1908.07836.pdf) for document layout analysis, the [ICDAR 2019 cTDaR dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978120) for table detection, and the [FUNSD dataset](https://arxiv.org/pdf/1905.13538.pdf) for text detection. 
- They evaluate the pre-trained DiT models on four publicly available Document AI benchmarks, including the [RVL-CDIP dataset](https://arxiv.org/pdf/1502.07058.pdf) for document image classification, the [PubLayNet dataset](https://arxiv.org/pdf/1908.07836.pdf) for document layout analysis, the [ICDAR 2019 cTDaR dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8978120) for table detection, as well as the [FUNSD dataset](https://arxiv.org/pdf/1905.13538.pdf) for OCR text detection.

#### Nougat : document reconstruction
[Nougat](https://github.com/facebookresearch/nougat) est quant à lui un modèle de Document Reconstruction sans OCR. Le modèle se base en effet sur les [Swin Transformers](https://arxiv.org/pdf/2103.14030.pdf), qui est un type spécifique de Visual Transformer, qui lui permet de traiter des patchs de toute petite taille (ressemblant à un pixel) sans augmenter la complexité computationnelle du modèle. En effet, cela est rendu possible grâce à l'approche "Shifted Window" du visual transformer: on découpe l'image (la page du document) en tout petits patchs, et la self-attention n'est calculée qu'entre chaque patchs d'une même window: le modèle possède une complexité de calcul linéaire par rapport à la taille de l'image d'entrée, grâce au calcul de self-attention uniquement au sein de chaque fenêtre locale (indiquée en rouge):
<img src="swin_transformer.PNG" width="500"/>

A chaque couche du modèle, la fenêtre se déplace de 2 patchs à droite et 2 patchs en bas. ainsi, in-fine, chaque patch a une visibilité sur tous les autres patchs. Cela permet d'avoir des patchs beaucoup plus petits et donc de se passer de l'OCR pour reconstruire le texte. 
Le Swin Transformer construit des cartes de caractéristiques hiérarchiques en fusionnant les patches d'image (représentés en gris) dans des couches plus profondes. Il peut donc servir de base générale pour les tâches de classification d'images et de reconnaissance dense. En revanche, les Transformers de vision précédents produisent des cartes de caractéristiques d'une seule basse résolution et ont une complexité de calcul quadratique par rapport à la taille de l'image d'entrée, en raison du calcul de self-attention globalement:
<img src="swin_transformer2.PNG" width="500"/>

Nougat a été entraîné sur un dataset de papiers de recherches arxiv (le papier en format d'origine (latex) a été téléchargé sur arxiv, et le pdf correspondant au papier). Le images ont été enlevées du papier (pdf & latex) et le modèle a appris à reconstruire  le latex à partir du pdf. Ainsi, le modèle que l'on trouve sur https://github.com/facebookresearch/nougat n'est adapté que pour reconstruire des papiers de recherche. Il faut fine-tuner le modèle sur d'autres types de documents pour pouvoir l'utiliser pour reconstruire des ppt, et autre.

D'autres méthodes, sans fine-tuning, permettent d'utiliser Nougat pour reconstruire tout type de documents: c'est le cas de [marker](https://github.com/VikParuchuri/marker) qui applique Nougat, et quand le processing ne fonctionne pas, applique des outils d'OCR tels que tesseract pour reconstruire le document. C'est un "bidouillage" de Nougat.

#### Hi-VT5: QA on Document
[Hierarchical multimodal transformers for Multi-Page DocVQA](https://arxiv.org/pdf/2212.05935.pdf) is a DocVQA model. Existing work on DocVQA only considers single-page documents. However, in real scenarios documents are mostly composed of multiple pages that should be processed altogether. Hi-VT5 extend DocVQA to the multipage scenario.

It is a hierarchical transformer architecture (T5-base) where the encoder summarizes the most relevant information of every page and then, the decoder takes this summarized information to generate the final answer.
The model is capable to naturally process multiple pages by extending the input sequence length up to 20480 tokens without increasing the model complexity. 

It works this way:
<img src="hivt5.PNG" width="500"/>

- The encoder processes separately each page of the document, providing a summary of the most relevant information conveyed by the page conditioned on the question. This information is encoded in a number of special [PAGE] tokens, inspired in the [CLS] token of the BERT model
- The decoder generates the final answer by taking as input the concatenation of all these summary [PAGE] tokens for all pages.
- The model includes an additional head to predict the index of the page where the answer has been found.

#### DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding
[DocLLM: A Layout-Aware Generative Language Model for Multimodal Document Understanding](https://arxiv.org/pdf/2401.00908.pdf) has been released the 31st of December 2023 by JPMorgan. 
As opposed to all model above, here, they do not use any Vision Encoder. They only treat spatial informations of bounding boxes of the document to incorporate the spatial layout structure information of the document to the transformers. They get these bounding boxes through OCR. As opposed to other methods doing this, involving either concatenating spatial and textual embeddings ([Unifying Vision, Text, and Layout for Universal Document Processing, May 2023](https://arxiv.org/pdf/2212.02623.pdf)) or summing the two ([LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/pdf/1912.13318.pdf)), here, they do not treat the spatial information as a distinct modality and compute its inter-dependency with the text modality in a disentangled manner, like what was presented in [Connecting What to Say With Where to Look by Modeling Human Attention Traces](https://arxiv.org/pdf/2105.05964.pdf). What they do is extending the self-attention mechanism of standard transformers to include new attention scores that capture cross-modal relationships. 

Also, although they use a causal architecture (possible because no Vision Encoder), they do not use a classical next token prediction objective during the self-supervised pre-training phase since it can be restrictive. In particular, the preceding tokens may not always be relevant due to the diverse arrangements of text, which can be positioned horizontally, vertically, or even in a staggered manner. To tackle this issue, they propose two modifications to the pre-training objective: 
- adopting cohesive blocks of text that account for broader contexts
- implementing an infilling approach by conditioning the prediction on both preceding and succeeding tokens. 
Due to these modifications, the model is better equipped to address misaligned text, contextual completions, intricate layouts, and mixed data types. This solution has been constructed by [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/pdf/2103.10360.pdf). Autoregressive Infilling. There are two main autoregressive infilling approaches: 
- “fill-in-the-middle” (FIM) where a single span is sampled
- “blank infilling” with multiple spans.
How to divide the document into segments? 
- The OpenAI FIM approach ([Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/pdf/2207.14255.pdf)) uses the template (prefix, middle, suffix) to divide a document into three segments. Next, these segments are reorganized into (prefix, suffix, middle), enabling the model to predict the middle segment. This process relies on three special tokens, [PRE], [SUF], and [MID], which structure a document as: [PRE] prefix [SUF] suffix [MID] middle. The [MID] token denotes the start for prediction, while the other two special tokens guide the model on where to infill. This method demonstrates that autoregressive models can learn to infill text where the middle part is missing.
- Fill-in Language Model (FiLM) ([FILM: FILL-IN LANGUAGE MODELS FOR ANY-ORDER GENERATION](https://arxiv.org/pdf/2310.09930.pdf)) is a subsequent development that enables flexible generation at arbitrary positions, unconstrained by a predefined generation order.
- Approaches like [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/pdf/2103.10360.pdf)  sample multiple spans for infilling. For each blank to be infilled, a pair of special tokens is used: [blank_mask] and [start_to_fill]. The multiple spans not only require special tokens but also global indicators to distinguish which middle span the model should infill. This global indicator is implemented with 1D token positions, ensuring that each pair of the two special tokens, i.e., [blank_mask] and [start_to_fill], share the same positions.
How is the attention computed?
- Disentangled attention is introduced in the [DeBERTa](https://arxiv.org/pdf/2006.03654.pdf) model where token embeddings and relative positional encodings were kept separate rather than summed together, and each used independently when computing attention weights using disentangled matrices. The motivation behind this was to facilitate the learning of decoupled attention alignments based on content and position separately. 
- In DocLLM, the input is represented as x = {(xi , bi)} T i=1, where bi = (left, top, right, bottom) is the bounding box corresponding to xi . To capture the new modality (i.e. spatial information), we encode the bounding boxes into hidden vectors represented by S ∈ R T×d . We then decompose the attention matrix computation into four different scores, namely text-to-text, text-to-spatial, spatial-to-text and spatial-to-spatial.


During the Fine-tuning phase, the pre-trained model is fine-tuned using a large-scale instruction dataset, covering four core document intelligence tasks:
- Visual Question Answering
- Natural Language Inference
- Key Information Extraction
- Document Classification

<img src="docllm_archi.PNG" width="500"/>

**to complete**

### 2.2.2. Handling documents as images: how does it work?
#### ViT: Visual Transformer
**to complete**
#### SwinViT: hierarchical Visual Transformer whose representation is computed with shifted windows.
Les [Swin Transformers](https://arxiv.org/pdf/2103.14030.pdf), qui est un type spécifique de Visual Transformer, qui lui permet de traiter des patchs de toute petite taille (ressemblant à un pixel) sans augmenter la complexité computationnelle du modèle. En effet, cela est rendu possible grâce à l'approche "Shifted Window" du visual transformer: on découpe l'image (la page du document) en tout petits patchs, et la self-attention n'est calculée qu'entre chaque patchs d'une même window: le modèle possède une complexité de calcul linéaire par rapport à la taille de l'image d'entrée, grâce au calcul de self-attention uniquement au sein de chaque fenêtre locale (indiquée en rouge):
<img src="swin_transformer.PNG" width="500"/>

A chaque couche du modèle, la fenêtre se déplace de 2 patchs à droite et 2 patchs en bas. ainsi, in-fine, chaque patch a une visibilité sur tous les autres patchs. Cela permet d'avoir des patchs beaucoup plus petits et donc de se passer de l'OCR pour reconstruire le texte. 
Le Swin Transformer construit des cartes de caractéristiques hiérarchiques en fusionnant les patches d'image (représentés en gris) dans des couches plus profondes. Il peut donc servir de base générale pour les tâches de classification d'images et de reconnaissance dense. En revanche, les Transformers de vision précédents produisent des cartes de caractéristiques d'une seule basse résolution et ont une complexité de calcul quadratique par rapport à la taille de l'image d'entrée, en raison du calcul de self-attention globalement:
<img src="swin_transformer2.PNG" width="500"/>

#### [Generative Pretraining from Pixels](https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf)
trained a sequence Transformer called iGPT to auto-regressively predict pixels without incorporating knowledge of the 2D input structure, which is the first attempt at self-supervised image transformer pre-training. After that, self-supervised pre-training for image Transformer became a hot topic in computer vision. 
**to complete**
#### [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.14294.pdf)
proposed DINO, which pre-trains the image Transformer using self-distillation with no labels. 
**to complete**
#### [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/pdf/2104.02057.pdf)
proposed MoCov3 that is based on Siamese networks for self-supervised learning 
**to complete**
#### [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf)
adopted a BERT-style pre-training strategy, which first tokenizes the original image into visual tokens, then randomly masks some image patches and feeds them into the backbone Transformer. Similar to the masked language modeling, they proposed a masked image modeling task as the pre-training objective that achieves SOTA performance.
**to complete**
#### [IBOT : IMAGE BERT PRE-TRAINING WITH ONLINE TOKENIZER](https://arxiv.org/pdf/2111.07832.pdf)
presented a self-supervised framework iBOT that can perform masked prediction with an online tokenizer. The online tokenizer is jointly learnable with the MIM objective and dispenses with a multi-stage pipeline where the tokenizer is pre-trained beforehand.
**to complete**

### 2.2.3. Multimodal LLMs
#### End-To-End-Approaches
In such approaches, there is a unified architecture handling different modalities, **composed of a Visual Encoder and a LLM as a core**. Most of these approaches use a pretrained vision encoder and a pretrained language models. 

The LLM can be of any type. However, the visual encoder is not random: it must be trained with an alignment of the visual and language modalities, because:
- Alignment ensures that the visual and textual information are semantically coherent. For example, if an image shows a cat, the vision encoder must be able to convey this information to the LLM in a way that the LLM recognizes it as a cat and generates relevant text.
- For training a multimodal model, it is important that data from different modalities are aligned in a way that the model can learn correspondences between them.

End-to-end approaches MLLMs are constructed like this:
<img src="projection_matrix.PNG" width="500"/>

##### Visual encoder
Visual encoders in multimodal models are typically **either based on Convolutional Neural Networks (CNNs)**, like the ResNet architecture, **or on Transformer-based architectures** known as Visual Transformers. These models are designed to process and understand visual information, transforming raw image data into structured, feature-rich representations. 
- CNNs are known for their deep layers of convolution operations. They are adept at extracting hierarchical visual features from images
- Visual Transformers, inspired by the success of Transformers in NLP, apply the self-attention mechanism to process images. They treat an image as a sequence of patches and learn to focus on different parts of an image, capturing complex relationships and contextual information within the image.

Then, **these visual encoders need to be trainined with a text - image alignment**, as explained below. Here are the **3 ways of aligning the 2 modalities**, and thus training the Visual Transformer:
###### Visual encoder having a Bi-Encoder Architecture
A bi-encoder architecture for visual encoders involves **two separate encoders**: one for processing text and another for processing images. These encoders **operate independently to encode their respective inputs** into a shared embedding space, where the **similarity between text and image representations can be measured**. This is how these visual encoders are trained: by **Constrastive Learning**: In the training set, there are 
batches of N pairs of similar image, text. The encoders are jointly trained to maximize the cosine similarity of these pairs, while for the batches of N² - N pairs of in correct pairs (image, text), the encoders are jointly trained to minimize the cosine similarity of these pairs.
- [CLIP](https://arxiv.org/pdf/2103.00020.pdf), for "Contrastive Language-Image Pretraining" was released by OpenAI and is a prime example of a bi-encoder architecture. It uses a Transformer for text and a ResNet (or Vision Transformer) for images. CLIP is trained using a contrastive learning approach, where the model learns to match corresponding text and image pairs among a batch of non-corresponding pairs.
- [FLORENCE](https://arxiv.org/pdf/2111.11432.pdf), introduced by Microsoft, is another model that employs a bi-encoder architecture. It leverages large-scale pretraining across various data sources to learn a universal visual representation.
- [ALIGN](https://arxiv.org/pdf/2102.05918.pdf), standing for "A Large-scale Image and Noisy-Text Embedding", is also a model that utilizes a bi-encoder structure with an EfficientNet-based image encoder and a BERT-based text encoder. It's trained on a large dataset of noisy text-image pairs, making it adept at handling real-world, uncurated data.
###### Visual encoder having an Encoder-Decoder Architecture
An encoder-decoder architecture in visual encoders typically involves an **encoder module to process the image** and a **decoder module to generate corresponding textual output**. This architecture is common in tasks that require the generation of text from images, such as **image captioning**.
- [SimVLM](https://arxiv.org/pdf/2108.10904.pdf), standing for "Simple Vision Language Model" uses an encoder-decoder architecture where the encoder processes visual inputs and the decoder generates textual descriptions. It employs a 'prefix language modeling' objective for pretraining, allowing it to understand and generate natural language descriptions of images effectively.
- [VirTex](https://arxiv.org/pdf/2006.06666.pdf), standing for "Vision and Text", is a model that uses a CNN encoder to process images and a Transformer-based decoder to generate textual descriptions. It's trained on a dataset of images and captions, learning to generate accurate and contextually relevant text descriptions for a given image.
###### Visual encoder having both Architectures: Bi-Encoder + Encoder-Decoder
Some visual encoders take the advantage of both techniques: they are trained using both contrastive learning and cross-modality prediction. This approach combines the **strengths of contrastive methods** like [CLIP](https://arxiv.org/pdf/2103.00020.pdf) and **generative methods** like [SimVLM](https://arxiv.org/pdf/2108.10904.pdf):
- [CoCa](https://arxiv.org/pdf/2205.01917.pdf), standing for "Contrastive Loss and Captioning Loss", is a model from Google that uses this type of dual approaches for the visual encoder. The contrastive loss is applied between unimodal image and text embeddings, while the captioning loss is used on the multimodal decoder outputs, which predict text tokens autoregressively.

##### Communication between Visual Encoder's output and the LLM
How, there is a **need to make the output of the visual encoder and the LLM communicate**: and this is not direct! For the LLM decoder to effectively understand and interact with data from the vision encoder, it is necessary for the representations generated by the encoder to be in a **format or context that is comprehensible to the LLM**. This means that the visual data must be transformed into a representation that makes sense in the linguistic domain. There are 2 ways of doing this link between the visual encoder output and the LLM:
###### Some models here take a frozen visual encoder, as well as a frozen LLM, and use a projection matrix to make the connection between the output of the vision encoder and the LLM

- [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) fuses vision and language modalities with **gated cross-attention** showing impressive few-shot capabilities (= projection matrix). 
- [BLIP-2](https://arxiv.org/pdf/2301.12597.pdf) uses a **Q-Former** to align the visual features from the frozen visual encoder and large language models.
- [MiniGPT-4](https://arxiv.org/pdf/2304.10592.pdf) uses a **pretrained Q-Former (from BLIP-2)**. Then, they **train a linear layer** on top of this. This linear layer aims at transforming the features outputed from Q-Former into a format that the Vicuna language model can understand and process as if they were linguistic inputs, to make this output interract with the LLM Vicuna. Then, the **linear layer is instruct-tune to adapt the model to visual instructions**.

###### Other models also have a projection matrix to modify the output of the visual encoder, but they then fine-tune the LLM in order to make it understand the visual encoder's output
Thanks to fine-tuning / instruct-tuning, the models are exposed to tasks or datasets that are both text and visual inputs. The LLM is thus fine-tuned to understand and relate the outputs of the visual encoder to their language understanding. This **fine-tuning helps the model learn how to integrate and interpret visual data alongside textual information**:
- [LLaVA](https://arxiv.org/pdf/2304.08485.pdf) utilizes a **straightforward linear layer as projection matrix** to connect the visual encoder's output to the language model. The simplicity of a linear projection layer means that it lacks the sophistication to handle complex relations or nuanced translations between visual features and language concepts. To compensate for this, LLaVA undergoes a fine-tuning process. During **fine-tuning, the language model is specifically trained to better understand and interpret these visual tokens**. This training involves exposing the model to tasks or datasets that include both visual and textual data, enabling it to learn how to effectively combine and make sense of these two types of information.

###### Other models also have a projection matrix, but fine-tune in the pretraining stage the visual encoder in order to make its output adapted to the LLM's input expected
- [mPLUG-Owl](https://arxiv.org/pdf/2304.14178.pdf) has a **projection matrix called "visual abstractor", that it trains**. But it also **fine-tunes its Vision Transformer (ViT-L/14) to adapt its output to the LLM**. This ViT is initialized from the CLIP ViT-L/14 model, fine-tuned using using image-caption pairs from several datasets, such as LAION-400M, COYO-700M, Conceptual Captions, and MSCOCO.

#### Systematic-Collaboration-Approaches
In such approaches, **LLMs act as the agents in such systems**, and are **prompted to select the appropriate experts and tools** for visual understanding / modality understanding. So these systems employ collaboration with various specialized models to handle tasks beyond the scope of traditional text-based models, particularly in the visual domain. Here are 3 examples of Systematic-Collaboration-Approaches systems:
##### Systems using a prompt manager between LLM & Visual Foundation Models (VFMs)
- [Visual ChatGPT](https://ar5iv.labs.arxiv.org/html/2303.04671) is a system that **integrates Visual Foundation Models (VFMs) with ChatGPT**, enabling it to interact not only through text but also with images : ChatGPT is not modified, nor are VFMs, and so, to allow ChatGPT to interract with VFMs, there is a **Prompt Manager in between**, which bridges the gap between ChatGPT and VFMs by informing ChatGPT about the capabilities of each VFM, and converting visual information into a language format for ChatGPT, and finally managing histories, priorities, and conflicts among different VFMs. So this process involves multiple steps which are guided by the Prompt Manager: 
<img src="visual_chatgpt.PNG" width="500"/>

- [HuggingGPT](https://ar5iv.labs.arxiv.org/html/2303.17580) **connects large language models like ChatGPT with various AI models** from the Hugging Face community to solve multimodal AI tasks through 4 stages: **Task Planning, Model Selection, Task Execution, and Response Generation**. In this concept, an LLM acts as a controller, managing and organizing the cooperation of expert models. The LLM first plans a list of tasks based on the user request and then assigns expert models to each task. After the experts execute the tasks, the LLM collects the results and responds to the user: 

<img src="hugginggpt.PNG" width="500"/>

    - In Task Planning, ChatGPT analyzes user requests and disassembles them into solvable tasks. 
    - During Model Selection, it chooses appropriate models from Hugging Face based on their descriptions. 
    - In Task Execution, these models are invoked and executed, and their results are returned to ChatGPT. 
    - In Response Generation, ChatGPT integrates all model predictions and generates user responses.

##### Systems modifying LLM to generate specific action-requests tokens calling VFMs
- [MM-REACT](https://ar5iv.labs.arxiv.org/html/2303.11381) is a system that empowers ChatGPT to process and understand multimodal information through action requests. Indeed, in MM-REACT, **ChatGPT is instructed to say specific watchwords in action request if a vision expert is required** to interpret the visual inputs. The action request is composed of textual prompts that represent the expert name called and file names for visual signals. Regular expression matching is applied to parse the expert’s name and the file path, which are then used to call the vision expert (action execution): 

<img src="mm_react.PNG" width="500"/>