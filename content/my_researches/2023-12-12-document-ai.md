---
title: Document AI Models
date: 2023-12-12
tags: ["DocumentAI", "Nougat", "OCR", "Information Retrieval", "LayoutLM"]
math: true
author: ["Camille Barboule"]
categories: ["my_researches"]
---

---

# <b>1) LayoutLM : Document Foundation Model</b>
Les modèles de type [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) sont des modèles de pointe développés par Microsoft pour la compréhension de documents visuellement riches, intégrant à la fois des éléments textuels et des informations de mise en page.

LayoutLM est une méthode de pré-entraînement multi-modale simple mais efficace pour la compréhension de documents et l'extraction d'informations. Il combine des techniques de traitement du langage naturel (NLP) et de vision par ordinateur (CV) pour modéliser conjointement les interactions entre le texte et les informations de mise en page dans les images de documents numérisés. Cela le rend particulièrement utile pour des tâches telles que la compréhension de formulaires et la compréhension de reçus.

[LayoutLMv3](https://github.com/microsoft/unilm/blob/master/layoutlmv3/README.md), quant à lui, est une évolution de LayoutLM qui vise à pré-entraîner des Transformateurs multimodaux pour l'IA documentaire avec un masquage unifié du texte et de l'image. Ce modèle est pré-entraîné avec un objectif d'alignement mot-patch pour apprendre l'alignement cross-modal en prédisant si le patch d'image correspondant à un mot de texte est masqué. La simplicité de son architecture unifiée et de ses objectifs de formation font de LayoutLMv3 un modèle pré-entraîné polyvalent pour les tâches centrées sur le texte et l'image dans l'IA documentaire. LayoutLMv3 a démontré des performances de pointe non seulement dans les tâches centrées sur le texte, telles que la compréhension de formulaires et de reçus, ainsi que la réponse visuelle aux questions sur les documents, mais aussi dans les tâches centrées sur l'image comme la classification des images de documents et l'analyse de la mise en page des documents.

LayoutLMv3 extrait le texte des documents (chaque page du document étant convertie en image) à l'aide d'un outil d'OCR, ici Tesseract. Le layout de la page du document est quant à lui analysé à l'aide d'un Visual Transformer, qui utilise une projection linéaire de "patches". Le texte issu de l'OCR est alors converti en tokens, et les patches de l'image sont également considérés comme des tokens. Ces éléments sont ensuite transformés en représentations vectorielles contextuelles. Cette représentation vectorielle est ensuite donnée à un Transformer qui est adaptée pour traiter à la fois du texte et des images dans les documents. Le modèle peut prendre une image de document ainsi que des informations de texte et de positionnement (layout) correspondantes. Ce-dernier est entraîné avec les objectifs de pré-entraînements suivants:
- Modélisation de Langue Masquée (MLM) : Cela implique de masquer certains mots du texte et d'entraîner le modèle à les prédire. C'est une méthode courante pour le pré-entraînement des modèles de langage.
- Modélisation d'Image Masquée (MIM) : De manière similaire à MLM, mais appliquée aux images, où certaines parties de l'image sont masquées et le modèle est entraîné pour les reconstruire.
- Alignement Mot-Patch (Word-Patch Alignment, WPA) : Un objectif supplémentaire pour apprendre l'alignement entre les modalités de texte et d'image. Le modèle tente de prédire si un patch d'image correspondant à un mot de texte donné est masqué.

![](layoutlmv3.PNG)

LayoutLMv3 peut être comparé aux modèles plus anciens de Document AI tels que DocFormer et SelfDoc:

![](other_models.PNG)

LayoutLMv3 utilise des "patches" linéaires pour l'intégration d'images. Cette approche vise à réduire le "goulot d'étranglement" computationnel associé aux réseaux de neurones convolutionnels (CNN). En d'autres termes, plutôt que de s'appuyer sur des CNN complexes pour traiter les images, LayoutLMv3 utilise une méthode plus simple et moins gourmande en ressources. De plus, cette méthode élimine la nécessité de superviser des détecteurs d'objets régionaux pendant l'entraînement, ce qui simplifie le processus d'apprentissage du modèle.
En ce qui concerne les objectifs de pré-entraînement pour la modalité image, LayoutLMv3 est conçu pour apprendre à reconstruire des tokens d'image discrets à partir de patches masqués, plutôt que de se concentrer sur les pixels bruts ou les caractéristiques régionales. Cela permet au modèle de capturer les structures de mise en page de haut niveau au lieu de se concentrer sur les détails bruyants. En d'autres termes, plutôt que de se perdre dans les détails minutieux de l'image, LayoutLMv3 se concentre sur la compréhension globale de la disposition et de la structure des éléments dans les documents.

## Implémentation de LayoutLM
Le modèle de base qu'on peut trouver sur huggingface ([microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) ou [microsoft/layoutlmv3-large](https://huggingface.co/microsoft/layoutlmv3-large) est juste le modèle préentraîné: il n'y pas été fine-tuné pour une tâche particulière (comme un base-model pour les LLMs). Pour les modèles de type LayoutLM, ils ont été préentraîné via un unified text and image masking.

Ainsi, ces modèles doivent être fine-tunés pour être utilisés sur une tâche particulière. LayoutLMv3 can be fine-tuned for both text-centric tasks, including form understanding, receipt understanding, and document visual question answering, and image-centric tasks such as document image classification and document layout analysis.

Dans les exemples, je vais prendre des modèles déjà fine-tunés sur des tâches particulières:  DocVQA pour le QA sur les documents [xhyi/layoutlmv3_docvqa_t11c5000](https://huggingface.co/xhyi/layoutlmv3_docvqa_t11c5000), et FUNSD pour le layout analysis dans les documents [nielsr/layoutlmv3-finetuned-funsd](https://huggingface.co/nielsr/layoutlmv3-finetuned-funsd)


Tout d'abord, les installations nécessaires pour faire de l'inférence avec LayoutLM pour du QA sur images (pages de document):
```
pip install transformers Pillow datasets
python -m pip install -q 'git+https://github.com/facebookresearch/detectron2.git'
sudo apt install tesseract-ocr
pip install -q pytesseract
```
Ensuite, voici un exemple d'utilisation:
```
from transformers import AutoProcessor, LayoutLMv3ForQuestionAnswering, set_seed
import torch
from PIL import Image
from datasets import load_dataset

set_seed(88)
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForQuestionAnswering.from_pretrained("xhyi/layoutlmv3_docvqa_t11c5000")

image = Image.open("itc_limited.png").convert("RGB")

question = "Which nationality is this brand?"

encoding = processor(image, question, return_tensors="pt")
outputs = model(**encoding)
predicted_start_idx = outputs.start_logits.argmax(-1).item()
predicted_end_idx = outputs.end_logits.argmax(-1).item()
predicted_start_idx, predicted_end_idx

predicted_answer_tokens = encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1]
predicted_answer = processor.tokenizer.decode(predicted_answer_tokens)

print(predicted_answer)
```
Par contre: Il ne faut pas prendre des images avec trop de texte, sinon ça ne marche pas. Dans les bouts de code que j'ai trouvé, ils découpent les images en boxes et traitent chaque box de manière indépendante. Voir : https://huggingface.co/docs/transformers/model_doc/layoutlmv3

# <b>2) Nougat : document reconstruction</b>
[Nougat](https://github.com/facebookresearch/nougat) est quant à lui un modèle de Document Reconstruction sans OCR. Le modèle se base en effet sur les [Swin Transformers](https://arxiv.org/pdf/2103.14030.pdf), qui est un type spécifique de Visual Transformer, qui lui permet de traiter des patchs de toute petite taille (ressemblant à un pixel) sans augmenter la complexité computationnelle du modèle. En effet, cela est rendu possible grâce à l'approche "Shifted Window" du visual transformer: on découpe l'image (la page du document) en tout petits patchs, et la self-attention n'est calculée qu'entre chaque patchs d'une même window: le modèle possède une complexité de calcul linéaire par rapport à la taille de l'image d'entrée, grâce au calcul de self-attention uniquement au sein de chaque fenêtre locale (indiquée en rouge):
![](images/swin_transformer.PNG)

A chaque couche du modèle, la fenêtre se déplace de 2 patchs à droite et 2 patchs en bas. ainsi, in-fine, chaque patch a une visibilité sur tous les autres patchs. Cela permet d'avoir des patchs beaucoup plus petits et donc de se passer de l'OCR pour reconstruire le texte. 
Le Swin Transformer construit des cartes de caractéristiques hiérarchiques en fusionnant les patches d'image (représentés en gris) dans des couches plus profondes. Il peut donc servir de base générale pour les tâches de classification d'images et de reconnaissance dense. En revanche, les Transformers de vision précédents produisent des cartes de caractéristiques d'une seule basse résolution et ont une complexité de calcul quadratique par rapport à la taille de l'image d'entrée, en raison du calcul de self-attention globalement:
![](images/swin_transformer2.PNG)

Nougat a été entraîné sur un dataset de papiers de recherches arxiv (le papier en format d'origine (latex) a été téléchargé sur arxiv, et le pdf correspondant au papier). Le images ont été enlevées du papier (pdf & latex) et le modèle a appris à reconstruire  le latex à partir du pdf. Ainsi, le modèle que l'on trouve sur https://github.com/facebookresearch/nougat n'est adapté que pour reconstruire des papiers de recherche. Il faut fine-tuner le modèle sur d'autres types de documents pour pouvoir l'utiliser pour reconstruire des ppt, et autre.

# <b>3) Marker : extension de Nougat</b>
D'autres méthodes, sans fine-tuning, permettent d'utiliser Nougat pour reconstruire tout type de documents: c'est le cas de [marker](https://github.com/VikParuchuri/marker) qui applique Nougat, et quand le processing ne fonctionne pas, applique des outils d'OCR tels que tesseract pour reconstruire le document. C'est un "bidouillage" de Nougat.