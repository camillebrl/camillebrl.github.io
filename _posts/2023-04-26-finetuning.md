---
layout: post
title: Fine-tuning
date: 2023-04-26
---

**Les modèles de langage ne sont que des modèles de prédiction de mots masqués, de mots suivants, etc. Si on veut les utiliser dans un cas particulier, pour en faire un chatbot, ou autre, il est nécessaire de les fine-tuner, c'est à dire de les réentraîner pour qu'ils répondent à une tâche particulière. Fine-tuner un modèle consiste à le réentraîner, mais cela devient de moins en moins faisable pour les modèles de langages, qui sont de plus en plus gros, et un réentraînement de ces modèles peut être très coûteux. C'est poruquoi plusieurs méthodes ont été mises en place pour contourner ce problème lié à la taille de ces gros modèles.**

# [PEFT : Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) 
PEFT est un outil qui permet d'appliquer des méthodes d'optimisation de la modification des paramètres des modèles.
PEFT permet d'entraîner les modèles avec plusieurs méthodes:
### [LoRA: Low-Rank Adaptation](https://github.com/microsoft/LoRA)
[LoRA](https://arxiv.org/pdf/2106.09685.pdf) est une méthode qui permet d'éviter de passer par un entraînement classique du modèle (calcule du gradient, etc). Au lieu de cela, la méthode LoRA trouve un "filtre" sous forme de matrice, qu’on applique aux poids pré-entraînés pour les modifier. Les poids du modèle sont figés, et on applique une matrice de décomposition de rang dans chaque couche du modèle.
En fait, cette méthode part de l'idée développée dans les papiers [Intrinsic Dimentionality Explains The Effectiveness Of Language Model FIne-Tuning](https://arxiv.org/pdf/2012.13255.pdf) qui montre qu'on peut se ramener à un espace de faible dimension pour décrire les LLMs. Effectivement, on connait la [PCA](https://medium.com/apprentice-journal/pca-application-in-machine-learning-4827c07a61db) qui projette les données sur les axes principaux de variation des données, mais aussi les [auto-encodeurs](https://pierre-schwartz.medium.com/introduction-aux-auto-encodeurs-61e8d74660f3), avec l'encodeur qui projette les données dans un espace de dimension inférieure, et le décodeur qui apprend à restituer ces données, pour faire simple. Ici, c'est un peu la même idée: le papier suppose que le changement des poids pendant le fine-tuning peut aussi résider dans un espace de plus faible dimension. 
LoRA permet ainsi de fine-tuner les modèles de langage avec 3x moins de mémoire requise.

### [Prefix-tuning](https://aclanthology.org/2021.acl-long.353.pdf)
L'idée du Prefix-tuning est de garder les poids du modèle d'origine, et de n'optimiser que une séquence de "vecteurs continus spécifiques à une tâche", appelés "préfix".

![prexis-tuning idea](/assets/images/prefix_tuning.png)

Le papier montre qu'en ne modifiant que 0.1% des paramètres, on obitient de bons résultats sur plusieurs tâches.
D'autres papiers ont étudié la possibilité de n'entraîner que certains paramètres des modèles de langage: insertion de nouveaux modules contenant les paramètres à entraîner, [ajout de couches entre les couches existantes](http://proceedings.mlr.press/v97/houlsby19a.html), [application d'un "task-specific mask"](https://arxiv.org/pdf/2004.12406.pdf), ...