---
layout: post
title: Fine-tuning
date: 2023-04-26
---

**Les modèles de langage ne sont que des modèles de prédiction de mots masqués, de mots suivants, etc. Si on veut les utiliser dans un cas particulier, pour en faire un chatbot, ou autre, il est nécessaire de les fine-tuner, c'est à dire de les réentraîner pour qu'ils répondent à une tâche particulière. Fine-tuner un modèle consiste à le réentraîner, mais cela devient de moins en moins faisable pour les modèles de langages, qui sont de plus en plus gros, et un réentraînement de ces modèles peut être très coûteux. C'est poruquoi plusieurs méthodes ont été mises en place pour contourner ce problème lié à la taille de ces gros modèles.**

# Comment fine-tuner ces gros modèles?

## [PEFT : Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft) 
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

## L'entraînement parallélisé des modèles de langage

Différentes façon de [paralléliser l'entraînement des modèles](https://arxiv.org/pdf/2101.03961.pdf):
![parallelized training](/assets/images/parallelized_training.PNG)

### En passant directement par le Trainer de Pytorch
Plutôt que d'essayer de réinventer la roue, [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) permet d'intégrer les techniques les plus récentes afin qu'elles puissent fonctionner ensemble de manière agréable et que votre code reste efficace et organisé. Il n'y a qu'à ajouter les paramètres nécessaires dans le [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) ou les [TrainingArguments](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/trainer#transformers.TrainingArguments). 

#### [Deepspeed](https://github.com/microsoft/DeepSpeed)
DeepSpeed est une bibliothèque de parallelized training de Microsoft. Elle a été utilisée pour entraîner des LLMs tels que Megatron-Turing NLG 530B et BLOOM.

[DeepSpeed Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)

![deepspeed1](/assets/images/deepspeed1.PNG)
![deepspeed2](/assets/images/deepspeed2.PNG)

[Azure and DeepSpeed empower easy-to-use and high-performance model training](azure-empowers-easytouse-highperformance-and-hyperscale-model-training-using-deepspeed/) montre le hardware requiert par modèle:

![hardware](/assets/images/hardware_deepspeed.PNG)

Voir aussi [DeepSpeed Data Efficiency: A composable library that makes better use of data, increases training efficiency, and improves model quality](https://www.deepspeed.ai/2022/12/11/data-efficiency.html), [ZeRO-Inference: Democratizing massive model inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)

Pour mieux comprendre comment DeepSpeed fonctionne, voir: [DeepSpeed training](https://www.deepspeed.ai/training/)

L'optimiseur de redondance zéro (ZERO) est au cœur de la manière dont DeepSpeed permet la mise à l'échelle. ZERO comporte trois étapes :
- Les états de l'optimiseur sont répartis entre les processus.
- Les gradients sont répartis entre les processus.
- Les paramètres du modèle sont répartis entre les processus.

![Deepspeed](/assets/images/deepspeed.gif)

Pour faire un entraînement parallélisé sur Pytorch avec DeepSpeed, ajouter dans le Trainer:

{% highlight lang %}
trainer = Trainer(devides=4, accelerator="gpu", strategy="deepspeed")
{% endhighlight %}

#### Fairscale FSDP
Fully-sharded data-parallel (FSDP) est la version de Meta du sharding, inspirée de DeepSpeed (stage 3) et optimisée pour la compatibilité avec PyTorch.
FSDP a été développé par l'équipe FairScale de Facebook (maintenant Meta) avec pour objectif d'optimiser la compatibilité avec PyTorch. 

Pour faire un entraînement parallélisé sur Pytorch avec DeepSpeed, ajouter dans le Trainer:
{% highlight lang %}
trainer = Trainer(devides=4, accelerator="gpu", strategy="fsdp")
{% endhighlight %}

#### Composer
Composer est une autre bibliothèque qui s'attaque à une partie différente du pipeline de formation. Composer ajoute certaines techniques telles que BlurPooling, ChannelsLast, CutMix et LabelSmoothing.
Ces techniques peuvent être ajoutées au modèle AVANT que l'optimisation ne commence. Cela signifie que pour utiliser composer avec PyTorch Lightning, vous pouvez simplement exécuter les optimisations manuellement sur le modèle avant de commencer votre boucle d'apprentissage.

![composer](/assets/images/composer.gif)

### [Accelerate: entraînement parallélisé sans passer par le Trainer](https://github.com/huggingface/accelerate)

Accelerate a été créé pour les utilisateurs de PyTorch qui aiment écrire la boucle d'entraînement des modèles PyTorch mais qui ne veulent pas utiliser le Trainer.

{% highlight lang %}
  import torch
  import torch.nn.functional as F
  from datasets import load_dataset
+ from accelerate import Accelerator

+ accelerator = Accelerator()
- device = 'cpu'
+ device = accelerator.device

  model = torch.nn.Transformer().to(device)
  optimizer = torch.optim.Adam(model.parameters())

  dataset = load_dataset('my_dataset')
  data = torch.utils.data.DataLoader(dataset, shuffle=True)

+ model, optimizer, data = accelerator.prepare(model, optimizer, data)

  model.train()
  for epoch in range(10):
      for source, targets in data:
          source = source.to(device)
          targets = targets.to(device)

          optimizer.zero_grad()

          output = model(source)
          loss = F.cross_entropy(output, targets)

-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
{% endhighlight %}

# Les tâches sur lesquelles fine-tuner les LLMs
### [Instruction-tuning](https://github.com/SinclairCoder/Instruction-Tuning-Papers)
La tendance de l'Instruction-Tuning a commencé avec le dataset [Natural-Instructions](https://github.com/allenai/natural-instructions). Il s'agit d'apprendre aux modèles de langage à suivre le langage naturel (y compris les prompts, les exemples positifs ou négatifs, les contraintes, etc.), afin d'améliorer l'apprentissage multitâche sur les tâches d'entraînement et la généralisation sur les tâches non apprises par le modèle.

L'engouement est énorme sur ce type de fine-tuning (rien qu'à voir le nombre de papiers dessus...). Voici quelques papiers intéressants qui parlent de ça:

#### [In-BoXBART: Get Instructions into Biomedical Multi-Task Learning](https://arxiv.org/pdf/2204.07600.pdf) 
In-BoXBART explore l'instruction-tuning et son efficacité pour répondre à plusieurs tâches dans le domaine biomédical. Ils ont dans le instruction dataset 32 tâches d'instruction (avec des prompts demandant de faire des résumé, d'autres de répondre à des questions, d'autres de traduire, etc). Ils montre que ce modèle "multi-tâches" est plus performant dans de nombreuses tâches qu'un modèle entraîné spécifiquement pour celles-ci. Leur code pour le fine-tuning de leur modèle est open-source et bien documenté: [github.com/Mihir3009/In-BoXBART](https://github.com/Mihir3009/In-BoXBART). Voir surtout le [code de fine-tuning](https://github.com/Mihir3009/In-BoXBART/blob/main/scripts/finetune_model.py) et le [json contenant les templates d'insturctions](https://github.com/Mihir3009/In-BoXBART/blob/main/templates/BoX_instructions.json), classées par tâche.

#### [reStructured Pre-training](https://arxiv.org/pdf/2206.11147.pdf)
Le papier est intéressant tant il montre l'évolution du paradigme d'entraînement des modèles de langage. Il propose un nouveau paradigme appelé "reStructured Pre-training". 

![évolution du paradigme d'entraînement des modèles de langage](/assets/images/evolution_entrainement_modeles.PNG)

- Fully Supervised Learning : Les chercheurs utilisaient des techniques d'apprentissage automatique traditionnelles, comme SVM et PGM, pour apprendre à partir de données étiquetées sans pré-entraînement sur de grands corpus non étiquetés. L'accent était mis sur l'ingénierie des caractéristiques pour extraire des caractéristiques pertinentes.
- Pre-train, Fine-tune - Non-contextual : L'avènement des réseaux de neurones a permis d'extraire automatiquement des caractéristiques utiles, nécessitant une ingénierie d'architecture de réseaux de neurones. Les représentations de mots pré-entraînées ont amélioré les performances, mais les modèles restaient spécialisés pour une tâche spécifique.
- Pre-train, Fine-tune - Contextual : Les modèles pré-entraînés à grande échelle (PLMs), tels que BERT et GPT, ont été développés pour apprendre des représentations textuelles contextuelles à partir de textes non étiquetés massifs. Les PLMs facilitent l'adaptation aux tâches en aval avec moins de paramètres et de données d'entraînement.
- Pre-train, Prompt, Predict : Avec GPT-3, les tâches en aval sont reformulées pour ressembler à des tâches de pré-entraînement, ce qui réduit encore les besoins en données étiquetées et permet un apprentissage avec peu ou pas d'exemples.
- reStructured Pre-training (proposition) : Ce nouveau paradigme vise à maximiser l'utilité des données disponibles en considérant le pré-entraînement comme un processus de stockage de données et l'entraînement en aval comme un processus d'accès aux données. L'idée est de pré-entraîner les modèles sur des signaux structurés, en couvrant autant de types de signaux que possible et en fournissant des mécanismes d'accès précis pour ces signaux. 

#### [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/pdf/2301.13688.pdf)
Ce papier est intéressant tant il regroupe les différents datasets d'instructions disponibles sur le marché:

![datasets d'instructions](/assets/images/datasets_instruction.PNG)

Ils ont construit leur propre collection de données d'instruction (Flan 2022) qui contient des instructions provenant de plusieurs sources, notamment Flan 2021, T0-SF, Super-Natural Instructions, Chain-of-Thought, Dialog et Program Synthesis. Ces instructions sont formulées sous forme de tâches de compréhension de texte, de question-réponse, de génération de texte, de classification, etc.
Le papier explique aussi la méthode d'investion d'entrée (input inversion) qui permet d'enrichir un dataset d'instruction en demandant aux modèles de générer une question à partir d'une réponse. 

#### Exemple d'un modèle récemment entraîné avec de l'instruction-tuning: [IGEL](https://www.philschmid.de/introducing-igel) 
IGEL est parti du modèle pré-entraîné [bloom-german](https://huggingface.co/malteos/bloom-6b4-clp-german) fine-tuné sur une base de données d'instructions traduite en allemand.

#### [ICIL: In Context Instruction Learning](https://arxiv.org/pdf/2302.14691.pdf)
Avec cette Méthode, il n'y aurait pas besoin de fine-tuner les modèles de langages auto-régressifs pour faire du QA: Il s'agit de construire les prompts d'une certaine manière pour que le modèle préside le mot suivant, qui correspond à l'output d'un instruction-base modèle. 

![ICIL](/assets/images/ICIL.png)

Cette méthode permet d'exploiter les connaissances déjà acquises par les modèles de langage au cours de leur pré-entraînement, et met l'accent sur la manière dont les modèles peuvent apprendre implicitement de nouvelles informations et appliquer cette connaissance aux tâches en aval, en utilisant le contexte de la tâche plutôt que de les adapter spécifiquement à chaque tâche individuelle.
l'approche ICIL zero-shot vise à exploiter le contexte de la tâche pour améliorer les performances du modèle sur ces tâches en aval sans nécessiter d'entraînement spécifique.

#### [Adding Instructions during Pretraining: Effective Way of Controlling Toxicity in Language Models](https://arxiv.org/pdf/2302.07388.pdf)

#### [GPTScore: Evaluate as You Desire](https://arxiv.org/pdf/2302.04166.pdf)

#### [Learning Instructions with Unlabeled Data for Zero-Shot Cross-Task Generalization](https://arxiv.org/pdf/2210.09175.pdf)
