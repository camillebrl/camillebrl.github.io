---
layout: post
title: Les approches d'explicabilité par les exemples appliquées aux LLMs
date: 2025-06-16 22:00:00
description: Méthodes d'explicabilité de la génération de texte par les LLMs
tags: XAI, influence, transformers
categories: sample-posts
maths: true
giscus_comments: true
thumbnail: assets/img/xai_llm.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

Un LLM génère des tokens à partir d'autres tokens selon un processus probabiliste. Pour chaque séquence de tokens fournie en entrée, le modèle retourne une distribution de probabilité sur l'ensemble des tokens suivants possibles. Un token représente un élément du vocabulaire du modèle, dont la taille est fixe et définie dans la configuration du tokenizer ou du modèle lui-même.

Prenons l'exemple des modèles Qwen avec un vocabulaire de 151 936 tokens : à chaque étape de génération, le modèle calcule une distribution de probabilité sur ces 151 936 possibilités. De manière autorégressive, chaque nouveau token généré est ajouté à la séquence d'entrée pour prédire le suivant, et ainsi de suite.

Comprendre pourquoi un modèle génère certains tokens plutôt que d'autres soulève plusieurs questions cruciales. La séquence d'entrée (prompt) peut contenir des éléments variés, notamment des chunks de texte dans le cas du RAG (Retrieval-Augmented Generation), qui servent de contexte pour répondre aux questions. Pour véritablement comprendre le processus de génération, nous devons répondre à trois interrogations principales :

- Quelles parties spécifiques du prompt influencent la génération de quels tokens ? Cette question est particulièrement pertinente dans le contexte du RAG où différents passages peuvent avoir des impacts variés sur la réponse finale.

- Sur quelles connaissances acquises pendant l'entraînement le modèle s'appuie-t-il pour générer chaque token ? Comment distinguer ce qui provient du prompt de ce qui provient de la mémoire du modèle ?

- Quelles parties du modèle (couches, têtes d'attention, neurones) sont responsables de quelles décisions ? Pourquoi des modèles de tailles différentes, même issus de la même famille, produisent-ils des réponses différentes ?

Pour répondre à ces questions, nous devons nous tourner vers les approches d'explicabilité, et plus particulièrement l'explicabilité par les données. Cette approche examine deux sources d'influence principales : les données présentes dans le prompt et les données utilisées lors de l'entraînement du modèle.
L'objectif est d'établir des liens causaux clairs entre ces sources de données et les tokens générés, qu'ils soient pris individuellement ou en groupes cohérents. Cette compréhension est essentielle pour améliorer la fiabilité, la transparence et le contrôle des systèmes basés sur les LLM.

## Rapide overview des approches d'explicabilité par les exemples appliquées aux LLMs

### Approches TDA (training data analysis) pour comprendre quelles connaissances apprises dans l'entraînement ont été influentes dans la génération de quels tokens
1. **Si on cherche à estimer l'impact qu'aurait un exemple d'entraînement sur la perte d'un exemple de test à un exemple d'entraînement** (qu'il soit dans le jeu de données d'entraînement de base ou pas) :
$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) = \frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\,\theta_\varepsilon\bigr)\Big|_{\varepsilon=0} = -\,g_{\mathrm{test}}^\top\,H_\theta^{-1}\,g_{\mathrm{train}}
$$
Voir les librairies [Kronfluence](https://github.com/pomonam/kronfluence) ou [TracIn](https://github.com/frederick0329/TracIn) ou [PINNfluence](https://github.com/aleks-krasowski/PINNfluence) ou [GraSS](https://github.com/TRAIS-Lab/GraSS). Des repo github proposent des tutos pour plusieurs de ces lib: cf [Influenciae](https://github.com/deel-ai/influenciae). Les fonctions d'influence permettent de répondre à ces questions:
    - Est-ce que je devrais ajouter cet exemple dans mon set d'apprentissage pour améliorer les performances de cette prédiction?
    - Quels exemples d'entraînement ont été utiles à la prédiction de mon modèle?
    - Le modèle s'est trompé: sur quels exemples d'entraînement s'est-il basé pour cette mauvaise prédiction?
    - Quel serait l'effet d'une re-labélisation de ma donnée d'entraînement sur la prédiction? Quelles sont les données que je devrais re-labéliser pour améliorer ma prédiction? => Pour répondre à ces questions, il faut modifier un peu l'approche: $\mathrm{Influence}(\text{relabeling}(z_{\rm train}) \to z_{\rm test}) = g_{\mathrm{test}}^\top\,H_\theta^{-1}\,\bigl[\nabla_\theta \mathcal{L}\bigl(z_{\text{train}}^{\text{modified}},\,\theta_\varepsilon\bigr)  \;-\;  \nabla_\theta \mathcal{L}\bigl(z_{\text{train}}^{\text{original}},\,\theta_\varepsilon\bigr)\bigr]$
  
Dans ce [repo](https://github.com/camillebrl/llm_training_data_attribution), j'ai créé une interface pour calculer l'influence des données d'entraînement (de pre-training) de LLMs (base : modèles pré-entraînés uniquement) en 2 phases: d'abord en diminuant les données d'entraînement sur lesquelles calculer l'influence à l'aide d'elasticsearch pour identifier les 50 phrases les plus similaires au prompt et au texte généré par le LLM, ensuite en utilisant [Kronfluence](https://github.com/pomonam/kronfluence) pour calculer l'influence de ces données sur la génération en question. Les scores d'influence sont ensuite normalisés par la norme au carrée de la loss, comme réalisé dans les approches d'état de l'art.

### Approches de Context Attribution: effet du prompt (input donné au modèle) dans la génération de quels tokens
2. **Si on veut estimer sur quelle partie de l'input le modèle s'est basé pour faire sa prédiction**, on utilise des approches de cartes de saillance. En gros, on mesure comment la sortie du modèle $f(x)$ varie si on modifie chaque composante $x_i$ de l’entrée. Pour se faire, des libraires comme [Captum](https://github.com/pytorch/captum) ou [Inseq](https://github.com/inseq-team/inseq) ou [Grad-cam](https://github.com/jacobgil/pytorch-grad-cam) ou [Investigate](https://github.com/albermax/innvestigate) ou [Alibi](https://github.com/SeldonIO/alibi) existent.

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/mirage_illustration.png" class="img-fluid rounded z-depth-1" zoomable=true %}

J'ai créé un [repo github](https://github.com/camillebrl/mirage_ui) qui reproduit un [papier (MIRAGE)](https://aclanthology.org/2024.emnlp-main.347.pdf) d'explication de la génération du modèle à partir des éléments du prompt, découpés document par document (cas du RAG)

## 1. Estimation de l’impact d’un exemple d’entraînement sur la prédiction d’un exemple de test: Influence functions

### 1.1 Introduction sur les fonctions d'influence
Pour introduire les fonctions d'influence appliquées au deep learning, nous nous basons sur le le papier [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730), et notamment sur l'annexe A pour expliquer les différentes formules.

**L’influence de** $z_{\rm train}$ **sur** $z_{\rm test}$ ($\mathrm{Influence}(z_{\rm train}\!\to\!z_{\rm test})$) **se définit comme** 
$$
\mathrm{Influence}(z_{\rm train}\to z_{\rm test})\;=\; \left.\frac{d}{d\varepsilon}\,\mathcal{L}\bigl(z_{\rm test},\,\theta_\varepsilon\bigr)\right|_{\varepsilon=0}
$$

En d’autres termes, elle mesure la sensibilité de la loss de $z_{\rm test}$ (ou de toute fonction $f(\theta)$) à un « up-weight » infinitésimal de la loss de $z_{\rm train}$.  


A noter: 
- C'est l'**impact de l"up-weight" de la loss sur $z_\text{train}$ sur qqch qu'on mesure avec les fonctions d'influence**. En fait pour mesurer l'impact de l"up-weight" de $z_{\text{train}}$ dans la loss globale, on se pose la question: "on se pose la question : "**si je donnais un peu plus de poids à ce terme de loss dans l’objectif global, comment cela ferait-il bouger mes paramètres et, avec ces nouveaux paramètres, ma performance sur un point de test, ou sur une fonction?**". En effet, en deep learning, modifier le poids d'une donnée dans l'entraînement, c'est modifier le poids qu'on donne à sa loss dans l'apprentissage.
- $f(\theta)$ peut être n'importe quelle fonction (exemple: la moyenne des prédictions sur un ensemble de données types (cf le papier [Which Data Attributes Stimulate Math and Code Reasoning? An Investigation via Influence Functions](https://arxiv.org/pdf/2505.19949) qui cherche à calculer l'influence des textes d'entraînement sur la génération de code (moyenne de des log probabilité de la génération de chaque token de code générés dans un benchmark sachant un problème de code en langage naturel à résoudre)), la différence entre 2 prédictions du modèle, ...)

On a la formule de l'influence:
$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr)
= I_{z_\text{test}}\bigl(z_{\mathrm{train}}\bigr)
= -\,g_{\mathrm{test}}^\top\,H_\theta^{-1}\,g_{\mathrm{train}}.
$$

Avec:
$$
g_{\mathrm{test}} \;=\; \nabla_\theta \,\mathcal{L}\bigl(z_{\mathrm{test}},\,\theta_\varepsilon\bigr)\,.
$$

$$
g_{\mathrm{train}} \;=\; \nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)
$$

**Détaillon un peu comment on a obtenu cette formule...**

On cherche 

$$
\frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\ \theta_\varepsilon\bigr) \Big|_{\varepsilon=0}
$$

On peut voir ça comme $\frac{d}{d\varepsilon} f(g(\varepsilon))$ avec $g(\varepsilon) = \theta_\varepsilon$ et $f(\theta_\varepsilon) = \mathcal{L}(z_{\rm test},\theta_\varepsilon)$, d'où, par chain rule :

$$
f'(g(\varepsilon)) = f'\bigl(g(\varepsilon)\bigr)\,g'(\varepsilon)
$$

Donc on a :

$$
\frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\ \theta_\varepsilon\bigr) \Big|_{\varepsilon=0} = \nabla_\theta \mathcal{L}(z_{\rm test}, \theta_\varepsilon)\Big|_{\varepsilon=0} \times \frac{d}{d\varepsilon} \theta_\varepsilon \Big|_{\varepsilon=0}
$$

Du coup, dans un premier temps, il nous faut calculer 

$$
\frac{d}{d\varepsilon} \theta_\varepsilon \Big|_{\varepsilon=0}
$$

Et ensuite on le multipliera à 

$$
\nabla_\theta \mathcal{L}(z_{\rm test}, \theta_\varepsilon)\Big|_{\varepsilon=0}
$$

qu'on sait calculer vu que c'est au voisinage de $\theta$, les poids du modèle de base.

**D'abord, commençons par calculer** 

$$
\frac{d}{d\varepsilon} \theta_\varepsilon \Big|_{\varepsilon=0}
$$

**en gros : comment $\theta_\varepsilon$ varie autour de $\theta$ quand on up-weight très légèrement (voisinage de 0) la loss de notre exemple $z_\text{train}$ :**

Pour faire cet "up-weight" de la loss de $z_{\text{train}}$ d'un tout petit $\varepsilon$, on perturbe la fonction de perte en ajoutant un petit coefficient $\varepsilon$ sur la perte de $z_{\rm train}$ et on voit comment les paramètres optimaux $\theta$ évoluent avec $\varepsilon$.

On repart de ce que ça veut dire de "perturber la fonction de perte en ajoutant un petit coefficient $\varepsilon$ sur la perte de $z_{\rm train}$": on obtient une nouvelle la loss totale du modèle ($R_\varepsilon(\theta)$) avec les nouveau poid $\varepsilon$ donné à la loss de $z_{\text{train}}$:
$$
R_\varepsilon(\theta) = \frac{1}{n}\sum_{i=1}^n \mathcal{L}(z_i,\theta)
  \;+\;\varepsilon\,\mathcal{L}(z_{\rm train},\theta).
$$

Puis, on cherche les poids $\theta_\varepsilon$ qui minimisent cette nouvelle loss:
$$
\theta_\varepsilon 
\;=\; 
\arg\min_{\theta}\;R_\varepsilon(\theta)
$$

Ce qui revient à chercher les $\theta_\varepsilon$ dont le gradient de cette nouvelle loss en $\theta$ est nul, car $\theta_\varepsilon$ est un minimum local de $R_\varepsilon$ si et seulement si sa dérivée première (gradient) s'annule:
$$
\nabla_\theta R_\varepsilon\bigl(\theta_\varepsilon\bigr) = 0.
$$
En développant:

$$
\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta_\varepsilon)
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)
= 0.
$$

Ici, si on peut faire une approximation de Taylor en $\theta$ de $\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta_\varepsilon)
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)$, puisqu'on veut voir la modification de $\theta_\varepsilon$ pour un tout petit $\varepsilon$, ie proche de 0, donc $\theta_\varepsilon$ proche de $\theta$.

Le formule de Taylor à l'ordre 1 nous donne:
$$
f(x) \approx f(x_0) + f'(x_0)\,(x - x_0)
$$

D'où, en approximant avec Taylor à l'ordre 1 $\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta_\varepsilon)\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)$ en $\theta$, on obtient:

$$
[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta)
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta) ] \\
+ \; \;(\theta_\varepsilon - \theta) \;\; [ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon)
\;+\;\varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) ]
$$

On a donc:

$$
\left[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta) + \varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta) \right] + (\theta_\varepsilon - \theta) \left[ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon) + \varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) \right] = 0
$$

D'où:

$$
(\theta_\varepsilon - \theta) = - \left[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta) + \varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta) \right] \times \left[ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon) + \varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) \right]^{-1}
$$

Or, puisque $\theta$ sont les poids optimaux pour le modèle de base, $\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta) = 0$ et $\varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) = 0$:

$$
(\theta_\varepsilon - \theta) = - \varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta)  \times  [\frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon)]^{-1}
$$

Et si on dérive par rapport à $\varepsilon$ on obtient:

$$
\frac{d}{d\varepsilon}(\theta_\varepsilon - \theta) = - \nabla_\theta \mathcal{L}(z_{\rm train},\theta)  \times  [\frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon)]^{-1} = - \nabla_\theta \mathcal{L}(z_{\rm train},\theta) H_\theta^{-1}
$$

Or $\frac{d}{d\varepsilon}(\theta_\varepsilon - \theta) = \frac{d}{d\varepsilon} \theta_\varepsilon$

D'où : 
$$
\frac{d}{d\varepsilon} \theta_\varepsilon = - \nabla_\theta \mathcal{L}(z_{\rm train},\theta) H_\theta^{-1}
$$

Ainsi, quand on multiplie par $\nabla_\theta \,\mathcal{L}\bigl(z_{\mathrm{test}},\,\theta_\varepsilon\bigr)$ on obtient a la formule de l'influence:
$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) 
= I_{z_\text{test}}\bigl(z_{\mathrm{train}}\bigr)
= \frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\ \theta_\varepsilon\bigr) \Big|_{\varepsilon=0} = -\nabla_\theta \,\mathcal{L}\bigl(z_{\mathrm{test}},\,\theta_\varepsilon\bigr)H_\theta^{-1}\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)
$$
ou:
$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to f(\theta)\bigr) 
= I_f(z_{\mathrm{train}})
= \frac{d}{d\varepsilon}\, f(\theta_\varepsilon) \Big|_{\varepsilon=0} = -\nabla_\theta \,f(\theta_\varepsilon)H_\theta^{-1}\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)
$$

### 1.2 "Corrections" des fonctions d'influence
### 1.2.1 Données d'entraînement mal apprises prédominantes: comment corriger?
A noter que le papier [Influence Functions in Deep Learning Are Fragile](https://arxiv.org/pdf/2006.14651) indique que les fonctions d'influence sont biaisées vers les exemples à forte perte. en effet, le gradient de la loss des points d'entraînement est plus élevé pour les exemples mal appris, entraînant un biais systématique vers ces exemples dans l’attribution d’influence, indépendamment de leur véritable effet sur la perte globale. Le papier propose de recalculer les scores d'influence en normalisant les gradients par leur norme, éliminant ainsi les gradients élevés pour des exemples mal appris (sans lien avec la fonction à maximiser).  

$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) = \frac{\frac{d}{d\varepsilon} \mathcal{L}\bigl(z_{\rm test},\theta_\varepsilon\bigr)}{||\nabla_\theta \mathcal{L}(z)||^2} \Big|_{\varepsilon=0} = \frac{-\nabla_\theta \,f(\theta_\varepsilon)H_\theta^{-1}\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)}{||\nabla_\theta \mathcal{L}(z)||^2}
$$

### 1.2.2 Mauvaise approximation du leave-one-out? Analyse du Proximal Bregman Response

Le papier [If Influence Functions are the Answer, Then What is the Question?](https://arxiv.org/pdf/2209.05364) explique ensuite que les functions d'influence n’approximent pas fidèlement le retraining « leave-one-out », mais qu’elles correspondent en fait au proximal Bregman response function (PBRF). [A FINIR]

### 1.3 Comment calcule-t-on l'inverse de la hessienne en deep learning

#### 1.3.1 Hesienne pas forcément inversible...

Dans les réseaux de neurones, la loss d'entraînement n'est pas fortement convexe (le minimum local n'est pas forcément un minimum global...) donc la hessienne peut être non inversible. Donc, des approches ont été étudiées pour garantir l'inversibilité de la hessienne, en ajoutant notamment un terme dit de "damping" $\lambda >0$. 

#### 1.3.2 Hessienne par rapport aux paramètres du réseau compliquée à calculer pour des réseaux avec un grand nombre de paramètres...

Le papier [If Influence Functions are the Answer, Then What is the Question?](https://arxiv.org/pdf/2209.05364)  propose d’approximer la Hessienne par la Hessienne de Gauss–Newton (GNH), notée $G_\theta$ :

$$
G_\theta = J_{y\theta}^T \, H_y \, J_{y\theta}
$$

où :

- $J_{y\theta}$ est la matrice Jacobienne des sorties du réseau par rapport aux paramètres ;
- $H_y = \nabla^2_y \mathcal{L}(y, \theta)$ est la Hessienne de la fonction de coût par rapport aux sorties du réseau.
propose d'approximer la Hessienne par la matrice d'information de Fisher (équivalente à la Hessienne de Gauss-Newton). 

En fait, on a:

$$
\underbrace{H_\theta}_{\text{Très difficile}}
=\;\;\;
\underbrace{J_{y\theta}^T \, H_y \, J_{y\theta}}_{\text{Facile (GNH)}}
\;\;\;+
\underbrace{\sum_i \frac{\partial \mathcal{L}}{\partial y_i} \,\nabla_\theta^2 y_i}_{\text{Cauchemar computationnel}}
$$

avec:
  - $J_{y\theta}$ : Déjà calculé par backpropagation standard  
  - $H_y$ : Matrice $k \times k$ avec $k =$ le nombre de tokens possibles (ex : 151936 pour Qwen)

Afin de ne pas calculer les dérivées secondes à travers tout le réseau (ce qui est très coûteux quand on a beaucoup de paramètres), on utilise, pour l'ensemble des calculs d'influence (surtout pour les LLMs), ce résultat:

$$
   H_\theta^{-1}\approx \bigl(G_\theta + \lambda I\bigr)^{-1}.
$$


#### 1.3.3 Factorisation par blocs de $G_\theta$ et factorisation en produit de Kronecker pour pourvoir stocker cette matrice & paralléliser les calculs entre couches

Au lieu d’inverser directement la grande matrice $\,G_\theta+\lambda I$, le papier [Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization](https://arxiv.org/pdf/2505.05017) exploite sa structure en blocs correspondant à chaque couche du réseau.  
$$
G 
= \begin{bmatrix}
  G_{1,1} & G_{1,2} & \cdots\\
  G_{2,1} & G_{2,2} & \\
  \vdots & & \ddots
\end{bmatrix},
$$
Pour un réseau à $L$ couches, on a donc $G = [G_{i,j}]_{1 \leq i, j \leq L}$, avec $G_{i,j}$ qui représente le bloc entre les paramètres de la couche $i$ et de la couche $j$.

Cette séparation en bloc permet au papier de simplifier $G$ en ne gardant que les blocs diagonaux :

$$
G \approx \tilde{G} = \mathrm{diag}(G_{1,1}, G_{2,2}, \dots, G_{L,L})
$$

Cela signifie qu’on ignore les interactions entre différentes couches et qu’on ne considère que les blocs $G_{l,l}$ pour chaque couche $l$.

Cette approche par blocs permet :

- De traiter chaque couche indépendamment
- D'éviter de stocker/calculer la matrice complète de taille $p \times p$ (où $p$ est nombre total de paramètres)
- De paralléliser les calculs entre couches

C'est ce qui rend la méthode scalable pour les grands modèles comme les LLMs avec des milliards (plutôt même billions...) de paramètres.


### 1.4 Le cas des LLMs: besoin d'une influence token-wise ou sentence-wise
Le papier [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/pdf/2308.03296) présente l'application des fonctions d'influence aux LLMs. Dans le cas des LLMs, la loss est la negative log-vraisemblance. La première particularité d'un LLM, c'est le fait qu'un datapoint est un peu compliqué à définir. On peut supposer qu'il s'agit d'une phrase (et son label, le token suivant la phrase), ou on peut considérer le token lui-même (l'input étant la phrase le précédent, le label le token en question, par exemple). Mais il est important de bien définir de quoi on parle quand on parle de "datapoint".

#### 1.4.1 L'influence à l'échelle de la phrase $z_m$

Supposons que $z_m$ soit cette phrase "le chat est gris", soit ce datapoint:

$$[\text{BOS, le, chat, est, gris}] \rightarrow \text{[EOS]}$$

Vu qu'on est dans un cas autorégressif (c'est-à-dire que les tokens sont prédits à partir des tokens précédents de la séquence) :

Si $z_m$ de taille $T$ :

$$\nabla_\theta L(z_m, \theta) = \sum_{t=1}^T(-\nabla_\theta \log p(z_{m,t} \mid z_{m, <t}, \theta))$$

Nous, $z_m$ est de taille 6 :

$$
\begin{align}
\nabla_\theta L(z_m, \theta) &= -\nabla_\theta \log p(\text{le} \mid \text{[BOS]}, \theta) \\
&\quad - \nabla_\theta \log p(\text{chat} \mid \text{[[BOS], le]}, \theta) \\
&\quad - \nabla_\theta \log p(\text{est} \mid \text{[BOS], le, chat}, \theta) \\
&\quad - \nabla_\theta \log p(\text{gris} \mid \text{[BOS], le, chat, est}, \theta) \\
&\quad - \nabla_\theta \log p(\text{[EOS]} \mid \text{[BOS], le, chat, est, gris}, \theta)
\end{align}
$$

#### 1.4.2 L'influence à l'échelle des tokens $t$ dans la phrase $z_m$

On peut aussi considérer l'échelle du token, comme on l'a mis plus haut, en considérant l'input comme étant la phrase précédant ce token, et le label ce token en question.

On a ici $\nabla_\theta L(z_m, \theta)$ qui est la somme des gradients de la loss au niveau de chaque token. Du coup, il suffit de prendre $- \nabla_\theta \log p(\text{gris} \mid \text{[BOS], le, chat, est}, \theta)$ pour avoir l'influence du token gris dans la séquence par exemple. On peut ainsi avoir l'information token par token.

Et ça c'est ce qu'on obtient ici (cf [Studying Large Language Model Generalization with Influence Functions, Grosse 2023](https://arxiv.org/pdf/2308.03296)) :

$$\nabla_\theta L(\text{token t dans } z_m, \theta) = \nabla_\theta \log p(\text{token t} \mid \text{ce qui est avant token t dans } z_m, \theta)$$

On obtient donc la formule:

$$I_f(z_{m,t}) = \nabla_{\theta}f(\theta)^{T} H^{-1} \nabla_{\theta}\log p(z_{m,t}\mid z_{m,<t}, \theta)$$

Prenons l'exemple suivant: on prend $f = \log p(\text{"hydrogen and oxygen"} \mid \text{"Water is composed of"})$ et $z_m$ qui est le texte ci-dessous. On peut afficher l'influence token par token dans le texte:

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### 1.6 Le cas des LLMs: beaucoup de données d'entraînement (eg. 36 trillions de tokens pour Qwen3) => query batching ou semantic matching pour ne pas calculer l'influence sur toutes les données (trop coûteux)
Le papier [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/pdf/2308.03296) propose une approche pour éviter de calculer les gradients de tous les exemples d'entraînement candidats pour chaque requête d'influence. Pour cela, ils "filtrent" les données d'entraînement par rapport à la phrase test via un filtrage TF-IDF et une approche qu'ils introduisent de "query batching".

#### 1.6.1 Le filtrage TF-IDF
Le filtrage TF-IDF utilise une technique classique de recherche d'information pour présélectionner les séquences d'entraînement les plus susceptibles d'être influentes. L'intuition derrière est que les séquences pertinentes devraient avoir au moins un certain chevauchement de tokens avec la requête.

Ils retiennent les top 10,000 séquences selon le score TF-IDF Calcul d'influence et calculent les influences uniquement sur ces séquences présélectionnées. 

#### 1.6.2 Le Query-Batching

Dans un LLM, on a beaucoup d'exemples $z_m$ d'entraînement. Donc, on calcule séparemment $\nabla_\theta\mathcal{L}(z_m, \theta_\varepsilon)$ et $\nabla_{\theta} f(\theta_\varepsilon)^\top \, H^{-1}$ qui se calcule en une fois. 

Pour stocker de nombreux gradients de requêtes en mémoire ($\nabla_\theta\mathcal{L}(z_m, \theta_\varepsilon) \; \forall \; z_m$), ils approximent chaque matrice de gradient préconditionné comme étant de rang faible (rank-32 dans leurs expériences).

Ainsi, pour chaque requête, ils n'ont pas à refaire les calculs! Ils ont juste à calculer $\nabla_{\theta} f(\theta_\varepsilon)$.

### 1.7 Le cas des LLMs: plusieurs couches d'entraînement (pretraining, fine-tuning, alignement, ...) => multi-stage influence functions

Le papier [Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization](https://arxiv.org/pdf/2505.05017) explique que la fonction d'influence classique $I_f(z_{m,t}) = \nabla_{\theta}f(\theta)^{T} H^{-1} \sum_{t=1}^T(-\nabla_\theta \log p(z_{m,t} \mid z_{m, <t}, \theta))$ permet de quantifier l'impact d'une phrase d'entraînement sur les prédictions du modèle. Cependant, on a des modèles qui sont passés par plusieurs phases d'entraînement pour les LLMs (avec plusieurs données différentes). En effet, les LLMs sont pré-entraînés (modèles "base"), puis instruct-tuné (modèles "chat"), puis passent par du reinforcement learning (ou du "faux" réinforcement learning (DPO, ...)) pour la phase d'alignement. Donc notre formule ne marche plus si on prend un modèle "chat" par exemple (les 3/4 des modèles qu'on trouve sur huggingface) et qu'on veut calculer l'influence d'une phrase du jeu de pre-entraînement par exemple. Or, ce sont ces données de pré-entraînement qui nous intéressent puisque la majorité des connaissances d'un LLM sont acquises pendant le pré-entraînement. Sans pouvoir les tracer, on ne peut pas expliquer d'où viennent les réponses du modèle.

Ainsi, les auteurs du papier proposent une connexion entre l'espace des paramètres du modèle pré-entraîné et celui du modèle fine-tuné. L'intuition est que le fine-tuning ne devrait pas trop éloigner les paramètres de leur état pré-entraîné. On reformule donc l'objectif de fine-tuning avec une contrainte de proximité euclidienne :

$$\theta^{ft} = \arg\min_\theta \mathcal{L}_{ft}(\theta) + \frac{\alpha}{2}||\theta - \theta^{pt}||_2^2$$

où :
- $\mathcal{L}_{ft}(\theta)$ est la loss de fine-tuning
- $\alpha \in \mathbb{R}^+$ est un hyperparamètre contrôlant la proximité
- $\;||\theta - \theta^{pt}||_2^2\;$ est la distance euclidienne entre les paramètres du modèle pré-entraîné avec le modèle fine-tuné (final)

Avec cette reformulation, on peut dériver la fonction d'influence multi-étapes :

$$I_f(z_m) = \nabla_\theta f(\theta^{ft})^T \left(\nabla^2_\theta \mathcal{L}_{ft}(\theta^{ft}) + \alpha I\right)^{-1} \left(\nabla^2_\theta \mathcal{L}_{pt}(\theta^{pt})\right)^{-1} \nabla_\theta \mathcal{L}(z_m, \theta^{pt})$$

Ainsi, on a 2 hessiennes:
- **Hessienne du pré-entraînement** : $\left(\nabla^2_\theta \mathcal{L}_{pt}(\theta^{pt})\right)^{-1}$
   - Calculée aux paramètres $\theta^{pt}$ (modèle pré-entraîné)
   - Capture la courbure de la loss de pré-entraînement
- **Hessienne du fine-tuning** : $\left(\nabla^2_\theta \mathcal{L}_{ft}(\theta^{ft}) + \alpha I\right)^{-1}$
   - Calculée aux paramètres $\theta^{ft}$ (modèle fine-tuné)
   - Inclut le terme de régularisation $\alpha I$ qui encode la contrainte de proximité

Cette double inversion de Hessienne permet de :
- **Première inversion** : Transformer le gradient de l'exemple de pré-entraînement en changement de paramètres
- **Seconde inversion** : Propager ce changement à travers le fine-tuning pour voir son impact final

C'est comme si on "remontait" l'influence à travers deux étapes d'entraînement successives.

### 1.8 Beaucoup de sujets récents de recherche utilisent les fonctions d'influence pour déterminer les données utiles (qu'on peut utiliser pour fine-tuner le modèle) pour améliorer la génération d'un LLM, ou ajouter une "connaissance" au modèle par exemple

TODO

## 2. Mesurer sur quelles parties de l'input le modèle s'est basé pour faire sa prédiction: les cartes de saillance (saliency maps)

- **Objectif** : voir comment la **sortie** du modèle $f(x)$ varie si on modifie chaque composante $x_i$ de l’entrée.
- **Formule** :
  $$
    S_i(x) \;=\; \frac{\partial\,f(x)}{\partial\,x_i}
    \quad\Longrightarrow\quad
    \text{Carte }S(x) = [S_1(x), S_2(x), \dots]
  $$
- Application :  
  - Pour une image, chaque $x_i$ est un pixel / un patch de l'image ;  
  - Pour du texte, chaque $x_i$ est un token.

Le résultat est une carte où chaque valeur $S_i(x)$ indique l’importance du pixel (ou du token) $i$ pour la prédiction du modèle.