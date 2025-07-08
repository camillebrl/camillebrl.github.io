---
layout: distill
title: Les approches d'explicabilité des LLMs
date: 2025-07-02 22:00:00
description: Méthodes d'explicabilité de la génération de texte par les LLMs
tags: XAI, influence, mechanistic, saliency, attention, transformers
categories: sample-posts
maths: true
featured: true
giscus_comments: true
thumbnail: assets/img/xai_llm.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
toc: true
---

Un LLM génère des tokens à partir d'autres tokens selon un processus probabiliste. <mark>Pour chaque séquence de tokens fournie en entrée, le modèle retourne une distribution de probabilité sur l'ensemble des tokens suivants possibles</mark>. Un token représente un élément du vocabulaire du modèle, dont la taille est fixe et définie dans la configuration du tokenizer ou du modèle lui-même. 
Prenons l'exemple des modèles Qwen avec un <mark>vocabulaire de 151 936 tokens : à chaque étape de génération, le modèle calcule une distribution de probabilité sur ces 151 936 possibilités</mark>. De manière autorégressive, chaque nouveau token généré est ajouté à la séquence d'entrée pour prédire le suivant, et ainsi de suite.
<mark>Comprendre pourquoi un modèle génère certains tokens plutôt que d'autres soulève plusieurs questions cruciales</mark>. La séquence d'entrée (prompt) peut contenir des éléments variés, notamment des chunks de texte dans le cas du RAG (Retrieval-Augmented Generation), qui servent de contexte pour répondre aux questions. Pour véritablement comprendre le processus de génération, nous devons répondre à trois interrogations principales :
- <mark>Quelles parties spécifiques du prompt influencent la génération de quels tokens ?</mark> Cette question est particulièrement pertinente dans le contexte du RAG où différents passages peuvent avoir des impacts variés sur la réponse finale.
- <mark>Sur quelles connaissances acquises pendant l'entraînement le modèle s'appuie-t-il pour générer chaque token ?</mark> Comment distinguer ce qui provient du prompt de ce qui provient de la mémoire du modèle ?
- <mark>Quelles parties du modèle (couches, têtes d'attention, neurones) sont responsables de quelles décisions ?</mark> Pourquoi des modèles de tailles différentes, même issus de la même famille, produisent-ils des réponses différentes ?
Pour répondre à ces questions, nous devons nous tourner vers les approches d'explicabilité. Nous pouvons classifier les approches d'explicabilité des LLMs en 4 "familles", comme présentées dans le papier [Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety](https://arxiv.org/pdf/2506.05451):
{% include figure.liquid loading="eager" path="assets/img/explainability_llms/survey.png" class="img-fluid rounded z-depth-1" zoomable=true %}
- celles de <mark>"training data attribution" (ou "training" dans l'image), identifiant les données d'entraînement qui ont un fort impact (positif ou négatif) sur la génération du LLM</mark> , notamment à l'aides des fonctions d'influence, 
- celles de <mark>"context attribution" (ou "input" dans l'image), identifiant quelles parties des données d'entrée (input) a un impact sur quelle partie de la génération du LLM</mark>  (à l'aide des cartes de saillance et des cartes d'attention notamment), 
- celles d'<mark>explicabilité mécanistique (ou "inference" dans l'image), consistant à trouver les circuits / éléments du LLM qui capturent certains concepts</mark>, avec des circuits identifiés par approches d'observation de l'espace latent (ACP sur les états cachés du LLM à différents niveaux du réseau), de dictionnary learning pour visualiser quels concepts sont capturés par chaque neurones / groupes de neurones, ou par "patching", en modifiant certains états cachés pour voir l'effet sur le modèle. 
- Celles <mark>étudiant la génération des modèles (ou "generation" dans l'image), notamment des modèles dits de "raisonnement" (thinking) qui détaillent leur raisonnement dans la réponse qu'ils fournissent</mark> : cette approche est d'autant plus utile aujourd'hui que les agents consistent en plusieurs appels du LLMs (un appel qui retourne un appel à un outil (tool), un autre qui retourne l'appel à un autre outil, etc).

Nous allons <mark>d'abord observer les approches d'explicabilité par les données, que ce soit par les données d'entraînement ou par les données d'input (du prompt)</mark>. L'objectif est d'établir des liens causaux clairs entre ces sources de données et les tokens générés, qu'ils soient pris individuellement ou en groupes de tokens.

# I/ Approches TDA (training data attribution) pour comprendre quelles connaissances apprises dans l'entraînement ont été influentes dans la génération de quels tokens

## 1. Les fonctions d'influence pour estimer l’impact d’un exemple d’entraînement sur la prédiction d’un exemple de test

Si on cherche à estimer l'<mark>impact qu'aurait un exemple d'entraînement sur la perte d'un exemple de test à un exemple d'entraînement</mark> (qu'il soit dans le jeu de données d'entraînement de base ou pas), on peut utiliser les fonctions d'influence:
$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) = \frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\,\theta_\varepsilon\bigr)\Big|_{\varepsilon=0} = -\,g_{\mathrm{test}}^\top\,H_\theta^{-1}\,g_{\mathrm{train}}
$$

Les librairies [Kronfluence](https://github.com/pomonam/kronfluence) ou [PINNfluence](https://github.com/aleks-krasowski/PINNfluence) ou [GraSS](https://github.com/TRAIS-Lab/GraSS). Des repo github proposent des tutos pour plusieurs de ces lib: cf [Influenciae](https://github.com/deel-ai/influenciae). Les fonctions d'influence permettent de répondre à ces questions:
    - Est-ce que je devrais <mark>ajouter cet exemple dans mon set d'apprentissage pour améliorer les performances de cette prédiction?</mark>
    - Quels <mark>exemples d'entraînement ont été utiles à la prédiction</mark> de mon modèle?
    - Le modèle s'est trompé: sur quels exemples d'entraînement s'est-il basé pour cette mauvaise prédiction?
  
Dans ce [repo](https://github.com/camillebrl/llm_training_data_attribution), j'ai créé une interface pour calculer l'influence des données d'entraînement (de pre-training) de LLMs (base : modèles pré-entraînés uniquement) en 2 phases: d'abord en diminuant les données d'entraînement sur lesquelles calculer l'influence à l'aide d'elasticsearch pour identifier les 50 phrases les plus similaires au prompt et au texte généré par le LLM, ensuite en utilisant [Kronfluence](https://github.com/pomonam/kronfluence) pour calculer l'influence de ces données sur la génération en question. Les scores d'influence sont ensuite normalisés par la norme au carrée de la loss, comme réalisé dans les approches d'état de l'art.

### 1.1 Introduction sur les fonctions d'influence
Pour introduire les fonctions d'influence appliquées au deep learning, nous nous basons sur le le papier [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/pdf/1703.04730), et notamment sur l'annexe A pour expliquer les différentes formules.

**L’influence de** $z_{\rm train}$ **sur** $z_{\rm test}$ ($\mathrm{Influence}(z_{\rm train} \to z_{\rm test})$) **se définit comme** 

<mark>
$$
\mathrm{Influence}(z_{\rm train}\to z_{\rm test})\;=\; \left.\frac{d}{d\varepsilon}\,\mathcal{L}\bigl(z_{\rm test},\,\theta_\varepsilon\bigr)\right|_{\varepsilon=0}
$$
</mark> 

**L’influence de** $z_{\rm train}$ **sur** $f(\theta)$ ($\mathrm{Influence}(z_{\rm train} \to f(\theta))$) **se définit comme** 

<mark>
$$
\mathrm{Influence}(z_{\rm train}\to f(\theta))\;=\; \left.\frac{d}{d\varepsilon}\,\mathcal{L}\bigl(f(\theta),\,\theta_\varepsilon\bigr)\right|_{\varepsilon=0}
$$
</mark> 

En d’autres termes, elle <mark>mesure la sensibilité de la loss de $z_{\rm test}$ (ou de toute fonction $f(\theta)$) à un « up-weight » infinitésimal de la loss de $z_{\rm train}$.</mark> 


A noter: 
- C'est l'**impact de l"up-weight" de la loss sur $z_\text{train}$ sur qqch qu'on mesure avec les fonctions d'influence**. En fait pour mesurer l'impact de l"up-weight" de $z_{\text{train}}$ dans la loss globale, on se pose la question: "on se pose la question : **"<mark>si je donnais un peu plus de poids à ce terme de loss dans l’objectif global, comment cela ferait-il bouger mes paramètres et, avec ces nouveaux paramètres, ma performance sur un point de test, ou sur une fonction?</mark>"** . En effet, en deep learning, modifier le poids d'une donnée dans l'entraînement, c'est modifier le poids qu'on donne à sa loss dans l'apprentissage.
- <mark>$f(\theta)$ peut être n'importe quelle fonction (exemple: la moyenne des prédictions sur un ensemble de données types</mark> (cf le papier [Which Data Attributes Stimulate Math and Code Reasoning? An Investigation via Influence Functions](https://arxiv.org/pdf/2505.19949) qui cherche à calculer l'influence des textes d'entraînement sur la génération de code (moyenne de des log probabilité de la génération de chaque token de code générés dans un benchmark sachant un problème de code en langage naturel à résoudre)), la différence entre 2 prédictions du modèle, ...)

Nous allons entrer dans le détail de comment on calcule cette influence:

$$
\begin{align}
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) &= I_{z_\text{test}}\bigl(z_{\mathrm{train}}\bigr) \\
&= \frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\ \theta_\varepsilon\bigr) \Big|_{\varepsilon=0} \\
&= -\nabla_\theta \,\mathcal{L}\bigl(z_{\mathrm{test}},\,\theta_\varepsilon\bigr)\,H_\theta^{-1}\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)
\end{align}
$$


**Détaillon un peu comment on a obtenu cette formule (le (3))...**

On cherche 

$$
\frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\ \theta_\varepsilon\bigr) \Big|_{\varepsilon=0}
$$

On peut voir ça comme $\frac{d}{d\varepsilon} f(g(\varepsilon))$ avec $g(\varepsilon) = \theta_\varepsilon$ et $f(\theta_\varepsilon) = \mathcal{L}(z_{\rm test},\theta_\varepsilon)$, d'où, <mark>par chain rule</mark> (
$$
f'(g(\varepsilon)) = f'\bigl(g(\varepsilon)\bigr)\,g'(\varepsilon)
$$
), on a :

$$
\colorbox{yellow}{
$\displaystyle
\frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\ \theta_\varepsilon\bigr) \Big|_{\varepsilon=0} = \nabla_\theta \mathcal{L}(z_{\rm test}, \theta_\varepsilon)\Big|_{\varepsilon=0} \times \frac{d}{d\varepsilon} \theta_\varepsilon \Big|_{\varepsilon=0}
$}
$$

Du coup, <mark>dans un premier temps, il nous faut calculer</mark>

$$
\colorbox{yellow}{
$\displaystyle
\frac{d}{d\varepsilon} \theta_\varepsilon \Big|_{\varepsilon=0}
$}
$$

**Calculer cela revient à se demander : <mark>comment $\theta_\varepsilon$ varie autour de $\theta$ quand on up-weight très légèrement (voisinage de 0) la loss de notre exemple $z_\text{train}$</mark>?**

<mark>Pour faire cet "up-weight" de la loss de $z_{\text{train}}$ d'un tout petit $\varepsilon$, on perturbe la fonction de perte en ajoutant un petit coefficient $\varepsilon$ sur la perte de $z_{\rm train}$ et on voit comment les paramètres optimaux $\theta$ évoluent avec $\varepsilon$.</mark> 

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

<mark>Ce qui revient à chercher les $\theta_\varepsilon$ dont le gradient de cette nouvelle loss en $\theta$ est nul</mark> , car $\theta_\varepsilon$ est un minimum local de $R_\varepsilon$ si et seulement si sa dérivée première (gradient) s'annule:
$$
\nabla_\theta R_\varepsilon\bigl(\theta_\varepsilon\bigr) = 0.
$$
<mark>
En développant à l'aide de l'approximation de Taylor en $\theta$ pour un tout petit $\varepsilon$, ie proche de 0, donc pour un $\theta_\varepsilon$ proche de $\theta$, cette formule</mark>:
$$
\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta_\varepsilon)
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)
$$
*PS: Rappel de la formule de Taylor à l'ordre 1:*
$$
f(x) \approx f(x_0) + f'(x_0)\,(x - x_0)
$$

On obtient:

$$
\begin{gather}
\nabla_\theta R_\varepsilon\bigl(\theta_\varepsilon\bigr)  = \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta_\varepsilon)
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) \approx \\
[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta)
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta) ]
+ \; \;(\theta_\varepsilon - \theta) \;\; [ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon)
\;+\;\varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) ]
\end{gather}
$$

$$\nabla_\theta R_\varepsilon\bigl(\theta_\varepsilon\bigr) = 0$$ nous ramène donc à la formule:

$$
\left[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta) + \varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta) \right] + (\theta_\varepsilon - \theta) \left[ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon) + \varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) \right] = 0
$$

Ou, en modifiant un peu l'équation:

$$
(\theta_\varepsilon - \theta) = - \left[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta) + \varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta) \right] \times \left[ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon) + \varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) \right]^{-1}
$$

Or, <mark>puisque $\theta$ sont les poids optimaux pour le modèle de base</mark>, $\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta) = 0$ et $$\varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon) = 0$$:

$$
\colorbox{orange}{$\displaystyle(\theta_\varepsilon - \theta)$} = - \colorbox{cyan}{$\displaystyle \varepsilon $} \,\colorbox{pink}{$\displaystyle \nabla_\theta \mathcal{L}(z_{\rm train},\theta)  \times  [\frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon)]^{-1} $}
$$

Et si on <mark>dérive par rapport à $\varepsilon$</mark> on obtient:

$$
\begin{split}
\frac{d}{d \colorbox{cyan}{$\displaystyle \varepsilon $}}\colorbox{orange}{$\displaystyle(\theta_\varepsilon - \theta)$} &= \colorbox{yellow}{$\displaystyle \frac{d}{d\varepsilon} \theta_\varepsilon $} \\
&= - \colorbox{pink}{$\displaystyle \nabla_\theta \mathcal{L}(z_{\rm train},\theta)  \times  [\underbrace{\frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\theta_\varepsilon)}_{\colorbox{red}{$\displaystyle H_\theta $}}]^{-1} $} \\
&= - \nabla_\theta \mathcal{L}(z_{\rm train},\theta)  \colorbox{red}{$\displaystyle  H_\theta $}^{-1}
\end{split}
$$

D'où : 

<mark>
$$
\colorbox{yellow}{
$\displaystyle
\frac{d}{d\varepsilon} \theta_\varepsilon = - \nabla_\theta \mathcal{L}(z_{\rm train},\theta)H_\theta^{-1}
$}
$$
</mark>

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
#### 1.2.1 Données d'entraînement mal apprises prédominantes: comment corriger?
A noter que le papier [Influence Functions in Deep Learning Are Fragile](https://arxiv.org/pdf/2006.14651) indique que les <mark>fonctions d'influence sont biaisées vers les exemples à forte perte</mark>. en effet, le <mark>gradient de la loss des points d'entraînement est plus élevé pour les exemples mal appris</mark>, entraînant un <mark>biais systématique vers ces exemples dans l’attribution d’influence, indépendamment de leur véritable effet sur la fonction qu'on veut mesurer ou sur la loss sur notre $z_\text{test}$</mark>. Le papier propose de recalculer les scores d'influence en <mark>normalisant les gradients par la norme de la loss au carré</mark>, éliminant ainsi les gradients élevés pour des exemples mal appris (sans lien avec la fonction à maximiser).  

$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) = \frac{\frac{d}{d\varepsilon} \mathcal{L}\bigl(z_{\rm test},\theta_\varepsilon\bigr)}{||\nabla_\theta \mathcal{L}(z)||^2} \Big|_{\varepsilon=0} = \frac{-\nabla_\theta \,f(\theta_\varepsilon)H_\theta^{-1}\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)}{||\nabla_\theta \mathcal{L}(z)||^2}
$$

### 1.3 Les limites de l'influence appliquée au deep learning et comment contrer cela

Le papier [If Influence Functions are the Answer, Then What is the Question?](https://arxiv.org/pdf/2209.05364) explique que les **fonctions d’influence** n’approximent pas fidèlement le retraining « leave-one-out », mais qu’elles correspondent en fait à la **fonction de réponse de Bregman proximale** (PBRF), une formulation plus locale autour des paramètres entraînés.

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/pbrf.png" class="img-fluid rounded z-depth-1" zoomable=true %}

En effet:

Dans les LLMs, souvent surparamétrés, les optima peuvent être non uniques. La matrice hessienne $H_\theta$ devient alors parfois singulière, empêchant l’existence d’une fonction de réponse unique. De plus, on n’entraîne généralement pas un réseau jusqu’à convergence totale, d’une part pour limiter le coût de calcul, d’autre part pour éviter le surapprentissage. Hors optimum, l’interprétation de l’équation :
$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) = \frac{\frac{d}{d\varepsilon} \mathcal{L}\bigl(z_{\rm test},\theta_\varepsilon\bigr)}{||\nabla_\theta \mathcal{L}(z)||^2} \Big|_{\varepsilon=0} = \frac{-\nabla_\theta \,f(\theta_\varepsilon)H_\theta^{-1}\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)}{||\nabla_\theta \mathcal{L}(z)||^2}
$$ n’est plus claire et la hessienne peut présenter des valeurs propres négatives.

<mark>La fonction de réponse de Bregman proximale offre une meilleure approximation des fonctions d’influence dans le contexte du deep learning : elle ajoute un terme d’amortissement $\lambda$ et utilise un linéarisé de Gauss–Newton $G$ pour corriger les problèmes de singularité, de non-convergence et de non-convexité</mark>. Concrètement, la <mark>PBRF repose sur une hessienne de Gauss–Newton amortie $G + \lambda I$, toujours définie positive, garantissant une réponse bien définie</mark>.

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/pbrf2.png" class="img-fluid rounded z-depth-1" zoomable=true %}

Les fonctions d’influence (trait noir pointillé) et la PBRF (trait rouge) appréhendent différemment la modification locale du paysage de perte :  
- En réglant la pondération d’un exemple $z_m$, la PBRF suit la trajectoire qui minimise/maximise la perte tout en restant proche de $\theta_\varepsilon$.  
- Les <mark>fonctions d’influence classiques se bornent à une expansion de Taylor d’ordre 1 autour de $\epsilon=0$, valable seulement en présence d’une fonction strictement convexe et d’un optimum unique</mark>.


On va voir en quoi ça consiste, cet "amortissement" ($\lambda$) (1.3.1) et ce "gauss-newton" ($G$) (1.3.2):

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


#### 1.3.4 Factorisation par blocs de $G_\theta$ et factorisation en produit de Kronecker pour pourvoir stocker cette matrice & paralléliser les calculs entre couches

Au lieu d’inverser directement la grande matrice $\,G_\theta+\lambda I$, le papier [Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization](https://arxiv.org/pdf/2505.05017) exploite sa structure en blocs correspondant à chaque couche du réseau.

$$
G 
= \begin{bmatrix}
  G_{1,1} & G_{1,2} & \cdots\\
  G_{2,1} & G_{2,2} & \\
  \vdots & & \ddots
\end{bmatrix},
$$

Pour un réseau à $L$ couches, on a donc $$G = [G_{i,j}]_{1 \leq i, j \leq L}$$, avec $$G_{i,j}$$ qui représente le bloc entre les paramètres de la couche $i$ et de la couche $j$.

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
Le papier [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/pdf/2308.03296) présente l'application des fonctions d'influence aux LLMs. Dans le cas des LLMs, la loss est la negative log-vraisemblance. La première <mark>particularité d'un LLM, c'est le fait qu'un datapoint est un peu compliqué à définir. On peut supposer qu'il s'agit d'une phrase (et son label, le token suivant la phrase), ou on peut considérer le token lui-même</mark> (l'input étant la phrase le précédent, le label le token en question, par exemple). Mais il est important de bien définir de quoi on parle quand on parle de "datapoint".

#### 1.4.1 L'influence à l'échelle de la phrase $z_m$

Supposons que $z_m$ soit cette phrase "le chat est gris", soit ce datapoint:

$$[\text{BOS, le, chat, est, gris}] \rightarrow \text{[EOS]}$$

Vu qu'on est dans un cas autorégressif (c'est-à-dire que les tokens sont prédits à partir des tokens précédents de la séquence) :

Si $$z_m$$ de taille $$T$$ :

$$
\nabla_\theta L(z_m, \theta) = \sum_{t=1}^T(-\nabla_\theta \log p(z_{m,t} \mid z_{m, \lt t}, \theta))
$$

Nous, $$z_m$$ est de taille 6 :

$$
\begin{split}
\nabla_\theta L(z_m, \theta) &= -\nabla_\theta \log p(\text{le} \mid \text{[BOS]}, \theta) \\
&\quad - \nabla_\theta \log p(\text{chat} \mid \text{[[BOS], le]}, \theta) \\
&\quad - \nabla_\theta \log p(\text{est} \mid \text{[BOS], le, chat}, \theta) \\
&\quad - \nabla_\theta \log p(\text{gris} \mid \text{[BOS], le, chat, est}, \theta) \\
&\quad - \nabla_\theta \log p(\text{[EOS]} \mid \text{[BOS], le, chat, est, gris}, \theta)
\end{split}
$$

#### 1.4.2 L'influence à l'échelle des tokens $t\;\;$ dans la phrase $z_m$

On peut aussi considérer l'échelle du token, comme on l'a mis plus haut, en considérant l'input comme étant la phrase précédant ce token, et le label ce token en question.

On a ici $\nabla_\theta L(z_m, \theta)$ qui est la somme des gradients de la loss au niveau de chaque token. Du coup, il suffit de prendre $- \nabla_\theta \log p(\text{gris} \mid \text{[BOS], le, chat, est}, \theta)$ pour avoir l'influence du token gris dans la séquence par exemple. On peut ainsi avoir l'information token par token.

Et ça c'est ce qu'on obtient ici (cf [Studying Large Language Model Generalization with Influence Functions, Grosse 2023](https://arxiv.org/pdf/2308.03296)) :

$$\nabla_\theta L(\text{token t dans } z_m, \theta) = \nabla_\theta \log p(\text{token t} \mid \text{ce qui est avant token t dans } z_m, \theta)$$

On obtient donc la formule:

$$I_f(z_{m,t}) = \nabla_{\theta}f(\theta)^{T} H^{-1} \nabla_{\theta}\log p(z_{m,t}\mid z_{m,\lt t}, \theta)$$

Prenons l'exemple suivant: on prend $f = \log p(\text{"hydrogen and oxygen"} \mid \text{"Water is composed of"})$ et $z_m$ qui est le texte ci-dessous. On peut afficher l'influence token par token dans le texte:

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/image.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### 1.5 Le cas des LLMs: beaucoup de données d'entraînement (eg. 36 trillions de tokens pour Qwen3) => query batching ou semantic matching pour ne pas calculer l'influence sur toutes les données (trop coûteux)
Le papier [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/pdf/2308.03296) propose une approche pour éviter de calculer les gradients de tous les exemples d'entraînement candidats pour chaque requête d'influence. Pour cela, ils "filtrent" les données d'entraînement par rapport à la phrase test via un filtrage TF-IDF et une approche qu'ils introduisent de "query batching".

#### 1.5.1 Le filtrage TF-IDF
Le filtrage TF-IDF utilise une technique classique de recherche d'information pour présélectionner les séquences d'entraînement les plus susceptibles d'être influentes. L'intuition derrière est que les séquences pertinentes devraient avoir au moins un certain chevauchement de tokens avec la requête.

Ils retiennent les top 10,000 séquences selon le score TF-IDF Calcul d'influence et calculent les influences uniquement sur ces séquences présélectionnées. 

#### 1.5.2 Le Query-Batching

Dans un LLM, on a beaucoup d'exemples $z_m$ d'entraînement. Donc, on calcule séparemment $\nabla_\theta\mathcal{L}(z_m, \theta_\varepsilon)$ et $\nabla_{\theta} f(\theta_\varepsilon)^\top \, H^{-1}$ qui se calcule en une fois. 

Pour stocker de nombreux gradients de requêtes en mémoire ($\nabla_\theta\mathcal{L}(z_m, \theta_\varepsilon) \; \forall \; z_m$), ils approximent chaque matrice de gradient préconditionné comme étant de rang faible (rank-32 dans leurs expériences).

Ainsi, pour chaque requête, ils n'ont pas à refaire les calculs! Ils ont juste à calculer $\nabla_{\theta} f(\theta_\varepsilon)$.

### 1.6 Le cas des LLMs: plusieurs couches d'entraînement (pretraining, fine-tuning, alignement, ...) => multi-stage influence functions

Le papier [Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization](https://arxiv.org/pdf/2505.05017) explique que la fonction d'influence classique $I_f(z_{m,t}) = \nabla_{\theta}f(\theta)^{T} H^{-1} \sum_{t=1}^T(-\nabla_\theta \log p(z_{m,t} \mid z_{m, <t}, \theta))$ permet de quantifier l'impact d'une phrase d'entraînement sur les prédictions du modèle. Cependant, on a des modèles qui sont passés par plusieurs phases d'entraînement pour les LLMs (avec plusieurs données différentes). En effet, les LLMs sont pré-entraînés (modèles "base"), puis instruct-tuné (modèles "chat"), puis passent par du reinforcement learning (ou du "faux" réinforcement learning (DPO, ...)) pour la phase d'alignement. Donc notre formule ne marche plus si on prend un modèle "chat" par exemple (les 3/4 des modèles qu'on trouve sur huggingface) et qu'on veut calculer l'influence d'une phrase du jeu de pre-entraînement par exemple. Or, ce sont ces données de pré-entraînement qui nous intéressent puisque la majorité des connaissances d'un LLM sont acquises pendant le pré-entraînement. Sans pouvoir les tracer, on ne peut pas expliquer d'où viennent les réponses du modèle.

Ainsi, les auteurs du papier proposent une connexion entre l'espace des paramètres du modèle pré-entraîné et celui du modèle fine-tuné. L'intuition est que le fine-tuning ne devrait pas trop éloigner les paramètres de leur état pré-entraîné. On reformule donc l'objectif de fine-tuning avec une contrainte de proximité euclidienne :

$$\theta^{ft} = \arg\min_\theta \mathcal{L}_{ft}(\theta) + \frac{\alpha}{2}||\theta - \theta^{pt}||_2^2$$

où :
- $\mathcal{L}_{ft}(\theta)$ est la loss de fine-tuning
- $\alpha \in \mathbb{R}^+$ est un hyperparamètre contrôlant la proximité
- $$\|\theta - \theta^{pt}\|_2^2$$ est la distance euclidienne entre les paramètres du modèle pré-entraîné avec le modèle fine-tuné (final)

Avec cette reformulation, on peut dériver la fonction d'influence multi-étapes :

$$I_f(z_m) = \nabla_\theta f(\theta^{ft})^T \left(\nabla^2_\theta \mathcal{L}_{ft}(\theta^{ft}) + \alpha I\right)^{-1} \left(\nabla^2_\theta \mathcal{L}_{pt}(\theta^{pt})\right)^{-1} \nabla_\theta \mathcal{L}(z_m, \theta^{pt})$$

Ainsi, on a 2 hessiennes:
- **Hessienne du pré-entraînement** : $$\left(\nabla^2_\theta \mathcal{L}_{pt}(\theta^{pt})\right)^{-1}$$
   - Calculée aux paramètres $\theta^{pt}$ (modèle pré-entraîné)
   - Capture la courbure de la loss de pré-entraînement
- **Hessienne du fine-tuning** : $$\left(\nabla^2_\theta \mathcal{L}_{ft}(\theta^{ft}) + \alpha I\right)^{-1}$$
   - Calculée aux paramètres $\theta^{ft}$ (modèle fine-tuné)
   - Inclut le terme de régularisation $\alpha I$ qui encode la contrainte de proximité

Cette double inversion de Hessienne permet de :
- **Première inversion** : Transformer le gradient de l'exemple de pré-entraînement en changement de paramètres
- **Seconde inversion** : Propager ce changement à travers le fine-tuning pour voir son impact final

C'est comme si on "remontait" l'influence à travers deux étapes d'entraînement successives.

### 1.7 Beaucoup de sujets récents de recherche utilisent les fonctions d'influence pour déterminer les données utiles (qu'on peut utiliser pour fine-tuner le modèle) pour améliorer la génération d'un LLM, ou ajouter une "connaissance" au modèle par exemple

Les fonctions d’influence sont un bon moyen d’évaluer l’impact de chaque exemple de données sur les performances d’un LLM dans un domaine donné. Pour l’instruction-tuning, elles permettent de mesurer précisément quels couples question-réponse contribuent le plus à la qualité des réponses générées dans un domaine donné. Pour le continual pretraining, elles identifient les phrases dont l’ajout ou la suppression modifie le plus la capacité du modèle à maîtriser un vocabulaire ou un style spécifique. Les fonctions d'influence sont de plus en plus utilisées pour affiner le corpus d’apprentissage et maximiser le gain de performance du LLM sur un domaine cible. Voici quelques exemples de papier faisant cela:

[IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment](https://arxiv.org/pdf/2505.12762)

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration IDEAL</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/IDEAL.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

[Not All Documents Are What You Need for Extracting Instruction Tuning Data](https://arxiv.org/pdf/2505.12250)

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration de EQUAL</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/EQUAL.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

## 2. Les approches d'attribution aux données d'entraînement par similarité sémantique
Pour trouver les données d'entraînement influentes pour la génération du LLM, on peut également simplement étudier la similarité sémantique entre la génération du LLM et les données d'entraînement.

Le papier [Wrapper Boxes: Faithful Attribution of Model Predictions to Training Data](https://aclanthology.org/2024.blackboxnlp-1.33.pdf) compare l'embedding de la génération du LLM à celui de la penultième couche, et entraîne ensuite trois types de « wrapper boxes » sur ces embeddings:
- un k-nearest neighbors (kNN) pour retrouver, à chaque inférence, les k exemples d’entraînement les plus proches ;
- un clustering k-means pour assigner l’input à un cluster dont on peut exposer le centroïde et les exemples qui le composent ;
- un arbre de décision dont chaque feuille correspond à un sous-ensemble d’exemples d’entraînement utilisés pour la classification.

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration de Wrapper Boxes</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/wrapper_boxes.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

## 3. Les approches d'attribution aux données d'entraînement basées sur le gradient

Ces approches sont différentes des fonctions d'influence classiques, qu'on a vu plus tôt. En effet, ces fonctions d'influence calculent comment la perte sur un échantillon test, ou comment une fonction du modèle sur un échantillon donné changerait si on retirait / up-weightait un échantillon d'entraînement. Ici, les approches basées sur le gradient utilisent une notion plus simple: on calcule la distance (cosinus) entre 2 gradients: celui de la loss sur l'échantillon d'entraînement dont on veut voir l'influence avec celui de la loss sur l'exemple cible sur lequel on veut mesurer l'influence. on a donc $$\text{DABUF-inf}(z_\text{train}, z_\text{target}) = \cos(\nabla L(z_\text{train}, \theta), \nabla L(z_\text{target}, \theta))$$. C'est ce que fait par exemple le papier [Detecting and Filtering Unsafe Training Data via Data Attribution](https://arxiv.org/pdf/2502.11411) pour identifier les données d'entraînement "unsafe". Plus précisément, ils normalisent ce calcul par la norme des gradients:

$$
\text{DABUF-inf}(z_\text{train}, z\text{target}) =
\eta \cdot \cos(\nabla\ell(z_\text{target}; \theta), \nabla\ell(z; \theta))
= \eta \cdot \frac{\nabla\ell(z_\text{target}; \theta) \cdot \nabla\ell(z; \theta)}{||\nabla\ell(z_\text{target}; \theta)|| \cdot ||\nabla\ell(z; \theta)||}
$$

Plus généralement, cette approche est similaire à [TracIn](https://github.com/frederick0329/TracIn), qui lui ne calcule pas la distance cosinus entre les gradient mais effectue un produit scalaire simple entre eux. 

## 4. Les approches d'attribution aux données d'entraînement basées sur les valeurs de Shapley

les valeurs de shapley estiment l'effet moyen de sous-ensembles d'un dataset en supprimant certaines variables explicatives (ici, certains datapoints du dataset) et en mesurant la différence de prédiction du modèle, puis en moyennant ces effets marginaux sur toutes les permutations possibles.

L'utilisation des valeurs de Shapley pour déterminer la contribution de datapoints à la prédiction du modèle peut être vue comme une évaluation des contributions marginales [Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning](https://arxiv.org/pdf/2110.14049). La contribution marginale d’un point de données $z_i$ à un sous-ensemble de taille $k$ de l’ensemble d’entraînement est définie comme:

$$
\Delta^{D_N}_{z_i}(k, D_T)
:= \frac{1}{\binom{n-1}{k}}
\sum_{\substack{S \subseteq N \setminus \{i\} \\ |S|=k}}
\bigl(U(S\cup\{i\},D_T) - U(S,D_T)\bigr).
$$ 

L'approche Leave-one-out (cf fonctions d'influences plus haut) considère la variation de la précision du modèle lorsqu’on retire le point cible de l’ensemble d’entraînement. Cette LOO peut être interprétée comme la contribution marginale de $z_i$ à $D_N\setminus\{z_i\}$. Les approches "leave-one-out" ou ses approximations (comme les fonctions d'influence, cf plus haut), ne considèrent que la contribution marginale de $z_i$ à un unique sous-ensemble de taille $n-1$ (avec $n$ étant la taille du jeu d'entraînement du modèle). Ainsi, le score d’instance d’un exemple chute fortement lorsqu’un autre exemple similaire apparaît dans l’ensemble d’entraînement (cf papier [Data Shapley: Equitable Valuation of Data for Machine Learning](https://arxiv.org/pdf/1904.02868)), ce qui réduit la fiabilité et la robustesse de la mesure.

Au lieu de ne considérer qu’un sous-ensemble, la valeur de Shapley la généralise en tenant compte de l’impact de $z_i$ sur tous les sous-ensembles. En gros, au lieu de considérer:
$$
g^{\mathrm{LOO}}(z_i, D_T, D_N)
= \Delta^{D_N}_{z_i}(n-1, D_T).
$$
On considère:
$$
g^{\mathrm{Shap}}(z_i, D_T, D_N)
= n^{-1}
\sum_{k=0}^{n-1}
\Delta^{D_N}_{z_i}(k, D_T).
$$
Ceci peut être vu comme la moyenne des contributions marginales de $z_i$ à des sous-ensembles de toutes tailles possibles. 

Cependant, calculer la valeur de Shapley sur un ensemble d’entraînement de taille $n$ exige un nombre exponentiel d’ajustements du modèle (soit $n!$), ce qui nécessite ajustements lorsque les datasets sont très grands (comme pour les LLMs).


Le papier [Helpful or Harmful Data? Fine-tuning-free Shapley Attribution for Explaining Language Model Predictions](https://openreview.net/pdf?id=WSpPC1Jm0p) propose une méthode d'attribution aux données d'entraînement basée sur les valeurs de Shapley, appelée FreeShap (Fine-tuning-free Shapley Attribution), basée sur la théorie du noyau tangent neuronal (NTK) pour éviter de multiples réentraînements du modèle. 

En effet, la théorie du noyau tangent neuronal (NTK) a été proposée pour étudier la dynamique d’entraînement des réseaux de neurone. Les travaux [On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/pdf/1904.11955) et [Toward a theory of optimization for over-parameterized systems of non-linear equations: the lessons of deep learning](https://www.researchgate.net/publication/339642345_Toward_a_theory_of_optimization_for_over-parameterized_systems_of_non-linear_equations_the_lessons_of_deep_learning) montrent que l’entraînement d’un réseau entièrement suffisamment large est équivalent à la résolution d’une régression par noyau avec le NTK à l’initialisation aléatoire. Cependant, deux défis se posent lorsque l’on applique la théorie NTK au fine-tuning des LMs : (1) l’utilisation quasi systématique de poids préentraînés au lieu d’une initialisation aléatoire ; (2) l’emploi de prompts. Le papier de 2023 [A Kernel-Based View of Language Model Fine-Tuning](https://arxiv.org/pdf/2210.05643) étend l’analyse pour montrer que résoudre la régression par noyau avec le NTK empirique (eNTK) calculé à partir des poids préentraînés peut reproduire le fine-tuning basé sur des prompts. De plus, la régression par noyau utilisant l’eNTK obtient des performances comparables au fine-tuning en computer vision (cf le papier [More Than a Toy: Random Matrix Models Predict How Real-World Neural Representations Generalize](https://arxiv.org/pdf/2203.06176)) et en NLP (cf papier [A Kernel-Based View of Language Model Fine-Tuning](https://arxiv.org/pdf/2210.05643)), ce qui en fait un substitut prometteur.

L’eNTK se calcule à partir du Jacobien de la sortie du modèle pour un point de données $x_i$ :
$$
\psi(x_i)\;:=\;\frac{\partial f(x_i;\theta_0)}{\partial \theta_0}
\;\in\;\mathbb{R}^{C\times P},
$$
où $\theta_0\in\mathbb{R}^P$ désigne les poids préentraînés.  
Pour un ensemble d’entraînement $D_S$ d’indices $S:=\{1,\dots,k\}$, on pose
$$
X_S \;:=\;
\begin{bmatrix}x_1 \\ \vdots \\ x_k\end{bmatrix}
\in\mathbb{R}^{k\times d}, 
\quad
Y_S \;:=\;
\begin{bmatrix}
y^1_1 & \cdots & y^C_1 \\
\vdots &        & \vdots \\
y^1_k & \cdots & y^C_k
\end{bmatrix}
\;\in\;\{0,1\}^{k\times C},
$$
où $y^p_j=1$ si la classe réelle de $x_j$ est $p$.  
Une instance de test $x_t$ est alors prédite par le modèle de régression eNTK :
$$
f_S^{\mathrm{eNTK}}(x_t)
\;=\;
K(x_t, X_S)^\top
\,K(X_S, X_S)^{-1}\,
Y_S,
\tag{5}
$$
avec
$$
K(x_t, X_S)
=\bigl[\psi(x_t)\,\psi(x_i)^\top\bigr]_{i=1}^k
\;\in\;\mathbb{R}^{C\times k},
\quad
K(X_S, X_S)
=\bigl[\psi(x_i)\,\psi(x_j)^\top\bigr]_{i,j=1}^k
\;\in\;\mathbb{R}^{kC\times kC}.
$$

# II/ Approche Input (prompt) attribution: mesurer sur quelles parties de l'input le modèle s'est basé pour faire sa prédiction 

Ces approches consistent à attributer la génération du LLM à des tokens du prompt pour comprendre quel(s) token(s) est/sont responsable(s) de quelle(s) parties de la génération du LLM. 

Dans ce [papier: A Close Look at Decomposition-based XAI-Methods for Transformer Language Models, 2025](https://arxiv.org/pdf/2502.15886), ils comparent plusieurs approches (qu'on va détailler ci-dessous) de context / input attribution:

<details>
  <summary style="cursor: pointer;">Cliquez la comparaison des tokens du prompts influents sur la génération du LLM par différentes approches comme présentées dans le papier</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/comparison.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>


## 1. Les approches basées sur l'attention

Ces approches affichent les poids d'attention de chaque token générés par rapport aux tokens du prompt (cas auto-regressif). Des poids d'attention élevés signifient que le modèle a donné plus d'importance à ce token pour la génération du token en question.

Cependant, les poids d'attention sont donnés dans le réseau au niveau de chaque tête d'attention. Il est compliqué d'obtenir une attention "globale" que le modèle donne aux tokens. En effet, les scores d'attention ne montrent que les connexions locales (au sein de chaque tête de chaque couche du réseau), et surtout, ces patterns d'attention ne donnent tel quel aucune information : il s'agit de patterns uniformes. Plusieurs approches visent donc à modéliser ce flux d'attention global à travers le réseau pour tracer l'influence de chaque token d'entrée sur les tokens prédits. 

Tout l'enjeu est d'arriver à mesurer, de manière la plus fidèle et informative possible, le flux d'attention au sein du réseau pour donner une vue globale de l'influence de chaque token d'entrée sur un token généré.

### 1.1 Visualisation du LLM comme un graphe d'attentions pour obtenir la contribution finale du token

Le papier [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385.pdf) propose deux approches: une d'attention rollout et une d'attention flow. Ces 2 approches modélisent le LLM comme un graphe d'attention et utilisent chacune une méthode différente pour calculer l'attention globale d'un token sur le token prédit.

La première approche proposée est dite d'"attention rollout". Elle consiste à multiplier récursivement les matrices d'attention à travers toutes les couches. L'idée est de représenter le LLM comme un graphe d'attentions: Si chaque arête représente la proportion d'information transférée, multiplier les poids le long d'un chemin donne le flux total. 

La seconde approche est dite d'"attention flow", avec dans l'idée de traiter le graphe d'attention comme un réseau de flux (flow network). Il utilise un algorithme de flux maximum où les capacités des arêtes sont les poids d'attention. Dans cette approche, le poids d'un chemin est le minimum des poids (pas le produit).

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration des poids d'attention selon ces 2 méthodes</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/vis_attn.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

### 1.2 La contribution globale d'un token associée aux logits au sein des différents blocs d'attention (et MLP)
ALTI-Logit [Explaining How Transformers Use Context to Build Predictions](https://aclanthology.org/2023.acl-long.301.pdf) suit comment chaque token d'entrée contribue à la prédiction finale en traversant le réseau, en prenant en compte la contribution de chaque élément au logit, ainsi que l'ensemble des transformations (notamment au niveau du MLP) réalisées sur les scores d'attention pour évaluer cette contribution. En gros, pour calculer la contribution finale d'un token, ALTI-logit trace comment ce token (au logit du mot évalué) se propage à travers les connexions d'attention et les connexions résiduelles de toutes les couches, en tenant compte du mélange d'information entre tokens à chaque étape.

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration la composition d'un block transformers</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/transformer_layer.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

$$
\begin{aligned}
x^L_T &= x^0 + \sum_{l=1}^L \Bigl(\Delta_{\mathrm{MHA}}^l + \Delta_{\mathrm{MLP}}^l\Bigr),\\
\Delta_{\mathrm{MHA}}^l &= \text{contribution du bloc MHA de la couche }l,\\
\Delta_{\mathrm{MLP}}^l &= \text{contribution du bloc MLP de la couche }l.
\end{aligned}
$$

Pour chaque couche $l$ :
$$
\begin{aligned}
\Delta_{\mathrm{MLP}}^l &= \mathrm{sortie}^l_{\mathrm{MLP}} \;-\; \mathrm{entrée}^l_{\mathrm{MLP}},\\
\mathrm{relevance}_{\mathrm{MLP}}^l &= \Delta_{\mathrm{MLP}}^l \;\cdot\; U_w,
\end{aligned}
$$
où $U_w$ est le vecteur d'unembedding de sortie pour le token suivant.

\textbf{Exemple (couche 10)} :
$$
\begin{aligned}
\mathrm{entrée}_{\mathrm{MLP}}^{10} &= [0.1,\;0.2,\;\dots],&
\mathrm{sortie}_{\mathrm{MLP}}^{10} &= [0.3,\;0.5,\;\dots],\\
\Delta_{\mathrm{MLP}}^{10} &= [0.2,\;0.3,\;\dots],&
\Delta_{\mathrm{MLP}}^{10} \cdot \mathrm{embedding}_{\text{dort}} &= 1.5.
\end{aligned}
$$

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration de ALTI-logit</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/alti_logit.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

### 1.3 Visualisation de l'attention (sa propagation dans le réseau) comme un graphe à plusieurs niveaux
On peut voir l'attention comme un graphe à plusieurs niveaux: en effet, il y a d'abord les tokens (premier niveau, groupe A) sur lequels le token prédit par le LLM porte son attention. Mais il y a ensuite les tokens sur lesquels les tokens du groupe A portent leur attention (second niveau, groupe B), et ainsi dessuite. On peut remonter ainsi l'attention à plusieurs niveaux. 

Afin d'identifier comment les tokens du prompt sont traitées par le LLM dans ses différentes têtes d'attention pour amener à l'output, le papier [Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking](https://arxiv.org/pdf/2402.14811) teste d'abord toutes les têtes d'attention pour voir lesquelles, quand patchées (modification par une version bruitée de la tête d'attention: ici obtenue à l'aide de 2 fine-tuning différents: l'un bruité et l'autre d'origine), affectent le plus le logit final. Les têtes qui causent la plus grande chute de performance sont celles qui "regardent" directement la réponse correcte (Ces têtes forment le Groupe A). Le papier cherche ensuite quelles têtes d'attention ont un effet direct important sur les têtes du Groupe A, en patchant les chemins allant de candidats potentiels vers le Groupe A.


<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration de l'approche de la propagation de l'attention par groupes</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/path_patching2.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

Le papier [On the Emergence of Position Bias in Transformers](https://arxiv.org/pdf/2502.01951) représente en effet le masque d'attention comme un graphe $G$ dirigé où:
- Les nœuds représentent les tokens de la séquence
- une arrête $(j,i) \in E(G)$ signifie que l'attention du token $i$ est portée sur le token $j$

La propagation de l'information au sein des différentes couches du réseau se fait comme cela:

Pour une couche unique: $$X^{(1)} = A^{(0)}X^{(0)}W_V^{(0)}$$,
Pour 2 couches: $$X^{(2)} = A^{(1)}A^{(0)}X^{(0)}W_V^{(0)}W_V^{(1)}$$
Pour t couches:
$$X^{(t+1)}_{i,:} = \sum_{j=1}^{N} \underbrace{(A^{(t)} \cdots A^{(0)})_{ij}}_{P^{(t)}(z_i=j|X^{(0)})} \cdot \underbrace{X^{(0)}_{j,:} W_V^{(0)} \cdots W_V^{(t)}}_{f^{(t)}(X^{(0)}_{z_i,:})}$$

Le produit cumulatif $P^{(t)} = A^{(t)} \cdots A^{(0)}$ représente la probabilité cumulative que le token $i$ sélectionne le token $j$ comme contexte après $t$ couches. Cela capture tous les chemins possibles de longueur $t+1$ entre les tokens $j$ et $i$ dans le graphe.

Le papier [Chain and Causal Attention for Efficient Entity Tracking](https://aclanthology.org/2024.emnlp-main.731.pdf) généralise ce pattern d'attention "multi-niveau" en proposant un nouveau mécanisme d'attention qui, en une seule couche, permet toutes les connexions d'attention précédemment faites entre les couches. Pour cela, il utilise une représentation formelle avec le produit cumulatif des matrices d'attention qui permet de capturer toutes les longueurs de chemins en une seule couche. En effet, le papier interprète la matrice d'attention $A$ comme une matrice d'adjacence d'un graphe dirigé pour capturer toutes les dépendances possibles :
$$A + A^2 + A^3 + \cdots = A(I - A)^{-1}$$, avec $A$ connexions directes (chemins de longueur 1), $A^2$ chemins de longueur 2, $A^3$ chemins de longueur 3, etc. L'attention qu'ils proposent est calculée comme :

$$
\boxed{Y = (1 - \gamma) \cdot A(I - \gamma A)^{-1}V}
$$

où $\gamma \in [0, 1)$ est un hyperparamètre qui assure la convergence de la série $A + \gamma A^2 + \gamma^2 A^3 + \cdots$ et permet l'interpolation entre attention standard ($\gamma = 0$) et ChaCAL ($\gamma \approx 1$).



### 1.4 Intérêt des approches basées sur l'attention: "facilement" manipulables pour "forcer" le modèle à porter / ne pas porter son attention sur certains tokens

Le papier [Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs](https://openreview.net/pdf?id=xZDWO0oejD) force l'attention du LLM vers des parties spécifiques du prompt désignées par l'utilisateur, à la manière dont nous utilisons le gras ou l'italique dans les textes humains pour diriger l'attention du lecteur.

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration de l'approche PASTA</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/pasta.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

Le papier [The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Analysis of Orthogonal Safety Directions](https://arxiv.org/pdf/2502.09674) propose une approche plus complexe, pour neutraliser l'attention sur les tokens déclencheurs de jailbreak, qu'il identifie en exploitant la structure multi-dimensionnelle de l'espace résiduel de sécurité, construit comme la transformation linéaire des activations pendant le safety fine-tuning. Les tokens déclencheurs sont identifiés non pas par une seule direction, mais par l'interaction entre une direction dominante qui prédit le comportement de refus global, et plusieurs directions non-dominantes qui capturent des patterns spécifiques de jailbreak. Ces directions sont extraites via SVD de la matrice représentant les changements d'activation.

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration du papier du "safety residual space"</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/safe_direction.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>


## 2. Approches basées sur le gradient: les cartes de saillance (saliency maps)

Ces approches mesurent comment la sortie du modèle $f(x)$ varie si on modifie chaque composante $x_i$ de l’entrée. Pour se faire, des libraires comme [Captum](https://github.com/pytorch/captum) ou [Inseq](https://github.com/inseq-team/inseq) ou [Grad-cam](https://github.com/jacobgil/pytorch-grad-cam) ou [Investigate](https://github.com/albermax/innvestigate) ou [Alibi](https://github.com/SeldonIO/alibi) existent.

J'ai créé un [repo github](https://github.com/camillebrl/mirage_ui) qui reproduit un [papier (MIRAGE)](https://aclanthology.org/2024.emnlp-main.347.pdf) d'explication de la génération du modèle à partir des éléments du prompt, découpés document par document (cas du RAG).

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/mirage_illustration.png" class="img-fluid rounded z-depth-1" zoomable=true %}

Ces approches basées sur le gradient consistent à voir comment la sortie du modèle $f(x)$ varie si on modifie chaque composante $x_i$ de l’entrée.

$$
  S_i(x) \;=\; \frac{\partial\,f(x)}{\partial\,x_i}
  \quad\Longrightarrow\quad
  \text{Carte }S(x) = [S_1(x), S_2(x), \dots]
$$

Le résultat est une carte où chaque valeur $S_i(x)$ indique l’importance du token $i$ pour la prédiction du modèle.

Ici, on parle d'attribution de l'importance des tokens du prompt dans la génération du LLM par calcul de gradient, c'est à dire simplement en dérivant la sortie du LLM par rapport aux tokens du prompt. Le problème, c'est que quand la fonction est localement plate (gradient nul), l'attribution est nulle même si l'input est important. L'approche des gradients intégrés (IG) (introduits dans le papier [Axiomatic Attribution for Deep Networks](https://arxiv.org/pdf/1703.01365)), résout ces problèmes en intégrant les gradients le long d'un chemin entre une baseline neutre et l'input réel.


Soit $x = (x_1, x_2, ..., x_n)$ le vecteur d'embeddings correspondant aux tokens du prompt, et $x' = (x'_1, x'_2, ..., x'_n)$ une baseline neutre (typiquement des embeddings nuls ou des tokens de padding). Pour un modèle $F$ produisant une sortie $y = F(x)$, l'importance du $i$-ème token est calculée par :

$$
IG_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha \times (x - x'))}{\partial x_i} d\alpha
$$


Cette formule capture l'accumulation des gradients lorsqu'on interpole linéairement entre la baseline $x'$ et l'entrée réelle $x$. Le terme $(x_i - x'_i)$ représente la différence entre l'embedding du token réel et celui de la baseline, tandis que l'intégrale accumule les sensibilités du modèle tout au long du chemin d'interpolation.

En pratique, l'intégrale est approximée par une somme de Riemann (cas discret) :

$$
IG_i(x) \approx (x_i - x'_i) \times \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m} \times (x - x'))}{\partial x_i} \times \frac{1}{m}
$$

où $m$ est le nombre de pas d'intégration. Cette méthode garantit deux propriétés importantes : la sensibilité (un token n'ayant aucun impact aura une attribution nulle) et la complétude (la somme des attributions égale la différence entre les sorties du modèle pour l'entrée réelle et la baseline).


Le papier [Barkan et al., Improving LLM Attributions with Randomized Path-Integration, 2024](https://aclanthology.org/2024.findings-emnlp.551.pdf) améliore la méthode des gradients intégrés (IG) en introduisant de la randomisation dans le processus d'intégration pour générer de meilleures attributions. En effet, contrairement à IG qui intègre les gradients par rapport aux embeddings d'entrée, RPI intègre les gradients par rapport aux scores d'attention du modèle. Ainsi, pour chaque couche, il calcule les gradients de la prédiction par rapport au tenseur d'attention interpolé le long d'un chemin randomisé entre une baseline aléatoire et les scores d'attention réels.

## 3. Approches basées sur les vecteurs: décomposition des représentations intermédiaires faites par le LLM de chaque token du prompt avec ACP

Une autre approche consiste à étudier l'espace latent (c'est à dire les représentations intermédiaires (à différents niveaux du réseau) du LLM) des tokens d'entrée pour étudier cet espace : comment une perturbation de cet espace modifie la génération du LLM? Comment est structuré cet espace? Est-ce qu'en modifiant / étudiant les directions de cet espace je peux en déduire quelque chose sur la sortie du LLM? 

Le papier [How do Language Models Bind Entities in Context?](https://arxiv.org/pdf/2310.17191) par exemple utilise des perturbations sur les représentations intermédiaires des tokens du prompt pour voir leur impact sur la génération du LLM. 

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration du papier binding ids, perturbant les représentations intermédiaires des tokens du prompt pour voir l'impact sur la génération du LLM</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/binding_ids.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

Le papier [Truth is Universal: Robust Detection of Lies in LLMs](https://arxiv.org/pdf/2407.12831) effectue quand à lui une ACP sur les représentations intermédiaires du dernier token de chaque phrase (le ".") - le modèle étudié étant autorégressif, le dernier token encode la globalité de la phrase. Ils effectuent cette ACP sur les représentations intermédiaires d'un ensemble de données comprenant des phrases vraies (triangles orange dans les figures) et fausses (carrés violets), qu'ils avaient annotées au préalable. Ils observent deux directions principales qui émergent de cette analyse : une direction "générale de vérité" (tG) qui sépare efficacement les déclarations vraies des fausses indépendamment de leur polarité (affirmative ou négative), et une direction "sensible à la polarité" (tP) qui distingue les phrases affirmatives des phrases négatives. Dans l'espace d'activation du modèle, tG pointe systématiquement des déclarations fausses vers les vraies, indépendamment de la polarité grammaticale (affirmative/négative). On peut donc prédire si la génération du LLM va être une hallucination ou non à l'aide de cet espace.

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration du papier qui fait l'ACP sur les représentations intermédiaires d'un ensemble de prompts vrai et faux</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/acp_true_false_statements.PNG" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>


#  III/ Les approches d'explicabilité mécanistique (inférence): comprendre quelle partie du LLM est responsable de quel concept

Le papier [Open Problems in Mechanistic Interpretability](https://arxiv.org/pdf/2501.16496) décrit tous les défis / enjeux d'explicabilité mécanistique.

## 1. Les sondes sur l'espace latent

Ces approches consistent à étudier l'espace latent du modèle en lui donnant différents concepts en entrée (les "sondes").

### 1.1 Les approches de sonde par ablation de l'espace latent

L'ablation de l'espace latent consiste à une mise à zéro ou suppression d’une activation.

- Ablation de couches, têtes d'attention ou paramètres
- Localisation de composants responsables :
 - Hallucinations : Jin et al. (2024), Li et al. (2024a)
 - Jailbreaks : Zhao et al. (2024d), Wei et al. (2024)
 - Biais : Yang et al. (2023b), Ma et al. (2023)

### 1.2 Les approches de sonde par patching de l'espace latent

Le patching consiste à remplacer une activation par une autre.

- Inspiré de l'analyse de médiation causale (Pearl, 2001)
- Remplacement d'activations intermédiaires
- Applications : Hallucinations (Monea et al., 2024), biais (Vig et al., 2020)

Pour étudier les circuits:
- Path patching : Wang et al. (2023), Goldowsky-Dill et al. (2023)


### 1.3 Les approches de sonde par observation (réduction de dimension et autre) de l'espace latent

Il s'agit de projections de l'espace latent calculé sur différents concepts. On affiche donc les représentations des tokens / phrases avec et sans un concept spécifique, pour essayer de trouver un élément de l'espace latent qui les sépare.

Pour se faire, on peut effectuer une moyenne des vecteurs appartenant à un certain concept (hallucinations ([Liu et al., On the Universal Truthfulness Hyperplane Inside LLMs](https://aclanthology.org/2024.emnlp-main.1012.pdf), 2024), jailbreaks ([Arditi et al. Refusal in Language Models Is Mediated by a Single Direction, 2024](https://arxiv.org/pdf/2406.11717))), ou encore des méthodes de réduction de dimension de l'espace latent, comme l'ACP ([Duan et al., Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States](https://arxiv.org/pdf/2402.09733)). D'autres approches entraînent un classifier sur les représentations de l'espace latent (Détection d'hallucinations [Burns et al., Discovering Latent Knowledge in Language Models Without Supervision, 2022](https://arxiv.org/pdf/2212.03827),  [Truth is Universal: Robust Detection of Lies in LLMs, Bürger et al., 2024](https://arxiv.org/pdf/2407.12831), jailbreaks [Zhou et al., How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States, 2024](https://aclanthology.org/2024.findings-emnlp.139.pdf)).

Cependant, ces différents approches supposent que ces concepts sont encodés comme directions linéaires, alors que ce n'est pas vraiment le cas dans les LLMs. 

### 1.4 Ces approches permettent de "corriger" le modèle en dirigeant d'une certaine manière les vecteurs latents

cf: hallucinations [Li et al., Inference-Time Intervention: Eliciting Truthful Answers from a Language Model, 2023](https://arxiv.org/pdf/2306.03341), jailbreaks [Turner et al.,Steering Language Models With Activation Engineering, 2023](https://arxiv.org/pdf/2308.10248)

Exemple des figures de ces papiers pour "corriger" le modèle selon ces direction:

Le papier [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/pdf/2306.03341) propose cette approche:

<details>
  <summary style="cursor: pointer;">Cliquez pour voir l’illustration de l'intervention sur les activations du modèle</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/intervention.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

Le papier [Steering Language Models With Activation Engineering](https://arxiv.org/pdf/2308.10248) propose cette approche:

<details>
  <summary style="cursor: pointer;">Cliquez pour voir une autre illustration d'une intervention sur les activations du modèle</summary>
  {% include figure.liquid 
     loading="eager" 
     path="assets/img/explainability_llms/intervention2.png" 
     class="img-fluid rounded z-depth-1" 
     zoomable=true 
  %}
</details>

Le papier [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/pdf/2406.11717) qui corrige les poids du modèle de la direction $r$ identifiée comme générant des jailbreaks : $x' = x - rr^Tx$. L’**ablation directionnelle** « met à zéro » la composante suivant $r$ pour chaque activation du flux résiduel $x \in \mathbb{R}^{d_{\text{model}}}$. 


## 1.2 Les auto-encodeurs pour apprendre un "dictionnaire" des éléments du réseau (Sparse Dictionnary Learning)
**Analyse des neurones individuels**


**Autoencodeurs épars (SAE)**
- Objectif : Désentrelement des concepts superposés
- Architecture : Encodeur → vecteur épars de concepts → Décodeur
- **Références clés** :
 - Fondamentaux : Sharkey et al. (2022), Bricken et al. (2023)
 - Améliorations : Rajamanoharan et al. (2024a), Templeton et al. (2024)
- **Applications sécurité** :
 - Hallucinations : Ferrando et al. (2025), Theodorus et al. (2025)
 - Jailbreaks : Härle et al. (2024), Muhamed et al. (2025)
 - Biais : Hegde (2024), Zhou et al. (2025a)

**Logit lens**
- Projection des vecteurs latents intermédiaires sur l'espace vocabulaire
- Origine : nostalgebraist (2020), Elhage et al. (2021)
- Améliorations : Belrose et al. (2023), Din et al. (2023)
- Applications : Mécanismes de stockage/rappel (Yu et al., 2023), hallucinations (Yu et al., 2024b)

### 1.2.1 Le principe de superposition : Analyse des neurones individuels

Selon l'hypothèse de superposition, un réseau peut représenter plus de caractéristiques qu'il n'a de dimensions, à condition que chaque feature s'active de manière parcimonieuse. (chaque caractéristique (feature) d’un réseau de neurones ne doit pas être activée tout le temps, mais seulement dans des cas spécifiques et rares)

Ces approches de SDL sont plus sophistiquées et "state of the art" que les approches de réduction de dimension avec ACP ou autre des activations, justement dû au fait de la polysémanticité des neurones. Si un neurone encode à la fois : "nature", "vieux", "lumière" dans un même sous-espace de représentation, l'ACP ne pourra pas dissocier ces trois concepts si leurs activations sont mélangées dans les mêmes dimensions.

En gros, les activations sont traitées par un petit réseau neuronal à deux couches, correspondant respectivement à un encodeur et un décodeur, avec un espace latent large: L’encodeur encode l’activation de chaque variable latente, et le décodeur est une matrice qui sert de dictionnaire des directions latentes.

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/superposition.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- Identification des entrées activant fortement un neurone
- Références : Geva et al. (2021), Foote et al. (2023)
- Défi : Polysémantique des neurones (Arora et al., 2018)
- Fondamentaux : Sharkey et al. (2022), Bricken et al. (2023)
- Améliorations : Rajamanoharan et al. (2024a), Templeton et al. (2024)
- Hallucinations : Ferrando et al. (2025), Theodorus et al. (2025)
- Jailbreaks : Härle et al. (2024), Muhamed et al. (2025)
- Biais : Hegde (2024), Zhou et al. (2025a)

Il existe plusieurs approches de SDL, qui traitent des activations différentes:
- Les autoencodeurs parcimonieux (SAE) (classique)
- Les transcodeurs (cf papier [Transcoders Find Interpretable LLM Feature Circuits](https://openreview.net/pdf?id=J6zHcScAo0))
- Les crosscodeurs (cf article [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html))

### 1.2.2 Analyse de groupes de neurones à différents niveaux dans le réseau
Les concepts sont ici représentées non par par neurone mais sur plusieurs layers. Du coup, il est naturel d'appliquer l'apprentissage de dictionnaires de manière conjointe entre les layers. Les crosscodeurs permettent de gérer la persistance d'une feature sur plusieurs couches.

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/residual_stream.png" class="img-fluid rounded z-depth-1" zoomable=true %}

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/superposition2.png" class="img-fluid rounded z-depth-1" zoomable=true %}

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/crosscoder.png" class="img-fluid rounded z-depth-1" zoomable=true %}

Les crosscodeurs peuvent nous aider en cas de superposition entre couches, mais ils peuvent également être utiles lorsqu'une feature calculée reste dans le flux résiduel pendant de nombreuses couches. Considérons le cycle de vie hypothétique suivant d'une feature à travers le flux résiduel :

{% include figure.liquid loading="eager" path="assets/img/explainability_llms/crosscoder2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
