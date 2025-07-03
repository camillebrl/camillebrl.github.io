---
layout: post
title: Biais Positionnels dans les transformers auto-régressifs
date: 2025-06-16 22:00:00
description: Description du biais positionnel dans les transformers auto-régressifs
tags: XAI, biais, transformers
categories: sample-posts
maths: true
giscus_comments: true
thumbnail: assets/img/positional_biais/u-shape.PNG
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
tikzjax: true
giscus_comments: true
---

## Introduction sur le biais positionnel

### Le biais positionnel : c'est quoi ?
Le papier [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/pdf/2307.03172) décrit comment, pour des contextes très longs, les LLMs se concentrent surtout sur les débuts et la fin du prompt.

<div style="display: flex;">
<div style="flex: 0 0 60%;">
Le biais positionnel dans les LLMs, c'est la tendance du modèle à se concentrer excessivement sur certaines parties de l'entrée, qu'importe la sémantique. Et cette concentration influence significativement les performances et la fiabilité des transformers: en fonction de l'ordre des éléments dans le prompt, le modèle va générer des réponses différentes. Notamment, plus on  a un contexte long, plus le modèle a tendance à se concentrer sur les éléments au début et à la fin du prompt (effet "lost in the middle").
</div>
<div class="image">
    <div class="u-shape">
        {% include figure.liquid loading="eager" path="assets/img/positional_biais/u-shape.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


### Le biais positionnel : ça vient d'où ?

- **Le masque causal**  
  Avec un masque causal, chaque token ne peut voir que les tokens qui le précèdent. Cela signifie que les tokens situés au début sont accessibles à presque tous les calculs d’attention ultérieurs, alors que ceux situés plus tard ne le sont qu’à partir d’un certain point.
  <script type="text/tikz">
  \begin{tikzpicture}[scale=0.2, rotate=90]
    \foreach \y in {0,...,4} {
        \foreach \x in {0,...,4} {
        % On remplit si y > x (au-dessus de la diagonale)
        \ifnum \x>\y
          % Rien
        \else

          \fill[blue] (\x,\y) rectangle (\x+1,\y+1);
        \fi
      }
    }
    \draw[step=1,gray] (0,0) grid (5,5);
  \end{tikzpicture}
  </script>


- **L’encodage de position** (relatif, RoPE…)  
  Ce type d'encodage de position favorisent la proximité avec le token courant. Ainsi, les jetons situés près de la fin de la séquence bénéficient d’une attention renforcée, car leur représentation est plus fortement influencée par la similarité de position avec le token en cours de génération.
<script type="text/tikz"> 
\begin{tikzpicture}[baseline=(x1.base)]
  %------------------------------------------
  % 1) Placement des noeuds sur une SEULE ligne
  %------------------------------------------
  \node (x1)  at (0,0)       {$x_1$};
  \node (x2)  at (1.5,0)     {$x_2$};
  \node (x3)  at (3.0,0)     {$x_3$};
  \node (dots)at (4.5,0)     {$\dots$};
  \node (xn2) at (6.0,0)     {$x_{n-2}$};
  \node (xn1) at (7.5,0)     {$x_{n-1}$};
  \node (xn)  at (9.0,0)     {$x_n$};

  %------------------------------------------
  % 2) Tracé des flèches courbes vers le HAUT
  %    depuis x_n vers les autres noeuds
  %------------------------------------------
  % On utilise les options out=... et in=... pour définir
  % des angles qui font monter les arcs au-dessus de la ligne.
  % Angles ajustables pour éviter tout chevauchement.
  \draw[->, thick, red!70!black]
       (xn.north) to[out=135, in=45] (x1.north);
  \draw[->, thick, red!90!black]
       (xn.north) to[out=130, in=50] (x2.north);
  \draw[->, thick, red!50]
       (xn.north) to[out=125, in=55] (x3.north);
  \draw[->, thick, green!80!black]
       (xn.north) to[out=120, in=60] (xn2.north);
  \draw[->, thick, green!50]
       (xn.north) to[out=115, in=65] (xn1.north);

\end{tikzpicture}
</script>

## Qu'est-ce que l'encodage de position ?
> L'encodage de position est nécessaire dans les transformers à cause du calcul de l'attention qui prend en compte tous les azutres tokens de la séquence.
> Sans encodage positionnel, chaque token identique aurait la même influence, peu importe l’endroit dans la séquence :  
> “Le chat mange la souris” et “La souris mange le chat” seraient équivalents.
> Avec l'encodage de position, deux tokens identiques mais à des positions différentes auront des vecteurs différents.

1. **Encodage absolu** (sinusoïdal ou appris)  
   Ajouté aux embeddings initiaux. Limité pour extrapoler à des longueurs supérieures à celles vues à l’entraînement.

2. **Encodages relatifs**  
   - T5 : biais appris pour chaque distance relative.  
   - Alibi : biais fonctionnel selon la distance.

3. **RoPE** (Rotary Positional Encoding)  
   rotation appliqué à la représentation intermédiaire de chaque token dépendante de la position du token au niveau du calcul du score d'attention (chaque paire de dimensions du vecteur est tournée dans le plan d’un angle qui dépend de sa position dans la séquence). Mais du coup, quand on applique cette rotation à query et key, et qu’on fait un produit scalaire, le dot product dépend de la position relative $i−j$, même si on encode chaque position de manière absolue.
   <div class="image">
        <div class="periodic_attn">
            {% include figure.liquid loading="eager" path="assets/img/positional_biais/periodic_attn.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
        </div>
   </div>


# SOTA des approches pour corriger le biais positionnel

## 1. Modifier le mécanisme d’attention

- **Stable Mask** ([2402.04779](https://arxiv.org/pdf/2402.04779))  
  Ajout d’un “pseudo-score” \(-\gamma\times(j-1)\) présente une approche de "compensation" du score d'attention excessif sur les premiers tokens en ajoutant un "pseudo-score": $-\gamma \times (j-1)$ pour la j-ième position, qui diminue au fur et à mesure de la séquence. 

- **Calibration du score d’attention** ([2406.16008](https://arxiv.org/pdf/2406.16008)) présente une approche dans laquelle on a le score d'attention de la query avec un doc à la position k qui est fonction ($f$) de la pertinence du doc $\text{rel}(\text{doc}_k)$ et du biais positionnel à la position k $b_k$. En simplifiant la fonction $f$ (rasoir d'Occam) par une fonction linéaire, on a alors $\text{Attn}(\text{query}, doc_k) = \text{rel}(\text{doc}_k) + b_k + \epsilon$. \textbf{Donc en gros ils supposent que le score d'attention entre le document et la query est une combinaison linéaire du biais de position et de la pertinence réelle du document avec la query}. Pour isoler $\text{rel}(\text{doc}_k)$, ils \textbf{introduisent un document "dummy" à la même position $k$ que le document}: $\text{doc}_\text{k;dum}$, on a alors $\text{rel}(\text{doc}_k) = \text{Attn}(\text{query}, doc_k) -  \text{Attn}(\text{query}, doc_{k;dum}) + \epsilon$. Grâce à la pertinence "effective" de chaque document, ils calculent, $\forall k$, un coefficient de rééchelonnement $\alpha_k$, qui est une fonction Softmax appliquée sur le score de pertinence du document $\text{doc}_k$.

- **Attention bidirectionnelle entre documents** (PCW [2212.10947](https://arxiv.org/pdf/2212.10947)) propose une modification de l'attention entre documents pour un traitement "indépendant": Au lieu d'utiliser l'attention causale (unidirectionnelle) qui impose un ordre strict. A contrario, le papier [l'approche d'attention bidirectionnelle entre documents](https://openreview.net/attachment?id=fvkElsJOsN&name=pdf) propose une approche d'attention bidirectionnelle qui permet à chaque document d'interagir équitablement avec tous les autres.
  <div class="image">
        <div class="bidirectional_attn">
            {% include figure.liquid loading="eager" path="assets/img/positional_biais/bidirectional_attn.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
        </div>
        <div class="cropped_attn">
            {% include figure.liquid loading="eager" path="assets/img/positional_biais/cropped_attn.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
        </div>
  </div>

- **Diminuer le biais positionnel dû à RoPE**: L'objectif principal de [ROPE](https://arxiv.org/pdf/2104.09864) est d'encoder l'information positionnelle de sorte que le produit scalaire des embeddings de requête et de clé contienne intrinsèquement l'information relative à la position, c'est-à-dire $f(q_m, m)^T f(k_n, n) = f(q_m, k_n, m - n)$. Ici, $f$ est la fonction d'encodage positionnel appliquée aux embeddings de requête et de clé aux positions $m$ et 
$n$, respectivement. Pour satisfaire cette condition, la fonction $f$ est définie comme une fonction complexe vectorielle : $f(x, m) = x e^{im\theta} = \left[(x_1 + i x_2)e^{im\theta_1};(x_3 + i x_4)e^{im\theta_2};\ldots;(x_{l-1} + i x_l)e^{im\theta_{l/2}},\right]^T$. Dans cette équation, 
$l$ représente la dimension des embeddings, \(\theta_k = 10000^{-2k/l}\), et 
$i$ est l'unité imaginaire. Pour le calcul du score d'attention, RoPE considère la partie réelle du produit, spécifiquement $\operatorname{Re}\Bigl(f(q_m, m)^T f(k_n, n)\Bigr).$ La fonction trigonométrique $\cos{((m-n) \theta)}$ est périodique, d'où la waveform qu'on obtient. pour certaines valeurs de $(m−n)$, le cosinus (et le sinus) prend des valeurs élevées (les "pics"), tandis que pour d'autres il prend des valeurs faibles (les "creux"). Ainsi, si une information cruciale se trouve à une position qui correspond à un trough de l'oscillation, son score d'attention sera relativement bas. La fréquence des oscillations est modulée par rapport à $\theta_k$: Donc ce qui se fait couremment pour contrebalancer ça, c'est de prendre plusieurs bases de $\theta$, qui fixe l'échelle exponentielle à laquelle les fréquences décroissent.

<div class="image">
  <div class="periodic_attn">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/periodic_attn.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

Ainsi, plusieurs approchyes visent à contre-balancer le biais positionnel induit par RoPE:


- **Attention Bucket** ([2312.04455](https://arxiv.org/pdf/2312.04455))  
  Plusieurs bases \(\theta\) traitées en parallèle.

- **MS-PoE** ([2403.04797](https://arxiv.org/pdf/2403.04797))  
  C'est une approche qui applique un facteur $r$ à la position des tokens par tête, modifiant du coup l'oscillation par tête $m/r \theta$ afin de garder la base RoPE sur laquelle a été entraîné le modèle (dans une approche de correction du biais à l'inférence)
  <div class="image">
        <div class="ms_poe">
            {% include figure.liquid loading="eager" path="assets/img/positional_biais/ms_poe.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
        </div>
        <div class="attn_bucket">
            {% include figure.liquid loading="eager" path="assets/img/positional_biais/attn_bucket.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
        </div>
  </div>

## 2. Jouer sur les données

- **IN2 training** ([2404.16811](https://arxiv.org/pdf/2404.16811))  
  Fine-tuning sur des QA avec contextes longs formés par concaténation aléatoire de segments courts.

- **Réordonnancement des documents** ([ACL-Long’24](https://aclanthology.org/2024.acl-long.91.pdf))  
  Placer les plus pertinents en début ou fin de prompt.


# Introduction du papier « Mitigate Positional Biais via Scaling a Single Dimension »

- Le biais positionnel provient des patterns d’attention (focus sur début/fin).  
- Causal mask + encodage positionnel génèrent des “dimensions positionnelles” dans les hidden-states.  
- **Objectifs** :  
  1. Identifier ces dimensions.  
  2. Les modifier (scaling) pour diminuer le biais.

<div class="image">
  <div class="pos_hidden_state">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/pos_hidden_state.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>


# Tâche de retrieval utilisée

- **Clés/valeurs aléatoires** pour isoler le pure retrieval.  
- On mesure  
  \[
    A_G = \frac{1}{|G|} \sum_{j\in G} a_{l,j}
  \]  
  où \(l\)=dernier token (interrogateur), \(G\)=positions de la clé, \(a_{l,j}\)=poids d’attention.  

<div class="image">
  <div class="task">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/task.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

# Identification des dimensions positionnelles

1. **Monotonie** : \(h(p)\) doit être strictement croissante ou décroissante.  
2. **Smoothness** : dérivée seconde faible (évolution “douce”).  
3. Choix de la dimension \(t\) et de l’échelle \(s<1\) minimisant la perte  
   \[
     \arg\min_{h_t,s<1} \mathbb{E}\Bigl[\sum_{i=1}^{|P|} \mathcal{L}(x,y,p_i; F(\theta,h_t,s))\Bigr]
   \]  
   avec \(F(\theta,h_t,s)\) le modèle scaled sur la \(t\)-ième dimension.

<div class="image">
  <div class="algo">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/algo.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>


# Construction de \(h(p_i)\)

- Approximation par **moindres carrés segmentés** pour lisser le signal.  
- Hypothèse : \(\textbf{hidden\_state}_i = h(p_i) + \epsilon_i\).  
- Permet de distinguer tendance monotone et bruit.


# Résultats de l’identification

- Tendance monotone dès la couche 1, s’amplifie dans les couches supérieures.  

<div class="image">
  <div class="example_mistral">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/example_mistral.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>


# Dimensions causées par le masque causal

- Réduction du PE de 200 pour positions 400–600 : effet mineur.  
- Rogner le masque causal pour ces positions : fortes fluctuations.  

<div class="image">
  <div class="causal_mask_modification_perturb_pos_dim">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/causal_mask_modification_perturb_pos_dim.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

# Correction proposée

- Prouvé que la performance biasée vient des patterns d’attention : en doublant le score à la position 25, on déplace l’attention.  

<div class="image">
  <div class="attn_weights_distrib">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/attn_weights_distrib.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

- **Modification** : scaler la dimension \(p\) uniquement pour le calcul du score d’attention du dernier token \(l\) :
  \[
  z =
  \begin{cases}
    \mathrm{Softmax}\bigl((q_i K^\top + \mathrm{Mask})/\sqrt d\bigr)V, & i<l,\\[0.5em]
    \mathrm{Softmax}\bigl((\bar q_l\;\bar K^\top)/\sqrt d\bigr)V, & i=l.
  \end{cases}
  \]


# Effet du scaling

- \(s>1\) → focus début ; \(s<0\) → focus fin.  
- \(s\in[-1,0.5]\) → distribution équilibrée.  
<div style="display: flex; justify-content: space-around;">
![Scaling hidden](assets/img/positional_biais/scaling_pos_hidden_states.PNG)
![Scaling factor](assets/img/positional_biais/scaling_factor.PNG)
</div>

- **Varie selon les modèles** :  
<div class="image">
  <div class="examples_scaling_factors">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/examples_scaling_factors.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

# Performances

Gain significatif sur LongBench :  
<div class="image">
  <div class="longbench_perf">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/longbench_perf.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>


# Limitations (selon moi)

1. Hypothèses de monotonie et smoothness de $h(p)$ très discutables et infondées.  
2. Lien causal masque => hidden-states pas rigoureusement démontré. le fait qu’il y ait du biais positionnel dans les états cachés est un postulat de départ de ce papier non prouvé proprement.  
3. Approche de scaling de "la dimension positionnelle" discutable

<script type="text/tikz">
\begin{tikzpicture}[
    font=\sffamily,
    >=stealth,
    line cap=round,
    line join=round,
    thick
]

%------------------------------------------
% 1) Causal Mask (remplir AU-DESSUS de la diagonale)
%------------------------------------------
\node[minimum width=1.8cm, minimum height=1.8cm, align=center] (mask) {
  \begin{tikzpicture}[scale=0.2, rotate=90]
    \foreach \y in {0,...,4} {
        \foreach \x in {0,...,4} {
        % On remplit si y > x (au-dessus de la diagonale)
        \ifnum \x>\y
          % Rien
        \else

          \fill[blue] (\x,\y) rectangle (\x+1,\y+1);
        \fi
      }
    }
    \draw[step=1,gray] (0,0) grid (5,5);
  \end{tikzpicture}
};
\node at ($(mask.north)+(0,5pt)$) {\footnotesize Causal Mask};

% Flèche horizontale vers la droite
\draw[->] (mask.east) -- ++(0.8,0);

%------------------------------------------
% 2) Positional information in hidden states (courbe pointillée)
%------------------------------------------
\begin{scope}[xshift=3.0cm]
  \node[minimum width=2.2cm, minimum height=2.2cm] (posinfo) {};
  \node at ($(posinfo.north)$) {\footnotesize Positional dim in hidden states};


  \begin{scope}[
    xshift={$(posinfo.south west)!0.5!(posinfo.south east)$},
    yshift={$(posinfo.south west)!0.5!(posinfo.north west)$},
    xscale=0.6, yscale=0.6
  ]
    % Axes
    \draw[->] (-1.0,-1.0) -- (-1.0,1.0);
    \draw[->] (-1.1,-0.9) -- (1.0,-0.9);

    % Courbe pointillée
    \draw[blue, very thick]
      (-1.0,-0.9) .. controls (-0.4,0) .. (1.0,0.7);
  \end{scope}
\end{scope}

%------------------------------------------
% 3) Attention Pattern (courbe en U)
%------------------------------------------
\begin{scope}[xshift=6.0cm]
  \node[minimum width=2.2cm, minimum height=2.2cm] (attn) {};
  \node at ($(attn.north)$) {\footnotesize Attention Pattern};

  \begin{scope}[
    xshift={$(attn.south west)!0.5!(attn.south east)$},
    yshift={$(attn.south west)!0.5!(attn.north west)$},
    xscale=0.6, yscale=0.6
  ]
    % Axes
    \draw[->] (-1.0,-1.0) -- (-1.0,1.0);
    \draw[->] (-1.1,-0.9) -- (1.0,-0.9);

    % Courbe en U
    \draw[black, very thick]
      (-1.0,0.5) .. controls (0,-0.9) .. (1.0,0.5);
  \end{scope}
\end{scope}

% Flèche horizontale entre posinfo et attn
\draw[->] (posinfo.east) -- (attn.west);

%------------------------------------------
% 4) Positional hidden states (en-dessous, rapproché verticalement)
%------------------------------------------
\begin{scope}[xshift=3.0cm, yshift=-4.0cm]
  \node[minimum width=2.2cm, minimum height=2.2cm] (poshidden) {};
  \node at ($(poshidden.north)$) {\footnotesize Modified Positional dim};

  \begin{scope}[
    xshift={$(poshidden.south west)!0.5!(poshidden.south east)$},
    yshift={$(poshidden.south west)!0.5!(poshidden.north west)$},
    xscale=0.6, yscale=0.6
  ]
    % Axes
    \draw[->] (-1.0,-1.0) -- (-1.0,1.0);
    \draw[->] (-1.1,-0.9) -- (1.0,-0.9);

    % Courbe moins pentue
    \draw[blue, very thick]
      (-1.0,-0.5) .. controls (-0.3,0) .. (1.0,0.2);
  \end{scope}
\end{scope}

% Flèche verticale (posinfo -> poshidden)
\draw[->, red, dashed] ($(posinfo.south)+(0,7pt)$) --
  node[right]{\scriptsize $\times s \in [-1:0.5]$ => Pourquoi on multiplie par un scalaire $s$? Approche très empirique}
  ($(poshidden.north)+(0,6pt)$);

%------------------------------------------
% 5) Deuxième Attention Pattern (U moins prononcé)
%------------------------------------------
\begin{scope}[xshift=6.0cm, yshift=-4.0cm]
  \node[minimum width=2.2cm, minimum height=2.2cm] (attn2) {};
  \node at ($(attn2.north)$) {\footnotesize Modified Attention Pattern};

  \begin{scope}[
    xshift={$(attn2.south west)!0.5!(attn2.south east)$},
    yshift={$(attn2.south west)!0.5!(attn2.north west)$},
    xscale=0.6, yscale=0.6
  ]
    % Axes
    \draw[->] (-1.0,-1.0) -- (-1.0,1.0);
    \draw[->] (-1.1,-0.9) -- (1.0,-0.9);

    % U moins prononcée
    \draw[black, very thick]
      (-1.0,0.2) .. controls (0,-0.3) .. (1.0,0.2);
  \end{scope}
\end{scope}

% Flèche du bloc "Modified positional hidden states" vers la nouvelle "Attention Pattern"
\draw[->] (poshidden.east) -- (attn2.west);

\end{tikzpicture}
</script>

# Points forts du papier

- Séparation mécanique sémantique vs. position. Un autre papier étudie cette séparation pour comprendre quelle tête ont les parties positionnelles et sémantiques dans les transformers (notamment l'aspect low-rank et low-frequency de la partie positionnelle): [Uncovering hidden geometry in Transformers via disentangling position and context](https://openreview.net/pdf?id=1M0qIxVKf6). Un autre papier ([Exploring Context Window of Large Language Models via Decomposed Positional Vectors](https://openreview.net/pdf?id=zeYyq0GpXO)) cherche à décomposer les états cachés de chaque token en partie sémantique et positionnelle en faisant une estimation empirique des vecteurs positionnels sur un grand nombre d'exemples.
- Peu de papiers visent à corriger le biais positionnel dû au masque causal, donc j'ai trouvé ce papier intéressant qui s'attaque à cela au lieu de s'attaquer au biais positionnel dû à l'encodage positionnel.


# Pour aller plus loin : découpler sémantique et position

## Spline-based Transformers (ECCV’25)
- Suppression de l’encodage positionnel.  
- Introduction de *tokens de contrôle* formant une spline :  
  \[
    s(t)=\sum_{i=0}^n N_{i,k}(t)\,\mathbf p_i
  \]  
- Position implicite via la géométrie.  
<div class="image">
  <div class="spline_based_transformers">
      {% include figure.liquid loading="eager" path="assets/img/positional_biais/spline_based_transformers.PNG" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

## Decomposed Positional Vectors (NeurIPS)
- Décomposition : \(h_{l,t}=p_{l,t}+cs_{l,t}\).  
- Estimation : \(p_{l,t}=\tfrac1N\sum_{s=1}^N h^{(s)}_{l,t}\).  
- Isolation : \(cs_{l,t}=h_{l,t}-p_{l,t}\).

