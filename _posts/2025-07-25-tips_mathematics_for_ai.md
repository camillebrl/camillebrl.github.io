---
layout: distill
title: Tips mathématiques / points utiles en IA
date: 2025-07-25 07:00:00
description: Tips mathématiques / points utiles en IA
tags: maths
categories: sample-posts
maths: true
featured: true
giscus_comments: true
thumbnail: assets/img/tips_maths/illustration.JPG
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
toc: true
---

# Introduction

> Dans ce post, je vais mettre pas mal de <mark>rappels mathématiques</mark> (je reviens aux bases des bases, mais j'ai encore besoin de revenir à ces bases des choses pour comprendre certains concepts...) <mark>qui m'ont été utiles pour comprendre certains papiers de recherche ou débloquer certains points dans mes recherches</mark>.

# Entropie, information mutuelle, cross-entropie
- L’entropie quantifie l’incertitude moyenne d'une variable aléatoire, ie la quantité d'information qu'elle contient: $H(X)=\mathbb{E}[-\log p(X)]$ (discret) et $h(X)=-\int f\log f$ (continu). Elle est maximale pour l’uniforme, nulle pour une variable déterministe ; le conditionnement ne l’augmente pas. Elle est au cœur du codage, de l’inférence statistique (cross-entropie, KL) et de l’information mutuelle. En continu, elle n’est pas invariante par changement d’échelle ; préférer souvent $I$, $D_{\mathrm{KL}}$ pour des comparaisons robustes.
- La cross-entropie entre une "vraie" loi $P$ et un modèle $Q$ est définie par $H(P,Q)=-\sum_x P(x)\log Q(x)$ (ou $\int P\log Q$ au continu). Elle mesure le coût moyen pour coder des échantillons tirés de $P$ avec un code optimal pour $Q$, et vérifie l’identité clé $H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)\ge H(P)$, avec minimum atteint quand $Q=P$. En apprentissage supervisé, la NLL (negative log-likelihood) utilisée pour la classification est exactement une cross-entropie entre les étiquettes (one-hot) et les probabilités du modèle (softmax).
- L’information mutuelle quantifie la dépendance entre deux variables : $I(X;Y)=H(X)-H(X\mid Y)=H(Y)-H(Y\mid X)=H(X)+H(Y)-H(X,Y)$. Elle s’écrit aussi comme une divergence de KL entre la conjointe et le produit des marginales : $I(X;Y)=D_{\mathrm{KL}}\!\big(P_{XY}\Vert P_XP_Y\big)\ge 0$. Elle est nulle **ssi** $X$ et $Y$ sont indépendantes, est **symétrique**, et obéit à l’inégalité de traitement de l’information (si $X\!\to\!Y\!\to\!Z$, alors $I(X;Z)\le I(X;Y)$).
- La Divergence de Kullback–Leibler (KL) est une métrique de "distance" directionnelle entre distributions non symétriques. Cette métrique permet de relier information mutuelle et cross-entropie ($\mathrm{D_{KL}}(P|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}\ \ (\ge 0)$ ; $=0$ ssi $P=Q$)

## L'entropie
L’entropie mesure l’**incertitude** (ou le **désordre**) d’une variable aléatoire. Plus une variable est imprévisible, plus son entropie est grande. On peut aussi la voir comme la quantité moyenne d’**information** (ou de **surprise**) révélée par l’observation de la variable.

### Dans le cas discret
Soit $X$ une variable aléatoire discrète à valeurs dans un ensemble fini $\mathcal{X}$, de loi $p(x)=\Pr[X=x]$. Pour une base de logarithme fixée (base $2$ pour les *bits*, base $e$ pour les *nats*), l’entropie de $X$ est
$$
H(X) \;=\; - \sum_{x\in\mathcal{X}} p(x)\,\log p(x)
\;=\; \mathbb{E}\!\left[-\log p(X)\right].
$$
Le terme $-\log p(x)$ est l’**information de Shannon** (ou **surprise**) de l’issue $x$.

**Unités.** Avec $\log_2$, $H(X)$ est en **bits** ; avec $\log$, en **nats** ; avec $\log_{10}$, en **hartleys**.

#### Propriétés
- **Bornes.** $0 \le H(X) \le \log |\mathcal{X}|$.  
  $H(X)=0$ ssi $X$ est déterministe (toute la masse sur une seule valeur).  
  $H(X)$ est maximale et vaut $\log|\mathcal{X}|$ si $X$ est **uniforme** sur $\mathcal{X}$.
- **Concavité.** $H(\cdot)$ est concave en la loi $p$ de $X$.
- **Transformation déterministe.** Pour toute fonction déterministe $g$, $H(g(X)) \le H(X)$, avec égalité si $g$ est bijective sur le support.
- **Sous-additivité et indépendance.** Pour $(X,Y)$ discrets,
  $$
  H(X,Y) = H(X\mid Y) + H(Y) = H(Y\mid X) + H(X).
  $$
  Si $X \perp Y$ (indépendants), alors $H(X,Y)=H(X)+H(Y)$.
- **Le conditionnement réduit l’entropie.** $H(X\mid Y) \le H(X)$, avec égalité ssi $X$ et $Y$ sont indépendants.

#### Exemples
**Bernoulli.** Si $X\sim\mathrm{Bernoulli}(p)$,
$$
H(X) = -\,p\log p - (1-p)\log(1-p),
$$
maximisée en $p=\tfrac12$ (vaut $1$ bit en base $2$) et nulle en $p\in\{0,1\}$.

**Uniforme.** Si $X$ est uniforme sur $n$ symboles, $H(X)=\log n$.

### Dans le cas continu: entropie différentielle
Si $X$ est réelle (densité $f$), l’**entropie différentielle** est
$$
h(X) \;=\; - \int f(x)\,\log f(x)\,dx.
$$

**Attention :** $h(X)$ peut être **négative** et **n’est pas invariante par changement d’échelle** :
$$
h(aX) = h(X) + \log|a| \quad (a\neq 0).
$$
En revanche, des quantités dérivées comme l’information mutuelle et la divergence KL restent bien définies et invariantes de coordonnées.

**Exemple gaussien.** Si $X\sim\mathcal{N}(\mu,\sigma^2)$,
$$
h(X) = \tfrac12 \log\!\big(2\pi e\,\sigma^2\big).
$$
En dimension $d$, pour $X\sim\mathcal{N}(\mu,\Sigma)$,
$$
h(X) = \tfrac12 \log\!\big((2\pi e)^d \det \Sigma\big).
$$

### Estimer l'entropie
- **Estimateur plug-in (discret).** Remplacer $p(x)$ par la fréquence empirique $\hat{p}(x)$ :
  $$
  \widehat{H(X)} = -\sum_x \hat{p}(x)\log \hat{p}(x)
  $$
  (biais négatif pour petits échantillons ; corrections type Miller–Madow existent).
- **Continu.** Estimateurs à noyau, $k$-plus proches voisins (Kozachenko–Leonenko), ou via modèles paramétriques/neuraux (flows, VAEs) pour approcher $f$ ou $D_{\mathrm{KL}}$.


## Principe du maximum d’entropie
Parmi toutes les lois satisfaisant des contraintes (p. ex. support, moyenne, variance), la loi à entropie maximale est la moins informative (au sens de Shannon) :
- Support fini $\Rightarrow$ uniforme.
- Support $[0,\infty)$ et contrainte de moyenne $\Rightarrow$ exponentielle.
- Contrainte de moyenne et variance en $\mathbb{R}$ $\Rightarrow$ gaussienne.


## L'Entropie conditionnelle et l'information mutuelle
L’entropie conditionnelle de $X$ sachant $Y$ est
$$
H(X\mid Y) \;=\; \mathbb{E}_Y\big[H(X\mid Y=y)\big] \;=\; -\,\sum_{x,y} p(x,y)\,\log p(x\mid y).
$$
L’**information mutuelle** mesure la réduction d’incertitude sur $X$ due à la connaissance de $Y$ :
$$
I(X;Y) \;=\; H(X) - H(X\mid Y) \;=\; H(Y) - H(Y\mid X)
\;=\; H(X)+H(Y)-H(X,Y) \;\ge 0.
$$
**Inégalité de traitement de l’information (data processing).** Si $X \to Y \to Z$ forme une chaîne de Markov, alors $I(X;Z)\le I(X;Y)$.

## La Cross-entropie et la divergence de Kullback–Leibler
Pour deux lois $P$ et $Q$ sur le même alphabet,
- **Cross-entropie :** $H(P,Q) = -\sum_x P(x)\log Q(x)$,
- **Divergence KL :** $D_{\mathrm{KL}}(P\Vert Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} \ge 0$,

et l’identité clé :
$$
H(P,Q) = H(P) + D_{\mathrm{KL}}(P\Vert Q).
$$
En apprentissage, minimiser la log-vraisemblance négative revient souvent à minimiser une cross-entropie (donc une KL) entre la vraie loi $P$ et le modèle $Q_\theta$.

## Lien avec le codage (théorème source de Shannon)
Pour une source i.i.d. $X$, la longueur moyenne minimale $\bar{\ell}^\star$ d’un code préfixe vérifie
$$
H(X) \;\le\; \bar{\ell}^\star \;<\; H(X) + 1 \quad \text{(en bits/symbole)}.
$$
L’entropie est donc une borne fondamentale sur le **taux de compression**.



# SVD, ACP

- Données : matrice centrée $X \in \mathbb{R}^{n \times d}$, avec $n$ observations (lignes) et $d$ variables (colonnes).  
  On centre toujours : $X = \tilde{X} - \mathbf{1}\mu^\top$, où $\mu \in \mathbb{R}^{d}$ est la moyenne colonne et $\mathbf{1}$ le vecteur de 1.
- Covariance empirique : $\displaystyle S = \frac{1}{n-1} X^\top X \in \mathbb{R}^{d \times d}$.

## SVD : décomposition en valeurs singulières

La décomposition en valeurs singulières (SVD) factorise toute matrice réelle $X \in \mathbb{R}^{n\times d}$ en $X = U\Sigma V^\top$, où $U \in \mathbb{R}^{n\times r}$ et $V \in \mathbb{R}^{d\times r}$ ont des colonnes orthonormées (vecteurs singuliers gauche et droit), $\Sigma = \mathrm{diag}(\sigma_1 \ge \cdots \ge \sigma_r > 0)$ contient les valeurs singulières, et $r = \mathrm{rang}(X)$. On a $X^\top X = V\Sigma^2 V^\top$ et $X X^\top = U\Sigma^2 U^\top$, d’où le lien avec les décompositions spectrales. La SVD fournit la **meilleure approximation de rang $k$** au sens de la norme de Frobenius : $X_k = U_k \Sigma_k V_k^\top$ minimise $\|X - Y\|_F$ sur toutes les matrices $Y$ de rang $\le k$ (théorème d’Eckart–Young–Mirsky), avec erreur $\|X - X_k\|_F^2 = \sum_{i>k} \sigma_i^2$. Elle est utilisée pour le débruitage, la compression et le calcul de l’ACP (où $V$ donne les chargements et $\sigma_i^2/(n-1)$ les variances expliquées), et reste numé


La SVD (Singular Value Decomposition) de $X$ est :
$$
X = U \Sigma V^\top,\quad
U \in \mathbb{R}^{n \times r},\;
V \in \mathbb{R}^{d \times r},\;
\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_r),\; r=\mathrm{rang}(X).
$$
- $U$ : vecteurs singuliers à gauche (orthonormés),
- $V$ : vecteurs singuliers à droite (orthonormés),
- $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$ : valeurs singulières.

### Lien avec la covariance
$$
X^\top X = V \Sigma^2 V^\top \quad\Rightarrow\quad
S = \frac{1}{n-1} X^\top X = V \left(\frac{\Sigma^2}{\,n-1\,}\right) V^\top.
$$
Donc :
- **Vecteurs propres de $S$** $=$ **vecteurs singuliers droits** $V$.
- **Valeurs propres** $\lambda_k = \sigma_k^2/(n-1)$.



## L'ACP

L’Analyse en Composantes Principales (ACP) est une méthode linéaire de réduction de dimension qui projette des données centrées $X$ sur des directions orthogonales — les **composantes principales** — choisies pour maximiser la variance expliquée. On diagonalise la matrice de covariance $S=\tfrac{1}{n-1}X^\top X$ (ou on calcule la SVD $X=U\Sigma V^\top$) : les vecteurs propres de $S$ (colonnes de $V$) sont les **chargements**, et les **scores** sont $T=XV=U\Sigma$. En ne conservant que $k$ composantes associées aux plus grandes valeurs propres $\lambda_i=\sigma_i^2/(n-1)$, on obtient une représentation à $k$ dimensions qui minimise l’erreur quadratique de reconstruction (théorème d’Eckart–Young) avec une **variance expliquée** cumulée $\sum_{i=1}^k \lambda_i \big/ \sum_j \lambda_j$. L’ACP sert à visualiser, compresser et débruiter des données et se pratique souvent après **standardisation** des variables si leurs échelles diffèrent fortement.


Il y a 3 façons de voir l'ACP:
### 1) Maximisation de variance (optimisation quadratique)
La première composante principale est la direction unitaire $v_1 \in \mathbb{R}^{d}$ qui maximise la variance projetée :
$$
v_1 \;=\; \arg\max_{\|v\|_2=1}\; \mathrm{Var}(X v) \;=\; \arg\max_{\|v\|_2=1}\; v^\top S\, v.
$$
Par multiplicateurs de Lagrange, on obtient l’équation aux valeurs propres :
$$
S v_1 \;=\; \lambda_1 v_1,\quad \lambda_1=\max \text{ eigenvalue}.
$$
Les composantes suivantes $v_k$ se construisent de manière itérative sous contrainte d’orthogonalité $(v_k^\top v_j=0, j<k)$, donnant les $d$ vecteurs propres de $S$ rangés par $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d \ge 0$.

**Scores (coordonnées factorielles)** : $t_k = X v_k \in \mathbb{R}^{n}$.  
**Chargements** : colonnes de $V=[v_1,\dots,v_d]$.

### 2) Meilleure approximation de rang $k$ (moindres carrés)
L’ACP de rang $k$ cherche une approximation de $X$ par une matrice de rang $k$ :
$$
\min_{\substack{W \in \mathbb{R}^{d \times k},\, H \in \mathbb{R}^{n \times k}\\ W^\top W = I_k}}
\; \|X - H W^\top\|_F^2.
$$
La solution optimale est obtenue en prenant $W=V_k$ (les $k$ premiers vecteurs propres de $S$) et $H = X V_k$.  
Cette vue mène au théorème d’Eckart–Young–Mirsky (voir SVD ci-dessous).

### 3) Diagonalisation de la covariance (vue statistique)
En décomposant $S$ :
$$
S = V \Lambda V^\top,\quad \Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d),
$$
la projection $Z = X V$ a covariance diagonale :
$$
\frac{1}{n-1} Z^\top Z = \Lambda,
$$
c’est-à-dire des **composantes non corrélées** dont les variances expliquées sont $\lambda_k$.

### Choix du nombre de composantes $k$
- **Cumul de variance expliquée** : choisir le plus petit $k$ tel que $\sum_{i=1}^k \sigma_i^2 / \sum_{j=1}^r \sigma_j^2 \ge \tau$ (p.ex. $\tau=0{,}90$ ou $0{,}95$).
- **Coude** (scree plot) sur $\sigma_i^2$.
- **Validation croisée** sur l’erreur de reconstruction ou la performance aval.


### Reconstruction & projection
- **Projection (scores)** : $T_k = X V_k$.
- **Reconstruction** : $\widehat{X} = T_k V_k^\top = U_k \Sigma_k V_k^\top$.
- **Hors échantillon** : pour un nouveau $x$, centrer $x_c=x-\mu$, puis $t = x_c^\top V_k$ et $\hat{x}= \mu + V_k t$.

### Centrage, standardisation et ACP sur corrélations
- **Centrage indispensable** : sinon la 1ʳᵉ CP peut simplement « pointer » la moyenne.
- **Standardisation** (PCA sur la matrice de corrélation) si les échelles des variables diffèrent fortement : travailler sur
  $$
  X_{\text{std}} = (X - \mathbf{1}\mu^\top) D^{-1},
  $$
  où $D$ contient les écarts-types des colonnes, puis appliquer ACP/SVD à $X_{\text{std}}$.

### Blanchiment (whitening)
L’ACP fournit une base orthonormée où les variances sont $\lambda_k$. Le **whitening** met ces variances à 1 :
$$
X_{\text{white}} = X V \Lambda^{-1/2}
= X V \left(\frac{\Sigma}{\sqrt{n-1}}\right)^{-1}
= \sqrt{n-1}\, U.
$$
Ainsi, $\frac{1}{n-1} X_{\text{white}}^\top X_{\text{white}} = I$.

### Propriétés et détails numériques
- **Rang** : $\mathrm{rang}(X) \le \min(n-1,d)$ (le centrage retire au plus un degré de liberté).
- **Stabilité** : calculer l’ACP via la SVD de $X$ est numériquement plus stable que l’EVD de $S$.
- **Cas $n \ll d$ ou $d \ll n$** : utiliser la SVD tronquée (méthodes itératives, p.ex. Lanczos/power method).
- **Indétermination de signe** : $v_k$ et $-v_k$ sont équivalents (mêmes sous-espaces).

### ACP noyau (kernel PCA) — en bref
Remplace la projection linéaire par un **noyau** $K$ (centré en RKHS) et diagonalise $K$ (taille $n\times n$) pour capturer des **variations non linéaires**. Les scores sont obtenus via combinaisons des noyaux, sans construire explicitement la base de caractéristiques.

## Lien ACP ↔ SVD (formules utiles)
Si $X$ est **centrée** :
- **Chargements ACP** : $V = [v_1,\dots,v_r]$ (vecteurs singuliers droits).
- **Scores ACP** : $T = X V = U \Sigma$ (chaque colonne $t_k = \sigma_k\, u_k$).
- **Variance expliquée** :
  $$
  \text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^r \lambda_j}
  = \frac{\sigma_k^2}{\sum_{j=1}^r \sigma_j^2}.
  $$
- **Approximation de rang $k$** (Eckart–Young–Mirsky, meilleure au sens $\|\cdot\|_F$) :
  $$
  X_k = U_k \Sigma_k V_k^\top
  \quad\text{et}\quad
  \|X - X_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2.
  $$



# Rang d'une matrice: permet de déterminer si un espace est sur / sous dimensionné
Soit $A \in \mathbb{R}^{n \times d}$ et $r=\mathrm{rang}(A)$.
- **Espace image (colonnes)** : $ \mathrm{Im}(A)=\{A x : x\in\mathbb{R}^d\} \subseteq \mathbb{R}^n$.
  $$
  \boxed{\;\mathrm{rang}(A)=\dim(\mathrm{Im}(A))=\dim(\mathrm{span}\{\text{colonnes de }A\})\;}
  $$
- **Rang des lignes** : $ \mathrm{rang}(A)=\dim(\mathrm{span}\{\text{lignes de }A\})$ (rang des colonnes = rang des lignes).
- **Mineurs** : $ \mathrm{rang}(A)$ est l’ordre maximal $k$ tel qu’il existe un mineur $k\times k$ de déterminant non nul.
- **SVD** : si $A=U\Sigma V^\top$ avec $\Sigma=\mathrm{diag}(\sigma_1\ge\cdots\ge\sigma_r>0)$,
  $$
  \boxed{\;\mathrm{rang}(A)=\#\{i:\sigma_i>0\}\;}
  $$

Bornes : $\;0 \le r \le \min(n,d)$.

## Théorème du rang (rang-noyau)
Le **noyau** (espace des solutions homogènes) est $\ker(A)=\{x\in\mathbb{R}^d: Ax=0\}$.
$$
\boxed{\; \dim(\ker(A)) + \mathrm{rang}(A) = d \;}
$$
- $\dim(\ker(A)) = d-r$ s’appelle parfois la **nullité**.
- Interprétation : chaque dépendance linéaire entre colonnes ajoute une dimension au noyau.

## Indépendance linéaire et base
Pour des vecteurs $a_1,\dots,a_m \in \mathbb{R}^d$, posons $A=[a_1~\cdots~a_m]\in\mathbb{R}^{d\times m}$.
- $ \mathrm{rang}(A) = $ **nombre maximal de vecteurs linéairement indépendants** parmi $a_1,\dots,a_m$.
- Si $ \mathrm{rang}(A)=m\le d$, la famille est libre.  
- Si $ \mathrm{rang}(A)<m$, la famille est redondante (sur-paramétrée) pour décrire son sous-espace vectoriel.

## Systèmes linéaires $Ax=b$ et (sur/sous)-détermination
- **Existence (compatibilité)** : le système $Ax=b$ a une solution ssi
  $$
  \mathrm{rang}(A) = \mathrm{rang}([A~|~b]).
  $$
- **Unicité** : si une solution existe, elle est **unique** ssi $\ker(A)=\{0\}$, i.e. $\mathrm{rang}(A)=d$ (colonnes libres).


### Cas selon la forme de $A$ (avec $A\in\mathbb{R}^{n\times d}$)
1. **Sur-déterminé (plus d’équations que d’inconnues)** : $n>d$.
   - Si $ \mathrm{rang}(A)=d$ (**plein rang colonne**), l’égalité exacte $Ax=b$ n’est pas garantie, mais la **moindre carrés** a solution unique :
     $$
     x^\star = (A^\top A)^{-1}A^\top b.
     $$
   - Si $ \mathrm{rang}(A)<d$, colonnes dépendantes $\Rightarrow$ solution LS **non unique** (ill-posée).

2. **Sous-déterminé (plus d’inconnues que d’équations)** : $d>n$.
   - Si $ \mathrm{rang}(A)=n$ (**plein rang ligne**), il y a soit **infinité de solutions** (si compatible), la solution **de norme minimale** est
     $$
     x^\star = A^\top (AA^\top)^{-1} b.
     $$
   - Si $ \mathrm{rang}(A)<n$, encore plus de liberté (noyau plus grand).

3. **Carré** : $n=d$.
   - $ \mathrm{rang}(A)=n$ $\Leftrightarrow$ $A$ **inversible**, $\det(A)\neq 0$, $\sigma_i>0$ $\forall i$.  
   - Sinon, $A$ est **singulière** : pas d’unicité (noyau non trivial) et potentielle incompatibilité.

## Diagnostic « sur/sous-dimensionné » via le rang

### Familles de vecteurs dans $\mathbb{R}^d$
- Vous avez $m$ vecteurs.  
  - Si $\mathrm{rang}(A)=m<d$ : la famille est **sous-dimensionnée pour $\mathbb{R}^d$** (elle ne peut pas engendrer tout $\mathbb{R}^d$), mais **bien dimensionnée** pour son sous-espace de dimension $m$.  
  - Si $\mathrm{rang}(A)<m$ : la famille est **sur-dimensionnée** pour son sous-espace (des vecteurs redondants).  
  - Pour obtenir une **base** de $\mathbb{R}^d$, il faut et il suffit d’avoir $\mathrm{rang}(A)=d$ avec $m\ge d$; une base minimale a $m=d$.

### Données $(n \times d)$ et colinéarités
Considérez une matrice de données $X\in\mathbb{R}^{n\times d}$ (lignes = observations, colonnes = variables).
- Si $\mathrm{rang}(X)=d$ et $n\ge d$ : les variables sont **non redondantes** (pas de colinéarité parfaite).  
- Si $\mathrm{rang}(X)<d$ : variables **redondantes** $\Rightarrow$ espace de caractéristiques **sur-dimensionné** par rapport à l’information disponible; on peut réduire la dimension (p.ex. ACP/SVD).  
- Si $d \gg n$ : même avec $\mathrm{rang}(X)=n$, l’espace des variables est **sous-contraint** par les données (risque de sur-apprentissage) ; régularisation ou réduction de dimension sont recommandées.

## Outils de calcul et liens utiles
- **RREF / pivots** : le nombre de pivots (après élimination de Gauss) = $ \mathrm{rang}(A)$.  
- **SVD** : $\mathrm{rang}(A)=\#\{ \sigma_i > 0\}$.  
- **Pseudo-inverse de Moore–Penrose** $A^+$ (via SVD) :
  $$
  A^+ = V\,\Sigma^+\,U^\top,\quad \Sigma^+=\mathrm{diag}\Big(\tfrac{1}{\sigma_i}\mathbf{1}_{\sigma_i>0}\Big),
  $$
  donne les solutions LS minimales en norme : $x^\star=A^+ b$.


# Information de Fisher et Bornes de Cramer-Rao

L’information de Fisher d’un modèle paramétrique $p_\theta(y)$ est la matrice $I(\theta)=\mathbb{E}_{y\sim p_\theta}\!\big[\nabla_\theta \log p_\theta(y)\,\nabla_\theta \log p_\theta(y)^\top\big]=-\mathbb{E}_{y\sim p_\theta}\!\big[\nabla_\theta^2 \log p_\theta(y)\big]$ (sous conditions de régularité) ; elle quantifie la quantité d’information sur $\theta$ contenue dans une observation et mesure la courbure moyenne de la log-vraisemblance. Pour $n$ échantillons i.i.d., $I_n(\theta)=n\,I(\theta)$, ce qui induit la borne de Cramér–Rao $\mathrm{Cov}(\hat\theta)\succeq I_n(\theta)^{-1}$ pour tout estimateur sans biais et l’optimalité asymptotique du MLE : $\sqrt{n}(\hat\theta-\theta^\star)\Rightarrow \mathcal{N}\!\big(0,\,I(\theta^\star)^{-1}\big)$. On distingue la **Fisher observée** $-\nabla_\theta^2 \log L(\theta)$ évaluée en $\hat\theta$ (avec $L$ la vraisemblance) et la **Fisher attendue** (son espérance). En apprentissage, on minimise la NLL $\mathcal{L}(\theta)=-\sum_i \log p_\theta(y_i)$, égale à la cross-entropie dans les modèles de langue ; au minimum $\hat\theta$, la Hessienne $\nabla_\theta^2 \mathcal{L}(\hat\theta)$ coïncide en moyenne avec $I_n(\hat\theta)$, d’où l’interprétation de la Fisher comme “courbure de la loss”. Enfin, l’approximation de second ordre de la divergence de KL donne $\mathrm{KL}\big(p_\theta\,\|\,p_{\theta+\delta}\big)\approx \tfrac{1}{2}\,\delta^\top I(\theta)\,\delta$, plaçant $I(\theta)$ au cœur de la **natural gradient** et de la géométrie de l’information.

A noter que dans les modèles de langue, c'est l'information de Fisher qui est utilisée comme loss (Trouver les $theta$ qui maximisent les probabilités des observations (log-likelihood), ou minimiser la negative log-likelihood).


# Jacobienne
Soit $X\in\mathbb{R}^d$ à densité $f_X$ et une application $T:\mathbb{R}^d\to\mathbb{R}^d$ de classe $C^1$ inversible presque partout, avec matrice jacobienne $J_T(x)=\big[\partial T_i/\partial x_j\big]_{i,j}$ et déterminant $\det J_T(x)$. Pour $Y=T(X)$, la densité de $Y$ est le **pushforward** de $f_X$ par $T$ : $f_Y(y)=f_X\!\big(T^{-1}(y)\big)\,\big|\det J_{T^{-1}}(y)\big|=f_X(x)\,/\,\big|\det J_T(x)\big|$ où $x=T^{-1}(y)$. Cette formule découle de la conservation de la masse $\int_{A} f_Y(y)\,dy=\int_{T^{-1}(A)} f_X(x)\,dx$ et du fait que $|\det J_T(x)|$ mesure la dilatation locale de volume (la valeur absolue assure des densités positives, l’orientation n’ayant pas d’effet probabiliste). En dimension 1, pour une transformation monotone $Y=g(X)$, $f_Y(y)=f_X\!\big(g^{-1}(y)\big)\,\big|\tfrac{d}{dy}g^{-1}(y)\big|$; si $g$ est décroissante la dérivée est négative mais la valeur absolue corrige le signe. En cas de transformation **non injective** mais $C^1$ par morceaux (par ex. $T$ $k$-à-1 sur une partition), on somme sur les antécédents : $f_Y(y)=\sum_{x\in T^{-1}(\{y\})} f_X(x)\,/\,\big|\det J_T(x)\big|$. Les espérances suivent la même règle : pour toute $\varphi$ intégrable, $\mathbb{E}[\varphi(Y)]=\int \varphi\!\big(T(x)\big)f_X(x)\,dx=\int \varphi(y)\,f_Y(y)\,dy$. Si $\det J_T(x)=0$ sur un ensemble de mesure non négligeable, l’image peut être **singulière** (masse portée par une variété de dimension $<d$) et $Y$ peut ne pas admettre de densité Lebesgue; sinon, dans le cas classique (diffeomorphisme a.e.), la jacobienne fournit la relation fondamentale de changement de variables en probabilité. *Ex. compact : coordonnées polaires* $T(r,\theta)=(r\cos\theta,r\sin\theta)$ sur $\mathbb{R}^2\setminus\{0\}$ ont $|\det J_T|=r$, d’où $f_{X,Y}(x,y)=f_{R,\Theta}(r,\theta)/r$ avec $r=\sqrt{x^2+y^2}$, $\theta=\mathrm{atan2}(y,x)$.


# Transformation d'un problème complexe à un problème convexe
**Transformer un problème complexe en problème convexe** consiste à remplacer un programme non convexe $\min_{x\in\mathcal{X}} f(x)$ (où $f$ ou $\mathcal{X}$ est non convexe) par un **relaxé convexe** $\min_{x\in\mathcal{X}_{\mathrm{cvx}}} f_{\mathrm{cvx}}(x)$ résoluble globalement avec certificats d’optimalité (dualité forte, KKT). La démarche canonique combine : (i) **relaxation convexe** par enveloppe convexe et conv(hull) : on prend $f_{\mathrm{cvx}}=f^{**}$ (biconjuguée de Fenchel, plus grand convexe l.s.c. majoré par $f$) et un sur-ensemble convexe $\mathcal{X}_{\mathrm{cvx}}\supseteq\mathcal{X}$, donnant un **borne inférieure** $p^\star_{\mathrm{cvx}}\le p^\star$ (ex.: $\ell_0\to\ell_1$, $\min_x \tfrac12\|Ax-b\|_2^2$ s.c. $\|x\|_0\le k$ $\leadsto$ $\min_x \tfrac12\|Ax-b\|_2^2+\lambda\|x\|_1$; **exactitude** si p.ex. $A$ vérifie une RIP/N.S.P., alors la solution $\ell_1$ coïncide avec $\ell_0$); (ii) **relaxation de rang** via norme nucléaire : $\min_X \operatorname{rank}(X)$ s.c. $\mathcal{A}(X)=b$ $\leadsto$ $\min_X \|X\|_\ast$ s.c. $\mathcal{A}(X)=b$ (exact sous incohérence/échantillonnage suffisant) ; (iii) **lifting/SDP** pour bilinéaires/quadratiques : $x^\top Qx$ avec contraintes non convexes $\leadsto$ $X=xx^\top\succeq 0$ et contraintes linéaires sur $X$, en relâchant $\operatorname{rank}(X)=1$ ; (iv) **épigraphe/perspective** pour convexifier max/fractions : minimiser $f(x)$ $\equiv$ $\min_{t}\{t: (x,t)\in\operatorname{epi} f\}$, et pour un objectif fractionnel linéaire $\min \frac{c^\top x+d}{e^\top x+f}$ s.c. $Ax\le b$, la **transformation de Charnes–Cooper** $t=(e^\top x+f)^{-1}$, $y=xt$ donne $\min c^\top y+dt$ s.c. $Ay\le bt$, $e^\top y+ft=1$, $t>0$ (LP) ; (v) **changements de variables** (mono. difféomorphes) pour rendre convexes des formes multiplicatives, p.ex. **programmation géométrique** : $\min \sum_k c_k \prod_i x_i^{a_{ik}}$ s.c. $\sum_k d_k \prod_i x_i^{b_{ik}}\le 1$ devient convex en posant $y_i=\log x_i$ et en log-transformant (somme de fonctions convexes log-somme-exp) ; (vi) **prox/ pénalisation exacte** pour absorber contraintes non convexes dans l’objectif, puis remplacer par une norme convexe (p.ex. pénalités SCAD/MCP $\to$ majorations convexes via MM). Sur le plan théorique, le relâché fournit un **dualisme de Lagrange** avec écart $p^\star-p^\star_{\mathrm{cvx}}\ge 0$ ; si la relaxation est **serrée** (p.ex. solution située à un **point extrême** de $\mathcal{X}_{\mathrm{cvx}}$ respectant les **KKT** du problème original), on a $p^\star=p^\star_{\mathrm{cvx}}$ (exactitude) et la solution du convexe résout l’original. Au final, on obtient un problème convexe de la forme standard $\min_x g(x)$ s.c. $h_i(x)\le 0,\;Ax=b$ (avec $g,h_i$ convexes), solvable en temps polynomial (méthodes de points intérieurs, descente proximale), garantissant un optimum **global** et des **certificats** via conditions KKT et **dualité forte** (écart nul).


# Transformation d'un signal du domaine temporel au domaine fréquentiel
Pour un signal continu $x(t)\in L^1(\mathbb{R})$ (ou $L^2$), sa transformée de Fourier est $X(f)=\mathcal{F}\{x\}(f)=\int_{-\infty}^{\infty} x(t)\,e^{-j2\pi f t}\,dt$ et l’inverse s’écrit $x(t)=\int_{-\infty}^{\infty} X(f)\,e^{j2\pi f t}\,df$; $X(f)$ encode l’**amplitude** et la **phase** des composantes sinusoïdales de fréquence $f$. La transformation est linéaire, convertit les **décalages temporels** en facteurs de phase $x(t-t_0)\;\leftrightarrow\;X(f)\,e^{-j2\pi f t_0}$, la **modulation** temporelle en **translation fréquentielle** $x(t)\,e^{j2\pi f_0 t}\;\leftrightarrow\;X(f-f_0)$, et surtout transforme la **convolution temporelle** en **produit fréquentiel** $(x*y)(t)\;\leftrightarrow\;X(f)Y(f)$ (et le produit temporel en convolution fréquentielle). L’**identité de Parseval/Plancherel** préserve l’énergie : $\int |x(t)|^2 dt=\int |X(f)|^2 df$. Pour un signal discret $x[n]=x(nT_s)$ échantillonné à $f_s=1/T_s$, la DTFT est $X(e^{j\omega})=\sum_{n=-\infty}^{\infty} x[n]\,e^{-j\omega n}$ ($\omega=2\pi f/f_s$) et, sur une fenêtre de longueur $N$, la **DFT** est $X[k]=\sum_{n=0}^{N-1} x[n]\,e^{-j2\pi nk/N}$, $k=0,\dots,N-1$, avec inverse $x[n]=\frac{1}{N}\sum_{k=0}^{N-1} X[k]\,e^{j2\pi nk/N}$; la résolution fréquentielle vaut $\Delta f=f_s/N$ et l’algorithme **FFT** calcule la DFT en $O(N\log N)$. L’**échantillonnage** d’un signal continu $x(t)$ produit un spectre périodisé : si $x_s(t)=\sum_{n} x(nT_s)\,\delta(t-nT_s)$, alors $X_s(f)=\frac{1}{T_s}\sum_{m\in\mathbb{Z}} X(f-mf_s)$; l’absence d’**aliasing** requiert un spectre borné $|f|\le B$ et $f_s>2B$ (théorème de Shannon–Nyquist), avec reconstruction idéale $x(t)=\sum_{n\in\mathbb{Z}} x[n]\,\mathrm{sinc}\!\big(\tfrac{t-nT_s}{T_s}\big)$. En pratique, l’analyse de signaux non stationnaires utilise des **fenêtres** (STFT) : $X(t_0,f)=\int x(t)\,w(t-t_0)\,e^{-j2\pi f t}\,dt$, où le choix de $w$ contrôle le compromis temps–fréquence (fuites spectrales et élargissement des pics). Ces outils fournissent une cartographie rigoureuse du contenu fréquentiel à partir du temps, ouvrant la voie au filtrage linéaire, à la débruitage, à la démodulation et à l’identification de systèmes via leurs réponses fréquentielles.



# Quand on a l'inverse d'une somme: on peut toujours revenir à une somme des puissances!

## Démo
> On va montrer que 
> $$
> \colorbox{cyan}{$\displaystyle(1-x)^{-1} = 1 + x + x^2 + x^3 + x^4 + ...$}
> $$

Prenons la somme $S = 1 + x + x^2 + x^3 + x^4 + ...$

$$
\begin{split}
S - Sx &= 1 + x + x^2 + x^3 + x^4 + ... - (x + x^2 + x^3 + x^4 + ...) \\
&= 1
\end{split}
$$

D'où:

$$
\begin{split}
S - Sx &= 1 \\
S(1-x) &= 1 \\
\colorbox{cyan}{$\displaystyle S = \frac{1}{1-x}$}
\end{split}
$$

A noter : quand on fait une approximation en $x$ proche de 0 (développement limité), on s'arrête à une certaine puissance $n$ (en fonction de l'approximation à l'ordre $n$).

## Utilité
Dans le papier [Chain and Causal Attention for Efficient Entity Tracking](https://aclanthology.org/2024.emnlp-main.731.pdf), j'essayais de comprendre :

$$
A + A^2 + A^3 + A^4 + ... = A(I - A)^{-1}
$$

Et j'ai compris que ça venait de l'extension aux matrices de la formule précédente: 

$$
\begin{split}
1 + x + x^2 + x^3 + x^4 + ... &= \frac{1}{1-x} \\
I + A + A^2 + A^3 + A^4 + ... &= I(I - A)^{-1} \\
A + A^2 + A^3 + A^4 + ... &= A(I - A)^{-1}
\end{split}
$$

# La formule de Taylor d'ordre n pour approximer une fonction en un point $x_i$

Pour une fonction $f(x)$ suffisamment dérivable au voisinage de $x_i$, la formule de Taylor d'ordre $n$ s'écrit :

$$
f(x) \approx \sum_{k=0}^{n} \frac{f^{(k)}(x_i)}{k!}(x - x_i)^k + R_n(x)
$$

où :
- $f^{(k)}(x_i)$ désigne la dérivée $k$-ième de $f$ évaluée en $x_i$
- $f^{(0)}(x_i) = f(x_i)$ par convention
- $R_n(x)$ est le reste de Taylor d'ordre $n$

En développant explicitement :

$$
f(x) \approx f(x_i) + f'(x_i)(x - x_i) + \frac{f''(x_i)}{2!}(x - x_i)^2 + \cdots + \frac{f^{(n)}(x_i)}{n!}(x - x_i)^n + R_n(x)
$$

On a donc par exemple, l'approximation de Taylor à l'ordre 1:
$$
f(x) \approx f(x_i) + f'(x_i)\,(x - x_i)
$$ 

## Démo (par récurrence et intégrations par parties)

**Étape 1 : Cas de base (n = 0)**

Partons du théorème fondamental du calcul intégral :
$$
f(x) - f(x_i) = \int_{x_i}^{x} f'(t) \, dt
$$

Donc :
$$
f(x) = f(x_i) + \int_{x_i}^{x} f'(t) \, dt
$$

C'est la formule de Taylor d'ordre 0 avec $R_0(x) = \int_{x_i}^{x} f'(t) \, dt$.

**Étape 2 : Passage de l'ordre k à l'ordre k+1**

Supposons que nous ayons établi la formule à l'ordre $k$ :
$$
f(x) = \sum_{j=0}^{k} \frac{f^{(j)}(x_i)}{j!}(x - x_i)^j + R_k(x)
$$

avec $R_k(x) = \int_{x_i}^{x} \frac{f^{(k+1)}(t)}{k!}(x - t)^k \, dt$.

Appliquons une intégration par parties à $R_k(x)$ :

Posons :
- $u = f^{(k+1)}(t)$, donc $du = f^{(k+2)}(t) \, dt$
- $dv = \frac{(x - t)^k}{k!} \, dt$, donc $v = -\frac{(x - t)^{k+1}}{(k+1)!}$

L'intégration par parties donne :
$$
R_k(x) = \left[ f^{(k+1)}(t) \cdot \left(-\frac{(x - t)^{k+1}}{(k+1)!}\right) \right]_{x_i}^{x} + \int_{x_i}^{x} \frac{(x - t)^{k+1}}{(k+1)!} f^{(k+2)}(t) \, dt
$$

Le terme entre crochets devient :
$$
\left[ -\frac{f^{(k+1)}(t)(x - t)^{k+1}}{(k+1)!} \right]_{x_i}^{x} = 0 - \left(-\frac{f^{(k+1)}(x_i)(x - x_i)^{k+1}}{(k+1)!}\right) = \frac{f^{(k+1)}(x_i)}{(k+1)!}(x - x_i)^{k+1}
$$

Donc :
$$
R_k(x) = \frac{f^{(k+1)}(x_i)}{(k+1)!}(x - x_i)^{k+1} + \int_{x_i}^{x} \frac{f^{(k+2)}(t)}{(k+1)!}(x - t)^{k+1} \, dt
$$

En substituant dans l'expression de $f(x)$ :
$$
f(x) = \sum_{j=0}^{k} \frac{f^{(j)}(x_i)}{j!}(x - x_i)^j + \frac{f^{(k+1)}(x_i)}{(k+1)!}(x - x_i)^{k+1} + R_{k+1}(x)
$$

où $R_{k+1}(x) = \int_{x_i}^{x} \frac{f^{(k+2)}(t)}{(k+1)!}(x - t)^{k+1} \, dt$.

Ceci établit la formule à l'ordre $k+1$ :
$$
f(x) = \sum_{j=0}^{k+1} \frac{f^{(j)}(x_i)}{j!}(x - x_i)^j + R_{k+1}(x)
$$

**Conclusion**

Par récurrence, la formule de Taylor d'ordre $n$ est démontrée pour tout $n \geq 0$.

## Utilité

Si on cherche à approximer les nouveaux poids du modèle $\theta_\varepsilon$ qui sont les poids du modèle modifié (avec retrait d'un exemple d'entraînement), on a la nouvelle loss finale du modèle $R_\varepsilon(\theta, z_\text{train})$ qui est: 

$$
R_\varepsilon(\theta, z_\text{train}) = \frac{1}{n}\sum_{i=1}^n \mathcal{L}(z_i,\theta)
  \;+\;\varepsilon\,\mathcal{L}(z_{\rm train},\theta).
$$

Et donc les nouveaux poids $\theta_\varepsilon$ qui minimisent cette loss:

$$
\theta_\varepsilon = \arg\min_{\theta}\;R_\varepsilon(\theta, z_\text{train})
$$

D'où:

$$
\begin{split}
R_\varepsilon(\theta_\varepsilon, z_\text{train}) &= \frac{1}{n}\sum_{i=1}^n \mathcal{L}(z_i,\theta_\varepsilon) \;+\;\varepsilon\,\mathcal{L}(z_{\rm train},\theta_\varepsilon) \\
&= 0
\end{split}
$$

Maintenant, on peut utiliser une approximation de Taylor à $\varepsilon$ proche de 0 de $$\frac{1}{n}\sum_{i=1}^n \mathcal{L}(z_i,\theta_\varepsilon) \;+\;\varepsilon\,\mathcal{L}(z_{\rm train},\theta_\varepsilon)$$, car on connait $\theta$ et donc on peut peut-être arriver à quelque chose!

Pour rappel, l'approximation de Taylor de $f$ à l'ordre 1 en $\varepsilon$ proche de 0 donne: $$f(\theta_\varepsilon) \approx f(\hat{\theta}) + f'(\hat{\theta})\,(\theta_\varepsilon - \hat{\theta})$$

$$
\begin{split}
\underbrace{\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\theta_\varepsilon)
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\theta_\varepsilon)}_{f(\theta_\varepsilon)} \approx \underbrace{[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\hat{\theta})
\;+\;\varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\hat{\theta}) ]}_{f(\hat{\theta})}
\;\; &+ \\
\;\; \underbrace{[ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\hat{\theta})
\;+\;\varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\hat{\theta}) ]}_{f'(\hat{\theta})} \;\;(\theta_\varepsilon - \hat{\theta})
\end{split}
$$

ça nous permet d'isoler $\theta_\varepsilon$ et donc d'écrire $\theta_\varepsilon$ en fonction de $\theta$!

$$
\begin{split}
\theta_\varepsilon \approx \hat{\theta} - \left[ \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i,\hat{\theta}) + \varepsilon\,\nabla^2_\theta \mathcal{L}(z_{\rm train},\hat{\theta}(z_\text{train})) \right]^{-1} \;\; &\times \\
\;\; \left[ \frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\hat{\theta}) + \varepsilon\,\nabla_\theta \mathcal{L}(z_{\rm train},\hat{\theta}) \right]
\end{split}
$$

Et ça c'est cool parce qu'on sait le simplifier puisqu'on connait $\hat{\theta}$! Notamment $\hat{\theta}$ sont les poids optimaux pour le modèle de base, donc $\frac{1}{n}\sum_{i=1}^n \nabla_\theta \mathcal{L}(z_i,\hat{\theta}) = 0$. D'autres simplifications peuvent ensuite être faites. Et donc on arrive à approximer les nouveaux poids du modèle!

# Penser à la chain rule: exemple pour calculer $\frac{\partial \mathcal{L}}{\partial \theta}$

On a par exemple $\mathcal{L}$ est une fonction de perte (loss) qui dépend de la sortie du modèle $y$, $y$ qui est la sortie du modèle, qui dépend des paramètres du modèle $\theta$, d'où on a: $\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L(y(\theta))}}{\partial \theta}$. ça doit nous faire penser à la chain rule! $$\frac{\partial}{\partial x} f(g(x))$$ avec $f = \mathcal{L}$, ou $f(y) = \mathcal{L(y)}$ et $g(\theta) = y(\theta)$.

## Rappel de la formule...

Il faut donc ici penser à la chain rule! Un petit rappel... 

$$\frac{\partial}{\partial x} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

## Application concrète pour calculer $\frac{\partial \mathcal{L}}{\partial \theta}$

$$
\frac{\partial \mathcal{L(y(\theta))}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial \theta}
$$


# Optimisation sous contrainte en IA
TODO