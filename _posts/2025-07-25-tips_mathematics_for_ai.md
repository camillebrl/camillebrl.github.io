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

