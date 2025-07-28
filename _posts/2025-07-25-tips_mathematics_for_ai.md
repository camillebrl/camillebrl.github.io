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