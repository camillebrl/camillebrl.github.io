---
layout: distill
title: Explicabilité en deep learning
date: 2025-08-15 11:00:00
description: Explcabilité des modèles de deep learning
tags: XAI, robustesse, attribution
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

# Analyse des données

## Représentativité des données

Nous allons détailler l'approche [MeGe de la librairie Xplain de Deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/representativity.py) qui évalue la représentativité des données via un protocole "K modèles / K splits". L’idée centrale repose sur le fait que si les données sont bien représentatives (couvrent le support/les variations pertinentes) et si l’explicabilité de la génération du modèle est stable, alors **les explications produites par des modèles entraînés sur des sous-échantillons différents devraient être semblables lorsque les modèles aboutissent à la même prédiction correcte**. À l’inverse, les **explications devraient s’écarter lorsque les modèles prédisent des classes différentes**.

On dispose d’un jeu de données $\mathcal{D}=\{(x_n,y_n)\}_{n=1}^N$, où $y_n$ est **one-hot** pour la classification. Le code impose un entier $K$ tel que $K$ divise $N$, puis découpe $\mathcal{D}$ en $K$ splits de même taille $m=N/K$ :
$$
\mathcal{D} = \bigsqcup_{i=1}^K S_i, \qquad |S_i|=m.
$$

Pour chaque $i\in\{1,\dots,K\}$, on entraîne un modèle $M_i$ avec les données hors du split $S_i$ :
$$
M_i \leftarrow \text{learning\_algorithm}\big(\underbrace{\mathcal{D}\setminus S_i}_{\text{train}}, \underbrace{S_i}_{\text{validation/test}}\big).
$$


e protocole “K modèles / K splits” ?**  
Il imite une **perturbation des données d’entraînement** : chaque modèle voit une **version partielle** du jeu, ce qui permet de **sonder la stabilité des explications** vis-à-vis de variations réalistes d’échantillonnage (mesure indirecte de la **représentativité des données** et de la **robustesse de l’explication**).

## Complétude des données: l'approche [ReCo de la librairie Xplain de Deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/representativity.py) qui évalue la représentativité des données




# Evaluation de la robustesse du modèle

[deel-lip de Deel-IA](https://github.com/deel-ai/deel-lip)

En vérification/robustesse, on cherche des **bornes déterministes** qui garantissent ce que le modèle **ne peut pas** faire quand l’entrée est perturbée dans un rayon \(\varepsilon\) (p. ex. attaques adversariales \(\|\delta\|_p \le \varepsilon\)). On borne :
- **Les sorties** \(f(x)\) (logits/scores) : obtenir des **intervalles** qui contiennent \(f(x+\delta)\) pour **toutes** \(\delta\) admissibles.
- **Le gradient/Jacobien** \(J_f(x)=\frac{\partial f}{\partial x}\) : contrôler la **sensibilité locale**. Borner \(\|J_f(x)\|\) **implique** borner la variation des sorties.
- **La perte** \(L(f(x),y)\) : certaines méthodes bornent directement une **perte robuste** (majorant/minorant du pire cas).


> **À quoi ça sert ?**  
> - **Certifier** qu’aucune perturbation \(\|\delta\|_p\le\varepsilon\) ne change la prédiction (certificat de robustesse).  
> - **Entraîner** le modèle sous contrainte (régularisations/architectures) pour **augmenter** la robustesse certifiée.  
> - **Auditer** la stabilité des explications basées gradients (sensibilité locale).

## 1) Constante de **Lipschitz** (via `deel-lip`)

### Définition
Pour \(f:(\mathbb{R}^n,\|\cdot\|_p)\to(\mathbb{R}^m,\|\cdot\|_q)\), \(f\) est \(K\)-Lipschitz si
\[
\|f(x)-f(y)\|_q \le K\,\|x-y\|_p,\quad \forall x,y.
\]
Le plus petit \(K\) est \(\mathrm{Lip}(f)\). Si \(f\) est différentiable,
\[
\mathrm{Lip}(f)\;\le\;\sup_{x}\,\|J_f(x)\|_{p\to q}.
\]

**Conséquence « marge certifiée ».**  
Soit la **marge** \(m(x)=f_y(x)-\max_{j\ne y}f_j(x)\). Si chaque logit est \(K\)-Lipschitz, alors pour toute \(\|\delta\|_p \le \varepsilon\),
\[
m(x+\delta)\;\ge\; m(x) - 2K\varepsilon.
\]
Donc si \(m(x) > 2K\varepsilon\), la prédiction est **certifiée** inchangée dans la boule \(\varepsilon\).

### Comment `deel-lip` calcule/contrôle \(K\)
`deel-lip` (et son pendant PyTorch `deel-torchlip`) vise à **construire des réseaux \(k\)-Lipschitz par design** en contrôlant la norme d’opérateur des couches et en choisissant des activations 1-Lipschitz.

- **Couches linéaires (denses/conv) avec contrainte spectrale.**  
  - **Power iteration** : approximer la plus grande valeur singulière \(\sigma_{\max}(W)\) et **la normaliser** (puis **rescaler** vers la cible \(k\)).  
  - **Björck orthonormalization** : itérer pour rapprocher \(W\) d’une matrice (quasi-)orthonormale (singulières \(\approx 1\)), puis appliquer un **facteur d’échelle** pour viser \(\|W\|_{\text{op}}\approx k\).

- **Convolutions : RKO (Reshaped Kernel Orthogonalization).**  
  Remodeler le noyau de conv en matrice, appliquer power iteration + Björck pour contrôler la **norme d’opérateur effective** de la convolution, puis rescaler.

- **Activations 1-Lipschitz.**  
  Ex. `GroupSort`, `FullSort`, `MaxMin`, `Householder`, etc. Si chaque activation est 1-Lipschitz et chaque couche linéaire est \(k_\ell\)-Lipschitz,
  \[
  \mathrm{Lip}(f)\;\le\;\prod_{\ell=1}^L k_\ell.
  \]

- **Métriques & pertes adaptées.**  
  La librairie expose des **métriques** pour suivre \(\sigma_{\max}\) / lip de couche et des **pertes** (p. ex. variantes type HKR) pour favoriser des classifieurs **certifiables**.

> **Calcul d’un rayon certifié simple.**  
> Avec une borne globale \(K\) et une marge \(m(x)\), un **rayon** \(\varepsilon^\star\) certifié vérifie \(m(x)>2K\varepsilon^\star\), d’où \(\varepsilon^\star<\tfrac{m(x)}{2K}\).

---

## 2) **Relaxations linéaires** (bornes par point)

Objectif : obtenir des **bornes plus serrées et spécifiques à l’exemple \(x\)** en linéarisant les non-linéarités via leurs **enveloppes convexes** dépendant des bornes **pré-activation** \([l,u]\).

### Exemple pour ReLU
Pour \(z=\mathrm{ReLU}(x)\) avec \(l\le x \le u\) :
- Si \(u \le 0\) : \(z=0\).  
- Si \(l \ge 0\) : \(z=x\).  
- Si \(l<0<u\) (zone ambiguë), enveloppe convexe standard :
  \[
  z \ge 0,\quad z \ge x,\quad z \le \frac{u}{u-l}\,(x-l).
  \]
Ces contraintes **linéaires** sont propagées couche par couche pour obtenir une **relaxation convexe** du réseau sur le domaine \(\{x+\delta:\|\delta\|_p\le \varepsilon\}\).

### IBP, CROWN, DeepPoly (idées clés)
- **IBP (Interval Bound Propagation)** : propage des **intervalles** \([l,u]\) (rapide, parfois lâche).  
- **CROWN / LiRPA** : propage des **bornes linéaires** (majorants/minorants affines) en arrière pour serrer les bornes de **sortie**.  
- **DeepPoly** : relaxation polyédrale plus fine que de simples intervalles.  
- **CROWN-IBP** : hybride (IBP pour bornes intermédiaires + CROWN pour sorties) = **compromis précision/coût**, utile en **entraînement certifié**.

### Lire un **certificat** avec relaxations
Pour chaque logit \(j\), on obtient :
\[
\underbrace{L_j}_{\text{borne inf}} \;\le\; f_j(x+\delta) \;\le\; \underbrace{U_j}_{\text{borne sup}}
\quad\text{pour toute }\|\delta\|_p\le\varepsilon.
\]
Si \(\;L_{y} > \max_{j\ne y} U_j\;\), alors la prédiction \(y\) est **provablement invariante** aux perturbations admissibles.

### Entraînement certifié
On peut **minimiser une borne supérieure** de la **perte robuste** (p. ex. en remplaçant les logits par leurs bornes \((L_y,U_j)\) dans la CE), ce qui **augmente** la robustesse certifiée sans résoudre un problème NP-difficile.

---

## 3) **JacobiNet** : encadrer le **gradient** (Lipschitz **locale**)

`jacobinet` construit explicitement le **réseau du backward** (chaîne des dérivées) comme un modèle Keras, ce qui permet :
- de **calculer/visualiser** le **Jacobien** \(J_f(x)\) ou des directions/colonnes spécifiques ;
- d’estimer des **normes opérateur** \(\|J_f(x)\|_{p\to q}\) (i.e., une **constante de Lipschitz locale**) ;
- de définir des **régularisations** basées Jacobien (pénaliser \(\|J_f(x)\|\)) ou d’auditer la **stabilité** des explications par gradient.

**Lien avec la robustesse locale.**  
Pour \(\|\delta\|_p\) petit,
\[
\|f(x+\delta)-f(x)\|_q \;\approx\; \|J_f(x)\|_{p\to q}\,\|\delta\|_p,
\]
donc borner \(\|J_f(x)\|\) **resserre** les certificats **locaux** et éclaire la **sensibilité** instance-dépendante.




# Evaluation de la stabilité du modèle
Une bonne explication \(\phi(x,y)\) doit être **stable localement** : de petites perturbations d’entrée qui ne changent (presque) pas la prédiction ne devraient **pas** faire varier fortement l’explication. La métrique [AverageStability de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/metrics/stability.py) quantifie cette sensibilité locale : plus sa valeur est **faible**, plus les explications sont **robustes** au bruit.

Soit un explainer \(\phi:\mathbb{R}^d\times\mathcal{Y}\to\mathbb{R}^p\).  
Pour chaque échantillon \((x_i,y_i)\), on calcule l’explication de base \(\phi_i=\phi(x_i,y_i)\).  
On génère \(K\) voisins bruités :
\[
\varepsilon_{ik}\sim \mathcal{U}([0,r])^{d},\qquad x_{ik}=x_i+\varepsilon_{ik},\qquad k=1,\dots,K,
\]
puis leurs explications \(\phi_{ik}=\phi(x_{ik},y_i)\) (même cible que \(y_i\), répliquée).

On mesure la distance moyenne entre explications bruitées et explication de base :
\[
\bar d_i \;=\; \frac{1}{K}\sum_{k=1}^{K} d\big(\phi_{ik},\,\phi_i\big),
\]
et on agrège sur \(N\) échantillons :
\[
S \;=\; \frac{1}{N}\sum_{i=1}^{N} \bar d_i.
\]

Ici \(d(\cdot,\cdot)\) est une métrique choisie par l’utilisateur, typiquement :
\[
d_{L1}(a,b)=\sum_{j=1}^{p} |a_j-b_j|,\qquad
d_{L2}(a,b)=\sqrt{\sum_{j=1}^{p} (a_j-b_j)^2}.
\]

On a comme paramètres :

- **Rayon \(r\)** : contrôle l’échelle des perturbations (plus \(r\) est grand, plus \(S\) tend à croître).  
- **Nombre d’échantillons \(K\)** : réduit la variance de l’estimation de \(S\).  


# Evaluation de la fidélité du modèle



# Explication de la prédiction du modèle

Il existe plusieurs approches pour expliquer la génération d'un modèle de deep learning. Chacune de ces familles et sous-familles dépendent notamment du type d'entrée (texte, image, données tabulaires, etc) et du type de modèle (convolution, transformers, etc). On illustre chacune des méthodes pour 2 types de cas d'usage: classification de textes et détection d'objets dans des images.

A noter que le cas d'usage de classification de texte qu'on va utiliser est sur la classification de NOTAMs (notes envoyées au pilote), via un [ModernBert](https://huggingface.co/answerdotai/ModernBERT-base) fine-tuné sur le jeu de données de classification de [NOTAMS](https://huggingface.co/datasets/DEEL-AI/NOTAM), et que le cas d'usage de détection d'objet dans les images qu'on va utiliser est pour la prédiction de piste d'atterrissage dans les image, avec un Yolov8 fine-tuné sur la détection de pistes d'atterrissage (LARD_train_BIRK_LFST: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/MZSH2Y). Le modèle utilisé est https://github.com/AnnabellePundaky/runway-bounding-box-detection-NEW.



## Explication de la génération par carte d'attribution de la donnée d'entrée obtenue par calcul de gradient par rapport à l'entrée pour identifier les zones d'intérêt sur l'input portées par le modèle pour qu'il effectue sa génération

Les méthodes d'attribution basées sur le gradient exploitent les dérivées partielles du modèle pour quantifier l'importance de chaque élément unitaire de la donnée d'entrée dans une prédiction donnée. Le principe fondamental repose sur le calcul du gradient de la fonction de sortie par rapport aux différents éléments de la donnée d'entrée. 

Voici un exemple de résultat obtenu pour l'ensemble des méthodes décrites pour la prédiction de la classe "Landing Navaids" (prédiction correcte) par un modèle entraîné sur le jeu de données de classification de NOTAM :

{% include figure.liquid loading="eager" path="assets/img/explainable_deeplearning/gradient_based_approaches_text_classif.png" class="img-fluid rounded z-depth-1" zoomable=true %}

A noter que pour appliquer les différentes fonctions présentées ici, il faut juste wrapper correctement le modèle pour qu'il soit compatible avec ce qu'attend Xplain, et avoir accès aux gradients du modèle.

Voici un exemple pour un modèle transformers, et si on veut calculer le gradient par rapport à l'embedding (chacune de ses dimension notamment) des tokens d'entrée::

```python
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from xplique.wrappers import TorchWrapper
from xplique.attributions import Saliency, IntegratedGradients, GradientInput, SmoothGrad, VarGrad, SquareGrad

class ModernBERTEmbedsWrapper(nn.Module):
    def __init__(self, base_model, attention_mask):
        super().__init__()
        self.base = base_model
        self.register_buffer("am", attention_mask)

    def forward(self, inputs):
        B, T, _ = inputs.shape
        am = self.am
        out = self.base(inputs_embeds=inputs, attention_mask=am)
        return out.logits

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device).eval()
wrap = ModernBERTEmbedsWrapper(model, attention_mask).to(device)
wrap.eval()
wrapped_model = TorchWrapper(wrap, device=device)
```

Et maintenant, on peut instancier n'importe quelle méthode d'explicabilité à base de ce wrapped_model:

```python
explainer = Saliency(wrapped, operator="classification", **params_of_the_explainer) # pour la détection d'objets, on peut avoir operator=xplique.Tasks.OBJECT_DETECTION. En fonction de l'approche, on a des paramètres différents (par exemple, IntegratedGradients nécessite de défininir une valeur de baseline)
```
Puis on peut expliquer tous les inputs / outputs que l'on souhaite, en les mettant en tensor pytorch:
```python
import torch.nn.functional as F
texts = ["liste de textes d'entrée du modèle à tester"]
labels = ["liste des labels"]
tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
input_ids = tok["input_ids"].to(device)
attention_mask = tok["attention_mask"].to(device)
emb_layer = model.get_input_embeddings()
with torch.no_grad():
    embeds = emb_layer(input_ids)
y = torch.tensor([label2id[s] for s in labels], device=device)
targets = F.one_hot(y, num_classes=model.config.num_labels).float().cpu().numpy()
E = explainer.explain(embeds.detach().cpu().float().numpy(), targets).numpy()
token_scores = np.linalg.norm(E, ord=2, axis=-1) # on fait la forme L2 des gradient par rapport à chaque dimension de l'embedding (car on a dim_model scores) par token
```



### La méthode [Saliency de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/saliency.py)
La méthode Saliency calcule le gradient absolu de la sortie par rapport à l'entrée du modèle (l'input):

$$ \forall \text{dimension } i \text{ de l'entrée x (pixel, l'embedding d'un token, etc):} S_i = \left| \frac{\partial f(x)}{\partial x_i} \right| $$

A notée qu'ici, chaque dimension de l'entrée peut elle-même être de plusieurs dimensions (exemple d'un pixel en 3 dimensions (R,G,B), ou d'un token en dim_model dimensions). Dans ce cas, soit on prend le maximum des gradients de la prédiction du modèle par rapport à chaque dimension de l'embedding du token d'entrée ou du pixel d'entrée, soit on prend la norme L2 des gradients des dimensions (exemple du papier https://aclanthology.org/2024.emnlp-main.347.pdf), soit on prend la moyenne, ...

Dans le code https://github.com/deel-ai/xplique/blob/master/xplique/attributions/saliency.py: 

```python
class Saliency(WhiteBoxExplainer):

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 reducer: Optional[str] = "max",):
        super().__init__(model, output_layer, batch_size, operator, reducer)

    @sanitize_input_output # Cette fonction s'assure que les inputs et outputs du modèle sont des tf.Tensors
    @WhiteBoxExplainer._harmonize_channel_dimension

    def explain(self, inputs, targets):
        # 1. Calcul du gradient via backpropagation
        gradients = self.batch_gradient(self.model, inputs, targets, self.batch_size)
        
        # 2. Application de la valeur absolue
        gradients = tf.abs(gradients)
        
        # 3. Pour les images RGB, réduction sur les canaux (max par défaut)
        # Cela donne l'importance maximale parmi R, G, B pour chaque pixel
        return gradients
```

A noter que cette approche a plusieurs limites, notamment:
- La valeur absolue du gradient de la prédiction du modèle par rapport aux unités de l'input ne permet pas de distinguer les unités de l'input qui ont un impact négatifs de ceux qui ont un impact positif sur la prédiction modèle
- Quand les unités d'entrée sont de haute dimension ; pour les images, il s'agit des pixels, qui sont de faibles dimensions (R, G, B uniquement), donc "faciles" à aggréger pour obtenir un score unique de saillance par unité (pixel) d'entrée. Par contre, quand on a des unités de l'input qui sont de haute dimension, cela peut être compliqué. C'est par exemple le cas pour les modèles de langue, où l'unité d'un input est un token, plus précisément l'embedding d'un token, qui lui est de très grande dimension (en fonction du modèle). Ainsi, il est compliqué d'agréger les D (D = dim_model) gradients obtenus pour avoir un score unique par unité de l'input. On utilise couremment la norme L2 des gradients de chaque dimensions de l'embedding des tokens d'entrée pour se faire: mais ceci est discutable. *(on peut penser notamment à deux ou plusieurs dimensions qui sont très corrélées, alors leurs composantes de gradient pointent en général dans la même direction, et quand on fait la somme des carrés, ces contributions s’additionnent comme si c’étaient deux informations indépendantes, alors qu’elles portent le même signal, du coup la norme L2 des gradients par rapport à chaque dimension du modèle n'est pas forcément optimale)*
- Le gradient de la prédiction par rapport aux unités d'entrée n'a pas forcément de sens en termes absolu: un gradient petit n’implique pas forcément que l'unité de l'input n'est pas important: une unité peut être très influente mais avoir un gradient (local) faible. $\nabla_{x_i}(f(x))$ approxime l’effet de micro-perturbations de l'unité $i$ de l'input, pas le remplacement de cette unité (opération discrète et non-locale). Une unité peut être cruciale pour la classe, tout en ayant un gradient local faible. En effet, Le gradient est une pente locale. Si, autour de l'unité $i$ de l'input, la fonction "s’aplatit", alors la pente est proche de 0, donc le gradient local est faible, même si l'unité $i$ porte une info décisive pour la prédiction. 

D'autres approches permettent de combler l'effet "local" du gradient, notamment les approches d'Integrated Gradients, ou de Gradient x Input.


### La méthode des [Gradients Intégrés de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/integrated_gradients.py)

Les **gradients intégrés (IG)** attribuent à chaque caractéristique $i$ une contribution **cumulative** le long d’un chemin qui relie une **référence (baseline)** $x'$ à l’entrée $x$. En choisissant le **chemin linéaire** $\gamma(\alpha)=x' + \alpha\,(x-x')$, $\alpha \in [0,1]$, l’attribution IG pour la $i$-ème dimension est
$$
\boxed{\;\mathrm{IG}_i(x; x') \;=\; (x_i - x'_i)\,\int_{0}^{1} \frac{\partial F\big(\gamma(\alpha)\big)}{\partial x_i}\, d\alpha\;}
$$
et, en pratique, on l’approxime par une somme de Riemann avec $m$ pas :
$$
\mathrm{IG}_i(x; x') \;\approx\; (x_i - x'_i)\,\frac{1}{m}\sum_{k=1}^{m} \left.\frac{\partial F(z)}{\partial x_i}\right|_{z\,=\,x' + \tfrac{k}{m}(x-x')}.
$$
En **NLP**, on applique IG sur l’**espace d’embedding** : si $e(x)\in\mathbb{R}^{d}$ est l’entrée réelle du réseau (concaténation des embeddings par token), on interpole $e' + \alpha\,(e-e')$ et on dérive $F$ par rapport aux composantes d’**embedding** (la baseline $e'$ est souvent le vecteur nul, un token [PAD], ou un embedding « neutre »).

IG contourne le biais **local** des approches de gradient pur en **agrégeant** l’information de gradient **le long du chemin** $\gamma(\alpha)$ depuis la baseline $x'$ (où la sortie est « neutre ») vers $x$. Intuitivement, même si $\nabla F(x)\approx 0$, il existe souvent des $\alpha\in(0,1)$ où $\nabla F(\gamma(\alpha))$ est **grand** (région non saturée) ; l’intégrale
$$
\int_{0}^{1} \frac{\partial F\big(x' + \alpha(x-x')\big)}{\partial x_i}\, d\alpha
$$
**accumule** ces contributions, produisant une attribution fidèle au **chemin causal continu** qui mène de $x'$ à $x$. IG est ainsi une version **chemin-intégrée** du gradient, reliée à la **valeur d’Aumann–Shapley** (analogue continu des valeurs de Shapley), et hérite d’une interprétation en termes de **coût marginal moyen** le long de l’activation du feature $i$.

A noter que le choix de la baseline $x'$ est majeur: il doit représenter une **entrée de référence** « absence d’information » (image noire, bruit faible, embedding nul/[\mathrm{PAD}], etc.). Le résultat dépend de ce choix, mais la **complétude** garantit $\sum_i \mathrm{IG}_i = F(x)-F(x')$.

Exemple d'application:

```python
from xplique.attributions import IntegratedGradients

explainer = IntegratedGradients(
    wrapped_model,
    steps=50, # Nombre de points d'interpolation
    baseline_value=0.0 # Valeur de référence pour la baseline
)
attributions = explainer(inputs, targets)
```

Attention en revanche: Dans la librairie Xplique, on approxime l’intégrale continue $I_i=\int_0^1 g_i(\alpha)\,d\alpha$ — i.e. la moyenne du gradient $g_i(\alpha)=\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}$ le long du chemin $\gamma(\alpha)=x'+\alpha(x-x')$ avec la règle du trapèze. Comme on ne dispose pas d’une primitive explicite pour un réseau de neurones, on discrétise l’intégrale en une somme finie sur $m$ points. Plusieurs schémas sont possibles :

- **Riemann (rectangle)** : $I_i \approx \frac{1}{m}\sum_{k=1}^{m} g_i\!\big(\tfrac{k}{m}\big)$ (erreur $O(1/m)$) ;
- **Milieu (midpoint)** : $I_i \approx \frac{1}{m}\sum_{k=1}^{m} g_i\!\big(\tfrac{k-\tfrac12}{m}\big)$ (erreur $O(1/m^2)$) ;
- **Trapèze (composite)** : $I_i \approx \frac{1}{m}\!\Big[\tfrac12 g_i(0)+\sum_{k=1}^{m-1} g_i\!\big(\tfrac{k}{m}\big)+\tfrac12 g_i(1)\Big]$ (erreur $O(1/m^2)$, plus précis et plus symétrique) ;

La librairie Xplique utilise la règle du trapèze, qui offre une meilleure précision (et respecte mieux la propriété de complétude $\sum_i \mathrm{IG}_i \approx F(x)-F(x')$) à coût identique.

A noter que la valeur de référence pour la baseline est majeure: par exemple, on peut prendre le token de padding pour le texte.

### La méthode [gradient x input de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/gradient_input.py)

Gradient × Input attribue l’importance d’une caractéristique $x_i$ comme le produit élément-par-élément entre sa sensibilité locale et sa présence effective : 
$$
\mathbf{A}(x)\;=\;x\;\odot\;\nabla_x g(f,x,y),\quad\text{soit}\quad A_i(x)=x_i\,\frac{\partial g(f,x,y)}{\partial x_i}.
$$
Le gradient seul $\partial g/\partial x_i$ mesure "à quel point" la sortie changerait si l’on bougeait $x_i$, mais ne tient pas compte de combien de cette caractéristique est présente dans l’entrée ; en le pondérant par $x_i$, on obtient une contribution sensibilité × magnitude. Dans un modèle linéaire $g(x)=w^\top x$, on a $\nabla_x g = w$ et donc $A_i = x_i w_i$, ce qui correspond exactement à la part de $x_i$ dans la sortie. Plus généralement, c’est la décomposition de Taylor d’ordre 1 autour de 0 : $g(x)\approx g(0)+\sum_i x_i\,\partial g/\partial x_i$, d’où une attribution locale et signée. En pratique (y compris dans Xplique), on calcule le gradient d’entrée du scalaire choisi (p. ex. logit de classe via un *operator*) puis on le multiplie élément-par-élément par $x$ pour produire la carte d’attributions.

```python
class GradientInput(WhiteBoxExplainer):
    @sanitize_input_output
    @WhiteBoxExplainer._harmonize_channel_dimension
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        gradients = self.batch_gradient(self.model, inputs, targets, self.batch_size)
        gradients_inputs = tf.multiply(gradients, inputs)
        return gradients_inputs
```

### La méthode [SmoothGrad de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/gradient_statistics/smoothgrad.py)

Les cartes de saillance basées sur le gradient pur $\nabla_x g(f,x,y)$ sont souvent granulaires : de très petites variations d’entrée (ou du point d’évaluation dans les activations) peuvent faire fluctuer fortement le gradient à cause des non-linéarités locales (ReLU, max-pool, normalisations) ou des changements de région affine notamment. Résultat : des pixels/tokens isolés "s’allument" ou "s’éteignent" sans cohérence spatiale/sémantique. Visuellement, ça donne du bruit.

Plutôt que de se fier au gradient en un point unique $x$, SmoothGrad moyenne les gradients dans un petit voisinage gaussien de $x$. Cette moyenne annule statistiquement les fluctuations idiosyncratiques (le « bruit ») et renforce les tendances stables (structure commune dans le voisinage). 

Soit $h(x) \equiv g(f,x,y)$ un scalaire (ex. logit de la classe $y$). On introduit un bruit additif
$$
\delta \sim \mathcal N(0,\sigma^2 I),
$$
où $\delta$ est un tenseur de même forme que $x$, dont chaque composante est tirée d’une loi normale centrée d’écart-type $\sigma$. On définit :
$$
\phi_{\mathrm{SG}}(x)
\;=\;
\mathbb E_{\delta}\big[\nabla_x h(x+\delta)\big]
\;\approx\;
\frac{1}{N}\sum_{i=1}^N \nabla_x h\!\big(x+\delta_i\big),
\quad
\delta_i \stackrel{\text{i.i.d.}}{\sim} \mathcal N(0,\sigma^2 I).
$$
- $\sigma$ (*noise*) règle la taille du voisinage ; on le choisit relatif à l’échelle des entrées (p.ex. $\sigma\approx 0{,}2$ si $x\in[0,1]$).
- $N$ (*nb\_samples*) contrôle la variance Monte-Carlo de l’estimateur ($\propto 1/\sqrt{N}$) et le coût (il faut $N$ gradients).

Pour plus de précision sur la façon dont cela est codé:
```python
class GradientStatistic(WhiteBoxExplainer, ABC):

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 reducer: Optional[str] = "mean",
                 nb_samples: int = 50,
                 noise: float = 0.2):
        super().__init__(model, output_layer, batch_size, operator, reducer)
        self.nb_samples = nb_samples
        self.noise = noise

    @sanitize_input_output
    @WhiteBoxExplainer._harmonize_channel_dimension
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        batch_size = self.batch_size or (len(inputs) * self.nb_samples)
        perturbation_batch_size = min(batch_size, self.nb_samples)
        inputs_batch_size = max(1, batch_size // perturbation_batch_size)

        smoothed_gradients = []
        # loop over inputs (by batch if batch_size > nb_samples, one by one otherwise)
        for x_batch, y_batch in batch_tensor((inputs, targets), inputs_batch_size):
            total_perturbed_samples = 0

            # reset online statistic values
            self._initialize_online_statistic()

            # loop over perturbation (a single pass if batch_size > nb_samples, batched otherwise)
            while total_perturbed_samples < self.nb_samples:
                nb_perturbations = min(perturbation_batch_size,
                                       self.nb_samples - total_perturbed_samples)
                total_perturbed_samples += nb_perturbations

                # add noise to inputs
                perturbed_x_batch = GradientStatistic._perturb_samples(
                    x_batch, nb_perturbations, self.noise)
                repeated_targets = repeat_labels(y_batch, nb_perturbations)

                # compute the gradient of each noisy samples generated
                gradients = self.batch_gradient(
                    self.model, perturbed_x_batch, repeated_targets, batch_size)

                # group by inputs and compute the average gradient
                gradients = tf.reshape(
                    gradients, (x_batch.shape[0], nb_perturbations, *gradients.shape[1:]))

                # update online estimation
                self._update_online_statistic(gradients)

            # extract online estimation
            reduced_gradients = self._get_online_statistic_final_value() # pour SmoothGrad, cette fonction retourne la moyenne, pour VarGrad la variance, etc.
            smoothed_gradients.append(reduced_gradients)

        smoothed_gradients = tf.concat(smoothed_gradients, axis=0)
        return smoothed_gradients

    @staticmethod
    @tf.function
    def _perturb_samples(inputs: tf.Tensor,
                         nb_perturbations: int,
                         noise: float) -> tf.Tensor:
        perturbed_inputs = tf.repeat(inputs, repeats=nb_perturbations, axis=0)
        perturbed_inputs += tf.random.normal(perturbed_inputs.shape, 0.0, noise, dtype=tf.float32)
        return perturbed_inputs
```

### La méthode [VarGrad de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/gradient_statistics/vargrad.py)
Alors que SmoothGrad estime la moyenne des gradients sous bruit, VarGrad estime quant à lui la variance de ces mêmes gradients. Il mesure à quel point le gradient fluctue quand on perturbe légèrement l’entrée. Si le gradient est cohérent/stable dans le voisinage (même signe, même amplitude), SmoothGrad peut être fort alors que VarGrad sera faible. S’il change beaucoup (amplitude/signe) selon les perturbations, VarGrad sera élevé et mettra en avant des zones instables/fragiles.

### La méthode [SquareGrad de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/gradient_statistics/square_grad.py)
SquareGrad est la somme de ce que captent SmoothGrad (moyenne) et VarGrad (variance). Il met en avant l’intensité totale de la sensibilité locale, indépendamment du signe du gradient.

## Explication de la génération par carte d'attribution de la donnée d'entrée obtenu par approches de substitution de certaines parties de l'entrée pour identifier les zones d'intérêt sur l'input portées par le modèle pour qu'il effectue sa génération

{% include figure.liquid loading="eager" path="assets/img/explainable_deeplearning/patch_based_approaches_text_classification.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### L'approche d'Occlusion

Cette approche consiste à faire glisser un patch de masquage sur l’entrée et, à chaque position, on remplace localement les valeurs par une constante (par exemple 0), puis on mesure la baisse du score de la classe cible par rapport au score de base sans masque. En gros, on calcule une différence de score pour la région avec et sans ce masque. Plus la baisse est grande quand une région est masquée, plus cette région est jugée importante. L’attribution finale est obtenue en représentant la baisse de score sur toutes les positions couvertes par le patch et en additionnant sur tous les patchs qui recouvrent chaque position. On produit ainsi une carte de saillance de même taille spatiale que l’entrée.

Formule compacte : si $s_0=g(f,x,y)$ est le score de base et $s_p=g\!\big(f,(1-m_p)\odot x+m_p\odot v, y\big)$ le score avec le patch $m_p$, alors l’attribution en $i$ est
$$
\Phi(x)_i=\sum_{p:\, i\in \mathrm{supp}(m_p)} \big(s_0 - s_p\big).
$$

```python
# Préparation
masks = all_sliding_window_masks(input_shape, patch_size, patch_stride)
s0 = g(f, x, y)  # score de base, sans masquage: f est le modèle, x est l'entrée, y est la prédiction du modèle

phi = zeros_like_spatial(x)  # carte d’attribution

for batch_masks in batch(masks, batch_size):
    # Appliquer les masques (broadcast sur canaux si image)
    X_occ = apply_masks(x, batch_masks, v)

    # Répéter y pour matcher le batch de masques
    Y_rep = repeat(y, len(batch_masks))

    # Scores occlus
    s_occ = g(f, X_occ, Y_rep)  # shape: [len(batch_masks)]

    # Variations de score
    delta = s0 - s_occ  # shape: [len(batch_masks)]

    # Peindre et sommer sur la dimension « patch »
    phi += sum_over_patches(delta[:, None, ...] * batch_masks, axis="patch")

return phi
```

Cependant, l’occlusion est coûteuse: pour chaque région (définie par patch_size / stride), on doit réaliser une inférence sur l’image masquée et comparer au score de l’image originale (calculé une seule fois). Le nombre d’inférences croît donc linéairement avec le nombre de patches balayés, et la résolution de la carte dépend directement de la taille du patch (patchs gros donne une carte grossière ; patchs petits donne un coût élevé).

### L'approche Rise

C’est pourquoi RISE remplace le balayage exhaustif par un échantillonnage Monte-Carlo de masques aléatoires. Rise estime une espérance conditionnelle via échantillonnage aléatoire de masques qui préservent/éteignent des régions, puis moyenne les masques pondérés par le score du modèle.







## Explication de la génération du modèle par attribution de sa génération aux données d'entraînements qu'il a vu qui ont positivement ou négativement influencées sa prédiction

Les fonctions d'influence permettent d'approximer le Leave-one-out, c'est à dire cherche à estimer l'<mark>impact qu'aurait un exemple d'entraînement sur la perte d'un exemple de test (ou sur plusieurs résultats du modèle sur un jeu de données test)</mark>.

$$
\begin{split}
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) &= \frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\,\theta_\varepsilon(z_\text{train})\bigr)\Big|_{\varepsilon=0} \\ 
&\approx -\nabla_\theta \,\mathcal{L}\bigl(z_{\mathrm{test}},\,\hat{\theta} \bigr)\,H_\theta^{-1}(\hat{\theta})\,\nabla_\theta \mathcal{L}(z_{\rm train},\hat{\theta}) \\
&\approx -\nabla_\theta \,\mathcal{L}\bigl(z_{\mathrm{test}},\,\hat{\theta} \bigr)\,(G_\theta(\hat{\theta}) + \lambda I)^{-1}\,\nabla_\theta \mathcal{L}(z_{\rm train},\hat{\theta})
\end{split}
$$

Où:
- $$\hat{\theta}$$ sont les poids du modèle préentraîné (poids supposés optimiaux pour le jeu de pretraining du LLM)
- $$\theta_\varepsilon(z_\text{train})$$ sont les poids du modèle "modifié", ie lorsqu'on upweight de $\varepsilon$ l'exemple d'entraînement $z_\text{train}$
- $$\theta$$ est une variable muette (eg pour le gradient, on dérive par rapport aux paramètres du modèle et on applique en un point)
- Tous les $\theta$ sont des vecteurs colonne des poids du modèle
- $\mathcal{L}$ est la loss du modèle, donc ici, vu qu'on est dans le cas des LLMs, c'est la cross-entropy loss (negative log-likelihood), ie pour une séquence de taille $T$ ($x_1$, $x_2$, ..., $x_T$): $$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$
- $f$ est une fonction (moyenne, ou autre) sur plusieurs résultats du modèle sur un jeu de données test noté $x$. Il nous permet de ne pas évaluer l'impact de l'upweight d'un $z_\text{train}$ sur la loss d'un $z_\text{test}$, mais sur un ensemble de type de données $x$ (car parfois, on ne veut pas calculer l'influence d'une donnée d'entraînement sur la loss d'un prompt, mais sur la performance (pas forcément la loss d'ailleurs) sur un certain type de prompt)
- $H_\theta$ est la hessienne (dérivée seconde par rapport aux paramètres du modèle)
- $G_\theta$ est la Hessienne de Gauss-Newton qu'on va voir plus loin dans ce post, qui est en fait une approximation de la hessienne
- $\lambda$ est un terme dit de "damping", dû au fait que la loss n'est pas convexe, qu'on verra plus loin dans le post

Plus précisément, avec les foncitons d'influence, on cherche à estimer l'**impact de l"up-weight" de la loss sur $z_\text{train}$ sur qqch, ie on se pose la question: "<mark>si je donnais un peu plus de poids à ce terme de loss dans l’objectif global, comment cela ferait-il bouger mes paramètres et, avec ces nouveaux paramètres, comment ça modifierait ma performance sur un point de test, ou sur une fonction?</mark>"**.
A noter que <mark>$f_{\theta_\varepsilon(z_\text{train})}(x)$ peut être n'importe quelle fonction (par exemple ça peut être la moyenne des prédictions sur un ensemble de données types $x$</mark> (cf le papier [Which Data Attributes Stimulate Math and Code Reasoning? An Investigation via Influence Functions](https://arxiv.org/pdf/2505.19949) qui cherche à calculer l'influence des textes d'entraînement sur la génération de code (moyenne de des log probabilité de la génération de chaque token de code générés dans un benchmark sachant un problème de code en langage naturel à résoudre)), la différence entre 2 prédictions du modèle, ...)

## Explication de la génération du modèle par la façon dont le modèle (couche par couche, neurone par neurone, etc) a traité l'entrée pour effectuer sa prédiction (quels concepts intermédiaires a-t-il représenté, etc)

Une des librairies qui explique les concepts d'un modèle de vision est [Craft de Deel-AI](https://github.com/deel-ai/Craft)
TODO