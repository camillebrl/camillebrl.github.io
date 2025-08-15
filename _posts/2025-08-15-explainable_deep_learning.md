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

# Evaluation de la robustesse du modèle

TODO : [deel-lip de Deel-IA](https://github.com/deel-ai/deel-lip)

# Explication de la prédiction du modèle

Il existe plusieurs approches pour expliquer la génération d'un modèle de deep learning:

## Explication de la génération par carte d'attribution de la donnée d'entrée pour identifier les zones d'intérêt sur l'input portées par le modèle pour qu'il effectue sa génération

Dépend du cas d'usage et du type de modèle et du type d'input.

Certaines fonctions de la librairie sont applicables à tout type de modèle (pytorch via wrapper, tensorflow, keras) et tout type de données et use-case, comme la fonction Saliency.

### La méthode [Saliency de Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/saliency.py)
La méthode Saliency calcule le gradient absolu de la sortie par rapport à l'entrée :

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
Exemple d'application:

```python
import torch
import torchvision.models as models
from xplique.wrappers import TorchWrapper
from xplique.attributions import Saliency
import numpy as np

# Exemple avec un modèle pytorch
model_pt = models.resnet18(pretrained=False)
model_pt.load_state_dict(torch.load('mon_modele.pt'))
model_pt.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
wrapped_model = TorchWrapper(model_pt, device) # pas besoin pour un modèle tensorflow

explainer = Saliency(wrapped_model, reducer="max") # reducer pour le traitement d'un input de plusieurs dimensions (exemple R, G, B pour des pixels)

attributions = explainer(inputs, targets) # les inputs et targets peuvent être en numpy array
```

A noter que cette approche a plusieurs limites, notamment:
- Quand les unités d'entrée sont de haute dimension ; pour les images, il s'agit des pixels, qui sont de faibles dimensions (R, G, B uniquement), donc "faciles" à aggréger pour obtenir un score unique de saillance par unité (pixel) d'entrée. Par contre, quand on a des unités de l'input qui sont de haute dimension, cela peut être compliqué. C'est par exemple le cas pour les modèles de langue, où l'unité d'un input est un token, plus précisément l'embedding d'un token, qui lui est de très grande dimension (en fonction du modèle). Ainsi, il est compliqué d'agréger les D (D = dim_model) gradients obtenus pour avoir un score unique par unité de l'input.
- La valeur absolue du gradient de la prédiction du modèle par rapport aux unités de l'input ne permet pas de distinguer les unités de l'input qui ont un impact négatifs de ceux qui ont un impact positif sur la prédiction modèle
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


```python
class IntegratedGradients(WhiteBoxExplainer):
    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 reducer: Optional[str] = "mean",
                 steps: int = 50,
                 baseline_value: float = .0):
        super().__init__(model, output_layer, batch_size, operator, reducer)
        self.steps = steps
        self.baseline_value = baseline_value

    @sanitize_input_output
    @WhiteBoxExplainer._harmonize_channel_dimension
    def explain(self, inputs, targets):
        # Créer la baseline (point de référence neutre)
        baseline = tf.ones(shape) * self.baseline_value
        
        # Créer le chemin interpolé
        for step in range(self.steps):
            alpha = step / self.steps
            interpolated = baseline + alpha * (inputs - baseline)
            
        # Calculer les gradients pour chaque point
        interpolated_gradients = self.batch_gradient(model, interpolated_inputs, targets)
        
        # Moyenner avec la règle trapézoïdale
        trapezoidal_gradients = gradients[:-1] + gradients[1:]
        averaged_gradients = tf.reduce_mean(trapezoidal_gradients) * 0.5
        
        # Multiplier par la différence input-baseline
        integrated_gradients = (inputs - baseline) * averaged_gradients
        
        return integrated_gradients
```

### Détection d'objets sur des images (exemple de LARD)

Exemple de Yolov8 fine-tuné sur la détection de pistes d'atterrissage (LARD_train_BIRK_LFST: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/MZSH2Y). Le modèle utilisé est https://github.com/AnnabellePundaky/runway-bounding-box-detection-NEW.

### Classification de texte

Exemple de [ModernBert](https://huggingface.co/answerdotai/ModernBERT-base) fine-tuné sur la classification de [NOTAMS](https://huggingface.co/datasets/DEEL-AI/NOTAM). Voici quelques métriques d'entraînement:

{% include figure.liquid loading="eager" path="assets/img/explainable_deeplearning/modernbert_notam.png" class="img-fluid rounded z-depth-1" zoomable=true %}

#### Exemple de l'approche [Saliency de la lib Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/saliency.py)

On utilise [Saliency de Xplain](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/saliency.py) pour calculer le gradient absolu de la prédiction (classe de NOTAM, one-hot) du modèle par rapport à chaque dimension de chaque token d'entrée et qu'on applique une norme L2 sur les gradients de chaque dimension d'un token pour obtenir une valeur de saliency par token.

##### Code d'implémentation de cette approche d'explicabilité sur un modèle de classification de texte

```python
import torch, numpy as np
from xplique.wrappers import TorchWrapper
from xplique.attributions import Saliency
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Le modèle: c'est un modèle de langue donc il faut loader le modèle et le tokenizer avec la lib transformers
MAX_LENGTH = 512
OUTPUT_DIR = "./modernbert_notam"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device).eval()
model.eval().to(device)
# Wrapper qui attend des embeddings et injecte le mask pour être adapté aux requirements de TorchWrapper de xplain
class ModernBERTEmbedsWrapper(nn.Module):
    def __init__(self, base_model, attention_mask):
        super().__init__()
        self.base = base_model
        self.register_buffer("am", attention_mask)
        self.base.eval() # On s'assure que le modèle interne est en eval
        self.train(False) # Et on met aussi le wrapper en eval

    def forward(self, inputs):
        out = self.base(inputs_embeds=inputs, attention_mask=self.am)
        return out.logits

# le dataset de test
dataset = load_dataset("DEEL-AI/NOTAM")
unique_labels = set(dataset['train']['label'])
num_labels = len(unique_labels)
label2id = {label: i for i, label in enumerate(sorted(unique_labels))}
id2label = {i: label for label, i in label2id.items()}
k = 16 # batch_size
texts = dataset['test']['text'][:k]
labels_str = dataset['test']['label'][:k]
y = torch.tensor([label2id[s] for s in labels_str], device=device)  # shape (batch_size, )
tok = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
input_ids = tok["input_ids"].to(device) # shape (batch_size, amount of tokens per sentence)
attention_mask = tok["attention_mask"].to(device) # shape (batch_size, amount of tokens per sentence)
# Embeddings d'entrée de shape (batch_size, amount of tokens per sentence, model_dim)
emb_layer = model.get_input_embeddings()
with torch.no_grad():
    embeds = emb_layer(input_ids) # shape (batch_size, amount of tokens per sentence, model_dim)

# On wrap le modèle avec le TorchWrapper de xplain
wrap = ModernBERTEmbedsWrapper(model, attention_mask).to(device)
wrapped = TorchWrapper(wrap, device=device)

# On appliquer la classe Saliency
explainer = Saliency(wrapped, operator="classification")
# Xplique attend des targets en format numpy arrays (ici, des one-hot (batch_size, num_labels))
num_labels = len(id2label)
targets = F.one_hot(y, num_classes=num_labels).float().cpu().numpy()
# Explanations: saliency sur les embeddings des tokens d'entrée (batch_size, amount of tokens per sentence, model_dim)
embeds_np = embeds.detach().cpu().float().numpy()
E = explainer.explain(embeds_np, targets).numpy()
# Obtention d'un score de saillance par token (norme L2 des gradients par rapport à chaque dim_model des tokens) + masquage padding -> (shape (batch_size, amount of tokens per sentence)
token_scores = np.linalg.norm(E, ord=2, axis=-1) # norme L2 sur dim_model (sur les dim_model gradients)
token_scores = token_scores * attention_mask.cpu().numpy()
```

Et maintenant on peut afficher les cartes de saillance sur les tokens pour les exemples test:

```python
from IPython.display import display, HTML
import numpy as np

text = texts[0] # on affiche pour le premier exemple du test set

enc = tokenizer(
    text,
    add_special_tokens=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_offsets_mapping=True,
    return_attention_mask=True
)

ids_single = enc["input_ids"]
offsets = enc["offset_mapping"]
mask_single = enc["attention_mask"]
scores = token_scores[i]

# Normalisation (pour avoir des scores entre 0 et 1)
smin, smax = float(scores.min()), float(scores.max())
denom = (smax - smin) if (smax - smin) > 1e-12 else 1.0
scores_norm = (scores - smin) / denom

# Construit l'HTML avec un <span> par token non-spécial
parts = []
cursor = 0
T_use = min(len(ids_single), len(scores)) # sécurité si longueurs diffèrent

for t_idx in range(T_use):
    tok_id = ids_single[t_idx]
    (s, e) = offsets[t_idx]
    if mask_single[t_idx] == 0: # padding atteint
        break
    if tok_id in tokenizer.all_special_ids: # saute [CLS], [SEP], etc.
        continue
    if e <= s:  # parfois des offsets vides
        continue

    # Ajoute tout "trou" éventuel non couvert par les offsets (espaces, etc.)
    if s > cursor:
        parts.append(text[cursor:s])

    # Couleur = rouge avec alpha = score normalisé
    alpha = float(scores_norm[t_idx])
    bg = f"rgba(255, 0, 0, {alpha:.3f})"

    seg = text[s:e]
    parts.append(
        f'<span title="score={float(scores[t_idx]):.6f}" '
        f'style="background-color:{bg}; border-radius:4px; padding:2px 1px;">'
        f'{seg}'
        f'</span>'
    )
    cursor = e

# Ajoute la fin de phrase si besoin
if cursor < len(text):
    parts.append(text[cursor:])

html = (
    '<div style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; '
    'line-height:1.8; font-size:14px;">'
    + ''.join(parts) +
    '</div>'
    '<div style="margin-top:8px; font-size:12px; color:#555;">'
    'Plus c’est <b>rouge</b>, plus le token est important (Saliency). '
    'On peut passer la souris pour voir le score de saillance de chaque token.'
    '</div>'
)

display(HTML(html))
```

{% include figure.liquid loading="eager" path="assets/img/explainable_deeplearning/example_notam.png" class="img-fluid rounded z-depth-1" zoomable=true %}

##### Les limites de cette approche

- La valeur absolue du gradient de la prédiction du modèle par rapport aux dimensions de l'embedding du token d'entrée: ça ne permet pas de distinguer les dimensions des tokens qui ont un impact négatifs de ceux qui ont un impact positif sur la prédiction modèle
- Norme L2 des gradients de chaque dimensions de l'embedding des tokens d'entrée: est-ce une bonne approche? *(on peut penser notamment à deux ou plusieurs dimensions qui sont très corrélées, alors leurs composantes de gradient pointent en général dans la même direction, et quand on fait la somme des carrés, ces contributions s’additionnent comme si c’étaient deux informations indépendantes, alors qu’elles portent le même signal, du coup la norme L2 des gradients par rapport à chaque dimension du modèle n'est pas forcément optimale)*
- Le gradient de la prédiction par rapport aux tokens d'entrée n'a pas forcément de sens en termes absolu: un gradient petit n’implique pas forcément que le token n'est pas important: un token peut être très influent mais avoir un gradient (local) faible. $\nabla_{x_i}(f(x))$ approxime l’effet de micro-perturbations de l’embedding du token $i$, pas le remplacement de ce token (opération discrète et non-locale). Un token peut être crucial pour la classe, tout en ayant un gradient local faible. En effet, Le gradient est une pente locale. Si, autour du token d'entrée $i$, la fonction "s’aplatit", alors la pente est proche de 0, donc le gradient local est faible, même si le token $i$ porte une info décisive pour la prédiction.

D'autres approches permettent de combler l'effet "local" du gradient, notamment les approches d'Integrated Gradients, ou de Gradient x Input.

#### Exemple de l'approche des [Gradients intégrés de la lib Xplain de deel-ai](https://github.com/deel-ai/xplique/blob/master/xplique/attributions/integrated_gradients.py)

Dans Xplique, IntegratedGradients accepte un argument baselines dans explain. Si on ne fournit rien, le comportement standard est d’utiliser un tenseur de zéros de même forme que l’entrée (baseline "no-signal"). Sauf que pour le NLP, un tenseur de 0 n'a pas forcément de sens. Une baseline "neutre" serait plutôt par exemple des [PAD] tokens.

TODO: tester IG avec 0 et [PAD] tokens

### Classification d'images

#### Avec un réseau à base de convolutions (type ResNet)

Resnet-18 entraîné sur http://cs231n.stanford.edu/tiny-imagenet-200.zip

- GradCam de la librairie xplique : https://github.com/deel-ai/xplique/blob/master/xplique/attributions/grad_cam.py
- SmoothGrad

#### Avec un ViT


### Exemple pour une entrée multimodale (texte + image)
TODO - cas layoutlmv3


## Explication de la génération du modèle par attribution de sa génération aux données d'entraînements qu'il a vu qui ont positivement ou négativement influencées sa prédiction

Si on cherche à estimer l'<mark>impact qu'aurait un exemple d'entraînement sur la perte d'un exemple de test (ou sur plusieurs résultats du modèle sur un jeu de données test) à un exemple d'entraînement</mark> (qu'il soit dans le jeu de données d'entraînement de base ou pas), on peut utiliser les fonctions d'influence. 

$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to z_{\mathrm{test}}\bigr) = \frac{d}{d\varepsilon}\, \mathcal{L}\bigl(z_{\rm test},\,\theta_\varepsilon(z_\text{train})\bigr)\Big|_{\varepsilon=0} = -\nabla_\theta \,\mathcal{L}\bigl(z_{\mathrm{test}},\,\hat{\theta} \bigr)\,H_\theta^{-1}(\hat{\theta})\,\nabla_\theta \mathcal{L}(z_{\rm train},\hat{\theta})
$$

$$
\mathrm{Influence}\bigl(z_{\mathrm{train}}\to f(x)\bigr) = \left.\frac{d}{d\varepsilon}\bigl(f_{\theta_{\varepsilon}(z_{\mathrm{train}})}(x)\bigr)\right|_{\varepsilon=0} = - \nabla_\theta f_{\hat{\theta}}(x)^\top \, H_\theta(\hat{\theta})^{-1} \, \nabla_\theta \mathcal{L}\bigl(x_{\mathrm{train}},\hat{\theta}\bigr)
$$ 

Pour voir l'explication de la formule et plus de détails sur comment c'est utilisé dans les LLMs, vous pouvez aller voir [ce post](camillebrl.github.io/blog/2025/influence_functions_applied_to_llms/).

Une des librairies possibles pour calculer l'influence est [Influenciae de Deel-AI](https://github.com/deel-ai/influenciae).
TODO

## Explication de la génération du modèle par la façon dont le modèle (couche par couche, neurone par neurone, etc) a traité l'entrée pour effectuer sa prédiction (quels concepts intermédiaires a-t-il représenté, etc)

Une des librairies qui explique les concepts d'un modèle de vision est [Craft de Deel-AI](https://github.com/deel-ai/Craft)
TODO