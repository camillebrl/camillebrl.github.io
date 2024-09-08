# Introduction

Large Multimodal
models (LMMs) [GPT4V_System_Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf), [liu2023llava](http://arxiv.org/pdf/2402.11690v1), [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1), [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1), [wang2023cogvlm](https://arxiv.org/pdf/2311.03079), [Qwen-VL](http://arxiv.org/pdf/2308.12966v3)
have shown strong performance in visual-linguistic understanding and
reasoning. Models such as
LLaVA [liu2023llava](http://arxiv.org/pdf/2402.11690v1), [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1), [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
first embed the input image with a fixed number of visual tokens, and
then feed them as prefix tokens to a Large Language Model
(LLM) [Vicuna](https://vicuna.lmsys.org/), [llama-3](https://ai.meta.com/blog/meta-llama-3/) to reason about the input image.
Similar model designs are borrowed in video
LMMs [lin2023video](http://arxiv.org/pdf/2311.10122v2), [zhang2023video](http://arxiv.org/pdf/2311.12919v2), where each frame
contributes a fixed number of tokens to form the final video
representation.

In reality, the number of visual tokens can be prohibitively large in
the case of high-resolution images, and even more so for long videos.
Existing
works [lin2023video](http://arxiv.org/pdf/2311.10122v2), [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [zhang2024llavanextvideo](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/), [geminiteam2024gemini](https://arxiv.org/pdf/2312.11805)
mainly tackle this issue by increasing the input context length and
consequently, feeding a large number -8k of visual tokens into the LLM.
This approach has a couple of significant drawbacks: (1) the extremely
long context makes both training and inference inefficient; (2) an
excessive number of visual tokens can actually *harm* the LMM’s
performance, distracting it from attending to the relevant information,
as we show in
Sec. <a href="#sec:exp:video understanding" data-reference-type="ref"
data-reference="sec:exp:video understanding">[sec:exp:video
understanding]</a>. Several recent
works [bolya2023tome](None), [chen2024image-fastv](http://arxiv.org/pdf/2403.06764v2), [shang2024LLaVA-PruMerge](http://arxiv.org/pdf/2403.15388v5)
use heuristics to prune and merge visual tokens to reduce the sequence
length. However, they produce a single-length output and *do not afford
control over the final sequence length*, which could be useful to trade
information density versus efficiency while accounting for resource
constraints in the deployment phase.

<figure id="fig:detail-specturm-visualization">
<embed src="/papers/text_rich/arXiv-2405.17430v1_md/figures/concept.png" />
<figcaption> <strong>.</strong> We enforce the coarser set of visual
tokens <span
class="math inline"><strong>X</strong><sub><em>S</em><sub><em>i</em> − 1</sub></sub></span>
to be derived from the finer level of visual tokens <span
class="math inline"><strong>X</strong><sub><em>S</em><sub><em>i</em></sub></sub></span>.
As a result, the granularity of Matryoshka visual tokens gradually
changes in a controllable manner. The image is from MSCOCO <span
class="citation" data-cites="lin2014microsoft"></span> validation set.
</figcaption>
</figure>

Images and videos naturally exhibit a hierarchical structure from coarse
to fine details, and our human visual system has evolved to recognize
visual information in this coarse to fine manner, as shown by biologists
and psychologists decades
ago [harris2000coarse](http://arxiv.org/pdf/2208.13560v1), [hegde2008time](http://arxiv.org/pdf/2108.02839v1). Can we create a
similar structure for LMMs, where within one suite of model weights, the
visual content tokens are organized into different scales of
granularities? Conceptually, our goal is to learn the visual tokens to
have a nested structure, similar to the Matryoshka
Doll [kusupati2022matryoshka](http://arxiv.org/pdf/2405.17430v1). Matryoshka Representation
Learning (MRL) [kusupati2022matryoshka](http://arxiv.org/pdf/2405.17430v1) builds the
Matryoshka mechanism over a neural network’s representation vector,
where each of the segments with various feature dimensions is capable of
handling tasks like classification or retrieval. However, for LMMs, the
inefficiency mainly comes from the number of tokens. Thus, inspired by,
but different from MRL, our work is motivated to build upon the *token
length dimension*, so that we can flexibly adjust it.

<div class="wrapfigure" markdown="1">

l0.5 <img src="/papers/text_rich/arXiv-2405.17430v1_md/figures/scale_plot_submission.png" style="width:48.0%"
alt="image" />

</div>

Specifically, we propose *:* , which enforces an LMM to learn a
hierarchy of visual representation granularities at the token sequence
level, instead of the feature dimension level as in
MRL [kusupati2022matryoshka](http://arxiv.org/pdf/2405.17430v1). With this representation,
at inference time, the visual granularity can be *flexibly controlled*
based on specific requirements, e.g., to account for the input image’s
information density and efficiency constraints. Our training process is
simple and straightforward. During training, we encode the image into
$M$ sets of visual tokens from coarse to fine, $\mathbf{X} _{S_i}$,
$i = 1, \cdots, M$, where the number of visual tokens gradually
increases, $|\mathbf{X}_{S_{i-1}} | < |\mathbf{X}_{S_i}|$. And
importantly, the visual tokens in a coarser level are derived from the
visual tokens in a finer level,
$\mathbf{X}_{S_{i-1}} \subset \mathbf{X}_{S_i}$, $\forall i$. In this
way, the visual information in
$[ {\mathbf{X}} _{S_1}, {\mathbf{X}} _{S_2}, \cdots, {\mathbf{X}} _{S_M}]$
gradually includes more fine-grained details. For example, given a
natural image as shown in
Figure <a href="#fig:detail-specturm-visualization" data-reference-type="ref"
data-reference="fig:detail-specturm-visualization">1</a>,
$\mathbf{X} _{S_1}$ includes high-level semantics such as the restaurant
and girl, while $\mathbf{X} _{S_M}$ includes more details such as the
Pepsi cup and white paper bag. All other training settings, such as the
loss function and model architecture, are kept the same as
LLaVA [liu2023llava](http://arxiv.org/pdf/2402.11690v1), [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1), [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/).

Our approach, , introduces several novel properties and benefits for
LMMs. First, our approach can adaptively and efficiently represent
visual content. Under *one suite of weights*, it generates multiple
nested sets of visual tokens with different granualarities in
information density. This enables flexibility in the number of visual
tokens used for any image during inference, enabling control over the
best tradeoff between cost and performance based on the image or video
content. For example, one can use all visual tokens for images with
dense details and use just a few tokens for simpler images. This
flexibility can be particularly significant when handling very long
visual sequences, such as videos. For instance, given a fixed budget of
2880 visual tokens, a user could represent a video of 2880 frames each
with one token or represent the same video by sampling 5 frames each
with 576 tokens.

Second, our approach can be used as a general framework to evaluate the
visual complexity of vision-language datasets or benchmarks, which level
of granularity is needed in order to perform the given task correctly.
Surprisingly, we find that most benchmarks, especially those mainly
crafted from natural scenes (such as
COCO) [goyal2017vqav2](http://arxiv.org/pdf/1612.00837v3), [li2023pope](http://arxiv.org/pdf/2402.15721v1), [liu2023mmbench](http://arxiv.org/pdf/2005.12661v2), can be
handled well with only $\sim9$ tokens per image. In contrast, dense
visual perception tasks such as document understanding or
OCR [singh2019textvqa](http://arxiv.org/pdf/1811.11903v1), [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177) require a
greater amount of tokens ($144-576$ tokens) per image to handle the task
well. The detailed findings are presented in
Sec. <a href="#sec:exp:Image Understanding" data-reference-type="ref"
data-reference="sec:exp:Image Understanding">[sec:exp:Image
Understanding]</a>.

Finally, our approach provides a foundation to tackle a critical task in
LMMs: *How to use the least amount of visual tokens while answering the
visual questions correctly?*. Based on the model’s predictions on the
test set, we find that compared to full visual tokens, the oracle can
use far fewer tokens while performing much better. For example, under
six common LMM benchmarks used in
LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), the oracle with the
trained  model can use as few as 8.9 visual tokens on average to achieve
performance that is 8% points better than LLaVA-NeXT which uses 576
tokens per image grid. This indicates that there is a large room for
improvement compared to the oracle upperbound, as we show in
Sec. <a href="#sec:exp:Image Understanding" data-reference-type="ref"
data-reference="sec:exp:Image Understanding">[sec:exp:Image
Understanding]</a>.

To enable further research on adaptive LMMs that learn diverse
information granularities, we publicly release our code and models.

# Related Work

**Large Multimodal Models.** Large Language Models (LLMs) like
ChatGPT [chatgpt](https://openai.com/blog/chatgpt/), GPT-4 [gpt4](http://arxiv.org/pdf/2311.15732v2), and
LLaMA [touvron2023LLaMA](touvron2023LLaMA) have demonstrated impressive
reasoning and generalization capabilities for text. The landscape of
LLMs has been significantly transformed by the recent introduction of
models that also incorporate visual information, such as
GPT-4V(ision)[GPT4V_System_Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf). Building upon
open-source LLMs [touvron2023LLaMA](touvron2023LLaMA), [Vicuna](https://vicuna.lmsys.org/), a plethora
of multimodal models have made significant strides, spearheaded by
models like LLaVA [liu2023llava](http://arxiv.org/pdf/2402.11690v1), [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1)
and MiniGPT-4 [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1), which combine
LLaMA’s [touvron2023LLaMA](touvron2023LLaMA) language capabilities with a
CLIP [radford2021learning](http://arxiv.org/pdf/2404.19696v1) based image encoder. Recently,
LMMs on more tasks and modalities have emerged, such as region level
LMMs [cai2024vipllava](http://arxiv.org/pdf/2312.00784v2), [zhang2023gpt4roi](http://arxiv.org/pdf/2309.12109v1), [chen2023shikra](http://arxiv.org/pdf/2306.15195v2), [peng2023kosmos](http://arxiv.org/pdf/2305.16103v1), [zhang2023llavagrounding](https://arxiv.org/pdf/2312.02949),
3D LMMs [3dllm](http://arxiv.org/pdf/2403.09631v1), and video
LMMs [lin2023video](http://arxiv.org/pdf/2311.10122v2), [zhang2023video](http://arxiv.org/pdf/2311.12919v2), [zhang2024llavanextvideo](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/).
However, existing LMMs typically represent the visual content with a
large and fixed number of tokens, which makes it challenging to scale to
very long visual sequences such as high-resolution images or long-form
videos. In this work, we propose to adaptively and efficiently represent
the visual content by learning multiple nested sets of visual tokens,
providing flexibility in the number of visual tokens used for any image
during inference.

**Matryoshka Representation Learning.** Matryoshka Representation
Learning (MRL) [kusupati2022matryoshka](http://arxiv.org/pdf/2405.17430v1) addresses the
need for flexible representations that can adapt to multiple downstream
tasks with varying computational resources. This approach, inspired by
the nested nature of Matryoshka dolls, encodes information at different
granularities within the same high-dimensional feature vector produced
by a neural network. The adaptability of MRL extends across different
modalities, including vision (ResNet [he2016deep](http://arxiv.org/pdf/1608.05895v1),
ViT [dosovitskiy2020vit](http://arxiv.org/pdf/2105.15075v2)), vision + language
(ALIGN [jia2021scaling](http://arxiv.org/pdf/2102.05918v2)), and language
(BERT [devlin2018bert](http://arxiv.org/pdf/1810.04805v2)), demonstrating its versatility
and efficiency. Recent work [li20242d](http://arxiv.org/pdf/1804.10975v1) extends MRL to
both the text embedding space and the Transformer layers space. Our
approach is inspired by MRL, but instead of learning multiple nested
embeddings for a high-dimensional feature vector, we learn *nested
visual tokens along the token length dimension* for the visual input. We
are the first to show that the idea of Matryosha learning can enable
explicit control over the visual granularity of the visual content that
an LMM processes.

**Token Reduction.** One of the main causes of inefficiency in recent
LMMs is their large number of prefix visual tokens that are fed into the
LLM [liu2023llava](http://arxiv.org/pdf/2402.11690v1), [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1). The quadratic
complexity in Transformers [vaswani2017attention](http://arxiv.org/pdf/2107.08000v1) is the
key issue in scaling the input sequence length for Transformers. Token
reduction serves as an effective technique to reduce computational costs
in Transformers. Sparse attention methods such as
Linformer [wang2020linformer](https://arxiv.org/pdf/2006.04768) and
ReFormer [kitaev2020reformer](https://openreview.net/forum?id=rkgNKkHtvB) conduct attention
operations within local windows rather than the full context, thereby
reducing the quadratic complexity of the vanilla attention operation.
Another notable method is Token Merging
(ToMe) [bolya2023tome](None), which utilizes full attention but
gradually reduces the number of tokens in each transformer block by
selecting the most representative tokens through bipartite matching for
the Vision Transformer (ViT). A recent
work [Haurum_2023_ICCVW](http://arxiv.org/pdf/2308.04657v1) further studies different
families of token reduction methods for ViT. However, prior approaches
produce a single length output per input image and do not offer multiple
granularities over the reduced token sequence. Our approach instead
learns a multi-granularity, coarse-to-fine token representation within
the same model architecture and weights, enabling it to easily be
adjusted to various computational or memory constraints.

# :  [sec:approach]

<figure id="fig:architecture">
<embed src="/papers/text_rich/arXiv-2405.17430v1_md/figures/approach.png" style="width:99.0%" />
<figcaption> <strong>Architecture of our proposed .</strong> The visual
features from CLIP are represented as several groups of coarse-to-fine
visual tokens. At test time, users can explicitly control the
granularity of the visual features. </figcaption>
</figure>

Our goal is to learn a Large Multimodal Model (LMM) that represents
visual content as nested sets of visual tokens capturing information
across multiple coarse-to-fine granularities, so that one can explicitly
control the visual granularity per test instance during inference. Here
we introduce how we learn a Matryoshka doll-like token sequence.

LMMs such as LLaVA [liu2023llava](http://arxiv.org/pdf/2402.11690v1) typically input a
sequence of visual tokens as prefix tokens to the LLM for
visual-linguistic reasoning. The visual encoder from pretrained
vision-language models, such as
CLIP [radford2021learning](http://arxiv.org/pdf/2404.19696v1) and
SigLIP [zhai2023sigmoid](http://arxiv.org/pdf/2303.15343v4), is typically utilized to
project the images into the set of visual tokens. In particular, the
CLIP visual encoder represents an input image $\mathbf{I}$ as an
$H\times W$ grid of visual tokens ${\mathbf{X}} _{H\times W}$, where
each $\mathbf{X}_i \in \mathbb{R}^{ C}$ is a $C$ dimensional feature
vector. Our goal is to learn nested sets of visual tokens
$[ {\mathbf{X}} _{S_1}, {\mathbf{X}} _{S_2}, \cdots, {\mathbf{X}} _{S_M}]$
which encode the visual information in a coarse-to-fine manner. To this
end, we enforce
${\mathbf{X}} _{S_i} \subset {\mathbf{X}} _{S_{i+1}}, \forall i$.
Importantly, we do not introduce any new learnable parameters to the
LMM. We instead optimize the CLIP visual encoder to learn the nested
visual representation directly, and train the ensuing LLM to adapt to
the learned nested set of tokens.

For ease of exposition, we consider
CLIP-ViT-L-336 [radford2021learning](http://arxiv.org/pdf/2404.19696v1) as the visual
encoder, where an image is encoded as $24\times24$ visual tokens (576
total). We create $M$ sets of tokens e.g.,
$|S_i| \in \{ 1, 9, 36, 144, 576 \}$, in which the visual tokens at the
coarser level are derived directly from those at the finer level.
Specifically, given the initial $24\times24$ visual tokens, We
sequentially apply $2\times2$ pooling with a stride 2, resulting in
$12\times12, 6\times6$, and $3\times3$ visual tokens. Finally, we apply
$3\times3$ pooling and get the most condensed single visual token. In
this way, the sets of Matryoshka visual tokens can gradually preserve
the spatial information in the original tokens while simultaneously
forming a coarse-to-fine nested representation.

We train by averaging the autoregressive next token prediction loss for
each scale $S_i$ for each image $\mathbf{I}_i$. Specifically, given a
Matryoshka visual representation ${\mathbf{X}} _{S_i}$ for scale $S_i$,
we maximize the likelihood of the predicted tokens matching the
ground-truth answer $\mathbf{X}_{\mathrm{a}}$:
$$P(\mathbf{X}_{\mathrm{a}} \mid {\mathbf{X}}_{S_i}, \mathbf{X}_{\text {q}})=\prod_{j=1}^L P_{\boldsymbol{\theta}}(x_j \mid {\mathbf{X}}_{S_i}, \mathbf{X}_{\text {q}}, \mathbf{X}_{\mathrm{a},<j}),$$
where $\boldsymbol{\theta}$ is the trainable parameters of the model,
which includes both the CLIP visual encoder and the ensuing LLM.
$\mathbf{X}_{\text {q}}$ denotes the question in text format, $L$
denotes the token length of the ground truth answer
$\mathbf{X}_{\mathrm{a}}$, and $\mathbf{X}_{\mathrm{a},<j}$ denotes all
the ground truth answer tokens before the current prediction token
$x_j$, where $j$ denotes the token index during text token generation.
We omit system messages for clarity, though they are part of the
conditioning.
Figure <a href="#fig:architecture" data-reference-type="ref"
data-reference="fig:architecture">1</a> shows our model architecture.

The final objective averages over all $M$ visual token scales:
$$\min_{\boldsymbol{\theta}} \frac{1}{M} \sum_{i=1}^M -\log P(\mathbf{X}_{\mathrm{a}} \mid {\mathbf{X}}_{S_i}, \mathbf{X}_{\text {q}}).$$

With this objective function, learns nested sets of visual tokens that
gradually include more details with increasing scale. For example, in
Figure <a href="#fig:detail-specturm-visualization" data-reference-type="ref"
data-reference="fig:detail-specturm-visualization">[fig:detail-specturm-visualization]</a>,
the smaller set of visual tokens describes the whole scene at a high
level while the larger set of visual tokens includes more details such
as the Pepsi cup. Our training objective affords our model to conduct
visual question answering under any granularity during inference. This
can be particularly useful in resource constrained applications; e.g.,
the visual granularity can be flexibly adjusted based on the anticipated
simplicity or complexity of the visual content while taking into account
compute and memory constraints.

# Experiments

In this section, we first detail the experiment settings in
Sec <a href="#sec:exp:setting" data-reference-type="ref"
data-reference="sec:exp:setting">1.1</a>. Then we show the performance
of on both image-level
benchmarks <a href="#sec:exp:Image Understanding" data-reference-type="ref"
data-reference="sec:exp:Image Understanding">1.2</a> and video-level
benchmarks <a href="#sec:exp:video understanding" data-reference-type="ref"
data-reference="sec:exp:video understanding">1.3</a>. Finally, we
analyze the behavior of and provide ablations in
Sec <a href="#sec:exp:analysis" data-reference-type="ref"
data-reference="sec:exp:analysis">1.4</a> and
 <a href="#sec:exp:ablation" data-reference-type="ref"
data-reference="sec:exp:ablation">1.5</a>.

## Experiment Settings [sec:exp:setting]

#### Model

We use LLaVA-1.5 [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) and
LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) as the base LMMs, both with
Vicuna 7B as the language model backbone. We finetune the whole model
using the exact visual instruction data from LLaVA-1.5 and LLaVA-NeXT,
respectively. The learning rate of LLM is $2\times10^{-5}$ and
$1\times10^{-5}$, respectively for LLaVA-1.5 and LLaVA-NeXT. The
learning rate for the visual encoder is $2\times10^{-5}$ for both
models. We train both models for 1 epoch using 8 NVIDIA H100 GPUs.

Instead of training the language model from scratch, we initialize the
language model weights from pre-trained LLaVA-1.5 and LLaVA-NeXT, which
we empirically works better. We name our LLaVA-1.5- and LLaVA-NeXT-.

#### Visual Token Scales

We design 5 scales for the visual tokens.
LLaVA-1.5 [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) and
LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) both leverage
CLIP-ViT-L-336 [radford2021learning](http://arxiv.org/pdf/2404.19696v1) as the visual
encoder, where an image is embedded into $24\times24$ visual tokens. We
gradually apply $2\times2$ pooling with stride 2, resulting in
$12\times12, 6\times6$, and $3\times3$ visual tokens, where we finally
apply a $3\times3$ pooling to get the final single visual token.
Therefore, the size of Matryoshka visual token sets are
$S \in \{ 1, 9, 36, 144, 576 \}$, following a nested manner. The
efficiency anlaysis on the system level is shown in
Appendix <a href="#sec: Efficiency Analysis" data-reference-type="ref"
data-reference="sec: Efficiency Analysis">[sec: Efficiency Analysis]</a>,
where boosts the speed of the LMM prefill process through diminished
floating-point operations (FLOPs) and lessens computational memory
requirements.

#### Evaluations.

For **image understanding**, we evaluate LLaVA-1.5 and LLaVA-NeXT on (a)
diverse multimodal benchmarks: POPE [li2023pope](http://arxiv.org/pdf/2402.15721v1),
GQA [hudson2019gqa](http://arxiv.org/pdf/2112.05136v1),
MMBench [liu2023mmbench](http://arxiv.org/pdf/2005.12661v2),
VizWiz [gurari2018vizwiz](http://arxiv.org/pdf/1802.08218v4),
SEEDBench [li2023seed](http://arxiv.org/pdf/2311.15759v1),
ScienceQA [lu2022learnscienceqa](http://arxiv.org/pdf/2209.09513v2),
MMMU [yue2023mmmu](http://arxiv.org/pdf/2311.16502v3), and (b) document
understanding/Optical character recognition (OCR) benchmarks:
DocVQA [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1),
ChartQA [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177),
AI2D [ai2d](http://arxiv.org/pdf/1603.07396v1) and
TextVQA [singh2019textvqa](http://arxiv.org/pdf/1811.11903v1).

For **video understanding**, we use both (a) open ended video question
answering benchmarks evaluated by GPT-3.5:
MSVD-QA [xu2017video](http://arxiv.org/pdf/1904.04357v1),
MSRVTT-QA [xu2017video](http://arxiv.org/pdf/1904.04357v1) and
ActivityNet-QA [yu2019activitynet](http://arxiv.org/pdf/1906.02467v1); and (b) multi-choice
video question answering benchmarks:
NExT-QA [xiao2021next](http://arxiv.org/pdf/2307.04412v1),
IntentQA [Li2023IntentQACV](http://arxiv.org/pdf/2002.08945v1), and
EgoSchema [mangalam2023egoschema](http://arxiv.org/pdf/2308.09126v1).

## Image Understanding [sec:exp:Image Understanding]

#### LLaVA-1.5-

We evaluate LLaVA-1.5- on the common multimodal understanding and
reasoning benchmarks. Results are shown in
Table <a href="#tab:image-level-llava-1.5-ff" data-reference-type="ref"
data-reference="tab:image-level-llava-1.5-ff">1</a>. LLaVA-1.5- with
full tokens maintains the performance of LLaVA-1.5 across diverse
benchmarks. More importantly, our approach shows strong performance even
with 1 or 9 tokens. Specifically, in MMBench, a comprehensive multimodal
understanding benchmark, LLaVA-1.5- with 9 tokens surpasses Qwen-VL-Chat
with 256 tokens, and achieves similar performance as Qwen-VL-Chat with
even 1 token. Compared with InstructBLIP [instructblip](http://arxiv.org/pdf/2311.00233v2),
LLaVA-1.5 with 9 tokens surpasses InstructBLIP-7B and InstructBLIP-13B
across all benchmarks. This demonstrates that our model has both
flexibility and strong empirical performance under diverse number of
visual tokens.

<div class="adjustbox" markdown="1">

max width=0.95

<div id="tab:image-level-llava-1.5-ff" markdown="1">

|  | \# Tokens | MMBench | GQA | POPE | VizWiz | SEEDBench |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|
|  [Qwen-VL](http://arxiv.org/pdf/2308.12966v3) | 256 | 38.2 | 59.3 | \- | 35.2 | 56.3 |
|  [Qwen-VL](http://arxiv.org/pdf/2308.12966v3) | 256 | 60.6 | 57.5 | \- | 38.9 | 58.2 |
|  [instructblip](http://arxiv.org/pdf/2311.00233v2) | 32 | 36.0 | 49.2 | \- | 34.5 | 53.4 |
|  [instructblip](http://arxiv.org/pdf/2311.00233v2) | 32 | \- | 49.5 | 78.9 | 33.4 | \- |
| LLaVA-1.5-7B [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) | 576 | 64.8 | **62.0** | 85.9 | 54.4 | 60.5 |
|  | 576 | 65.9 | 61.9 | **87.4** | **54.9** | **60.6** |
|  | 144 | **66.4** | 61.3 | 87.0 | 53.1 | 59.7 |
|  | 36 | 64.8 | 60.3 | 85.5 | 52.8 | 58.0 |
|  | 9 | 63.1 | 58.0 | 83.4 | 51.9 | 55.4 |
|  | 1 | 59.5 | 52.6 | 78.4 | 49.4 | 50.1 |

Comparison between LLaVA-1.5-$M^3$ across various benchmarks under video
understanding benchmarks. LLaVA-1.5- maintains the performance of
LLaVA-1.5 while outperforming Qwen-VL and InstructBLIP with fewer
tokens.

</div>

</div>

<span id="tab:image-level-llava-1.5-ff"
label="tab:image-level-llava-1.5-ff"></span>

#### LLaVA-NeXT-

We use the proposed to finetune LLaVA-NeXT, and compare LLaVA-NeXT- with
, which denotes the setting where the LLaVA-NeXT is trained under a
**S**pecific **S**cale of visual tokens also for 1 epoch. We also
include the oracle upperbound performance. Specifically, ‘Oracle’
denotes the case where the best tradeoff between visual tokens and
performance is picked for each test instance. Specifically, for each
test instance, we select the the scale with the fewest amount of tokens
but can answer the question correctly. Results are shown in
Table <a href="#tab:image-level-LLaVA-NeXT-ff" data-reference-type="ref"
data-reference="tab:image-level-LLaVA-NeXT-ff">2</a>. Our approach, , is
at least as good as , while performing better on tasks such as document
understanding (TextVQA and ChartQA) and common benchmarks such as
MMBench [liu2023mmbench](http://arxiv.org/pdf/2005.12661v2).

<div class="adjustbox" markdown="1">

max width=

<div id="tab:image-level-LLaVA-NeXT-ff" markdown="1">

| \# Tokens Per Grid | Approach | TextVQA | AI2D | ChartQA | DocVQA | MMBench | POPE | ScienceQA | MMMU |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|  |  | 64.53 | 64.83 | 59.28 | 75.40 | 66.58 | 87.02 | 72.29 | 34.3 |
|  | $M^3$ | 63.13 | 66.71 | 58.96 | 72.61 | 67.96 | 87.20 | 72.46 | 34.0 |
|  |  | 62.16 | 65.77 | 55.28 | 67.69 | 67.78 | 87.66 | 72.15 | 36.4 |
|  | $M^3$ | 62.61 | 68.07 | 57.04 | 66.48 | 69.50 | 87.67 | 72.32 | 36.1 |
|  |  | 58.15 | 65.90 | 45.40 | 56.89 | 67.01 | 86.75 | 71.87 | 36.2 |
|  | $M^3$ | 58.71 | 67.36 | 50.24 | 55.94 | 68.56 | 87.29 | 72.11 | 36.8 |
|  |  | 50.95 | 65.06 | 37.76 | 44.21 | 65.29 | 85.62 | 72.37 | 36.8 |
|  | $M^3$ | 51.97 | 66.77 | 42.00 | 43.52 | 67.35 | 86.17 | 71.85 | 35.2 |
|  |  | 38.39 | 63.76 | 28.96 | 33.11 | 61.43 | 82.83 | 72.32 | 35.3 |
|  | $M^3$ | 38.92 | 64.57 | 31.04 | 31.63 | 62.97 | 83.38 | 71.19 | 34.8 |
|  | <span style="color: blue">\# Tokens</span> | <span style="color: blue">31.39</span> | <span style="color: blue">11.54</span> | <span style="color: blue">41.78</span> | <span style="color: blue">64.09</span> | <span style="color: blue">8.90</span> | <span style="color: blue">6.08</span> | <span style="color: blue">7.43</span> | <span style="color: blue">22.85</span> |
|  | <span style="color: blue">Performance</span> | <span style="color: blue">70.51</span> | <span style="color: blue">76.36</span> | <span style="color: blue">70.76</span> | <span style="color: blue">81.73</span> | <span style="color: blue">74.35</span> | <span style="color: blue">94.29</span> | <span style="color: blue">76.07</span> | <span style="color: blue">50.44</span> |

Comparison of approaches with the baseline and $M^3$ across various
benchmarks under LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/). Here \#
Tokens denotes the number of visual tokens per image grid in LLaVA-NeXT.
denotes the baseline model trained with a **S**pecific **S**cale of
visual tokens. is at least as good as , while performing better on tasks
such as TextVQA, ChartQA, and MMBench.
<span style="color: blue">Oracle</span> denotes the case where the best
tradeoff between visual tokens and performance is picked.

</div>

</div>

<span id="tab:image-level-LLaVA-NeXT-ff"
label="tab:image-level-LLaVA-NeXT-ff"></span>

Our results also show that dataset level biases towards the visual token
scales do exist. For example, ScienceQA maintains consistent performance
across all visual token scales. AI2D and MMBench only encounter a small
performance drop for even as few as 9 to 1 tokens. On the other hand,
dense visual perception tasks such as TextVQA and DocVQA show a
significant performance with fewer tokens. This analysis shows that
could serve as a framework to analyze the granularity that a benchmark
needs.

Furthermore, there is a large gap between the model’s actual performance
under full tokens and the upper-bound oracle. This indicates that using
full tokens cannot always result in the optimal performance for all
samples; i.e., there is a large room of improvement towards the oracle
point.

## Video Understanding [sec:exp:video understanding]

Following IG-VLM [kim2024image](http://arxiv.org/pdf/2403.18406v1), we directly conduct
zero-shot inference on diverse video benchmarks using LLaVA-NeXT-.
Specifically, 6 frames are uniformly sampled over the entire video, then
arranged as a collage, which is fed into LLaVA-NeXT along with the
question to get the response. Results under LLaVA-NeXT- and recent video
LMMs are show in
Table <a href="#tab:LLaVA-NeXT-performance-video" data-reference-type="ref"
data-reference="tab:LLaVA-NeXT-performance-video">3</a>.

LLaVA-NeXT- with full visual tokens again shows comparable performance
with LLaVA-NeXT. More interestingly, results indicate that full visual
tokens usually *do not lead to the best performance* in video
understanding tasks. Specifically, on 4 out of 6 benchmarks, full visual
tokens show less desirable performance compared to 720 or 180 visual
tokens. We suspect that very long visual context could bring distraction
(e.g., too much focus on potentially irrelevant background) to the
model’s prediction, where a compact representation of the video focusing
on the more relevant information may be more advantageous.

Finally, for most video understanding tasks such as ActivityNet,
IntentQA and EgoSchema, with 9 tokens per image grid (45 tokens in
total), the accuracy difference compared to full tokens (2880 in total)
is less than 1%. This demonstrates that the video questions in these
benchmarks usually require very sparse visual information, as the source
of such video understanding benchmarks mostly comes from natural scenes,
which matches our observation in image understanding benchmarks.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:LLaVA-NeXT-performance-video" markdown="1">

|  | \# Tokens | MSVD | MSRVTT | ActivityNet | NextQA | IntentQA | EgoSchema |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|  [zhang2023VideoLLAMA](None) | \- | 51.6 | 29.6 | 12.4 | \- | \- | \- |
|  [Zhang2023LLaMAAdapterEF](http://arxiv.org/pdf/2207.10858v1) | \- | 54.9 | 43.8 | 34.2 | \- | \- | \- |
|  [Maaz2023VideoChatGPTTD](http://arxiv.org/pdf/2311.18445v1) | \- | 64.9 | 49.3 | 35.2 | \- | \- | \- |
|  [Lin2023VideoLLaVALU](http://arxiv.org/pdf/2311.10122v2) | 2048 | 70.7 | 59.2 | 45.3 | \- | \- | \- |
| InternVideo [Wang2022InternVideoGV](http://arxiv.org/pdf/2212.03191v2) | \- | \- | \- | \- | 59.1 | \- | 32.1 |
| LLaVA-NeXT-7B [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | 2880 | 78.8 | 63.7 | 54.3 | **63.1** | **60.3** | 35.8 |
|  | 2880 | 78.2 | **64.5** | 53.9 | **63.1** | 58.8 | 36.8 |
|  | 720 | **79.0** | **64.5** | **55.0** | 62.6 | 59.6 | 37.2 |
|  | 180 | 77.9 | 63.7 | **55.0** | 61.4 | 59.3 | 37.6 |
|  | 45 | 75.8 | 63.0 | 53.2 | 59.5 | 58.7 | **38.8** |
|  | 5 | 73.5 | 62.7 | 50.8 | 56.5 | 56.7 | 36.2 |

Overall accuracy of LLaVA-NeXT- and recent video LMMs on various video
understanding benchmarks. Here \# Tokens denotes the overall number of
visual tokens across all frames.

</div>

</div>

<span id="tab:LLaVA-NeXT-performance-video"
label="tab:LLaVA-NeXT-performance-video"></span>

## In-depth Analysis [sec:exp:analysis]

####  shows much stronger performance compared to heuristics based sampling at test time.

A simple way to reduce the number of visual tokens via a training-free
way is to conduct heuristic token merging or reduction. In
Table <a href="#tab:compare inference time sampling" data-reference-type="ref"
data-reference="tab:compare inference time sampling">4</a>, we compare
with three training-free approaches: average pooling, spatial sampling,
and sequential sampling. is much more resilient when the number of
tokens decreases, while the heuristic based sampling approaches show
dramatic performance drop. A visualization of the spatial and sequential
sampling is shown in
Figure <a href="#fig:vis sampling inference" data-reference-type="ref"
data-reference="fig:vis sampling inference">[fig:vis sampling
inference]</a>.

<span id="tab:compare inference time sampling"
label="tab:compare inference time sampling"></span>

<div class="adjustbox" markdown="1">

max width=0.9

<div id="tab:compare inference time sampling" markdown="1">

| \# Tokens |     | Average Pooling | Spatial Sampling | Sequential Sampling |
|:----------|:----|:---------------:|:----------------:|:-------------------:|
|           |     |      67.18      |      67.18       |        67.18        |
|           |     |      61.68      |      65.81       |        60.14        |
|           |     |      50.77      |      60.05       |        44.76        |
|           |     |      45.45      |      45.45       |        31.96        |
|           |     |      19.33      |      26.29       |        22.42        |

Comparison between , and heuristics based sampling baselines—average
pooling, spatial sampling, and sequential sampling—at inference time on
MMBench with the LLaVA-NeXT architecture.

</div>

</div>

####  serves as a good metric for image complexity.

We extract the response from LLaVA-NeXT- in the TextVQA benchmark, and
show the samples where using visual tokens across different scales can
answer the question correctly and incorrectly. Shown in
Figure <a href="#fig:textvqa-visualization" data-reference-type="ref"
data-reference="fig:textvqa-visualization">1</a>, the OCR performance
aligns with the complexity of the images, which indicates that can be
utilized as a metric towards sample level complexity.

<figure id="fig:textvqa-visualization">
<embed src="/papers/text_rich/arXiv-2405.17430v1_md/figures/correct-wrong-samples.png" style="width:99.0%" />
<figcaption> TextVQA test samples with correct and incorrect predictions
upon different scales. Answers vary with different number of visual
tokens. In addition, can serve as a framework to evaluate the complexity
of images. </figcaption>
</figure>

#### Large gap between oracle and actual performance.

As shown in
Table <a href="#tab:image-level-LLaVA-NeXT-ff" data-reference-type="ref"
data-reference="tab:image-level-LLaVA-NeXT-ff">2</a>, the oracle
upper-bound can use very few ($6\sim64$) tokens yet achieve at least 10%
better performance compared to full visual tokens. This suggests that a
visual token scale predictor, where the model learns to automatically
select the best visual token scale given the input images or both input
images and questions, has potential to achieve a better tradeoff. This
would be interesting future work.

#### Zero-shot generalization to longer visual sequences.

Here we extend the length of the visual tokens at inference time to
study the model’s zero-shot generalization behavior. Results under
LLaVA-NeXT are shown in
Table <a href="#tab:image-grid-performance-generation"
data-reference-type="ref"
data-reference="tab:image-grid-performance-generation">5</a>. Here
LLaVA-NeXT- is trained on $2\times2$ image grids but evaluated on
$3\times3$ grids. We set the number of visual tokens to be 144 in each
image during evaluation. The model obtains a significant improvement in
document understanding by 2.12, 1.80, and 4.11 on TextVQA, ChartQA, and
DocVQA, respectively, while maintaining the same performance on
benchmarks mainly composed of natural scene images. $3\times3$ image
grids with 144 tokens per grid own 1440 tokens, yet achieve similar
performance with the default LLaVA-NeXT $2\times2$ image grids with 2880
total tokens (576 tokens per grid). This indicates it is promising to
feed more subimages while making the number of visual tokens within each
subimage much smaller.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:image-grid-performance-generation" markdown="1">

| \# Grids | \# Tokens per grid | Overall \# Tokens | TextVQA | AI2D | ChartQA | DocVQA | MMBench | POPE | ScienceQA |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| $2\times2$ | 144 | 720 | 62.61 | 68.07 | 57.04 | 66.48 | 69.50 | 87.67 | 72.32 |
| $3\times3$ | 144 | 1440 | 64.73 | 67.75 | 58.84 | 70.59 | 69.50 | 87.67 | 72.22 |
| $2\times2$ | 576 | 2880 | 63.13 | 66.71 | 58.96 | 72.61 | 67.96 | 87.20 | 72.46 |

Performance comparison of different image grid configurations with
LLaVA-NeXT-.

</div>

</div>

<span id="tab:image-grid-performance-generation"
label="tab:image-grid-performance-generation"></span>

<div class="wrapfigure" markdown="1">

l0.5 <embed src="/papers/text_rich/arXiv-2405.17430v1_md/figures/sequential-spatial.png" />

</div>

## Ablation Studies [sec:exp:ablation]

We ablate the key designs in , including the sampling method of
Matryoshka visual tokens, and training strategy.

#### Matryoshka visual token sampling. 

Here we compare three different ways to select the visual tokens for ,
including average pooling, spatial sampling, and sequential sampling,
which is illustrated in
Figure <a href="#fig:vis sampling inference" data-reference-type="ref"
data-reference="fig:vis sampling inference">[fig:vis sampling
inference]</a>. Shown in
Table <a href="#tab:ablation_study_sampling" data-reference-type="ref"
data-reference="tab:ablation_study_sampling">6</a>, averaging pooling
shows better performance than the two alternatives across diverse
benchmarks. In general, sequential sampling performs the worst. We
hypothesize that this is due to the visual tokens having spatial
information, while sequential sampling does not naturally align with the
spatial distribution of the visual tokens.

<div class="adjustbox" markdown="1">

max width=

<div id="tab:ablation_study_sampling" markdown="1">

|  | TextVQA |  |  | MMBench |  |  | AI2D |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 2-10 Num of Vis Tokens | Avg Pooling | Sequential | Spatial | Avg Pooling | Sequential | Spatial | Avg Pooling | Sequential | Spatial |
| 576 | 63.13 | 59.37 | 60.45 | 67.96 | 64.60 | 64.43 | 66.71 | 65.61 | 64.96 |
| 144 | 62.61 | 55.80 | 58.33 | 69.50 | 64.18 | 64.52 | 68.07 | 64.90 | 64.96 |
| 36 | 58.71 | 52.79 | 52.39 | 68.56 | 63.92 | 64.69 | 67.36 | 64.51 | 64.02 |
| 9 | 51.97 | 44.05 | 44.19 | 67.35 | 63.14 | 62.11 | 66.77 | 63.70 | 63.92 |
| 1 | 38.92 | 28.03 | 29.91 | 62.97 | 59.36 | 57.47 | 64.57 | 63.21 | 63.08 |

Ablation on Matryoshka visual token sampling including average pooling,
sequential sampling, and spatial sampling.

</div>

</div>

<div class="adjustbox" markdown="1">

max width=

<div id="tab:ablation train llm" markdown="1">

| Num of Vis Tokens | TextVQA |         | MMBench |         |  AI2D  |         | DocVQA |         |
|:-----------------:|:-------:|:-------:|:-------:|:-------:|:------:|:-------:|:------:|:-------:|
|        2-9        | w/ LLM  | w/o LLM | w/ LLM  | w/o LLM | w/ LLM | w/o LLM | w/ LLM | w/o LLM |
|        576        |  63.13  |  61.16  |  67.96  |  63.66  | 66.71  |  63.92  | 72.61  |  69.15  |
|        144        |  62.61  |  57.79  |  69.50  |  65.21  | 68.07  |  63.73  | 66.48  |  59.77  |
|        36         |  58.71  |  49.75  |  68.56  |  63.92  | 67.36  |  62.89  | 55.94  |  44.08  |
|         9         |  51.97  |  36.15  |  67.35  |  61.08  | 66.77  |  62.05  | 43.52  |  28.36  |
|         1         |  38.92  |  19.72  |  62.97  |  51.80  | 64.57  |  60.59  | 31.63  |  17.37  |

Performance comparison of training LLaVA-NeXT- with and without training
the LLM across diverse benchmarks. We see a clear drop when freezing the
LLM.

</div>

</div>

<div class="adjustbox" markdown="1">

max width=0.95

<div id="tab:ablation smaple or average" markdown="1">

|           Technique            | TextVQA |       |       |       | AI2D  |       |       |       |
|:------------------------------:|:-------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  Init LLM weights from LLaVA   |         |       |       |       |       |       |       |       |
| Average losses over all scales |         |       |       |       |       |       |       |       |
|              576               |  60.36  | 62.25 | 61.01 | 63.13 | 62.40 | 65.06 | 65.84 | 66.71 |
|              144               |  59.61  | 61.02 | 59.80 | 62.61 | 63.67 | 65.61 | 65.77 | 68.07 |
|               36               |  54.86  | 55.91 | 55.32 | 58.71 | 63.67 | 65.32 | 66.68 | 67.36 |
|               9                |  46.84  | 47.04 | 48.80 | 51.97 | 63.02 | 64.83 | 65.38 | 66.77 |
|               1                |  33.78  | 33.68 | 36.05 | 38.92 | 61.53 | 63.21 | 63.37 | 64.57 |

Impact of (a) initializing the LLM weights from LLaVA, and (b) averaging
the loss from all scales vs randomly selecting a scale for each sample
during training.

</div>

</div>

<span id="tab:ablation smaple or average"
label="tab:ablation smaple or average"></span>

#### Training the entire LMM vs only training CLIP.

Since the nested behavior of Matryoshka visual tokens is learned within
the CLIP visual encoder, we next evaluate whether it is necessary to
also finetune the LLM. Shown in
Table <a href="#tab:ablation train llm" data-reference-type="ref"
data-reference="tab:ablation train llm">7</a>, training the whole LLM
achieves better performance. This demonstrates that by also training the
LLM, the model can better adapt to the patterns of the visual tokens
distributed in the Matryoshka manner.

As explained in Sec. <a href="#sec:approach" data-reference-type="ref"
data-reference="sec:approach">[sec:approach]</a>
and <a href="#sec:exp:setting" data-reference-type="ref"
data-reference="sec:exp:setting">1.1</a>, we (a) initialize the LLM
weights from LLaVA and (b) minimize the loss averaged upon all visual
token scales for each sample during training. An alternative choice is
to randomly sample a visual token scale. Shown in
Table <a href="#tab:ablation smaple or average" data-reference-type="ref"
data-reference="tab:ablation smaple or average">8</a>, initializing the
LLM weights from LLaVA and minimizing the losses over all scales shows
consistent performance boost compared to using the vanilla text-only
pre-trained LLM weights [Vicuna](https://vicuna.lmsys.org/) and randomly selecting a
visual token scale. Initializing the LLM weights from LLaVA makes the
training process of more stable. By learning all scales at once, the
model is forced to learn the nested behavior for each sample, which
leads to better performance.

# Conclusion and Future Work [sec:conclusion and limitation]

We introduced : , which learns to represent visual content as nested
sets of visual tokens, capturing information across multiple
coarse-to-fine granularities. LMMs equipped with afford explicit control
over the visual granularity per test instance during inference. We also
showed that can serve as an analysis framework to investigate the visual
granularity needed for existing datasets, where we discovered that a
large number of multimodal benchmarks only need as few as  9 visual
tokens to obtain accuracy similar to that of using all visual tokens,
especially for video understanding. Furthermore, we disclosed a large
performance-efficiency gap between the oracle upper-bound and the
model’s performance.

Our work can be naturally extended to other domains. For example, the
long context in a text-only LLM or vision tokens in dense vision tasks
can also be represented as nested sets of tokens in a Matryoshka manner.
One limitation of our current approach is that we are lacking an
effective visual token predictor that can bridge the gap between the
oracle and LMM’s actual performance at a specific scale. We believe this
would be an exciting next direction of research in this space.

# Acknowledgement [acknowledgement]

This work was supported in part by NSF CAREER IIS2150012, and Institute
of Information & communications Technology Planning & Evaluation(IITP)
grants funded by the Korea government(MSIT) (No. 2022-0-00871,
Development of AI Autonomy and Knowledge Enhancement for AI Agent
Collaboration) and (No. RS2022-00187238, Development of Large Korean
Language Model Technology for Efficient Pre-training), and Microsoft
Accelerate Foundation Models Research Program.

# Broader Impact [sec:boarder_impact]

The broader impact of , a framework with nested visual representations,
has potential benefits and risks associated with its deployment and
release. Our model is trained using the exact same architecture and data
of LLaVA-1.5 [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) and
LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/). All the concerns are same
as LLaVA. Specifically, as one example, LLaVA conducts instruction
tuning using GPT-4 and GPT-4V generated data. The bias from GPT-4 and
GPT-4V would still exist in LLaVA.

# Efficiency Analysis [sec: Efficiency Analysis]

To illuminate the computational benefits conferred by , we employ the
roofline-based LLM-Viewer analysis as detailed
in [yuan2024llm](http://arxiv.org/pdf/2402.16363v6). Our analysis is set within a
hypothetical context designed to emphasize the effects of on processing
efficiency in LMMs. We study the LLaVA-1.5 case where a $336 \times 336$
resolution image is processed using a CLIP-ViT image encoder, resulting
in 576 visual tokens. Accompanied by a text prompt with an assumed
number of 30 tokens, the nested visual tokens in substantially lowers
the visual token count. The consequences of this reduction are
substantial as outlined in
Table <a href="#tab:computation Cost" data-reference-type="ref"
data-reference="tab:computation Cost">1</a>, detailing the computational
costs involved in the LMM prefill process. Notably, not only boosts the
speed of the LMM prefill process through diminished floating-point
operations (FLOPs) but also lessens computational memory requirements.

It is crucial to highlight that the advantages of are not limited to
just efficiency improvements. The token reduction approach of can also
enhance other LMM acceleration methods, such as quantization and
factorization, as referenced in [yuan2023asvd](http://arxiv.org/pdf/2403.07378v4). This
complementary relationship accentuates the broad potential of to
contribute to a wider array of efficiency-boosting strategies.

<div id="tab:computation Cost" markdown="1">

| \# Tokens | FLOPs (TB) | Prefill Time (ms) | Total Memory (GB) | Storing Activation (GB) |
|:--:|:--:|:--:|:--:|:--:|
| 576 | 8.0 | 58.1 | 21.6 | 3.8 |
| 144 | 2.2 | 19.5 | 15.0 | 0.7 |
| 36 | 0.9 | 18.0 | 13.8 | 0.3 |
| 9 | 0.5 | 17.7 | 13.6 | 0.2 |
| 1 | 0.4 | 17.6 | 13.5 | 0.1 |

Computation Cost Analysis. The development device is Tesla V100 GPU, and
time estimated by the roofline model represents the theoretical
performance that the hardware can achieve.

</div>

# More Visualizations on Nested Visual Representation

Shown in Figure <a href="#fig:vis-more" data-reference-type="ref"
data-reference="fig:vis-more">1</a>, with more visual tokens, LMMs can
discover more details, such as furniture and human attributes. Besides,
LMMs can generate higher quality descriptions with more visual tokens,
as demonstrated by the OCR capability in
Figure <a href="#fig:vis-more" data-reference-type="ref"
data-reference="fig:vis-more">1</a> (b).

<figure id="fig:vis-more">
<embed src="/papers/text_rich/arXiv-2405.17430v1_md/figures/vis-more-examples.png" />
<figcaption> <strong>More visualization examples.</strong> With more
visual tokens, LMMs can discover more details, and generate higher
quality descriptions. The images are from MSCOCO <span class="citation"
data-cites="lin2014microsoft"></span> validation set. </figcaption>
</figure>