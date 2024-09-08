### Acknowledgments [acknowledgments]

We thank Xiaohan Zhang from Zhipu AI for managing the data annotation
team, and Zhao Xue, Aohan Zeng, Yifan An, Chenxu Guo from Zhipu AI and
Tsinghua for data management.

[^1]: Work was done when interned at Zhipu AI.

[^2]: Corresponding authors





# Introduction

<div class="figure*" markdown="1">

<img src="/papers/vision_rich/arXiv-2312.08914v2_md/figures/main_demo.png" style="width:99.0%" alt="image" />

</div>

Autonomous agents in the digital world are ideal assistants that many
modern people dream of. Picture this scenario: You type in a task
description, then relax and enjoy a cup of coffee while watching tasks
like booking tickets online, conducting web searches, managing files,
and creating PowerPoint presentations get completed automatically.

Recently, the emergence of agents based on large language models (LLMs)
is bringing us closer to this dream. For example,
AutoGPT [autogpt](https://github.com/Significant-Gravitas/AutoGPT), a 150,000-star open-source project,
leverages ChatGPT [openai2022chatgpt](https://openai.com/blog/chatgpt) to integrate
language understanding with pre-defined actions like Google searches and
local file operations. Researchers are also starting to develop
agent-oriented
LLMs [zeng2023agenttuning](http://arxiv.org/pdf/2310.12823v2), [chen2023fireact](http://arxiv.org/pdf/2402.01469v1). However, the
potential of purely language-based agents is quite limited in real-world
scenarios, as most applications interact with humans through Graphical
User Interfaces (GUIs), which are characterized by the following
perspectives:

-   Standard APIs for interaction are often lacking.

-   Important information including icons, images, diagrams, and spatial
    relations are difficult to directly convey in words.

-   Even in text-rendered GUIs like web pages, elements like canvas and
    iframe cannot be parsed to grasp their functionality via HTML.

Agents based on visual language models (VLMs) have the potential to
overcome these limitations. Instead of relying exclusively on textual
inputs such as HTML [nakano2021webgpt](http://arxiv.org/pdf/2310.03184v2) or OCR
results [rawles2023android](http://arxiv.org/pdf/1209.0687v1), VLM-based agents directly
perceive visual GUI signals. Since GUIs are designed for human users,
VLM-based agents can perform as effectively as humans, as long as the
VLMs match human-level vision understanding. In addition, VLMs are also
capable of skills such as extremely fast reading and programming that
are usually beyond the reach of most human users, extending the
potential of VLM-based agents. A few prior studies utilized visual
features merely as auxiliaries in specific scenarios. e.g.
WebShop [yao2022webshop](http://arxiv.org/pdf/2207.01206v4) which employs visual features
primarily for object recognition purposes. With the rapid development of
VLM, can we naturally achieve universality on GUIs by relying solely on
visual inputs?

In this work, we present CogAgent, a visual language foundation model
specializing in GUI understanding and planning while maintaining a
strong ability for general cross-modality tasks. By building upon
CogVLM [wang2023cogvlm](http://arxiv.org/pdf/2210.00066v1)—a recent open-source VLM,
CogAgent tackles the following challenges for building GUI agents:

-   **Training Data.** Most current VLMs are pre-trained on datasets
    like LAION [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1), consisting of
    natural images on the Web. However, we notice that the GUI images
    share a different distribution from natural images. We thus
    construct a large-scale annotated dataset about GUIs and OCR for
    continual pre-training.

-   **High-Resolution vs. Compute.** In GUIs, tiny icons and text are
    ubiquitous, and it is hard to recognize them in commonly-used
    $224\times224$ resolution. However, increasing the resolution of
    input images results in significantly long sequence length in
    language models. For example, a $1120\times 1120$ image corresponds
    to a sequence of $6400$ tokens if the patch size is $14$, demanding
    excessive training and inference compute. To address this, we design
    a cross-attention branch that allows for a trade-off between the
    resolution and the hidden size within a proper computation budget.
    Specifically, we propose to combine the original large
    ViT [dosovitskiy2020image](http://arxiv.org/pdf/2105.15075v2) (4.4B parameters) used in
    CogVLM [wang2023cogvlm](http://arxiv.org/pdf/2210.00066v1) and a new small
    *high-resolution cross-module* (with image encoder of 0.30B
    parameters) to jointly model visual features.

Our experiments show that:

-   CogAgent tops popular GUI understanding and decision-making
    benchmarks, including AITW [rawles2023android](http://arxiv.org/pdf/1209.0687v1) and
    Mind2Web [deng2023mind2web](http://arxiv.org/pdf/2306.06070v3). To the best of our
    knowledge, this is the first time that a generalist VLM can
    outperform LLM-based methods with extracted structured text.

-   Though CogAgent focuses on GUIs, it achieves state-of-the-art
    generalist performance on nine visual question-answering benchmarks
    including VQAv2 [antol2015vqa](http://arxiv.org/pdf/1309.1125v1),
    OK-VQA [marino2019ok](http://arxiv.org/pdf/1906.00067v2),
    TextVQA [singh2019towards](http://arxiv.org/pdf/1811.11903v1),
    ST-VQA [biten2019scene](http://arxiv.org/pdf/2304.01603v1),
    ChartQA [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1),
    infoVQA [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2),
    DocVQA [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1),
    MM-Vet [yu2023mm](http://arxiv.org/pdf/2402.15896v1), and
    POPE [li2023evaluating](http://arxiv.org/pdf/2402.15721v1).

-   The separated design of high- and low-resolution branches in
    CogAgent significantly lows the compute cost for consuming
    high-resolution images, e.g., the number of the floating-point
    operations (FLOPs) for CogAgent-18B with $1120 \times 1120$ inputs
    is less than half that of CogVLM-17B with its default
    $490\times 490$ inputs.

CogAgent is open-sourced at <https://github.com/THUDM/CogVLM>. It
represents an effort to promote the future research and application of
AI agents, facilitated by advanced VLMs.

# Method

In this section, we will first introduce the architecture of CogAgent,
especially the novel high-resolution cross-module, and then illustrate
the process of pre-training and alignment in detail.

## Architecture

The architecture of CogAgent is depicted in
<a href="#fig:arch" data-reference-type="ref+label"
data-reference="fig:arch">1</a>. We build our model based on a
pre-trained VLM (on the right side of the image), and propose to add a
cross-attention module to process high-resolution input (on the left
side of the image). As our base VLM, We select
CogVLM-17B [wang2023cogvlm](http://arxiv.org/pdf/2210.00066v1), an open-sourced and
state-of-the-art large vison-language model. Specifically, We employ
EVA2-CLIP-E [sun2023eva](http://arxiv.org/pdf/2303.15389v1) as the encoder for
low-resolution images (224$\times$`<!-- -->`{=html}224 pixels),
complemented by an MLP adapter that maps its output into the feature
space of the visual-language decoder. The decoder, a pre-trained
language model, is enhanced with a visual expert module introduced by
 [wang2023cogvlm](http://arxiv.org/pdf/2210.00066v1) to facilitate a deep fusion of visual
and language features. The decoder processes a combined input of the
low-resolution image feature sequence and text feature sequence, and
autoregressively outputs the target text.

Similar to most VLMs, the original CogVLM can only accommodate images of
relatively low resolution (224 or 490), which hardly meets the demands
of GUI where the screen resolution of computers or smartphones is
typically 720p ($1280\times720$ pixels) or higher. It is a common
problem among VLMs, e.g. LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1) and
PALI-X [chen2023pali](http://arxiv.org/pdf/2109.04653v1) are pre-trained at a low resolution
of $224\times224$ on the general domain. The primary reason is that
high-resolution image brings prohibitive time and memory overhead: VLMs
usually concatenate text and image feature sequence as input to the
decoder, thus the overhead of self-attention module is quadratic to the
number of visual tokens (patches), which is quadratic to the image’s
side length. There are some initial attempts to reduce costs for
high-resolution images. For instance,
Qwen-VL [bai2023qwen](http://arxiv.org/pdf/1412.3919v1) proposes a position-aware
vision-language adapter to compress image features, but only reduces
sequence length by four and has a maximum resolution of $448\times448$.
Kosmos-2.5 [lv2023kosmos](http://arxiv.org/pdf/2309.11419v1) adopts a Perceiver Resampler
module to reduce the length of the image sequence. However, the
resampled sequence is still long for self-attention in the large
visual-language decoder (2,048 tokens), and can only be applied to
restricted text recognition tasks.

Therefore, we propose a novel *high-resolution cross-module* as a potent
complement to the existing structure for enhancing understanding at high
resolutions, which not only maintains efficiency confronting
high-resolution images, but also offers flexible adaptability to a
variety of visual-language model architectures.

<figure id="fig:arch">
<embed src="/papers/vision_rich/arXiv-2312.08914v2_md/figures/architecture.png" />
<figcaption>Model architecture of CogAgent. We adopt CogVLM as the
original VLM. </figcaption>
</figure>

## High-Resolution Cross-Module

The structural design of *high-resolution cross-module* is mainly based
on the following observations:

1.  At a modest resolution such as $224\times224$, images can depict
    most objects and layouts effectively, yet the resolution falls short
    in rendering text with clarity. Hence, our new high-resolution
    module should emphasize text-related features, which are vital for
    understanding GUIs.

2.  While pre-trained VLMs in general domain often need large hidden
    sizes (e.g. 4,096 in PALI-X and CogVLM, 5,120 in LLaVA), VLMs
    tailored for text-centered tasks like document OCR require smaller
    hidden sizes to achieve satisfying performance (e.g. 1,536 in
    Kosmos-2.5 and Pix2Struct [lee2023pix2struct](http://arxiv.org/pdf/2210.03347v2)). This
    suggests that text-related features can be effectively captured
    using smaller hidden sizes.

As shown in <a href="#fig:arch" data-reference-type="ref+label"
data-reference="fig:arch">1</a>, the high-resolution cross-module acts
as a new branch for higher-resolution input, which accepts images of
size $1120\times1120$ pixels in our implementation. Different from the
original low-resolution input branch, the high-resolution cross-module
adopts a much smaller pre-trained vision encoder (visual encoder of
EVA2-CLIP-L [sun2023eva](http://arxiv.org/pdf/2303.15389v1) in our implementation, 0.30B
parameters), and uses cross-attention of a small hidden size to fuse
high-resolution image features with every layer of VLLM decoder, thus
reducing the computational cost. To be concrete, for an input image, it
is resized to $1120\times1120$ and $224\times224$ and fed into the
high-resolution cross-module and the low-resolution branch respectively,
then encoded into image feature sequences $X_{\text{hi}}$ and
$X_{\text{lo}}$ with two distinct-sized image encoders in parallel. The
visual language decoder retains its original computations, while the
only change is to integrate a cross-attention between $X_{\text{hi}}$
and hidden states in every decoder layer.

Formally, suppose that the input hidden states of the i-th attention
layer in the decoder are
$X_{\text{in}_i} \in \mathbb{R}^{B\times (L_{I_{\text{lo}}}+L_T) \times D_{\text{dec}}}$,
and the output hidden states of cross-module’s image encoder are
$X_{\text{hi}} \in \mathbb{R}^{B\times (L_{I_{\text{hi}}}) \times D_{\text{hi}}}$,
where B is the batch size, $L_{I_{\text{lo}}}$, $L_{I_{\text{hi}}}$ and
$L_T$ are the lengths of the low-resolution image, high-resolution image
and text sequences, $D_{\text{dec}}$ and $D_{\text{hi}}$ is the hidden
size of the decoder and high-resolution encoder’s output respectively.
Each layer’s attention procedure can be formulated as $$\begin{aligned}
    X_{i}' &= \text{MSA}(\text{layernorm}(X_{\text{in}_i})) + X_{\text{in}_i}, \label{msa} \\
    X_{\text{out}_i} &= \text{MCA}(\text{layernorm}(X_{i}'), X_{\text{hi}}) + X_{i}', \label{eq:mca}
\end{aligned}$$ where MSA and MCA represent multi-head self-attention
with visual expert and multi-head cross-attention, while $X_{i}'$ and
$X_{\text{out}_i}$ represent their respective output features with the
residual connection. To implement cross-attention between them, we add
learnable transformation matrices
$W_{K_{\text{cross}}}^i, W_{V_{\text{cross}}}^i \in \mathbb{R}^{D_{\text{hi}} \times D_{\text{cross}}}$
to get $K_{\text{cross}}^i=X_{\text{hi}} W_{K_{\text{cross}}}^i$,
$V_{\text{cross}}^i=X_{\text{hi}} W_{V_{\text{cross}}}^i \in \mathbb{R}^{L_{I_{\text{hi}}} \times D_{\text{cross}}}$,
and
$W_{Q_{\text{cross}}}^i \in \mathbb{R}^{D_{\text{dec}} \times D_{\text{cross}}}$
to get
$Q_{\text{cross}}^i=X_i' W_{Q_{\text{cross}}}^i \in \mathbb{R}^{(L_{I_{\text{lo}}}+L_T) \times D_{\text{cross}}}$
in every decoder layer. With the residual connection in
Eq. <a href="#eq:mca" data-reference-type="ref"
data-reference="eq:mca">[eq:mca]</a>, the cross-attention with
high-resolution images can be perceived as a complement to the features
of low-resolution images, thereby effectively utilizing the previous
pre-trained model in low resolution.

**Computational complexity.** Let the number of attention head be
$H_{\text{cross}}$ and $H_{\text{dec}}$ in cross-attention and
self-attention, and the dimension of each head be
$d_{\text{cross}} = D_{\text{cross}}/{H_{\text{cross}}}$ and
$d_{\text{dec}} = D_{\text{dec}}/{H_{\text{dec}}}$. If using our
high-resolution cross-module, the computational complexity of attention
is $$\begin{split}
\text{T}_{\text{improved}} = \mathbf{O}\bigl( &(L_{I_{\text{lo}}} + L_T) L_{I_{\text{hi}}} H_{\text{cross}} d_{\text{cross}} \\
&+ (L_{I_{\text{lo}}} + L_T)^2 H_{\text{dec}} d_{\text{dec}} \bigr).
\end{split}$$ Note that $d_{\text{cross}}$ and $H_{\text{cross}}$ can be
flexibly adjusted according to computational budget and model
performance. If not utilizing the high-resolution cross-module and
directly substituting low-resolution images with high-resolution ones,
the computational complexity would be $$\begin{aligned}
\text{T}_{\text{original}} = \mathbf{O}\bigl((L_{I_{\text{hi}}} + L_T)^2 H_{\text{dec}} d_{\text{dec}} \bigr).
\end{aligned}$$

In our implementation, $d_{\text{cross}}=32$, $H_{\text{cross}}=32$, and
we inherits $d_{\text{dec}}=128$, $H_{\text{dec}}=32$ from CogVLM-17B.
Both high- and low-resolution encoders patchify images with
$14\times14$-pixel patches, thus $L_{I_{\text{hi}}}=6400$,
$L_{I_{\text{lo}}}=256$. Our method leads to at least
$\frac{L_{I_{\text{hi}}}+L_{T}}{L_{I_{\text{lo}}}+L_{T}} = \frac{6400+L_{T}}{256+L_{T}} \times$
acceleration which is a stringent lower bound (refer to Appendix for
detailed derivation), and reduces memory overhead at the same time.

## Pre-training

To enhance the model’s ability to comprehend high-resolution images and
adapt it for GUI application scenarios, we focus our pre-training
efforts on the following aspects: the capability to recognize texts of
various sizes, orientations, and fonts in high-resolution images, the
grounding ability of text and objects in the image, and a specialized
understanding capability for GUI imagery such as web page. We divide our
pre-train data into three parts based on the aforementioned aspects,
with samples in the Appendix. All the pre-training data are derived from
publicly available datasets. The construction methods are detailed
below.

**Text recognition.** Our data includes (1) Synthetic renderings with
text from language pre-training dataset (80M). This is similar to the
Synthetic Document Generator in  [kim2022ocr](http://arxiv.org/pdf/2305.09520v1), with text
of varying font, size, color and orientation, and diverse image
background from LAION-2B [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1). (2)
Optical Character Recognition (OCR) of natural images (18M). We collect
natural images from COYO [kakaobrain2022coyo-700m](https://github.com/kakaobrain/coyo-dataset) and
LAION-2B [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1) and employ
Paddle-OCR [du2020pp](http://arxiv.org/pdf/2109.03144v2) to extract the texts and their
bounding boxes, and filter out images with no text boxes. Paddle-OCR may
introduce some errors, which can be ameliorated through integration with
other pre-training datasets and subsequent fine-tuning processes. (3)
Academic documents (9M). We follow
Nougat [blecher2023nougat](http://arxiv.org/pdf/2308.13418v1) to construct image-text pairs
including text, formula and tables from the source code (LaTeX) release
on arXiv. For (1)(3), we apply the same data augmentation as Nougat
which includes erosion, gaussian noise, gaussian blur, image
compression, and elastic transform, etc. For (2), we additionally
employed more aggressive rotation and flipping data augmentation
techniques, thereby enhancing the model’s robustness in recognizing
text.

**Visual grounding.** It is imperative for GUI agents to possess the
capability to accurately comprehend and locate diverse elements within
images. Therefore, we incorporated a range of grounding data into
pre-training. We follow CogVLM [wang2023cogvlm](http://arxiv.org/pdf/2210.00066v1) to use a
constructed visual grounding dataset of 40M images with image-caption
pairs sampled from LAION-115M [li2023blip](http://arxiv.org/pdf/2301.12597v3), which
associate entities in the caption with bounding boxes to indicate their
positions. The format of the bounding box is $[[x_0, y_0, x_1, y_1]]$,
where $(x_0, y_0)$ and $(x_1, y_1)$ represent the coordinates of
upper-left and lower-right corners which are normalized to $[000, 999]$.
If multiple objects are indicated by a single noun phrase, their boxes
are separated by semicolons in double square brackets. We have also
collected grounding data on web page elements, which will be introduced
in the next part.

**GUI imagery.** Our approach innovatively addresses the scarcity and
limited relevance of GUI images in datasets like LAION and COYO, which
predominantly feature natural images. GUI images, with their distinct
elements such as input fields, hyperlinks, icons, and unique layout
characteristics, require specialized handling. To boost the model’s
capability in interpreting GUI imagery, we have conceptualized two
pioneering GUI grounding tasks: (1) GUI Referring Expression Generation
(REG) – where the model is tasked with generating HTML code for DOM
(Document Object Model) elements based on a specified area in a
screenshot, and (2) GUI Referring Expression Comprehension (REC) – which
involves creating bounding boxes for given DOM elements. To facilitate
robust training in GUI grounding, we have constructed the CCS400K
(Common Crawl Screenshot 400K) dataset. This extensive dataset is formed
by extracting URLs from the latest Common Crawl data, followed by
capturing 400,000 web page screenshots. Alongside these screenshots, we
compile all visible DOM elements and their corresponding rendered boxes
using Playwright[^1], supplementing the dataset with 140 million REC and
REG question-answer pairs. This rich dataset ensures comprehensive
training and understanding of GUI elements. To mitigate the risk of
overfitting, we employ a diverse range of screen resolutions for
rendering, selected randomly from a list of commonly used resolutions
across various devices. Additionally, to prevent the HTML code from
becoming overly extensive and unwieldy, we perform necessary data
cleaning by omitting redundant attributes in the DOM elements, following
the method outlined in  [lee2023pix2struct](http://arxiv.org/pdf/2210.03347v2).

We also incorporate publicly available text-image datasets including
LAION-2B and COYO-700M (after removing the broken URLs, NSFW images, and
images with noisy captions and political bias) during pre-training.

We pre-train our CogAgent model for a total of 60,000 iterations with a
batch size of 4,608 and a learning rate of 2e-5. We freeze all
parameters except the newly added high-resolution cross-module for the
first 20,000 steps, resulting in a total number of 646M (3.5%) trainable
parameters, then additionally unfreeze the visual expert in CogVLM for
the next 40,000 steps. We warm up with curriculum learning by first
training on easier text recognition (synthetic renderings and OCR on
natural images) and image captioning, then sequentially incorporating
harder text recognition (academic document), grounding data and web page
data, as we observed that it leads to faster convergence and more stable
training in our preliminary experiments.

## Multi-task Fine-tuning and Alignment

To enhance our model’s performance for diverse tasks and ensure it
aligns with free-form human instructions in the GUI setting, we further
fine-tune our model on a broad range of tasks. We manually collected
over two thousand screenshots from mobile phones and computers, each
annotated with screen elements, potential tasks, and methods of
operation in the question-answering format by human annotators (details
illustrated in the Appendix). We also utilize
Mind2Web [deng2023mind2web](http://arxiv.org/pdf/2306.06070v3) and
AITW [rawles2023android](http://arxiv.org/pdf/1209.0687v1), datasets focusing on web and
Android behaviors which comprise tasks, sequences of actions and
corresponding screenshots, and convert them into a natural language
question-and-answer format using GPT-4. Besides, we incorporate multiple
publicly available visual question-answering (VQA) datasets encompassing
a variety of tasks into our alignment dataset. We unfreeze all model
parameters during this stage and train for 10k iterations with a batch
size of 1024 and a learning rate of 2e-5.

[^1]: <https://playwright.dev>

# Experiments

To evaluate the foundational capabilities and GUI-related performance of
our model, we conduct extensive experiments on a broad range of
datasets. First, we conduct evaluations on eight VQA benchmarks, as well
as MM-Vet [yu2023mm](http://arxiv.org/pdf/2402.15896v1) and
POPE [li2023evaluating](http://arxiv.org/pdf/2402.15721v1), which validate our model’s
enhanced ability in visual understanding, especially on those that are
reliant on text recognition. Then we evaluate our model on Mind2Web and
AITW datasets, as the representative of two major GUI scenarios —
computers and smartphones.

## Foundational Visual Understanding

We first extensively evaluate CogAgent’s foundational visual
understanding capability across eight VQA benchmarks, covering a wide
range of visual scenes. The benchmarks can be divided into two
categories: general VQA, including VQAv2 [antol2015vqa](http://arxiv.org/pdf/1309.1125v1)
and OK-VQA [marino2019ok](http://arxiv.org/pdf/1906.00067v2), and text-rich VQA, including
TextVQA [singh2019towards](http://arxiv.org/pdf/1811.11903v1),
OCR-VQA [mishra2019ocr](http://arxiv.org/pdf/2010.02582v1),
ST-VQA [biten2019scene](http://arxiv.org/pdf/2304.01603v1),
DocVQA [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1),
InfoVQA [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2) and
ChartQA [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1). The latter category
emphasizes the understanding of visually-situated text, including
documents, charts, photographs containing text, etc. Contrary to models
individually fine-tuned for optimal performance on each downstream task,
our model is fine-tuned collectively on all datasets simultaneously,
yielding a single generalist model which is then evaluated across all
datasets. The goal of generalist evaluation is to better mirror
real-world situations of visual agents where typically a single model is
used, and to demonstrate the model’s versatility and robustness across
tasks.

The results are presented in
<a href="#table:vqa" data-reference-type="ref+label"
data-reference="table:vqa">[table:vqa]</a>. For general VQA, CogAgent
achieves state-of-the-art generalist results on both datasets. For
text-rich VQA, CogAgent achieves state-of-the-art results on 5 out of 6
benchmarks, significantly surpassing generalist competitors
(TextVQA+8.0, ChartQA+2.1, InfoVQA+2.3, DocVQA+16.2), even outperforming
the task-specific state-of-the-art models on TextVQA(+4.7), STVQA(+0.6)
and DocVQA(+1.6). Notably, compared to the generalist results of CogVLM
which CogAgent is initially based on, CogAgent demonstrates certain
improvements on both general and Text-rich VQA tasks, suggesting the
efficacy of our proposed model architecture and training methods.

Furthermore, we conducted zero-shot tests of our model on the
challenging MM-Vet [yu2023mm](http://arxiv.org/pdf/2402.15896v1) and
POPE [li2023evaluating](http://arxiv.org/pdf/2402.15721v1) datasets, both of which are
instrumental in gauging the multi-modal capabilities and the
generalization performance in complex tasks including conversation
question-answering, detailed descriptions, complex reasoning tasks.
MM-Vet is designed with six core tasks to assess multi-modal models’
proficiency in handling intricate assignments, and POPE-adversarial
models on their susceptibility to hallucinations. Our experimental
results, as detailed in
Table <a href="#tab:LLaVA_results" data-reference-type="ref"
data-reference="tab:LLaVA_results">[tab:LLaVA_results]</a>, showcase
that our model significantly outperforms other existing models in both
datasets. Notably, on the MM-Vet dataset, our model achieved a
remarkable score of 52.8, surpassing the closest competitor, LLaVA-1.5,
by a substantial margin (+16.5). On the POPE-adversarial evaluation, our
model attained a score of 85.9, demonstrating superior handling of
hallucinations compared to other models.

These results indicate CogAgent’s robust performance in foundational
visual understanding, especially in the interpretation of images with
embedded text. With these core competencies, the model can be feasibly
applied to various visual agent tasks across different GUI environments.

## GUI Agent: Computer Interface

We evaluate CogAgent on Mind2Web, a dataset for web agents that includes
over 2,000 open-ended tasks collected from 137 real-world websites
across 31 domains. Each entry in the dataset comprises a high-level task
description, a sequence of actions, and webpage snapshots in a variety
of formats, including HTML and screenshots. Given task description,
current webpage snapshot and previous actions as inputs, agents are
expected to predict the subsequent action. We follow the setting of
[deng2023mind2web](http://arxiv.org/pdf/2306.06070v3) in our experiments, and report step
success rate (step SR) metric. Further details are attached in the
Appendix.

Several language models were evaluated on this benchmark. For instance,
AgentTuning [zeng2023agenttuning](http://arxiv.org/pdf/2310.12823v2) and
MindAct [deng2023mind2web](http://arxiv.org/pdf/2306.06070v3) evaluated Llama2-70B and
Flan-T5-XL in a fine-tuned setting, and GPT-3.5 and GPT-4 in a
in-context learning setting. However, limited by the input modality of
language models, these models could only use heavily cleansed HTML as
the representation of screen inputs. To the best of our knowledge, no
visually-based web agents have been experimented with on this benchmark.

We fine-tune our model on the train set and evaluate on three
out-of-domain subsets, i.e. cross-website, cross-domain, and cross-task.
We additionally fine-tune LLaMA2-7B and LLaMA2-70B as the baseline of
fine-tuned LLMs, and adopt the same HTML cleansing process as
[deng2023mind2web](http://arxiv.org/pdf/2306.06070v3) to construct HTML input. The results
are presented in <a href="#tab:mind2web" data-reference-type="ref+label"
data-reference="tab:mind2web">[tab:mind2web]</a>. Compared to other
methods, our approach achieved significant performance improvements
across all three subsets, surpassing LLaMA2-70B, which is nearly
4$\times$ the scale of CogAgent, by 11.6%, 4.7%, and 6.6%, respectively.
This reflects not only the capability of our model but also the
advantages of employing a visual agent in computer GUI scenarios.

## GUI Agent: Smartphone Interface

To evaluate our model on diverse smartphone interfaces and tasks, we
utilize Android in the Wild (AITW)
dataset [rawles2023android](http://arxiv.org/pdf/1209.0687v1) , a large-scale dataset for
Android device agents. It comprises 715k operation episodes, covering
30k distinct task instructions, four Android versions, and eight device
types featuring varying screen resolutions. Each episode in the dataset
consists of a goal described in natural language, followed by a sequence
of actions and corresponding screenshots. The training target is to
predict the next action based on the given goal, historical actions, and
the screenshot. AITW considers a wide range of action types, including
tapping, swiping, typing, going home, going back, entering, etc. For
each action, models are required to predict the exact action type; for
tap, swipe and type, models are further required to predict the
position, direction, and content to be typed, respectively.

We conduct comparisons with two kinds of baselines: language models
using the textual description of UI elements provided by the original
dataset (text OCR and icon) as the representations of screen inputs[^1],
and visual-language models using images as the screen inputs. We
simultaneously fine-tuned on all the subsets, yielding a unified model
which is then evaluated on all test sets. As the GoogleApps subset is
10-100 times larger than other subsets, we downsample it to 10% to avoid
data imbalance.

Results are shown in <a href="#tab:aitw" data-reference-type="ref+label"
data-reference="tab:aitw">[tab:aitw]</a>. CogAgent achieves
state-of-the-art performance compared to all previous methods. In
comparison to language-based methods, our model surpasses both baselines
by a large margin. In comparison to the visual-language baseline,
Auto-UI, our model achieves +2.61 improvements in the overall
performance. In instances of inaccuracies, we randomly sample hundreds
of cases, and upon reassessment, more than 40% are determined to be
correct (refer to the appendix for details). This diversity arises from
the multiple valid pathways inherent in mobile interactions, resulting
in a range of responses.

[^1]: Some Android applications may have View Hierarchy which is more
    friendly to language-based agents, but most of them tend to be poor
    quality or missing altogether. Therefore, as a large-scale,
    general-purpose dataset, AITW retained the results of OCR detection
    and icon detection as textual representations of screenshots.

# Ablation Study [subsec:ablation]

To thoroughly comprehend the impact of various components in the
methodology, we conduct ablation studies on two aspects, model
architecture and training data. The evaluation is conducted on diverse
datasets, including multiple VQA datasets (STVQA, OCRVQA, DocVQA) and a
web agent dataset (Mind2Web). For VQA datasets, we fine-tune the model
on four datasets together for 3,000 iters with a batch size of 1,280,
and report the generalist score; for Mind2Web, models are fine-tuned for
2,400 iters with a batch size of 128 and use top-10 setting. Training
iterations are fewer than those in the main experiment, aiming to
control variables within the constraints of a limited budget.

## Model Architecture

To ascertain the efficacy of the high-resolution cross-module, we
compare it with directly increasing the resolution using the original
model architecture of CogVLM, and ablate on two perspectives:
computational efficiency and model performance.

To measure computational overhead, we use floating point operations
(FLOPs) as the metric, and conduct experiments on multiple resolutions
including 224, 490, 756, and 1120. From
<a href="#fig:flop" data-reference-type="ref+label"
data-reference="fig:flop">1</a> we can see that, as the image resolution
increases, models that use a high-resolution cross-module experience
only a modest rise in computational overhead, demonstrating an almost
linear relationship with the number of image patches. In contrast, using
the original model structure, i.e. CogVLM, leads to a significant
increase in the number of FLOPs at higher resolutions. Its FLOPs can
even be more than 10 times higher compared to employing a cross-module
at a resolution of 1120, which is the resolution utilized by CogAgent.

<figure id="fig:flop">
<embed src="/papers/vision_rich/arXiv-2312.08914v2_md/figures/flops.png" style="width:90.0%" />
<figcaption>Comparison of FLOPs during forward propagation for different
model architectures and resolutions.</figcaption>
</figure>

We further compare the model performance in
<a href="#tab:ablation-architecture" data-reference-type="ref+label"
data-reference="tab:ablation-architecture">[tab:ablation-architecture]</a>,
which indicates that models with high-resolution cross-module at the
resolution of 756 require only 1/2 of the computational resources used
by the original structure at the resolution of 490, while delivering
significantly better performance. Additionally, the high-resolution
cross-module allows for further increasing models’ acceptable resolution
within a limited computational budget, thereby yielding additional
performance improvements.

## Pre-train Data

We further conduct an ablation study on pre-training data, which is an
integral part of training visual agents. Building upon the image-caption
data commonly used in visual-language training, we sequentially add OCR
data (denoted as Cap+OCR), as well as GUI and grounding data (denoted as
All). The results in
<a href="#tab:ablation-data" data-reference-type="ref+label"
data-reference="tab:ablation-data">[tab:ablation-data]</a> indicate that
each part of data broadly contributes to enhanced performance. Notably,
web and grounding data have a significant impact on the Mind2Web
dataset, underscoring the importance of constructing domain-specific
pre-train data in the training of GUI agents.

# Conclusion

We introduce CogAgent, a VLM-based GUI agent with enhanced pre-train
data construction and efficient architecture for high-resolution input.
CogAgent achieves state-of-the-art performance on a wide range of VQA
and GUI benchmarks, and will be open-sourced.

CogAgent is an initial exploration of VLM-based GUI agent, and still has
some shortcomings, e.g. imprecise output coordinates and incapability of
processing multiple images, necessitating further research.