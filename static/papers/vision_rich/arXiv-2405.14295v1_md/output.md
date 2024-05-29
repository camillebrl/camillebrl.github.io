# Introduction [intro]

Recently, research on Large Vision-Language
Models [GPT4](https://arxiv.org/pdf/arXiv preprint arXiv:2303.08774), [minigpt4](http://arxiv.org/pdf/2402.17510v1), [Flamingo](http://arxiv.org/pdf/2205.07065v1) has been an attractive
direction. These models not only easily handle some conventional vision
tasks (*e.g.*, Image Caption [coco_text](http://arxiv.org/pdf/1707.08831v1),
OCR [OCRVQA](http://arxiv.org/pdf/2010.02582v1)), but also demonstrate powerful reasoning
capabilities like humans.

<figure id="fig:intro">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/intro.png" style="width:100.0%" />
<figcaption>(a) Multiple vision vocabularies are catalyzed using
synthetic cross-vocabulary data to handle interleaved pages. (b) Fox
achieves fine-grained document-level understanding by focusing anywhere,
such as region-level OCR/translation and in-page figure caption. (c) Fox
impressively supports the entire 8-page input and can focus on multiple
cross-page RoIs in a single-turn conversation.</figcaption>
</figure>

The LVLMs mostly give responses by leveraging large language
models [OPT](http://arxiv.org/pdf/2405.04515v2), [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [T5](http://arxiv.org/pdf/1910.10683v4) to follow language instructions
while referring to the vision vocabulary to understand the input image.
Some researchers attempt to adopt LVLMs to advance the understanding of
large-resolution (*e.g.*, 833$\times$`<!-- -->`{=html}1132) document
pages. For example, UReader [ye2023ureader](http://arxiv.org/pdf/2311.13165v1) crops the
input image into smaller patches to align with a CLIP-style vision
vocabulary of input size 224$\times$`<!-- -->`{=html}224. Later,
TextMonkey [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1) divides the input image
into 448$\times$`<!-- -->`{=html}448 patches and uses Openclip’s
ViT-bigG [openclip_ilharco_2024_10469088](openclip_ilharco_2024_10469088) along with a
resampling strategy to retain useful image tokens.
LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) adopts CLIP-ViT-L-336px to
perform visual perception and splits the input image into smaller
patches to encode independently.
InternVL-V1.5 [chen2024far_intervl1.5](http://arxiv.org/pdf/2404.16821v2) proposes a
stronger vision vocabulary InternViT-6B with the input size of
448$\times$`<!-- -->`{=html}448. Similarly, to capture more details of
the input image, InternVL-V1.5 [chen2024far_intervl1.5](http://arxiv.org/pdf/2404.16821v2)
dynamically divides the input image into 1 to 12 tiles. Different from
the methods above, without cropping patches,
Vary [wei2023vary](http://arxiv.org/pdf/2312.06109v1) writes an extra
SAM-style [SAM](http://arxiv.org/pdf/2305.01275v1) vision vocabulary specific to document
and chart data, running in parallel with the CLIP branch. Vary can
directly encode 1024$\times$`<!-- -->`{=html}1024 page into 256 image
tokens with a high compression ratio.

The patch-based
models [ye2023ureader](http://arxiv.org/pdf/2311.13165v1), [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1), [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [chen2024far_intervl1.5](http://arxiv.org/pdf/2404.16821v2)
mostly employ CLIP-style vision vocabulary with small resolution, so a
large-scale document needs to be decomposed into many patches/tiles. A
patch/tile is independently encoded to 256 image tokens, and
InternVL-V1.5 [chen2024far_intervl1.5](http://arxiv.org/pdf/2404.16821v2) even produces
3,328 image tokens during training. However, numerous image tokens are
difficult to extend to multi-page documents for contextual
understanding. More importantly, there may still be dense characters on
these cropped patches, but CLIP-style vision vocabulary compresses
limited sparse information of small input images via global contrastive
learning, preventing these models from losslessly recovering the content
of the original document (, full-page OCR). Although
Vary [wei2023vary](http://arxiv.org/pdf/2312.06109v1) enjoys a high compression ratio and
avoids cropping patches by directly encoding the document page, the lack
of full collaboration across multiple vision vocabularies limits the
performance. For example, given an input document page,
Vary [wei2023vary](http://arxiv.org/pdf/2312.06109v1) tends to only activate the SAM-style
ViT branch due to the specific-vocabulary visual bias. In addition, the
above models are sensitive to document format (*e.g.*, multi-column) and
do not support fine-grained user interaction on specific regions on
documents.

Another key point for the document understanding is how to carry out
fine-grained interaction, such as OCR/summarizing/captioning a region of
interest. Actually, LVLMs with human-like referential dialogue
capability for natural scenes have been investigated, such as
Shikra [chen2023shikra](http://arxiv.org/pdf/2306.15195v2) and
ChatSpot [zhao2023chatspot](http://arxiv.org/pdf/2307.09474v1). They introduce referring
spatial coordinates to refer to the special region of the input natural
image, lifting the user experience and leading to more precise
conversations. But these models can not handle the document images due
to vision vocabulary CLIP-ViT [CLIP_radford2021learning](http://arxiv.org/pdf/2404.19696v1)
which is specific to natural scenes and has low input resolution.
Besides, CLIP-style pre-training method based on
Laion-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) (image-phrase pairs) only
weakly write sparse visual knowledge, leading to a gap in understanding
the dense document. Thus, we may ask: *Can we devise an effective and
efficient pipeline for LVLMs to achieve the fine-grained multi-page
document understanding?*

In this paper, we propose Fox, an effective pipeline, hybrid data, and
tunning strategy, giving a pleasing answer to the above question. The
proposed Fox efficiently catalyzes the LVLM’s attention to anywhere on
single/multi-page documents in a user-friendly manner. Our solution has
three highlights: 1) *Focusing anywhere:* We introduce a novel task that
boosts document understanding by focusing on the region of interest via
fine-grained position-aware prompts, *i.e.*, click points, dragged
bounding boxes, and drawn color boxes. Notably, the dense full-page OCR
sub-task can be further optimized by being redefined as foreground
focus. 2) *Full reaction across multiple vision vocabularies:* To fully
interpret hybrid visual knowledge on interleaved document pages, we
synthesize cross-vocabulary vision data to activate multiple visual
vocabularies simultaneously to break down the specific-vocabulary bias
of visual content, catalyzing multiple vision vocabularies to a full
reaction. 3) *Supporting multi-column format and multiple pages:* With
the position-aware prompts, the pipeline of focusing anywhere can yield
robust performance regardless of document format. Moreover, benefiting
from the high compression ratio (one 1024$\times$`<!-- -->`{=html}1024
page to 256 image tokes), we demonstrate the Fox can be efficiently
tuned to achieve the above fine-grained capabilities on multi-page
documents without modifying parameters of vision vocabulary.

As a result of the focusing catalytic process, the proposed Fox can not
only give specific-vocabulary responses (*e.g.*, page foreground OCR,
region/line-level OCR/translation) but also gain the noticeable ability
to utilize the cross-vocabulary visual knowledge (*e.g.*, color-guided
OCR, in-document figure caption). Furthermore, for more impressive
multi-page document features, Fox can give the OCR results of $region_1$
on $page_1$ and $region_n$ on $page_n$ by only one question. Note that
tasks like this with reference to cross-page content are of great
research significance. We encourage researchers to rethink the framework
design for LVLM-based document understanding and not be limited to
conventional single-page sparse QA tasks. Our contributions can be
summarized as follows:

-   We introduce a series of novel tasks to boost document understanding
    by enabling LVLMs to focus on document-level regions of interest. We
    propose an effective and efficient solution named Fox to focus
    anywhere on single/multi-page documents.

-   To catalyze multiple vision vocabularies for figure-text interleaved
    documents, we provide methods for generating hybrid data containing
    cross-vocabulary visual elements.

-   Fox is robust to documents of various formats due to the flexible
    position-aware prompts. Without training vision vocabulary, our Fox
    can be easily tuned to multi-page documents and gain cross-page
    parsing capabilities.

-   We build a fine-grained document benchmark, including 9 sub-tasks,
    such as dense page OCR, region-level OCR/translation/summary,
    color-guided OCR, multi-page OCR/VQA. Experimental results show that
    our Fox outperforms other LVLMs by a large margin.

# Related Works

## Visual Document Understanding

Visual document understanding is widely investigated in the research
field of computer vision. Optical Character Recognition (OCR) is a basic
task, which plays a key role in document
digitalization [smith2007overview](http://arxiv.org/pdf/1003.5893v1), [moysset2017full](http://arxiv.org/pdf/1704.08628v1). The
layout analysis task [zhong2019publaynet](http://arxiv.org/pdf/1908.07836v1) aims to detect
various document elements and facilitate to understanding of spatial
relationships between them. We believe that OCR is a good task to test
whether LVLMs can compress documents losslessly. Besides, for
translation and
summary [vaswani2017attention](http://arxiv.org/pdf/2107.08000v1), [dong2019unified](http://arxiv.org/pdf/2212.06742v2) tasks, the
proposed Fox can directly give answers for document images via the
multimodal framework.

## Large Language Models

In recent times, the success of LLMs has ignited the fields of natural
language processing (NLP) and artificial general intelligence (AGI). The
LLMs are built with the popular transformer framework which is explored
by earlier NLP research, *e.g.*, BERT [Bert](http://arxiv.org/pdf/1810.04805v2),
GPT-2 [GPT-2](http://arxiv.org/pdf/2203.12926v1), T5 [T5](http://arxiv.org/pdf/1910.10683v4), and so on.
Afterward, it is discovered that when the model parameters are expanded
to a certain size, the language model will be greatly boosted due to the
so-called "emergent ability" [wei2022emergent](http://arxiv.org/pdf/2403.15796v2). Further,
the "GPT time" comes with amazing dialogue robots optimized by
Reinforcement Learning with Human
Feedback [RLHF_christiano2017deep](http://arxiv.org/pdf/2007.12904v2), *e.g.*,
InstructGPT [InstructGPT](http://arxiv.org/pdf/2302.05206v1) and
ChatGPT [ChatGPT](https://openai.com/blog/chatgpt/). Following that,
OPT [OPT](http://arxiv.org/pdf/2405.04515v2), LLaMA [llama](http://arxiv.org/pdf/2402.08075v1), and
GLM [GLM](http://arxiv.org/pdf/2004.13270v1) are accessible to the community to pursue the
performance like the GPT family. Based on the open-source LLMs, for more
specific requirements, some fine-tuned models have merged, such as
Alphaca [alpaca](https://github.com/tatsu-lab/stanford_alpaca) and Vicuna [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/),
which also play critical roles in later Large Vision-Language Models.

## Large Vision-Language Models

For vision-centric tasks, Large Vision-Language Models
(LVLMs) [llava](http://arxiv.org/pdf/2402.11690v1), [Flamingo](http://arxiv.org/pdf/2205.07065v1), [lu2024deepseek](http://arxiv.org/pdf/2402.17510v1) have been
developed by connecting the vision networks to LLMs.
CLIP-ViT [CLIP_radford2021learning](http://arxiv.org/pdf/2404.19696v1) is a mature
pre-trained vision vocabulary widely used to inject visual modality into
language models. To ensure that LLMs can understand the visual context,
LLaVA [llava](http://arxiv.org/pdf/2402.11690v1) places the linear layers to project visual
tokens into text space. Later, beyond natural scenes, LVLMs for
large-resolution documents have emerged.
UReader [ye2023ureader](http://arxiv.org/pdf/2311.13165v1) is developed based on the LVLM
mPLUG-Owl [ye2023mplug](http://arxiv.org/pdf/2405.00390v2).
UReader [ye2023ureader](http://arxiv.org/pdf/2311.13165v1) devise a shape-adaptive approach
to crop input images into 224$\times$`<!-- -->`{=html}224 patches and
feed them into CLIP vision encoder. Following
Qwen-VL [Qwen-VL](http://arxiv.org/pdf/2308.12966v3),
TextMonkey [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1) uses a more powerful
vision vocabulary Openclip’s
ViT-bigG [openclip_ilharco_2024_10469088](openclip_ilharco_2024_10469088) with
448$\times$`<!-- -->`{=html}448 input size to endoce each cropped patch.
With the strategy of cropping patches,
LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) adopts CLIP-ViT-L-336px to
perform visual perception. Similarly, to capture more details,
InternVL-V1.5 [chen2024far_intervl1.5](http://arxiv.org/pdf/2404.16821v2) dynamically
divides the input image into 1 to 12 tiles of
448$\times$`<!-- -->`{=html}448. In contrast, without cropping patches,
Vary [wei2023vary](http://arxiv.org/pdf/2312.06109v1) writes an extra
SAM-style [SAM](http://arxiv.org/pdf/2305.01275v1) 1024$\times$`<!-- -->`{=html}1024 vision
vocabulary specific to document and chart data, running in parallel with
the CLIP branch.

Compared to the above models, we believe that document understanding
should move towards more fine-grained (*e.g.,* region-level
OCR/translation) and multi-page tasks. Imagine how cool it would be if
we could use the LVLM like a reading pen! In this paper, we introduce
Fox which can achieve fine-grained features by focusing anywhere on
multi-page documents.

# Methods

<figure id="fig:architecture">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/archi.png" style="width:100.0%" />
<figcaption>Overall framework of the proposed Fox. All image tokens of
multiple pages are unified into a sequence to achieve multi-page
understanding. We devise position-aware prompts (point, color, and box)
to make the model focus anywhere on single/multi-page documents. We
catalyze multiple vision vocabularies into a full reaction of hybrid
visual knowledge for interleaved pages.</figcaption>
</figure>

In this section, we will elaborate on the details of the proposed Fox.
First, we introduce the flexible pipeline which supports
single/multi-page document understanding. Second, we provide the
strategy to produce the data containing hybrid visual elements to
activate multiple vocabularies concurrently. Last, we unify multi-task
data with position-aware prompts to conduct the focusing process.

## Framework for Focusing Anywhere

As illustrated in
Figure <a href="#fig:architecture" data-reference-type="ref"
data-reference="fig:architecture">2</a>, the architecture of the
proposed Fox is built with two vision vocabularies, a large language
model, and embedding linear layers. Specifically, to better handle
figure-text interleaved large-resolution documents, there are two vision
vocabularies, including natural content-aware
CLIP-ViT [CLIP_radford2021learning](http://arxiv.org/pdf/2404.19696v1) and artificial
content-aware Vary-tiny [wei2023vary](http://arxiv.org/pdf/2312.06109v1). The overall
framework is neat and provides more user-friendly fine-grained
interactions, which can focus on the entire page and more specific
regions of interest (RoI). Impressively, the proposed Fox also supports
users to select RoIs on multiple pages at the same time, enabling
cross-page contextual understanding.

Given a set of input document pages $I=\{p_i\}_{i=1}^N$, users can
further indicate regions of interest $r_i$ on each page by clicking a
point, dragging boxes, or drawing color boxes, and then give some
language instructions $L^{instruct}$ about the questioning RoIs. $N$ is
the number of input pages. The spatial coordinates or color information
of $\{r_i\}_{i=1}^N$ is transformed into position-aware prompts and
combined with $L^{instruct}$ to produce complete referential
instructions. Meanwhile, two vision vocabularies will produce 256 image
tokens $v^C_i \in \mathbb{R}^{256\times1024}$ and
$v^S_i \in \mathbb{R}^{256\times1024}$ for each page $p_i$. These image
tokens $\{v^C_i\}_{i=1}^N$ and $\{v^S_i\}_{i=1}^N$ are sent into linear
layers $W^C$ and $W^S$ to align with linguistic space. Then, the final
image tokens $v_i \in \mathbb{R}^{256\times2048}$ can be obtained by
concatenation. Note that $v_i$ is compressed into cross-vocabulary
content, including dense characters and figures. Finally, with the
projected image tokens and referential instructions, LLM will generate
the response sequence $Q$ in an auto-regressive manner. The above
process can be formulated as follows:
$$\{v_i\}_{i=1}^N = \left[ W^C \circ \{v^C_i\}_{i=1}^N || W^S \circ \{v^S_i\}_{i=1}^N\right]$$
$$Q = \mathcal{LLM} \left( \{v_i\}_{i=1}^N, \left(L^{instruct}, \Psi \left(\{r_i\}_{i=1}^N \right)\right) \right)$$
where $\left[\cdot || \cdot \right]$ is the concatenation operation.
$\Psi(\cdot)$ denotes the normalization for spatial coordinates. Note
that multi-page ($N$ pages) image tokens $\{v_i\}_{i=1}^N$ are unified
into a sequence for cross-page contextual understanding. With the causal
masked sequence modeling, the training objective can be expressed as:
$$\mathcal{L}_t=-E_{(Q, V)\sim D}\operatorname{log} P_{\theta} \left( q_m | q_{<m}, \{v_i\}_{i=1}^N \right)$$
where $m$ denotes the current index of the target token and $D$ is the
multi-page multi-grained dataset.

## Activating Multiple Vision Vocabularies by Cross-Vocabulary Hybrid Data [sec:hybrid data]

We hope to catalyze new capabilities more efficiently while freezing
pre-trained multiple vision vocabularies. Note that each vocabulary is
written with visual knowledge of its specific-domain data.
CLIP [CLIP_radford2021learning](http://arxiv.org/pdf/2404.19696v1) using
400M [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) image-text pairs is geared
toward perceiving natural images, and
Vary-tiny [wei2023vary](http://arxiv.org/pdf/2312.06109v1) using about 10M artificial data
is good at page-level documents. There is a specific-vocabulary
perceptual bias during querying vision vocabs due to the simple stacking
of samples in two domains. Hence, we synthesize hybrid data containing
the cross-vocabulary elements for the full reaction of multiple vision
vocabularies. Instead of focusing too much on a specific vocabulary,
richer cross-visual knowledge will be activated by breaking down the
barriers of visual content.

#### Preparing PDF data. [para:prepare pdf]

We download enough open-source files in PDF format from e-books,
CC-MAIN, and arXiv. Then, we parse the PDFs with some useful Python
packages. We save each document page as a PNG-style image. Meanwhile, we
locate each potential paragraph/line and record the corresponding
bounding box and text content within it.

#### Figure-text interleaved data.

We choose the BLIP558k [llava](http://arxiv.org/pdf/2402.11690v1),
Laion-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1), and
RegionChat [zhao2023chatspot](http://arxiv.org/pdf/2307.09474v1) datasets that contain
descriptions for natural images, and we randomly sample the same number
of document pages from the prepared PDF data. Clearly, before we render
a natural image of $W^n\times H^n$ pixels into a document page with the
resolution of $W^d \times H^d$ pixels, we should make the size of the
natural image smaller than that of the document page. The scaling
process can be formulated as follows:

<div class="small" markdown="1">

$$\label{eq1}
\left\{ \begin{aligned}
    W_{new}^n & = \operatorname{randint}\left(\left[\alpha \cdot W^d \right], \left[\beta \cdot W^d\right] \right), H_{new}^n = \left[W_{new}^n/W^n \cdot H^n \right], & \text{if} \  W^n/H^n > W^d/H^d \\
    H_{new}^n & = \operatorname{randint}\left(\left[\eta \cdot H^d \right], \left[\gamma \cdot H^d\right] \right), W_{new}^n = \left[H_{new}^n/H^n \cdot W^n \right], & \text{if} \ W^n/H^n \leq W^d/H^d\\
\end{aligned} \right.$$

</div>

where $W_{new}^n$/$H_{new}^n$ denote the desired width/height of the
scaled natural image. $\left[\cdot\right]$ means the integral function.
$\alpha$, $\beta$, $\eta$, and $\gamma$ are the hyperparameters that
control the scaling ratio, and they are set to 0.3, 0.9, 0.4, and 0.9,
respectively. Then, we randomly pick a suitable location
$(x^n_1, y^n_1, x^n_2, y^n_2)$ on the page to place the scaled natural
image. What’s more, to make the interleaved data reasonable and delete
the occluded text on this page, we calculate the intersection of union
(IoU) between $(x^n_1, y^n_1, x^n_2, y^n_2)$ and the vanilla text boxes
$\left\{ (x^d_{i,1}, y^d_{i,1}, x^d_{i,2}, y^d_{i,2}) \right\}_{i=1}^{N_d}$,
and fill the text boxes overlapped by the natural image with the white
color. $N_d$ is the number of text boxes on this document page. So, we
can obtain cross-vocabulary image-text pairs for in-document figure
caption. The text for each interleaved page includes the filtered
optical characters and the description of the pasted natural image.

#### Color-text hybrid data.

CLIP is written with the knowledge for recognizing colors, while the
Vary-tiny is not. We produce color-text hybrid data to further activate
multiple vocabularies, which is the key to enabling Fox to support the
conversations for users’ color-guided RoI. We randomly select three text
boxes and paint them directly on the document page in red, blue, and
green colors. The proposed Fox is expected to directly give the OCR
results in the area with the questioning color.

## Triggering Focusing Process via Fine-grained Instruction-following Tasks

We devise fine-grained instructions based on several position-aware text
prompts, such as points, boxes, and colors, to catalyze Fox to focus any
fine-grained region on single/multi-page documents.

#### Fine-grained document understanding.

We define several novel sub-tasks to drive the model to focus on
fine-grained regions for flexible document-level understanding: 1)
Foreground OCR. We redefine the page OCR task as the foreground focus to
further boost the dense perception. The instruction can be “*Give the
OCR results of the box $(x^f_{i,1}, y^f_{i,1}, x^f_{i,2}, y^f_{i,2})$*”.
The foreground box can be obtained by some simple operations. 2)
Region-level OCR. Based on the obtained text boxes, we transform the
content of one page into multiple region-level OCRs via multi-turn
conversations. An example can be “*Give the OCR results of the box
$(x^d_{i,1}, y^d_{i,1}, x^d_{i,2}, y^d_{i,2})$*”. 3) Line-level OCR. We
pick a point near the left side of each line as the position prompt.
Then, we construct the line-level multi-turn conversations and an
example can be like “*OCR the line $(x^d_{j}, y^d_{j})$*”. 4)
Color-guided OCR. Using the color-text hybrid data in
Section <a href="#sec:hybrid data" data-reference-type="ref"
data-reference="sec:hybrid data">3.2</a>, we define the corresponding
cross-vocabulary task by some color-guided questions, such as “*OCR red
box*” and “*OCR blue box*”. 5) Region-level translation and summary. We
filter and retain the boxes with text lengths over 400 on each page.
Then, we employ GPT-3.5 [ChatGPT](https://openai.com/blog/chatgpt/) to generate the
translation and summary for each long in-box text as the corresponding
annotations. The instruction can be “*Translate/Summarize the content of
the box $(x^d_{i,1}, y^d_{i,1}, x^d_{i,2}, y^d_{i,2})$*”. 6) Document
layout: We convert the 330K high-quality annotations of
PubLayNet [zhong2019publaynet](http://arxiv.org/pdf/1908.07836v1) to the unified
conversation format. Further, we sample 1M extra PDF pages and use
PaddleOCRv2 [paddleocrv2_du2021pp](http://arxiv.org/pdf/2109.03144v2) tools to generate
pseudo layout annotations.

#### In-document figure understanding.

Based on the synthetic interleaved data, we organize the
cross-vocabulary image-text pairs into two sub-tasks: 1) In-document
figure caption. As a result of the added position-aware prompts, an
example language instruction is as follows: “*Give a brief description
for the region $(x^n_1, y^n_1, x^n_2, y^n_2)$ of the image*”. The box
denotes the boundary of the figure. 2) In-document in-figure chat. The
RegionChat [zhao2023chatspot](http://arxiv.org/pdf/2307.09474v1) dataset is built for
referential dialogue on natural images. After rendering it on PDF pages,
with spatial coordinates of the referring region, we can ask the
proposed Fox the following question: “*What can you see in this region?
$(x^n_1, y^n_1, x^n_2, y^n_2)$*”. At a more fine-grained level, the RoI
can be the box within the figure on the document page.

#### Extension for multi-page documents.

The proposed Fox can be easily tuned to focus on multiple regions of
multi-page documents using simple instructions. As a forerunner, we
define two basic yet interesting multi-page sub-tasks and give
position-aware instruction examples. 1) Multi-page region-level OCR:
“*OCR boxes on multiple pages. Page 1: $(x^1_1, y^1_1, x^1_2, y^1_2)$,
Page 2: $(x^2_1, y^2_1, x^2_2, y^2_2)$, $\dots$ Page N:
$(x^N_1, y^N_1, x^N_2, y^N_2)$*”. 2) Cross-page VQA: “*Which page’s box
contains more characters? Page 1: $(x^1_1, y^1_1, x^1_2, y^1_2)$, Page
2: $(x^2_1, y^2_1, x^2_2, y^2_2)$, $\dots$ Page N:
$(x^N_1, y^N_1, x^N_2, y^N_2)$*”.

It is worth noting that all the above methods are independent of
document format. The PDF data with any format or layout, such as
single-column, double-column, interleaved, *etc.*, can be parsed to
extract positional prompts and formulated into the corresponding
conversations. With the fine-grained position-aware instructions, the
vision query pipeline enjoys high human-AI interactivity and is robust
to different formats (multi-column) and multi-page documents.

## Catalyzing Fox by Multi-page and Multi-grained Data Engine

The data engine is a key part of the proposed Fox. To ensure the
performance on multiple tasks, We carefully control the quantity and
ratio of training data, and more details are reported in
Table <a href="#tab:data" data-reference-type="ref"
data-reference="tab:data">[tab:data]</a>.

#### Pre-training data.

In the pre-training stage, we formulate a large number of multimodal
task-driven data. Specifically, for hybrid images of in-document caption
and chat sub-tasks, we render the BLIP558K [llava](http://arxiv.org/pdf/2402.11690v1) data,
1M natural images sampled in
Laion-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) and
RegionChat100K [zhao2023chatspot](http://arxiv.org/pdf/2307.09474v1) data into an equal
amount of document pages sampled in prepared PDF data. For fine-grained
optical character understanding, we formulate 6 types of 4.6M document
image-text pairs, containing box/line/color position-aware prompts and
OCR/translation/summary interactive task forms. Further, we generate
800K multi-page data, including multi-page multi-region OCR and
cross-page QA. In addition, to maintain the general conversational
capabilities of our model, we sample 1M natural data from
Laion-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) and NLP dialogue data
from Alpaca [alpaca](https://github.com/tatsu-lab/stanford_alpaca), Baize [xu2023baize](http://arxiv.org/pdf/2404.02406v1)
and ShareGPT.

#### SFT data.

In the supervised fine-tuning stage, To make the conversation experience
more comfortable, we sample 10K image-text pairs for each data type of
the above pre-training data, and adopt GPT3.5 [ChatGPT](https://openai.com/blog/chatgpt/)
to rewrite prompts ten times more diversified. Besides,
LLaVA80K [llava](http://arxiv.org/pdf/2402.11690v1) is also added to further tune our model
to generate pleasing answers.

<div class="table*" markdown="1">

<div class="center" markdown="1">

| **Task** | **Region-level Dataset** | **Sample** | **Task** | **Page-level Dataset** | **Sample** |
|:--:|:--:|:--:|:--:|:--:|:--:|
| In-document Cap. | PDF$\times$BLIP558K [llava](http://arxiv.org/pdf/2402.11690v1) | 558K | Layout | PubLayNet [zhong2019publaynet](http://arxiv.org/pdf/1908.07836v1) | 33K |
|  | PDF$\times$ Laion-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) | 1M |  | Annots. by PaddleOCRv2 [paddleocrv2_du2021pp](http://arxiv.org/pdf/2109.03144v2) | 1M |
| In-document Chat | PDF$\times$ RegionChat [zhao2023chatspot](http://arxiv.org/pdf/2307.09474v1) | 22K | Cap. | Laion-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) | 500K |
| Doc. Understanding | foreground OCR | 1M | NLP | Alpaca [alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 52K |
|  | region-level OCR | 1M |  | Baize [xu2023baize](http://arxiv.org/pdf/2404.02406v1) | 112K |
|  | line-level OCR | 600K |  | ShareGPT | 125K |
|  | color-guided OCR | 1M | ——— | ———————————— | ————- |
|  | region-level translation | 500K | PDF | Page OCR | 1M |
|  | region-level summary | 500K |  | Page Markdown | 1M |
| Multi-page Doc. | multi-region OCR | 400K | \- | \- | \- |
|  | cross-page VQA | 400K | \- | \- | \- |

</div>

<span id="tab:data" label="tab:data"></span>

</div>

#### Input and Conversation Format

For each input image, we resize it with a fixed resolution
1024$\times$`<!-- -->`{=html}1024 before feeding it into the
SAM-style [SAM](http://arxiv.org/pdf/2305.01275v1) ViT branch and we perform a resize
operation to obtain a new image of 224$\times$`<!-- -->`{=html}224 as
the input of the CLIP vision network. We choose
Qwen-1.8B [qwen](http://arxiv.org/pdf/2309.16609v1) with rich linguistic vocabulary as our
language model. Following the
LLaVA-MPT [llava](http://arxiv.org/pdf/2402.11690v1), [team2023introducing](http://arxiv.org/pdf/2311.16429v1) dialogue style, the
input conversation format can be summarized as follows:
\<\|im_start\|\>user: \<img\>"\<image\>"\</img\> "*human question
\[position-aware prompts\]*"\<\|im_end\|\> \<\|im_start\|\>assistant:
"*AI responses*" \<\|im_end\|\>.

# Experiments

## Implementation Details

During the multi-task pre-training and SFT phase, the multiple vision
vocabularies (CLIP and SAM-style ViT) are frozen and only the parameters
of the embedding linear layers and language model are optimized. We
train our model using the optimizer AdamW [AdamW](http://arxiv.org/pdf/2311.11446v2) and a
cosine annealing scheduler [loshchilov2016sgdr](http://arxiv.org/pdf/1608.03983v5). The
learning rate is set to 1e-4 in pretraining and then to 2e-5 in SFT. In
both stages, we use 48 A800 GPUs with a per device batch of 4 and the
data epoch is set to 1.

## Multi-grained Benchmark and Metrics

To advance fine-grained document understanding, we build a bilingual
benchmark including 9 sub-tasks. We collect 112 English pages and 100
Chinese pages, including single/multi-column formats. The number of
words per page exceeds 1,000. These images are used to evaluate page
OCR, line-level OCR, color-guided OCR, region-level
OCR/translation/summary, multi-page multi-region OCR, and cross-page
VQA. Besides, to monitor the performance of interleaved data, we render
200 natural images sampled from
Laion-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) onto 200 PDF pages to
evaluate the document-level in-figure caption task. The comprehensive
evaluation metrics contain normalized edit distance, F1-score,
BLEU [papineni2002bleu](http://arxiv.org/pdf/2202.11027v1),
METEOR [banerjee2005meteor](http://arxiv.org/pdf/2312.00536v1),
ROUGE [lin2004rouge](http://arxiv.org/pdf/2209.06517v2), and *etc*.

<div id="tab:en_page_ocr" markdown="1">

| **Method** | Params | Edit Distance $\downarrow$ | F1-score $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ | BLEU $\uparrow$ | METEOR $\uparrow$ |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | 34B | 0.430 | 0.647 | 0.573 | 0.881 | 0.478 | 0.582 |
| InternVL-ChatV1.5 [chen2024far_intervl1.5](http://arxiv.org/pdf/2404.16821v2) | 26B | 0.393 | 0.751 | 0.698 | 0.917 | 0.568 | 0.663 |
| Nougat [blecher2023nougat](http://arxiv.org/pdf/2308.13418v1) | 250M | 0.255 | 0.745 | 0.720 | 0.809 | 0.665 | 0.761 |
| Vary [wei2023vary](http://arxiv.org/pdf/2312.06109v1) | 7B | 0.092 | 0.918 | 0.906 | 0.956 | 0.885 | 0.926 |
| Vary-toy [wei2024small_varytoy](http://arxiv.org/pdf/2401.12503v1) | 1.8B | 0.082 | 0.924 | 0.919 | 0.938 | 0.889 | 0.929 |
| Qwen-VL-Plus [Qwen-VL](http://arxiv.org/pdf/2308.12966v3) | \>100B | 0.096 | 0.931 | 0.921 | 0.950 | 0.893 | 0.936 |
| Qwen-VL-Max [Qwen-VL](http://arxiv.org/pdf/2308.12966v3) | \>100B | 0.057 | **0.964** | 0.955 | **0.977** | **0.942** | **0.971** |
| Fox (foreground focus) | **1.8B** | **0.046** | 0.952 | **0.957** | 0.948 | 0.930 | 0.954 |

Dense English text recognition on the single document page.

</div>

<div id="tab:cn_page_ocr" markdown="1">

| **Method** | Params | Edit Distance $\downarrow$ | F1-score $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ | BLEU $\uparrow$ | METEOR $\uparrow$ |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| InternVL-ChatV1.5 [chen2024far_intervl1.5](http://arxiv.org/pdf/2404.16821v2) | 26B | 0.265 | 0.816 | 0.784 | 0.866 | 0.622 | 0.717 |
| Vary-toy [wei2024small_varytoy](http://arxiv.org/pdf/2401.12503v1) | 1.8B | 0.142 | 0.914 | 0.928 | 0.907 | 0.718 | 0.832 |
| Qwen-VL-Plus [Qwen-VL](http://arxiv.org/pdf/2308.12966v3) | \>100B | 0.121 | 0.895 | 0.903 | 0.890 | 0.684 | 0.828 |
| Vary [wei2023vary](http://arxiv.org/pdf/2312.06109v1) | 7B | 0.113 | 0.952 | 0.961 | 0.944 | 0.754 | 0.873 |
| Qwen-VL-Max [Qwen-VL](http://arxiv.org/pdf/2308.12966v3) | \>100B | 0.091 | 0.931 | 0.917 | 0.946 | 0.756 | 0.885 |
| Fox (foreground focus) | **1.8B** | **0.061** | **0.954** | **0.964** | **0.946** | **0.842** | **0.908** |

Dense Chinese text recognition on the single document page.

</div>

<div class="tabular" markdown="1">

clcccccc & & &  
(rl)3-5 (rl)6-8 & & color & region & line & color & region & line  
& Edit Distance $\downarrow$ & 0.064 & 0.059 & 0.116 & 0.114 & 0.042 &
0.084  
& F1-score $\uparrow$ & 0.940 & 0.957 & 0.879 & 0.884 & 0.955 & 0.918  
& Precision $\uparrow$ & 0.942 & 0.962 & 0.879 & 0.902 & 0.966 & 0.931  
& Recall $\uparrow$ & 0.942 & 0.955 & 0.883 & 0.873 & 0.947 & 0.909  
& BLEU $\uparrow$ & 0.868 & 0.914 & 0.845 & 0.778 & 0.885 & 0.825  
& METEOR $\uparrow$ & 0.938 & 0.955 & 0.878 & 0.848 & 0.934 & 0.886  

</div>

## Evaluation Results

#### Foreground focus for dense text recognition on a single page.

For the dense text recognition on the entire page, we directly input the
normalized box $\left[2, 2, 998, 998\right]$ as the foreground prompts.
As shown in Table <a href="#tab:en_page_ocr" data-reference-type="ref"
data-reference="tab:en_page_ocr">1</a> and
 <a href="#tab:cn_page_ocr" data-reference-type="ref"
data-reference="tab:cn_page_ocr">2</a>, Fox showcases strong English and
Chinese dense OCR ability by almost lossless compression for the
document page. Specifically, Fox achieves the best edit distance of
0.046 and 0.061 in English and Chinese, respectively. Compared to
Vary-toy using the image-level prompts, the proposed Fox lifts the
English F1-score by 2.8% by redefining the task as foreground focus.
Note that the performance of LLaVA-NeXT and InternVL-ChatV1.5 which use
the CLIP-style vocabulary is bottle-necked, indicating that the dense
texts of each patch are not completely encoded.

#### Region focusing performance of in-document fine-grained tasks.

As shown in Table <a href="#tab:boxline" data-reference-type="ref"
data-reference="tab:boxline">[tab:boxline]</a>, Fox can yield excellent
OCR results on various metrics under several
color-guided/region-level/line-level settings, indicating that our model
can accurately recognize the content in these randomly sampled RoIs. In
Table <a href="#tab:translation_summary" data-reference-type="ref"
data-reference="tab:translation_summary">3</a>, for the region-level
translation, Fox yields an acceptable METEOR of 0.366 due to the smaller
language model of 1.8B parameters. In addition, we evaluate our model on
the fine-grained summary task and obtain a decent ROUGE-L-F score of
0.282. It is worth mentioning that this kind of usage similar to a
reading pen is exactly what users need more.

<div id="tab:translation_summary" markdown="1">

| **Fine-grained Translation** |  | **Fine-grained Summary** |  |  | **Fine-grained Caption** |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1-2 (rl)3-5 (rl)6-7 BLEU | METEOR | ROUGE-L R | ROUGE-L P | ROUGE-L F | METEOR | ROUGE-L F |
| 0.138 | 0.366 | 0.261 | 0.316 | 0.282 | 0.359 | 0.396 |

The performance of in-document fine-grained understanding tasks. The
fine-grained translation/summary/caption tasks are targeted at
interpreting in-document text/figure regions.

</div>

<div id="tab:multipage" markdown="1">

| **Method** | **Multi-page (8 pages) multi-region OCR** |  |  |  | **Cross-page (8 pages) VQA** |
|:---|:--:|:--:|:--:|:--:|:--:|
| 2-5 (rl)6-6 | Edit Distance $\downarrow$ | F1-score $\uparrow$ | BLEU $\uparrow$ | METEOR $\uparrow$ | Accuracy $\uparrow$ |
| Fox (Ours) | 0.084 | 0.946 | 0.836 | 0.805 | 0.827 |

The performance of fine-grained tasks on the multi-page (8 pages)
documents.

</div>

#### Cross-vocabulary focusing tasks on interleaved pages.

The color-guided task requires cross-vocabulary visual knowledge,
*i.e.*, CLIP for recognizing colors and Vary-tiny for capturing texts.
Table <a href="#tab:boxline" data-reference-type="ref"
data-reference="tab:boxline">[tab:boxline]</a> shows that the decent
results (0.940 and 0.884 on English and Chinese F1-score) meet our
expectations due to the collaboration across multiple vision
vocabularies. For the in-document figure caption task, we render natural
images onto document pages and ask our model “*What is this in the box
$<box>$?*”, where $<box>$ is the boundary of the natural image that is
pasted into the document page. As shown in
Table <a href="#tab:translation_summary" data-reference-type="ref"
data-reference="tab:translation_summary">3</a>, when handling
interleaved data, Fox reaches the METEOR of 0.359 and ROUGE-L-F of 0.396
due to the full reaction of activating multiple vocabularies.

#### Exploration for focusing on multiple pages.

To verify the focusing capability of Fox on multi-page documents, we
report two relevant results in
Table <a href="#tab:multipage" data-reference-type="ref"
data-reference="tab:multipage">4</a>. For the multi-page OCR task, we
ask the model to output the OCR results of 8 boxes on 8 complex pages
(in mixed English/Chinese and mixed single/multi-column formats) in a
single-turn conversation. Our Fox still performs an amazing F1-score of
0.946 and achieves true focus anywhere by parsing the entire 8-page
document simultaneously. For the cross-page visual question-answering
task which requires the model to answer which box has the largest number
of characters in multiple cross-page boxes, Fox yields a high accuracy
of 0.827, demonstrating that it is easier to perform VQA reasoning based
on successfully perceiving dense text of multiple pages.

<figure id="fig:vis">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/vis.png" style="width:100.0%" />
<figcaption>Visualization results. Fox can focus anywhere by supporting
fine-grained features, such as in-document figure caption, color-guided
OCR, VQA in the cartoon book, and <em>etc</em>. </figcaption>
</figure>

#### Visualization.

Figure <a href="#fig:vis" data-reference-type="ref"
data-reference="fig:vis">3</a> shows our Fox can perform impressive
features with high human-AI interactivity. For the figure on the
academic page, Fox gives the response “global seismic hazards” which is
relevant to the content of the document. Fox can also give precise OCR
results by dense text perception. For the cartoon book, Fox can
recognize the interesting “lion” and can read the story texts for users.
This indicates that our Fox enjoys fine-grained focusing capabilities in
various scenarios.

# Conclusion and Limitations [discussion]

This paper proposes a user-friendly LVLM named Fox, which enjoys amazing
fine-grained capabilities of focusing anywhere on single/multi-page
documents. Further, after catalyzing the multiple vision vocabularies
into a full reaction, Fox gains impressive cross-vocabulary features on
figure-text interleaved pages. To advance the fine-grained document
understanding, we provide a benchmark containing comprehensive
sub-tasks. Our Fox can achieve promising scores in these experiments,
making a successful step to high human-AI interactivity on dense-content
documents. We believe that the proposed method has considerable room for
improvement (*e.g.*, the low-resolution CLIP), and we encourage more
researchers to focus on more reasonable multi-page document-level tasks.

# Appendix

We show more amazing output results of our model Fox. All testing images
are from the Internet.

<figure id="fig:append1">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/append8page.png" style="width:100.0%" />
<figcaption>Fox can give precise responses when focusing on the 8-page
document. These pages contain bilingual content, have well over a
thousand characters per page, and have a variety of single and
multi-column layouts. This extreme case demonstrates powerful focusing
capabilities.</figcaption>
</figure>

<figure id="fig:append2">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/append8page2.png" style="width:100.0%" />
<figcaption>The left case shows Fox can handle the cross-page VQA task
on the multi-page (8 pages as an example) document. The right case shows
Fox can perform the dense Chinese text recognition by foreground focus
and obtain precise results.</figcaption>
</figure>

<figure id="fig:append3">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/append_en.png" style="width:100.0%" />
<figcaption>The proposed Fox easily performs dense English text
recognition by foreground focus.</figcaption>
</figure>

<figure id="fig:append4">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/append_story.png" style="width:100.0%" />
<figcaption>Fox can achieve text-associative in-page figure caption and
fine-grained document understanding. Fox enjoys high flexibility and
robustness when performing fine-grained region-level
translation/summary/OCR tasks in multi-column documents.</figcaption>
</figure>

<figure id="fig:append5">
<embed src="/papers/vision_rich/arXiv-2405.14295v1_md/fig/append_natural.png" style="width:100.0%" />
<figcaption>Of course, Fox can yield interesting results in cartoon and
natural scenes.</figcaption>
</figure>

[^1]: This work was done when the first author was interning at Megvii
    Technology Inc.