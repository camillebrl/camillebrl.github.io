<figure id="fig1">
<embed src=".//papers/vision_rich/arXiv-2311.06607v3_md/figure/fig1_big-crop.png" />
<figcaption>Comparisons of existing pipelines for document
understanding. Contrasting with (a) vision-constrained, (b)
language-constrained, and (c) unconstrained methods, our DocPedia
efficiently processes high-resolution document images and performs
logical reasoning using the world knowledge of large language models.
The instructions Q1, Q2, and Q3 evaluate the text recognition, world
knowledge, and text localization abilities, respectively.</figcaption>
</figure>

# Introduction

Document understanding [srihari1986document](http://arxiv.org/pdf/2304.06447v5) is a
critical and challenging task that sits at the intersection of computer
vision and natural language processing. It involves the *perception* and
*comprehension* in terms of visual and textual content embedded within
document images. The difficulty of this task stems from the diverse and
complex formats of high-resolution documents, where the sparse or dense
texts are intertwined with graphics and tables. The accurate parsing of
documents not only propels the digitization of archival materials but
also facilitates the document automation in current data-rich world,
such as information
extraction [hwang2019post](None), [kim2022ocr](None), [luo2023geolayoutlm](None)
and visual question
answering [ye2023mplug](None), [feng2023unidoc](None), [ye2023ureader](None), [lv2023kosmos](None).

Many early
attempts [xu2021layoutxlm](None), [xu2020layoutlm](None), [huang2022layoutlmv3](None), [hong2022bros](http://arxiv.org/pdf/2108.04539v5), [bai2022wukong](None), [tang2023unifying](http://arxiv.org/pdf/2212.02623v3), [li2021structext](None), [peng2022ernie](None), [appalaraju2021docformer](None)
in the field follow a perceive-then-comprehend paradigm, initially
involving Optical Character
Recognition (OCR) [liao2020real](http://arxiv.org/pdf/1911.08947v2), [shi2016end](http://arxiv.org/pdf/1507.05717v1) of document
images, followed by the fusion of textual, layout, and visual features
for content parsing. However, the individual processing step of OCR may
precipitate the accumulation of errors. Furthermore, considering the
intrinsic interweaving of visual elements and textual segments within
documents, the reciprocity between perception and comprehension awaits
further exploration.

To attack the issue, OCR-free solutions have emerged as recent
prevailing approaches in the field. Among them, most models commonly
generate a sequence of tokens that can be converted into a target
string [ye2023mplug](None), [feng2023unidoc](None), [ye2023ureader](None), [zhang2023llavar](None), [ye2023mplug-doc](None)
or a structured format
data [kim2022ocr](None), [lv2023kosmos](None), [lee2023pix2struct](None). Such
generative models are skilled at synthesizing and rephrasing
information, which naturally can unveil the implicit content or purpose
behind the source material, as well as provide deeper insights and more
versatile responses to inquiries. As depicted in
Fig. <a href="#fig1" data-reference-type="ref" data-reference="fig1">1</a>,
they can be mainly categorized into three groups, namely (a)
*vision-constrained*, (b) *language-constrained*, and (c)
*unconstrained* types, described next.

Specifically, in vision-constrained methodologies such as
LLaVAR [zhang2023llavar](None),
mPLUG-DocOwl [ye2023mplug-doc](None), and
UniDoc [feng2023unidoc](None), the visual encoders largely rely
on a pre-trained CLIP-ViT [radford2021learning](http://arxiv.org/pdf/2404.19696v1),
operating at input resolutions of 224 or 336. These resolutions are
designed for images featuring texts in medium or large font sizes,
*e.g.*, scene text, but prove inadequate for text-intensive
high-resolution documents where more details are
indispensable [liu2023hidden](None). As shown in
Fig. <a href="#fig1" data-reference-type="ref" data-reference="fig1">1</a> (a),
when a high-resolution supermarket receipt is downscaled to 224 for
model input, the text becomes unreadable, rendering these methods
incapable of answering the three presented instructions. In contrast,
language-constrained approaches, including
Donut [kim2022ocr](None),
KOSMOS-2.5 [lv2023kosmos](None), and
Pix2Struct [lee2023pix2struct](None), employ high-resolution
input for training their models with a vision encoder. They abandon the
use of large language models (LLMs) in vision-constrained
methods [zhang2023llavar](None), [ye2023mplug-doc](None), [feng2023unidoc](None),
and instead opt for a lightweight language
decoder [vaswani2017attention](http://arxiv.org/pdf/2107.08000v1). While these approaches
demonstrate promising perception ability, their comprehension
performance is often compromised. This is because the vital components
of robust logical reasoning and extensive world knowledge, typically
provided by the LLM, are not adequately incorporated. Taking
Fig. <a href="#fig1" data-reference-type="ref" data-reference="fig1">1</a> (b)
for example, in response to the instruction Q2, these models falter in
providing accurate answers due to a deficiency in pertinent knowledge.

The *status quo* triggers a question: *Is there a feasible approach to
maintain both perception and comprehension abilities without
compromising vision and language?*

To mitigate the problem in above both categories, unconstrained
method [ye2023ureader](None) (Fig. <a href="#fig1" data-reference-type="ref" data-reference="fig1">1</a> (c))
takes a further step by proposing a shape-adaptive cropping strategy.
This strategy involves cropping high-resolution images into patches,
which are then used in conjunction with a frozen low-resolution
CLIP-ViT [radford2021learning](http://arxiv.org/pdf/2404.19696v1) and LLM. However, this
heuristic-based crop strategy may lead to semantic discontinuities, even
after fusion is performed. Furthermore, the features extracted by
CLIP-ViT [radford2021learning](http://arxiv.org/pdf/2404.19696v1) are not well-suited for
tasks that require fine-grained local detail, such as text
detection [feng2023unidoc](None) or grounding (refer to Q3 in
Fig. <a href="#fig1" data-reference-type="ref" data-reference="fig1">1</a>
(c)).

<div class="figure*" markdown="1">

<embed src=".//papers/vision_rich/arXiv-2311.06607v3_md/figure/overviewv9.png" />

</div>

To answer the question aforementioned, this work reinspects the problem
through the lens of frequency and proposes DocPedia, a novel yet
effective Large Multimodal Model (LMM), aiming to achieve versatile
OCR-free document understanding. DocPedia is capable of parsing
high-resolution document images up to
2,560$\times$`<!-- -->`{=html}2,560, and harnessing the extensive world
knowledge and powerful inference capabilities offered by
LLMs [touvron2023llama](None), [chiang2023vicuna](None). This
integration aims to enhance both perception and comprehension aspects.
Technically, contrasting with previous LMMs in the filed, DocPedia
directly processes visual input in the frequency
domain [ahmed1974discrete](http://arxiv.org/pdf/1109.0337v1), [wallace1991jpeg](http://arxiv.org/pdf/1305.0020v1), [liu2023devil](http://arxiv.org/pdf/2204.08227v1), [liu2022nommer](None)
rather than the pixel space. This unique characteristic of the frequency
domain enables DocPedia to capture a greater amount of visual and
textual information using a limited number of visual tokens.

Employing this effective architecture, we train our DocPedia with two
phases: i) *text-aware pre-training* and ii) *context-aware
fine-tuning*. During the pre-training stage, the vision encoder is
trained to align the frequency domain features with a
LLM [chiang2023vicuna](None), incorporating various perception
tasks across both document and natural scene contexts, such as text
detection [liao2020real](http://arxiv.org/pdf/1911.08947v2),
spotting [liu2018fots](http://arxiv.org/pdf/1801.01671v2), paragraph reading, image
captioning [hossain2019comprehensive](http://arxiv.org/pdf/1810.04020v2), and *etc*. In the
subsequent fine-tuning stage, the focus shifts to the simultaneous
learning of perception and comprehension, *e.g.*, lower-level
reading-related tasks and higher-level document understanding. To ensure
the robustness of the model as well as a consistent response style, we
enrich the instructions and annotations of all these tasks with
GPT [brown2020language](http://arxiv.org/pdf/2112.07522v2). Extensive quantitative and
qualitative experiments are performed on this constructed large-scale
instruction tuning dataset covering multiple document types. The results
demonstrate the mutual benefits of jointly learning perception and
comprehension tasks.

The contributions are summarized as follows:

-   To the best of our knowledge, we are the first to scale a large
    multimodal model for document understanding tasks to the resolution
    of 2,560$\times$`<!-- -->`{=html}2,560.

-   We innovatively transform image domain inputs into frequency ones,
    enabling capturing more visual and textual information using a
    limited number of visual tokens.

-   We achieved superior performance on multiple publicly available
    benchmark datasets and conducted extensive experiments to validate
    the effectiveness of DocPedia.

# Related Work

In the following, we provide an overview of existing research in the
field of document understanding. This body of work is categorized into
two distinct types: OCR-driven and OCR-free methodologies, discussed
next.

## OCR-driven Document Understanding

This section outlines methods that initiate with text extraction from
document images, followed by the integration of textual, layout, and
visual features for thorough content analysis. Prominent among these are
the LayoutLM
series [xu2021layoutxlm](None), [xu2020layoutlm](None), [huang2022layoutlmv3](None),
which enhance text and layout modeling and integrate complex multimodal
pre-training for richer representation learning.
Wukong-Reader [bai2022wukong](None) employs pre-training
objectives to exploit the structural knowledge of document textlines,
incorporating textline-region contrastive learning for advanced visual
document understanding. StrucTexT [li2021structext](None)
combines a segment-token aligned encoder with diverse pre-training
tasks, targeting enhanced structured text analysis in visually rich
documents. DocFormer [appalaraju2021docformer](None) fuses
text, vision, and spatial information using a distinct transformer
architecture. However, the dependence of these methods on Optical
Character Recognition (OCR) can result in error accumulation, raising
efficacy concerns regarding the segregation of OCR in the context of the
intertwined nature of visual and textual elements in documents.

## OCR-free Document Understanding

To address this issue, prevailing OCR-free models excel in generating
token sequences for varied responses and structured information
synthesis, thereby offering enhanced insights and versatility in content
creation and inquiry response. Typically,
LLaVAR [zhang2023llavar](None) enhances document understanding
by improving interaction skills with humans and boosting performance on
text-rich image tasks, building upon its predecessor
LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1) with advanced visual instruction
tuning techniques. Based on the large multimodal model
mPLUG-Owl [ye2023mplug](None),
mPLUG-DocOwl [ye2023mplug-doc](None) integrates a unified
instruction tuning strategy across diverse document data.
UniDoc [feng2023unidoc](None) combines foundational OCR
learning with text-rich image comprehension tasks, markedly boosting
text scene image understanding. Despite their strong representational
skills and world knowledge from extensively pre-trained
CLIP-ViT [radford2021learning](http://arxiv.org/pdf/2404.19696v1) and large language models,
these methods are limited to processing images with larger, sparser text
due to the pre-trained visual models’ lower resolution constraints.
StrucTexTv2 [yu2023structextv2](None) employs self-supervised
pre-training for document images, adeptly integrating masked image and
language modeling [he2022masked](http://arxiv.org/pdf/2208.00173v1), [devlin2018bert](None) to
enhance performance. Donut [kim2022ocr](None) introduces an
end-to-end trainable model that overcomes OCR limitations by using a
synthetic document image generator for pre-training, enhancing
performance in document understanding tasks.
Pix2Struct [lee2023pix2struct](None) through its unique
pretraining on web page screenshots parsed into HTML, introduces a
variable-resolution input and integrated language-vision approach. As an
evolution of Kosmos-2 [peng2023kosmos](None),
Kosmos-2.5 [lv2023kosmos](None) processes text-rich images,
skillfully blending spatial text block creation and structured markdown
generation in a streamlined, decoder-only model.
UReader [ye2023ureader](None) innovatively employs a
shape-adaptive cropping module for high-resolution image processing.

While these work exhibit outstanding outcomes in various aspects, they
either struggle with handling high-resolution input or face challenges
due to a lack of world knowledge. This underscores the future research
endeavors: the development of an intelligent system adept at handling
documents of various types and resolutions.

# Method

Fig. <a href="#fig:overview" data-reference-type="ref"
data-reference="fig:overview">[fig:overview]</a> presents an overview of
DocPedia. It consists of two training phases: (a) text-aware
pre-training to align the visual features from the frequency domain to
the large language model, and (b) context-aware fine-tuning for learning
the parsing of documents. In the following, we first delineate the
network architecture of DocPedia, followed by a detailed exposition of
its two training phases.

## Architecture of DocPedia

Given an input RGB document image, we first resize it to our designated
training scale of $H\times W$ to obtain the image $\bm{I}$. By default,
both $H$ and $W$ are set as 2,560. Here we preserve the aspect ratio
during the resizing process to prevent distortion of textual elements.
Then, as shown in Fig. <a href="#fig:dct" data-reference-type="ref"
data-reference="fig:dct">2</a>, we apply the JPEG DCT
extraction [ahmed1974discrete](http://arxiv.org/pdf/1109.0337v1), [wallace1991jpeg](http://arxiv.org/pdf/1305.0020v1) to
retrieve the DCT coefficients for the $\bm{Y}$, $\bm{Cb}$, and $\bm{Cr}$
channels. The DCT coefficients are scaled down due to
8$\times$`<!-- -->`{=html}8 block processing for the luminance component
($\bm{Y}$) and additional chroma subsampling for color components
($\bm{Cb}$ and $\bm{Cr}$), resulting in $\frac{1}{8}$ and $\frac{1}{16}$
scales respectively. Each of them features $C$ channels. After that, we
upscale $\bm{Cb}$ and $\bm{Cr}$ to a $\frac{1}{8}$ scale based on
bilinear interpolation, followed by a concatenation along the channel
dimension. Subsequent to this is a 1$\times$`<!-- -->`{=html}1
convolutional layer, employed to map the channel dimension of the
concatenated map to that of the following backbone’s input. Through
these operations, we acquire the frequency domain counterpart of image
$\bm{I}$, denoted as $\bm{F}$.

Next, we feed $\bm{F}$ into the Swin
Transformer [liu2021swin](http://arxiv.org/pdf/2306.13776v1), a visual backbone that
leverages shifted windowing schemes to efficiently model spatial
hierarchies. In our implementation, we remove the 1/4 scale downsampling
module originally present before stage 1. The output of the visual
backbone is a feature map downsampled by a factor of 1/64. It is
subsequently flattened, resulting in $\frac{H}{64}\times \frac{W}{64}$
tokens, each with a dimensionality of 1,024. Drawing inspiration from
the paradigms of advanced large multimodal
models [zhu2023minigpt](None), [liu2023visual](http://arxiv.org/pdf/2402.11690v1), we employ a linear
layer to align these tokens with the input token dimension of the
following large language model [chiang2023vicuna](None).
Finally, the dimensionally aligned visual tokens are concatenated with
the tokens transformed from the language instructions. This concatenated
sequence is then fed into the LLM, generating the output response.

<figure id="fig:dct">
<embed src=".//papers/vision_rich/arXiv-2311.06607v3_md/figure/DCTv4.png" />
<figcaption>Schematic illustration of the DCT transformation and
frequency adapter module in DocPedia.</figcaption>
</figure>

## Text-aware Pre-training [sec:pre]

To develop a vision encoder capable of processing frequency domain
representation input and align it with the feature space of the
following large language model [chiang2023vicuna](None), we
first undertook extensive text-aware pre-training. During this stage, we
freeze the large language model, focusing on the optimization of the
vision encoder and its subsequent linear projector, as illustrated in
Fig. <a href="#fig:overview" data-reference-type="ref"
data-reference="fig:overview">[fig:overview]</a>.

Specifically, our pre-training encompassed a variety of perception
tasks, including text detection [liao2020real](http://arxiv.org/pdf/1911.08947v2),
recognition [wang2011end](http://arxiv.org/pdf/1811.10003v1),
spotting [liu2018fots](http://arxiv.org/pdf/1801.01671v2), paragraph reading, full-text
reading [kim2022ocr](None), and image
captioning [hossain2019comprehensive](http://arxiv.org/pdf/1810.04020v2). The first three
tasks are foundational OCR tasks. “Paragraph reading" denotes the
reading of the text within a specified bounding box (see bottom case in
Fig. <a href="#fig:demo_perc" data-reference-type="ref"
data-reference="fig:demo_perc">3</a>), whereas “full-text reading"
refers to deciphering all text in the image. It is worth noting that the
first five tasks focus on a diverse array of document images, while the
final task targets natural scene images. This comprehensive pre-training
enables the vision encoder of our DocPedia to effectively perceive
textual and visual information from both document and natural scene
images.

<div id="table:dataset-summary" markdown="1">

| **Stage** | **Image** | **Instruction** | **Task** | **\# Conv** |  |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Pre-training | Scene | LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1) | $\mathcal{C}$ | 595K |  |
|  | PDF | OCR | $\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{R}_p,\mathcal{R}_f$ | 325K |  |
|  | PPT | OCR | $\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{R}_p,\mathcal{R}_f$ | 600K |  |
| Fine-tuning | PDF | OCR | $\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{R}_p,\mathcal{R}_f$ | 325K |  |
|  | PPT | OCR | $\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{R}_p,\mathcal{R}_f$ | 600K |  |
|  | Scene | LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1) | $\mathcal{U}$ | 158K |  |
|  | Benchmark | GPT | $\mathcal{U}$ | 370K |  |

Summary of the training data statistics across two stages. The symbols
represent various instruction-following tasks as follows: $\mathcal{D}$
for text detection, $\mathcal{R}$ for text recognition, $\mathcal{S}$
for text spotting, $\mathcal{R}_p$ for paragraph reading,
$\mathcal{R}_f$ for full-text reading, $\mathcal{C}$ for image
captioning, and $\mathcal{U}$ for document understanding.

</div>

<div class="table*" markdown="1">

</div>

## Context-aware Fine-tuning

In the fine-tuning phase, we concurrently cultivate the perception and
comprehension capabilities of DocPedia. Concretely, within each batch of
training data, one half is dedicated to the five types of OCR tasks
outlined in the pre-training phase, while the other half comprises tasks
that demand a higher level of semantic understanding related to
document [mathew2021docvqa](None) and
scene [liu2023visual](http://arxiv.org/pdf/2402.11690v1). We argue that the concurrent
learning of lower-level perceptual abilities and the cultivation of
higher-level understanding capabilities can maximize the performance of
the model. During this stage, we unfreeze the LLM and fine-tune the
entire model.

# Dataset Construction

To train our DocPedia, we construct a large-scale multimodal instruction
following dataset. The statistical data employed during the pre-training
and fine-tuning phases are summarized in
Table <a href="#table:dataset-summary" data-reference-type="ref"
data-reference="table:dataset-summary">1</a>. We detail them in the
following.

<div id="tab:vision_commands" markdown="1">

| **Type** | **Example** |
|:---|:---|
| Detection | “Where are the texts located in the photo?" |
| Recognition | “Recognize all the text in this image." |
| Spotting | “Identify all the text in the shot return their coordinates in the format of \[x1,y1,x2,y2\]." |
| Paragraph Reading | “Tell me about the content in the area marked as \[0.124,0.276,0.353,0.487\] of the frame." |
| Full Text Reading | “Convey the entire content of this pic to me." |

Different types of OCR instructions and their examples.

</div>

## Pre-training

During the pre-training phase, our focus was on the learning of
perceptual abilities, particularly in the context of text perception. As
illustrated in Table 1, we amassed a dataset comprising 600,000
PowerPoint (PPT) images and 325,000 PDF images. The PowerPoint images
are sourced from the “Common Crawl" dataset[^3], an extensive web corpus
encompassing publicly accessible web pages. The PDF images are sourced
from arXiv[^4], an established online platform for scientists to publish
pre-print research papers.

For each of these images, we randomly selected an Optical Character
Recognition (OCR) task type as described in
Sec. <a href="#sec:pre" data-reference-type="ref"
data-reference="sec:pre">3.2</a> and then constructed corresponding
instructions and responses [feng2023unidoc](None). On one hand,
to ensure instruction diversity, we generated multiple variations of
instructions for each OCR task using
GPT-3.5 [brown2020language](http://arxiv.org/pdf/2112.07522v2). In
Table <a href="#tab:vision_commands" data-reference-type="ref"
data-reference="tab:vision_commands">2</a>, we present one exemplar for
each of the five text-aware perceptual tasks. For further examples,
please refer to the supplementary materials. On the other hand, for
their responses, we employed a standardized format (see
Fig. <a href="#fig:demo_perc" data-reference-type="ref"
data-reference="fig:demo_perc">3</a>). In addition to the aforementioned
data, we enriched our dataset with 595,000 caption entries from
LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1), aiming to enhance the DocPedia’s
perceptual abilities in natural scenes.

## Fine-tuning

Furthermore, during the fine-tuning phase, we first employed the same
data utilized during the pre-training phase, comprising 325,000 PDF and
600,000 PPT images. Building upon this, we introduced an extra 370,000
entries from seven visual question answering benchmark datasets,
including DocVQA [mathew2021docvqa](None),
OCRVQA [mishra2019ocr](None),
TextVQA [singh2019towards](None),
InfoVQA [mathew2022infographicvqa](None),
ChartVQA [masry2022chartqa](None),
FigureVQA [kahou2017figureqa](None), and PlotVQA. Notably, as
the responses in these datasets are typically concise, containing only
the answer itself, we employed
GPT-3.5 [brown2020language](http://arxiv.org/pdf/2112.07522v2) to expand these responses
into complete sentences. This adaptation was done to align with the
characteristic comprehensive and detailed response style of large
language models [chiang2023vicuna](None). Besides, we
supplemented the training data with 158,000 instruction tuning data for
natural scene understanding from LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1).
Our experiments demonstrate the effectiveness of a fine-tuning strategy
that concurrently learns perceptual and understanding abilities.

# Experiment

## Implementation Details

To implement DocPedia, we adopted a one-cycle learning rate
strategy [smith2019super](http://arxiv.org/pdf/1708.07120v3). For the pre-training phase,
the peak learning rate was established at 1e-3, which was set as 1e-5
during the subsequent fine-tuning phase. We maintained batch sizes of 64
and 8 for the pre-training and fine-tuning stages, respectively. We
employ the AdamW optimizer [loshchilov2017decoupled](http://arxiv.org/pdf/2311.11446v2) and
both training stages were performed on eight A100 GPUs, each spanning
just a single epoch.

For performance assessment, a temperature parameter of 0.2 was utilized
in both quantitative and qualitative evaluations. We adopted the
accuracy metric, where a response generated by the model is considered
correct if it contains the string present in the ground
truth [liu2023hidden](None).

<figure id="fig:demo_perc">
<embed src=".//papers/vision_rich/arXiv-2311.06607v3_md/figure/demo_percpv3.png" />
<figcaption>Exemplary demonstrations of DocPedia’s advanced text
perception capabilities. The three instances illustrate its adeptness in
accurately identifying and localizing text in scene and document images,
and demonstrating proficient paragraph reading skills. We visualized the
bounding boxes within the responses in the images. For the last case,
subsequent text readouts have been omitted for display convenience. Zoom
in for best view.</figcaption>
</figure>

<div class="figure*" markdown="1">

<embed src=".//papers/vision_rich/arXiv-2311.06607v3_md/figure/understanding-crop.png" />

</div>

## Results

We further conducted both quantitative and qualitative evaluations of
the current state-of-the-art multimodal large-scale models in comparison
to our proposed method.

**Qualitative results.** We qualitatively evaluate DocPedia’s perception
and comprehension capabilities on high-resolution scene text and
document images. Firstly, in terms of the perception capabilities, as
illustrated in Fig. <a href="#fig:demo_perc" data-reference-type="ref"
data-reference="fig:demo_perc">3</a>, our DocPedia can accurately locate
and identify text in both scenes and high-resolution documents, which is
attributed to the training of fundamental OCR tasks in
Table <a href="#table:dataset-summary" data-reference-type="ref"
data-reference="table:dataset-summary">1</a>. Secondly, regarding
comprehension abilities, as demonstrated in
Fig. <a href="#fig:demo" data-reference-type="ref"
data-reference="fig:demo">[fig:demo]</a>, the examples in the first two
rows indicate that DocPedia can perceive and understand the visual and
textual information in images to provide accurate responses, based on
the intention of the instructions. Moreover, the examples in the bottom
row illustrate that DocPedia is capable of integrating the content of
instructions, visual and textual information within images, and its
large language model’s rich world knowledge to formulate responses.
These results demonstrate DocPedia’s robust multimodal comprehension
capabilities. For additional examples, please refer to the supplementary
materials.

**Quantitative results.** Furthermore, we conduct a quantitative
evaluation of existing large multimodal models and our DocPedia. The
results are summarized in
Table <a href="#tab:per_com" data-reference-type="ref"
data-reference="tab:per_com">[tab:per_com]</a>. The benchmarks used for
this assessment consist of 3 Key Information Extraction (KIE) datasets,
including FUNSD [jaume2019funsd](http://arxiv.org/pdf/1905.13538v2),
SROIE [huang2019icdar2019](None) as well as
POIE [kuang2023visual](http://arxiv.org/pdf/2102.06732v1), and 6 Visual Question Answering
(VQA) datasets, including DocVQA [mathew2021docvqa](None),
ChartVQA [masry2022chartqa](None),
STVQA [biten2019icdar](None),
OCRVQA [mishra2019ocr](None),
TextVQA [singh2019towards](None), and
InfoVQA [mathew2022infographicvqa](None).

As we can see, on several high-resolution document image
benchmarks [jaume2019funsd](http://arxiv.org/pdf/1905.13538v2), [huang2019icdar2019](None), [kuang2023visual](http://arxiv.org/pdf/2102.06732v1), [mathew2021docvqa](None), [masry2022chartqa](None),
where the text is dense and tiny, our DocPedia demonstrates significant
performance improvements over existing state-of-the-art multimodal large
models. Notably, compared to the state-of-the-art LMMs, DocPedia
achieved an increase in accuracy by 40.20$\%$ on
DocVQA [mathew2021docvqa](None) and 28.67$\%$ on
FUNSD [jaume2019funsd](http://arxiv.org/pdf/1905.13538v2), respectively. These results
underscore the distinct advantages of our approach. Moreover, our method
also achieved considerable improvements on high-resolution scene text
benchmarks [biten2019icdar](None), [mishra2019ocr](None), [singh2019towards](None), [mathew2022infographicvqa](None),
though the enhancements were less pronounced. This can be attributed to
two primary factors: firstly, our pre-trained vision encoder was not
exposed to large-scale natural scene data as extensively as pre-trained
Vision Transformer (ViT) [radford2021learning](http://arxiv.org/pdf/2404.19696v1) employed
in previous
LMMs [feng2023unidoc](None), [zhu2023minigpt](None), [liu2023visual](http://arxiv.org/pdf/2402.11690v1);
secondly, in such images, the text often appears more sparsely and is
generally larger compared to the dense and tiny textual content in
document images.

## Ablation Studies

We further conduct ablation studies to validate the efficacy of core
settings and components in DocPedia. Note that all experiments were
conducted on two benchmark datasets:
DocVQA [mathew2021docvqa](None) and
TextVQA [singh2019towards](None).
DocVQA [mathew2021docvqa](None) is centered around document
comprehension, whereas TextVQA [singh2019towards](None) focuses
on scene text image understanding. Both datasets are notable for their
substantial sample sizes, comprising 5,000 and 5,349 test samples,
respectively.

<div id="aba1" markdown="1">

| Method |  |  | VQA |  |
|:--:|:--:|:--:|:--:|:--:|
| 1-3(lr)4-5 Input | Resolution | Tokens | DocVQA [mathew2021docvqa](None) | TextVQA [singh2019towards](None) |
| RGB | 640$\times$`<!-- -->`{=html}640 | 400 | 13.78 | 27.56 |
| RGB | 960$\times$`<!-- -->`{=html}960 | 900 | 21.15 | 41.18 |
| RGB | 1280$\times$`<!-- -->`{=html}1280 | 1600 | 29.54 | 48.80 |
| DCT | 1280$\times$`<!-- -->`{=html}1280 | 400 | 21.09 | 45.05 |
| DCT | 1920$\times$`<!-- -->`{=html}1920 | 900 | 37.83 | 53.35 |
| DCT | 2560$\times$`<!-- -->`{=html}2560 | 1600 | **47.08** | **60.18** |

Ablation experiments regarding the use of various resolutions in the RGB
domain and frequency domain as inputs for the vision encoder in
DocPedia. “Tokens" refers to the number of tokens outputted by the
vision encoder.

</div>

<div id="aba2" markdown="1">

| Pre-training | Fine-tuning |  | VQA |  |
|:--:|:--:|:--:|:--:|:--:|
| 2-3(lr)4-5 | Perception | Understanding | DocVQA [mathew2021docvqa](None) | TextVQA [singh2019towards](None) |
|  |  |  | 21.59 | 34.17 |
|  |  |  | 27.13 | 48.47 |
|  |  |  | **37.83** | **53.35** |

Ablation experiments concerning the training strategies of DocPedia
during the pre-training and fine-tuning phases. All ablations are
conducted at a resolution of 1,920$\times$`<!-- -->`{=html}1,920.

</div>

**Impact of training in the frequency domain.** One of the significant
contributions of our DocPedia lies in utilizing the frequency domain
representation of images as the input for the vision encoder. In
Table <a href="#aba1" data-reference-type="ref" data-reference="aba1">3</a>,
we evaluate our method’s performance using image inputs and frequency
domain inputs on varying scales. For image inputs, three resolution
settings were evaluated: 640, 960, and 1,280. Given that the backbone
Swin [liu2021swin](http://arxiv.org/pdf/2306.13776v1) downsamples input by a factor of 32,
the resultant token counts are 400, 900, and 1,600, respectively. In
experiments with our frequency domain inputs, we tested image
resolutions of 1,280, 1,920, and 2,560 for the DCT, resulting in token
counts corresponding to the three image-based experimental settings.

As we can see, with the same number of visual tokens, our DocPedia
yields better performance. This is attributed to the increased
resolution enabling enhanced perception of texture content within
images. In experiments where the input resolution is constant (1,280 in
Table <a href="#aba1" data-reference-type="ref" data-reference="aba1">3</a>),
we observe a slightly enhanced performance with image inputs compared to
frequency ones. Note that the number of visual tokens for the latter is
only a quarter of that used for the former. This is likely because our
frequency-based approach retains a limited number of tokens, leading to
some information loss. However, this constraint simultaneously
facilitates the incorporation of higher-resolution inputs, up to
2,560$\times$`<!-- -->`{=html}2,560.

In Fig. <a href="#fig:reso_aba" data-reference-type="ref"
data-reference="fig:reso_aba">4</a>, we further compare the responses of
DocPedia to the same academic image and instruction under varying input
resolutions. It is observed that the response becomes accurate when the
input resolution reaches 2,560.

**Impact of the training strategy.** We further study the impact of our
training strategies. Initially, we omitted the pre-training phase,
opting instead for a random initialization of the vision encoder. In
Table <a href="#aba2" data-reference-type="ref" data-reference="aba2">4</a>,
significant performance degradation was observed in the absence of
pre-training, underscoring the critical role of feature alignment
between the vision encoder and subsequent
LLM [chiang2023vicuna](None).

Additionally, we examined the fine-tuning strategies. Under default
settings, we concurrently learn perceptual and understanding
capabilities, incorporating tasks OCR, image captioning, document
understanding, and scene comprehension. Subsequently, we eliminated the
OCR and image captioning from the fine-tuning. The results clearly
indicated a notable decline in performance, affirming the efficacy of
our joint training strategy. This implies that the simultaneous
development of foundational perceptual skills augments the acquisition
of comprehension abilities.

<figure id="fig:reso_aba">
<embed src=".//papers/vision_rich/arXiv-2311.06607v3_md/figure/aba_case-crop.png" />
<figcaption>Comparison of DocPedia’s responses to varying resolutions of
DCT inputs for the same high-resolution document image, encompassing
scales of 1,280, 1,920, and 2,560. The response becomes accurate at the
scale of 2,560. Zoom in for best view.</figcaption>
</figure>

## Limitation Discussion

Furthermore, we discuss the limitations of our DocPedia. Firstly, as
illustrated in Table <a href="#tab:per_com" data-reference-type="ref"
data-reference="tab:per_com">[tab:per_com]</a>, we observe minimal
performance improvements on the InforVQA
dataset [mathew2022infographicvqa](None). This highlights one
of the constraints of DocPedia. Many images in
InforVQA [mathew2022infographicvqa](None) possess extremely
high aspect ratios, akin to vertically concatenating multiple pages of
images, with some even reaching dimensions of
6,400$\times$`<!-- -->`{=html}800. In addition, our DocPedia currently
lacks the capability to process multi-page document
images [tito2023hierarchical](None) and also exhibits a
deficiency in multilingual proficiency [Qwen-VL](None).

# Conclusion

This work introduces DocPedia, an innovative Large Multimodal Model
tailored for versatile OCR-free document understanding, capable of
handling images with high resolutions. Unlike existing methods, DocPedia
directly processes visual input in the frequency domain, where more
visual and textual information is captured in a limited number of visual
tokens. Thanks to the dual-stage training strategy designed and the
polished instructions/annotations for all tasks, DocPedia shows superior
performance on several public datasets. In conclusion, we provide a
successful attempt at pathways for handling complex high-resolution
documents. We expect our success in exploring LMM dealing with
high-resolution images from frequency perspective could trigger more
insights for the community.

[^1]: Equal contribution. $\spadesuit$ $\ddag$

[^2]: Corresponding authors: Wengang Zhou and Can Huang.

[^3]: https://commoncrawl.org/

[^4]: https://arxiv.org/