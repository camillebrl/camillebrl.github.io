[^1]: Equal contribution

[^2]: $^{\dagger}$ Corresponding authors

# Introduction

Leveraging strong Large Language Models as the language decoder, some
recent works propose Multimodal Large Language Models (MLLMs)
[minigpt4](http://arxiv.org/pdf/2402.17510v1), [llava](http://arxiv.org/pdf/2402.11690v1), [mplugowl](http://arxiv.org/pdf/2405.00390v2), [blip2](None) and achieve promising
vision-and-language understanding performance. Surprisingly, without
in-domain training, these MLLMs exhibit shallow zero-shot visual text
recognition ability when fed a low-resolution image with salient text
information [mplugowl](http://arxiv.org/pdf/2405.00390v2), [llmocr](http://arxiv.org/pdf/2305.07895v5). However, due to the
variety of image types and the wide range of image sizes, they are still
far from universal visually-situated language understanding, such as
extracting information from documents, reading texts from webpages, and
visual question and answering on tables, as shown in
<a href="#fig:intro_case" data-reference-type="ref+Label"
data-reference="fig:intro_case">[fig:intro_case]</a>.

Existing works for visually-situated language understanding can be
categorized into two-stage [layoutlmv2](http://arxiv.org/pdf/2310.16527v1), [layoutlmv3](None), [tap](None)
and end-to-end [dessurt](http://arxiv.org/pdf/2203.16618v3), [donut](http://arxiv.org/pdf/2305.09520v1), [pix2struct](None) methods
according to whether relying on an off-the-shelf OCR model or API. These
works all follow a domain-specific pretraining and finetuning paradigm,
thus leading to high training costs, e.g. end-to-end model Donut
[donut](http://arxiv.org/pdf/2305.09520v1) costs more than 192 A100 days.

Inspired by the shallow text recognition ability of existing MLLMs, in
this work, we propose for universal OCR-free visually-situated language
understanding, which leverages the multimodal Large Language Model via
low-cost instruction tuning [instructblip](None). Different
from previous works, we forgo pretraining tasks by leveraging the
existing MLLM and directly finetune MLLM by taking full advantage of
various Visually-situated Language Understanding datasets. To make the
most of the strong language understanding ability of MLLM, we convert
all tasks into the vision-language instruction tuning format. Besides,
to enhance text recognition and semantic understanding ability across
diverse domains, we design auxiliary text reading and key points
generation tasks in the same instruction format. To utilize the
low-resolution encoder of MLLM for processing high-resolution images and
avoid blurry and distortion problems due to resizing, we propose a
shape-adaptive cropping module to cut a high-resolution image into
multiple local images. Each image is firstly independently encoded with
the frozen visual encoder and a trainable visual abstractor and then
concatenated to feed into the language decoder. Moreover, we add
learnable crop position encoding to help the model correlate local
images and add a resized global image to alleviate salient information
loss due to cropping.

Our contributions in this work are four-fold: =0.1em

-   We first propose instruction tuning with Multimodal Large Language
    Models for OCR-free Visually-situated Language Understanding.

-   We build an instruction-tuning dataset covering 5 domains of
    visually-situated language understanding: document, table, chart,
    natural image, and webpage screenshot.

-   We design a shape-adaptive cropping module to utilize the frozen
    low-resolution vision encoder for processing high-resolution images.

-    achieves state-of-the-art OCR-free performance in 8 out of 10
    tasks, across 5 domains.

# Related Work

aims to comprehend images containing rich text information. The image
types are quite diverse, covering document
[docvqa](None), [infovqa](http://arxiv.org/pdf/2104.12756v2), [klc](None), [deepform](http://arxiv.org/pdf/2303.13839v1), [mpmqa](None), table
[wikitableqa](http://arxiv.org/pdf/2009.13845v2), [TabFact](http://arxiv.org/pdf/2311.06592v1), chart
[chartqa](None), [plotqa](http://arxiv.org/pdf/1906.04124v2), [dvqa](None), [figureqa](http://arxiv.org/pdf/2109.02226v1), natural image
[textvqa](None), [ocrvqa](None), [stvqa](http://arxiv.org/pdf/2304.01603v1), [qctextcap](http://arxiv.org/pdf/2302.02124v2), webpage screenshot
[visualmrc](http://arxiv.org/pdf/2101.11272v2), [websrc](http://arxiv.org/pdf/2004.14797v1), etc. Tasks of Visually-situated
Language Understanding range from visual question answering, image
captioning, information extraction to natural language inference.

According to whether using off-the-shelf OCR models or APIs to recognize
texts from images, existing work can be divided into two-stage models
[layoutlmv2](http://arxiv.org/pdf/2310.16527v1), [layoutlmv3](None), [udop](http://arxiv.org/pdf/2212.02623v3), [tap](None) and end-to-end models
[donut](http://arxiv.org/pdf/2305.09520v1), [dessurt](http://arxiv.org/pdf/2203.16618v3), [pix2struct](None). Two-stage work always
designs pretrianing tasks to learn cross-modality alignment between
visual inputs and text inputs. For example, for document understanding,
UDOP [udop](http://arxiv.org/pdf/2212.02623v3) design a Joint Text-Layout Reconstruction
task to recover masked texts and layout information given the visual
inputs and retained text inputs. LayoutLMv3 [layoutlmv3](None)
applies a Masked Image Modeling task to recover masked image tokens with
the context of their surrounding text and image tokens. Without the help
of an off-the-shelf OCR model, end-to-end models need to learn text
recognition with a high-resolution image encoder during the pretraining
stage. For example, Pix2Struct [pix2struct](None) proposes a
Screenshot Parsing pretraining task, where the model needs to generate
the complete HTML DOM tree with only a masked webpage screenshot as the
input. Donut [donut](http://arxiv.org/pdf/2305.09520v1) designs a pretraining task to
generate all texts in the document image. These work all follow a
domain-specific pretraining and finetuning paradigm and therefore ask
for high training costs, e.g. Donut is trained for more than 192 A100
days. In this work, by leveraging the shallow text recognition ability
of Multimodal Large Language Models, we propose to directly perform
instruction tuning across various types of images and greatly reduce the
training cost for universal visually-situated Language Understanding.

is developed to empower the Large Language Model with multi-modality
understanding ability, especially for vision information. These work
[kosmos](http://arxiv.org/pdf/2302.14045v2), [minigpt4](http://arxiv.org/pdf/2402.17510v1), [llava](http://arxiv.org/pdf/2402.11690v1), [mplugowl](http://arxiv.org/pdf/2405.00390v2), [blip2](None), [instructblip](None)
mainly connect a pre-trained vision encoder (usually CLIP VIT-L/14
[clip](http://arxiv.org/pdf/2404.19696v1)) with a strong large language model, such as LLaMA
[llama](http://arxiv.org/pdf/2402.08075v1). These MLLMs show some emergent abilities,
including shallow zero-shot text recognition ability
[llmocr](http://arxiv.org/pdf/2305.07895v5). However, they are still far from universal
visually-situated language understanding. Firstly, due to the
pretraining data for the vision encoder being mostly natural images,
MLLMs show barely acceptable text understanding performance on natural
images but bad performance on other types, such as document
[llmocr](http://arxiv.org/pdf/2305.07895v5). Secondly, most images for visuall-situated
language understanding are high-resolution. Rescaling them to low
resolution to adapt to the vision encoder can cause the texts blurry and
distorted. In this work, we propose to fully leverage the shallow text
recognition ability of MLLMs and perform instruction tuning to enhance
its universal understanding ability across 5 domains. Besides, we design
a shape-adaptive cropping module to alleviate the text blur and
distortion problem.

# 

<figure id="fig:overall_model">
<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/model.png" style="width:100.0%" />
<figcaption>The overall architecture of .</figcaption>
</figure>

The primary goal of  is to efficiently utilize existing MLLMs for
Visually-situated Language Understanding tasks. In this work, we utilize
but are not limited to, the mPLUG-Owl [mplugowl](http://arxiv.org/pdf/2405.00390v2) as our
basic MLLM. <a href="#fig:overall_model" data-reference-type="ref+Label"
data-reference="fig:overall_model">1</a> presents an overall
architecture of . The input image is firstly pre-processed by a
shape-adaptive cropping module (in
<a href="#sec:crop" data-reference-type="ref+Label"
data-reference="sec:crop">[sec:crop]</a>). The resulting sub-images are
then simultaneously passed through the visual encoder and visual
abstractor. To enable the large language model to correlate multiple
cropped sub-images, we apply a crop position encoding module to
introduce spatial information across sub-images. (in
<a href="#sec:modelling" data-reference-type="ref+Label"
data-reference="sec:modelling">1.2</a>).

## Shape-Adaptive Cropping Module

Images with texts have various aspect ratios and a great range of
resolutions. Simply resizing the image to $H_v, W_v$ (raw resolution of
the MLLM) can result in text being blurred, distorted, and
unrecognizable. Thus we propose a shape-adaptive cropping module.
Specifically, as shown in
<a href="#fig:crop_module" data-reference-type="ref+Label"
data-reference="fig:crop_module">2</a>, we pre-define grids
$\{g=(n_h\times n_w)|n_h\cdot n_w\le N_c, n_h \in \mathbb{N}, n_w \in \mathbb{N}\}$
with various shapes, where $n_h$ and $n_w$ denote the number of rows and
columns of the grid $g$ and $N_c$ denotes the maximum number of the
cells (sub-images). To select a suitable grid for an image $I$ with
shape $H \times W$, two rules should be followed: (1) The grid should
preserve the resolution of the image as much as possible, and (2) the
grid should fit the aspect ratio of the input image. To measure the
resolution coherence and shape similarity between the image and each
grid, we calculate the resolution-related and resolution-agnostic
insection over union $\mathrm{S_{rr}}$ and $\mathrm{S_{ra}}$ as follows:
$$\begin{aligned}
\mathrm{S_{rr}}(I, g)&=\mathrm{IoU}\left((H,W),(n_hH_v,n_wW_v)\right) \\
\mathrm{S_{ra}}(I, g)&=\mathrm{IoU}\left((\frac{n_wH}{W},n_w),(n_h, n_w)\right)
\end{aligned}$$ where $\mathrm{IoU}$ denotes the insection over the
union between two rectangles centered and aligned with each other. The
matched grid is selected by maximizing the matching score:
$$g^{*}=\argmax_{g} {\mathrm{S_{ra}}(I, g)+\mathrm{S_{rr}}(I, g)}$$
where $g^{*}$ is the selected grid. Then, we resize the input image to
$(n_hH_v,n_wW_v)$ and crop it to $n_h \cdot n_w$ local images. To
maintain the global structure information of the image, we also resize
the input image to $(H_v,W_v)$ as a global image. All images are then
passed on to the visual encoder and visual abstractor in parallel.

The visual encoder extracts visual feature
$V\in \mathbb{R}^{N \times (H'\cdot W')\times d_v}$ from the input
images $\mathbf{I}\in \mathbb{R}^{N\times H\times W \times 3}$, where
$N=(n_h\cdot n_w)+1$, $H'\cdot W'$ and $d_v$ denote the number and
dimension of the extracted visual features, respectively. The visual
abstractor further summarizes visual information and obtains higher
semantic visual representations
$V^{l} \in \mathbb{R}^{N\times N_q\times d_l}$ in language feature space
by several learnable queries, where $d_l$ denotes the dimension of
language feature space and $N_q$ denotes the number of learnable
queries. <span id="sec:crop" label="sec:crop"></span>

<figure id="fig:crop_module">
<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/crop.png" style="width:100.0%" />
<figcaption>The Shape-Adaptive Cropping Module.</figcaption>
</figure>

## Cropped Images Modeling with LLM [sec:modelling]

MLLMs are mostly trained with a single image as the input. Due to the
cropping module, we need to input visual features from multiple images
into the language model. The 1-dimensional position embeddings of LLM
can not reflect the spatial position of each sub-image, which is
critical to correlate local images. Therefore, we incorporate a
2-dimensional crop position encoding to help the language model to
understand the spatial relationship between cropped images.
Specifically, we assign a location index $(i, j)$ for each cell of the
selected grid and obtain their row embedding and column embedding by two
auxiliary embedding layers as follows: $$\begin{aligned}
\mathbf{e}^{row}_{i,j}&=\mathrm{Embedding_{row}}(i) \\
\mathbf{e}^{column}_{i,j}&=\mathrm{Embedding_{column}}(j) \\
\mathbf{e}_{i,j}&=\mathbf{e}^{row}_{i,j} + \mathbf{e}^{column}_{i,j}
\end{aligned}$$ where $\mathbf{e}_{i,j}\in \mathbb{R}^{D_l}$ denotes the
crop position embedding of the cell $(c_i, c_j)$. We add the embedding
to the visual feature of each cell in the language space via
broadcasting along the dimension of learnable queries:
$\bar{V}^l_{i,j}=V^l_{i,j}+\mathbf{e}_{i,j}$. We then reshape the visual
features into
$\bar{\mathbf{V}}^l\in \mathbb{R}^{(N\cdot N_q)\times d_l}$. The
resulting spatial-aware visual features and word embeddings of the input
sentences are concatenated at sequence dimension and sent to the large
language model.

In order to enhance the language model’s ability to effectively model
multiple images while keeping low training costs, we freeze the origin
language model and adopt the low-rank adaptation approach (LoRA)
 [hu2022lora](https://openreview.net/forum?id=nZeVKeeFYf9).

# Instruction Tuning

For developing a universal visually-situated language understanding
model that could process various types of images and perform different
comprehension tasks, we conduct low-cost instruction tuning with a
Multimodal Large Language Model. Without introducing any large-scale
pretraining datasets, we directly ensemble multiple downstream datasets
and perform joint training. Different downstream tasks are all
reorganized to the unified instruction format
[instructblip](None). Besides, we design auxiliary text reading
and key points generation tasks to enhance text recognition and semantic
understanding ability.

## Tuning Tasks

Downstream tasks of Visuall-situated Language Understanding cover Visual
Question Answering, Information Extraction, Natural Language Inference,
and Image Captioning. For developing a universal model, we reorganize
all tasks into the instruction tuning format
[instructblip](None). Concretely, for the Visual Question
Answering task, the question is directly used as the instruction:
"Human: {question} AI: {answer}". For the Information Extraction task,
each category and value pair is expressed with a prompt as "Human: What
is the value for the {category}? AI: {value}". If some categories don’t
exist in the image, the value is ‘None’. In the raw annotation of the
Natural Language Inference task, ‘1’ means ‘Entailed’ and ‘0’ means
‘Refuted’. We reorganize the NLI task by constructing the instruction
"Human: {statement}, Yes or No? AI: {answer}", where ‘Yes’ means
‘Entailed’. For the Image captioning task, we refer to 11 prompts from
LLaVa [llava](http://arxiv.org/pdf/2402.11690v1) to instruct the model to briefly describe
the image and randomly choose 1 prompt for each caption, such as "Human:
Provide a brief description of the given image. AI: {caption}".

Text Recognition is a basic ability for OCR-free Visuall-situated
Language Understanding. Therefore, we apply an auxiliary Text Reading
task to strengthen text recognition ability across different domains.
With the text and position information in the image, we organize the
texts in the common reading order: from top to down, from left to right.
Directly utilizing all texts as targets [donut](http://arxiv.org/pdf/2305.09520v1) will
result in the model focusing on generating the starting texts and
neglecting others to reduce the loss. Instead, we randomly choose a
split position $p$ from
$\{0, \frac{L}{6},\frac{2L}{6}, ...,\frac{5L}{6}\}$, where $L$ is the
text sequence length. The left part is used as the input and the right
one is the target. $p=0$ means to generate all texts while other cases
ask the model to continue reading following the input texts. Such a
design could enforce the model to read different parts of texts with the
context. Starting texts always convey key information about the image,
such as the chart title. Therefore, we apply a bigger sample rate (0.5)
for the ‘0’ position and 0.1 for other positions. To distinguish reading
from the beginning and continuing reading, we design two groups of
prompts and randomly choose 1 prompt for each sample. For example, an
instruction of reading from the beginning can be "Human: Recognize text
in the image. AI: {all texts}" and an instruction of continuing reading
can be "Human: The words on this picture are {left texts}. Continue
reading the text. AI: {right texts}".

Large Language Models learn strong understanding ability from the tough
language modeling task. Therefore, for stronger vision-and-language
semantic comprehension ability, we propose to design an auxiliary Key
Points Generation task, which requires the model to give some key points
about the image. To support this task, we collect QA pairs of each image
and convert them to declarative sentences with Vicuna
[vicuna](https://github.com/lm-sys/FastChat). These declarative sentences are finally
regarded as key points about the image. We also build a set of templates
to instruct this task, such as "Human: Identify some key points in this
picture. AI: {key points}".

All templates for Text Reading and Key Points Generation tasks can be
found in Appendix
<a href="#sec:appendix_template" data-reference-type="ref"
data-reference="sec:appendix_template">[sec:appendix_template]</a>.

## Instruction Data Resources

<div class="table*" markdown="1">

<div class="tabular" markdown="1">

p0.11p0.04p0.02\|p0.03p0.03p0.03p0.04\|p0.03p0.06\|p0.07\|p0.06p0.07\|p0.06
\***Model** & **Train** & **TS** & **Doc** & **Info** & **Deep** &
\***KLC** & \***WTQ** & \***TabFact** & \***ChartQA** & \***TextVQA** &
\***TextCaps** & **Visual**  
  & **Param** & **FT** & **VQA** & **VQA** & **Form** &   &   &   &   &
  &   & **MRC**  
Dessurt & 127M & & 63.2 & -& - & - & - & - & - & - & - & -  
Donut & 176M & & 67.5 & 11.6 & 61.6 & 30.0 & 18.8 & 54.6 &41.8 & 43.5 &
74.4 & 93.91  
Pix2Struct$_{base}$ & 282M & $\checkmark$ & 72.1 & 38.2 &- & - & - & - &
56.0 & -& 88.0 & -  
Pix2Struct$_{large}$ & 1.3B & $\checkmark$ & **76.6** & 40.0 & - & - & -
& - & 58.6 & -& 95.5 & -  
& 86M & $\times$ & <u>65.4</u> & **42.2** & <u>49.5</u> & **32.8** &
**29.4** & **67.6** & **59.3** & **57.6** & **118.4** & **221.7**  

</div>

</div>

DocVQA [docvqa](None) comprises 50k question and answer(QA)
paris on 12k document images from UCSF Industry Documents Library.
InfographicsVQA (InfoVQA) [infovqa](http://arxiv.org/pdf/2104.12756v2) collects 5k diverse
infographics from the internet and annotates 30k QA pairs.
DeepForm$^*$[^1] [deepform](http://arxiv.org/pdf/2303.13839v1) and Kleister Charity (KLC)
[klc](None) are two Information Extraction datasets.
DeepForm$^*$ contains 1.1k documents related to election spending. 2.7k
documents of KLC come from published reports of charity organizations.

WikiTableQuestions (WTQ$^*$) [wikitableqa](http://arxiv.org/pdf/2009.13845v2) comprises 2.1k
table images from Wikipedia and is annotated with 23k question and
answer pairs demanding comparison and arithmetic operations. TabFact$^*$
[TabFact](http://arxiv.org/pdf/2311.06592v1) is a Natural Language Inference dataset, which
contains 112k ‘entailed’ or ‘refuted’ statements about 16k Wikipedia
tables.

ChartQA [chartqa](None) collects various topics and types of
charts from four sources: Statista (statista.com), The Pew research
(pewresearch.org), OWID (ourworldindata.org) and OECD (oecd.org). It
totally contains 21k chart images and 32k QA pairs.

TextVQA [textvqa](None) filters 28k natural images with texts
from Open Images V3 [openimages](http://arxiv.org/pdf/1809.05929v7) and annotates 45k QA
pairs. To support image captioning with reading comprehension, TextCaps
[textcaps](None) further collects 145k captions based on
TextVQA.

VisualMRC [visualmrc](http://arxiv.org/pdf/2101.11272v2) collects 5k full screenshots of
webpages from 35 websites. There are 30k annotated QA pairs where
answers are expressed in fluent sentences (avg. 9.53 words) and much
longer than the ones of QA datasets mentioned above.

[^1]: Superscript $^*$ means the reformulated or modified version in
    DUE-benchmark [due](None)

# Experiments

## Implementation Details

We conduct experiments on a recently proposed MLLM named
mPLUG-Owl [mplugowl](http://arxiv.org/pdf/2405.00390v2) without modifying its
hyperparameters. The number of learnable queries of visual abstractor is
$65$. The dimension of hidden states $d_v$ and $d_l$ are 1024. For the
shape-adaptive cropping module, we set the maximum number of cells $N_c$
to 20 by default. During instruction tuning, the maximum sequence length
is limited to 2048, and $H_v, W_v$ are set to 224 to match the
pretrained resolution of the visual encoder. For LoRA, we set the rank
$r=8$. The learning rate schedule uses a linear warmup of 36 steps to
$1e^{-4}$, followed by cosine decay to 0. The batch size is set to 256.
For better convergence of each dataset, DocVQA is up-sampled 3 times,
InfoVQA, WTQ, DeepForm, and KLC are up-sampled 2 times. The total number
of training samples including Text Reading and Key Points Generation is
514,764. The instruction tuning process takes 16 A100 days for 20k
training steps (10 epochs).

## Evaluation

We use official training splits as tuning data and evaluate models on
test splits. Following previous works [due](None), [pix2struct](None),
DocVQA and InfoVQA are evaluated by ANLS [stvqa](http://arxiv.org/pdf/2304.01603v1),
DeepForm and KLC are evaluated by F1 score. WTQ, TabFact and TextVQA are
evaluated by accuracy. ChartQA is evaluated with the relaxed accuracy
[plotqa](http://arxiv.org/pdf/1906.04124v2). TextCaps and VisualMRC are measured by CIDEr
[cider](http://arxiv.org/pdf/2106.15553v1). Evaluation of TextVQA and TextCaps are performed
with the official challenge website.

<div class="table*" markdown="1">

<div class="tabular" markdown="1">

p0.02\|p0.02p0.02\|p0.04\<p0.04\<\|p0.05\<\|p0.04\<p0.06\<p0.07\<\|p0.06p0.04p0.06p0.06p0.06
  & & & **Doc** & &\***DocVQA** & \***WTQ** & \***ChartQA** &
\***TextVQA** & **Visual**  
  & KPG & TR & Abs & LoRA & **Data** &Global & CropPos & Crops &   &   &
  &   & **MRC**  
r1 & & & & & & & & 20 & 56.7 & 22.9 & 56.7 & 54.3 & 205.0  
r2 & & & & & & & & 20 & 64.3 & 28.1 & 58.6 & 56.0 & 213.5  
r3 & & & & & & & & 20 & 52.4 & 20.5 & 43.5 & 54.9 & 194.9  
r4 & & & & & & & & 20 & 59.5 & 23.5 & 58.5 & 53.3 & 177.0  
r5 & & & & & & & & 20 & 46.2 & 27.4 & 59.8 & 54.0 & 185.6  
r6 & & & & & & & & - & 22.0 & 13.4 & 24.2 & 34.4 & 157.4  
r7 & & & & & & & & 9 &58.0 & 24.7 & 58.9 & 55.5 & 215.3  
r8 & & & & & & & & 20 & 64.1 & 27.6 & **60.7** & 56.5 & 210.7  
r9 & & & & & & & & 20 & 62.8 & 26.7 & 58.7 & 55.4 & 181.1  
r10 & & & & & & & & 20 & **65.4** & **29.4** & 59.3 & **57.6** &
**221.7**  

</div>

</div>

## Main Results

We first compare  with state-of-the-art ocr-free models on 10 datasets.
For fair and consistent comparison across all datasets, we finetune the
strong and accessible baseline Dount on unreported datasets. As shown in
Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>,  achieves state-of-the-art
performance in 8 tasks across 5 domains, covering Visual Question
Answering, Information Extraction, Natural Language Inference and Image
Captioning tasks. With much fewer trainable parameters (86M vs 1.3B) and
without a specific finetuning stage,  outperforms the strong
pretriaining model Pix2Struct$_{large}$ in InfoVQA, ChartQA, and
TextCaps. Considering that Pix2Struct$_{large}$ is trained more than
170k steps with a batch size of 1024 on 128 TPUs, this validates that
with the help of open-domain Multimodal Large Language Models, learning
costs for universal visually-situated language understanding can be
greatly reduced. More detailed analysis can be found in
<a href="#sec:main_weak" data-reference-type="ref+Label"
data-reference="sec:main_weak">[sec:main_weak]</a>.

## Ablation Study

We perform comprehensive ablation experiments to validate the
contribution of two auxiliary tasks, trainable architectures,
cross-domain joint training and the design of shape-adaptive cropping
module.

#### Auxiliary Tasks.

As shown in Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a>, dropping the Key
Points Generation task (r10 vs r2) causes a performance decrease on all
domains of datasets, demonstrating that this task helps the model better
understand the vision-and-language semantic. Further removing the Text
Reading task (r2 vs r1) causes more significant performance degradation,
which validates the importance of enhancing text recognition ability
across different domains.

#### Trainable Architectures.

Both the visual abstractor and LoRA in LLM are finetuned in  (r10).
Freezing either the visual abstractor (r3) or LoRA (r4) causes
performance decrease, which demonstrates that both the vision and
language parts should be finetuned for adjusting to Visually-situated
Language Understanding.

<figure id="fig:cut_map">
<img src="/papers/vision_rich/arXiv-2310.05126v1_md/cut_map.png" style="width:100.0%" />
<figcaption>Visualization of the frequency of selected grid with
shape-adaptive cropping module. The cell at row <span
class="math inline"><em>i</em></span> and column <span
class="math inline"><em>j</em></span> denotes the selected frequency of
grid <span
class="math inline">(<em>n</em><sub><em>h</em></sub> = <em>i</em>, <em>n</em><sub><em>w</em></sub> = <em>j</em>)</span>.
Deeper colors represent higher selection frequencies.</figcaption>
</figure>

#### Cross-domain Joint Training.

After removing 4 document datasets from the training data,  achieves
worse performance (r10 vs r5) on the table, natural image, and webpage
domains, validating that images of different domains share some common
characteristics and cross-domain joint training improves the universal
performance. Besides, although trained without document data, our model
achieves a 46.2 score on the DocVQA dataset, showing the potential
out-of-domain understanding ability of our training paradigm.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/case.png" style="width:100.0%" />

</div>

#### Shape-adaptive Cropping.

The r6 in Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a> represents directly
tuning the mPLUG-Owl without any model revisions. With the
shape-adaptive cropping,  achieves significantly better performance (r7
vs r6), showing that our cropping module is indispensable to leverage
pretrained low-resolution vision encoder for universal visually-situated
language understanding. Besides, increasing the cropping numbers (r8 vs
r7) improves the model’s performance. Due to the resolution of each
local image being constant (224x224), more crops mean higher overall
resolution and therefore achieves better performance. Furthermore,
adding a resized global image bring a slight improvement in most
datasets (r10 vs r8), validating that a complete image could alleviate
possible information loss due to image cropping. Finally, dropping crop
position encoding also hurts the model’s performance (r10 vs r9),
proving the effectiveness of crop position encoding for correlating
local images.

For alleviating the distortion problem due to resizing, we propose to
crop images according to their raw aspect ratio.
<a href="#fig:cut_map" data-reference-type="ref+Label"
data-reference="fig:cut_map">1</a> shows the frequency distribution of
grids selected by our shape-adaptive cropping module on DocVQA,
VisualMRC and WikiTableQuestions (the distribution on more datasets can
be found in the Appendix
<a href="#sec:appendix_grid" data-reference-type="ref"
data-reference="sec:appendix_grid">[sec:appendix_grid]</a>). For
aesthetic purposes, we present the distribution with $N_c=9$.
Apparently, different domains of images have different shape
distributions. For most document images in DocVQA, their height is
greater than the width, while table images are the opposite. As webpages
are scrollable, their screenshots are always in the form of a long
rectangular shape. With the shape-adaptive cropping design, our model
can easily adapt to various image shapes without domain-specific
fine-tuning.

Text distortion may pose little influence on visual question answering
because they are always about partial text information. But it is
harmful for reading texts in the image because every text matters. For
quantitative analysis of the influence of shape-adaptive design, we
directly evaluate the performance of reading all texts. We choose the
Bleu [bleu](http://arxiv.org/pdf/2202.11027v1) as the metric because it directly measures
the n-gram overlap between the ground-truth and predicted text sequence.
The evaluation set is built by combining 100 randomly-selected test
images from each dataset. As shown in
<a href="#tab:shape-adap" data-reference-type="ref+Label"
data-reference="tab:shape-adap">1</a>, compared with cropping all images
with a fixed grid,  could better recognize texts in the image due to our
shape-adaptive design that alleviates the text distortion problem.

<div id="tab:shape-adap" markdown="1">

| **Model**  | **Bleu1** | **Bleu2** | **Bleu3** | **Bleu4** |
|:-----------|:----------|:----------|:----------|:----------|
|  w/o adapt | 21.4      | 15.4      | 12.0      | 9.7       |
|            | **24.9**  | **18.1**  | **14.3**  | **11.7**  |

The Text Reading performance of  under the condition of $N_c=9$. ‘w/o
adapt means removing the shape-adaptive design and cropping the image
with a fixed grid $3 \times 3$.

</div>

## Qualitative Results [sec:quali_analysis]

<a href="#fig:case" data-reference-type="ref+Label"
data-reference="fig:case">[fig:case]</a> show some qualitative results
produced by our  on different types of images.  could not only extract
information from the document (case a), but also understand different
instructions and provide corresponding answers by attending to different
regions (case b). Table understanding always involves layout
comprehension and statistics. As shown in case c, given a table image,
 could well relate different columns to answer the ‘first movie’ and
perform simple statistics about the ‘total number’. As for images with
multiple paragraphs of text, e.g. webpage screenshot in case e,  could
also locate the relevant paragraph, understand the texts and answer the
question accurately. Case d shows the text reading performance. With the
help of the Text Reading task,  is able to read texts from top left to
bottom right. But, due to the language decoding manner, when given an
image with rich texts, such as a page of a book, the model often reads
the beginning texts and then continues writing without watching the
image. More qualitative results can be found in
<a href="#sec:appendix_case" data-reference-type="ref+Label"
data-reference="sec:appendix_case">[sec:appendix_case]</a>. Finally, as
shown in case f,  is able to list some key points about the chart by
combining the title and line information. Listing key points in this
work is just a superficial attempt at open-ended generation, and its
performance is far from promising, e.g.,  makes a mistake about the
lowest line. More effort is needed towards a comprehensive understanding
of images with rich text.

# Conclusion

We first propose to leverage existing Multimodal Large Language Models
for universal ocr-free visually-situated language understanding through
low-cost instruction tuning. All downstream tasks are reorganized into a
unified instruction-tuning format. Besides, we design the Text Reading
task and Key Points Generation task to enhance text recognition and
vision-and-language semantic comprehension abilities. To utilize the
pre-trained vision encoder for processing high-resolution images, we
design a shape-adaptive cropping module, which cuts the image into
multiple local images considering its raw aspect ratio and resolution.
 achieve state-of-the-art ocr-free performance in 8 out of 10 datasets,
ranging from documents, tables, charts, and natural images to webpage
screenshots.

# Limitations [limitations]

Our experiments validate that  is able to correlate local images after
cropping a high-resolution image. However,  struggles to understand
multi-page documents (e.g. books and papers) due to lacking ability to
correlate different pages and the limited sequence length of the
decoder. Besides,  feeds an equal number of features for each local
image into the language decoder. But, not all local images contain rich
vision or text information. In the future, we will explore a more
efficient way to encode different crops. Furthermore, the open-ended
generation about Visually-situated Language understanding is far from
well studied. We try developing key points generation ability in this
work but more difficult generation tasks are not currently considered,
such as giving the chain-of-the-thought of the answer. How to simulate
such abilities through instruction tuning is a topic worth studying.
Finally, the Text Reading task helps the model recognize texts, but the
text reading performance with the LLM as the decoder is far from
satisfactory due to the hallucination problem. Instructing the LLM to
read texts strictly according to images is a challenging topic.

# Ethics Statement [ethics-statement]

Our  relies on multi-modal large language models that are trained on
large-scale image and text data from the web and therefore may be
subject to issues such as toxic language and
bias [bender2021dangers](http://arxiv.org/pdf/1705.07451v1). However, our model is further
fine-tuned on publicly available datasets and is used specifically in
the domain of visually-situated language understanding, where these
issues have minimal impact.

<div class="figure*" markdown="1">

<img src="/papers/vision_rich/arXiv-2310.05126v1_md/cut_map_full.png" style="width:100.0%" alt="image" />

</div>

# Grid Distribution on Downstream Datasets [sec:appendix_grid]

We visualize the frequency distribution of grids selected by our
shape-adaptive cropping module on all ten datasets in
<a href="#fig:cut_map_full" data-reference-type="ref+Label"
data-reference="fig:cut_map_full">[fig:cut_map_full]</a>. The wide
variety of image shapes in downstream tasks highlights the crucial role
of the shape-adaptive cropping module.

# Detailed Analysis on Performance [sec:main_weak]

## Underperforms Ocr-Free Baselines on DocVQA and DeepForm

It can be seen that underperforms ocr-free baselines on DocVQA and
DeepForm. There are two main factors: (1) Donut performs the pretraining
on large-scale document dataset IIT-CDIP (11M document images), which is
the same domain as DocVQA and DeepForm. But UReader does no have a
pretraining process and is just instruction finetuned on ensembled
datasets (less than 0.5M assorted images). Training with more document
images brings better performance. (2) The pretraining task of Pix2struct
is to predict the HTML dom tree of a masked web screenshot, which
requires the model to fully understand the layout information of the
image. But UReader is trained to read texts from top to down, from left
to right, which requires a weaker layout understanding ability. The
pretraining on layout understanding also leads to improved performance
on DocVQA.

The conclusion can also be substantiated by the observations on the
other two datasets (i.e., InfoVQA and KLC) included in the document
domain as previous work [udop](http://arxiv.org/pdf/2212.02623v3). For the InfoVQA dataset,
the image is poster style and the layout is not as important as DocVQA
and DeepForm but the relationship between text and vision objects
matters more, like natural image and chart image. As for the KLC
dataset, ocr-free models are only fed with the first page (always the
cover of a report) , where the layout is much simpler than DocVQA and
DeepForm. Therefore, can outperform baselines on these two document
datasets.

In summary, compared with ocr-free model Donut and Pix2Struct, due to
the pretrianing of MLMM on open-domain datasets, is better at
understanding cross-modality relationships in the image but weaker at
comprehending text layout information without large-scale document
pretraining and specific layout understanding tasks.

## Compared with Pipeline Methods

<div class="table*" markdown="1">

</div>

We list the performance of state-of-the-art pipeline models in
<a href="#tab:pipeline" data-reference-type="ref+Label"
data-reference="tab:pipeline">[tab:pipeline]</a>. We can summarize from
the results that there are two distinct aspects. Firstly, our model
achieves comparable or slightly worse results compared to the pipeline
methods on TextVQA, ChartQA, InfoVQA, TextCaps and TabFact. Secondly,
there is a obvious gap between our model and pipeline methods on DocVQA,
DeepForm, KLC, WTQ and VisualMRC.

For the first aspect, there are two reasons for the similarity
performance: (1) Modeling the diverse relationship between visual
objects and text presents challenges for both pipeline-based methods and
OCR-free methods. TextVQA, TextCaps and InfoVQA requires the relation
understanding between text and visual objects (i.e. logos, icons and
common objects). ChartQA asks for trend comprehension of lines.
Understanding such complex cross-modality relation is challenging for
both ocr-free and pipeline methods. (2) The simplicity of task formats
can reduces performance gaps. Tabfact is a simply binary classification
task resulting the small performance gap.

For this second aspect, the main performance gap appears in three
categories of datasets: document, table, and webpage screenshot. The
reasons are two folds: (1) The gap in terms of text recognition and
layout extraction. In document, table and website, text is the dominant
information source and the layout(e.g. row and column layout in table)
is relatively uniformer than the chart and natural images. Therefore,
with pre-extracted texts and layout information, it is more easy to
understand the image. But for OCR-Free models, such as our UReader and
Donut, it’s still challenging to fully recognize all texts. (2) The gap
in terms of modeling capacity on multi-page document input. for
multiple-page document datasets KLC (98% \> 4 pages) and DeepForm (75%
\> 1 pages), OCR-Free models only input the first page and lose much
information.

## Zero-shot Performance

We test the zero-shot performance of UReader on unseen dataset OCR-VQA.
With the same evaluation metrics, UReader outperforms mPLUG-Owl (41.1 vs
28.6) and a recent work UniDoc [Feng2023UniDocAU](https://api.semanticscholar.org/CorpusID:261065237) (41.1
vs 34.5) with the training of layout prediction. The results show that
the zero-shot performance of our method on unseen domains is acceptable.

# More Qualitative Results [sec:appendix_case]

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/appendix_case_natural.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/appendix_case_chart.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/appendix_case_table.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/appendix_case_doc.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/appendix_case_webpage.png" style="width:100.0%" />

</div>

## Downstream Results

More qualitative results on natural images, charts, tables, documents
and webpage screenshots are shown in Figure
<a href="#fig:natural_case" data-reference-type="ref"
data-reference="fig:natural_case">[fig:natural_case]</a>-<a href="#fig:web_case" data-reference-type="ref"
data-reference="fig:web_case">[fig:web_case]</a>.

<a href="#fig:web_case" data-reference-type="ref+Label"
data-reference="fig:web_case">[fig:web_case]</a> show a sample of Text
Reading and Visual Question Answering about a webpage screenshot from
VisualMRC. As mentioned in
<a href="#sec:quali_analysis" data-reference-type="ref+Label"
data-reference="sec:quali_analysis">[sec:quali_analysis]</a>, when given
an instruction about reading all texts in the image,  can read the
beginning texts but sometimes is easy to continue to generate
vision-irrelevant texts. With appropriate instructions,  could indeed
recognize texts in other regions, such as ‘exercise increases cellular
recycling’. Therefore, the hallucination problem during text reading is
not because  cannot recognize texts, but the generating manner of LLM
decoder. When beginning texts are read from the image, the decoder may
generate the following texts according to the closer text context rather
than the image.

## Open-domain Results

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2310.05126v1_md/open_domain_cases.png" style="width:90.0%" />

</div>

We present open-domain examples in
<a href="#fig:open_domain_case" data-reference-type="ref+Label"
data-reference="fig:open_domain_case">[fig:open_domain_case]</a>. We use
randomly collected images and freely ask questions to the model based on
the content of these images. The original mPLUG-Owl is used for
comparison.

In <a href="#fig:open_domain_case" data-reference-type="ref+Label"
data-reference="fig:open_domain_case">[fig:open_domain_case]</a> (a),
 is able to accurately recognize and answer questions about the small
text in natural images ("Name of passenger" and "MORRIS/KARLA"). In
contrast, mPLUG-Owl does not respond with the name in the first round
and gives an incorrect answer even with a prompt in the second round.

In <a href="#fig:open_domain_case" data-reference-type="ref+Label"
data-reference="fig:open_domain_case">[fig:open_domain_case]</a> (b), we
raise a query consisting of two cascaded questions, which requires the
model to simultaneously understand the spatial position of the
non-textual objects referred to by the query and locate the
corresponding fields. It can be seen that the  completes this task well,
while mPLUG-Owl answers incorrectly in both object recognition and price
extraction.

In <a href="#fig:open_domain_case" data-reference-type="ref+Label"
data-reference="fig:open_domain_case">[fig:open_domain_case]</a> (c), we
conduct multi-turn conversions with  on a screenshot. The questions
included references to the history of the conversation. Some questions
also require a certain amount of common sense. For example, the time of
account creation is equivalent to the time of joining Twitter, and the
inactive state of the Follow button indicates that the user has not
followed Iron Musk.  answers these questions well. mPLUG-Owl can
correctly recognize the Iron Mush, but is it prone to generating content
that is unrelated to the image, leading to some erroneous statements.

In <a href="#fig:open_domain_case" data-reference-type="ref+Label"
data-reference="fig:open_domain_case">[fig:open_domain_case]</a> (d), we
ask the  about the price and its components based on an image consisting
of multiple forms. Although  wrongly includes the header in the answer
and does not list the prices for each component, we notice that it
proactively filters out the components with a price of \$0, making the
answer more consistent with the user’s intention. It indicates that  can
find the form related to the question and comprehensively understand the
meaning of each field in the form. In contrast, mPLUG-Owl generates
responses that are full of illusions due to the loss of textual
information in the image.

These results reveal that  maintains some interactive ability of MLMM in
the open domain and shows stronger visually-situated language
understanding ability.

# Instruction Templates [sec:appendix_template]

The instruction templates of the auxiliary Text Reading and Key Points
Generation tasks are shown in
<a href="#tab:instruct_templates" data-reference-type="ref+Label"
data-reference="tab:instruct_templates">[tab:instruct_templates]</a>.

<div class="table*" markdown="1">

<div class="tabular" markdown="1">

c\|c\|l **Task** & **Part** &**Instruction Template**  
\*text reading from the beginning & \*- & \<Image\>Human: what words are
in the image? AI: {all texts}.  
  &   & \<Image\>Human: what texts are in the picture? AI: {all
texts}.  
  &   & \<Image\>Human: what does the image read? AI: {all texts}.  
  &   & \<Image\>Human: what does the picture say? AI: {all texts}.  
  &   & \<Image\>Human: what is written in the image? AI: {all texts}.  
  &   & \<Image\>Human: list the words in the image. AI: {all texts}.  
  &   & \<Image\>Human: list the texts in the picture. AI: {all
texts}.  
  &   & \<Image\>Human: Recognize text in the image. AI: {all texts}.  
  &   & \<Image\>Human: Identify text in the picture. AI: {all texts}.  
  &   & \<Image\>Human: Deciphering written content in the photo. AI:
{all texts}.  
  &   & \<Image\>Human: Extract words from the graphic. AI: {all
texts}.  
  &   & \<Image\>Human: Parse text from imagery. AI: {all texts}.  
  &   & \<Image\>Human: Read written language in the visuals. AI: {all
texts}.  
  &   & \<Image\>Human: Decode text from the snapshot. AI: {all
texts}.  
  &   & \<Image\>Human: Translate text in the picture. AI: {all
texts}.  
  &   & \<Image\>Human: Retrieve written information from the image. AI:
{all texts}.  
  &   & \<Image\>Human: Detect words in the photograph. AI: {all
texts}.  
\*continue reading & \*A & \<Image\>Human: The picture reads {left
texts}.  
  &   & \<Image\>Human: The image says {left texts}.  
  &   & \<Image\>Human: There are words {left texts} in the image.  
  &   & \<Image\>Human: Words {left texts} are in the picture.  
  &   & \<Image\>Human: The texts in this image read {left texts}.  
  &   & \<Image\>Human: The words on this picture are {left texts}.  
  &   & \<Image\>Human: The script depicted in this image reads {left
texts}.  
  &   & \<Image\>Human: The writing on this visual representation states
{left texts}.  
  &   & \<Image\>Human: The content presented in this diagram states
{left texts}.  
  &   & \<Image\>Human: The language used in this photograph says {left
texts}.  
  &   & \<Image\>Human: The inscription on this picture explain {left
texts}.  
  & \*B & Continue reading the text. AI: {right texts}.  
  &   & Read the following text. AI: {right texts}.  
  &   & Read the text behind. AI: {right texts}.  
  &   & What is the following text? AI: {right texts}.  
\*key points generation & \*- & \<Image\>Human: Identify some key points
in this picture. AI: {key points}.  
  &   & \<Image\>Human: Point out several critical features in this
image. AI: {key points}.  
  &   & \<Image\>Human: Highlight a few significant elements in this
photo. AI: {key points}.  
  &   & \<Image\>Human: Give some essential details in this
illustration. AI: {key points}.  
  &   & \<Image\>Human: Draw attention to some important aspects in this
diagram. AI: {key points}.  
  &   & \<Image\>Human: Mention a couple of crucial points in this
snapshot. AI: {key points}.  
  &   & \<Image\>Human: Indicate a few pertinent items in this graphic.
AI: {key points}.  
  &   & \<Image\>Human: Outline some significant characteristics in this
image. AI: {key points}.  
  &   & \<Image\>Human: Specify some key components in this picture. AI:
{key points}.  
  &   & \<Image\>Human: List a handful of essential elements in this
visual. AI: {key points}.  

</div>

</div>