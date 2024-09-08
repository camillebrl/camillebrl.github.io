# Introduction

Research on the interaction between language and vision has
traditionally focused on tasks where images and text can be separated
into distinct channels, e.g. visual question answering or image
captioning. However, *visually-situated language* is a far more
pervasive way in which these modalities interact and blend together. For
example, documents, tables, infographics, and user interfaces (UIs) are
intended to be consumed holistically, without clear boundaries between
textual and visual elements
(Figure <a href="#fig:tasks" data-reference-type="ref"
data-reference="fig:tasks">[fig:tasks]</a>). Comprehensive understanding
of this information requires a deep set of skills, including the ability
to recognize text, understand language, and incorporate diverse visual
context.

Previous work on understanding visually-situated language is scattered.
The focus is typically on complex task-specific combinations of
available inputs and tools. For example, document-understanding
models [layoutlmv3](None) rely on external OCR systems,
UI-understanding models rely on platform-specific metadata (e.g. Android
view hierarchy) [uibert](https://doi.org/10.24963/ijcai.2021/235), and diagram-understanding
models rely on diagram parses [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1).
Domain-specific engineering can be effective for high-resource settings
such as documents, where there is an abundance of tools and data
available. However, these pipelined models lack sharing of the
underlying data, model architectures, and objectives across domains,
limiting their general applicability. Moreover, relying on external
systems like OCR increases engineering complexity, limits adaptability,
and can increase overall computational cost. Recent work on OCR-free,
end-to-end document understanding from
images [donut](https://arxiv.org/abs/2111.15664), [dessurt](https://arxiv.org/abs/2203.16618) has attempted to remove such
task-specific engineering and reliance on external components during
inference by learning to decode OCR outputs during pretraining—a
significant step towards more general-purpose models. However, the focus
on text at the surface level limits the depth of knowledge transferred
from unsupervised data.

<div class="figure*" markdown="1">

<span id="fig:tasks" label="fig:tasks"></span>

</div>

We present `Pix2Struct`[^1], a pretrained model that combines the
simplicity of purely pixel-level inputs with the generality and
scalability provided by self-supervised pretraining from diverse and
abundant web data. Specifically, we propose a *screenshot parsing*
objective that requires predicting an HTML-based parse from a masked
screenshot of a web page. HTML provides clean signals about text,
images, and layouts, while the masked inputs encourage joint reasoning
about their co-occurrence. With the diversity and complexity of textual
and visual elements found on the web, `Pix2Struct` learns rich
representations of the underlying structure of web pages, which we show
can effectively transfer to a variety of downstream visual language
understanding tasks.

A key ingredient which enables this transfer is processing inputs
visually and holistically as they are intended for human readers. We
introduce variable-resolution inputs for vision transformers (ViT) that
prevent distortion of the original aspect ratio, which can vary greatly
across documents, figures, and UIs. During finetuning, we render other
inputs (e.g., questions in VQA and bounding boxes in UI tasks) onto the
image input for the task. In effect, we consume all our inputs through a
single modality, simplifying the modality combination problem in
previous work.

We train two variants with 282M and 1.3B parameters, which we refer to
as `Pix2Struct`-Base and `Pix2Struct`-Large respectively, on 80M
screenshots of web pages collected from the URLs in the C4
corpus [t5](http://jmlr.org/papers/v21/20-074.html)[^2]. Experiments on four domains and nine
tasks show that our finetuned models strongly outperform Donut (ranging
from 9 to 53 points), the strongest existing baseline without pipelines.
Compared with models with domain-specific pipelines, we lag behind the
state of the art in high-resource domains such as documents and natural
images but observe significant improvements (ranging from 1 to 44
points) in low-resource domains such as illustrations and UIs. We hope
these results encourage the community to continue developing such
general-purpose methods and further enable new applications in this
currently fragmented intersection of language and vision.

To summarize, our major contributions are as follows:

-   We introduce the area of general-purpose visually-situated language
    understanding, which consists of diverse tasks but common
    challenges.

-   We propose a *screenshot parsing* pretraining objective based on the
    HTML source of web pages. Our objective is shown to be more
    effective than prior attempts to enable the elegant pixel-to-text
    design for general-purpose visually-situated language understanding.

-   We introduce variable-resolution input representations to ViT and
    new fine-tuning strategies that seamlessly integrate language and
    vision inputs by directly rendering any text prompts on top of the
    input image.

# Method

## Background

Prior attempts at pixel-only modeling of visually situated language have
largely focused on documents and natural images. For documents,
Donut [donut](https://arxiv.org/abs/2111.15664) and Dessurt [dessurt](https://arxiv.org/abs/2203.16618)
combine pretrained objectives based on surface-level features from
synthetic images or predicted OCR outputs. For natural images, recent
work—GIT2 [wang2022git](http://arxiv.org/pdf/2204.07780v1) and
PaLI [pali](https://doi.org/10.48550/ARXIV.2209.06794)—focuses on collecting and training on large
scale image captioning data that transfers well to datasets with natural
images (e.g. TextCaps).

We aim to provide a single pretrained model that can be finetuned on a
wider variety of tasks and domains. The input to our model is an image
in the form of raw pixels only, and the output is text in the form of
token sequences, similar to Donut. The goal is a visual analog of models
like T5 [t5](http://jmlr.org/papers/v21/20-074.html), where the generality of simple inputs and
outputs is combined with the power of pretraining on large unsupervised
sources of data. During finetuning, the complexity of adapting to
diverse downstream tasks resides only in data preprocessing.

Even without visual context, pixel-only language modeling for text has
only recently been attempted [rust2022language](http://arxiv.org/pdf/2207.06991v2)—perhaps
because it requires solving multiple hard sub-problems. First, the
ability to read with high fidelity while also building rich high-level
representations poses a difficult optimization problem. Second, encoding
text-heavy inputs (e.g. long documents) involves processing
high-resolution images with variable aspect ratios. State-of-the-art
document understanding models [layoutlmv3](None) therefore
rely on the combination of (possibly noisy) OCR outputs with low
resolution images.

We show the components of `Pix2Struct` that address these challenges.
Section <a href="#sec:architecture" data-reference-type="ref"
data-reference="sec:architecture">2.2</a> discusses modifications to the
transformer inputs to handle variable aspect ratios and resolutions.
Section <a href="#sec:pretraining" data-reference-type="ref"
data-reference="sec:pretraining">2.3</a> details our proposed screenshot
parsing objective and
Section <a href="#sec:curriculum" data-reference-type="ref"
data-reference="sec:curriculum">2.4</a> describes curriculum learning
for more robust transfer learning. Finally,
Section <a href="#sec:finetuning" data-reference-type="ref"
data-reference="sec:finetuning">2.5</a> shows how `Pix2Struct` consumes
textual and visual inputs for downstream tasks (e.g. questions and
images) in the same space by rendering text inputs onto images.

## Architecture [sec:architecture]

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<embed src="/papers/visionrich_small_dec/figures/arXiv-2210.03347v2_md/input_rep.png" style="width:100.0%" />

</div>

</div>

`Pix2Struct` is an image-encoder-text-decoder based on
ViT [vit](http://arxiv.org/pdf/2105.15075v2). While the bulk of the model is fairly
standard, we propose one small but impactful change to the input
representation to make `Pix2Struct` more robust to various forms of
visually-situated language. Before extracting fixed-size patches, the
standard ViT scales the input images to a predefined resolution, which
creates two undesirable effects: (1) rescaling the image distorts the
true aspect ratio, which can be highly variable for documents, mobile
UIs, and figures. (2) transferring these models to downstream tasks with
higher resolution is
non-trivial [train-test-resolution](https://proceedings.neurips.cc/paper/2019/file/d03a857a23b5285736c4d55e0bb067c8-Paper.pdf), [simvlm](https://arxiv.org/abs/2108.10904), since the
model only observes one specific resolution during pretraining.

We instead propose to always scale our input image up or down such that
we extract the maximal number of fixed-size patches that fit within the
given sequence length
(Figure <a href="#fig:input_rep" data-reference-type="ref"
data-reference="fig:input_rep">[fig:input_rep]</a>). In order for the
model to handle variable resolutions unambiguously, we use 2-dimensional
absolute positional embeddings for the input patches. Together these
changes to the standard ViT inputs provide two major advantages in terms
of robustness to: (1) extreme aspect ratios, which is common in the
domains that we experiment with, and (2) on-the-fly changes to the
sequence length and resolution.

## Pretraining [sec:pretraining]

The goal of pretraining is for `Pix2Struct` to represent the underlying
structure of the input image. To that end, we create self-supervised
pairs of input images and target text from web pages. For each page in
the pretraining corpus, we start by collecting its HTML source and a
screenshot using a viewport of 1024 x 1024.

**Screenshot parsing inputs & outputs**   The screenshot and HTML are
modified to ensure rich and dense learning signal during pretraining.
These modifications provide a reasonable trade-off between preserving
the semantics of the page and requiring a practical decoder sequence
length.

We condense the HTML DOM tree by (1) only keeping nodes with *visible*
elements or descendants with visible elements and (2) if a node does not
contain visible elements and it only has a single child, replacing the
singleton child with any grandchildren to remove chained nesting. In
each node, we only use the text, along with filenames and alt-text of
images. Much more information could be retained (e.g. element tags,
style, titles and URLs) in future work. The decoder sequence length is
further reduced by finding the largest linearized subtree that fits
within a predefined sequence length. A bounding box indicating the
region covered by the chosen subtree is also drawn on the screenshot.

For better context modeling, we introduce a
BART-like [lewis-etal-2020-bart](https://doi.org/10.18653/v1/2020.acl-main.703) learning signal by
masking 50% of the text and decoding the entire subtree. The masked
regions are randomly sampled spans of text from the chosen subtree where
we render masks
(Figure <a href="#fig:screenshot_parsing_running" data-reference-type="ref"
data-reference="fig:screenshot_parsing_running">[fig:screenshot_parsing_running]</a>).

<div class="figure*" markdown="1">

:

$\rightarrow$

\<\<\<Python\> \</papers/visionrich_small_dec/figures/arXiv-2210.03347v2_md/img_src=py_logo.png img_alt=Python\>\> \<\<C++\>
\</papers/visionrich_small_dec/figures/arXiv-2210.03347v2_md/img_src=cpp_logo.png img_alt=C++\>\> \<\<Java\>
\</papers/visionrich_small_dec/figures/arXiv-2210.03347v2_md/img_src=java_logo.png img_alt=Java\>\> \<Submit\>\>

</div>

**Comparison to existing pretraining strategies**   Our proposed
screenshot parsing seamlessly integrates signals reminiscent of several
well-known pretraining strategies:

-   Recovering the unmasked parts of the parse is similar to OCR, a
    prerequisite skill for understanding language. OCR pretraining was
    proposed in Donut which uses synthetic renderings or OCR outputs. In
    Figure <a href="#fig:screenshot_parsing_running" data-reference-type="ref"
    data-reference="fig:screenshot_parsing_running">[fig:screenshot_parsing_running]</a>,
    predicting `<C++>` exemplifies this learning signal.

-   Recovering the masked parts of the parse is much like masked
    language modeling [bert](https://doi.org/10.18653/v1/N19-1423). A major difference is that
    the visual context often provides additional powerful cues. In
    Figure <a href="#fig:screenshot_parsing_running" data-reference-type="ref"
    data-reference="fig:screenshot_parsing_running">[fig:screenshot_parsing_running]</a>,
    predicting `<Python>` exemplifies this signal.

-   Recovering the alt-text from images is a common pretraining strategy
    for image
    captioning [conceptual-captions](https://doi.org/10.18653/v1/P18-1238), [wang2022git](http://arxiv.org/pdf/2204.07780v1), [pali](https://doi.org/10.48550/ARXIV.2209.06794).
    A major difference is that the model is permitted to use the web
    page as additional context. In
    Figure <a href="#fig:screenshot_parsing_running" data-reference-type="ref"
    data-reference="fig:screenshot_parsing_running">[fig:screenshot_parsing_running]</a>,
    predicting `img_alt=C++` exemplifies this learning signal.

Appendix <a href="#sec:pretraining_ex" data-reference-type="ref"
data-reference="sec:pretraining_ex">13</a> contains more details
including examples of screenshots paired with their gold and predicted
parses.

## Warming up with a reading curriculum [sec:curriculum]

While we can directly pretrain `Pix2Struct` on the screenshot parsing
task, we find that doing this naively can result in instability and slow
learning. However, if we first expose the model to a short “warmup”
stage of simply learning to read, we find a strong curriculum learning
effect where (1) pretraining is more stable and converges faster, and
(2) we observe better finetuning performance, as discussed in
Section <a href="#sec:ablations" data-reference-type="ref"
data-reference="sec:ablations">5</a>. We create images of text snippets
with random colors and fonts. The model is simply trained to decode the
original text (see
Appendix <a href="#sec:warmup_example" data-reference-type="ref"
data-reference="sec:warmup_example">12</a> for examples). This type of
curriculum learning was also used in Dessurt [dessurt](https://arxiv.org/abs/2203.16618)
and can also be viewed as a simplified version of Donut’s pretraining.

## Finetuning [sec:finetuning]

Finetuning  `Pix2Struct` is straightforward and largely a matter of
preprocessing the downstream data to unambiguously reflect the task in
the image inputs and text outputs, analogous to the way
T5 [t5](http://jmlr.org/papers/v21/20-074.html) is used for text-based tasks. In this section,
we cover the preprocessing strategies for the tasks described in
Table <a href="#tab:datasets" data-reference-type="ref"
data-reference="tab:datasets">[tab:datasets]</a>. Examples of this
preprocessing are shown in
Figure <a href="#fig:tasks" data-reference-type="ref"
data-reference="fig:tasks">[fig:tasks]</a>.

Captioning is the most straightforward, since the input image and the
output text can be directly used (as in TextCaps, Screen2Words). In the
case where the focus of the caption is a specific bounding box (as in
Widget Captioning), we draw the target bounding box on the image itself.

For visual question answering (as in OCR-VQA, ChartQA, DocVQA,
InfographicsVQA), while multimodal models typically reserve a
specialized text channel for the question, we opt to instead directly
render the question as a header at the top of the original
image. `Pix2Struct` reads both the question and the image jointly via
the visual modality. This strategy is analogous to the common practice
of simply concatenating all inputs during finetuning of pretrained text
models, first proposed in GPT [gpt](http://arxiv.org/pdf/2310.01427v1) and has been the
default method in NLP since then. Intuitively, this strategy is
effective because `Pix2Struct` has been pretrained to be sensitive to
long-range interactions between various parts of the input image. In the
case of multiple choice answers (as in AI2D), we also render the choices
in the header as part of the question.

The most complex scenario is RefExp, where the task is choosing between
UI components that a natural language expression could be referring to.
For each candidate, we create a training instance where the input image
contains the bounding box and referring expression, and the decoding
target is “true” or “false”. We sample five negative candidates per
positive candidate during training. During inference, we pick the
candidate for which the model generates “true” with the highest
score.[^3]

# Experimental Setup

## Benchmarks

We evaluate `Pix2Struct` on multiple benchmarks for visually-situated
language understanding across four domains: illustrations, user
interfaces, natural images, and documents. Since we are the first to
aggregate datasets with this scope, we optimized for diversity in
domains and in task-format. Evaluation is restricted to standard splits
without additional labeled data.
Table <a href="#tab:datasets" data-reference-type="ref"
data-reference="tab:datasets">[tab:datasets]</a> in
Appendix <a href="#sec:finetuning_datasets" data-reference-type="ref"
data-reference="sec:finetuning_datasets">10</a> provides a summary of
the datasets with details in
Section <a href="#sec:results" data-reference-type="ref"
data-reference="sec:results">4</a>.

We use evaluation metrics as defined in the original papers: (a) average
normalized Levenshtein similarity (ANLS) for DocVQA and InfographicVQA,
(b) exact match (EM) for AI2D, RefExp, and OCR-VQA, (c) relaxed accuracy
(RA) for ChartQA, and (d) CIDEr for the generation tasks.

## Implementation and Baselines

**Pretraining**  We pretrain two model variants: (a) a *base* model with
282M parameters including 12 encoder and 12 decoder layers with a hidden
size of 768, and (b) a *large* model with 1.3B parameters including 18
layers with a hidden size of 1536. Both models have the same warmup
stage using text rendered from BooksCorpus [books](http://arxiv.org/pdf/1506.06724v1)
lasting 30K steps with a maximum input sequence length of 128 patches.
The base model is then pretrained further for 270K steps with the
screenshot parsing objective using a batch size of 2048 on 64 Google
Cloud TPUs. The large model is pretrained for 170K steps with a batch
size of 1024 on 128 Google Cloud TPUs. Both models use an input sequence
length of 2048 patches and are optimized using
Adafactor [shazeer2018adafactor](http://arxiv.org/pdf/1604.06174v2). The learning rate
schedule uses a linear warmup of 1000 steps to 0.01, followed by cosine
decay to 0. The decoder sequence length is 128 tokens, and we choose
pretraining targets to have at most 1024 characters. As a reference
point, the base model reaches  30 BLEU and the large model reaches  32
BLEU on the pretraining validation set. Details about finetuning can be
found in Appendix <a href="#sec:hyperparams" data-reference-type="ref"
data-reference="sec:hyperparams">11</a>.

**Baselines**   Across all tasks, we found a large number of methods
which could serve as baselines. We compare `Pix2Struct`  against state
of the art (SotA) methods in each domain (see
Section <a href="#sec:results" data-reference-type="ref"
data-reference="sec:results">4</a> for method descriptions). Several
methods use model ensembles, multitask with labeled training data from
other datasets [powalski2021going](http://arxiv.org/pdf/2102.09550v3), [wang2022git](http://arxiv.org/pdf/2204.07780v1), or train
with validation data [li2021structurallm](https://doi.org/10.18653/v1/2021.acl-long.493). For fair
comparison and ease of experimentation, we focus on single-model and
single-task baselines trained on standard splits. Several (per-task)
SotA [li2021vut](http://arxiv.org/pdf/2107.13731v2), [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177) use
domain-specific inputs (e.g. view hierarchies for UIs or gold data
tables for charts) making it difficult to apply them to other domains.
For a strong, consistent visual baseline across domains, we finetuned
Donut on tasks where a purely visual baseline was unavailable.[^4]

# Results [sec:results]

Table <a href="#tab:main_res" data-reference-type="ref"
data-reference="tab:main_res">[tab:main_res]</a> compares
`Pix2Struct` with prior work.

<div class="table*" markdown="1">

<div class="tabular" markdown="1">

lllccccccccc & & & & & & & & & &  
& - & & & & & & & & &  
& GIT2 & & - & - & 70.3 & - & - & - & 145.0 & - & -  
& Donut & & 41.8 & 30.8 & 66.0 & - & 127.4 &   56.4 &   74.4 & 67.5 &
11.6  
&`Pix2Struct`  
& & & 56.0 & 40.9 & 69.4 & 92.2 & 133.1 & 107.0 &   88.0 & 72.1 & 38.2  
&    Large & & **58.6 & **42.1 & **71.3 & **94.2 & **136.7 & **109.4 &
  95.5 & 76.6 & 40.0  
************

</div>

</div>

## Illustrations

**ChartQA** [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177) is a VQA dataset
with questions based on charts, i.e. visual representations of tabular
data.[^5]. VisionTaPas [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177), the
current SotA, is a pipeline which operates on data tables predicted from
the given charts. It consists of (1) a ViT encoder for encoding the
chart image, (2) a TaPas encoder for encoding the question and the data
table, and (3) a cross-modal encoder. In contrast, `Pix2Struct` does not
rely on table extractors and uses the chart directly—improving the SotA
from 45.5 to 58.6 with the large variant.

**AI2D** [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1) contains multiple choice
questions based on illustrative science diagrams (about geological
processes, biological structures etc.). The dataset comes with train and
test splits. We set aside 1% of the train split for validation. The
current SotA DQA-NET [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1) focuses on
modeling entity relationships via a pipeline of tools for extracting
arrows, blobs, and other visual elements. `Pix2Struct`-Large outperforms
DQA-NET and Donut by 3.6 and 11.27 points respectively without any
domain-specific modifications.

**OCR-VQA** [mishra2019ocr](http://arxiv.org/pdf/2010.02582v1) is a VQA dataset on images
of book covers. The questions are based on book metadata such as title,
author, genre etc. Much of work on OCR-VQA, including the pipeline SotA
LATr [biten2022latr](http://arxiv.org/pdf/2309.17133v2), uses off-the-shelf OCR. Recent
work, GIT2 [wang2022git](http://arxiv.org/pdf/2204.07780v1), the current SotA, is
pretrained on 12.9B image caption pairs. Their final finetuning stage is
preceded by intermediate finetuning on eight VQA datasets including
VQAv2 [goyal2017making](http://arxiv.org/pdf/1612.00837v3),
VizWiz-VQA [chen2022grounding](http://arxiv.org/pdf/2202.01993v3), and
OCR-VQA [mishra2019ocr](http://arxiv.org/pdf/2010.02582v1) amongst others. Despite not
using more labeled training data, we outperform GIT2 by almost 1 point.

## UIs

**RefExp** [uibert](https://doi.org/10.24963/ijcai.2021/235) Given a natural language referring
expression, an app screenshot, and a set of components (via bounding
boxes on the screenshot), the goal is to retrieve the component that the
expression refers to. UIBert [uibert](https://doi.org/10.24963/ijcai.2021/235), the current SotA,
is pretrained on a combination of inputs from mobile apps including
screenshots, OCR text, and Android view hierarchies. Our models
substantially ourperform UI Bert by 1.4 and 3.4% absolute,
with `Pix2Struct`-Large setting the new SotA.

**Widget Captioning** [li-etal-2020-widget](https://doi.org/10.18653/v1/2020.emnlp-main.443) is an image
captioning task where the input is an app screenshot annotated with a
single bounding box denoting a widget (e.g. a button or a scroll bar).
The caption describes the functionality of the widget (e.g. *find
location*). VUT [li2021vut](http://arxiv.org/pdf/2107.13731v2), the current SotA uses a
specialized UI encoder combining images, bounding boxes, and view
hierarchies. `Pix2Struct`-Large improves the SotA CIDEr from 127.4 to
136.7.

**Screen2Words** [screen2words](https://doi.org/10.1145/3472749.3474765) is an image captioning
task where the input is an app screenshot and the caption describes the
functionality of the page (see
Figure <a href="#fig:tasks" data-reference-type="ref"
data-reference="fig:tasks">[fig:tasks]</a> for an example).
 `Pix2Struct`-Large improves the state of the art CIDEr from 64.3 to
109.4.

## Natural Images

**TextCaps** Recently, GIT2 (5.1B parameters) and PaLI (17B parameters)
have advanced the state of the art on TextCaps by pretraining on 10B+
image-caption pairs extracted from the web. PaLI (CIDEr 135.4) and GIT2
(CIDEr 145) show comparable performance without OCR inputs. PaLI
achieves SotA (CIDEr 160.4) performance when finetuned with OCR,
indicating that even for large-scale methods, end-to-end pixel-only
performance lags behind pipeline SotA. While their image
captioning-based pretraining understandably improves TextCaps, previous
work [donut](https://arxiv.org/abs/2111.15664) shows that captioning may not transfer to
other domains (e.g. documents). Moreover, screenshot parsing subsumes
signals from captioning
(Section <a href="#sec:pretraining" data-reference-type="ref"
data-reference="sec:pretraining">2.3</a>) while using a fraction of the
data used for pretraining GIT2 and PaLI. These results suggest that
 `Pix2Struct` could further benefit from scaling in pretraining data and
model size.

## Documents

**DocVQA** [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1) is a dataset of questions
about scanned documents,[^6] including typewritten, printed, handwritten
and born-digital text. `Pix2Struct`-Large outperforms Donut, the
previous visual SotA on DocVQA by 9 points. Top-performing single-task
methods like UDOP [tang2022unifying](http://arxiv.org/pdf/2212.02623v3) (ANLS 84.7)
typically use three components: (a) an off-the-shelf OCR system, (b)
pretrained text and image encoders, and (c) additional pretraining on
the IIT-CDIP scanned documents corpus. Despite using purely visual
representations and no in-domain pretraining data, `Pix2Struct` achieves
competitive performance (ANLS 76.6).

**InfographicVQA** [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2) is a
dataset of questions about infographics from the web. A unique challenge
of this dataset is its large images with extreme aspect ratios. Donut
scales images to a fixed aspect ratio, which we speculate is the cause
of its poor performance with an ANLS of 11.6. `Pix2Struct`-Large sets
the state of the art amongst visual models with an ANLS of 40.

For both DocVQA and InfographicVQA, text-only baselines are at or near
the state of the art. A T5-based model (T5 + 2D + U) with 2D positional
biases [borchmann2021due](http://arxiv.org/pdf/2111.08609v1) achieves ANLS of 81 on DocVQA
and 46.1 on InfographicVQA. This is in part due to the text-heavy nature
of the data (especially DocVQA) where visual context plays a lesser
role, and the more mature pretrained text-based encoders can do the
heavy lifting.

**Common trends**  Overall, `Pix2Struct` outperforms Donut in all tasks,
underscoring the effectiveness of our pretraining. We also advance the
single-task state of the art on six of nine benchmarks across four
domains. Scaling up from base to large results in considerable
improvements on all tasks despite the base model being trained for
3$\times$ more iterations than the large model. Previous
work [liu2019roberta](http://arxiv.org/pdf/1907.11692v1), [t5](http://jmlr.org/papers/v21/20-074.html) has shown that large batch
sizes and many training steps contribute greatly to the quality of the
pretrained model. Results indicate that further scaling up of
`Pix2Struct` is a promising direction.

# Analysis [sec:ablations]

<div id="tab:ablations" markdown="1">

|                         |      |       |       |
|:------------------------|-----:|------:|------:|
| Pretraining             |      |       |       |
| VQA                     |      |       |       |
| Captioning              |      |       |       |
|  Full                   | 67.8 | 137.5 | 84.2  |
|    – Warmup             | 56.2 | 128.0 | 71.7  |
|    – Masking            | 55.7 | 129.4 | 77.4  |
|    – Screenshot Parsing | 12.2 |  35.1 | 24.2  |

Ablations of pretraining components. Each ablation is a modification
with respect to the full model, while keeping the total number of
pretraining steps constant.

</div>

**Ablating pretraining objectives**  
Table <a href="#tab:ablations" data-reference-type="ref"
data-reference="tab:ablations">1</a> analyzes the importance of each
component of our pretraining recipe on DocVQA, Widget Captioning, and
TextCaps validation sets. The full pretraining method consists of a
warmup reading stage on the BooksCorpus followed by pretraining using
the screenshot parsing objective. For these experiments, we use the base
variant with a total of 100K steps of pretraining including 30K warmup
steps followed by 70K steps of screenshot parsing. The screenshot
parsing ablation removes the screenshot parsing stage altogether and
uses an extended warmup stage of 100K steps. The warmup ablation skips
the warmup stage and directly pretrains from random initialization for
100K steps. The masking ablation uses 30K steps warmup followed by 70K
steps of screenshot parsing without masking.[^7]

The biggest drop in performance comes from ablating the screenshot
parsing stage, effectively reducing the pretraining to reading linear
text. Ablating the warmup and masking is nearly equivalent on DocVQA and
Widget Captioning while the warmup is slightly more important in
TextCaps. Overall, our results seem to indicate that reading and
understanding visually-situated language is a complex problem involving
skills including recognizing text, understanding language, and
incorporating visual context.

**Ablating variable-resolution inputs**  
Figure <a href="#fig:aspect_ratio" data-reference-type="ref"
data-reference="fig:aspect_ratio">1</a> compares various ways to convert
input images into a constant number of patches. This ablation is
performed on the warmup stage
(Section <a href="#sec:curriculum" data-reference-type="ref"
data-reference="sec:curriculum">2.4</a>), where we measure full sequence
accuracy. The ‘padded’ variant maintains the original aspect ratio, but
introduces significant padding, which sacrifices the effective
resolution. The ‘stretched’ variant, typically used in ViT, introduces
no padding but distorts the original image. Our variable-resolution
inputs get the best of both worlds by maintaining the original aspect
ratio while maximally utilizing the budget specified by the sequence
length. Experiments in
Appendix <a href="#sec:resolution" data-reference-type="ref"
data-reference="sec:resolution">8</a> show that this benefit leads to
more effective learning, even for a task as simple as transcribing text
in the input image.

<figure id="fig:aspect_ratio">

<figcaption>Our variable-resolution inputs prevent aspect-ratio
distortion while minimizing padding. </figcaption>
</figure>

# Discussion

This section lays out some of the challenges in training general-purpose
visual language understanding models, and discuss a road map for future
work.

**Resolution**  Like Donut, we found that pretraining and finetuning
performance are extremely sensitive to the input resolutions.[^8] The
difficulty in using high-resolution images has been a bottleneck for
pixel-only models since higher resolutions often lead to longer sequence
lengths. This bottleneck has in part been responsible for the dominance
of OCR-based pipelines which are able to use lower image resolutions due
to a dedicated text encoder.[^9] However, steady progress with Donut
and `Pix2Struct` combined with recent progress in long range
transformers [press2021train](https://openreview.net/forum?id=R8sQPpGCv0) provides hope that
pixel-only models will bridge the gap with OCR-based pipelines.

**The visual web**  As a first attempt towards a general-purpose visual
language understanding model, we focused on simplicity both in terms of
how we use the HTML source and our choice for the pretraining corpus,
C4—a known public corpus used in previous work [t5](http://jmlr.org/papers/v21/20-074.html) that
is significantly smaller and narrower than corpora used to train the
largest language models today. However, web data includes even richer
multimodal signals such as videos and interactions. We posit that future
versions of general-purpose visual language understanding models will
benefit from better data curation. This opportunity also comes with a
caveat: just like text-based models, we must be careful of harmful
content on the web, which multimodal models would also be sensitive to.

**Generality**  While we have focused on general pixel-only models, we
do acknowledge that using OCR-pipelines or metadata can be appropriate
or even necessary in certain domains. For NLP, the scaling of pretrained
text based models has led to not only simpler model architectures and
preprocessing, but also emergent abilities on newer tasks which were
hitherto considered far too difficult [wei2022emergent](https://openreview.net/forum?id=yzkSU5zdwD).
A general-purpose model may also enable broader applications for visual
language, e.g. filling in missing accessibility
annotations [zhang2021screen](http://arxiv.org/pdf/2101.04893v1). Finally, given that the
overwhelming majority of prior work has leveraged OCR-based features, it
seems necessary to advance OCR-free alternatives (as this paper does) in
order to enable a clearer longer-term understanding around the proper
role for OCR. The broader objective of this work is to bring pretraining
for visually-situated language understanding a step closer to text-based
counterparts and pave the way for similar benefits from data and model
scaling.

# Related Work

To the best of our knowledge, no prior work has pretrained and evaluated
a visually-situated language understanding model on tasks spanning all
four domains of documents, illustrations, user interfaces, and natural
images. [^10] We build on prior work primarily focused on a single
domain and briefly highlight the similarities as well as the points of
departure with respect to such work here.

**Document understanding**  State-of-the-art models in this domain are
based on a pipeline of an external OCR system and a model that combines
images and OCR
annotations [docformer](None), [powalski2021going](http://arxiv.org/pdf/2102.09550v3), [layoutlmv2](http://arxiv.org/pdf/2310.16527v1),
*inter alia*. Prominent representatives are
LayoutLMv3 [layoutlmv3](None), which uses a simplified
transformer-based architecture and losses that encourage patch–OCR
alignment. TILT [powalski2021going](http://arxiv.org/pdf/2102.09550v3) pretrains a text
decoder and an image + OCR-output encoder followed by intermediate
finetuning on multiple QA tasks. `Pix2Struct` is more closely related to
Donut and Dessurt [dessurt](https://arxiv.org/abs/2203.16618), both image-to-text models
without OCR at inference time; the main difference stems from our more
powerful pretraining task from ground truth structures and resolution
flexibility enabling transfer to a variety of visual language domains.

**UI understanding**  Models in this group have focused solely on the UI
domain using pretraining data from mobile and web apps. While some
models use image-only
inputs [Liu2018LearningDS](http://arxiv.org/pdf/2309.10328v1), [Chen2020UnblindYA](http://arxiv.org/pdf/2003.00380v2), higher
accuracy approaches tend to benefit from often-noisy structures of view
hierarchies [li-etal-2020-mapping](https://doi.org/10.18653/v1/2020.acl-main.729) and element
annotations, e.g. UIBert [uibert](https://doi.org/10.24963/ijcai.2021/235),
ActionBert [actionbert](http://arxiv.org/pdf/2402.07938v2),
VUT [li2021vut](http://arxiv.org/pdf/2107.13731v2). One exception is concurrent
work [li2023spotlight](https://openreview.net/forum?id=9yE2xEj0BH7) which achieves comparable
performance with image-only inputs. The screen parsing
task [wu2021screen](None), while similar in name, is an
amalgamation of pipelines over domain-specific structures that are not
intended to produce transferable representations.

**Natural image understanding**  Pix2Seq uses the image-to-text
architecture for core vision tasks such as object detection and instance
segmentation [chen2022unified](http://arxiv.org/pdf/2206.07669v2), [chen2021pix2seq](http://arxiv.org/pdf/2305.18279v1).
Additionally, a variety of model
architectures [singh2019towards](http://arxiv.org/pdf/1811.11903v1), [sidorov2019textcaps](http://arxiv.org/pdf/1709.08299v2), [wang2020multimodal](http://arxiv.org/pdf/2108.02059v1)
and objectives [yang2021tap](http://arxiv.org/pdf/2311.01038v2) have been proposed for
understanding natural images containing short segments of text (e.g.
street signs). The predominant source of pretraining data has been
image-caption pairs often in conjunction with the output of
OCR [pali](https://doi.org/10.48550/ARXIV.2209.06794), [yang2021tap](http://arxiv.org/pdf/2311.01038v2).
GIT2 [wang2022git](http://arxiv.org/pdf/2204.07780v1), the pixel-only SoTA, learns from
12.9 billion image-caption pairs and is about 4 times larger than
`Pix2Struct`— it outperforms our model significantly on natural images
(TextCaps) but underperforms on illustrations (OCR-VQA). PaLI benefits
from using a pipeline with OCR, obtaining higher performance on
TextCaps. These methods have not been evaluated on more text-dense input
domains.

**Illustrations**  Models for illustrations have not been fully
pretrained on large scale data, perhaps because such data is not readily
available. Some components of such models, e.g. T5 and
TaPas [eisenschlos-etal-2020-understanding](https://doi.org/10.18653/v1/2020.findings-emnlp.27) used in the
VL-T5 and VisionTaPas models of
 [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177) or LATr’s OCR output
encoder [biten2022latr](http://arxiv.org/pdf/2309.17133v2) have been pretrained on
digital-born or OCR-ed documents. Our approach outperforms current SotA
models, without relying on other intermediate structures.

**Models learning from markup
structure**  MarkupLM [li2021markuplm](https://doi.org/10.18653/v1/2022.acl-long.420) and
Webformer [wang2022webformer](http://arxiv.org/pdf/2202.00217v1) learn encoders of HTML
from web pages. HTLM [aghajanyan2021htlm](https://openreview.net/forum?id=P-pPW1nxf1r) and
CM3 [aghajanyan2022cm3](http://arxiv.org/pdf/2201.07520v1) are generative models of
simplified HTML to enable zero-shot prompting with text and natural
images. Im2Tex [deng2017image](http://arxiv.org/pdf/1709.06308v1) is conceptually the most
relevant in showing that a pixel-only parser can be learned from
freely-available pairs of markup and renders, but doesn’t focus on
transferring this signal to wider applications.

**Datasets**  We have selected datasets representing challenges in
visually-situated language understanding in a variety of domains, but
our selection is not aimed to be exhaustive. The DUE
benchmark [borchmann2021due](http://arxiv.org/pdf/2111.08609v1) focuses on a more limited
domain of visual document understanding (e.g. excluding natural images
and UIs), but integrates a more comprehensive set of tasks within the
document understanding domain.

# Resolution in visually-situated language understanding tasks [sec:resolution]

<div class="figure*" markdown="1">

</div>

Previous methods rescale input images to fixed resolutions, which can
introduce severe aspect ratio distortions for inputs such as webpages
and documents. In contrast, we prevent aspect ratio distortion by
rescaling input images up or down such that we extract the maximal
number of patches that fit within the given sequence length
(Figure <a href="#fig:input_rep" data-reference-type="ref"
data-reference="fig:input_rep">[fig:input_rep]</a>).

Figure <a href="#fig:resolution" data-reference-type="ref"
data-reference="fig:resolution">[fig:resolution]</a> gives an overview
of the importance of input resolutions in visually-situated language
understanding tasks. Though `Pix2Struct` is more efficient at making use
of the input resolution, both `Pix2Struct` and Donut require high
resolutions to perform well on DocVQA (note the log scale). For example,
we only see significantly diminishing returns after about 1M pixels
(4096 patches of $16\times16$ pixels for `Pix2Struct` and
$1024\times1024$ for fixed-resolution models). However, ViT models
typically pretrain with resolutions of $224\times224$ and finetune with
up to $512\times512$. This is a subtle but critical detail that makes
using standard ViT out of the box suboptimal.

On the right of
Figure <a href="#fig:resolution" data-reference-type="ref"
data-reference="fig:resolution">[fig:resolution]</a>, we also present
example inference speeds on a v3-8 Cloud TPU when performing inference
on DocVQA. At full resolution (4096 sequence length or 1M pixels), the
base model processes 62 documents per second, and the large model
processes 20 documents per second.

# Full Results [sec:full_results]

<div class="table*" markdown="1">

|  |  |  |  |  |  |  |  |  |  |  |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Method |  |  |  |  |  |  |  |  |  |  |
| QA |  |  |  |  |  |  |  |  |  |  |
| VQA |  |  |  |  |  |  |  |  |  |  |
| Exp |  |  |  |  |  |  |  |  |  |  |
| Cap |  |  |  |  |  |  |  |  |  |  |
| Words |  |  |  |  |  |  |  |  |  |  |
| Caps |  |  |  |  |  |  |  |  |  |  |
| VQA |  |  |  |  |  |  |  |  |  |  |
| VQA |  |  |  |  |  |  |  |  |  |  |
|  | TILT | \- | \- | \- | \- | \- | \- | \- | 87.1 | -  |
|  | VUT | \- | \- | \- | \- | 94.8 | 64.3 | \- | \- | -  |
|  | TAP | \- | \- | \- | \- | \- | \- | 99.5 | \- | -  |
|  | LATr | \- | \- | 67.5 | \- | \- | \- | \- | \- | -  |
|  | PLC | \- | \- | \- | \- | 97.0 | \- | \- | \- | -  |
|  |  | \- | \- | \- | \- | \- | \- | \- | 81.0 | 46.1  |
|  | RoBERTa | \- | \- | \- | \- | \- | \- | \- | 69.5 | -  |
|  | LayoutLMv3 | \- | \- | \- | \- | \- | \- | \- | 83.4 | -  |
|  | DQA-NET | \- | 38.5 | \- | \- | \- | \- | \- | \- | -  |
|  | UI Bert | \- | \- | \- | 90.8 | \- | \- | \- | \- | -  |
|  | M4C | \- | \- | 63.9 | \- | \- | \- | 81 | \- | 14.7  |
|  | VisionTaPas | 45.5 | \- | \- | \- | \- | \- | \- | \- | -  |
|  | PaLI | \- | \- | \- | \- | \- | \- | **160.4** | \- | -  |
|  | UDOP | \- | \- | \- | \- | \- | \- | \- | **84.7** | **47.4 ** |
|  | GIT2 | \- | \- | 70.3 | \- | \- | \- | 145.0 | \- | -  |
|  | Donut | 41.8 | 30.8 | 66.0 | \- | 127.4 | 56.4 | 74.4 | 67.5 | 11.6  |
|  | `Pix2Struct`-Base | 56.0 | 40.9 | 69.4 | 92.2 | 133.1 | 107.0 | 88.0 | 72.1 | 38.2  |
|  | `Pix2Struct`-Large | **58.6** | **42.1** | **71.3** | **94.2** | **136.7** | **109.4** | 95.5 | 76.6 | 40.0  |

</div>

Table <a href="#tab:full_res" data-reference-type="ref"
data-reference="tab:full_res">[tab:full_res]</a> reports full results
for pipeline and pixel-only methods. For fair comparison and ease of
experimentation, we focus on single-model and single-task baselines
trained on standard splits. Several (per-task)
SotA [li2021vut](http://arxiv.org/pdf/2107.13731v2), [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177) use
domain-specific inputs (e.g. view hierarchies for UIs or gold data
tables for charts) making it difficult to apply them to other domains.

<div class="table*" markdown="1">

| Dataset | Domain | Description  |
|:---|:---|:---|
| OCR-VQA | Illustrations | VQA over book covers.  |
| ChartQA | Illustrations | VQA over charts (visualization of tabular data) |
| AI2D | Illustrations | VQA over science diagrams |
| RefExp | UIs | Detect UI component matching a natural language query  |
| Widget Captioning | UIs | Captioning a UI component on a screen  |
| Screen2Words | UIs | Captioning a UI screen to describe functionality |
| TextCaps | Natural images | Captioning of natural images containing text |
| DocVQA | Documents | VQA over scanned documents.  |
| InfographicsVQA | Documents | VQA over high-res infographics. |

</div>

# Finetuning Dataset Details [sec:finetuning_datasets]

Table <a href="#tab:datasets" data-reference-type="ref"
data-reference="tab:datasets">[tab:datasets]</a> show the datasets in
our benchmark for visually-situated language understanding.

# Hyperparameters [sec:hyperparams]

The base and large models are finetuned with an input sequence length of
4096 and 3072 respectively, except the base model on InfographicVQA
which benefits from a longer sequence length of 6144. We cannot use a
longer sequence length for the large variant due to TPU/GPU memory
constraints. We finetune for 5000 or 10000 steps with a batch size of
32, 128, or 256, with hyperparameter tuning and early stopping based on
the validation set.
Table <a href="#tab:hyperparams" data-reference-type="ref"
data-reference="tab:hyperparams">[tab:hyperparams]</a> contains
hyperparameter values for all tasks.

<div class="table*" markdown="1">

| Dataset        |    Base |       |       |     |   Large |       |         |
|:---------------|--------:|------:|------:|:---:|--------:|------:|--------:|
| 2-4            | Seq Len | Batch | Steps |     | Seq Len | Batch |   Steps |
| DocVQA         |    4096 |   256 | 10000 |     |    3072 |   128 | 10000   |
| InfographicVQA |    6144 |    64 | 10000 |     |    3072 |   128 | 10000   |
| AI2D           |    4096 |    32 |  5000 |     |    3072 |    32 |  5000   |
| ChartQA        |    4096 |   256 | 10000 |     |    3072 |   128 | 10000   |
| OCR-VQA        |    4096 |   256 | 10000 |     |    3072 |   128 | 10000   |
| RefExp         |    4096 |   256 | 10000 |     |    3072 |   128 | 10000   |
| Screen2Words   |    4096 |    32 | 10000 |     |    3072 |    32 | 10000   |
| Widget Cap.    |    4096 |   256 |  5000 |     |    3072 |   128 |  5000   |
| TextCaps       |    4096 |   256 |  5000 |     |    3072 |   128 |  5000   |

</div>

# Warmup Stage Data [sec:warmup_example]

<div class="figure*" markdown="1">

$\rightarrow$

The elves, it seemed, were possessed of some mysterious power over the
arts; without eve

</div>

For the warmup stage, we create images of text snippets from the
BooksCorpus [books](http://arxiv.org/pdf/1506.06724v1) with random colors (uniformly sampled
from all possible RGB values), fonts (uniformly sampled from all
possible Google Fonts [^11]), and font sizes (uniformly sampled from
12pt to 36pt) on a white background. The text snippets are up to 128
bytes long. The width of the images are 640 pixels, and the text is
wrapped of it exceeds the width of the image. The height of the image is
fit to the content height. The text is unmasked as this stage is
intended purely as a learning-to-read task.

Exposing the model to a short “warmup” stage of simply learning to read,
results in a strong curriculum learning effect where (1) pretraining is
more stable and converges faster, and (2) we observe better finetuning
performance. Figure <a href="#fig:warmup" data-reference-type="ref"
data-reference="fig:warmup">[fig:warmup]</a> shows an example of
rendered text from the BooksCorpus with its “parse”.

# Pretraining Data [sec:pretraining_ex]

The pretraining data is constructed from URLs in the C4 corpus. We
collect 80M (about one third of the total number of documents) pairs of
screenshots paired with their HTML source. The screenshots have a width
of 1024 pixels, and the height of the image is fit to the content
height.

The figures below show screenshots of our pretraining data along with
ground-truth and predicted parses.

<div class="figure*" markdown="1">

<div class="tcolorbox" markdown="1">

#### Ground-truth Parse

    <<<<CrossFit Thunderhawk | Rio Rancho>
       <dedicated to promote healthy kids and teens in Rio Rancho, NM>>
      <<Home> <About> <Schedule> <Media> <Blog> <Contact Us> <Free Class>>>
     <<Drop-ins>
      <Bring your child in for a drop-in to get a WOD in!>>
     <<<If you are visiting from out of town or traveling for club sports,
        make sure your child’s routine is not disrupted. Bring them in for
        a drop in to get a WOD in!>
       <<1-day CrossFit Athlete $15>
        <1-day Competitor $25>>>
      <<Become A Member>
       <We’d love to meet you and show you around.>>>>

</div>

<div class="tcolorbox" markdown="1">

#### Predicted Parse

    <<<<img_src=thunderhawk-logo-white img_alt=Thunderhawk Sports & Fitness>
       <Thunderhawk Sports & Fitness>>
      <<Home> <About> <Programs> <Team> <Blog> <Contact Us> <Get Started>>>
     <<<Drop-Ins>
       <Bring your child in for a drop-in to get a workout>>
      <<<If you are visiting from out of town or traveling for club sports,
         make sure your child’s routine is not disrupted. Bring them to our
         drop-in for a full session!> <<1:1 drop-in for

</div>

</div>

<div class="figure*" markdown="1">

<div class="tcolorbox" markdown="1">

#### Ground-truth Parse

    <, I tried something Valentine's themed. If you'd like to help
    raise money for fighting children's cancer you can follow the link right
    above and help out, too. As inspiration for this semi-homemade recipe, 
    I looked at the two recipes on the bag of sweet dough, I got an idea and 
    today I'm going to share with you how that worked out.
    \xa0 I got the bag of Sweet Dough using a coupon for a free product
    that was sent to my by Rhodes BakeNServ in exchange for testing out
    their products and sharing the results with all of you; no other form of 
    compensation was received.>

</div>

<div class="tcolorbox" markdown="1">

#### Predicted Parse

    <, I tried something Valentine's themed. If you'd like to help
    out, I think you'd go right ahead and do a post. Click on the link right
    above and help out, too. As inspiration for this semi-homemade recipe, 
    I've shared up two recipes on the bag of sweet dough. I got an idea and 
    today I'm going to share with you the second one. 
    Thank you for any of the amazing baking ideas plus this free product 
    that was sent to my by Rhodes BakeNServ in exchange for testing. 
    I'm really excited and sharing this recipe with all of you

</div>

</div>

<div class="figure*" markdown="1">

<div class="tcolorbox" markdown="1">

#### Ground-truth Parse

    <<<100% FEMALE 100% UV PROTECTION SINCE 1999>
      <FAST FREE SHIPPING>> 
     <img_alt=Velvet Eyewear> 
     <<<<Fringe Benefits>
        <<Posted by> <Lindsay Sperin> <on> <August 19, 2016>>> 
       <<img_src=img>
        <Fall is undeniably the best season for fashion
         for a multitude of reasons.> 
        <img_src=img>>>
      <<NEWS> 
       <<Polarized vs. UV Protection - What's The Difference?>
        <What's Hot in The Hamptons>>>> 
     <<img_src=en-us img_alt=en> <English>>>

</div>

<div class="tcolorbox" markdown="1">

#### Predicted Parse

    <<<10% OFF YOUR FIRST ORDER WITH CODE: FIRST10>
      <FAST FREE SHIPPING>> 
     <img_alt=Velvet>
     <<<<Fringe Benefits>
        <<Posted by> <Velvet Fashion> <on> <October 1, 2018>>>
       <<Fall is undeniably the best season for fashion 
         for a multitude of reasons.> 
        <img_alt=Fringe Benefits>>>
      <<Search> 
       <<Polarized vs. UV Protection: Velvet's Best Sunscreen> 
        <The Best Sunblock Sunscreen>>>>>

</div>

</div>

<div class="figure*" markdown="1">

<div class="tcolorbox" markdown="1">

#### Ground-truth Parse

    <<Menu>
     <img_src=ftg_webheader>
     <<<Spin-Off Games>
       <<Fairytale Games is a growing universe. Because of this, we have and
       will continue to grow spin-off games that utilize characters,
       storylines, and even poke fun of our games. Keep checking back and
       you just might be surprised at what you see!>
        <<Rumplestiltskin!>
         <Super Fairytale Fighters 2>>
        <<<Share this:>
          <<Twitter> <Facebook>>>
         <Loading...>>>> 
      <<Leave a Reply> 
       <<<Your email address will not be published.> 
         <<Required fields are marked> <*>>>
        <<Comment> <*>>>>>>

</div>

<div class="tcolorbox" markdown="1">

#### Predicted Parse

    <<Menu>
     <img_src=cropped-blogheader>
     <<<Fairytale Games>
       <<Fairytale Games is a growing universe. Because of this, we are
         excited to continue to grow spin-off games that utilize characters,
         storylines, and even poke fun of our games. Keep checking back and
         you just might be surprised at what you see!> 
        <<Fairytale Games>
         <Fairytale Games on Steam>>
        <<<Share this:>
          <<Twitter> <Facebook>>>
         <Loading...>>>> 
      <<Leave a Reply> 
       <<<Your email address will not be published.> 
         <<Required fields are marked

</div>

</div>

<div class="figure*" markdown="1">

<div class="tcolorbox" markdown="1">

#### Ground-truth Parse

    <<<<Coronavirus Update! We are open and ready to help you.>
       <We are conducting most of our appointments via phone to help prevent 
        the spread of the virus.>>
      <Chapter 13 Coronavirus Update>> 
     <<img_src=Logoo img_alt=Stamps & Stamps Attorneys At Law>
      <img_src=Phone>
      <Contact for a free Initial Consultation> 
      <<Call Us> <(937) 247-6447>> 
      <<Text Us> <(937) 265-6418>>> 
     <<Home> <About> <Articles> <Videos> <Testimonials> <Tax Relief> <News> 
      <Podcasts> <Rate Us> <Contact>> 
     <<We can provide the guidance you need to get through stressful family> 
      <disputes with your rights and interests intact.>> 
     <<<img_src=Bankruptcy img_alt=Bankruptcy Overview>
       <<Bankruptcy> <Overview>>>
      <img_src=Criminal-Defense1
       img_alt=Criminal Defense & Traffic Offenses>>>

</div>

<div class="tcolorbox" markdown="1">

#### Predicted Parse

    <<<<Coronavirus Update! We are open and ready to help you.> 
       <We are conducting most of our appointments via phone to help prevent 
        the spread of infection.>> 
      <CLICK HERE FOR MORE INFO>> 
     <<img_src=logo img_alt=Stamps & Stamps Attorneys At Law> 
      <img_src=phone>
      <<<Call Us> <(904) 222-2222>> 
       <<Text Us> <(904) 222-2222>>>>
     <<Home> <About> <Articles> <

</div>

</div>

[^1]: For pretrained checkpoints and code, see
    <https://github.com/google-research/pix2struct>.

[^2]: We do not use the released text in C4. The web page content and
    screenshots were crawled directly from the URLs.

[^3]: or lowest score if something other than “true” was generated

[^4]: Except RefExp due to the complexity inference.

[^5]: We evaluate on the task without the gold data table.

[^6]: from the UCSF Industry Documents Library
    <https://www.industrydocuments.ucsf.edu>

[^7]: All models use the same hyperparameters.

[^8]: See Appendix <a href="#sec:resolution" data-reference-type="ref"
    data-reference="sec:resolution">8</a> for a concrete comparison.

[^9]: OCR pipelines, while noisy, often result in manageable sequence
    lengths for large-scale text encoders.

[^10]: Some prior approaches have been evaluated on two domains.

[^11]: <https://developers.google.com/fonts>

<figure>
<p><strong>Screenshot Parsing Pretraining</strong></p>
</figure>

<figure>
<p><strong>AI2D</strong></p>
</figure>

<figure>
<p><strong>Screen2Words</strong></p>
</figure>

<figure>
<p><strong>DocVQA</strong></p>
</figure>

<figure>

</figure>

<figure>

</figure>

<figure>

</figure>

<figure>

</figure>

  

<figure>

</figure>

<figure>
<div class="snugshade*">
<pre><code>&lt;&lt;Pro&gt;
 &lt;&lt;&lt;$15&gt; &lt;/mo&gt;&gt;
  &lt;&lt;20 users included&gt;
   &lt;10 GB of storage&gt; 
   &lt;Priority email support&gt;
   &lt;Help center access&gt;&gt;
  &lt;Get started&gt;&gt;&gt;</code></pre>
</div>
</figure>

<figure>
<div class="snugshade*">
<pre><code>carnivore





</code></pre>
</div>
</figure>

<figure>
<div class="snugshade*">
<pre><code>list of videos
for weather
reports in
different
locations

</code></pre>
</div>
</figure>

<figure>
<div class="snugshade*">
<pre><code>Fred LeCrone





</code></pre>
</div>
</figure>