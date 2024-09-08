# Introduction

Building intelligent agents that can read and comprehend real-world
documents, such as webpages, office documents, lecture slides, etc., has
been a long-standing goal of artificial intelligence. To achieve this
goal, machine reading comprehension (MRC), a central task in natural
language understanding, has been intensively studied. The typical
definition of the MRC task is quite simple, wherein given a short
natural language text as a context and a question about it, a machine
reads the text and then answers the question by extracting a span from
the text [RajpurkarZLL16](None), [RajpurkarJL18](None). However, this
definition is far from real-world applications, such as customer service
chatbots on e-commerce websites [CuiHWTDZ17](None) and
assistant systems for reading professional
literature [HongWJZW19](None), in that the context is composed
entirely of text, with no graphical elements.

To this end, visual question answering on document images (document VQA)
has received much attention. It is a challenging vision and language
task that requires methods to reason about document layout, textual
content, and visual
elements [Mathew_2021_WACV](None), [DBLP:conf/aaai/TanakaNY21](None), [Mathew_2022_WACV](None).
When the primary content in a document is text (e.g., e-mails and forms)
and the task is to understand it on the basis of its layout information,
state-of-the-art models have already achieved nearly human-level
performance [xu2020layoutlmv2](None), [powalski2021going](None). On the
other hand, challenges remain when it comes to handling diverse
real-world documents. First and foremost is that current models are not
capable of performing reasoning across multiple images since the
existing datasets focus on testing reasoning ability on a single image.
Moreover, compared with humans, document VQA models still have trouble
understanding documents that contain visual elements and understanding
questions that require numerical
reasoning [Mathew_2022_WACV](None).

To address the above challenges, we introduce a new document VQA
dataset[^1], SlideVQA, for tasks wherein given a slide deck composed of
multiple slide images and a corresponding question, a system selects a
set of evidence images and answers the question. Slide decks are one of
the most efficient document types that arrange visual and textual
elements for communication. As shown in
Figure <a href="#fig:example_dataset" data-reference-type="ref"
data-reference="fig:example_dataset">[fig:example_dataset]</a>, SlideVQA
requires complex reasoning over slide images, including single-hop,
multi-hop, and numerical reasoning. These reasoning skills play
essential roles in MRC
tasks [Yang0ZBCSM18](None), [dua-etal-2019-drop](None).

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2301.04883v1_md/Images/example_slidevqa_v5.png" />

</div>

Our main contributions are summarized as follows:

-   We introduce a novel task and dataset, SlideVQA, wherein to answer
    its questions, a machine has to read and comprehend a slide deck. It
    is the largest multi-image document VQA dataset containing 2.6k+
    slide decks (each consisting of 20 slides) and 14.5k questions. It
    also provides bounding boxes around textual and visual elements for
    understanding document layout and arithmetic expressions for
    numerical reasoning.

-   We developed a **M**ulti-**M**odal **M**ulti-image **D**ocument VQA
    model, M3D, to jointly perform evidence selection and question
    answering tasks and to enhance numerical reasoning by generating
    arithmetic expressions.

-   Our model outperformed existing state-of-the-art QA models on
    SlideVQA, but its performance is still below that of humans by a
    large margin.

# Related Work

### Datasets for VQA on document images.

Document VQA is the task of answering questions about document images,
and some useful datasets have been published, such as
DocVQA [Mathew_2021_WACV](None),
VisualMRC [DBLP:conf/aaai/TanakaNY21](None),
WebSRC [ChenZCJZLX021](None), and
InfographicVQA [Mathew_2022_WACV](None). The task assumes that
the datasets have a single relevant image, containing all the facts
required to answer.

The work most related to ours is
DocCVQA [tito2021document](None), wherein a large collection of
document images is used to answer a given question. Our dataset differs
from DocCVQA, as follows. First, SlideVQA consists of 14.5k questions,
wheres DocCVQA provides only 20 questions. Second, SlideVQA requires
multi-hop reasoning over multiple slides to find the answer, while
DocCVQA requires only single-hop reasoning on individual images to find
the answer. Besides these differences, SlideVQA provides questions that
require numerical reasoning and arithmetic expression annotations to
answer numerical questions (e.g., “30 - 28" for the answer “2"): no
other VQA dataset, including InfographicVQA that requires numerical
reasoning, provides such annotations. Furthermore, SlideVQA provides the
largest number of bounding boxes on all of the collected images among
the related datasets.

<div class="table*" markdown="1">

</div>

### Document VQA Models.

In parallel with the development of datasets,
Transformer [VaswaniSPUJGKP17](None) has come to be used for
understanding unstructured text in document images.
LayoutLM [XuLCHWZ20](None),
LayoutLMv2 [xu2020layoutlmv2](None),
LayoutT5 [DBLP:conf/aaai/TanakaNY21](None), and
TILT [powalski2021going](None) have achieved impressive results
in single-image document VQA tasks by combining textual, layout, and
visual features. By contrast, we focus on endowing models with the
ability to reason and comprehend multiple images. Moreover, while
[tito2021document](None) used a pipeline of retrieval and
reading models for DocCVQA, we use multi-task learning that jointly
performs evidence selection and question answering.

### Multi-modal question answering.

This type takes textual and visual information as input contexts, which
is different from document VQA that takes only a document image as the
input context. TQA [KembhaviSSCFH17](None) is comprised of
middle-school science lessons containing diagrams and text.
MultiModalQA [talmor2021multimodalqa](None) requires joint
reasoning over text, tables, and images in Wikipedia.

### VQA on videos or image sets.

VideoQA focuses on answering questions about video frames of TV
shows [lei-etal-2018-tvqa](None), [lei-etal-2020-tvqa](None) and
movies [tapaswi2016movieqa](None). A similar task is VQA on
image sets (ISVQA), which involves handling photos taken from different
viewpoint indoors [bansal2020visual](None). By contrast, our
dataset also requires a model to understand the text in images.

### Slide images understanding.

[haurilet2019spase](None), [haurilet2019wise](None) introduced a
benchmark for object segmentation on slide-pages.
[sun-etal-2021-d2s](None), [fu2021doc2ppt](None) tackled the task of
generating slides from research papers. Our work is the first to focus
on answering questions on sets of slide images.

### Reasoning over textual documents.

Numerical reasoning plays an important role in NLP
tasks [dua-etal-2019-drop](None), [zhang-etal-2020-language](None), [zhang-etal-2021-noahqa-numerical](None).
Moreover, multi-hop reasoning has taken the spotlight as it aligns with
the multi-hop nature of how humans reason to acquire knowledge, and has
led to a proliferation of
benchmarks [talmor-berant-2018-web](None), [Yang0ZBCSM18](None).
However, there is as yet no dataset for developing models to perform
both multi-hop and numerical reasoning on document images.

# The SlideVQA Task and Dataset

## Task Overview and Formulation

The SlideVQA task, requires a system to answer a question about a slide
deck, which is composed of an ordered set of slide images and to select
evidence slide images. We formulate the end-to-end SlideVQA task as
follows: <span class="smallcaps">MainTask</span> (SlideVQA).
<span id="prob:main" label="prob:main"></span> Given a question $q$ and
a slide deck $\mathbf{I} = \{I_1, \ldots, I_{K}\}$ ($K=20$), a model
outputs an answer $y$ and selects relevant slides
$\mathbf{\hat{I}} = \{\hat{I}_1, \ldots, \hat{I}_{K'}\}$.

The task can be decomposed into two subtasks:

<div id="prob:es" class="subtask" markdown="1">

**Subtask 1** (Evidence Selection). *Given a question $q$ and a slide
deck $\mathbf{I}$, a model identifies the images $\mathbf{\hat{I}}$ from
which to derive the answer $y$.*

</div>

<div id="prob:qa" class="subtask" markdown="1">

**Subtask 2** (Question Answering). *Given a question $q$ and the slide
images ($\mathbf{I}$ or $\mathbf{\hat{I}}$), a model outputs an answer
$y$.*

</div>

SlideVQA has three answer types (see the examples in
Figure <a href="#fig:example_dataset" data-reference-type="ref"
data-reference="fig:example_dataset">[fig:example_dataset]</a>). A
single-span answer is a contiguous sequence of tokens in the reading
order extracted from the image, and a multi-span answer is formed from
multiple spans from the image. A non-span answer is not extracted and is
composed of numerical values and visual appearances.

We can also use annotations of bounding boxes around the objects (and
their categories) to understand the semantic structure of images and
annotations of arithmetic expressions to understand numerical reasoning
as additional input at training. These annotations are not given at
inference.

## Dataset Collection

In this section, we describe the collection process of the SlideVQA
dataset. To control the annotation quality, we recruited crowd workers
located in English-speaking countries and who had passed a rigorous
qualification procedure. Additionally, we asked other workers to assess
the quality of the annotated samples after each collection step.

### Slide decks collection.

First, we selected and downloaded 25,327 slide decks composed of more
than 20 slides from slideshare[^2] and covering 39 topics. We kept the
first 20 slides and truncated the rest of the pages. Then, the workers
filtered the collected decks that did not meet the following criteria:
(i) the main language is English; (ii) the content is easy for workers
to understand; (iii) the decks must contain one or more graphs, tables,
figures, or numerical data to avoid creating questions requiring only
text-level understanding.

### Bounding boxes and categories annotation.

<figure id="fig:bbox">
<embed src="/papers/multipage_ocr/arXiv-2301.04883v1_md/Images/example_bbox.png" />
<figcaption>Example of collected bounding boxes. Colored boxes and words
were annotated by workers. The image can be viewed at <a
href="https://www.slideshare.net/andrybrewok/big-data-analytics-a-social-network-approach"
class="uri">https://www.slideshare.net/andrybrewok/big-data-analytics-a-social-network-approach</a>.</figcaption>
</figure>

To facilitate understanding of the semantic components of images, we
annotated all images with bounding boxes and their categories. The
workers indicated specific objects in each image by annotating bounding
boxes around the objects and classifying them into nine classes that
were based on SPaSe [haurilet2019spase](None) as follows:

-   **Title**: presentation title, slide title

-   **Page-text**: text in slide, bullet-point text list, text list

-   **Obj-text**: text in a figure, image, diagram or table

-   **Caption**: description of figure, image, diagram, or table

-   **Other-text**: footnote, date, affiliation, code, URL

-   **Diagram**: a graphical representation of data, a process

-   **Table**: data arranged in rows and columns

-   **Image**: drawing, logo, map, screenshot, realistic image

-   **Figure**: graph with data points and coordinates

As shown in Figure <a href="#fig:bbox" data-reference-type="ref"
data-reference="fig:bbox">1</a>, SlideVQA provides densely annotated
bounding boxes in images.

<figure id="fig:distributions">
<p><br />
</p>
<figcaption>Distribution of bounding box categories, reasoning types,
numerical operations, and answer types in the test set.</figcaption>
</figure>

### Single-hop QA creation.

We asked the workers to create 12,466 QA pairs by selecting a single
slide image from a slide deck. The selected slide can be used as
evidence to tell whether a system arrived at the right answer for the
right reasons. We encouraged questions that needed numerical reasoning,
including operations of arithmetic expressions with $\{+, -, /, *\}$,
counting, and comparisons. Additionally, the workers avoided creating
questions that (i) contained selected page numbers; (ii) required
external knowledge; (iii) were common to all of the slides (e.g., “What
is the title?").

### Multi-hop questions creation.

We created 2,018 QA pairs for multi-hop reasoning by editing the
single-hop questions created in the previous step. For example at the
left of Figure <a href="#fig:example_dataset" data-reference-type="ref"
data-reference="fig:example_dataset">[fig:example_dataset]</a>, “North"
is replaced by the phrase “the region with 70% of journals". To this
end, we first identified one or two bridge entities in the created
questions, and the workers selected related slides as evidence that
mentioned the identified ones. Then, the content of the selected slides
was utilized to replace the entities in the created questions. The
process of creating multi-hop questions by editing may produce unnatural
questions, as mentioned in the “Limitations" section, but is easily
scalable. A similar approach was taken with
MultiModalQA [talmor2021multimodalqa](None), which requires
multi-hop reasoning over text, tables, and images in Wikipedia.

### Arithmetic expression annotation.

We provided arithmetic expressions like “30 - 28" in which the final
numerical answer can be arrived at with the four arithmetic operations.
The interpretation of the answer generation process is important for
creating explainable QA models.

## Statistics and Analysis

SlideVQA contains 14,484 QA pairs from 2,619 slide decks, consisting of
52,480 slide images annotated with 890,945 bounding boxes. We split the
dataset into 10,617 questions for training, 1,652 (2,215) questions for
development (test), making sure that each deck appears in the same
split.

### Images.

SlideVQA provides the largest number of images covering broad range of
topics among the datasets shown
in Table <a href="#tab:statistics_dataset" data-reference-type="ref"
data-reference="tab:statistics_dataset">[tab:statistics_dataset]</a>.
Moreover, SlideVQA provides the largest number of bounding box
annotations, where the number of the annotations in SlideVQA is 14.7
times that of VisualMRC.
Figure <a href="#fig:distributions" data-reference-type="ref"
data-reference="fig:distributions">2</a>a shows the distribution of
bounding boxes broken down into nine categories, which cover all
classes, including visually related ones (Image and Figure), unlike
DocVQA and DocCVQA. To analyze the OCR tokens, we extracted the text
shown in the images by using the Google Cloud Vision API[^3]. As a
result, the number of OCR tokens the system should consider
simultaneously is larger (1488.88 tokens) than those of single-image
document VQA datasets; the largest dataset (InfographicVQA) has 217.89
tokens.

<figure id="fig:sunburst">
<embed src="/papers/multipage_ocr/arXiv-2301.04883v1_md/Images/sunburst_questions.png" />
<figcaption>Distribution of the first three words of the
questions.</figcaption>
</figure>

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2301.04883v1_md/Images/proposed_model.png" style="width:90.0%" />

</div>

### Questions and answers.

As shown in
Table <a href="#tab:statistics_dataset" data-reference-type="ref"
data-reference="tab:statistics_dataset">[tab:statistics_dataset]</a>,
SlideVQA requires complex reasoning including single/multi-hop, and
numerical reasoning.
Figure <a href="#fig:distributions" data-reference-type="ref"
data-reference="fig:distributions">2</a>b shows the diverse distribution
of questions related to reasoning types. 49.3% of the questions require
multi-hop or numerical reasoning. Moreover, SlideVQA provides
annotations of arithmetic expressions to improve numerical reasoning.
Figure <a href="#fig:distributions" data-reference-type="ref"
data-reference="fig:distributions">2</a>c shows the distribution of
numerical operations. 25.5% of the numerical questions require
arithmetic operations, which current systems have particular difficulty
answering. Figure <a href="#fig:distributions" data-reference-type="ref"
data-reference="fig:distributions">2</a>d shows that multi-span and
non-span account for 32.4% of the answers, indicating systems also need
to generate answers as well as extract multiple spans.

Figure <a href="#fig:sunburst" data-reference-type="ref"
data-reference="fig:sunburst">3</a> shows the sunburst pattern of the
first three words of the questions. “In" and “Regarding" are frequent
first words because SlideVQA needs to search for evidence images from a
slide deck, which is a special pattern in multi-text document
QA [Yang0ZBCSM18](None).

# Our Model

Figure <a href="#fig:proposed_model" data-reference-type="ref"
data-reference="fig:proposed_model">[fig:proposed_model]</a> shows an
overview of our model, called M3D (**M**ulti-**M**odal **M**ulti-image
**D**ocument VQA model). We use Fusion-in-Decoder
(FiD) [izacard2020leveraging](None), which is a
state-of-the-art multi-text encoder-decoder model, as our base model and
initialize FiD with a pre-trained T5 [RaffelSRLNMZLL20](None).
We extend FiD to perform the end-to-end SlideVQA task (defined in
<span class="smallcaps">MainTask</span>) by (i) performing evidence
selection and question answering tasks as a unified sequence-to-sequence
format using multi-task learning, (ii) predicting arithmetic expressions
as intermediate reasoning steps instead of generating answers directly
to enhance numerical reasoning, and (iii) modifying the input sequence
to learn the visual layout and content of the image.

## Multi-modal Task-Specific Input

### Input token sequence.

For each image $I_k$, we first use Faster-RCNN
 [ren2015faster](None), which was trained on SlideVQA, to
extract $N$ semantic regions (bounding boxes) and their labels (e.g.,
Title and Image). We parse the slide image for each extracted region $r$
by using an OCR engine and apply a sub-word tokenizer to obtain OCR
tokens $\mathbf{W}^r_k = \{w^{r}_{k,1},\ldots, w^{r}_{k,n}\}$ and
corresponding OCR bounding boxes. To jointly train the evidence
selection and question answering tasks, we add different task prefixes
$t \in$ {`Evidence Selection`, `Question Answering`} to the encoder
input. Specifically, the input sequence is as follows: $$\nonumber
x_k = (\texttt{task:} t \texttt{ question:} q \texttt{ page:} e_k \texttt{ context:} c_k),$$
where the sequence concatenates each slide and page number pair ($c_k$,
$e_k$) with the question $q$ and task prefix $t$. To tell the role of
each region, we insert region labels `[R`$^{r_i}_{k}$`]`, corresponding
to the region label of the $i$-th region $r_i$ in $k$-th page, before
the OCR tokens $\mathbf{W}^{r_i}_{k}$ extracted in $r_i$: $$\nonumber
c_k = 
( [{\rm \texttt{R}}^{r_1}_{k}], \mathbf{W}^{r_1}_{k}, [{\rm \texttt{R}}^{r_2}_{k}], \mathbf{W}^{r_2}_{k}, \dots,
[{\rm \texttt{R}}^{r_N}_{k}], \mathbf{W}^{r_N}_{k} )$$

### Input embedding.

Following LayoutT5 [DBLP:conf/aaai/TanakaNY21](None), the input
embeddings $\mathbf{z}$ of the encoder are defined by utilizing
multi-modal information, including token $\mathbf{z}^{{\rm token}}$,
segment $\mathbf{z}^{{\rm seg}}$, layout $\mathbf{z}^{{\rm lay}}$, and
visual embeddings $\mathbf{z}^{{\rm vis}}$ as follows: $$\nonumber
\mathbf{z} = {\rm LN}(\mathbf{z}^{{\rm token}} + \mathbf{z}^{{\rm seg}} + \mathbf{z}^{{\rm lay}} + \mathbf{z}^{{\rm vis}})  \in \mathbb{R}^{L \times d},$$
where LN is a layer normalization [BaKH16](None), and $L$ and
$d$ are the length of the input sequence and a hidden vector size,
respectively. The segment embedding indicates which regions are included
in the input sequence. The layout embedding denotes the encoded bounding
box coordinates of the token within the image. We normalize all
coordinates by the size of images and use embedding layers to embed
x-axis and y-axis features separately. The visual embedding is the
appearance feature of each region and the OCR bounding boxes, which were
obtained from Faster-RCNN. Note that the layout and visual embeddings
are set to zero vectors for the task prefix, question, and page number.

## Multi-modal Encoder-Decoder

### Multi-modal encoder.

Our encoder is a stack of $m$ Transformer blocks, consisting of a
self-attention layer and a fully-connected layer with residual
connections. Following FiD [izacard2020leveraging](None), all
$K$ input sequences are encoded independently and then concatenated to
form a unified input representation. Formally, we transform each input
sequence $x_k$ into $\mathbf{x}_k \in \mathbb{R}^{L \times d}$ and
concatenate them into $\mathbf{X} \in \mathbb{R}^{K \times L \times d}$.

### Answer/Arithmetic-expression decoder.

Our decoder is another stack of $m$ Transformer blocks similar to the
multi-modal encoder, where each block has an additional layer of
cross-attention between the output sequence and $\mathbf{X}$. The answer
decoder is modeled as a conditional generation $p_\theta(y|\mathbf{X})$,
where $\theta$ represents the set of all model parameters. To allow the
model to perform numerical reasoning, we train the system to predict
annotated arithmetic expressions $y'$ (e.g., “$30 - 28$") instead of
numeric values $y$ (e.g., “$2$") by modeling $p_\theta(y'|\mathbf{X})$.
During inference, the model itself decides whether numerical reasoning
is required or not for each question by predicting an indicator token
`Answer:` or `Expression:` at the beginning of the output sequence.

### Evidence selector.

The selector shares the weights and the architecture of the
answer/arithmetic-expression decoder. Instead of only modeling answer
generation, we devise a simple method to train evidence selection in a
unified sequence. Specifically, we define the output sequence as
$\hat{\mathbf{I}}_{\text{pages}}$ $=$ (`Evidence pages:` $\hat{e}_1$,
$\ldots$, $\hat{e}_{K'}$), where each $\hat{e}$ is the page number of
the selected slide.

### Training and inference.

Our model is trained by minimizing the weighted sum of two losses
$\mathcal{L} = \mathcal{L}_{\text{dec}} + \mathcal{L}_{\text{sel}}$,
where $\mathcal{L}_{\text{dec}}$ and $\mathcal{L}_{\text{sel}}$ are the
negative log-likelihood between the ground-truth and the prediction
regarding the decoder and selector, respectively. During inference, we
obtain the final prediction to post-process the decoded sequence by
removing the task indicator. If an arithmetic expression is generated
(i.e., `Expression:` is generated), we use a calculator to obtain the
final results.

# Experiments

<div class="table*" markdown="1">

<div class="center" markdown="1">

</div>

<div class="center" markdown="1">

</div>

<div class="center" markdown="1">

</div>

</div>

## Experimental Setup

We conducted experiments on the SlideVQA task, evidence selection task,
and question answering task respectively defined in
<span class="smallcaps">MainTask</span>,
<span class="smallcaps">Subtasks</span>
<a href="#prob:es" data-reference-type="ref"
data-reference="prob:es">1</a> and
<a href="#prob:qa" data-reference-type="ref"
data-reference="prob:qa">2</a>.

### Main task baselines.

We mainly evaluated pipeline models as baselines, consisting of evidence
selection that produces top-3 evidences and question answering that
takes the selection results as input. Here, we introduced a hierarchical
LayoutLMv2 (H-LayoutLMv2) inspired
by [tu2020select](None), [xu2020layoutlmv2](None), which encodes all
slides simultaneously by using another Transformer layer, as the
evidence selector. It achieved 96.0% on Recall@3 on the test set. We
used three generative QA models: a textual model
**T5** [RaffelSRLNMZLL20](None), a numerical and multi-hop
model **PreasM** [yoran-etal-2022-turning](None), and a
document VQA model
**LayoutT5** [DBLP:conf/aaai/TanakaNY21](None). We also used an
extractive document VQA model **LayoutLMv2** to predict the single span.

### Evidence selection baselines.

We also evaluated the evidence selection task alone.
**BM25** [robertson2009probabilistic](None) is a non-neural
retrieval framework to estimate the relevance of texts to a search
query. For the neural models,
**CLIP** [radford2021learning](None) encodes the question and
each image to predict the highest similar pair. BM25 and CLIP used the
top-1 slide as the prediction. **BERT** [DevlinCLT19](None) is
a pre-trained language model which only uses text information with the
Transformer architecture. **LayoutLM** [XuLCHWZ20](None)
incorporates layout information into the input embeddings of BERT.
**LayoutLMv2** includes image features produced by a CNN backbone in
input embeddings. To model the interactions between the slides, we used
**H-LayoutLMv2** described in the previous section. For neural evidence
selection baselines (except for CLIP), we use a hidden state of `[CLS]`
in the last layer to feed into an MLP classifier with a sigmoid
activation. Evidence is selected if its confidence of binary
classification is above the optimal value on the development set.

To evaluate the effectiveness of our generative evidence selection
module, we introduced **BinaryClass** as a classification baseline,
which uses a two-layer MLP classifier with a sigmoid activation on top
of each encoder representation at the start-of-sequence. We also
introduced a generative baseline, **ChainGen**, which generates a
sequence of selected slide page numbers before the
answer [wei2022chain](None).

### Question answering baselines.

In addition to the pipeline models, we developed **Q-only**, which takes
only the question into T5. We also used a VideoQA model
**UniVL** [Luo2020UniVL](None) that can take all of the slide
images as input. Furthermore, we evaluated our base model
**FiD** [izacard2020leveraging](None).

### Human performance.

We asked six crowdworkers (not among those recruited to collect our
dataset) to select slide images relevant to the question and answer the
question.

### Evaluation metrics.

Following HotpotQA [Yang0ZBCSM18](None), we used exact match
(EM) and F1 on each question answering and evidence selection task and
also used Joint EM (JEM) and Joint F1 (JF1) to evaluate both tasks.
These joint metrics penalize models that perform poorly on either task
and assess the accuracy and explainability of the question answering
models.

## Implementation Details

We implemented all of the models in PyTorch and experimented on eight
Tesla V100 32GB GPUs. The size of CLIP was `Large` and the size of the
other models was `Base`. We fine-tuned the models using
AdamW [loshchilov2017decoupled](None) with a learning rate of
5e-5 and a dropout rate of 10%, and we linearly warmed up the learning
rate over 1000 steps. The batch size was set to 32. We evaluated models
every 500 steps and selected the best one on the development set on the
basis of the loss. We used a maximum length of 200 tokens for each input
sequence of M3D, and set the maximum target sequence length to 50. We
trained Faster-RCNN [ren2015faster](None) with a
ResNet-101 [HeZRS16](None) backbone by using stochastic
gradient descent (SGD) [ruder2016overview](None) with a
learning rate of 1e-3 and batch size of one. Standard anchor scales of
\[8, 16, 32\] and anchor ratios of \[0.5, 1.0, 2.0\] were used. For the
VideoQA baseline, we created a new video at a rate of five frames per
second. We used the Google Cloud Vision API to extract text and bounding
boxes from images. When the OCR word is tokenized into sub-word tokens,
the bounding box coordinates of a sub-word token are the same as those
of its whole word.

## Experimental Results and Analysis

### Does our model outperform the baselines?

Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a> summarizes the results of the
main tasks. As shown in
Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>a, M3D outperformed the
baselines on joint EM/F1, where the metrics evaluate the consistency
between the predicted evidence and answers. For the evidence selection
task, Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>b shows that H-LayoutLMv2 and
M3D performed better than the baselines. This indicates that modeling
the interaction between multiple slides simultaneously is needed to
improve performance. For the QA task,
Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>c shows that M3D outperformed
the pipeline methods in all metrics. Our end-to-end M3D model is better
at ignoring the slides irrelevant to the question than the answer
generator in the pipeline methods that strongly depend on the slides
narrowed down by the evidence selector. However, M3D$_{\texttt{GT}}$ in
Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>a achieved a significant
improvement by knowing the ground-truth slides. There is room for
improving the correctness of evidence selection.

<figure id="fig:compare">
<embed src="/papers/multipage_ocr/arXiv-2301.04883v1_md/Images/compare_type.png" />
<figcaption>Performance of models and humans on the answer types,
reasoning types and numerical operation types in the test set. AE stands
for “arithmetic expression”.</figcaption>
</figure>

### What are the characteristics of our dataset?

Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a> shows that adding modality
information tended to improve performance in all tasks. This
demonstrates that SlideVQA requires methods to have the ability to
jointly understand the text, layout, and visual modalities of documents.
As shown in Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>c, Q-only had the lowest
performance, showing that the systems could not answer the question
without reading documents in the SlideVQA task. Additionally, UniVL has
a comparative result to Q-only, indicating that SlideVQA requires
different abilities from VideoQA [le-hoi-2020-video](None),
especially the ability to read texts in images.
Tables <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>a and
<a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>c show that LayoutT5, a
generative model, significantly outperformed LayoutLMv2, an extractive
approach. This result is inline with observations on the DROP
dataset [dua-etal-2019-drop](None), which also has non-span
answers [geva-etal-2020-injecting](None). Additionally, all of
the models performed all of the tasks significantly worse than humans.
To be specific, Figure <a href="#fig:compare" data-reference-type="ref"
data-reference="fig:compare">4</a> illustrates that (i) better multi-hop
reasoning over multiple images is needed and (ii) non-span answers to
questions involving arithmetic operations have to be improved.

### Do our sub-modules improve performance?

Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a> lists the results of an
ablation study. Here, performance consistently decreased as individual
modules were removed from M3D. This indicates that each of the modules
is effective. More precisely, the arithmetic expression (AE) generation
was influential on the QA and Joint performance, meaning that predicting
the arithmetic expression instead of the numerical value enhances the
ability to generate answers with numerical reasoning. As shown in
Figure <a href="#fig:compare" data-reference-type="ref"
data-reference="fig:compare">4</a>, applying AE prediction increased F1
by a large margin (+10.4%) in the arithmetic type.

### What are the effective evidence selection methods?

Table <a href="#tab:qa_classification" data-reference-type="ref"
data-reference="tab:qa_classification">[tab:qa_classification]</a> shows
that our method, which generates the evidence selection and question
answering results separately, obtained the highest performance. It seems
that the generative methods (MultiGen and ChainGen) benefited from the
text-to-text pre-training of T5 more than the classification-based
method (BinaryClass). Our MultiGen decoder that separately trains
evidence selection and question answering had the advantage of being
easier to train than the ChainGen baseline decoder that trains the two
tasks as a single sequence generation task.

### On which categories does the object detection model not work well?

Table <a href="#tab:object_detection" data-reference-type="ref"
data-reference="tab:object_detection">[tab:object_detection]</a> lists
the object detection performance of Faster-RCNN broken down by bounding
box categories. These results show that detecting randomly placed and
small boxes, such as Obj-text, is more difficult than mostly fixed and
large boxes, such as Title.

<figure id="fig:qualitative">
<embed src="/papers/multipage_ocr/arXiv-2301.04883v1_md/Images/qualitative_example.png" />
<figcaption>Qualitative example. GT denotes the ground-truth. (<span
class="math inline">⋅</span>) means the generated arithmetic expression.
The slide deck can be viewed at <a
href="https://www.slideshare.net/musicbizassoc/nielsen-2015-music-biz-presentation-final"
class="uri">https://www.slideshare.net/musicbizassoc/nielsen-2015-music-biz-presentation-final</a>.</figcaption>
</figure>

### Qualitative examples.

Figure <a href="#fig:qualitative" data-reference-type="ref"
data-reference="fig:qualitative">5</a> demonstrates our model’s
performance by visualizing a qualitative example. This example needs
multi-hop reasoning and an answer involving an arithmetic operation. FiD
gave an incorrect answer because it did not consider the visual layout
of the slides. Moreover, while LayoutT5 could not understand the process
of getting numerical answers, M3D successfully extracted information
(“11%" and “12%") and generated the same answer as the ground-truth.

# Discussion and Limitations

SlideVQA is the largest document VQA benchmark that uses multiple images
as input and requires multi-hop reasoning; its limitation is that the
multi-hop questions created by editing are different from the questions
humans might actually ask the system. We argue that developing models
that can reason over multiple images is an important research direction,
and therefore, we employed an editing method that guarantees multi-hop
questions and easily extends the dataset size. Also, our model uses
cross-attention on all evidence candidates, which may cause a
computational problem when there are a lot of input images (e.g., as in
the open-domain QA setting like DocCVQA). To remedy this problem, we
consider that models that train a two-stage selector that roughly
narrows down candidates to a small number of images and then accurately
selects evidence images and an answer generator in an end-to-end manner
are promising [sachan-etal-2021-end](None), [sachan2021endtoend](None).

# Conclusion

We introduced a new document VQA dataset, SlideVQA, focused on the task
of understanding slide decks composed of multiple images. We also
introduced a unified end-to-end model, M3D, that can perform evidence
selection and question answering tasks and enhance numerical reasoning
by generating arithmetic expressions. While our evaluation highlighted
the promise of this approach, it also revealed a huge gap compared with
human performance, and several challenges emerge from multi-hop
reasoning on multiple images and generating answers with arithmetic
operations. We believe that our dataset will contribute to the
development of intelligent assistant agents that can comprehend diverse
real-world documents.

[^1]: Our dataset and codes are publicly available
    at <https://github.com/nttmdlab-nlp/SlideVQA>

[^2]: <https://www.slideshare.net/>

[^3]: https://cloud.google.com/vision