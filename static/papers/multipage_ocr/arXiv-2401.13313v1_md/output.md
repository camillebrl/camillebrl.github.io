# Introduction

Building document artificial intelligence (Document AI) capable of
reading and comprehending real-world documents, including webpages,
office documents, mobile UIs, etc., has been a long-cherished goal.
Toward this goal, numerous works on visual document understanding (VDU)
have addressed a wide range of tasks, such as document question
answering (QA) [Mathew_2021_WACV](None) and information
extraction [jaume2019funsd](None). Document data contain both
textual and visual objects, with content spread structurally across
various locations depending on diverse document types and formats. To
address this complexity, previous works have proposed models that aim to
improve interactions among text/layout/visual
modalities [xu2020layoutlmv2](None), [appalaraju2021docformer](None).
However, the diversity of documents and tasks poses a challenge in
developing a unified model that can comprehend intricate relationships
between text and visual objects across a wide range of document types,
formats, and tasks.

To improve the generalizability and adaptivity of unseen vision-language
tasks, visual instruction
tuning [xu-etal-2023-multiinstruct](None), [liu2023llava](None) has been
introduced. This approach involves training multimodal large language
models (MLLMs) on a collection of images, task inputs, and instructions.
However, according to  [liu2023hidden](None), most of the
previous visual instruction tuning datasets have primarily focused on
understanding visual (non-textual) objects in scene images and existing
models struggle with accomplishing tasks that require visual document
understanding abilities. While recent
works [zhang2023llavar](None), [ye2023mplugdocowl](None) attempt to deal
with the issue, they still exhibit limitations when generalizing to
unseen tasks and documents.

In this paper, we propose **InstructDoc**[^1], the first large-scale
visual instruction tuning dataset that covers a wide range of VDU tasks
and datasets (12 diverse tasks created from 30 openly available
datasets). Each dataset has diverse instructions annotated by experts,
following a unified instruction schema, composed of user’s *intent* and
*answer style*, for VDU tasks. As shown in
Figure <a href="#fig:samples" data-reference-type="ref"
data-reference="fig:samples">[fig:samples]</a>, InstructDoc requires a
rich set of abilities, including understanding document layout, visual
representations of texts, and relation extraction of objects (e.g.,
graphs and charts) over open document types/formats with handcrafted
instructions.

Furthermore, to enhance the generalization performance on VDU tasks, we
present a **Instruct**ion-based **D**ocument **r**eading and
understanding model, InstructDr, which unifies the visual, text, and
layout modalities of documents by bridging the gap between a vision
encoder and a large language model (LLM) through a new bridging module
called Document-former. The Document-former converts documents into a
useful feature for the LLM. Experiments show that InstructDr achieves
the highest zero-shot performance among existing MLLMs and outperforms
ChatGPT on a wide range of VDU datasets with instructions.

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/samples_instructdoc_dataset.png" />

</div>

# Related Work

### Visual document understanding.

Visual documents are ubiquitous and used in diverse applications,
including QA on business documents [Mathew_2021_WACV](None),
information extraction on receipts [jaume2019funsd](None), and
classification over large document
collections [harley2015evaluation](None). Due to this
diversity, previous works have generally been domain/task-specific,
lacking the sharing of underlying data, model architectures, and
objectives [XuLCHWZ20](None), [appalaraju2021docformer](None), [huang2022layoutlmv3](None).
Although pixel-based
methods [kim2022ocr](None), [lee2023pix2struct](None) can simplify
architectures, these methods have high computational costs (due to the
encoding of high-resolution images) and can have degraded performance on
new tasks. We leverage the reasoning abilities of LLMs and perform all
VDU tasks in a unified sequence-to-sequence format with instructions,
resulting in improved generalization performance.

### Instruction-following language models.

Training LLMs with instructions on various NLP tasks has proven
effective in improving zero-shot performance of unseen
tasks [wei2021finetuned](None), [iyer2022opt](None).
Flan [wei2021finetuned](None), [longpre2023flan](None),
PromptSource [bach-etal-2022-promptsource](None), and Natural
Instructions [mishra-etal-2022-cross](None) collected
instructions and datasets for a variety of general NLP tasks, such as
machine reading comprehension and summarization tasks on plain-text
documents. In contrast, we tackle the challenge of understanding
real-world documents organized in non-plain text formats (e.g., HTML and
PDF).

### Visual instruction tuning.

Researchers have recently explored the application of LLMs to
vision-language tasks by distilling the output of
LLMs [liu2023llava](None), [zhu2023minigpt](None), [ye2023mplugowl](None) or
training with handcrafted
instructions [xu-etal-2023-multiinstruct](None), [instructblip](None).
However, as pointed out in [liu2023hidden](None), these models
struggle with tasks requiring document understanding abilities because
they do not assume that text might be contained in images during
instruction tuning. To mitigate this issue,
LLaVAR [zhang2023llavar](None) and
LLMDoc [ye2023mplugdocowl](None) fine-tune MLLMs with
instruction tuning on document images. However, these approaches have
trouble understanding diverse real-world documents because (i) the
datasets provide a few document and task types, hindering zero-shot
generalization; and (ii) the models simply encode documents via vision
encoders and cannot explicitly learn document meta-information (e.g.,
document layout). In contrast, the InstructDoc covers diverse VDU tasks
and open document types/formats, and InstructDr learns rich
representations of the underlying structure of documents with
instructions.

# InstructDoc Dataset

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/instructdoc_dataset.png" />

</div>

## Problem Formulation

All of the tasks in InstructDoc are simply defined as: given an
instruction $T$ and a document image $I$, a model outputs an answer $A$.
Each task is composed of one or more datasets, where the dataset
$\mathcal{D}$ is associated with the set of $K$ instructions
$\mathcal{T^{\mathcal{D}}} = \{T^{\mathcal{D}}_1, ..., T^{\mathcal{D}}_K\}$
and contains $N$ instances
$\{(\mathcal{T^{\mathcal{D}}}, I_j, A_j)\}^{N}_{j=1}$. Here, we randomly
select the instruction from $\mathcal{T^{\mathcal{D}}}$ for every
instance. Note that we allow the utilization of external OCR engines to
derive the answer in our setting, as in the previous VDU
benchmark [borchmann2021due](None). Our goal is to enable the
model to perform a wide range of VDU tasks with instructions rather than
improving the accuracy of text
recognition [zhang2023llavar](None).

We mainly evaluate the models’ ability to perform zero-shot learning
scenarios. Specifically, we fine-tune a model on a collection of
instruction tasks and evaluate it on unseen datasets defined three
types: (i) **Test$_{\text{Cross-Dataset}}$**: datasets not used during
training, but whose tasks exist in training set; (ii)
**Test$_{\text{Cross-Task}}$**: datasets and associated tasks entirely
unseen during training; and (iii) **Test$_{\text{Cross-Domain}}$**:
datasets, tasks, and document types entirely unseen during training.

## Dataset Collection

In this section, we describe the collection process of the InstructDoc
dataset. InstructDoc is designed to cover a wide range of VDU tasks with
instructions that require reasoning among document layout, images, and
text.

### Source dataset collection.

Figure <a href="#fig:dataset" data-reference-type="ref"
data-reference="fig:dataset">[fig:dataset]</a> shows the source datasets
in InstructDoc. We collected 30 publicly available datasets and 12 tasks
in VDU areas from DUE [borchmann2021due](None) as well as
through manual searches. Following the task clusters defined in previous
works [wei2021finetuned](None), [instructblip](None), we divided the QA
datasets that require different reasoning abilities into different
tasks. As a result, we divided the collected datasets into the following
tasks:

-   **Key Information Extraction (KIE)** assigns each word a semantic
    entity label from predefined
    categories [simsa2023docile](None), [jaume2019funsd](None), [sun2021spatial](None), [park2019cord](None), [huang2019icdar2019](None).

-   **Single-page QA** is a task of QA on single-page documents and
    focuses on document layout and textual content
    understanding [DBLP:conf/aaai/TanakaNY21](None), [ChenZCJZLX021](None), [MishraSSC19](None), [Mathew_2021_WACV](None), [tuselmann2022recognition](None).

-   **Single-page QA w/ Discrete Reasoning** requires various arithmetic
    abilities, including addition, sorting, or
    counting [zhu2022towards](None).

-   **Single-page QA w/ Visual Reasoning** requires a set of abilities,
    including object (e.g., icon) recognition, commonsense
    understanding, and relation extraction on single-page
    documents [lu2021iconqa](None), [kembhavi2016diagram](None), [lu2022learn](None), [kembhavi2016diagram](None).

-   **Single-page QA w/ Discrete & Visual Reasoning** requires both
    discrete and visual
    reasoning [Mathew_2022_WACV](None), [masry-etal-2022-chartqa](None)
    on single-page documents.

-   **Multi-page QA w/ Multi-hop & Discrete & Visual Reasoning**
    requires understanding the content relationship via multi-hop
    reasoning as well as discrete/visual reasoning on multi-page
    documents [SlideVQA2023](None), [landeghem2023document](None).

-   **Document NLI** is a task of natural language inference that
    predicts the entailment relationship between two sentences in a
    document [borchmann2021due](None)

-   **Dialogue** involves a human-agent interaction on the basis of
    document images [zhang2023llavar](None).

-   **Captioning** involves producing descriptions of
    documents [hsu-etal-2021-scicap-generating](None), [wang2021screen2words](None).

-   **Classification** involves classifying a document from a set of
    candidate labels [harley2015evaluation](None).

-   **Document Layout Analysis (DLA)** determines a document’s
    components with bounding
    boxes [li-etal-2020-docbank](None), [doclaynet](None)

-   **Image-Text Matching (ITM)** requires the model to determine
    whether a given OCR text and image match.

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/instructdlip.png" />

</div>

### Query rephrasing.

We found that two KIE datasets (FUNSD and CORD) are challenging because
they contain abbreviated queries that are difficult for humans to
comprehend. To bridge the gap between humans and machines, we replace
these queries with complete and more easily understandable phrases
(e.g., `menu.vatyn` $\to$ `menu_whether_price_tax_included`).

### Instruction annotation.

For each dataset, we manually crafted five to ten distinct instruction
templates in a unified format. For QA tasks, the answers have diverse
styles in the original datasets; for example, DocVQA’s answer is
extractive, which requires the model to extract a contiguous span of
words from the document, but VisualMRC’s answer is generative, which is
not limited to the word spans. Hence, an instruction that sufficiently
describes an arbitrary VDU task should include *intent* and *answer
style* or only *intent*. Specifically, as shown in
Figure <a href="#fig:samples" data-reference-type="ref"
data-reference="fig:samples">[fig:samples]</a>, *intent* describes how
the task can be performed and *answer style* describes how the model
generates the output. If each dataset provides *query and options*, we
fill it in annotated instruction templates.

### Data split.

We split InstructDoc into 23 held-in and seven held-out datasets. For
the held-out evaluation, we aim to understand how instruction tuning on
the held-in datasets improves the zero-shot generalization performance
on unseen datasets, including (i) **Test$_{\text{Cross-Dataset}}$**:
FUNSD and CORD datasets, (ii) **Test$_{\text{Cross-Task}}$**: ChartQA,
InfoVQA, and TabFact datasets, and (iii)
**Test$_{\text{Cross-Domain}}$**: DUDE and SlideVQA datasets. All other
datasets were held-in ones to train our model. Note that the held-out
datasets were carefully selected in order to avoid data contamination.

## Comparison with Related Datasets

Table <a href="#tab:comparison" data-reference-type="ref"
data-reference="tab:comparison">[tab:comparison]</a> shows the
statistics of InstructDoc and other VDU instruction tuning datasets,
including LLaVAR [zhang2023llavar](None) and
DocOwl [ye2023mplugdocowl](None). InstructDoc has three unique
key properties; First, it is the first dataset to address open document
types, including multi-page documents and has the highest standard
deviation in the number of OCR tokens (1442.8) compared with LLaVAR
(93.1) and DocOwl (807.2). This implies that our dataset is a more
challenging setting. Second, InstructDoc covers the widest range of
tasks, offering four times more tasks compared with DocOwl, while LLaVAR
provides only a single task. Finally, InstructDoc provides a more
extensive set of instructions (20.3 words and 7.4 templates) and
annotates various answer styles within the instructions to deal with
various VDU tasks that require diverse abilities. In contrast, the
instructions in DocOwl are limited (five words and a single template)
and LLaVAR has only machine-generated instructions, and they may not
generalize well to reformulations and new tasks.

# Our Model

Figure <a href="#fig:instructdlip" data-reference-type="ref"
data-reference="fig:instructdlip">[fig:instructdlip]</a> depicts our
model, InstructDr (**Instruct**ion-based **D**ument **r**eading and
understanding model). We use pre-trained
BLIP-2 [li2023blip2](None), a state-of-the-art MLLM connected
with instruction-tuned FlanT5 [chung2022scaling](None), as the
base model. We extend BLIP-2 in three key ways; (i) equipping it with
Document-former, an enhanced Q-former module that can capture and
convert the visual and textual content/layout of documents into
representations of the LLM, (ii) conducting multi-task instruction
tuning with unified formats, and (iii) encoding multiple images in
parallel to facilitate understanding of multi-page documents.

<div class="table*" markdown="1">

</div>

## Spatial-aware Document Feature Extraction

### Document image/OCR and instruction encoding.

To encode a document image, we use a pre-trained
CLIP [radford2021learning](None) vision encoder to extract its
visual features $\mathbf{z}^{\text{vis}}$. Additionally, we process the
document image using an OCR engine and apply a sub-word tokenizer to
obtain $M$ word tokens $\{s_i\}_{i=1}^M$ and their corresponding
bounding boxes $\{ (x_i^1, y_i^1, x_i^2, y_i^2)\}_{i=1}^M$, where
($x^1$, $y^1$) and ($x^2$, $y^2$) represent the coordinates of the
top-left and bottom-right corners, respectively. To learn the visual
layout of the image, we construct a spatially aware OCR representation
$\mathbf{z}_i^{\text{ocr}} = \mathbf{z}_i^{\text{word}} + \mathbf{z}_i^{\text{bbox}}$
with learnable embedding layers $\mathbf{W}^{\{s, x, y, h, w\}}$, where
OCR text features are calculated as
$\mathbf{z}^{\text{word}}_i = \mathbf{W}^s(s_i)$ and spatial features
are calculated as
$\mathbf{z}^{\text{bbox}}_i = \mathbf{W}^x(x^1_i, x^2_i) + \mathbf{W}^y(y^1_i, y^2_i) + \mathbf{W}^h(y^2_i - y^1_i) + \mathbf{W}^w(x^2_i - x^1_i)$.
Similarly, we encode an instruction by $\mathbf{W}^{s}$ and obtain its
features $\mathbf{z}^{\text{ins}}$.

### Document-former.

We introduce Document-former, which is a trainable module to bridge the
gap between an image encoder and an LLM, enabling extraction of document
content/layout that LLMs can understand. The architecture of
Document-former is a stack of Transformer blocks with cross-attention
layers. To map document features into the LLM’s space, we use a set of
$m$ learnable tokens $\mathbf{z}^{\text{token}} \in \mathbb{R}^{d}$,
where $d$ is the dimension of the hidden size. These tokens
$\mathbf{z}^{\text{token}}$ interact with $\mathbf{z}^{\text{vis}}$
through cross-attention layers and with the input sequence, composed of
$\mathbf{z}^{\text{ins}}$ and $\mathbf{z}^{\text{ocr}}$, through
self-attention layers. As a result, we obtain $\mathbf{z}^{\text{doc}}$
and transform it via a projection feed-forward network (FFN) layer to
$\mathbf{h}^{\text{doc}} \in \mathbb{R}^{m \times d^{\text{LLM}}}$,
which have the same dimension $d^{\text{LLM}}$ as the LLM’s input
embedding.

## Multimodal Document Large Language Model

### Connecting document features to LLM.

The LLM receives the document embeddings $\mathbf{h}^{\text{doc}}$, the
instruction, and OCR tokens as input and outputs the answer
$\mathbf{A}$, token by token. The parameters of the LLM are initialized
from an instruction-tuned FlanT5.

### Parameter-efficient multi-task instruction tuning.

To achieve task-agnostic learning, we formulate the process of learning
all held-in tasks in a unified sequence-to-sequence abstraction through
instructions. To train the LLM efficiently, we update only the
parameters of the Document-former (including
$\mathbf{W}^{\{s, x, y, h, w\}}$) and the projection FFN layer, while
keeping other parameters frozen during training. We optimize the model
by minimizing the negative log-likelihood between the ground-truth and
predictions.

### Multi-page document understanding.

We also support performing reasoning across multiple document pages. As
shown in Figure <a href="#fig:instructdlip" data-reference-type="ref"
data-reference="fig:instructdlip">[fig:instructdlip]</a>b, each image is
processed individually by the image encoder and Document-former, and
their resulting document embeddings are mean-pooled together before
being fed into the LLM. The OCR input to the LLM consists of
concatenated tokens extracted from each page.

# Experiments

<div class="table*" markdown="1">

</div>

## Experimental Setup

We mainly conducted evaluations under three zero-shot settings,
including **Test$_{\text{Cross-Dataset}}$**,
**Test$_{\text{Cross-Task}}$**, and **Test$_{\text{Cross-Domain}}$**.
Furthermore, we evaluated our model under the task-specific fine-tuning
setting.

### Baselines.

We compared InstructDr with seven state-of-the-art (SOTA) MLLMs,
including **LLaVA** [liu2023llava](None),
**MiniGPT-4** [zhu2023minigpt](None) and
**mPLUG-Owl** [ye2023mplugowl](None), which align CLIP visual
encoder with Vicuna [vicuna2023](None) trained on a dialogue
generated by GPT-4 [openai2023gpt4](None);
**BLIP-2** [li2023blip2](None), which connects a FlanT5 with a
vision encoder; **InstructBLIP** [instructblip](None), which
fine-tunes BLIP-2 with instructions on scene images; and
**LLMDoc** [ye2023mplugdocowl](None) and
**LLaVAR** [zhang2023llavar](None), which fine-tune
mPULG-Owl/LLaVA on the DocOwl/LLaVAR datasets. Additionally, we used
**Supervised SOTA
models** [appalaraju2023docformerv2](None), [chen2023pali](None), [huang2022layoutlmv3](None), [landeghem2023document](None)
on each dataset and two text-based LLMs, **ChatGPT**
(`gpt-3.5-turbo-0613`) and **GPT-4**. To control the answer’s length, we
added control phrases (e.g., *use 1 to 3 words to answer*) to the
selected instructions.

### Evaluation metrics.

We followed the evaluation protocol of each dataset, we used
**ANLS** [BitenTMBRJVK19](None) for InfoVQA, DUDE, Text-VQA and
ST-VQA, **EM** for SlideVQA, Relaxed Accuracy (**RAcc.**) for ChartQA,
entity F1 (**eF1**) for FUNSD and CORD, Accuracy (**Acc.**) for TabFact,
and **ROUGE-L** for VisualMRC as evaluation metrics. Additionally, we
used **F1** as the optional metrics.

### Implementation details.

Following [wei2021finetuned](None), we balanced the training
instances of different tasks by sampling a maximum of 5k instances for
each held-in dataset while keeping all evaluation instances. We used the
AdamW [loshchilov2017decoupled](None) with a weight decay of
0.05. We applied a linear warmup during the initial 1,000 steps and used
a cosine learning rate decay with a minimum learning rate of 0. We set
the number of learnable tokens $m$ to $32$. All images of the model
input were resized to $224$. We trained on eight A100 (40G) GPUs for
three epochs and completed the training within two hours. If each
dataset does not provide OCR, we extracted it via the Google Vision API.

## Experimental Results and Analysis

### Does our model outperform existing MLLMs?

Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a> shows that our model achieved
the highest performance on all datasets compared with other MLLMs.
InstructDr consistently outperformed its original backbone, BLIP-2, by a
significant margin, indicating that instruction tuning on InstructDoc
effectively enhances performance on unseen VDU datasets, tasks, and
domains. In contrast, InstructBLIP, which is instruction-tuned BLIP-2
trained on scene images, performed worse than BLIP-2. This is because
that InstructBLIP does not assume that the images might contain text
during instruction tuning. BLIP-2 fine-tuned on InstructDoc falls short
of achieving the same level of performance compared with InstructDr,
indicating that InstructDr is better suited for comprehending diverse
real-world documents. This conclusion is further supported by the
results presented in
Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a>, where ablations of
Document-former, spatial information, and strategy of gathering
multi-page features have a significant negative impact on performance.

### How well does our model perform in comparison with supervised SOTA models and powerful LLMs?

As shown in
Table <a href="#tab:compare_chatgpt" data-reference-type="ref"
data-reference="tab:compare_chatgpt">[tab:compare_chatgpt]</a>, our
model outperformed ChatGPT on all datasets. Additionally, InstructDr
achieved competitive results with supervised SOTA models and GPT-4 on
the DUDE and SlideVQA datasets that require multiple reasoning skills
(e.g., discrete, visual, and multi-hop reasoning). This indicates that
our model can effectively learn diverse skills through instruction
tuning with InstructDoc.

<figure id="fig:robustnss">
<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/instruct_robustness.png" />
<figcaption>Comparison of zero-shot performance on DUDE for five
different instructions. w/o Multiple instructions denotes our model
trained with a single instruction per dataset.</figcaption>
</figure>

<figure id="fig:ablation_task">
<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/task_ablation.png" />
<figcaption>Model performance as the number of task clusters used in
training. (<span class="math inline">⋅</span>) denotes the number of
tasks.</figcaption>
</figure>

### What is the role of instructions?

As shown in Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a>, removing instructions
(i.e., only *query and options* as the model input) significantly
decreased zero-shot performance during training or/and test time,
indicating the effectiveness of incorporating instructions. This result
was observed on the high-quality instruction-tuning
datasets [wei2021finetuned](None), [xu-etal-2023-multiinstruct](None).
Moreover, our instruction annotations, including query rephrasing and
answer styles, helped to improve the zero-shot performance.

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/examples_output.png" />

</div>

### Does our model have robustness towards diverse instructions?

Figure <a href="#fig:robustnss" data-reference-type="ref"
data-reference="fig:robustnss">1</a> shows the performance variance when
the models were given five different instructions; InstructDr exhibited
the smallest performance variance and outperformed the other models.
This indicates InstructDoc empowers the model with the ability to deal
with a variety of instructions. Our results also suggest that using
multiple instructions per dataset is important for achieving decent
performance.

### What is the impact of diverse task clusters?

As shown in
Figure <a href="#fig:ablation_task" data-reference-type="ref"
data-reference="fig:ablation_task">2</a>, as the number of task clusters
increases, we can observe an improvement in models’ zero-shot
performance.

### Are our model weights effective for task-specific fine-tuning?

We further fine-tuned InstructDr (only Document-former module) on a
specific dataset to investigate the knowledge and transferability of our
instruction-tuned model weights.
Table <a href="#tab:finetune" data-reference-type="ref"
data-reference="tab:finetune">[tab:finetune]</a> shows the fine-tuning
performance on held-in (VisualMRC) and held-out (DUDE, SlideVQA) tasks.
InstructDr achieved state-of-the-art finetuning performance on
VisualMRC, DUDE, and SlideVQA using a unified model. Compared with
BLIP-2, InstructDr exhibited superior fine-tuning performance on both
held-in/out datasets, validating InstructDr as a better weight
initialization model for task-specific fine-tuning.

### Can our model also understand images other than documents?

Table <a href="#tab:textvqa" data-reference-type="ref"
data-reference="tab:textvqa">[tab:textvqa]</a> shows the zero-shot
performance of scene-text
VQA [SinghNSJCBPR19](None), [BitenTMBRJVK19](None) on scene images,
which are the unseen image types in InstructDoc but were used for
training our base model, BLIP-2. Note that ST-VQA’s images include the
part of COCO [lin2014microsoft](None) that InstructBLIP was
trained on. This result indicates that InstructDr can effectively learn
visual reasoning skills without forgetting the abilities of the original
backbone.

### Qualitative examples.

Figure <a href="#fig:output" data-reference-type="ref"
data-reference="fig:output">[fig:output]</a> visualizes output examples,
where the left/center/right examples require table/visual/hand-written
text understanding skills. ChatGPT gave incorrect answers because it can
only consider text information. Moreover, while BLIP-2 could not follow
instructions (e.g., *use 5 to 10 words*) and extract items from
structured text, InstructDr accomplished diverse VDU tasks with
instructions. As shown in the right example, all models affected OCR
quality, causing incorrect answers.

# Limitations

Despite its impressive performance on various VDU tasks with
instructions, InstructDr suffers from noisy OCR predictions, whose
performance depends highly on OCR text qualities (right of
Figure <a href="#fig:output" data-reference-type="ref"
data-reference="fig:output">[fig:output]</a>). We argue that our
approach is more cost-efficient and accurate because another approach,
the pixel-based ones [kim2022ocr](None), [chen2023pali](None), requires
a large amount of computation to encode high-resolution images and
cannot use document meta-information (e.g., bounding boxes). Moreover,
since InstructDoc only contains a single document-text pair per
instance, it cannot learn the correlation among multiple document-text
pairs and lacks an in-context learning capability. The same observation
has also been reported in the
Flamingo [alayrac2022flamingo](None) and BLIP-2. Finally, while
we have constructed diverse VDU tasks, the number of tasks and
corresponding instructions are still limited. We plan to consider
utilizing automatic generation and augmentation techniques to increase
the variety of instructions available.

# Conclusion

We introduced a new large-scale instruction-tuning dataset, InstructDoc,
to lay the foundation for building general-purpose VDU models that can
follow natural language instructions. We also introduced a simple yet
effective instruction tuning model, InstructDr, which unifies the
vision, text, and layout modalities of documents by bridging the gap
between a vision encoder and an LLM with Document-former. We performed a
comprehensive study on instruction tuning with InstructDoc and
demonstrated its generalization capability to a wide range of VDU
datasets, tasks, and domains with instructions. We believe that our
dataset will facilitate research on developing general-purpose document
artificial intelligence systems.

# Further InstructDoc Details

<div class="table*" markdown="1">

</div>

## Dataset Collection

### Dataset list.

Table <a href="#tab:datasets" data-reference-type="ref"
data-reference="tab:datasets">[tab:datasets]</a> shows the detail of all
datasets we used in InstructDoc. It contains 5,917,602 held-in instances
and 30,177 held-out instances.

### Query rephrasing.

Table <a href="#tab:query" data-reference-type="ref"
data-reference="tab:query">[tab:query]</a> shows the detail of the query
rephrasing annotation. The rephrased queries are more easily
understandable phrases than the original queries.

### Instruction annotation.

Table <a href="#tab:cord" data-reference-type="ref"
data-reference="tab:cord">[tab:cord]</a>-<a href="#tab:doclaynet" data-reference-type="ref"
data-reference="tab:doclaynet">[tab:doclaynet]</a> show the examples of
instructions for each task in InstructDoc.

## Dataset Analysis

### Starting words of the instructions.

<figure id="fig:sunburst">
<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/instruct_trigram.png" />
<figcaption>Distribution of first three words of the
instructions.</figcaption>
</figure>

Figure <a href="#fig:sunburst" data-reference-type="ref"
data-reference="fig:sunburst">3</a> shows the sunburst pattern of the
first three words of the instructions. It can be seen that the
instructions contain various types, such as questions (e.g., “*What is
the*") and requests (e.g., “*I want to*") used in real-world situations.

### Answer styles.

Figure <a href="#fig:answer_types" data-reference-type="ref"
data-reference="fig:answer_types">4</a> shows InstructDoc has five
diverse answer types.

### Word clouds.

Figure <a href="#fig:statistics" data-reference-type="ref"
data-reference="fig:statistics">[fig:statistics]</a> shows how diverse
the vocabulary space is in InstructDoc.

<figure id="fig:answer_types">
<embed src="/papers/multipage_ocr/arXiv-2401.13313v1_md/Images/answer_types.png" />
<figcaption>Distribution of the answer styles in QA
datasets.</figcaption>
</figure>

<div class="figure*" markdown="1">

</div>

# Further Evaluation Setup Details

## Main Evaluation Datasets Details

### FUNSD.

Form Understanding in Noisy Scanned Documents
(FUNSD) [jaume2019funsd](None) evaluates on the *KIE* task:
predicting the entity, “title", “key", “value", or “other", for the
assigned text token.

### CORD.

Consolidated Receipt Dataset for Post-OCR Parsing
(CORD) [park2019cord](None) is the *KIE* dataset with 30 labels
under 4 categories such as “total" or “subtotal".

### InfographicVQA.

This dataset focuses on the task of *single-page QA w/ discrete & visual
reasoning* on infographics. It requires understanding plots/graphs,
texts, and layout [Mathew_2022_WACV](None).

### ChartQA.

This dataset focuses on the task of *single-page QA w/ discrete & visual
reasoning* on chart images. We used both two subsets: (i)
machine-generated set and (ii) human-written
set [masry-etal-2022-chartqa](None).

### TabFact.

This dataset studies the task of *Document NLI* with semi-structured
evidence over tables. It predicts the entailment relationship between
two sentences in a document [borchmann2021due](None).

### DUDE.

Document Understanding Dataset and Evaluation
(DUDE) [landeghem2023document](None) focuses on the task of
*multi-page QA w/ discrete & visual & multi-hop reasoning*. It is a
multi-page, multi-domain, and multi-industry Document VQA for real-world
document understanding.

### SlideVQA.

This dataset focuses on the task of *multi-page QA w/ discrete & visual
& multi-hop reasoning* on the slide deck composed of multiple images. It
requires selecting a set of evidence and answering the
question [SlideVQA2023](None).

## Other Evaluation Datasets Details

### VisualMRC.

Visual Machine Reading Comprehension
(VisualMRC) [DBLP:conf/aaai/TanakaNY21](None) is the task of
abstractive single-page QA on the Web screenshot. We used the end-to-end
setting where answers are derived from OCR results and images without
ROI detection.

### TextVQA.

It contains scene images from Open Images
dataset [kuznetsova2020open](None), with questions asking to
reason about text in the image [SinghNSJCBPR19](None).

### ST-VQA.

It contains scene images from multiple sources, such as Visual
Genome [KrishnaZGJHKCKL17](None). We used the Open Dictionary
setting where answer candidates and vocabularies are not provided at
test time [BitenTMBRJVK19](None).

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

|                                                                |
|:---------------------------------------------------------------|
| <span class="smallcaps">**<u>Input</u>**</span>                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
|                                                                |
| Please output the category corresponding to the text “16,500". |
| <span class="smallcaps">**<u>Target</u>**</span>               |
| menu_price                                                     |

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

[^1]: Our dataset and codes are publicly available at
    <https://github.com/nttmdlab-nlp/InstructDoc>