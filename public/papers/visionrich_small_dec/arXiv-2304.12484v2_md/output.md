# Introduction

Information extraction from visually rich documents (VRDs) is an
important research topic that continues to be an active area of research
[chargrid](None), [visualwordgrid](http://arxiv.org/pdf/2010.02358v5), [Cutie](http://arxiv.org/pdf/1903.12363v4), [cloudscan](http://arxiv.org/pdf/1708.07403v1), [layoutlm](http://arxiv.org/pdf/2204.08387v3), [docreader](http://arxiv.org/pdf/2307.02499v1), [trie++](http://arxiv.org/pdf/1903.11279v1), [Layout-aware](http://arxiv.org/pdf/2005.11017v1), [Graph_based-1](http://arxiv.org/pdf/1903.11279v1)
due to its importance in various real-world applications.

The majority of the existing information extraction from visually rich
documents approaches [layoutlm](http://arxiv.org/pdf/2204.08387v3), [Lambert](None), [TILIT](http://arxiv.org/pdf/2102.09550v3), [Bros](http://arxiv.org/pdf/2108.04539v5) depend
on an external deep-learning-based Optical Character Recognition (OCR)
[text_detection](http://arxiv.org/pdf/1904.01941v1), [text_recognition](http://arxiv.org/pdf/1904.01906v4) engine. They follow a
two-step pipeline: First they read the text using an off-the-shelf OCR
system then they extract the fields of interest from the OCR’ed text.
These two-step approaches have significant limitations due to their
dependence on an external OCR engine. First of all, these approaches
need positional annotations along with textual annotations for training.
Also, training an OCR model requires large scale datasets and huge
computational resources. Using an external pre-trained OCR model is an
option, which can degrade the whole model performance in the case of a
domain shift. One way to tackle this is to fine-tune these off-the-shelf
OCR models which is still a delicate task. In fact, the documents full
annotations are generally needed to correctly fine-tune off-the-shelf
OCR models, which is time-consuming and difficult to obtain. OCR
post-correction
[OCR_Post_Correction](None), [OCR_Post_Correction_2](http://arxiv.org/pdf/2309.11549v1) is an option
to correct some of the recognition errors. However, this brings extra
computational and maintenance cost. Moreover, these two-step approaches
rarely fully exploit the visual information because incorporating the
textual information is already computationally expensive.

Recent end-to-end OCR-free information extraction approaches
[eaten](http://arxiv.org/pdf/2403.00724v1), [trie++](http://arxiv.org/pdf/1903.11279v1), [donut](http://arxiv.org/pdf/2305.09520v1) were proposed to tackle some of the
limitations of OCR-dependant approaches. The majority of these
approaches follow an encoder-decoder scheme. However, the used encoders
are either unable to effectively model global dependence when they are
primarily composed of Convolutional neural network (CNN) blocks
[docreader](http://arxiv.org/pdf/2307.02499v1), [eaten](http://arxiv.org/pdf/2403.00724v1) or they don’t give enough privilege to
character-level features extraction when they are are primarily composed
of Swin Transformer [Swin](http://arxiv.org/pdf/2306.13776v1) blocks
[donut](http://arxiv.org/pdf/2305.09520v1), [Dessurt](http://arxiv.org/pdf/2203.16618v3). In this paper, we argue that capturing
both intra-character local patterns and inter-character long-range
connections is essential for the information extraction task. The former
is essential for character recognition and the latter plays a role in
both the recognition and the localization of the fields of interest.

Motivated by the issues mentioned above, we propose an end-to-end
OCR-free information extraction model named DocParser. DocParser has
been designed in a way that allows it to efficiently perceive both
intra-character patterns and inter-character dependencies. Consequently,
DocParser is up to two times faster than state-of-the-art methods while
still achieving state-of-the-art results on various datasets.

# Related Work

## OCR-dependant Approaches

Most of the OCR-dependant approaches simply use an off-the-shelf OCR
engine and only focus on the information extraction task.

Prior to the development of deep learning techniques, earlier approaches
[earlier_approaches_0](http://arxiv.org/pdf/2402.14871v1), [earlier_approaches_1](http://arxiv.org/pdf/2005.01646v1), [earlier_approaches_2](http://arxiv.org/pdf/2311.11856v1)
either followed a probabilistic approach, relied on rules or used
manually designed features which often results in failure when applied
to unfamiliar templates. The initial deep learning approaches only
relied on textual information and simply used pre-trained language
models [Bert](None), [RoBERTa](http://arxiv.org/pdf/1907.11692v1). Later, several approaches tried to
take the layout information into consideration. First,
[chargrid](None) proposed Chargrid, a new type of text
representation that preserves the 2D layout of a document by encoding
each document page as a two-dimensional grid of characters. Then,
[Bert_grid](http://arxiv.org/pdf/1909.04948v2) added context to this representation by using
a BERT language model. Later, [visualwordgrid](http://arxiv.org/pdf/2010.02358v5) improved
the Chargrid model by also exploiting the visual information.
Graph-based models were also proposed to exploit both textual and visual
information [Graph_based-1](http://arxiv.org/pdf/1903.11279v1), [Graph_based-2](http://arxiv.org/pdf/2103.14470v1).

To successfully model the interaction between the visual, textual and
positional information, recent approaches
[layoutlm](http://arxiv.org/pdf/2204.08387v3), [Lambert](None), [TILIT](http://arxiv.org/pdf/2102.09550v3), [Bros](http://arxiv.org/pdf/2108.04539v5) resorted to pre-training
large models. First [layoutlmv0](None) tried to bring the
success of large pre-trained language models into the multi-modal domain
of document understanding and proposed LayoutLM. LayoutLMv2
[layoutlmv1](None) was later released where new pre-training
tasks were introduced to better capture the cross-modality interaction
in the pre-training stage. The architecture was also improved by
introducing spatially biased attention and thus making the spatial
information more influential. Inspired by the Vision Transformer (ViT)
[VIT](http://arxiv.org/pdf/2105.15075v2), [layoutlm](http://arxiv.org/pdf/2204.08387v3) modified LayoutLMv2 by
using patch embeddings instead of a ResNeXt [Resnext](http://arxiv.org/pdf/2007.06257v2)
Feature Pyramid Network [FPN](http://arxiv.org/pdf/2108.00580v3) visual backbone and
released LayoutLMv3. Pre-training tasks were also improved compared to
previous versions. [Lambert](None) proposed LAMBERT which used
a modified RoBERTa [RoBERTa](http://arxiv.org/pdf/1907.11692v1) that also exploits the
layout features obtained from an OCR system. [TILIT](http://arxiv.org/pdf/2102.09550v3)
proposed TILT, a pre-trained encoder-decoder model.
[Bros](http://arxiv.org/pdf/2108.04539v5) tried to fully exploit the textual and layout
information and released Bros which achieves good results without
relying on the visual features. However, the efficiency and the
computational cost of all the previously cited works are still hugely
impacted by the used OCR system.

## End-to-end Approaches

In recent years, end-to-end approaches were proposed for the information
extraction task among many other Visually-Rich Document Understanding
(VRDU) tasks. [eaten](http://arxiv.org/pdf/2403.00724v1), [docreader](http://arxiv.org/pdf/2307.02499v1) both used a CNN-based
encoder and a recurrent neuronal network coupled with an attention
mechanism decoder. However, the accuracy of these two approaches is
limited and they perform relatively badly on small datasets.
[trie++](http://arxiv.org/pdf/1903.11279v1) proposed TRIE++, a model that learns
simultaneously both the text reading and the information extraction
tasks via a multi-modal context block that bridges the visual and
natural language processing tasks. [VIES](http://arxiv.org/pdf/2102.06732v1) released VIES
which simultaneously performs text detection, recognition and
information extraction. However, both TRIE++ and VIES require the full
document annotation to be trained. [donut](http://arxiv.org/pdf/2305.09520v1) proposed
Donut, an encoder-decoder architecture that consists of a Swin
Transformer [Swin](http://arxiv.org/pdf/2306.13776v1) encoder and a Bart
[Bart](None)-like decoder. [Dessurt](http://arxiv.org/pdf/2203.16618v3) released
Dessurt, a model that processes three streams of tokens, representing
visual tokens, query tokens and the response. Cross-attention is applied
across different streams to allow them to share and transfer information
into the response. To process the visual tokens, Dessurt uses a modified
Swin windowed attention that is allowed to attend to the query tokens.
Donut and Dessurt achieved promising results, however, they don’t give
enough privilege to local character patterns which leads to sub-optimal
results for the information extraction task.

# Proposed Method

This section introduces DocParser, our proposed end-to-end information
extraction from VRDs model.

Given a document image and a task token that determines the fields of
interest, DocParser produces a series of tokens representing the
extracted fields from the input image. DocParser architecture consists
of a visual encoder followed by a textual decoder. An overview of
DocParser’s architecture is shown on figure
<a href="#fig:docparser_overview" data-reference-type="ref"
data-reference="fig:docparser_overview">[fig:docparser_overview]</a>.
The encoder consists of a three-stage progressively decreased height
convolutional neural network that aims to extract intra-character local
patterns, followed by a three-stage progressively decreased width Swin
Transformer [Swin](http://arxiv.org/pdf/2306.13776v1) that aims to capture long-range
dependencies. The decoder consists of $n$ Transformer layers. Each layer
is principally composed of a multi-head self-attention sub-layer
followed by a multi-head cross-attention sub-layer and a feed-forward
sub-layer as explained in [attention](http://arxiv.org/pdf/2107.08000v1).

## Encoder

The encoder is composed of six stages. The input of the encoder is an
image of size $H \times W \times 3$. It is first transformed to
$\frac{H}{4} \times \frac{W}{4}$ patches of dimension $C_0$ via an
initial patch embedding. Each patch either represents a fraction of a
text character or a fraction of a non-text component of the input image.
First, three stages composed of ConvNext [ConvNext](http://arxiv.org/pdf/2007.00649v1)
blocks are applied at different scales for character-level
discriminative features extraction. Then three stages of Swin
Transformer blocks are applied with varying window sizes in order to
capture long-range dependencies. The output of the encoder is a feature
map of size $\frac{H}{32} \times \frac{W}{32} \times C_5$ that contains
multi-grained features. An overview of the encoder architecture is
illustrated in figure
<a href="#fig:encoder_architecture" data-reference-type="ref"
data-reference="fig:encoder_architecture">[fig:encoder_architecture]</a>.

### Patch Embedding

  
  
Similar to [SVTR](http://arxiv.org/pdf/2401.09802v1), we use a progressive overlapping patch
embedding. For an input image of size $W \times H \times 3$, a
$3 \times 3$ convolution with stride $2$ is first applied to have an
output of size $\frac{W}{2} \times \frac{H}{2} \times \frac{C_0}{2}$. It
is then followed by a normalization layer and another $3 \times 3$
convolution with stride $2$. The size of the final output is
$\frac{W}{4} \times \frac{H}{4} \times C_0$.

### ConvNext-based Stages

  
  
The first three stages of DocParser’s encoder are composed of ConvNext
blocks. Each stage is composed of several blocks. The kernel size is set
to $7$ for all stages. At the end of each stage, the height of the
feature map is reduced by a factor of two and the number of channels
$C_i,$ $i \in [1,2,3]$ is increased to compensate for the information
loss. The feature map width is also reduced by a factor of two at the
end of the third stage. The role of these blocks is to capture the
correlation between the different parts of each single character and to
encode the non-textual parts of the image. We don’t reduce the width of
the feature map between these blocks in order to avoid encoding
components of different characters in the same feature vector and thus
allowing discriminative character features computation. We note that
contrary to the first encoder stages where low-level features extraction
occurs, encoding components of different characters in the same feature
vector doesn’t affect performance if done in the encoder last stages
where high-level features are constructed. This is empirically
demonstrated in section <a href="#abla" data-reference-type="ref"
data-reference="abla">[abla]</a>. We chose to use convolutional blocks
for the early stages mainly due to their good ability at modeling local
correlation at a low computational cost.

### Swin Transformer-based Stages

  
  
The last three stages of the encoder are composed of Swin Transformer
blocks. We modify Swin’s window-based multi-head self-attention to be
able to use rectangular attention windows. At the output of the fourth
and fifth stages, the width of the feature map is reduced by a factor of
two and the number of channels is increased to compensate for the
information loss. The role of these layers is to capture the correlation
between the different characters of the input image or between textual
and non-textual components of the image. In the forth and fifth stage,
the encoder focuses on capturing the correlation between characters that
belong to adjacent sentences. This is accomplished through the use of
horizontally wide windows, as text in documents typically has an
horizontal orientation. In the last stage, the encoder focuses on
capturing long-range context in both directions. This is achieved
through the use of square attention windows. As a result, the output of
the encoder is composed of multi-grained features that not only encode
intra-character local patterns which are essential to distinguish
characters but also capture the correlation between textual and
non-textual components which is necessary to correctly locate the fields
of interest. We note that positional embedding is added to the encoder’s
feature map before the encoder’s forth stage.

## Decoder

The decoder takes as input the encoder’s output and a task token. It
then outputs autoregressively several tokens that represent the fields
of interest specified by the input token. The decoder consists of
$n$[^2] layers, each one is similar to a vanilla Transformer decoder
layer. It consists of a multi-head self-attention sub-layer followed by
a multi-head cross-attention sub-layer and a feed-forward sub-layer.

### Tokenization

We use the tokenizer of the RoBERTa model [RoBERTa](http://arxiv.org/pdf/1907.11692v1) to
transform the ground-truth text into tokens. This allows to reduce the
number of generated tokens, and so the memory consumption as well as
training and inference times, while not affecting the model performance
as shown in section <a href="#abla" data-reference-type="ref"
data-reference="abla">[abla]</a>. Similar to [donut](http://arxiv.org/pdf/2305.09520v1),
special tokens are added to mark the start and the end of each field or
group of fields. Two additional special tokens $<item>$ and $<item/>$
are used to separate fields or group of fields appearing more than once
in the ground truth. An example is shown in figure
<a href="#fig:token" data-reference-type="ref"
data-reference="fig:token">[fig:token]</a>.

### At Training Time

When training the model, we use a teacher forcing strategy. This means
that we give the decoder all the ground truth tokens as input. Each
input token corresponding last hidden state is used to predict the next
token. To ensure that each token only attends to previous tokens in the
self-attention layer, we use a triangular attention mask that masks the
following tokens.

# Expriments and Results

## Pre-training

We pre-train our model on two different steps :

### Knowledge Transfer Step

Using an $L2$ Loss, we teach the ConvNext-based encoder blocks to
produce the same feature map as the PP-OCR-V2 [Paddle](http://arxiv.org/pdf/2109.03144v2)
recognition backbone which is an enhanced version of MobileNetV1
[mobilenet](http://arxiv.org/pdf/1909.02765v2). A pointwise convolution is applied to the
output of the ConvNext-based blocks in order to obtain the same number
of channels as the output of PP-OCR-V2 recognition backbone. The goal of
this step is to give the encoder the ability to extract discriminative
intra-character features. We use 0.2 million documents from the IIT-CDIP
[CDIP](http://arxiv.org/pdf/2305.06148v1) dataset for this task. We note that even though
PP-OCR-V2 recognition network was trained on text crops, the features
generated by its backbone on a full image are still useful thanks to the
translation equivariance of CNNs.

### Masked Document Reading Step

After the knowledge transfer step, we pre-train our model on the task of
document reading. In this pre-training phase, the model learns to
predict the next textual token while conditioning on the previous
textual tokens and the input image. To encourage joint reasoning, we
mask several $32 \times 32$ blocks representing approximately fifteen
percent of the input image. In fact, in order to predict the text
situated within the masked regions, the model is obliged to understand
its textual context. As a result, DocParser learns simultaneously to
recognize characters and the underlying language knowledge. We use 1.5
million IIT-CDIP documents for this task. These documents were annotated
using Donut. Regex rules were applied to identify poorly read documents,
which were discarded.

## Fine-tuning

After the pre-training stage, the model is fine-tuned on the information
extraction task. We fine-tune DocParser on three datasets: SROIE and
CORD which are public datasets and an in-house private Information
Statement Dataset.

#### SROIE

: A public receipts dataset with 4 annotated unique fields : company,
date, address, and total. It contains 626 receipts for training and 347
receipts for testing.

#### CORD

: A public receipts dataset with 30 annotated unique fields of interest.
It consists of 800 train, 100 validation and 100 test receipt images.

#### Information Statement Dataset (ISD) 

: A private information statement dataset with 18 annotated unique
fields of interest. It consists of 7500 train, 3250 test and 3250 eval
images. The documents come from 15 different insurers, each insurer has
around 4 different templates. We note that for the same template, the
structure can vary depending on the available information. On figure
<a href="#fig:inhouse_dataset" data-reference-type="ref"
data-reference="fig:inhouse_dataset">1</a> we show 3 samples from 3
different insurers.

<figure id="fig:inhouse_dataset">
<p><img src="/papers/visionrich_small_dec/arXiv-2304.12484v2_md/figures/4091269.17906_CCLIE_page0_direct.jpg"
alt="image" /> <img
src="/papers/visionrich_small_dec/arXiv-2304.12484v2_md/figures/1331082.16606_MIGRATION_Groupama_0sinistrepage0.jpg"
alt="image" /> <img
src="/papers/visionrich_small_dec/arXiv-2304.12484v2_md/figures/3262770.14406_CCLIE_page0_generali.jpg"
alt="image" /></p>
<figcaption> <strong>Anonymized samples from our private in-house
dataset.</strong> The fields of interest are located within the red
boxes.</figcaption>
</figure>

## Evaluation Metrics

We evaluate our model using two metrics:

### Field-level F1 Score

The field-level F1 score checks whether each extracted field corresponds
exactly to its value in the ground truth. For a given field, the
field-level F1 score assumes that the extraction has failed even if one
single character is poorly predicted. The field-level F1 score is
described using the field-level precision and recall as:
$$\text{Precision} = \frac{\text{The number of exact field matches}}{\text{The number of the detected fields}}$$
$$\text{Recall} = \frac{ \text{The number of exact field matches}}{\text{The number of  the ground truth fields}}$$
$$\text{F1} = \frac{ 2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Document Accuracy Rate (DAR)

This metric evaluates the number of documents that are completely and
correctly processed by the model. If for a given document we have even
one false positive or false negative, the DAR assumes that the
extraction has failed. This metric is a challenging one, but requested
in various industrial applications where we need to evaluate at which
extent the process is fully automatable.

## Setups

The dimension of the input patches and the output vectors of every stage
$C_i,$ $i \in [0\mathrel{{.}\,{.}}\nobreak 5]$ are respectively set to
$64$, $128$, $256$, $512$, $768$, and $1024$. We set the number of
decoder layers to $1$. This choice is explained in
Section <a href="#abla" data-reference-type="ref"
data-reference="abla">[abla]</a>. For both pre-training and fine-tuning
we use the Cross-Entropy Loss, AdamW [ADAMW](http://arxiv.org/pdf/2311.11446v2) optimizer
with weight decay of $0.01$ and stochastic depth [stocha](http://arxiv.org/pdf/1603.09382v3)
with a probability equal to $0.1$. We also follow a light data
augmentation strategy which consists of light re-scaling and rotation as
well as brightness, saturation, and contrast augmentation applied to the
input image. For the pre-training phase, we set the input image size to
$2560 \times 1920$. The learning rate is set to $1e-4$. The pre-training
is done on 7 A100 GPUs with a batch size of 4 on each GPU. We use
gradient accumulation of 10 iterations, leading to an effective batch
size of $280$. For the fine-tuning, the resolution is set to
$1600 \times 960$ for CORD and SROIE datasets and $1600 \times 1280$ for
the Information Statement Dataset. We pad the input image in order to
maintain its aspect ratio. We also use a Cosine Annealing scheduler
[cosine](http://arxiv.org/pdf/1608.03983v5) with an initial learning rate of $3e-5$ and a
batch size of $8$.

## Results

<div id="tab2" markdown="1">

|  |  |  | SROIE |  | CORD |  | ISD |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 4-5 |  |  |  |  |  |  |  |  |  |  |
|  | OCR | Params(M) | F1(%) | Time(s) | F1(%) | Time(s) | F1(%) | Time(s) |  |  |
| LayoutLM-v3 |  | $87+\alpha^{*}$ | $77.7$ | $2.1 + t^{*}$ | $80.2$ | $2.1+t^{*}$ | $90.8$ | $4.1+t^{*}$ |  |  |
| Donut |  | 149 | 81.7 | 5.3 | 84 | 5.7 | 95.4 | 6.7 |  |  |
| Dessurt |  | 87 | 84.9 | 16.7 | 82.5 | 17.9 | 93.5 | 18.1 |  |  |
| **DocParser** |  | **70** | **87.3** | 3.5 | **84.5** | 3.7 | **96.2** | **4.4** |  |  |

**Performance comparisons on the three datasets.** The field-level
F1-score and the extraction time per image on an Intel Xenion W-2235 CPU
are reported. In order to ensure a fair comparison, we exclude
parameters related to vocabulary. Additional parameters $\alpha^{*}$ and
time $t^{*}$ for the OCR step should be considered for LayouLM-v3. For
the ISD dataset $t^{*}$ is equal to 3.6 seconds.

</div>

We compare DocParser to Donut, Dessurt and LayoutLM-v3. The results are
summarized in table
<a href="#tab2" data-reference-type="ref" data-reference="tab2">1</a>. A
comparison of inference speed on an NVIDIA Quadro RTX 6000 GPU is
presented in table <a href="#tabgpu" data-reference-type="ref"
data-reference="tabgpu">2</a>. Per-field extraction performances on our
Information Statement Dataset can be found in table
<a href="#tab3" data-reference-type="ref" data-reference="tab3">3</a>.
DocParser achieves a new state-of-the-art on SROIE, CORD and our
Information Statement Dataset with an improvement of respectively 2.4,
0.5 and 0.8 points over the previous state-of-the-art. In addition,
Docparser has a significantly faster inference speed and less
parameters.

<span id="tabgpu" label="tabgpu"></span>

<div id="tabgpu" markdown="1">

|               | SROIE           | CORD            | ISD             |     |
|:--------------|:----------------|:----------------|:----------------|:----|
| LayoutLM-v3   | 0.041 + $t^{*}$ | 0.039 + $t^{*}$ | 0.065 + $t^{*}$ |     |
| Donut         | 0.38            | 0.44            | 0.5             |     |
| Dessurt       | 1.2             | 1.37            | 1.39            |     |
| **DocParser** | 0.21            | 0.24            | **0.25**        |     |

**Comparison of inference speed on GPU.** Extraction time (seconds) per
image on an NVIDIA Quadro RTX 6000 GPU is reported. Additional time
$t^{*}$ for the OCR step should be considered for LayouLM-v3. For the
ISD dataset $t^{*}$ is equal to 0.5 seconds.

</div>

<span id="OURDATASET" label="OURDATASET"></span>

<div id="tab3" markdown="1">

|                        | LayoutLM | DONUT | Dessurt | **DocParser** |
|:-----------------------|:---------|:------|:--------|:--------------|
|                        |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
| first driver           |          |       |         |               |
|                        |          |       |         |               |
| second driver          |          |       |         |               |
|                        |          |       |         |               |
| third driver           |          |       |         |               |
|                        |          |       |         |               |
| of contract            |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
| the document           |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
| driver                 |          |       |         |               |
|                        |          |       |         |               |
| driver                 |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
| driver of the accident |          |       |         |               |
|                        |          |       |         |               |
| driver of the accident |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |
|                        |          |       |         |               |

**Extraction performances on our Information Statement Dataset.** Per
field (field-level) F1-score, field-level F1-score mean, DAR, and
extraction time per image on an Intel Xenion W-2235 CPU are reported.
The OCR engine inference time $t^{*}$ should be considered for
LayouLM-v3.

</div>

Regarding the OCR required by the LayoutLM-v3 approach, we use, for both
SROIE and CORD datasets, Microsoft Form Recognizer[^3] which includes a
document-optimized version of Microsoft Read OCR (MS OCR) as its OCR
engine. We note that we tried combining a ResNet-50
[Resnet](http://arxiv.org/pdf/1608.05895v1)-based DBNet++ [DB](http://arxiv.org/pdf/2202.10304v1) for text
detection and an SVTR [SVTR](http://arxiv.org/pdf/2401.09802v1) model for text recognition
and fine-tuned them on the fields of interest of each dataset. However,
the obtained results are worse than those obtained with Microsoft Form
Recognizer OCR engine. For the Information Statement Dataset, we don’t
use MS OCR for confidentiality purposes. Instead, we use an in-house OCR
fine-tuned on this dataset to reach the best possible performances. Even
though the best OCRs are used for each task, LayoutLM-v3 extraction
performances are still lower than those of OCR-free models. This proves
the superiority of end-to-end architectures over the OCR-dependent
approaches for the information extraction task. We note that for Donut,
we use the same input resolution as DocParser. For Dessurt, we use a
resolution of $1152 \times 768$, which is the resolution used to
pre-train the model.

# Primary Experiments and Further Investigation <span id="abla" label="abla"></span>

## Primary Experiments

In all the experiments, the tested architectures were pre-trained on 0.5
Million synthesized documents and fine-tuned on a deskewed version of
the SROIE dataset. We report the inference time on a an Intel Xenion
W-2235 CPU, as we aim to provide a model suited for low resources
scenarios.

<div id="tabencoders" markdown="1">

|                        |     |     |     |     |
|:-----------------------|:----|:----|:----|:----|
| EasyOCR-based encoder  |     |     |     |     |
| PP-OCRv2-based encoder |     |     |     |     |
| Proposed encoder       |     |     |     |     |

**Comparison of different encoder architectures.** The dataset used is a
deskewed version of the SROIE dataset. The field-level F1 score is
reported.

</div>

### On the Encoder’s Architecture

The table <a href="#tabencoders" data-reference-type="ref"
data-reference="tabencoders">4</a> shows a comparison between an
EasyOCR[^4]-based encoder, a PP-OCRv2 [Paddle](http://arxiv.org/pdf/2109.03144v2)-based
encoder and our proposed DocParser encoder. Concerning the EasyOCR and
PP-OCRv2 based encoders, each one consists of its corresponding OCR’s
recognition network followed by few convolutional layers that aim to
further reduce the feature map size and increase the receptive field.
Our proposed encoder surpasses both encoders by a large margin.

<div id="finalfmsize" markdown="1">

|                      |                           |     |     |     |
|:---------------------|:--------------------------|:----|:----|:----|
|                      |                           |     |     |     |
| where the feature    |                           |     |     |     |
| map width is reduced |                           |     |     |     |
| (seconds)            | F1(%)                     |     |     |     |
|                      |                           |     |     |     |
| (3,4,5) (proposed)   | Transformer               |     |     |     |
| (3,4,5)              | LSTM + Additive attention |     |     |     |
| (1,2,3)              | Transformer               |     |     |     |
| (1,2,3)              | LSTM + Additive attention |     |     |     |
| No reduction         | Transformer               |     |     |     |
| No reduction         | LSTM + Additive attention |     |     |     |

**The effect of decreasing the width of the feature map in various
stages of DocParser’s encoder.** The dataset used is a deskewed version
of the SROIE dataset. The field-level F1-score and the extraction time
per image on an Intel Xenion W-2235 CPU are reported.

</div>

### On the Feature Map Width Reduction

While encoding the input image, the majority of the text recognition
approaches reduce the dimensions of the feature map mainly vertically
[SVTR](http://arxiv.org/pdf/2401.09802v1) [text_recognition](http://arxiv.org/pdf/1904.01906v4). Intuitively,
applying this approach for the information extraction task may seem
relevant as it allows different characters to be encoded in different
feature vectors. Our empirical results, however, show that this may not
always be the case. In fact, we experimented with reducing the encoder’s
feature map width at different stages. As a decoder, we used both a one
layer vanilla Transformer decoder and a Long Short-Term Memory (LSTM)
[LSTM](http://arxiv.org/pdf/2103.15232v1) coupled with an attention mechanism that uses an
additive attention scoring function [additive](http://arxiv.org/pdf/2201.01706v1).
Table <a href="#finalfmsize" data-reference-type="ref"
data-reference="finalfmsize">5</a> shows that reducing the width of the
feature map in the early stages affects drastically the model’s accuracy
and that reducing the width of the feature map in the later stages
achieves the the best speed-accuracy trade-off.
Table <a href="#finalfmsize" data-reference-type="ref"
data-reference="finalfmsize">5</a> also shows that while the LSTM-based
decoder struggles with a reduced width encoder output, the performance
of the vanilla Transformer-based decoder remains the same in both cases.
This is probably due to the multi-head attention mechanism that makes
the Transformer-based decoder more expressive than an LSTM coupled with
an attention mechanism.

### On the Tokenizer Choice

In addition to the RoBERTa tokenizer, we also tested a character-level
tokenizer.
Table <a href="#tok" data-reference-type="ref" data-reference="tok">6</a>
shows that the RoBERTa tokenizer allows faster decoding while achieving
the same performance as the character-level tokenizer.

<div id="tok" markdown="1">

|                           |     |     |     |     |
|:--------------------------|:----|:----|:----|:----|
|                           |     |     |     |     |
| RoBERTa tokenizer         |     |     |     |     |
| Character-level tokenizer |     |     |     |     |

**Comparison between different tokenization techniques.** The dataset
used is a deskewed version of the SROIE dataset. The field-level
F1-score and the decoding time per image on an Intel Xenion W-2235 CPU
are reported.

</div>

### On the Number of Decoder Layers

Table <a href="#decoderlayers" data-reference-type="ref"
data-reference="decoderlayers">7</a> shows that increasing the number of
decoder layers doesn’t improve DocParser’s performance. Therefore, using
one decoder layer is the best choice as it guarantees less computational
cost.

### On the Data Augmentation Strategy

Additionally to the adopted augmentation techniques, we experimented
with adding different types of blur and noise to the input images for
both the pre-training and the fine-tuning. We concluded that this does
not improve DocParser’s performance. The lack of performance improvement
when using blur may be attributed to the fact that the datasets used for
evaluating the model do not typically include blurred images.
Additionally, it is challenging to accurately create realistic noise,
thus making the technique of adding noise to the input images
ineffective.

<div id="decoderlayers" markdown="1">

|     |     |     |     |     |
|:----|:----|:----|:----|:----|
|     |     |     |     |     |
|     |     |     |     |     |
|     |     |     |     |     |
|     |     |     |     |     |

**Effect of the number of decoder layers on the performance and the
decoding inference time of DocParser.** The dataset used is a deskewed
version of the SROIE dataset. The field-level F1-score and the decoding
time per image on an Intel Xenion W-2235 CPU are reported.

</div>

## Further Investigation

<div id="abc" markdown="1">

|                                              |     |     |     |     |
|:---------------------------------------------|:----|:----|:----|:----|
|                                              |     |     |     |     |
| Knowledge transfer                           |     |     |     |     |
| Knowledge transfer + Document reading        |     |     |     |     |
| Knowledge transfer + Masked document reading |     |     |     |     |

**Comparison between different pre-training strategies.** All the models
are pre-trained for a total of 70k steps. The field-level F1-score is
reported.

</div>

### On the Pre-training Strategy

Table
<a href="#abc" data-reference-type="ref" data-reference="abc">8</a>
presents a comparison between different pre-training strategies. To
reduce compute used, all the models were pre-trained for 70k
back-propagation steps, with 7k knowledge transfer steps in the case of
two pre-training tasks. The results show that masking text regions
during the document reading pre-training task does effectively lead to
an increase in performance on all three datasets. It also confirms, as
demonstrated in [donut](http://arxiv.org/pdf/2305.09520v1) and [Dessurt](http://arxiv.org/pdf/2203.16618v3),
that document reading, despite its simplicity, is an effective
pre-training task.

### On the Input Resolution

Figure <a href="#resolutionimpact" data-reference-type="ref"
data-reference="resolutionimpact">2</a> shows the effect of the input
resolution on the performance of DocParser on the SROIE dataset.
DocParser shows satisfying results even with a low-resolution input. It
achieves 83.1 field-level F1 score with a $960 \times 640$ input
resolution. The inference time for this resolution on an Intel Xenion
W-2235 CPU is only 1.7 seconds. So, even at this resolution, DocParser
still surpasses Donut and LayoutLM-v3 on SROIE while being more than
three times faster. However, if the input resolution is set to
$640 \times 640$ or below, the model’s performance shows a drastic drop.
This may be due to the fact that the characters start to be illegible at
such a low resolution.

<figure id="resolutionimpact">
<img src="/papers/visionrich_small_dec/arXiv-2304.12484v2_md/figures/Investigation.png" style="width:75.0%" />
<figcaption><strong>The impact of the input resolution on DocParser’s
performance on the SROIE dataset.</strong> The field-level F1 score is
reported.</figcaption>
</figure>

# Conclusion

We have introduced DocParser, a fast end-to-end approach for information
extraction from visually rich documents. Contrary to previously proposed
end-to-end models, DocParser’s encoder is specifically designed to
capture both intra-character local patterns and inter-character
long-range dependencies. Experiments on both public and private datasets
showed that DocParser achieves state-of-the-art results in terms of both
speed and accuracy which makes it perfectly suitable for real-world
applications.

### Acknowledgments

The authors wish to convey their genuine appreciation to Prof. Davide
Buscaldi and Prof. Sonia Vanier for providing them with valuable
guidance. Furthermore, the authors would like to express their gratitude
to Paul Wassermann and Arnaud Paran for their assistance in proofreading
previous versions of the manuscript.

[^1]: The corresponding author

[^2]: For our final model, we set $n$=1

[^3]: https://learn.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/concept-read?view=form-recog-3.0.0

[^4]: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr