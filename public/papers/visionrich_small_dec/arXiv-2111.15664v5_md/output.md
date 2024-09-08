# Introduction

# Method

## Preliminary: background

# Experiments and Analyses [sec:exp]

# Related Work

# Conclusions

# Appendix

## Details of OCR Engines (MS, CLOVA, Easy, Paddle) [sec:detail_of_ocr_engines]

Current state-of-the-art visual document understanding (VDU) backbones,
such as BROS [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322),
LayoutLM [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172) and
LayoutLMv2 [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), are dependent on
off-the-shelf OCR engines. These backbones take the output of OCR as
their (one of) input features. For the OCR-dependent methods, in our
experiments, we use state-of-the-art OCR engines that are publicly
available, including 2 OCR API products (i.e., MS OCR[^3] and CLOVA
OCR[^4]) and 2 open-source OCR models (i.e., Easy OCR[^5] and Paddle
OCR[^6]). In the main paper, Paddle OCR is used for the Chinese train
ticket dataset [eaten](eaten) and CLOVA OCR is used for the rest
datasets in the document information extraction (IE) tasks. MS OCR is
used to measure the running time of the LayoutLM family in document
classification and visual question answering (VQA) tasks, following the
previous work of Xu et al. [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201).
Each OCR engine is explained in the following.

### MS OCR

MS OCR is the latest OCR API product from Microsoft and used in several
recent VDU methods, e.g.,
LayoutLMv2 [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201). This engine
supports 164 languages for printed text and 9 languages for handwritten
text (until 2022/03).

### CLOVA OCR

CLOVA OCR is an API product from NAVER CLOVA and is specialized in
document IE tasks. This engine supports English, Japanese and Korean
(until 2022/03). In the ablation experiments on the CORD
dataset [park2019cord](park2019cord) (Figure 9 in the main paper), the
CLOVA OCR achieved the best accuracy.

### Easy OCR

Easy OCR is a ready-to-use OCR engine that is publicly available at
GitHub. This engine supports more than 80 languages (until 2022/03).
Unlike the aforementioned two OCR products (i.e., MS OCR and CLOVA OCR),
this engine is publicly opened and downloadable. The entire model
architecture is based on the modern deep-learning-based OCR
modules [baek2019craft](baek2019craft), [baek2019wrong](baek2019wrong) with some
modifications to make the model lighter and faster. The total number of
model parameters is 27M which is small compared to the state-of-the-art
models [baek2019craft](baek2019craft), [baek2019wrong](baek2019wrong).

### Paddle OCR

Paddle OCR is an open-source OCR engine available at GitHub. We used a
lightweight (i.e., mobile) version of the model which is specially
designed for a fast and light OCR of English and Chinese texts. The
model is served on a CPU environment and the size of the model is
extremely small, which is approximately 10M.

<figure id="fig:more_synthdog">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/figA.png" />
<figcaption><span><strong>Examples of SynthDoG.</strong></span> English,
Chinese, Japanese and Korean samples are shown (from top to bottom).
Although the idea is simple, these synthetic samples play an important
role in the pre-training of <span>Donut</span>. Please, see Figure 7 in
the main paper for details</figcaption>
</figure>

## Details of Synthetic Document Generator (SynthDoG) [sec:detail_of_synthdog]

In this section, we explain the components of the proposed Synthetic
Document Generator (SynthDoG) in detail. The entire pipeline basically
follows Yim et al. [synthtiger](synthtiger). Our source code is
available at <https://github.com/clovaai/donut>. More samples are shown
in Figure <a href="#fig:more_synthdog" data-reference-type="ref"
data-reference="fig:more_synthdog">1</a>.

### Background

Background images are sampled from
ImageNet [deng2009imagenet](deng2009imagenet). Gaussian blur is randomly
applied to the background image to represent out-of-focus effects.

### Document

Paper textures are sampled from the photos that we collected. The
texture is applied to an white background. In order to make the texture
realistic, random elastic distortion and Gaussian noise are applied. To
represent various view angles in photographs, a random perspective
transformation is applied to the image.

### Text Layout and Pattern

To mimic the layouts in real-world documents, a heuristic rule-based
pattern generator is applied to the document image region to generate
text regions. The main idea is to set multiple squared regions to
represent text paragraphs. Each squared text region is then interpreted
as multiple lines of text. The size of texts and text region margins are
chosen randomly.

### Text Content and Style

We prepare the multi-lingual text corpora from Wikipedia.[^7] We use
Noto fonts[^8] since it supports various languages. SynthDoG samples
texts and fonts from these resources and the sampled texts are rendered
in the regions that are generated by the layout pattern generator. The
text colors are randomly assigned.

### Post-processing

Finally, some post-processing techniques are applied to the output
image. In this process, the color, brightness, and contrast of the image
are adjusted. In addition, shadow effect, motion blur, Gaussian blur,
and JPEG compression are applied to the image.

## Details of Document Information Extraction

Information Extraction (IE) on documents is an arduous task since it
requires (a) reading texts, (b) understanding the meaning of the texts,
and (c) predicting the relations and structures among the extracted
information. Some previous works have only focused on extracting several
pre-defined key information [eaten](eaten). In that case, only
(a) and (b) are required for IE models. We go beyond the previous works
by considering (c) also. Although the task is complex, its interface
(i.e., the format of input and output) is simple. In this section, for
explanation purposes, we show some sample images (which are the raw
input of the IE pipeline) with the output of Donut.

In the main paper, we test four datasets including two public benchmarks
(i.e., *CORD* [park2019cord](park2019cord) and
*Ticket* [eaten](eaten)) and two private industrial datasets
(i.e., *Business Card* and *Receipt*).
Figure <a href="#fig:ticket_example" data-reference-type="ref"
data-reference="fig:ticket_example">2</a> shows examples of *Ticket*
with the outputs of Donut.
Figure <a href="#fig:cord_example" data-reference-type="ref"
data-reference="fig:cord_example">3</a> shows examples of *CORD* with
the outputs of Donut. Due to strict industrial policies on the private
industrial datasets, we instead show some real-like high-quality samples
of *Business Card* and *Receipt* in
Figure <a href="#fig:kor_jpn_example" data-reference-type="ref"
data-reference="fig:kor_jpn_example">4</a>.

<figure id="fig:ticket_example">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/figB.png" />
<figcaption><span><strong>Examples of <em>Ticket</em> <span
class="citation" data-cites="eaten"></span> with <span>Donut</span>
predictions.</strong></span> There is no hierarchy in the structure of
information (i.e., depth <span class="math inline"> = 1</span>) and the
location of each key information is almost fixed. Failed predictions are
marked and bolded (red)</figcaption>
</figure>

<figure id="fig:cord_example">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/figC.png" />
<figcaption><span><strong>Examples of <em>CORD</em> <span
class="citation" data-cites="park2019cord"></span> with
<span>Donut</span> predictions.</strong></span> There is a hierarchy in
the structure of information (i.e., depth <span
class="math inline"> = 2</span>). <span><strong>Donut</strong></span>
not only reads some important key information from the image, but also
predicts the relationship among the extracted information (e.g., the
name, price, and quantity of each menu item are grouped)</figcaption>
</figure>

<figure id="fig:kor_jpn_example">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/figD.png" />
<figcaption><span><strong>Examples of <em>Business Card</em> (top) and
<em>Receipt</em> (bottom).</strong></span> Due to strict industrial
policies on the private industrial datasets from our active products,
real-like high-quality samples are shown instead</figcaption>
</figure>

<figure id="fig:onecol">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/figE.png" />
<figcaption><strong>Donut training scheme with teacher forcing and
decoder output format examples.</strong> The model is trained to
minimize cross-entropy loss of the token classifications simultaneously.
At inference, the predicted token from the last step is fed to the
next</figcaption>
</figure>

## Details of Model Training Scheme and Output Format [sec:detail_of_scheme_and_format]

In the model architecture and training objective, we basically followed
the original Transformer [vaswani2017transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), which
uses a Transformer encoder-decoder architecture and a teacher-forcing
training scheme. The teacher-forcing scheme is a model training strategy
that uses the ground truth as input instead of model output from a
previous time step.
Figure <a href="#fig:onecol" data-reference-type="ref"
data-reference="fig:onecol">5</a> shows a details of the model training
scheme and decoder output format.

## Implementation and Training Hyperparameters [sec:detail_of_implementation_and_hyperparams]

The codebase and settings are available at GitHub.[^9] We implement the
entire model pipeline with Huggingface’s
`transformers`[^10] [wolf-etal-2020-transformers](https://aclanthology.org/2020.emnlp-demos.6) and an
open-source library `TIMM` (PyTorch image
models)[^11] [rw2019timm](https://github.com/rwightman/pytorch-image-models).

For all model training, we use a half-precision (fp16) training. We
train Donut using Adam optimizer [Adamoptim](http://arxiv.org/abs/1412.6980) by
decreasing the learning rate as the training progresses. The initial
learning rate of pre-training is set to 1e-4 and that of fine-tuning is
selected from 1e-5 to 1e-4. We pre-train the model for 200K steps with
64 NVIDIA A100 GPUs and a mini-batch size of 196, which takes about 2-3
GPU days. We also apply a gradient clipping technique where a maximum
gradient norm is selected from 0.05 to 1.0. The input resolution of
Donut is set to 2560$\times$`<!-- -->`{=html}1920 at the pre-training
phase. In downstream tasks, the input resolutions are controlled. In
some downstream document IE experiments, such as,
*CORD* [park2019cord](park2019cord), *Ticket* [eaten](eaten)
and *Business Card*, smaller size of input resolution, e.g.,
1280$\times$`<!-- -->`{=html}960, is tested. With the
1280$\times$`<!-- -->`{=html}960 setting, the model training cost of
Donut was small. For example, the model fine-tuning on *CORD* or
*Ticket* took approximately 0.5 hours with one A100 GPU. However, when
we set the 2560$\times$`<!-- -->`{=html}1920 setting for larger
datasets, e.g., *RVL-CDIP* or *DocVQA*, the cost increased rapidly. With
64 A100 GPUs, *DocVQA* requires one GPU day and *RVL-CDIP* requires two
GPU days approximately. This is not surprising in that increasing the
input size for a precise result incurs higher computational costs in
general. Using an efficient attention
mechanism [wang2020linformer](wang2020linformer) may avoid the problem in
architectural design, but we use the original
Transformer [vaswani2017transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) as we aim to present
a simpler architecture in this work. Our preliminary experiments in
smaller resources are available in
Appendix <a href="#sec:smaller_resources" data-reference-type="ref"
data-reference="sec:smaller_resources">6.6</a>.

For the implementation of document IE baselines, we use the
`transformers` library for BERT [devlinBERT2018](https://aclanthology.org/N19-1423),
BROS [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322),
LayoutLMv2 [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), [layoutxlm](layoutxlm) and
WYVERN [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271). For the
SPADE [hwang-etal-2021-spatial](https://aclanthology.org/2021.findings-acl.28) baseline, the official
implementation[^12] is used. The models are trained using NVIDIA P40,
V100, or A100 GPUs. The major hyperparameters, such as initial learning
rate and number of epochs, are adjusted by monitoring the scores on the
validation set. The architectural details of the OCR-dependent VDU
backbone baselines (e.g., LayoutLM and LayoutLMv2) are available in
Appendix <a href="#sec:detail_of_VDU_backbone" data-reference-type="ref"
data-reference="sec:detail_of_VDU_backbone">6.7</a>.

## Preliminary Experiments in Smaller Resources [sec:smaller_resources]

In our preliminary experiments, we pre-trained Donut with smaller
resources (denoted as Donut$_{\text{Proto}}$), i.e., smaller data
(SynthDoG 1.2M) and fewer GPUs (8 V100 GPUs for 5 days). The input size
was 2048$\times$`<!-- -->`{=html}1536. In this setting,
Donut$_{\text{Proto}}$ also achieved comparable results on *RVL-CDIP*
and *CORD*. The accuracy on *RVL-CDIP* was 94.5 and *CORD* was 85.4.
After the preliminaries, we have scaled the model training with more
data.

## Details of OCR-dependent Baseline Models [sec:detail_of_VDU_backbone]

In this section, we provide a gentle introduction to the general-purpose
VDU backbones, such as LayoutLM [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172) and
LayoutLMv2 [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201). To be specific, we
explain how the conventional backbones perform downstream VDU tasks;
document classification, IE, and VQA. Common to all tasks, the output of
the OCR engine is used as input features of the backbone. That is, the
extracted texts are sorted and converted to a sequence of text tokens.
The sequence is passed to the Transformer encoder to get contextualized
output vectors. The vectors are used to get the desired output. The
difference in each task depends on a slight modification on the input
sequence or on the utilization of the output vectors.

### Document Classification

At the start of the input sequence, a special token `[CLS]` is appended.
The sequence is passed to the backbone to get the output vectors. With a
linear mapping and softmax operation, the output vector of the special
token `[CLS]` is used to get a *class-label* prediction.

### Document IE

With a linear mapping and softmax operation, the output vector sequence
is converted to a *BIO-tag* sequence [hwang2019pot](hwang2019pot).

#### IE on 1-depth structured documents

When there is no hierarchical structure in the document (See
Figure <a href="#fig:ticket_example" data-reference-type="ref"
data-reference="fig:ticket_example">2</a>), the tag set is defined as
{“B$_{k}$”, “I$_{k}$”, “O” $\mid k\in$ pre-defined keys}. “B$_{k}$” and
“I$_{k}$” are tags that represent the beginning (B) and the inside (I)
token of the key $k$ respectively. The “O” tag indicates that the token
belongs to no key information.

#### IE on $n$-depth structured documents

When there are hierarchies in the structure (See
Figure <a href="#fig:cord_example" data-reference-type="ref"
data-reference="fig:cord_example">3</a>), the BIO-tags are defined for
each hierarchy level. In this section, we explain a case where the depth
of structure is $n=2$. The tag set is defined as {“B$_{g}$.B$_{k}$”,
“B$_{g}$.I$_{k}$”, “I$_{g}$.B$_{k}$”, “I$_{g}$.I$_{k}$”, “O” $\mid g\in$
pre-defined parent keys, $k\in$ pre-defined child keys}. For instance,
the Figure <a href="#fig:cord_example" data-reference-type="ref"
data-reference="fig:cord_example">3</a> shows an example where a parent
key is “menu” and related child keys are {“cnt”, “nm”, “price”}.
“B$_{g}$” represents that one group (i.e., a parent key such as “menu”)
starts, and “I$_{g}$” represents that the group is continuing.
Separately from the BI tags of the parent key (i.e., “B$_{g}$” and
“I$_{g}$”), the BI tags of each child key (i.e., “B$_{k}$” and
“I$_{k}$”) work the same as in the case of $n=1$. This BIO-tagging
method is also known as *Group BIO-tagging* and the details are also
available in Hwang et al. [hwang2019pot](hwang2019pot).

### Document VQA

With a linear mapping and softmax operation, the output vector sequence
is converted to a *span-tag* sequence. For the input token sequence, the
model finds the beginning and the end of the answer span. Details can
also be found in the Section 4.2 of Devlin et
al. [devlinBERT2018](https://aclanthology.org/N19-1423).

[^1]:  Corresponding author: gwkim.rsrch@gmail.com

[^2]:  This work was done while the authors were at NAVER CLOVA.

[^3]: <https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/overview-ocr>.<span id="footnote_ms_url"
    label="footnote_ms_url"></span>

[^4]: <https://clova.ai/ocr/en>.<span id="footnote_clova_url"
    label="footnote_clova_url"></span>

[^5]: <https://github.com/JaidedAI/EasyOCR>.<span id="footnote_easy_url"
    label="footnote_easy_url"></span>

[^6]: <https://github.com/PaddlePaddle/PaddleOCR>.<span id="footnote_paddle_url"
    label="footnote_paddle_url"></span>

[^7]: <https://dumps.wikimedia.org>.

[^8]: <https://fonts.google.com/noto>.

[^9]: <https://github.com/clovaai/donut>.

[^10]: <https://github.com/huggingface/transformers>.

[^11]: <https://github.com/rwightman/pytorch-image-models>.

[^12]: <https://github.com/clovaai/spade>.

Understanding document images (*e.g.*, invoices) is a core but
challenging task since it requires complex functions such as *reading
text* and a *holistic understanding of the document*. Current Visual
Document Understanding (VDU) methods outsource the task of reading text
to off-the-shelf Optical Character Recognition (OCR) engines and focus
on the understanding task with the OCR outputs. Although such OCR-based
approaches have shown promising performance, they suffer from 1) high
computational costs for using OCR; 2) inflexibility of OCR models on
languages or types of documents; 3) OCR error propagation to the
subsequent process. To address these issues, in this paper, we introduce
a novel OCR-free VDU model named , which stands for **Do**cume**n**t
**u**nderstanding **t**ransformer. As the first step in OCR-free VDU
research, we propose a simple architecture (*i.e.*, Transformer) with a
pre-training objective (*i.e.,* cross-entropy loss). Donut is
conceptually simple yet effective. Through extensive experiments and
analyses, we show a simple OCR-free VDU model, , achieves
state-of-the-art performances on various VDU tasks in terms of both
speed and accuracy. In addition, we offer a synthetic data generator
that helps the model pre-training to be flexible in various languages
and domains. The code, trained model, and synthetic data are available
at <https://github.com/clovaai/donut>.

<div class="figure*" markdown="1">

<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/fig1.png" />

</div>

<figure id="fig:teaser_summary">
<div class="center">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/fig2.png" />
</div>
<p><span>(a) Pipeline Overview.</span> <span>(b) System
Benchmarks.</span></p>
<figcaption><strong>The pipeline overview and benchmarks.</strong> The
proposed end-to-end model, , outperforms the recent OCR-dependent VDU
models in memory, time cost and accuracy. Performances on visual
document IE <span class="citation" data-cites="park2019cord"></span> are
shown in (b). More results on various VDU tasks are available at
Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">[sec:exp]</a> showing the same
trend</figcaption>
</figure>

Document images, such as commercial invoices, receipts, and business
cards, are easy to find in modern working environments. To extract
useful information from such document images, Visual Document
Understanding (VDU) has not been only an essential task for industry,
but also a challenging topic for researchers, with applications
including document
classification [Kang2014ConvolutionalNN](Kang2014ConvolutionalNN), [7333933](7333933),
information
extraction [hwang2019pot](hwang2019pot), [majumder2020representation](https://www.aclweb.org/anthology/2020.acl-main.580), and
visual question
answering [mathew2021docvqa](mathew2021docvqa), [icdar21docvqa](icdar21docvqa).

Current VDU
methods [hwang2019pot](hwang2019pot), [hwang2020spade](https://aclanthology.org/2021.findings-acl.28), [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172), [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322)
solve the task in a two-stage manner: 1) reading the texts in the
document image; 2) holistic understanding of the document. They usually
rely on deep-learning-based Optical Character Recognition
(OCR) [baek2019craft](baek2019craft), [baek2019wrong](baek2019wrong) for the text reading
task and focus on modeling the understanding part. For example, as shown
in Figure <a href="#fig:problem_definition" data-reference-type="ref"
data-reference="fig:problem_definition">[fig:problem_definition]</a>, a
conventional pipeline for extracting structured information from
documents (a.k.a. document parsing) consists of three separate modules
for text detection, text recognition, and
parsing [hwang2019pot](hwang2019pot), [hwang2020spade](https://aclanthology.org/2021.findings-acl.28).

However, the OCR-dependent approach has critical problems. First of all,
using OCR as a pre-processing method is expensive. We can utilize
pre-trained off-the-shelf OCR engines; however, the computational cost
for inference would be expensive for high-quality OCR results. Moreover,
the off-the-shelf OCR methods rarely have flexibility dealing with
different languages or domain changes, which may lead to poor
generalization ability. If we train an OCR model, it also requires
extensive training costs and large-scale
datasets [baek2019craft](baek2019craft), [baek2019wrong](baek2019wrong), [Liu_2020_CVPR](Liu_2020_CVPR), [spts](https://arxiv.org/abs/2112.07917).
Another problem is, OCR errors would propagate to the VDU system and
negatively influence subsequent
processes [ocr_error_negative](ocr_error_negative), [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271).
This problem becomes more severe in languages with complex character
sets, such as Korean or Chinese, where the quality of OCR is relatively
low [rijhwani-etal-2020-ocr](https://aclanthology.org/2020.emnlp-main.478). To deal with this, post-OCR
correction
module [schaefer-neudecker-2020-two](https://aclanthology.org/2020.latechclfl-1.6), [rijhwani-etal-2020-ocr](https://aclanthology.org/2020.emnlp-main.478), [duong-etal-2021-unsupervised](https://aclanthology.org/2021.nodalida-main.24)
is usually adopted. However, it is not a practical solution for real
application environments since it increases the entire system size and
maintenance cost.

We go beyond the traditional framework by modeling a direct mapping from
a raw input image to the desired output without OCR. We introduce a new
OCR-free VDU model to address the problems induced by the
OCR-dependency. Our model is based on Transformer-only architecture,
referred to as **Do**cume**n**t **u**nderstanding **t**ransformer (),
following the huge success in vision and
language [devlinBERT2018](https://aclanthology.org/N19-1423), [dosovitskiy2020vit](https://openreview.net/forum?id=YicbFdNTTy), [pmlr-v139-kim21k](http://proceedings.mlr.press/v139/kim21k.html).
We present a minimal baseline including a simple architecture and
pre-training method. Despite its simplicity, shows comparable or better
overall performance than previous methods as shown in
Figure <a href="#fig:teaser_summary" data-reference-type="ref"
data-reference="fig:teaser_summary">1</a>.

We take pre-train-and-fine-tune
scheme [devlinBERT2018](https://aclanthology.org/N19-1423), [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172) on training. In
the pre-training phase, learns *how to read the texts* by predicting the
next words by conditioning jointly on the image and previous text
contexts. is pre-trained with document images and their text
annotations. Since our pre-training objective is simple (*i.e.*, reading
the texts), we can realize domain and language flexibility
straightforwardly pre-training with synthetic data. During fine-tuning
stage, learns *how to understand the whole document* according to the
downstream task. We demonstrate has a strong understanding ability
through extensive evaluation on various VDU tasks and datasets. The
experiments show a simple OCR-free VDU model can achieve
state-of-the-art performance in terms of both speed and accuracy.

The contributions are summarized as follows:

1.  We propose a novel OCR-free approach for VDU. To the best of our
    knowledge, this is the first method based on an OCR-free Transformer
    trained in end-to-end manner.

2.  We introduce a simple pre-training scheme that enables the
    utilization of synthetic data. By using our generator SynthDoG, we
    show can easily be extended to a multi-lingual setting, which is not
    applicable for the conventional approaches that need to retrain an
    off-the-shelf OCR engine.

3.  We conduct extensive experiments and analyses on both public
    benchmarks and private industrial datasets, showing that the
    proposed method achieves not only state-of-the-art performances on
    benchmarks but also has many practical advantages (e.g.,
    *cost-effective*) in real-world applications.

4.  The codebase, pre-trained model, and synthetic data are available at
    GitHub.[^1]

[^1]: <https://github.com/clovaai/donut><span id="footnote_code"
    label="footnote_code"></span>.

There have been various visual document understanding (VDU) methods to
understand and extract essential information from the semi-structured
documents such as
receipts [8977955](8977955), [hwang-etal-2021-spatial](https://aclanthology.org/2021.findings-acl.28), [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322),
invoices [8978079](8978079), and form
documents [7333829](7333829), [8977962](8977962), [majumder-etal-2020-representation](https://aclanthology.org/2020.acl-main.580).

Earlier VDU attempts have been done with OCR-independent visual
backbones [Kang2014ConvolutionalNN](Kang2014ConvolutionalNN), [7333933](7333933), [7333910](7333910), [eaten](eaten), [docreader](https://doi.org/10.1007/978-3-030-86549-8\_29),
but the performances are limited. Later, with the remarkable advances of
OCR [baek2019craft](baek2019craft), [baek2019wrong](baek2019wrong) and
BERT [devlinBERT2018](https://aclanthology.org/N19-1423), various OCR-dependent VDU models
have been proposed by combining
them [hwang2019pot](hwang2019pot), [hwang2020spade](https://aclanthology.org/2021.findings-acl.28), [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271).
More recently, in order to get a more general VDU, most
state-of-the-arts [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322)
use both powerful OCR engines and large-scale real document image data
(e.g., IIT-CDIP [iitcdip](https://doi.org/10.1145/1148170.1148307)) for a model pre-training.
Although they showed remarkable advances in recent years, extra effort
is required to ensure the performance of an entire VDU model by using
the off-the-shelf OCR engine.

<div class="figure*" markdown="1">

<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/fig3.png" />

</div>

## Document Understanding Transformer

is an end-to-end (i.e., self-contained) VDU model for general
understanding of document images. The architecture of is quite simple,
which consists of a
Transformer [vaswani2017transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), [dosovitskiy2020vit](https://openreview.net/forum?id=YicbFdNTTy)-based
visual encoder and textual decoder modules. Note that does not rely on
any modules related to OCR functionality but uses a visual encoder for
extracting features from a given document image. The following textual
decoder maps the derived features into a sequence of subword tokens to
construct a desired structured format (e.g., JSON). Each model component
is Transformer-based, and thus the model is trained easily in an
end-to-end manner. The overall process of is illustrated in
Figure <a href="#fig:teaser" data-reference-type="ref"
data-reference="fig:teaser">[fig:teaser]</a>.

### Encoder.

The visual encoder converts the input document image
$\mathbf{x}{\in}\mathbb{R}^{H\times W\times C}$ into a set of embeddings
$\{\mathbf{z}_{i} | \mathbf{z}_{i}{\in}\mathbb{R}^{d}, 1{\le}i{\le}n\}$,
where $n$ is feature map size or the number of image patches and $d$ is
the dimension of the latent vectors of the encoder. Note that CNN-based
models [HeZRS16](HeZRS16) or Transformer-based
models [dosovitskiy2020vit](https://openreview.net/forum?id=YicbFdNTTy), [Liu_2021_ICCV](Liu_2021_ICCV) can be used as
the encoder network. In this study, we use Swin
Transformer [Liu_2021_ICCV](Liu_2021_ICCV) because it shows the best
performance in our preliminary study in document parsing. Swin
Transformer first splits the input image $\mathbf{x}$ into
non-overlapping patches. Swin Transformer blocks, consist of a shifted
window-based multi-head self-attention module and a two-layer MLP, are
applied to the patches. Then, patch merging layers are applied to the
patch tokens at each stage. The output of the final Swin Transformer
block $\{\mathbf{z}\}$ is fed into the following textual decoder.

### Decoder.

Given the $\{\mathbf{z}\}$, the textual decoder generates a token
sequence $(\mathbf{y}_{i})_{i=1}^{m}$, where
$\mathbf{y}_{i}{\in}\mathbb{R}^{v}$ is an one-hot vector for the $i$-th
token, $v$ is the size of token vocabulary, and $m$ is a hyperparameter,
respectively. We use BART [lewis-etal-2020-bart](https://aclanthology.org/2020.acl-main.703) as the
decoder architecture. Specifically, we initialize the decoder model
weights with those from the publicly available[^1] pre-trained
multi-lingual BART model[liu-etal-2020](https://aclanthology.org/2020.tacl-1.47).

### Model Input.

Following the original
Transformer [vaswani2017transformer](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), we use a
teacher-forcing scheme [williams1989learning](williams1989learning), which is a
model training strategy that uses the ground truth as input instead of
model output from a previous time step. In the test phase, inspired by
GPT-3 [NEURIPS2020_1457c0d6](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf), the model generates a token
sequence given a prompt. We add new special tokens for the prompt for
each downstream task in our experiments. The prompts that we use for our
applications are shown with the desired output sequences in
Figure <a href="#fig:teaser" data-reference-type="ref"
data-reference="fig:teaser">[fig:teaser]</a>. Illustrative explanations
for the teacher-forcing strategy and the decoder output format are
available in
Appendix <a href="#sec:detail_of_scheme_and_format" data-reference-type="ref"
data-reference="sec:detail_of_scheme_and_format">[sec:detail_of_scheme_and_format]</a>.

### Output Conversion.

The output token sequence is converted to a desired structured format.
We adopt a JSON format due to its high representation capacity. As shown
in Figure <a href="#fig:teaser" data-reference-type="ref"
data-reference="fig:teaser">[fig:teaser]</a>, a token sequence is
one-to-one invertible to a JSON data. We simply add two special tokens
`[START_`$\ast$`]` and `[END_`$\ast$`]`, where $\ast$ indicates each
field to extract. If the output token sequence is wrongly structured, we
simply treat the field is lost. For example, if there is only
`[START_name]` exists but no `[END_name]`, we assume the model fails to
extract “name” field. This algorithm can easily be implemented with
simple regular expressions [Friedl06](https://www.safaribooksonline.com/library/view/mastering-regular-expressions/0596528124/).

## Pre-training

### Task. [sec:pretraining]

The model is trained to read all texts in the image in reading order
(from top-left to bottom-right, basically). The objective is to minimize
cross-entropy loss of next token prediction by jointly conditioning on
the image and previous contexts. This task can be interpreted as a
pseudo-OCR task. The model is trained as a visual language model over
the visual corpora, i.e., document images.

### Visual Corpora.

We use IIT-CDIP [iitcdip](https://doi.org/10.1145/1148170.1148307), which is a set of 11M scanned
english document images. A commercial CLOVA OCR API is applied to get
the pseudo text labels. As aforementioned, however, this kind of dataset
is not always available, especially for languages other than English. To
alleviate the dependencies, we build a scalable ***Synth**etic
**Do**cument **G**enerator*, referred to as **SynthDoG**. Using the
SynthDog and Chinese, Japanese, Korean and English Wikipedia, we
generated 0.5M samples per language.

<figure id="fig:synthdog">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/fig4.png" style="width:70.0%" />
<figcaption><span><strong>Generated English, Chinese, Japanese, and
Korean samples with <strong>SynthDoG</strong>.</strong></span> Heuristic
random patterns are applied to mimic the real documents</figcaption>
</figure>

### Synthetic Document Generator.

The pipeline of image rendering basically follows Yim et al.
[synthtiger](synthtiger). As shown in
Figure <a href="#fig:synthdog" data-reference-type="ref"
data-reference="fig:synthdog">1</a>, the generated sample consists of
several components; background, document, text, and layout. Background
image is sampled from ImageNet [deng2009imagenet](deng2009imagenet), and a
texture of document is sampled from the collected paper photos. Words
and phrases are sampled from Wikipedia. Layout is generated by a simple
rule-based algorithm that randomly stacks grids. In addition, several
image rendering
techniques [Gupta16](Gupta16), [long2020unrealtext](long2020unrealtext), [synthtiger](synthtiger) are
applied to mimic real documents. The generated examples are shown in
Figure <a href="#fig:synthdog" data-reference-type="ref"
data-reference="fig:synthdog">1</a>. More details of SynthDoG are
available in the code and
Appendix <a href="#sec:detail_of_synthdog" data-reference-type="ref"
data-reference="sec:detail_of_synthdog">[sec:detail_of_synthdog]</a>.

## Fine-tuning

After the model learns *how to read*, in the application stage (i.e.,
fine-tuning), we teach the model *how to understand* the document image.
As shown in Figure <a href="#fig:teaser" data-reference-type="ref"
data-reference="fig:teaser">[fig:teaser]</a>, we interpret all
downstream tasks as a JSON prediction problem.

The decoder is trained to generate a token sequence that can be
converted into a JSON that represents the desired output information.
For example, in the document classification task, the decoder is trained
to generate a token sequence `[START_class][memo][END_class]` which is
1-to-1 invertible to a JSON {“class”: “memo”}. We introduce some special
tokens (e.g., `[memo]` is used for representing the class “memo”), if
such replacement is available in the target task.

[^1]: <https://huggingface.co/hyunwoongko/asian-bart-ecjk>.

In this section, we present fine-tuning results on three VDU
applications on six different datasets including both public benchmarks
and private industrial service datasets. The samples are shown in
Figure <a href="#fig:datasets" data-reference-type="ref"
data-reference="fig:datasets">[fig:datasets]</a>.

<div class="figure*" markdown="1">

<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/fig5.png" />

</div>

## Downstream Tasks and Datasets

### Document Classification.

To see whether the model can distinguish across different types of
documents, we test a classification task. Unlike other models that
predict the class label via a softmax on the encoded embedding, generate
a JSON that contains class information to maintain the uniformity of the
task-solving method. We report overall classification accuracy on a test
set.

#### RVL-CDIP.

The RVL-CDIP dataset [harley2015icdar](harley2015icdar) consists of 400K
images in 16 classes, with 25K images per class. The classes include
letter, memo, email, and so on. There are 320K training, 40K validation,
and 40K test images.

### Document Information Extraction.

To see the model fully understands the complex layouts and contexts in
documents, we test document information extraction (IE) tasks on various
real document images including both public benchmarks and real
industrial datasets. In this task, the model aims to map each document
to a structured form of information that is consistent with the target
ontology or database schema. See
Figure <a href="#fig:problem_definition" data-reference-type="ref"
data-reference="fig:problem_definition">[fig:problem_definition]</a> for
an illustrative example. The model should not only read the characters
well, but also understand the layouts and semantics to infer the groups
and nested hierarchies among the texts.

We evaluate the models with two metrics; field-level F1
score [hwang2019pot](hwang2019pot), [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172), [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322) and
Tree Edit Distance (TED) based
accuracy [ted](ted), [teds](teds), [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271). The F1 checks
whether the extracted field information is in the ground truth. Even if
a single character is missed, the score assumes the field extraction is
failed. Although F1 is simple and easy to understand, there are some
limitations. First, it does not take into account partial overlaps.
Second, it can not measure the predicted structure (e.g., groups and
nested hierarchy). To assess overall accuracy, we also use another
metric based on TED [ted](ted), that can be used for any
documents represented as trees. It is calculated as,
$\max(0, 1-\text{TED}(\text{pr},\text{gt})/\text{TED}(\phi,\text{gt}))$,
where $\text{gt}$, $\text{pr}$, and $\phi$ stands for ground truth,
predicted, and empty trees respectively. Similar metrics are used in
recent works on document IE [teds](teds), [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271)

We use two public benchmark datasets as well as two private industrial
datasets which are from our active real-world service products. Each
dataset is explained in the followings.

#### CORD.

The Consolidated Receipt Dataset (CORD)[^1][park2019cord](park2019cord)
is a public benchmark that consists of 0.8K train, 0.1K valid, 0.1K test
receipt images. The letters of receipts is in Latin alphabet. The number
of unique fields is 30 containing menu name, count, total price, and so
on. There are complex structures (i.e., nested groups and hierarchies
such as `items>item>``{``name, count, price``}`) in the information. See
Figure <a href="#fig:problem_definition" data-reference-type="ref"
data-reference="fig:problem_definition">[fig:problem_definition]</a> for
more details.

#### Ticket.

This is a public benchmark dataset [eaten](eaten) that consists
of 1.5K train and 0.4K test Chinese train ticket images. We split 10% of
the train set as a validation set. There are 8 fields which are ticket
number, starting station, train number, and so on. The structure of
information is simple and all keys are guaranteed to appear only once
and the location of each field is fixed.

#### Business Card (In-Service Data).

This dataset is from our active products that are currently deployed.
The dataset consists of 20K train, 0.3K valid, 0.3K test Japanese
business cards. The number of fields is 11, including name, company,
address, and so on. The structure of information is similar to the
*Ticket* dataset.

#### Receipt (In-Service Data).

This dataset is also from one of our real products. The dataset consists
of 40K train, 1K valid, 1K test Korean receipt images. The number of
unique field is 81, which includes store information, payment
information, price information, and so on. Each sample has complex
structures compared to the aforementioned datasets. Due to industrial
policies, not all samples can publicly be available. Some real-like
high-quality samples are shown in
Figure <a href="#fig:datasets" data-reference-type="ref"
data-reference="fig:datasets">[fig:datasets]</a> and in the
supplementary material.

### Document Visual Question Answering.

To validate the further capacity of the model, we conduct a document
visual question answering task (DocVQA). In this task, a document image
and question pair is given and the model predicts the answer for the
question by capturing both visual and textual information within the
image. We make the decoder generate the answer by setting the question
as a starting prompt to keep the uniformity of the method (See
Figure <a href="#fig:teaser" data-reference-type="ref"
data-reference="fig:teaser">[fig:teaser]</a>).

#### DocVQA.

The dataset is from Document Visual Question Answering competition[^2]
and consists of 50K questions defined on more than 12K
documents [mathew2021docvqa](mathew2021docvqa). There are 40K train, 5K
valid, and 5K test questions. The evaluation metric is ANLS (Average
Normalized Levenshtein Similarity) which is an edit-distance-based
metric. The score on the test set is measured via the evaluation site.

## Setups

We use Swin-B [Liu_2021_ICCV](Liu_2021_ICCV) as a visual encoder of with
slight modification. We set the layer numbers and window size as
$\{2, 2, 14, 2\}$ and 10. In further consideration of the speed-accuracy
trade-off, we use the first four layers of BART as a decoder. As
explained in
Section <a href="#sec:pretraining" data-reference-type="ref"
data-reference="sec:pretraining">[sec:pretraining]</a>, we train the
multi-lingual using the 2M synthetic and 11M IIT-CDIP scanned document
images. We pre-train the model for 200K steps with 64 A100 GPUs and a
mini-batch size of 196. We use Adam [Adamoptim](http://arxiv.org/abs/1412.6980)
optimizer, the learning rate is scheduled and the initial rate is
selected from 1e-5 to 1e-4. The input resolution is set to
2560$\times$`<!-- -->`{=html}1920 and a max length in the decoder is set
to 1536. All fine-tuning results are achieved by starting from the
pre-trained multi-lingual model. Some hyperparameters are adjusted at
fine-tuning and in ablation studies. We use
960$\times$`<!-- -->`{=html}1280 for Train Tickets and Business Card
parsing tasks. We fine-tune the model while monitoring the edit distance
over token sequences. The speed of is measured on a P40 GPU, which is
much slower than A100. For the OCR based baselines, states-of-the-art
OCR engines are used, including MS OCR API used in
[xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201) and CLOVA OCR API[^3] used in
[hwang2020spade](https://aclanthology.org/2021.findings-acl.28), [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271). An analysis on
OCR engines is available in
Section <a href="#sec:ablation_and_analysis" data-reference-type="ref"
data-reference="sec:ablation_and_analysis">[sec:ablation_and_analysis]</a>.
More details of OCR and training setups are available in
Appendix <a href="#sec:detail_of_ocr_engines" data-reference-type="ref"
data-reference="sec:detail_of_ocr_engines">[sec:detail_of_ocr_engines]</a>
and <a href="#sec:detail_of_implementation_and_hyperparams"
data-reference-type="ref"
data-reference="sec:detail_of_implementation_and_hyperparams">[sec:detail_of_implementation_and_hyperparams]</a>.

<div class="table*" markdown="1">

<div class="threeparttable" markdown="1">

|                     | OCR |        \#Params        | Time (ms) | Accuracy (%) |
|:--------------------|:---:|:----------------------:|:---------:|:------------:|
| BERT                |     | 110M + $\alpha^{\dag}$ |   1392    |    89.81     |
| RoBERTa             |     | 125M + $\alpha^{\dag}$ |   1392    |    90.06     |
| LayoutLM            |     | 113M + $\alpha^{\dag}$ |   1396    |    91.78     |
| LayoutLM (w/ image) |     | 160M + $\alpha^{\dag}$ |   1426    |    94.42     |
| LayoutLMv2          |     | 200M + $\alpha^{\dag}$ |   1489    |    95.25     |
| **(Proposed)**      |     |          143M          |  **752**  |  **95.30**   |

</div>

</div>

## Experimental Results

### Document Classification.

The results are shown in
Table <a href="#tbl:docclass" data-reference-type="ref"
data-reference="tbl:docclass">[tbl:docclass]</a>. Without relying on any
other resource (e.g., off-the-shelf OCR engine), shows a
state-of-the-art performance among the general-purpose VDU models such
as LayoutLM [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172) and
LayoutLMv2 [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201). In particular,
surpasses the LayoutLMv2 accuracy reported in
[xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), while using fewer parameters
with the 2x faster speed. Note that the OCR-based models must consider
additional model parameters and speed for the entire OCR framework,
which is not small in general. For example, a recent advanced OCR-based
model [baek2019craft](baek2019craft), [baek2019wrong](baek2019wrong) requires more than
80M parameters. Also, training and maintaining the OCR-based systems are
costly [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271), leading to needs for the
-like end-to-end approach.

<div class="table*" markdown="1">

<div class="adjustbox" markdown="1">

max width=

<div class="threeparttable" markdown="1">

|  |  |  | CORD [park2019cord](park2019cord) |  |  | Ticket [eaten](eaten) |  |  | Business Card |  |  | Receipt |  |  |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 4-6(lr)7-9(lr)10-12(lr)13-15 | OCR | \#Params | Time (s) | F1 | Acc. | Time (s) | F1 | Acc. | Time (s) | F1 | Acc. | Time (s) | F1 | Acc. |
| BERT$^{\ast}$ [hwang2019pot](hwang2019pot) |  | $86^{\dag}_{\text{M}}+\alpha^{\ddag}$ | 1.6 | 73.0 | 65.5 | 1.7 | 74.3 | 82.4 | 1.5 | 40.8 | 72.1 | 2.5 | 70.3 | 54.1 |
| BROS [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322) |  | $86^{\dag}_{\text{M}}+\alpha^{\ddag}$ | 1.7 | 74.7 | 70.0 |  |  |  |  |  |  |  |  |  |
| LayoutLM [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172) |  | $89^{\dag}_{\text{M}}+\alpha^{\ddag}$ | 1.7 | 78.4 | 81.3 |  |  |  |  |  |  |  |  |  |
| LayoutLMv2$^{\ast}$ [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), [layoutxlm](layoutxlm) |  | $179^{\dag}_{\text{M}}+\alpha^{\ddag}$ | 1.7 | 78.9 | 82.4 | 1.8 | 87.2 | 90.1 | 1.6 | 52.2 | 83.0 | 2.6 | 72.9 | 78.0 |
|  |  | $143^{\dag}_{\text{M}}$ | **1.2** | **84.1** | **90.9** | **0.6** | **94.1** | **98.7** | **1.4** | **57.8** | **84.4** | **1.9** | **78.6** | **88.6** |
| SPADE$^{\ast}$ [hwang-etal-2021-spatial](https://aclanthology.org/2021.findings-acl.28) |  | $93^{\dag}_{\text{M}} + \alpha^{\ddag}$ | 4.0 | 74.0 | 75.8 | 4.5 | 14.9 | 29.4 | 4.3 | 32.3 | 51.3 | 7.3 | 64.1 | 53.2 |
| WYVERN$^{\ast}$ [hwang-etal-2020-towards](https://aclanthology.org/2020.vardial-1.15) |  | $106^{\dag}_{\text{M}} + \alpha^{\ddag}$ | 1.2 | 43.3 | 46.9 | 1.5 | 41.8 | 54.8 | 1.7 | 29.9 | 51.5 | 3.4 | 71.5 | 82.9 |

</div>

</div>

</div>

### Document Information Extraction.

Table <a href="#tbl:information_extraction" data-reference-type="ref"
data-reference="tbl:information_extraction">[tbl:information_extraction]</a>
shows the results on the four different document IE tasks. The first
group uses a conventional BIO-tagging-based IE
approach [hwang2019pot](hwang2019pot). We follows the conventions in
IE [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172), [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322). OCR extracts texts and
bounding boxes from the image, and then the serialization module sorts
all texts with geometry information within the bounding box. The
BIO-tagging-based named entity recognition task performs token-level tag
classification upon the ordered texts to generate a structured form. We
test three general-purpose VDU backbones,
BERT [devlinBERT2018](https://aclanthology.org/N19-1423),
BROS [hong2021bros](https://ojs.aaai.org/index.php/AAAI/article/view/21322),
LayoutLM [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172), and
LayoutLMv2 [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), [layoutxlm](layoutxlm).

We also test two recently proposed IE models,
SPADE [hwang2020spade](https://aclanthology.org/2021.findings-acl.28) and
WYVERN [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271). SPADE is a graph-based
IE method that predicts relations between bounding boxes. WYVERN is an
Transformer encoder-decoder model that directly generates entities with
structure given OCR outputs. WYVERN is different from in that it takes
the OCR output as its inputs.

For all domains, including public and private in-service datasets, shows
the best scores among the comparing models. By measuring both F1 and
TED-based accuracy, we observe not only can extract key information but
also predict complex structures among the field information. We observe
that a large input resolution gives robust accuracies but makes the
model slower. For example, the performance on the CORD with
1280$\times$`<!-- -->`{=html}960 was 0.7 sec./image and 91.1 accuracy.
But, the large resolution showed better performances on the low-resource
situation. The detailed analyses are in
Section <a href="#sec:ablation_and_analysis" data-reference-type="ref"
data-reference="sec:ablation_and_analysis">[sec:ablation_and_analysis]</a>.
Unlike other baselines, shows stable performance regardless of the size
of datasets and complexity of the tasks (See
Figure <a href="#fig:datasets" data-reference-type="ref"
data-reference="fig:datasets">[fig:datasets]</a>). This is a significant
impact as the target tasks are already actively used in industries.

### Document Visual Question Answering.

Table <a href="#tbl:docvqa" data-reference-type="ref"
data-reference="tbl:docvqa">1</a> shows the results on the DocVQA
dataset. The first group is the general-purposed VDU backbones whose
scores are from the LayoutLMv2
paper [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201). We measure the running
time with MS OCR API used in [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201).
The model in the third group is a DocVQA-specific-purposed fine-tuning
model of LayoutLMv2, whose inference results are available in the
official leader-board.[^4]

As can be seen, achieves competitive scores with the baselines that are
dependent on external OCR engines. Especially, shows that it is robust
to the handwritten documents, which is known to be challenging to
process. In the conventional approach, adding a post-processing module
that corrects OCR errors is an option to strengthen the
pipeline [schaefer-neudecker-2020-two](https://aclanthology.org/2020.latechclfl-1.6), [rijhwani-etal-2020-ocr](https://aclanthology.org/2020.emnlp-main.478), [duong-etal-2021-unsupervised](https://aclanthology.org/2021.nodalida-main.24)
or adopting an encoder-decoder architecture on the OCR outputs can
mitigate the problems of OCR
errors [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271). However, this kind of
approaches tend to increase the entire system size and maintenance cost.
shows a completely different direction. Some inference results are shown
in Figure <a href="#fig:doc_vqa_example" data-reference-type="ref"
data-reference="fig:doc_vqa_example">1</a>. The samples show the current
strengths of as well as the left challenges in the -like end-to-end
approach. Further analysis and ablation is available in
Section <a href="#sec:ablation_and_analysis" data-reference-type="ref"
data-reference="sec:ablation_and_analysis">[sec:ablation_and_analysis]</a>.

<div class="adjustbox" markdown="1">

max width=

<div class="threeparttable" markdown="1">

<div id="tbl:docvqa" markdown="1">

|  | Fine-tuning set | OCR | \#Params$^{\dag}$ | Time (ms) | $^{\text{ANLS}^{\:}}_{\text{test set}}$ | $^{\text{ANLS}^\ast}_{\text{handwritten}}$ |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| BERT [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201) | train set |  | 110M + $\alpha^{\ddag}$ | 1517 | 63.5 | n/a |
| LayoutLM[xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172) | train set |  | 113M + $\alpha^{\ddag}$ | 1519 | 69.8 | n/a |
| LayoutLMv2[xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201) | train set |  | 200M + $\alpha^{\ddag}$ | 1610 | 78.1 | n/a |
|  | train set |  | 176M | **782** | 67.5 | **72.1** |
| LayoutLMv2-Large-QG[xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201) | train + dev + QG |  | 390M + $\alpha^{\ddag}$ | 1698 | **86.7** | 67.3 |

**Average Normalized Levenshtein Similarity (ANLS) scores on DocVQA.**
shows a promising result without OCR. $^{\ast}$shows a high ANLS score
on the handwritten documents which are known to be challenging due to
the difficulty of handwriting OCR (See
Figure <a href="#fig:doc_vqa_example" data-reference-type="ref"
data-reference="fig:doc_vqa_example">1</a>). $^\dag$Token embeddings for
English is counted for a fair comparison. $^\ddag$\# parameters for OCR
should be considered

</div>

</div>

</div>

<figure id="fig:doc_vqa_example">
<embed src="/papers/visionrich_small_dec/arXiv-2111.15664v5_md/figures/fig6.png" />
<figcaption><span><strong>Examples of Donut and LayoutLMv2 outputs on
DocVQA.</strong></span> The OCR-errors make a performance upper-bound of
the OCR-dependent baselines, e.g., LayoutLMv2 (left and middle
examples). Due to the input resolution constraint of the end-to-end
pipeline, Donut miss some tiny texts in large-scale images (right
example) but this could be mitigated by scaling the input image size
(See Section <a href="#sec:ablation_and_analysis"
data-reference-type="ref"
data-reference="sec:ablation_and_analysis">[sec:ablation_and_analysis]</a>)</figcaption>
</figure>

[^1]: <https://huggingface.co/datasets/naver-clova-ix/cord-v1>.

[^2]: <https://rrc.cvc.uab.es/?ch=17>.

[^3]: <https://clova.ai/ocr>.

[^4]: <https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1>.

## Optical Character Recognition

Recent trends of OCR study are to utilize deep learning models in its
two sub-steps: 1) text areas are predicted by a detector; 2) a text
recognizer then recognizes all characters in the cropped image
instances. Both are trained with a large-scale datasets including the
synthetic images [Jaderberg14c](Jaderberg14c), [Gupta16](Gupta16) and real
images [7333942](7333942), [Phan_2013_ICCV](Phan_2013_ICCV).

Early detection methods used CNNs to predict local segments and apply
heuristics to merge
them [Huang10.1007/978-3-319-10593-2_33](Huang10.1007/978-3-319-10593-2_33), [Zhang_2016_CVPR](Zhang_2016_CVPR).
Later, region proposal and bounding box regression based methods were
proposed [LiaoSBWL17](https://ojs.aaai.org/index.php/AAAI/article/view/11196). Recently, focusing on the
homogeneity and locality of texts, component-level approaches were
proposed [CTPN](CTPN), [baek2019craft](baek2019craft).

Many modern text recognizer share a similar
approach [starnet](https://dx.doi.org/10.5244/C.30.43), [Shi2016RobustST](Shi2016RobustST), [Shi2017AnET](Shi2017AnET), [jianfeng2017deep](https://proceedings.neurips.cc/paper/2017/file/c24cd76e1ce41366a4bbe8a49b02a028-Paper.pdf)
that can be interpreted into a combination of several common deep
modules [baek2019wrong](baek2019wrong). Given the cropped text instance
image, most recent text recognition models apply CNNs to encode the
image into a feature space. A decoder is then applied to extract
characters from the features.

## Visual Document Understanding

Classification of the document type is a core step towards automated
document processing. Early methods treated the problem as a general
image classification, so various CNNs were
tested [Kang2014ConvolutionalNN](Kang2014ConvolutionalNN), [7333933](7333933), [7333910](7333910).
Recently, with BERT [devlinBERT2018](https://aclanthology.org/N19-1423), the methods based
on a combination of CV and NLP were widely
proposed [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172), [li-etal-2021-structurallm](https://aclanthology.org/2021.acl-long.493). As
a common approach, most methods rely on an OCR engine to extract texts;
then the OCR-ed texts are serialized into a token sequence; finally they
are fed into a language model (e.g., BERT) with some visual features if
available. Although the idea is simple, the methods showed remarkable
performance improvements and became a main trend in recent
years [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201), [selfdoc](selfdoc), [Appalaraju_2021_ICCV](Appalaraju_2021_ICCV).

Document IE covers a wide range of real
applications [hwang2019pot](hwang2019pot), [majumder2020representation](https://www.aclweb.org/anthology/2020.acl-main.580),
for example, given a bunch of raw receipt images, a document parser can
automate a major part of receipt digitization, which has been required
numerous human-labors in the traditional pipeline. Most recent
models [hwang-etal-2021-spatial](https://aclanthology.org/2021.findings-acl.28), [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271)
take the output of OCR as their input. The OCR results are then
converted to the final parse through several processes, which are often
complex. Despite the needs in the industry, only a few works have been
attempted on end-to-end parsing. Recently, some works are proposed to
simplify the complex parsing
processes [hwang-etal-2021-spatial](https://aclanthology.org/2021.findings-acl.28), [hwang2021costeffective](https://aclanthology.org/2021.emnlp-main.271).
But they still rely on a separate OCR to extract text information.

Visual QA on documents seeks to answer questions asked on document
images. This task requires reasoning over visual elements of the image
and general knowledge to infer the correct
answer [mathew2021docvqa](mathew2021docvqa). Currently, most
state-of-the-arts follow a simple pipeline consisting of applying OCR
followed by BERT-like
transformers [xu2019_layoutLM](https://doi.org/10.1145/3394486.3403172), [xu-etal-2021-layoutlmv2](https://aclanthology.org/2021.acl-long.201).
However, the methods work in an extractive manner by their nature.
Hence, there are some concerns for the question whose answer does not
appear in the given image [icdar21docvqa](icdar21docvqa). To tackle the
concerns, generation-based methods have also been
proposed [10.1007/978-3-030-86331-9_47](10.1007/978-3-030-86331-9_47).

In this work, we propose a novel end-to-end framework for visual
document understanding. The proposed method, , directly maps an input
document image into a desired structured output. Unlike conventional
methods, does not depend on OCR and can easily be trained in an
end-to-end fashion. We also propose a synthetic document image
generator, SynthDoG, to alleviate the dependency on large-scale real
document images and we show that can be easily extended to a
multi-lingual setting. We gradually trained the model from *how to read*
to *how to understand* through the proposed training pipeline. Our
extensive experiments and analysis on both external public benchmarks
and private internal service datasets show higher performance and better
*cost-effectiveness* of the proposed method. This is a significant
impact as the target tasks are already practically used in industries.
Enhancing the pre-training objective could be a future work direction.
We believe our work can easily be extended to other domains/tasks
regarding document understanding.