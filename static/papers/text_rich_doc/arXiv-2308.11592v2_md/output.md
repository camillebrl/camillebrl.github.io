<div class="figure*" markdown="1">

<embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/fig1v2-crop.png" />

</div>

# Introduction

Nowdays, considerable advancements have been observed in the domain of
Large Language Models (LLMs), such as ChatGPT, [^1]
BLOOM [scao2022bloom](http://arxiv.org/pdf/2106.06683v2), and
LLaMA [touvron2023llama](http://arxiv.org/pdf/2402.08075v1), [touvron2023llama2](http://arxiv.org/pdf/2403.00858v4). These
developments constitute significant strides towards the achievement of
artificial general intelligence (AGI) and exhibit superior zero-shot
proficiency across various linguistic applications. By employing these
LLMs as language decoders, their Multimodal counterparts (LMMs), which
include models like BLIP [li2023blip](http://arxiv.org/pdf/2301.12597v3),
MiniGPT-4 [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1),
LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1), and
mPLUG-Owl [ye2023mplug](http://arxiv.org/pdf/2405.00390v2), have showcased noteworthy
efficacy in understanding visual and linguistic data.

While these large multimodal models exhibit astonishing zero-shot
multimodal understanding capabilities, their comprehension of text-rich
images remains limited [liu2023hidden](http://arxiv.org/pdf/2305.07895v5). To address this
gap, LLaVAR [zhang2023LLaVAR](zhang2023LLaVAR) proposes incorporating a
text recognition pre-training task to enhance the understanding of
text-rich images. Besides, mPLUG-DocOwl [ye2023mplug](http://arxiv.org/pdf/2405.00390v2)
constructs a large-scale dataset about the document image understanding.
Although their text-rich scene understanding capabilities have shown
notable promise, the vast potential of these pretrained large visual and
language models remains largely unexplored and underutilized, analyzed
next.

Firstly, a salient absence of text detection capabilities is observed in
the current large multimodal models. Since these large visual and
linguistic models are pre-trained on extremely large-scale datasets,
they possess powerful representational capabilities and a wealth of
world knowledge, endowing them with the ability to localize objects/text
in images. Their potential can be further harnessed and explored.
Secondly, the training strategies of advanced methods suffer from data
distribution inconsistencies between the pre-training and fine-tuning
phases [brown2020language](http://arxiv.org/pdf/2112.07522v2), leading to suboptimal
performance. Typically, LLaVAR [zhang2023LLaVAR](zhang2023LLaVAR) solely
conducts text recognition tasks during the pre-training phase and
proceeds with document understanding training in the fine-tuning phase.
Thirdly, text detection and recognition inherently fall under the
umbrella of high-level scene understanding tasks, with the location and
content of the text being associated with scene semantics. Existing LMMs
for text-rich image understanding have not effectively capitalized on
these beneficial connections among OCR
tasks [li2017towards](http://arxiv.org/pdf/1707.03985v1) to enhance the performance on the
individual tasks.

Formally, we introduce UniDoc, a universal large multimodal model for
simultaneous text detection, recognition, spotting, and understanding.
UniDoc aims to establish comprehensive OCR and multimodal understanding
capabilities tailored for text-rich images. We integrate all these tasks
into a cohesive framework driven by natural language instructions for
multimodal understanding, as shown in
Fig. <a href="#fig1" data-reference-type="ref"
data-reference="fig1">[fig1]</a>. Based on such a unified multimodal
instruct tuning, not only have we endowed our UniDoc with various OCR
capabilities, but the beneficial interactions among these tasks have
also enhanced the performance across individual task. To implement our
UniDoc, we collected and annotated a large-scale instruction following
dataset for this tasks. Extensive quantitative and qualitative
experimental results demonstrate the superior performance of UniDoc and
its strong generalization ability. To our best knowledge, this is the
first large multimodal model capable of simultaneous text detection,
recognition, spotting, and understanding.

In summary, we make three-fold contributions as follows:

-   We introduce UniDoc, the first large multimodal model capable of
    simultaneous text detection, recognition, spotting, and multimodal
    understanding of text-rich images.

-   We contribute a large-scale multimodal instruction tuning dataset,
    tailored for tasks of text detection, recognition, and spotting
    within text-rich images.

-   We achieve state-of-the-art performance on multiple publicly
    available benchmark datasets. Moreover, we conduct extensive
    quantitative and qualitative experiments to validate the
    effectiveness of UniDoc.

# Related Work

In this section, we broadly review the recent research on instruction
tuning and multimodal instruction tuning.

<div class="figure*" markdown="1">

<embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/frameworkv3-crop.png" />

</div>

## Instruction Tuning

Instruction tuning is an effective technique to align large language
models (LLMs) with human intents. It aims to teach language models to
follow natural language (including prompt, positive or negative
examples, and constraints etc.), to perform better multi-task learning
on training tasks and generalization on unseen tasks. Recently, models
like GPT-3 [brown2020language](http://arxiv.org/pdf/2112.07522v2) and others have
significantly leveraged instructional fine-tuning. Typically, Stanford’s
Alpaca [alpaca](https://github.com/tatsu-lab/stanford_alpaca) employs
self-instruction [wang2022self](http://arxiv.org/pdf/2311.00233v2) to provide a
cost-effective approach to obtain instruction data for fine-tuning
LLaMA. Vicuna [chiang2023vicuna](None) that is a instructional
fine-tuned LLaMA based on dialogues between users and ChatGPT, achieves
performance comparable to ChatGPT [zheng2023judging](https://arxiv.org/pdf/2306.05685).

## Multimodal Instruction Tuning

Recent advancements in the confluence of natural language processing and
computer vision have seen the rise of Large Multimodal Models (LMMs),
which integrate large language models and visual encoders to address
complex tasks involving both text and vision. Prominent works in this
domain include MiniGPT-4 [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1), which fuses
components from BLIP-2 [li2023blip](http://arxiv.org/pdf/2301.12597v3) and
Vicuna [chiang2023vicuna](None) for modality mapping and adopts
a two-stage fine-tuning strategy. The LLaVA model, on the other hand,
employs a supplementary linear layer to map visual features to the text
space and undergoes additional fine-tuning under multimodal
instructions. In the same vein, mPLUG-Owl from Alibaba’s DAMO Academy
incorporates Flamingo’s Perceiver Resampler structure to facilitate
visual and language modalities alignment. Another significant
contribution is from InstructBLIP, which introduces a novel multimodal
instruction dataset and uses Q-Former and Vicuna as an image encoder and
language model respectively. Finally, X-LLM has introduced a Chinese
multimodal instruction dataset and employs several adapters to map
different modalities to the text space. While these multimodal large
models exhibit promising visual-linguistic understanding capabilities,
their potential are yet to be fully harnessed in specific domains.

To bridge this divide, LLaVAR [zhang2023LLaVAR](zhang2023LLaVAR) puts
forward the inclusion of a text recognition pre-training task, thus
bolstering the comprehension of text-heavy imagery. In addition,
mPLUG-DocOwl [ye2023mplug](http://arxiv.org/pdf/2405.00390v2) has compiled an expansive
dataset designed specifically for the fine-tuning of document
comprehension tasks. Shikra [chen2023shikra](http://arxiv.org/pdf/2306.15195v2) integrates
LMMs with visual grounding ability by recasting detection task as a
prompt-guided seq2seq task. Although these approaches somewhat augment
the multimodal comprehension ability of models in text-rich scenarios,
they fall short in offering a comprehensive ability for text detection,
recognition and spotting. Moreover, they do not effectively harness the
potential reciprocal enhancements that could be achieved by learning
these capabilities in tandem.

# Methodology

## Model Architecture

Fig. <a href="#frame" data-reference-type="ref"
data-reference="frame">[frame]</a> presents an overview of our UniDoc.
Our design follows the paradigm established by
MiniGPT-4 [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1) and
LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1).

Specifically, given an input *RGB* image
$\bm{I} \in \mathbb{R}^{H\times W\times3}$ and a natural language
instruction $\bm{Q}$, UniDoc first abstracts the visual features from
$\bm{I}$ utilizing CLIP-ViT-L/14 [radford2021learning](http://arxiv.org/pdf/2404.19696v1) as
the visual encoder. Both pre- and post- Transformer layer grid features
are incorporated in our method. The extracted feature map is then
flattened into a sequence of visual embedding sequence and projected
into the embedding dimension of the LLM with a linear layer. The output
sequence $\bm{E}_v$ and then concatenated with embedding sequence
$\bm{E}_l$ tokenized from the language instruction $\bm{Q}$.

Thereafter, the concatenated embedding sequence are fed into
Vicuna [chiang2023vicuna](None), a large language model
originating from the LLaMA [touvron2023llama](http://arxiv.org/pdf/2402.08075v1) and
specifically tuned with the instruction following data.
Vicuna [chiang2023vicuna](None) then generates the response
based on the received visual and text cues. Note that the visual
embedding here can be considered as a soft prompt for LLM.

## Unified Multimodal Instruct Tuning

Our training process is divided into two stages. Both stages employ our
unified multimodal instruct tuning. The first pre-training phase aims to
align the output features from the pre-trained visual encoder with the
feature space of the large language model. During the second fine-tuning
stage, we further optimize the weights of the large language model.

Concretely, during the pre-training phase, we freeze both the
pre-trained large visual and language models, training only the linear
projector to align the visual and language features. Our instruction
following data involves four tasks: text detection, recognition,
spotting, and image captioning. We argue that detection, recognition,
and spotting inherently involve high-level semantic understanding, as
the position and content of text within an image often have a strong
correlation with their surrounding context. The image captioning task
enhances the model’s understanding of natural scene images. All of these
tasks were performed in a natural language instruction following manner.

<div id="data_distri" markdown="1">

| **Satge** | **Data** | **Image** | **Instruction** | **\# Conv** | **Task** |
|:---|:---|:--:|:--:|:--:|:--:|
| Pre-train | LLaVA | CC3M | CC3M | 595K | $\mathcal{C}$ |
|  | UniDoc | LAION | OCR | 600K | $\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{C}$ |
| Fine-tune | LLaVA | COCO | GPT-4 | 158K | $\mathcal{U}$ |
|  | LLaVAR | LAION | GPT-4 | 16K | $\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{U}$ |
|  | UniDoc | LAION | GPT-4 + OCR | 186K | $\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{U}$ |

Summary of the dataset statistics. The symbols
$\mathcal{C},\mathcal{D},\mathcal{R},\mathcal{S},\mathcal{U}$ correspond
to the different instruction following tasks, namely, captioning,
detection, recognition, spotting, and multimodal understanding.

</div>

<div class="table*" markdown="1">

</div>

In the fine-tuning phase, we unfreeze both the large language model and
the projector. Besides the training tasks involved in the pre-training
stage, we further incorporate an additional multimodal understanding
task for text-rich images which requires a more advanced level of
semantic comprehension. The learning of these tasks mutually enhance
each other. Through this unified multi-modal unified instruction
fine-tuning, UniDoc achieves a comprehensive recognition and
understanding capability for text-rich scenarios.

# Dataset Construction

To train the UniDoc, we construct a large-scale multimodal instruction
following dataset. We detail it in the following.

**Pre-training.** The pre-training data consists of two parts: one
portion includes 595K natural scene images along with their captions,
sourced from the CC3M dataset and filtered by
LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1); the other portion comprises 600K
image-text pairs from PowerPoint presentations that we created. The data
were collected from the “Common Crawl" dataset, a vast web corpus
containing publicly available web page. [^2] We opt for PowerPoint files
based on two primary considerations. On one hand, PowerPoint
presentations are characterized by a rich assortment of elements and
their complex combinations, such as various fonts, images, tables, as
shown in Fig. <a href="#dataset" data-reference-type="ref"
data-reference="dataset">1</a>. These elements are interrelated, making
them highly conducive to training multimodal understanding tasks in
text-rich scenarios. On the other hand, the text within the slides is
relatively large, making it legible for existing pre-trained visual
models [radford2021learning](http://arxiv.org/pdf/2404.19696v1). In other words, if the text
in an image is too small, it becomes unrecognizable when input into the
model.

To ensure high-quality visuals suitable for our purposes, we conducted
rigorous quality assurance checks, eliminating the noisy data to avoid
any negative impact on training. Specifically, we first applied text
size optimization, excluding images with small-sized text. Then, an
in-house OCR tool accurately extracts the text and box annotations from
each image and we constructed OCR instruction based on them. The
instructions here are categorized into three types: text detection,
recognition, and understanding. Furthermore, we employed GPT-4 to
generate diverse expressions for each type of instruction. The data for
detection, recognition, and spotting each account for one-third of the
total.

<figure id="dataset">
<embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/dataset.png" />
<figcaption>Example instances from the proposed dataset, featuring
diverse fonts in terms of size, style, and color, and a rich array of
visual elements.</figcaption>
</figure>

**Fine-tuning.** During fine-tuning, we extend the 16K instruction
following data collected from
LAION-5B [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1) and constructed by
LLaVAR [zhang2023LLaVAR](zhang2023LLaVAR). Initially, we curated this
dataset, employing the same cleansing methodology as used for the
pre-training set. Subsequently, for each image, we constructed OCR
instruction following data, adhering to the approach established during
the pre-training phase. The data for detection, recognition, and
spotting each account for one-third of the total. Furthermore, we
further incorporated 150K OCR instruction data as the pre-training
stage, in which detection, recognition, and spotting each constitute
one-third of the total.

<div class="figure*" markdown="1">

<embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/detect-recognize-ctw1500.png" />

</div>

<div class="figure*" markdown="1">

<embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/recognize-wordart-totaltext.png" />

</div>

# Experiments

## Training Details

To implement UniDoc, we employed a one-cycle learning rate
policy [smith2019super](http://arxiv.org/pdf/1708.07120v3). During the pre-training phase,
the maximum learning rate was set to 1e-3, and for the fine-tuning
phase, it was reduced to 1e-5. Moreover, the batch size was 128 for the
pre-training and 32 for the fine-tuning phase, respectively. The
AdamW [loshchilov2017decoupled](http://arxiv.org/pdf/2311.11446v2) optimizer was chosen for
weight updates. Both the pre-training and fine-tuning phases were
executed using eight A100 GPUs. Each of these phases consisted of a
single epoch. In this study, for both the training and inference phases,
the default input image resolution is set at
224$\times$`<!-- -->`{=html}224. It is noteworthy that larger input
resolutions are almost certain to yield better results due to the
presence of more discernible
text [zhang2023LLaVAR](zhang2023LLaVAR), [ye2023mplug-doc](http://arxiv.org/pdf/2403.14252v1). Unless otherwise
specified, the performance reported in this study is based on image
inputs with an input resolution of 224$\times$`<!-- -->`{=html}224.

<div id="tab:det_results" markdown="1">

|   Method   | Detection |           |       |
|:----------:|:---------:|:---------:|:-----:|
|    2-4     |  CTW1500  | TotalText | TD500 |
| **UniDoc** |   38.27   |   12.60   | 17.36 |

Quantitative performance of UniDoc (F-score) on several scene text
detection benchmark datasets. Here the input instruction is “Output all
the text locations in this photo".

</div>

## Evaluation Metrics

We evaluate our UniDoc in a series of text-rich scenes from three
perspectives (*i.e.,* detection, recognition, and multimodal
understanding). For the task of text detection, we employed the F-score
metric. For text recognition and visual question answering tasks, we
adopted the accuracy metric, where a response generated by the model is
considered correct if it contains the string present in the ground
truth [liu2023hidden](http://arxiv.org/pdf/2305.07895v5). In this paper, F-score and
accuracy are respectively denoted as $\mathcal{F}$ and $\mathcal{A}$.

<div class="table*" markdown="1">

</div>

<div id="tab:multitask" markdown="1">

| Training Task |  | Detection | Recognition | Understanding |
|:--:|:--:|:--:|:--:|:--:|
| 1-2 (rl)3-3 (rl)4-4 (rl)5-5 Pre-train | Fine-tune | $\mathcal{F}$ | $\mathcal{A}$ | $\mathcal{A}$ |
|  |  | 0.00 | 20.01 | 35.78 |
|  |  | 0.00 | 84.13 | **41.28** |
|  |  | 27.89 | 88.93 | 40.46 |
|  |  | **38.27** | **90.60** | 40.72 |

Ablation studies about the training tasking settings. The “" indicates
that the corresponding training phase including the detection,
recognition, and spotting task.

</div>

<div id="tab:otheraba" markdown="1">

|     Experiment      |   Setting   |   Detection   |  Recognition  | Understanding |
|:-------------------:|:-----------:|:-------------:|:-------------:|:-------------:|
| 3-3 (rl)4-4 (rl)5-5 |             | $\mathcal{F}$ | $\mathcal{A}$ | $\mathcal{A}$ |
|    index tokens     |     w/      |     31.28     |      \-       |      \-       |
|                     |     w/o     |   **38.27**   |      \-       |      \-       |
|  instruction type   |  detection  |     38.27     |      \-       |      \-       |
|                     |  spotting   |   **43.33**   |      \-       |      \-       |
|  instruction type   | recognition |      \-       |     90.60     |      \-       |
|                     |  spotting   |      \-       |   **91.30**   |      \-       |

Ablation studies about variations in detection task configurations, and
the impacts of the instruction type on text detection and recognition
during inference.

</div>

## Comparison with Other LMMs

We perform an exhaustive evaluation of publicly accessible large
multimodal models (LMMs) and our UniDoc, assessing their efficacy across
various benchmarks. In the following, we compare and analyze the
experimental results.

**Text Detection.** Compared with the existing large multimodal models
(LLMs), a unique capability of our UniDoc is its text detection ability.
This stems from our approach of incorporating text detection as part of
the unified multimodal instruction tuning. In
Table <a href="#tab:det_results" data-reference-type="ref"
data-reference="tab:det_results">2</a>, we present the quantitative
performance of our method on multiple scene text detection datasets,
including CTW1500 [liu2019curved](http://arxiv.org/pdf/1712.02170v1),
TotalText [ch2017total](http://arxiv.org/pdf/1710.10400v1), and
TD500 [yao2012detecting](http://arxiv.org/pdf/1703.01086v3). Moreover, as illustrated in
Fig. <a href="#fig_spotting" data-reference-type="ref"
data-reference="fig_spotting">[fig_spotting]</a>, we provide examples
showcasing UniDoc’s text detection performance on the CTW1500
dataset [liu2019curved](http://arxiv.org/pdf/1712.02170v1). It can be seen that the text is
consistently detected in these images. Notably, the words in these
images are located irregularly instead of in a straight horizontal line,
and our training phase also does not involve the text detection tasks
for such scene images. These findings validate our learning strategy and
underscore the substantial generalization ability of LLMs.

<figure id="zhuzhuang">
<p><embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/acc.png" /> <span id="fig0"
label="fig0"></span></p>
<figcaption>Quantitative comparison on multiple recognition datasets
based on the recognition instructions and spotting instructions. The
<span class="math inline"><em>x</em></span>-axis represents the
datasets. Spotting instruction consistently performs
better.</figcaption>
</figure>

<figure id="instruct_type_case">
<embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/case3-crop.png" />
<figcaption>A case study illustrating the impact of detection (left) and
spotting (right) instructions on the response. Spotting effectively
mitigates recognition omissions. </figcaption>
</figure>

**Text Recognition.** Furthermore, we extend our evaluation to assess
the text recognition capacity of UniDoc. To commence, as shown in
Table <a href="#tab:text_reco" data-reference-type="ref"
data-reference="tab:text_reco">[tab:text_reco]</a>, UniDoc achieves a
series of state-of-the-art scores across numerous benchmark datasets for
text recognition. It is noteworthy that these datasets encompass a
diverse array of text-rich images, including document text, artistic
text, handwritten text, scene text, and more. Moreover, as depicted in
Fig. <a href="#fig_spotting" data-reference-type="ref"
data-reference="fig_spotting">[fig_spotting]</a> and
Fig. <a href="#fig_recognize" data-reference-type="ref"
data-reference="fig_recognize">[fig_recognize]</a>, we showcase
recognition results of UniDoc on CTW1500 [liu2019curved](http://arxiv.org/pdf/1712.02170v1),
WordArt [xie2022toward](http://arxiv.org/pdf/1812.05824v3) and
TotalText [ch2017total](http://arxiv.org/pdf/1710.10400v1) dataset. Although these images
involve varying fonts, styles, image blurriness, and non-horizontal text
distributions, UniDoc consistently manifests a remarkable ability to
accurately recognize the embedded text within them.

<div class="figure*" markdown="1">

<embed src=".//papers/text_rich_doc/arXiv-2308.11592v2_md/figure/example2-crop.png" />

</div>

**Multimodal Understanding.** We conduct both quantitative and
qualitative assessments of UniDoc’s multimodal understanding
performance. Specifically, as presented in
Table <a href="#tab:text_reco_vqa_kie_res" data-reference-type="ref"
data-reference="tab:text_reco_vqa_kie_res">[tab:text_reco_vqa_kie_res]</a>,
UniDoc achieves state-of-the-art and comparable performance on several
benchmark datasets. Besides, as illustrated in the
Fig. <a href="#fig_understanding" data-reference-type="ref"
data-reference="fig_understanding">[fig_understanding]</a>, we provide
examples of multimodal question-answering focused on text-based
scenarios. It can be seen that UniDoc effectively integrates the visual
cues from the input image and the textual cues from both the image and
instructions. Leveraging the inherent world knowledge of the large
language model (LLM), it then engages in coherent reasoning to generate
corresponding responses.

## Ablation Studies

In this section, we conduct ablation studies to validate the efficacy of
core settings and components in our UniDoc. In all experiments, for the
tasks of text detection, recognition, and multimodal understanding, we
report the performance on the CTW1500 [liu2019curved](http://arxiv.org/pdf/1712.02170v1),
IIIT5K [mishra2012scene](http://arxiv.org/pdf/1907.09705v1), and
TextVQA [singh2019towards](http://arxiv.org/pdf/1811.11903v1) benchamrk datasets,
respectively.

**Impact of Unified Multimodal Instruct Tuning.** During the
pre-training phase, the instruction-following data we trained
encompasses text detection, recognition, and spotting tasks. In the
fine-tuning phase, the instruction-following data was further augmented
with tasks concerning multimodal understanding. we investigate the
impact of learning these tasks (ı.e., text detection, recognition, and
spotting) on the final performance. As illustrated in
Table <a href="#tab:multitask" data-reference-type="ref"
data-reference="tab:multitask">3</a>, incorporating the learning of them
in individual phases led to enhancements not only in detection and
recognition performance, but also in multimodal understanding.
Furthermore, incorporating these tasks in both stages yielded the best
performance. These results demonstrate that there exists a beneficial
interplay and synergy among these tasks. We argue that such a multi-task
learning strategy not only endows Large Multimodal Models (LMMs) with
comprehensive capabilities, but also bolsters their inherent abilities.

**Impact of the Formulation of the Detection Task.** In our default
setting, we directly predict the integer coordinates of the text region
bounding boxes. Given that our input images are all of the size
224$\times$`<!-- -->`{=html}224, these coordinates are normalized to the
range \[0, 223\]. An alternative approach is to set up an additional 224
tokens to represent both the horizontal and vertical coordinates in the
range \[0, 223\] [chen2021pix2seq](http://arxiv.org/pdf/2305.18279v1). As shown in
Table <a href="#tab:otheraba" data-reference-type="ref"
data-reference="tab:otheraba">4</a>, in terms of text detection
capabilities, the introduction of additional positional index tokens did
not yield a performance gain.

**Impact of Instruction Template Type.** In our UniDoc, the detection
results can originate from either the detection or the spotting
instructions. Similarly, our recognition outcomes can be sourced from
either the recognition or the spotting instructions. Consequently, we
evaluate the impact of using different types of instructions on the
performance of detection and recognition. As shown in
Table <a href="#tab:otheraba" data-reference-type="ref"
data-reference="tab:otheraba">4</a>, the text detection and recognition
performance based on the spotting instruction works better. This is
likely because in autoregressive generation, spotting instruction
template makes model provide explicit location information in its
responses, enhancing the recognition performance. The same applies to
detection tasks. The two tasks are mutually complementary. In
Fig. <a href="#zhuzhuang" data-reference-type="ref"
data-reference="zhuzhuang">2</a>, we perform quantitative comparisons on
a broader range of recognition benchmarks. Besides, as shown in
Fig. <a href="#instruct_type_case" data-reference-type="ref"
data-reference="instruct_type_case">3</a>, we further provide a case to
illustrate this finding.

# Conclusion

In this work, we introduce UniDoc, a universal large multimodal model
for simultaneous text detection, recognition, spotting, and
understanding. Through our proposed unified multimodal instruct tuning,
UniDoc effectively leverages the beneficial interactions among
text-based tasks, not only addressing the shortcomings of existing large
multimodal models, but also enhancing their original capabilities. To
implement UniDoc, we contribute a large-scale multimodal instruction
following dataset. Experiments show that our UniDoc sets
state-of-the-art scores across multiple benchmarks. Besides, we perform
extensive studies to validate its effectiveness. Currently, UniDoc is
unable to extract fine-grained visual features for detection and
recognition, and the resolution of input images remains a limitation. In
the future, we will consider addressing these issues.

[^1]: https://openai.com/blog/chatgpt

[^2]: https://commoncrawl.org/