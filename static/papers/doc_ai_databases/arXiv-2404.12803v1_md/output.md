# Introduction

Recent research on multimodal large language models (MLLMs) has achieved
significant advancements in the text-centric visual
question-answering(VQA) domain
[Text-MLLM-1](None), [Text-MLLM-2](None), [Text-MLLM-3](None), [docpedia](None), with
several closed-source state-of-the-art (SOTA) models leading the way.
Two representative examples are GPT4V [gpt4v](http://arxiv.org/pdf/2312.04344v2) and Gemini
[gemini-pro](http://arxiv.org/pdf/2312.17661v1), which have demonstrated remarkable
performance and have even surpassed human-level capabilities in certain
aspects. Nevertheless, as illustrated in Figure
<a href="#img-performance" data-reference-type="ref"
data-reference="img-performance">1</a>, the performance of open-source
models still lags significantly behind that of pioneering closed-source
models. This phenomenon can be attributed to various factors, including
model architecture, the scale of model parameters, image resolution, the
volume of pretraining and instruction tuning data, and training
strategies, among others.

<figure id="img-performance">
<embed src="/papers/doc_ai_databases/arXiv-2404.12803v1_md/radar.png" />
<figcaption> The performance of TextSquare in various VQA tasks compared
to existing models. (a) shows the comparison with state-of-the-art
closed-source models (Gemini <span class="citation"
data-cites="gemini-pro"></span> and GPT4V <span class="citation"
data-cites="gpt4v"></span>), and (b) shows the comparison with the
leading open-source models. The numbers in parentheses after the model
names in the legend indicate the average performance ranking across 10
text-centric multimodal benchmarks. TextSquare is marginally superior to
GPT4V. Best viewed on screen.</figcaption>
</figure>

Many pioneering studies [allava](None), [bonito](None), [sharegpt4v](None), [llavar](None)
have recently conducted data-centric research into the challenges of
insufficient instruction tuning data. For instance, Monkey
[monkey](None) initially employed expert models to generate
descriptions of different aspects of images, which were then summarized
by GPT-4 to produce high-quality and detailed image caption data. For
better text-based knowledge injection, For better text-based knowledge
injection, LLaVAR [llavar](None) and TG-Doc
[tg-doc](None) used GPT-4 to generate conversations for
text-rich images by integrating OCR results into the instructions. In
order to improve the image caption ability for MLLMs, ShareGPT4V
[sharegpt4v](None) constructs a high-quality image caption
dataset through GPT4V. While these efforts have achieved remarkable
success, they also left some challenges unresolved. Image caption data
and VQA data belong to different domains, with inconsistencies in the
granularity and scope of image content presentation. Furthermore, the
scale of synthetic data remains relatively small, preventing MLLMs from
fully realizing their potential. The exploration of methods that
leverage large-scale text-centric VQA data for instruction tuning of
existing open-source models remains limited.

To bridge the gap, this paper proposes a strategy termed Square for
obtaining massive, high-quality text-centric VQA data from sophisticated
and versatile closed-source MLLMs, resulting in the construction of a
dataset (Square-10M) comprising tens of millions of instances for
instruction tuning. Specifically, the method consists of four steps:
Self-Questioning, Answering, Reasoning, and Evaluation. The
self-questioning step involves utilizing the MLLM’s capabilities in
text-image analysis and understanding to generate questions related to
the textual content of images. The answering step involves answering
these generated questions, leveraging various prompting techniques such
as Chain-of-Thought and few-shot prompting. The reasoning step entails
probing the model for the reasoning behind its answers, leveraging the
powerful reasoning abilities of MLLMs. The evaluation step involves
evaluating the question-answer pairs, assessing the validity of the
questions and their relevance to the textual content of the images, as
well as the correctness of the answers, thereby improving data quality
and mitigating hallucinations. Overall, Square comprehensively leverages
the capabilities of MLLMs in various aspects, significantly enhancing
the data quality.

Besides, enriching the diversity of images is also crucial. We collect a
diverse set of text-rich images from various public sources, including
natural scenes, charts, tables, receipts, books, slides, PDFs,
documents, products, and web images. Subsequently, deduplication is
performed on this collection. By applying the Square method to these
images, Square-10M is constructed.

Based on Square-10M, we achieve several remarkable results with
extensive and rigorous experiments. First, as shown in Figure
<a href="#img-performance" data-reference-type="ref"
data-reference="img-performance">1</a>, our model (TextSquare) achieves
comparable or superior performance to advanced closed-source models and
substantially outperforms recent state-of-the-art open-source models on
various benchmarks. It is notable that the image resolution of
TextSquare is $700$ and the parameters are $8.6$B. Second, our
experiments validate the beneficial impact of reasoning data on VQA
tasks, demonstrating its ability to enhance model performance while
mitigating hallucinations. With reasoning data for instruction tuning,
TextSquare has a strong reasoning capability to provide elaborate
explanations for VQA scenarios. Last but not least, by leveraging the
dataset’s massive scale, we unveil the relationships between instruction
tuning data scale, training convergence loss, and model performance.
Whereas a few instruction tuning data can motivate MLLM well, it is not
sufficient. Large amounts of high-quality data can still significantly
reduce convergence loss and improve performance. The performance of
TextSquare grows and the loss of convergence decreases while
continuously scaling up the instruction tuning data, which also
demonstrates the effectiveness of our dataset.

In summary, the main contributions of this paper can be categorized into
four points:

-   A high-quality dataset (Square-10M) comprising tens of millions of
    instances for text-centric VQA instruction tuning is constructed by
    comprehensively collecting text-rich images from various scenarios
    and employing the Square (Self-Questioning, Answering, Reasoning,
    and Evaluation) strategy on closed-source MLLMs.

-   Leveraging Square-10M, TextSquare achieves a significant
    outperformance of existing open-source models and even comparable or
    superior performance to SOTA closed-source models on various
    benchmarks, e.g., +0.9% on ChartQA, +2.1% on WTQ, +4.3% on SROIE.
    Notably, TextSquare outperforms GPT4V in overall rankings across ten
    text-centric benchmarks (ranking 2.2 *v.s.* 2.4).

-   Reasoning data is demonstrated to be beneficial in improving model
    performance and mitigating hallucinations in text-centric VQA
    scenarios, as it can deliver rich question-specific contextual
    information.

-   Through extensive experiments, we reveal the relationships between
    data scale, convergence loss, and model performance for text-centric
    VQA instruction tuning, which demonstrates the effectiveness and
    necessity of Square-10M.

# Related Work

## Multi-modal Large Language Models

Recent work has increasingly focused on introducing visual knowledge
into LLMs [MLLM-1](None), [MLLM-2](http://arxiv.org/pdf/2308.12966v3), [MLLM-3](None). General attempts
connect a visual encoder and an LLM with intermediate modules like
Projector [llava](None), Q-Former [blip2](None),
Perceiver Resampler [flamingo](None), etc, and go through
pre-training alignment and instruction fine-tuning for vision-language
understanding.

Recently, several researches
[Text-MLLM-1](None), [Text-MLLM-2](None), [docpedia](None), [structextv2](None), [vary](None), [omniparser](None), [layoutllm](None), [hrvda](None)
propose to enhance MLLMs’ capabilities in understanding textual elements
(OCR, text-centric VQA, etc). Among them, mPLUG-DocOwl
[Text-MLLM-1](None) creates novel instruction-following
datasets to enhance the tuning process. TextMonkey
[MLLM-3](None) adopts shifted window attention and filters out
significant tokens. DocPedia [docpedia](None) and HRVDA
[hrvda](None) enlarges input resolution to bridge the gap
between MLLMs and visual document understanding.

Despite the extraordinary progress of existing open-source MLLMs, they
still suffer from the huge gap against SOTA closed-source models like
GPT4V [gpt4v](http://arxiv.org/pdf/2312.04344v2) and Gemini Pro [gemini-pro](http://arxiv.org/pdf/2312.17661v1).
In this paper, we propose to mitigate this gap by training with
large-scale and high-quality instruction-following data.

## Text-Centric Visual Question Answering

Text-Centric Visual Question Answering aims to understand the
interactions between the textual and the visual elements in the image.
Donut [donut](None) first proposes an end-to-end training
method based on a Transformer without OCR. Pix2Struct
[pix2struct](None) introduces a variable-resolution input
representation to adapt to document images. DoCo [doco](None)
enhances the visual representation of the image encoder in LVLMs by
aligning the document object of multi-modal inputs. BLIVA
[bliva](None) enlarges the input token space by concatenating
learned query embeddings and encoded patch embeddings. Several studies
[Text-MLLM-2](None), [tg-doc](None), [llavar](None) have performed data-centric
attempts in this regard. UniDoc [Text-MLLM-2](None) construct
600k document-oriented image-text pairs from PowerPoint presentations.
LLaVAR [llavar](None) and TG-Doc [tg-doc](None) prompt
text-only GPT-4 to generate conversations for text-rich images by
integrating OCR results into the instructions. These researches are
restricted to small-scale annotations or generation based on uni-modal
inputs.

## Generating Instruction-Tuning Data via LLMs

The success of LLMs has inspired recent work to employ them as training
data generators
[sharegpt4v](None), [allava](None), [self-instruct](None), [synthetic-prompting](None). In
this regard, we anchor on generating instruction-following data.
Self-Instruct [self-instruct](None) took the initial step
towards synthesizing instructions via language models and improving the
instruction-following capabilities. Llama-GPT4
[llama-gpt4](None) uses GPT-4 to generate instruction-following
data for LLM fine-tuning. Synthetic Prompting
[synthetic-prompting](None) leverages a few handcrafted
examples to prompt LLMs to generate more examples. Bonito
[bonito](None) converts unannotated text into task-specific
training datasets for instruction tuning. Recently, ALLAVA
[allava](None) employs GPT4V to generate reasoning instructions
and detailed answers from unlabeled images. All of the above attempts
suffer from the low quality of the generated data and are typically
performed on a small scale. In contrast, we collect massive text-centric
images (*i.e.*, tens of millions) and devise comprehensive generating
methods and filtering rules to ensure the quantity and quality of the
instruction tuning dataset.

<figure id="data_distribution">
<embed src="/papers/doc_ai_databases/arXiv-2404.12803v1_md/data_distribution.png" />
<figcaption>Overview of Square-10M: the distribution of images, the
average tokens of the QAs, etc.</figcaption>
</figure>

# Square-10M: A Massive and High-quality Text-Centric VQA Instruction Tuning Dataset

Square-10M is synthesized by our proposed Square pipeline, *i.e.*,
Self-Questioning, Answering, Reasoning, and Evaluation.

## Overview of Square

Figure <a href="#algorithm" data-reference-type="ref"
data-reference="algorithm">3</a> presents an overview of our proposed
Square. Square generally consists of three stages for synthesizing
high-quality instruction tuning data for text-centric VQA: (1) Data
Collection for collecting large-scale images with textual elements of
diverse properties. (2) Data Generation involves self-questioning,
answering, and reasoning of the collected data. In this phase, the MLLM
is prompted to generate VQA pairs based on the given image, as well as
the reasoning behind its answers. (3) Data Filtering for self-evaluation
of the generated content, aiming to discard meaningless questions and
erroneous answers by employing the evaluation capabilities of MLLMs.

The above procedures result in our Square-10M dataset, standing out with
its massive and high-quality text-centric VQA pairs and reasoning
context. To be more specific, a total of 3.8 million images with rich
textual elements are collected from diverse sources. After that 20
million question-answer pairs are obtained from Data Generation.
Finally, 9.1 million QA pairs as well as the reasoning context are
distilled with our Square strategy. A more precise analysis of
Square-10M is depicted in Figure
<a href="#data_distribution" data-reference-type="ref"
data-reference="data_distribution">2</a>.

## Data Collection

The data collection strategy is driven by the primary objective of
encompassing a broad range of real-world text-rich scenarios. To this
end, we collect 3.8 million unlabeled text-rich images (Figure
<a href="#data_distribution" data-reference-type="ref"
data-reference="data_distribution">2</a>). These images exhibit diverse
properties. For instance, Chart and Table focus on textual elements with
intense statistical information; Slide, Screenshot, and WebImage are
designed for the interaction between text and prominent visual messages;
Document/PDF, Receipt, and e-commerce contain images with fine and dense
text; Street-View is derived from natural scenes. The collected images
form a mapping of the textual elements in the real world and constitute
the foundation of our research on text-centric VQA.

<figure id="algorithm">
<embed src="/papers/doc_ai_databases/arXiv-2404.12803v1_md/algorithm.png" />
<figcaption>Pipeline for the proposed Square strategy. Gemini’s
versatile multi-modal comprehension capabilities are utilized to
synthesize Square-10M, which consists of four stages, self-questioning,
answering, reasoning, and evaluation.</figcaption>
</figure>

## Data Generation: Self-Questioning, Answering, and Reasoning

We build our Square-10M dataset by employing the multi-modal
understanding capabilities of Gemini Pro, one of the most advanced LLMs.
For each image selected from a specific data source, Gemini Pro is
instructed to generate VQA pairs and reasoning context through the
subsequent three stages:

**Stage 1: Self-Questioning.** In this stage, Gemini Pro is prompted to
generate profound, meaningful, and non-trivial questions about the given
image. We ask Gemini Pro to first comprehensively analyze the image and
then raise questions based on its understanding, as shown in Figure
<a href="#algorithm" data-reference-type="ref"
data-reference="algorithm">3</a>. Considering that advanced MLLMs
typically have weaker understanding capabilities of the textual elements
than visual elements, we also prepend the extracted text to the prompt
by employing expert OCR models.

**Stage 2: Answering.** Gemini Pro is then instructed to give
appropriate answers to the generated questions. We leverage various
prompting techniques to enrich the contextual information and improve
the reliability of the generated answers, such as Chain-of-Thought and
few-shot prompting. Figure
<a href="#algorithm" data-reference-type="ref"
data-reference="algorithm">3</a> shows an example prompt for generating
answers to a given question.

**Stage 3: Reasoning.** We require Gemini Pro to elaborate on the
detailed reasons behind its answers. Such an effort enforces Gemini Pro
to think more about the connections between the questions and the visual
elements, thus reducing hallucinations and providing accurate answers.
Moreover, the generated reasons could serve as extra contextual
information specific to individual questions, favoring possible research
on the mechanism behind in-context learning. We present an example
prompt for self-reasoning in Figure
<a href="#algorithm" data-reference-type="ref"
data-reference="algorithm">3</a>.

## Data Filtering: Self-Evaluation and Answering Consistency

Despite the effectiveness of Self-Questioning, Answering, and Reasoning,
the generated image-text pairs could face hallucinatory content,
meaningless questions, and erroneous answers. We thus devise filtering
rules based on the Evaluation capabilities of LLMs to select
high-quality VQA pairs. The whole filtering system is established upon
three aspects.

**Self-Evaluation of MLLMs.** We prompt Gemini Pro as well as other
advanced MLLMs to judge whether the generated questions are meaningful
and whether the answers are good enough to correctly address the
questions.

Figure <a href="#algorithm" data-reference-type="ref"
data-reference="algorithm">3</a> depicts an example prompt for
self-evaluation.

**Multi-Prompt Consistency.** Besides direct evaluation of the generated
content, we manually augment the prompt and context space in Data
Generation. A correct and meaningful VQA pair should be semantically
consistent when provided with different prompts. Specifically, in the
stage of Answering we provide Gemini Pro with different but semantically
similar prompts to answer the given question. Then we discard the VQA
pairs if the generated answers are not stable in semantics. An example
is given in Figure <a href="#algorithm" data-reference-type="ref"
data-reference="algorithm">3</a>.

**Multi-Context Consistency.** Similar to Multi-Prompt Consistency, we
further validate the VQA pairs by prepending the question with varied
context information. Given the generated question, three types of
answers are produced by Gemini Pro with different contexts: (1)
Answering with reasoning. Gemini Pro answers the question with a
detailed explanation prepended (*i.e.*, content generated in the stage
of Reasoning). (2) In-Context answering. Gemini Pro answers the question
with chain-of-thought or few-shot prompts prepended. (3) Naive
answering. Gemini Pro answers the question with no extra context. We
then discard the VQA pairs if the generated answers are not semantically
consistent.

# TextSquare: A Text-Centric Multimodal Large Language Model

## Model Architecture

The model architecture of TextSquare follows the paradigm established by
InternLM-Xcomposer2 [internlm-xcomposer2](None), including
three integral components: (1) A Vision Encoder modified from OpenAI
CLIP ViT-L-14-336 [clip](None), where the resolution is
increased to 700 for improved performance. (2) A LLM based on InternLM-2
[internlm2](None), utilizing InternLM2-7B-ChatSFT as the
practical variant. (3) A Projector, which semantically aligns the vision
token and the text token.

## Supervised Fine-Tuning with Square-10M

TextSquare is achieved by performing Supervised Fine-Tuning (SFT) with
Square-10M. The SFT process comprises three stages: In the first stage,
we unfreeze all the three components (*i.e.*, the Vision Encoder, the
LLM, and the Projector) and train the model in a resolution of 490. In
the second stage, the input resolution is increased to 700 and only the
Vision Encoder is trained to adapt to the resolution change. In the
third stage, we further perform full-parameter fine-tuning in the
resolution of 700. TextSquare demonstrates that with our Square-10M
dataset, a model with 8B parameters and normal-size image resolution can
achieve extraordinary performance on text-centric VQA, surpassing most
available MLLMs and even the closed-source SOTA models.

# Experiment

## Implementation Details

The training data contains Square-10M and in-domain datasets (consistent
with Monkey’s SFT data). The training process is divided into three
phases, using the same data and the AdamW [adamw](None)
optimizer with 64 A100-80G GPUs. In the first phase, we fine-tune
InternLM-Xcomposer2 with full parameters, and the learning rate
decreases from 1e-5 to 1e-6, taking about 9520 GPU hours; In the second
phase we scale up the image resolution to 700, and train only VIT, with
the learning rate decreasing from 1e-4 to 1e-5, taking about 7280 GPU
hours; In the third stage, we perform full-parameter fine-tuning at 700
image resolution, and the learning rate drops from 1e-5 to 1e-6,
spending about 12350 GPU hours.

## Benchmark Evaluation

We report the results on Scene Text-centric VQA, Document-oriented VQA,
Table VQA, Text-centric KIE, OCRBench, and General VQA for a
comprehensive comparison of the performance of our model with existing
models. The metrics of each benchmark are listed in Table
<a href="#benchmark" data-reference-type="ref"
data-reference="benchmark">[benchmark]</a> in the Supplementary
Material.

<span id="table-text-bench" label="table-text-bench"></span>

**Document-Oriented Benchmark.** While the documents have a clean
background, dense text and complex typography pose distinct challenges.
To effectively evaluate our model, we select representative benchmarks
including DocVQA [docvqa](None), ChartQA
[chartqa](None), and InfographicVQA
[infographicvqa](None). The results, detailed in Table
<a href="#table-text-bench" data-reference-type="ref"
data-reference="table-text-bench">[table-text-bench]</a>, show that
TextSquare outperforms all the open-source models in these three
document-oriented VQA tasks with an average improvement of $3.5$%,
specifically, DocVQA $84.3$% *vs.* $81.6$% (Cogagent and mPLUG-DocOwl
1.5), ChartQA $79.4$% *vs.* $72.7$% (Intern-Xcomposer2), InfographicVQA
$51.5$% *vs.* $50.4$% (mPLUG-DocOwl 1.5). On the ChartQA dataset,
TextSquare outperforms GPT4V and Gemini Pro by a slight margin. Note
that TextSquare employs an image resolution of 700, which is smaller
than most document-oriented MLLMs. Our model relies on comprehensively
high-quality VQA information specific to the text in the document,
improving its ability to recognize and understand various document
elements such as text, diagrams, infographics, and so on. If the image
resolution is further increased, it is believed that the model
performance will be further improved, as demonstrated by Monkey et al.

**Scene Text-centric Benchmark.** The ability to answer text-based
questions in images becomes an important aspect of the answering task as
textual information is usually present in real-world scenes. In the
evaluation, we utilize two datasets: TextVQA [textvqa](None)
and AI2D [ai2d](None). As shown in Table
<a href="#table-text-bench" data-reference-type="ref"
data-reference="table-text-bench">[table-text-bench]</a>, in this
scenario, although TextSquare achieves SOTA performance on the AI2D
dataset, there is no major improvement over our baseline
Intern-Xcomposer2, which may be due to the fact that Intern-Xcomposer2
has been adequately optimized with high-quality in-domain data.

**Table VQA Benchmark.** Due to the complex structure of tables and the
dense text, the understanding of the content of tables remains a
challenging issue. In order to evaluate the performance of the
comprehension of table content and structure, we choose two widely
utilized datasets, Wiki Table Questions (WTQ) [wtq](None) and
Table Fact (TabFact) [tabfact](None), as shown in Table
<a href="#table-text-bench" data-reference-type="ref"
data-reference="table-text-bench">[table-text-bench]</a>. On the Table
VQA benchmarks, TextSquare achieves optimal performance among the
leading models with an average $3.0$% improvement. This demonstrates
that our model has reached a new level of table understanding, where
high-quality generated table VQA and reasoning data play a key role.

**Text-centric KIE Benchmark.** Text-centric key information extraction
tasks are frequently encountered in the information processing of
various types of products, certificates, and receipts. We select a
receipt information extraction dataset (SROIE) [sroie](None)
and a product information extraction dataset (POIE)
[poie](None), and the KIE task is converted to the VQA task.
TextSquare achieves optimal performance in both datasets, with a major
average lift of $14.8$% (shown in Table
<a href="#table-text-bench" data-reference-type="ref"
data-reference="table-text-bench">[table-text-bench]</a>). It is worth
noting that there is no training set of POIE added to the training set
and there is not much data in the domain of product scenarios. This
illustrates the extensive textual comprehension capabilities of our
model.

**OCRBench.** OCRBench [ocrbench](None) is a comprehensive
benchmark consisting of 29 OCR-related assessments, with text
recognition, formula recognition, text-centric VQA, KIE, etc. TextSquare
achieves optimal performance in OCRBench except for the closed-source
models and becomes the first MLLM that exceeds $600$ points with about
$10$B parameters. It indicates that the model performs well in both
text-centric perception and comprehension tasks, especially in text
recognition, where little in-domain data is included in the training
set.

<div id="table-general-bench" markdown="1">

|  |  |  |  |  |  |
|:---|:--:|:--:|:--:|:--:|:--:|
| Method | General VQA and Hallucination Evaluation |  |  |  |  |
|  | VizWiz | VQAv2 | GQA | POPE$^{adv}$ | Average |
| Qwen-VL [MLLM-2](http://arxiv.org/pdf/2308.12966v3) | 35.2 | 79.5 | 59.3 | \- | \- |
| Monkey [monkey](None) | 61.2 | 80.3 | 60.7 | 80.3$^*$ | 70.6 |
| Cogagent [cogagent](None) | 36.7$^*$ | <u>**83.7**</u> | 62.3$^*$ | 85.9 | 67.2 |
| DocOwl 1.5 [docowl-1.5](None) | 43.5$^*$ | 68.0$^*$ | 48.5$^*$ | 79.7$^*$ | 59.9 |
| Llava Next 34B [llava-next](http://arxiv.org/pdf/2404.05465v1) | 63.8 | <u>**83.7**</u> | <u>**67.1**</u> | 83.4 | 74.5 |
| GPT4V [gpt4v](http://arxiv.org/pdf/2312.04344v2) | 64.9$^*$ | 77.2 | 48.4$^*$ | 79.6$^*$ | 67.5 |
| Gemini Pro [gemini-pro](http://arxiv.org/pdf/2312.17661v1) | 42.8$^*$ | 71.2 | 52.2$^*$ | 84.5$^*$ | 62.7 |
| Xcomposer2 [internlm-xcomposer2](None) | 58.9$^*$ | 81.8 | 64.5 | 78.5 | 70.9 |
| TextSquare (ours) | <u>**71.4**</u> | 78.0 | 64.5 | <u>**86.6**</u> | <u>**75.1**</u> |

Quantitative comparison of our model with existing MLLMs on
representative General VQA and hallucination evaluation benchmarks.
VizWiz and POPE are relevant to both VQA and hallucination. Following
Cogagent, we evaluate the adversarial part of POPE.

</div>

<span id="table-general-bench" label="table-general-bench"></span>

**General VQA and Hallucination Evaluation Benchmark.** General VQA
requires the ability to learn both visual and textual information and a
deep understanding of their inter-relationships. For general VQA, we
validate on four benchmarks: VizWiz [vizwiz](None), VQAv2
[vqav2](None), GQA [gqa](None), and POPE
[pope](None). The VizWiz and POPE benchmarks are also relevant
for hallucination evaluation. The results are shown in Table
<a href="#table-general-bench" data-reference-type="ref"
data-reference="table-general-bench">1</a>. On VQAv2 and GQA, TextSquare
does not have a significant degradation compared to InternLM-Xcomposer2
and still maintains comparable performance. TextSquare exhibits superior
capabilities in VizWiz and POPE, outperforming the closest competing
method by an average of $3.6$%. These results highlight the
effectiveness of our approach, which is also able to mitigate model
hallucinations in particular with large-scale instruction tuning. We
observe that it is partly attributed to the high-quality reasoning data
that provides detailed explanations for VQA.

## Qualitative Analysis

As illustrated in Figure
<a href="#reasoning_case" data-reference-type="ref"
data-reference="reasoning_case">4</a>, TextSquare has a formidable
capability to provide plausible explanations of the answers to questions
in a variety of text-centric VQA scenarios. Figure
<a href="#reasoning_case" data-reference-type="ref"
data-reference="reasoning_case">4</a>(a) shows that TextSquare has
simple arithmetic capabilities. Figure
<a href="#reasoning_case" data-reference-type="ref"
data-reference="reasoning_case">4</a>(b) shows the ability to understand
textual content and provide approximate location in dense text. Figure
<a href="#reasoning_case" data-reference-type="ref"
data-reference="reasoning_case">4</a>(c) shows the comprehension of
table structure and the ability to extract contextual information
relevant to the question.

<figure id="reasoning_case">
<embed src="/papers/doc_ai_databases/arXiv-2404.12803v1_md/reasoning_case.png" />
<figcaption>Qualitative results of VQA and reasoning for various
text-centric scenarios.</figcaption>
</figure>

| Model          | OCRBench | DocVQA | ChartQA | InfoVQA | WTQ  | SROIE | Average |
|:---------------|:--------:|:------:|:-------:|:-------:|:----:|:-----:|:-------:|
| Xcomposer2$^*$ |   571    |  74.8  |  73.2   |  41.6   | 40.3 | 44.7  |  54.9   |
| TextSquare     |   622    |  84.3  |  79.4   |  46.2   | 49.7 | 53.2  |  62.6   |

Ablation study on Incorporating Square-10M for Instruction Tuning.

<div class="table*" markdown="1">

</div>

## Ablation Study

**The Effect of Incorporating Square-10M for Instruction Tuning.**

In order to verify the effectiveness of Square-10M, we fine-tune the
baseline model InternLM-Xcomposer2 on the public text-centric VQA
instruction tuning dataset (consistent with Monkey’s training data). As
shown in Table, TextSquare substantially outperforms Xcomposer2$^*$
(fine-tuned) on various text-centric VQA benchmarks by $7.7$%, which
corroborates that Square-10M can fully exploit MLLM’s ability in
text-centric VQA scenarios and that a large amount of high-quality
instruction tuning data has a major improvement in performance.

**The Effect of Evaluation Step of the Square Strategy.** As shown in
Table <a href="#Tab1" data-reference-type="ref"
data-reference="Tab1">[Tab1]</a>, there is a distinct improvement in
model performance after incorporating the evaluation of the generated
VQA data, which verifies that the evaluation step of the Square strategy
improves the quality of VQA instruction tuning data.

**The Effect of VQA Reasoning Data on Model Performance and
Hallucination Evaluation.** From Table
<a href="#Tab2" data-reference-type="ref"
data-reference="Tab2">[Tab2]</a>, we can find that VQA Reasoning data is
helpful in both improving VQA performance and mitigating hallucinations.
Specifically, in terms of enhancing VQA performance, there is a 1.4% and
1.3% gain on DocVQA and ChartQA. In terms of mitigating hallucinations,
there is a $2.7$% and $3.2$% gain on POPE and WizViz.

<figure id="data-scale">
<embed src="/papers/doc_ai_databases/arXiv-2404.12803v1_md/data_scale_1.png" style="width:95.0%" />
<figcaption>The relationship between instruction tuning dataset scale,
convergence loss, and model performance in text-centric VQA scenarios.
Figure (a) and Figure (b) show the relationship between data scale and
convergence loss, distinguished by a scaling of the horizontal
coordinate of Figure (b) with log<span
class="math inline"><sub>10</sub></span>. Figure (c) and Figure (d) show
the relationship between data scale and model performance, distinguished
by a scaling of the horizontal coordinate of figure (e) with log<span
class="math inline"><sub>10</sub></span>.</figcaption>
</figure>

## Relationships between Instruction Tuning Data Scale, Convergence Loss, and Model Performance

To explore the relationship between instruction tuning data scale,
convergence loss, and model performance based on the merged large-scale
Square-10M and the in-domain instruction tuning dataset, we conduct 10
sets of experiments for different data scales. The average performance
of the models is evaluated on DocVQA, ChartQA, InfoVQA, WTQ, and SROIE.
As shown in Figure <a href="#data-scale" data-reference-type="ref"
data-reference="data-scale">5</a>(a)(b), the convergence loss of the
model continues to decrease as the data scale grows, whereas the rate of
decrease becomes progressively slower. The relationship between the
convergence loss and the instruction tuning data scale approximately
conforms to a logarithmic function. Similarly, from Figure
<a href="#data-scale" data-reference-type="ref"
data-reference="data-scale">5</a>(c)(d), it can be seen that as the
instruction tuning data grows, the model performs better and better, but
the rate of growth continues to slow down. Their relationship is also
approximately in accordance with a logarithmic function. Holistically,
there is a corresponding scaling law in the instruction tuning phase in
text-centric VQA scenarios, where model performance is proportional to
the logarithm of the scale of data. It can guide the construction of
potentially larger datasets and predict model performance.

# Limitation

Although our approach achieves remarkable results in various scenarios,
there are some limitations. Firstly, large-scale data requires plenty of
GPUs for long-time training, which greatly increases the training
consumption. Second, while the Square strategy improves the quality of
synthetic data, it still cannot reach the human level.

# Conclusion

In this paper, we present the Square strategy for constructing a
high-quality text-centric instruction tuning dataset(Square-10M).
Leveraging this dataset, TextSquare significantly surpasses recent
open-source models and even achieves performance comparable to GPT4V
across various benchmarks. Furthermore, we derive the relationship
between instruction tuning dataset scale, convergence loss, and model
performance in order to pave the way for constructing even much larger
datasets. Our approach provides a data-centric perspective that revisits
the role of instruction-tuning data in text-centric VQA, confirming that
both the quantity and quality of data are crucial to model performance.
We believe that there is a promising direction on how to further improve
the data quantity and quality for closing the gap between open-source
models and the leading ones.

# Supplementary Material

## Summary of the Evaluation Benchmarks

We summarize the evaluation benchmarks used in this paper in Table
<a href="#benchmark" data-reference-type="ref"
data-reference="benchmark">[benchmark]</a>.

<span id="benchmark" label="benchmark"></span>