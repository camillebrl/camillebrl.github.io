# Introduction

<figure id="fig:radar">
<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/perform_and_speed_0422.png" style="width:90.0%" />
<figcaption>Our TinyChart-3B outperforms several 13B MLLMs on a variety
of chart understanding benchmarks (a), while achieving larger inference
throughput (b).</figcaption>
</figure>

As an important information source, charts can intuitively visualize
data in various visual presentation forms and have become an
indispensable part of information dissemination, business
decision-making, and academic research [chartsurvey](chartsurvey).
With the rapid growth of multimodal data, automatically comprehending
charts has become a pressing need and received increasing attention from
the research
community [chartllama](chartllama), [chartast](chartast), [chartinstruct](chartinstruct), [onechart](onechart).
Recently, Multimodal Large Language Models (MLLMs) have shown strong
capability in comprehending images and following
instructions [gpt4](gpt4), [llava](llava), [mplugowl](mplugowl), [llava1.5](llava1.5), [sphinx](sphinx), [mplugowl2](mplugowl2), [xcomposer](xcomposer), [xcomposer2](xcomposer2), [xcomposer2-4k](xcomposer2-4k).
Based on these MLLMs, some recent
works [chartllama](chartllama), [chartast](chartast), [chartinstruct](chartinstruct), [paperowl](paperowl)
further build chart understanding models by collecting and constructing
versatile chart comprehension datasets and performing supervised
fine-tuning.

However, despite their remarkable success, current chart understanding
models still face three main limitations: (1) Considerable amount of
parameters makes training and deployment challenging. For example,
ChartLlama [chartllama](chartllama) is a model with 13 billion
parameters, which is hard to deploy on a single consumer GPU with less
than 26GB of VRAMs. (2) They are prone to errors when tackling questions
involving numerical calculations [chartast](chartast), which are
difficult to directly answer without any reasoning steps. (3) They
struggle with efficiently encoding for high-resolution images since the
standard vision transformer would produce lengthy feature sequences.

To overcome such limitations in chart understanding, we propose an
efficient and powerful MLLM, namely **TinyChart**. As shown in
Figure <a href="#fig:radar" data-reference-type="ref"
data-reference="fig:radar">1</a>, through the efficient visual encoding
and Program-of-Thoughts learning strategy, TinyChart achieves
state-of-the-art performances on various chart understanding benchmarks
with only 3B parameters, while excelling in faster inference throughput.

For efficient visual encoding, we propose to merge visual tokens based
on the observation that chart images often contain large areas of color
and white spaces. Inspired by [tome](tome), we adopt a
parameter-free Visual Token Merging module inside each vision
transformer layer, which aggregates the most similar visual tokens and
gradually reduces the length of the visual feature sequence, thus making
it possible to efficiently encode high-resolution chart images. This
enables the model to maintain high-resolution chart image input while
controlling the computation load.

Moreover, inspired by  [pot](pot), we propose
Program-of-Thoughts learning that enhances the model’s ability to
resolve mathematical problems. According to statistics on
ChartQA [chartqa](chartqa), 42% of questions for charts require
numerical answers, and most existing models struggle to perform
numerical question answering [matcha](matcha), [chartast](chartast). To learn
chart understanding more efficiently, we train the model to generate
Python programs for the computation problems step by step. The programs
are then passed to a Python interpreter to produce the final answer. To
support Program-of-Thoughts learning, we further construct the
ChartQA-PoT dataset based on ChartQA [chartqa](chartqa). The QA
pairs in our ChartQA-PoT are constructed in two ways: (1) Template-based
PoT construction, which generates questions and programs by filling in
the manually written templates based on chart data. (2) GPT-based PoT
construction, which leverages `gpt-3.5-turbo` [gpt3.5](gpt3.5) to
generate programs based on human-written questions. Experimental results
show that Program-of-Thoughts learning can significantly improve the
question-answering, especially numerical question answering ability of
TinyChart.  
The main contributions of this work are as follows:

-   We introduce TinyChart, an efficient multimodal chart understanding
    model, which outperforms several 13B MLLMs and achieves
    state-of-the-art performances on a variety of chart understanding
    benchmarks, while excelling in faster inference speed at the same
    time.

-   We propose a Program-of-Thoughts (PoT) learning strategy to enhance
    the model in learning numerical calculation and carefully build a
    PoT dataset ChartQA-PoT.

-   We adopt Visual Token Merging for efficient vision encoding, which
    significantly reduces the length of vision feature sequences and
    enables the model to encode high-resolution chart images with
    constrained computing resources.

# Related Work

## Chart Understanding

Chart understanding requires the model to comprehend chart contents and
accomplish related tasks specified by the instructions. This field
encompasses low-level recognition tasks, such as data
extraction [deplot](deplot), and high-level tasks, such as
question-answering (QA) [chartqa](chartqa), [plotqa](plotqa), [dvqa](dvqa),
summarization [chart2text](chart2text), [chart2text-8k](chart2text-8k), and
re-rendering [chartllama](chartllama). As charts often contain OCR
text pivotal for data interpretation, and many instructions require the
model to perform numerical calculations, chart understanding demands
robust text recognition capabilities and computational reasoning from
the model. Early
approaches [lorra](lorra), [plotqa](plotqa), [deplot](deplot), [chartstamp](chartstamp), [mpmqa](mpmqa), [qc_cap](qc_cap)
rely on pipeline methods that use off-the-shelf OCR tools or component
detectors to transform charts into data tables and other textual
representations. They then employ language models to complete the
specified tasks. These pipeline approaches, limited by their inability
to optimize jointly, were hampered by error accumulation. Recent
studies [unichart](unichart), [matcha](matcha), [chartllama](chartllama), [chartast](chartast), [chartinstruct](chartinstruct), [mmc](mmc)
have shifted towards end-to-end methods based on multimodal large
language models. These studies adopt the structure of multimodal large
language
models [llava](llava), [llava1.5](llava1.5), [mplugowl](mplugowl), [mplugowl2](mplugowl2), [sphinx](sphinx) and
enhance chart understanding abilities through supervised
fine-tuning [instructgpt](instructgpt) with substantial chart
instruction data [chartllama](chartllama), [chartast](chartast), [chartinstruct](chartinstruct).
Although these models demonstrate improvement in performance, their
extensive parameter size prevents them from being easily trained or
deployed under resource-constrained scenarios. In this paper, we
demonstrate that a 3B MLLM is enough to achieve state-of-the-art
performance on several chart understanding tasks. Meanwhile, it has been
well observed that these models are prone to numerical
errors [matcha](matcha), [chartinstruct](chartinstruct), [chartast](chartast).
Though [chartast](chartast) try to construct executable command
lines in JSON format based on a template to eliminate numerical errors,
we argue that it is insufficient to fully address this issue for two
reasons: 1) The executable command lines in JSON format produced
by [chartast](chartast) relies on a specific computational
backend, which limits their potential versatility. 2) Template-based
programs can only cover rather limited scenarios. Instead, we construct
the Program-of-Thoughts learning dataset with the combination of both
templates and GPT-generated programs. This allows the model to more
effectively learn how to solve numerical problems.

## Multimodal Large Language Model

Multimodal large language models (MLLM) exhibit strong capabilities in
visual understanding and instruction
following [gpt4](gpt4), [gemini](gemini). They typically comprise
transformer-based visual encoders, large language models, and
vision-language
connectors [llava](llava), [llava1.5](llava1.5), [tinyllava](tinyllava), [mplugowl](mplugowl), [mplugowl2](mplugowl2), [xcomposer](xcomposer), [xcomposer2](xcomposer2), [mplug-octopus](mplug-octopus).
These models are generally trained on extensive general image-text data
for cross-modal alignment and instruction fine-tuning. Although some
studies have showcased a degree of OCR capability in these multimodal
large language models [ocr_mllm](ocr_mllm), [trie](trie), their performance
on document and chart understanding benchmarks remains suboptimal due to
their low input resolution [ureader](ureader), [xcomposer2-4k](xcomposer2-4k).
Efforts in the general document domain have attempted to improve the
fine-grained understanding capabilities of MLLMs by increasing
resolution [qwenvl](qwenvl), segmenting
images [ureader](ureader), [sphinx](sphinx), [docowl1.5](docowl1.5), [xcomposer2-4k](xcomposer2-4k),
utilizing frequency domain signals [docpedia](docpedia), and
introducing additional high-resolution
encoders [cogagent](cogagent). However, these models often suffer
from low efficiency, primarily due to the excessive length of the
high-resolution visual sequences. The visual token merging method
adopted in this paper can significantly reduce the length of visual
feature sequences and relax the computational requirements with
high-resolution input.

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/overview.png" style="width:90.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/token_merger.png" style="width:90.0%" />

</div>

# TinyChart

## Model Architecture

Figure <a href="#fig:overview" data-reference-type="ref"
data-reference="fig:overview">[fig:overview]</a> shows the overview
framework of our proposed TinyChart. It follows the typical architecture
of the multimodal large language model (MLLM), which consists of a
vision transformer encoder, a vision-language connector, and a large
language model. To encode high-resolution visual input effectively, we
insert the visual token merging module inside each vision transformer
layer.

### Vision Transformer Encoder

The vision transformer encoder aims to encode chart images into vision
features. A standard vision transformer [vit](vit) first
resizes the input image $I$ into a fixed resolution and crops the image
into patches. Then the patches are treated as vision tokens and
processed with transformer encoder layers [transformer](transformer).
Suppose the input image $I^{N\times N}$ is in resolution $N \times N$,
and the patch size is $P \times P$, the length of vision tokens would be
$(N // P)^2$. Since the standard transformer layer does not reduce the
sequence length, the vision transformer finally produces a vision
feature in length $(N // P)^2$. In practice, when $N$ is large, the
vision feature can be very long and inefficient for the language model
to handle.

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/pot_construction_2.png" style="width:90.0%" />

</div>

Since key information (such as OCR words) in a chart can be
unrecognizable in low-resolution images [docowl1.5](docowl1.5),
high-resolution input is essential for chart understanding. However,
charts typically contain a large number of color blocks and blank
spaces, where patches are visually similar. To achieve efficient and
effective chart understanding, we apply Visual Token
Merging [tome](tome) in each transformer layer. The process of
Visual Token Merging is shown in
Figure <a href="#fig:tokenmerge" data-reference-type="ref"
data-reference="fig:tokenmerge">[fig:tokenmerge]</a>. By merging the $r$
most similar token pairs, it reduces the length of the vision feature by
$r$ in each layer. We measure the similarity between two tokens using
the cosine distance between Keys from self-attention
following [tome](tome). As shown in the lower part of
Figure <a href="#fig:tokenmerge" data-reference-type="ref"
data-reference="fig:tokenmerge">[fig:tokenmerge]</a>, Vision Token
Merger finds the top-$r$ similar token pairs through bipartite graph
matching. It first divides the vision tokens into two disjoint sets.
Then, for each token in one set, it finds the most similar tokens in the
other set and draws an edge between the two tokens. After that, it only
keeps the top-$r$ most similar edges and merges the features of the two
endpoints through average pooling. Note that not only spatially adjacent
visual tokens are subject to merging. Non-adjacent tokens can also be
merged if they belong to different subsets and are similar enough.

The visual token merging operation aggregates tokens with a similar
feature into one. Therefore, it will reduce the proportion of this
visual feature in the attention calculation in the following transformer
layer, since the number of this feature has decreased. To solve this
issue, we let the attention operation consider the actual number of
patches $s$ represented by each token as follows: $$\begin{aligned}
\mathrm{Attention}=\mathrm{softmax}\left( \frac{QK^\top}{\sqrt{d}} + \log s \right) V
\end{aligned}$$

Where $Q$, $K$, $V$ denotes the query, key, and value of self-attention
which are linear projected from the hidden
states [transformer](transformer). By adding $\log s$ inside
$\mathrm{softmax}$, the token that merged from $s$ patches are
duplicated by $s$ times in the attention
calculation [tome](tome).

### Vision-Language Connector

The vision language connector aims to project the vision features into
the embedding space of the large language model.
Following [llava1.5](llava1.5), [tinyllava](tinyllava), we implement the
vision-language connector as a multiple-layer perceptron with one hidden
layer and GeLU [gelu](gelu) activation.

### Large Language Model

The large language model aims to comprehend both visual features and
language instructions, and then generate responses to accomplish chart
understanding tasks. It is implemented as a transformer
decoder [transformer](transformer) with a causal attention mask. The
training objective of the model is language modeling. Assuming the
visual features is $V$, the language instruction is $L$, and the
response is $R$, then the loss function is defined as follows:
$$\begin{aligned}
\mathcal{L}=\frac{1}{T}\sum_{i=1}^T\mathrm{LLM}(R_i|V, L, R_{<i})
\label{eq:loss}
\end{aligned}$$ Where $T$ is the number of tokens in $R$. Note that we
only calculate loss over tokens in the responses following the
supervised fine-tuning setting in [llava1.5](llava1.5).

## Program-of-Thoughts Learning

Program-of-Thoughts (PoT) learning aims to enhance the learning
efficiency of models for numerical computation. In PoT learning, the
model is trained to generate executable Python codes as the target of a
question. The final answer is obtained by executing the code with a
Python interpreter. Compared to short answers that only contain the
calculated values, the Python code includes natural language comments
and multi-step reasoning processes, offering a form of learning closely
aligned with the pre-training of the large language model.

To support PoT learning on chart understanding, we construct the
ChartQA-PoT dataset based on the training split of
ChartQA [chartqa](chartqa). ChartQA-PoT contains 140,584
(question, PoT answer) pairs. Each PoT answer consists of multiple lines
of Python code. We provide natural language comments for almost all code
lines to explain their behaviors. We employ two approaches for
constructing (question, PoT answer) pairs: Template-based PoT, and
GPT-based PoT.

### Template-based PoT

Based on the chart images in ChartQA, we construct template-based
(question, PoT answer) pairs. As illustrated in the upper half of
Figure <a href="#fig:pot_construction" data-reference-type="ref"
data-reference="fig:pot_construction">[fig:pot_construction]</a>, the
Template-based PoT is constructed based on human-written templates
containing placeholders for both questions and code. The template
questions involve common numerical operations such as calculating the
sum, average, minimal, and maximum values. We adopt the 40 template
questions proposed by PlotQA [plotqa](plotqa) and manually write
their corresponding template Python code to solve them. As shown in the
top-left part of
Figure <a href="#fig:pot_construction" data-reference-type="ref"
data-reference="fig:pot_construction">[fig:pot_construction]</a>, the
template code consists of several variable assignment operations with
NumPy [numpy](numpy) functions to perform calculations. The
beginning steps usually involve extracting the relevant data from the
chart and assigning them to variables. The final computed result is
stored in a variable named "Answer". For each placeholder in the
template, we identify all possible values from the data table of the
chart and randomly select one to fill in the placeholder. After removing
incorrect or unreasonable filled-ins using rule-based methods, we
finally successfully construct 119,281 (question, PoT pairs) over 17,498
images from ChartQA.

### GPT-based PoT

Although the template-based method allows for the construction of a
large number of question-answer pairs, the diversity of these pairs is
limited due to the fixed templates. To improve the generalization
ability of PoT learning, we have additionally built GPT-generated PoT
data by leveraging the powerful command-following and code-generation
capabilities of large language models. Specifically, we prompt
`gpt-3.5-turbo` [gpt3.5](gpt3.5) to generate PoT answers similar
to the template PoT format for questions annotated in ChartQA using
in-context examples. As shown in
Figure <a href="#fig:pot_construction" data-reference-type="ref"
data-reference="fig:pot_construction">[fig:pot_construction]</a>, since
`gpt-3.5-turbo` does not accept image input, we also provide the data
table corresponding to the chart as text input to `gpt-3.5-turbo`. We
screen the quality of the generated PoT answers by running them through
a Python interpreter. If the annotated PoT answer can not run on the
Python interpreter, or if the answer obtained is different from the
annotated one in ChartQA, then the corresponding PoT Answer is deleted.
In the end, we construct 21,303 (question, PoT Answer) pairs on 15,521
chart images.

<div id="tab:training_data" markdown="1">

| Dataset                                      | Benchmark |       Samples |
|:---------------------------------------------|:---------:|--------------:|
| ***Chart question answer***                  |           |               |
| ChartQA [chartqa](chartqa)             |           |        28,299 |
| ChartQA-PoT                                  |           |       140,584 |
| PlotQA [plotqa](plotqa)               |           |       157,070 |
| DVQA [dvqa](dvqa)                   |           |       200,000 |
| OpenCQA [opencqa](opencqa)             |           |         5,407 |
| ***Chart-to-text generation***               |           |               |
| Pew [chart2text](chart2text)              |           |         7,892 |
| Statista [chart2text](chart2text)         |           |        29,589 |
| OpenCQA [opencqa](opencqa)             |           |         5,407 |
| Vistext [vistext](vistext)             |           |        11,171 |
| ChartSumm [chartsumm](chartsumm)         |           |        75,255 |
| Chart2Text-8k [chart2text-8k](chart2text-8k) |           |         7,862 |
| ***Chart-to-table generation***              |           |               |
| ChartQA [chartqa](chartqa)             |           |        19,373 |
| PlotQA [plotqa](plotqa)               |           |       190,720 |
| Chart2Text-8k                                |           |         8,305 |
| DVQA [dvqa](dvqa)                   |           |       300,000 |
| Statista [chart2text](chart2text)         |           |        29,589 |
| ***Chart instruction following***            |           |               |
| ChartLlama [chartllama](chartllama)       |           |       148,398 |
| **Total**                                    |           | **1,364,921** |

Datasets used for training TinyChart. The benchmark datasets consist of
basic chart understanding evaluations including QA, summary, and
chart-to-table generation. Note that in ablation studies, we only use
the benchmark datasets for training due to limited computational
resources.

</div>

<span id="tab:training_data" label="tab:training_data"></span>

## Multitask Learning

We perform multitask learning to train our TinyChart model. We collect a
chart understanding dataset that contains 1.36M samples for supervised
fine-tuning. It covers various chart understanding tasks including chart
question answering, chart-to-text generation, chart-to-table generation,
and chart instruction following.
Table <a href="#tab:training_data" data-reference-type="ref"
data-reference="tab:training_data">1</a> shows the collection of our
training dataset. We mix data in different tasks together to jointly
train the model, and use task-specified instructions to enable the model
to differentiate between them. The training objective is language
modeling on response tokens as presented in
Eq.<a href="#eq:loss" data-reference-type="ref"
data-reference="eq:loss">[eq:loss]</a>. Note that in ablation studies,
we train solely with benchmark datasets due to limited computational
resources.

<div class="table*" markdown="1">

<span id="tab:main_result" label="tab:main_result"></span>

<div class="tabular" markdown="1">

@lrcccccccc@ & & & & & Chart-to-Text & Chart-to-Table & OpenCQA  
(l)5-10 & & & & Aug. & Hum. & Avg. & BLEU4 & RMS$_{F1}$ & BLEU4  
  
GPT-4V [gpt4](gpt4) & - & - & - & - & - & 78.50 & - & - & -  
Gemini-Ultra [gemini](gemini) & - & - & - & - & - & 80.80 & - & -
& -  
Qwen-VL-Max [qwenvl](qwenvl) & - & - & - & - & - & 79.80 & - & -
& -  
Deplot+Codex [deplot](deplot) & 1.3B+175B & - & - & 91.00 & 67.60
& 79.30 & - & 87.22 & -  
  
Llava1.5 [llava1.5](llava1.5) & 13B &
336$\times$`<!-- -->`{=html}336 & 1.94 it/s & 72.96 & 37.68 & 55.32 &
7.16 & 48.95 & -  
Qwen-VL [qwenvl](qwenvl) & 9.6B & 448$\times$`<!-- -->`{=html}448
& 1.65 it/s & 78.90 & 44.30 & 61.60 & - & - & -  
UReader [ureader](ureader) & 7B &
224$\times$`<!-- -->`{=html}224($\times$`<!-- -->`{=html}20) & 1.67 it/s
& 79.42 & 39.12 & 59.30 & - & - & -  
DocOwl1.5 [docowl1.5](docowl1.5) & 8B &
448$\times$`<!-- -->`{=html}448($\times$`<!-- -->`{=html}9) & 1.56 it/s
& 91.38 & 49.62 & 70.50 & - & - & -  
ChartInstruct [chartinstruct](chartinstruct) & 7B & - & - & 87.76 &
45.52 & 66.64 & 13.83 & - & 15.59  
ChartLlama [chartllama](chartllama) & 13B &
336$\times$`<!-- -->`{=html}336 & 1.94 it/s & 90.36 & 48.96 & 69.66 &
14.23 & 90.00 & -  
ChartAst [chartast](chartast) & 13B &
448$\times$`<!-- -->`{=html}448 & 1.47 it/s & **93.90** & 65.90 & 79.90
& 15.50 & 91.60 & 15.50  
TinyChart@512 & 3B & 512$\times$`<!-- -->`{=html}512 & **3.65** it/s &
93.60 & 72.16 & 82.88 & **17.93** & 92.93 & 19.62  
TinyChart@768 & 3B & 768$\times$`<!-- -->`{=html}768 & 3.14 it/s & 93.86
& **73.34** & **83.60** & 17.18 & **93.78** & **20.39**  

</div>

</div>

# Experiment

## Implementation Details

TinyChart is initialized from TinyLlava [tinyllava](tinyllava),
which utilizes the SigLIP [siglip](siglip) as the vision encoder
and Phi-2 [phi1.5](phi1.5) as the large language model. The
origin input resolution of the vision encoder is
384$\times$`<!-- -->`{=html}384. We extend the input resolution to
512$\times$`<!-- -->`{=html}512 and 768$\times$`<!-- -->`{=html}768 and
apply visual token merging with $r=20$ and $r=84$ in each transformer
layer respectively. We train the entire model for 3 epochs with a batch
size of 512. The learning rate is set to $1e-4$, with a warmup in the
beginning 3% steps, and then decays to 0 at the end of training. The
total training process costs 3 days on 32 Tesla V100 GPUs with 32 GB
VRAMs.

## Evaluation Benchmarks

ChartQA [chartqa](chartqa) aims to generate a short answer to the
question based on the chart content. It includes a lot of questions that
require numerical calculation. We report the relaxed accuracy that
allows numerical error within 5% as the metric
following [chartqa](chartqa), [chartllama](chartllama), [chartast](chartast). Note that our
TinyChart with Program-of-Thoughts learning can perform ChartQA in the
following four settings:

-   **Direct**: the model produces short answers directly.

-   **PoT**: the model produces Python code. The answer is then
    calculated through the Python interpreter.

-   **Combine**: the model produces Python code for questions that
    require calculation, and Direct answers for others. We determine
    whether a question requires calculation with a simple rule-based
    keyword detector. If the question contains one of the calculative
    keywords[^1], the detector will treat it as a computational question
    and prompt the model to generate a PoT answer. Otherwise, the model
    is instructed to produce a Direct answer. Additionally, if the
    generated program of a calculative question encounters syntax
    errors, we let the model produce Direct answers for this question in
    the Combine setting.

-   **Oracle** We further introduce the Oracle setting for ChartQA
    evaluation. Under this setting, we always choose the correct one
    between the Direct and PoT answers after evaluating under both
    settings. It is the upper bound of the combination across the two
    answers.

We evaluate TinyChart under the Combine setting by default.

Chart-to-Text aims to generate a chart summarization based on chart
content. We evaluate the model with the Pew
benchmark [chart2text](chart2text), and report
BLEU4 [bleu](bleu) as the metric.

Chart-to-Table aims to extract the underlying data table presented by
the chart. We evaluate the performance of Chart-to-Table with the data
table annotation provided by ChartQA [chartqa](chartqa)
following [chartllama](chartllama), [chartast](chartast). We report
RMS$_{F1}$ [deplot](deplot) as the metric.

Different from ChartQA, OpenCQA [opencqa](opencqa) evaluates the
ability of models to generate free-form answers to the chart-related
questions. We report BLEU4 [bleu](bleu) as the metric
following [chartinstruct](chartinstruct), [chartast](chartast).

ChartX [chartx](chartx) is a recently proposed benchmark that
contains more chart types. We evaluate the ChartX cognition tasks since
they are more challenging. It covers Question Answering, Chart
Description Generation, Chart Summary Generation, and Chart Redrawing.
We report the GPT-Accuracy for QA and GPT-score for the remaining 3
tasks as the metrics following ChartX [chartx](chartx).

## Main Results

Table <a href="#tab:main_result" data-reference-type="ref"
data-reference="tab:main_result">[tab:main_result]</a> shows an
extensive comparison between TinyChart and existing multimodal large
language models on 4 chart understanding benchmarks. Our TinyChart model
achieves state-of-the-art performance on ChartQA, Chart-to-Text,
Chart-to-Table, and OpenCQA, while excels in larger inference
throughput. Specifically, with the input resolution set at
768$\times$`<!-- -->`{=html}768, TinyChart achieves an accuracy of 83.60
on ChartQA [chartqa](chartqa), surpassing several closed-source
models including GPT-4V, Gemini-Ultra, and
Qwen-VL-Max [qwenvl](qwenvl). It also outperforms previous
open-source SOTA ChartAst [chartast](chartast) on chart
understanding.

We find that previous models performed poorly on the ChartQA human
subset, with none of them achieving over 70%. In contrast, the
performance on the ChartQA-augmentation has approached 93.9%. This is
because the questions posed by human annotators involve more
computational problems [chartqa](chartqa) and are more
challenging. By leveraging the Program-of-Thoughts learning, TinyChart
achieves performance of 73.34% on ChartQA-human, which is an improvement
of 7.44% over the previous state-of-the-art
ChartAst [chartast](chartast). This demonstrates the effectiveness
of our proposed learning method based on the Program-of-Thoughts.

We observed that models with higher input resolutions generally perform
better on chart understanding tasks. However, encoding high-resolution
charts leads to a decrease in inference speed (e.g., Qwen-VL vs.
Llava1.5, DocOwl1.5 vs. UReader, ChartAst vs. ChartLlama). By leveraging
visual token merging, TinyChart is able to accept higher-resolution
input images with a limited increase in computing demands, thus
achieving better performance. Due to the smaller model size and the
efficient visual token merging strategy, TinyChart achieves
significantly larger inference throughput compared to previous models.
In summary, these results demonstrate that TinyChart can achieve
efficient chart understanding with enhanced performance and faster
inference.

Table <a href="#tab:chartqa_setting" data-reference-type="ref"
data-reference="tab:chartqa_setting">1</a> shows the performance
comparison under different settings. Note that the performance of
ChartAst under the Combine setting is from [chartast](chartast),
which leverages a combination of Direct answer and executive JSON to
produce the final answer. The results indicate that our TinyChart model
could achieve SOTA performance on the Direct answer. By combining with
PoT answers, TinyChart could make further improvements. In addition,
since the combination of Direct and PoT answers is very simple, the
performance under the Combine setting falls behind the Oracle setting a
lot. Further study can be conducted to better combine the two answers.

We divide the questions in ChartQA test set [chartqa](chartqa)
into two categories: calculative questions (761 of 2500) and
non-calculative questions (1739 of 2500) by checking whether they
contain calculative keywords mentioned above.
Table <a href="#tab:cal_questions" data-reference-type="ref"
data-reference="tab:cal_questions">[tab:cal_questions]</a> shows the
performance of TinyChart@768 on these two types of questions under
different settings. We observe that PoT significantly improves the
performance on calculative questions compared to Direct settings (78.98
vs. 56.64) and thus it shows overall performance gains (80.84 vs.
76.36). And the simple combination of Direct and PoT strategies further
makes improvements.

<div id="tab:chartqa_setting" markdown="1">

| Model | ChartQA |  |  |  |
|:---|:--:|:--:|:--:|:--:|
| 2-5 | Direct | PoT | Oracle | Combine |
| ChartLlama [chartllama](chartllama) | 69.66 | \- | \- | \- |
| ChartAst [chartast](chartast) | 75.10 | \- | \- | 79.90 |
| TinyChart@512 | **76.92** | 79.64 | 88.76 | 82.88 |
| TinyChart@768 | 76.36 | **80.84** | **89.12** | **83.60** |

Performance on ChartQA under different settings.

</div>

To further assess the generalizability of TinyChart, we compare our
model with end-to-end General MLLM and Chart MLLM on ChartX-Cognition
benchmark [chartx](chartx), since it covers visually diverse
chart types. We use TinyChart@768 to perform inference on ChartX without
additional fine-tuning. As shown in
Table <a href="#tab:chartx" data-reference-type="ref"
data-reference="tab:chartx">2</a>, benefiting from our
Program-of-Thoughts learning method, TinyChart achieves a 33.35
GPT-Accuracy on the QA task, even surpassing the GPT-4V model. Though it
falls behind GPT-4V in Summary, Description, and Redrawing tasks,
TinyChart still performs better than open-source Chart MLLMs including
ChartLlama and ChartAst. It indicates that TinyChart has a strong
capability to generalize across various chart types.

<div id="tab:chartx" markdown="1">

| Model              | ChartX Cognition |          |             |           |
|:-------------------|:----------------:|:--------:|:-----------:|:---------:|
| 2-5                |        QA        | Summary  | Description | Redrawing |
| ***General MLLM*** |                  |          |             |           |
| Llava1.5           |      17.19       |   1.48   |    1.29     |   0.75    |
| GPT-4V             |      33.04       | **3.17** |  **3.12**   | **2.63**  |
| ***Chart MLLM***   |                  |          |             |           |
| ChartLlama         |      13.80       |   1.04   |    1.02     |   0.94    |
| ChartAst           |      30.99       |   0.33   |    1.03     |   0.82    |
| TinyChart@768      |    **33.35**     |   1.53   |    1.64     |   1.89    |

Evaluation results on ChartX [chartx](chartx).

</div>

<div class="table*" markdown="1">

</div>

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/vis_tokenmerge.png" style="width:100.0%" />

</div>

## Ablation Studies

To verify the effectiveness of visual token merging and
program-of-thoughts learning, we conduct ablation studies in
Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a>.

The upper block in
Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a> shows the performance
of the model with and without the use of Program-of-Thoughts training
data. Comparing Row 2 with Row 1, we observe that training solely with
template-based PoT improves the model’s ability to generate direct
answers (71.12 vs. 70.72). This improvement is attributed to PoT
learning enhances the model’s reasoning abilities. At this point, the
PoT answers produced by the model are less accurate than direct answers
(55.44 vs. 71.12), which may be due to the inability of template-based
PoT to cover all questions. However, when we ask the model to generate
PoT answers for questions that require calculation and combine with
direct answers, it outperforms solely direct answers (73.00 vs. 71.12).
This indicates that PoT answers have advantages in computational
problems. After incorporating GPT-based PoT into training, the
performance of PoT answering surpasses direct answering (76.88 vs.
72.44), and both direct (72.44 vs. 71.12) and combined answering (79.48
vs. 73.00) show further improvements. These results confirm the
effectiveness of our proposed Program-of-Thoughts learning method,
suggesting that it not only strengthens the model’s computational
capabilities but also enhances overall problem-solving capability.

The middle block in
Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a> compares the
performance with and without using visual token merging when the input
resolution is 512$\times$`<!-- -->`{=html}512, and with different
numbers of tokens to merge in each layer. Comparing Row 4 and Row 3,
increasing the input resolution from 384 to 512 significantly improves
the model’s performance on three chart understanding benchmarks,
demonstrating that high resolution is crucial for comprehending chart
images. However, a direct increase in resolution leads to a substantial
drop in the inference throughput (2.38 it/s vs. 3.73 it/s). The reason
is that, given high-resolution images, the standard vision transformer
produces a lengthy visual feature sequence that is then processed by the
large language model. This brings considerable computational expenses.
By adopting the visual token merging, we can control the length of the
visual feature sequence by regulating the number of tokens to merge at
each layer, and, thereby achieving efficient high-resolution encoding.
When setting $r$=20, we attain an inference throughput nearly equal to
that with an input resolution of 384$\times$`<!-- -->`{=html}384 (3.65
it/s vs. 3.73 it/s), while providing the performance benefits of higher
resolutions.

To further highlight the advantages of visual token merging, we increase
the input resolution to 768 in the bottom block of
Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">[tab:ablation]</a>. At this point, the
length of the visual feature sequence is 2,916, which could not be
trained using 32GB V100 due to insufficient VRAM. However, after
employing the visual token merging module with $r$=84, the input
sequence length is reduced to 732 and we can perform training normally.
In this setting, the model’s inference throughput is 3.14 it/s, and
demonstrates a certain performance advantage in ChartQA (81.04 vs.
80.76) and Chart-to-Table (88.90 vs. 87.81). It illustrates that by
utilizing visual token merging, we are able to leverage
higher-resolution chart images under constrained resources, thereby
improving performance.

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/vis_cases.png" style="width:97.5%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/table_cases.png" style="width:97.5%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/summary_cases.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/redraw_cases.png" style="width:100.0%" />

</div>

## Visualization

To investigate the effects of visual token merging, we visualized the
token merging results at the final layer of the vision transformer. In
Figure <a href="#fig:vis_tokenmerge" data-reference-type="ref"
data-reference="fig:vis_tokenmerge">[fig:vis_tokenmerge]</a>, we
visualize the top ten groups with the largest numbers of tokens. Each
group is outlined with a different color. The visualization reveals that
these largest groups typically correspond to blank or colored areas. By
compressing these areas down to a single token for encoding, our visual
token merging module can thus reduce the length of the encoded sequence
without losing much information, thereby achieving efficient visual
encoding.

## Case study

We conduct case studies with TinyChart when conducting chart question
answering, chart-to-table, chart-to-text, and chart redrawing in
Figure <a href="#fig:vis_cases" data-reference-type="ref"
data-reference="fig:vis_cases">[fig:vis_cases]</a>,
<a href="#fig:table_cases" data-reference-type="ref"
data-reference="fig:table_cases">[fig:table_cases]</a>,
<a href="#fig:summary_cases" data-reference-type="ref"
data-reference="fig:summary_cases">[fig:summary_cases]</a>, and
<a href="#fig:redraw_cases" data-reference-type="ref"
data-reference="fig:redraw_cases">[fig:redraw_cases]</a>.

In Figure <a href="#fig:vis_cases" data-reference-type="ref"
data-reference="fig:vis_cases">[fig:vis_cases]</a>, we present a case
study on ChartQA. As shown in
Figure <a href="#fig:vis_cases" data-reference-type="ref"
data-reference="fig:vis_cases">[fig:vis_cases]</a> (a-c), much key
information within the chart is provided by visually situated texts
within the image, which requires the model to have the ability to
process high-resolution images. Since ChartLlama only supports 336
resolutions, it struggles to retrieve accurate information in these
charts. In contrast, thanks to the visual token merging, our TinyChart
can accept higher-resolution inputs without introducing excessive
computations. Thus it can successfully find clues related to the
questions. Meanwhile, ChartLlama suffers from numerical errors when
faced with calculative questions in
Figure <a href="#fig:vis_cases" data-reference-type="ref"
data-reference="fig:vis_cases">[fig:vis_cases]</a> (d-e), and our PoT
(Program-of-Thoughts) learning method can accurately solve these
problems. These examples further illustrate the advantages of our
methods. For chart-to-table extraction, we find that our TinyChart model
can successfully extractive values from several visually diverse charts
in Figure <a href="#fig:table_cases" data-reference-type="ref"
data-reference="fig:table_cases">[fig:table_cases]</a> (a-c), thanks to
its excellent text recognition ability with high-resolution input.
However, as shown in
Figure <a href="#fig:table_cases" data-reference-type="ref"
data-reference="fig:table_cases">[fig:table_cases]</a> (d), the model
struggles to estimate the values of data points in the absence of OCR
words. It seems that the model could make reasonable predictions based
on surrounding points, but hardly provide accurate values based on the
coordinate axis. This indicates that the model still lacks the ability
to understand spatial relationships across large areas. From
Figure <a href="#fig:summary_cases" data-reference-type="ref"
data-reference="fig:summary_cases">[fig:summary_cases]</a>, we observe
that the model can understand the data presented in the chart and
generate descriptions and summaries in natural language. Though it can
retrieve the data values correctly, we find it sometimes produces
contents that do match the chart as shown in
Figure <a href="#fig:summary_cases" data-reference-type="ref"
data-reference="fig:summary_cases">[fig:summary_cases]</a> (c-d). This
may be due to the inherent limitations of hallucination in
MLLMs [chair](chair), [pope](pope), [wang2023evaluation](wang2023evaluation), [amber](amber), and may be
alleviated by addressing
hallucinations [vcd](vcd), [opera](opera), [jiang2024hallucination](jiang2024hallucination), [less_eos](less_eos).
We present four cases of chart redrawing in
Figure <a href="#fig:redraw_cases" data-reference-type="ref"
data-reference="fig:redraw_cases">[fig:redraw_cases]</a>. As shown in
Figure <a href="#fig:redraw_cases" data-reference-type="ref"
data-reference="fig:redraw_cases">[fig:redraw_cases]</a> (a-c), our
TinyChart model can generate Python code to redraw visually diverse
chart types including lines, heatmaps, and rings. However, it can be
hard to draw unseen chart types such as 3D bar charts
(Figure <a href="#fig:redraw_cases" data-reference-type="ref"
data-reference="fig:redraw_cases">[fig:redraw_cases]</a> (d)). This may
be mitigated by improving the coverage of different chart types in
training data through automatic data construction
techniques [chartllama](chartllama), [chartx](chartx).

[^1]: sum, mean, average, ratio, mode, divide, dividing, differ,
    subtract, add, division, times, absolute, minus, exceed, below,
    less, fewer, bigger, biggest, greater, higher, longer, tallest,
    lowest, number, how many colors, what is the value

# Conclusion

This paper introduces TinyChart, a chart understanding Multimodal Large
Language Model with 3 billion parameters. To address the inefficiency of
lengthy visual token sequences with high-resolution images, TinyChart
injects a visual token merging module that merges similar vision tokens
together, thereby enabling efficient encoding of high-resolution images.
To tackle the challenges of learning numerical computations, we propose
a Program-of-Thoughts learning method that trains the model to generate
Python programs to answer questions. Our TinyChart model achieves
state-of-the-art (SOTA) performance on multiple chart understanding
benchmarks, surpassing existing 13 billion parameter chart MLLMs, and
outperforms closed-source models like GPT-4V on ChartQA. Extensive
ablation studies confirm the effectiveness of our methods. Our code and
model are released at
<https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/TinyChart>.

<div class="figure*" markdown="1">

</div>

# ChartQA-PoT Details

## Dataset Statistic

We build ChartQA-PoT based on the images and questions in the training
split of ChartQA [chartqa](chartqa). ChartQA-PoT consists of two
subsets: Template-based PoT and GPT-based PoT. We present the statistics
over ChartQA-PoT in
Table <a href="#tab:chartqa_pot" data-reference-type="ref"
data-reference="tab:chartqa_pot">[tab:chartqa_pot]</a>. We find that
answers provided by `gpt-3.5-turbo` are longer than template-based PoT,
since they cover more diverse scenarios.

<div class="tabular" markdown="1">

lrrr Statistic & & &  
Num. of samples & 119,281 & 21,303 & 140,584  
Num. of images & 17,498 & 15,521 & 18,133  
Avg. answer characters & 319.38 & 381.23 & 328.75  
Avg. answer tokens & 117.70 & 136.01 & 120.48  

</div>

We further present the first 2-gram words of the questions after
removing stop words in Template-based PoT and GPT-based PoT in
Figure <a href="#fig:gram_sun" data-reference-type="ref"
data-reference="fig:gram_sun">3</a>. It is observed that GPT-PoT covers
more diverse questions for ‘what’ type questions, and questions in
Template-based PoT are more evenly distributed across all question
types.

<figure id="fig:gram_sun">
<figure id="fig:temp_pot">
<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/temp_sun_cropped_new.png" />
<figcaption>Template PoT.</figcaption>
</figure>
<figure id="fig:gpt_pot">
<embed src="/papers/doc_ai_databases/arXiv-2404.16635v1_md/figures/gpt_sun_cropped_new.png" />
<figcaption>GPT PoT.</figcaption>
</figure>
<figcaption>First 2-gram of the questions in ChartQA-PoT after removing
stop words.</figcaption>
</figure>

## Instructions for GPT-based PoT

Figure <a href="#fig:gpt_prompt" data-reference-type="ref"
data-reference="fig:gpt_prompt">[fig:gpt_prompt]</a> shows the
instructions for constructing GPT-based PoT answers. Note that we prompt
`gpt-3.5-turbo` to provide Python code consisting of assignment
statements and avoid using loops or judgment statements. This can
simplify the program and reduce syntax errors. We also provide meta
information including the chart title, type, and colors to
`gpt-3.5-turbo` since some questions rely on this information to answer.