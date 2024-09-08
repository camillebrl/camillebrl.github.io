# Introduction [sec:intro]

Multimodal large language models (MLLM) have received widespread
attention from the research community in recent years. It inherits the
advanced capabilities of Large Language Models (LLMs) such as powerful
language expression and logical reasoning. The integration of visual and
textual information not only enhances the understanding of visual
content but also provides a more comprehensive context for language
understanding and generation. MLLM has shown great potential in solving
visual problems in the real world and has rich applications in the
fields of vision and language, such as image
captioning `\cite{Karpathy2014DeepVA,Vinyals2014ShowAT}`{=latex},
referring expression comprehension
(REC) `\cite{yu2018mattnet,qiao2020referring}`{=latex}, visual question
answering (VQA) `\cite{Agrawal2015VQAVQ,Schwenk2022AOKVQAAB}`{=latex},
etc. Leveraging Transformer-based
architectures `\cite{Vaswani2017AttentionIA}`{=latex} and large amounts
of training data from web sources, MLLM has become a fundamental
component in artificial intelligence research.

Although Transformers improve the ability of long-range dependencies and
greatly enhance the performance of the model, this architecture is
usually very computationally intensive. This is due to the inherent
computational and memory complexity of the self-attention mechanism used
by Transformer. The computational burden and memory requirements
increase quadratically with the sequence length.

To solve the bottleneck of long sequence modeling, the state space model
(SSM) has been widely studied `\cite{LSSL, s5}`{=latex}. It can be seen
as a blend of recurrent neural networks (RNNs) and convolutional neural
networks (CNNs). Among these studies, the representative works are
structured state space (S4) `\cite{s4}`{=latex} and its
variants `\cite{s5, gupta2022diagonal-dss, S4D}`{=latex}. The latest
work Mamba `\cite{gu2023mamba}`{=latex} further improves S4, with a
selection mechanism that allows the model to select relevant information
in an input-dependent manner, combined with a hardware-aware
implementation to achieve efficient training and inference. Mamba
outperforms Transformer on large-scale data and enjoys linear scaling in
sequence length, which has proven to be a promising alternative to
Transformer for language modeling. Some concurrent works extended this
architecture from 1D language to 2D vision
domain `\cite{Ma2024UMambaEL,Liu2024VMambaVS,Yang2024VivimAV}`{=latex}
such as image classification, biomedical image segmentation, To the best
of our knowledge, no work has explored how to utilize this efficient
architecture to solve multimodal tasks.

Inspired by the successes of SSM, in the paper, we introduce VL-Mamba,
the first work that utilizes state space models for multimodal learning
tasks. To be specific, as illustrated in
Fig. <a href="#fig:vl-mamba" data-reference-type="ref"
data-reference="fig:vl-mamba">[fig:vl-mamba]</a>, we leverage the
pre-trained Mamba language model as our backbone language model instead
of conventional Transformer-based language models such as
LLama `\cite{Touvron2023LLaMAOA}`{=latex} or
Vicuna `\cite{vicuna2023}`{=latex}. Furthermore, we empirically explore
the way to apply 2D vision selective scan mechanisms for VL-Mamba and
introduce a novel MultiModal Connector (MMC) architecture, comprising a
Vision Selective Scan (VSS) module and two linear layers, tailored to
enrich the 2D-causal modeling of visual sequences. For the VSS module,
we explore two distinct scan mechanisms: the Bidirectional-Scan
Mechanism (BSM) and the Cross-Scan Mechanism (CSM). The BSM conducts
scans of visual sequences in both forward and backward directions, while
the CSM extends scanning capability to four directions. In addition, we
study the combinations of different vision encoders, variants of
pretrained Mambe language models, and multimodal connectors to find the
effect of different components for VL-Mamba. Extensive experiments are
conducted on various multimodal learning benchmarks to verify the
effectiveness of VL-Mamba. Our model achieves competitive performance
with other small MLLMs of similar size and even outperforms large MLLMs
(e.g., 7B and 13B versions of
LLaVA-1.5 `\cite{liu2023improvedllava}`{=latex}) on some popular
benchmarks.

In summary, our contributions are as follows:

-   We propose VL-Mamba, which is the first work to explore and exploit
    the state space model in solving multimodal learning tasks, which
    provides a novel framework option for multimodal large language
    models other than transformer-based architectures.

-   We empirically explore the effect of different components for
    VL-Mamba and introduce a novel MultiModal Connector containing a
    Vision Selective Scan (VSS) module to improve the representational
    capabilities.

-   We conduct extensive experiments on diverse multimodal learning
    benchmarks. The experiments demonstrate that VL-Mamba achieves
    competitive performance compared to existing multimodal large
    language models.

-   We make the code open source to promote the research of applying
    state space models for multimodal learning.

# Related Work [sec:related work]

## State Space Models (SSMs)

Modern state space models (SSMs) are derived from the classical state
space model `\cite{kalman1960new}`{=latex} and have become an efficient
building block for constructing deep networks, thereby achieving
cutting-edge performance in analyzing continuous long-sequence data.
They particularly excel at capturing long-range dependencies (LRDs) and
leveraging parallel training methods to increase efficiency. Initiated
by a HiPPO matrix `\cite{gu2020hippo}`{=latex}, Linear State Space Layer
(LSSL) `\cite{LSSL}`{=latex} combines the advantages of continuous-time
models (CTMs), RNNs, and CNNs, which demonstrates the potential of deep
SSMs to solve long-range dependencies. However, the practical
feasibility of LSSL is hampered by the large computational and memory
requirements imposed by the state representation. Subsequently, the
Structured State Space (S4) `\cite{s4}`{=latex} addresses the main
computational bottleneck in prior research. This is achieved through
novel parameterizations catering to continuous-time, recurrent, and
convolutional views of the state space model, thereby effectively
modeling long-range dependencies. S4 has subsequently seen some
variants `\cite{s5, gupta2022diagonal-dss, S4D}`{=latex}, such as the
Diagonal State Space (DSS) model `\cite{gupta2022diagonal-dss}`{=latex},
which forces the state matrix to be a diagonal matrix, making it easier
to formulate, implement, and analyze, and can be proven to be as
expressive as a general state space, while S4D `\cite{S4D}`{=latex}
provides a new mathematical analysis for DSS initialization, making it
simpler and more efficient.

A recent work, named Mamba `\cite{gu2023mamba}`{=latex}, further
improves S4 with a selection mechanism that incorporates time-varying
parameters into SSM, allowing the model to select relevant information
in an input-dependent manner. It proposes a hardware-aware algorithm to
achieve efficient training and inference. Mamba’s superior scaling
performance shows that it is a promising alternative to the Transformer
in long-sequence modeling. Many works extend Mamba from Natural Language
Processing (NLP) to other
fields `\cite{Yang2024VivimAV, Xing2024SegMambaLS,ruan2024vm}`{=latex}.
Vision Mamba (Vim) `\cite{Zhu2024VisionME}`{=latex} applies Mamba to the
Vision Transfomer (ViT) architecture, and combines bidirectional SSM for
data-dependent global visual context modeling and position embedding for
location-aware visual understanding. Visual State Space Model
(VMamba) `\cite{Liu2024VMambaVS}`{=latex} designs a cross-scan mechanism
to bridge the gap between 1-D array scanning and 2-D plain traversing.
U-Mamba `\cite{Ma2024UMambaEL}`{=latex} proposes a hybrid CNN-SSM
architecture to capture both localized fine-grained features and
long-range dependencies in images, to solve the biomedical image
segmentation task. In this work, we explore how to transfer the success
of Mamba to solve the more challenging multimodal learning tasks, which
often require modeling of both vision and language modalities and
complex reasoning.

## Multimodal Large Language Model (MLLM)

With the development of the powerful Large Language Models
(LLMs) `\cite{Touvron2023LLaMAOA,Zhang2022OPTOP,Chowdhery2022PaLMSL}`{=latex},
many
studies `\cite{achiam2023gpt4,Driess2023PaLMEAE,chen2023minigptv2,Qwen-VL,ye2023mplug,Chu2023MobileVLMA}`{=latex}
extend LLMs to multimodal domains by combining visual input with LLM to
build the multimodal large language model (MLLM).
Flamingo `\cite{alayrac2022flamingo}`{=latex} freezes pre-trained visual
encoders and large language models and fuses visual and language
modalities with gated cross-attention, demonstrating excellent few-shot
learning performance. BLIP `\cite{Li2022BLIPBL}`{=latex} uses a dataset
bootstrapped from large-scale noisy image-text pairs to pre-train a
multi-modal mixture of encoder-decoder models by injecting different
synthetic captions and removing noisy captions. Based on this,
BLIP-2 `\cite{Li2023BLIP2BL}`{=latex} uses Querying Transformer
(Q-Former) to bridge the modal gap.
InstructBLIP `\cite{instructblip}`{=latex} further proposes an
instruction-aware visual feature extraction mechanism that can flexibly
and effectively extract visual information features according to the
given instructions.
LLaVA `\cite{liu2023improvedllava, liu2023llava}`{=latex} leverages
advanced LLMs (LLaMA `\cite{Touvron2023LLaMAOA}`{=latex} and
Vicuna `\cite{vicuna2023}`{=latex}) as the language model and
CLIP `\cite{Radford2021LearningTV}`{=latex} as the visual encoder, it
transforms visual tokens into language tokens with a simple MLP layer.
MiniGPT-4 `\cite{zhu2023minigpt}`{=latex} directly aligns visual
information with the language model to accomplish diverse
vision-language tasks without using external vision models. Usually, the
training of MLLMs contains two stages, of which the first stage is to
pretrain the model on a large collection of image-text pairs to acquire
the alignment of vision-language knowledge, and the second stage is to
finetune the model with a smaller but high-quality multimodal
instruction tuning dataset with a designed conversational template.

These MLLM works have greatly advanced research in the fields of
computer vision and natural language processing. However, since the main
framework of these models relies on Transformers, the attention
mechanism in Transformers inherently has high computational complexity
in inference for long sequences. To alleviate the abovementioned issues
related to modeling long-range sequences in the area of multi-modal
learning, we propose the VL-Mamba, which is based on the state space
model. To be specific, we utilize pretrained
Mamba `\cite{gu2023mamba}`{=latex} language model as our backbone
language model, rather than Transformer-based LLMs such as
LLama `\cite{Touvron2023LLaMAOA}`{=latex} or
Vicuna `\cite{vicuna2023}`{=latex}. Moreover, we empirically explore the
effective application of 2D selective scan mechanism in the multimodal
VL-Mamba and the combination of different vision encoders and variants
of Mamba language models.

# Method [sec:method]

In this section, we first introduce the preliminary concepts of state
space models (Sec. <a href="#subsec:pre" data-reference-type="ref"
data-reference="subsec:pre">1.1</a>). Then, we describe the details of
our proposed VL-Mamba (Sec.
<a href="#subsec:model" data-reference-type="ref"
data-reference="subsec:model">1.2</a>), which mainly includes the Vision
Encoder, MultiModal Connector, and the Mamba LLM.

## Preliminaries [subsec:pre]

State space models (SSMs) are commonly considered linear time-invariant
systems that map stimulation $x(t) \in \mathbb{R}^L$ to response
$y(t) \in \mathbb{R}^M$ through a hidden state $h(t) \in \mathbb{R}^N$.
Mathematically, these models are typically formulated as linear ordinary
differential equations (ODEs), where the parameters include
$\mathbf{A} \in \mathbb{C}^{N \times N}$,
$\mathbf{B} \in \mathbb{C}^{N}$ for a state size $N$, and the skip
connection $\mathbf{D} \in \mathbb{C}^1$. The system dynamics and output
equations are given by:

$$\begin{aligned}
\label{eq:lti}
h'(t) &= \mathbf{A}h(t) + \mathbf{B}x(t), \\
y(t) &= \mathbf{C}h(t) +  \mathbf{D}h(t).
\end{aligned}$$

Subsequently, the process of discretization is commonly employed to
incorporate Eq. <a href="#eq:lti" data-reference-type="ref"
data-reference="eq:lti">[eq:lti]</a> practical deep learning algorithms.
In this context, $\mathbf{\Delta}$ represents the timescale parameter
that is used to convert the continuous parameters
$\mathbf{A}, \mathbf{B}$ into discrete parameters,
$\mathbf{\bar{A}}, \mathbf{\bar{B}}$. The zero-order hold (ZOH) method
is commonly utilized for this discretization, and it is described as
follows: $$\begin{aligned}
\label{eq:zoh}
\mathbf{\overline{A}} &= \exp{(\mathbf{\Delta}\mathbf{A})}, \\
\mathbf{\overline{B}} &= (\mathbf{\Delta} \mathbf{A})^{-1}(\exp{(\mathbf{\Delta} \mathbf{A})} - \mathbf{I}) \cdot \mathbf{\Delta} \mathbf{B}.
\end{aligned}$$

Once discretized, Eq. <a href="#eq:zoh" data-reference-type="ref"
data-reference="eq:zoh">[eq:zoh]</a> can be reformulated with the step
size $\Delta$ as: $$\begin{aligned}
\label{eq:discrete_lti}
h_t &= \mathbf{\overline{A}}h_{k-1} + \mathbf{\overline{B}}x_{k}, \\
y_t &= \mathbf{C}h_k + \mathbf{D}x_k.
\end{aligned}$$

Nevertheless, the formulation in
<a href="#eq:discrete_lti" data-reference-type="ref"
data-reference="eq:discrete_lti">[eq:discrete_lti]</a> is predicated on
a Linear Time Invariance (LTI) system where parameters are invariant
despite changes in the input. To address this constraint, the recent
work Mamba `\cite{gu2023mamba}`{=latex} explored integrating a selective
scan technique, in which the matrices $\mathbf{\overline{B}}$,
$\mathbf{C}$, and $\mathbf{\Delta}$ are derived from the input data.
This change equipped Mamba with the ability to dynamically focus on
information from the input sequence, which increased the model’s
capability.

<figure id="fig:vl-mamba">
<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/pic/overall.png" style="width:100.0%" />
<figcaption>The architecture of VL-Mamba. It contains a Vision Encoder,
a MultiModal Connector (MMC), and a language model. We utilize the
pre-trained Mamba Large Language Model (Mamba LLM) as its language
model, and the pre-trained Vision Transformer model as its vision
encoder. </figcaption>
</figure>

## VL-Mamba Model [subsec:model]

### Overall Architecture [subsubsec:all]

The architecture of VL-Mamba consists of a pretrained vision encoder, a
randomly initialized MultiModal Connector (MMC) which incorporates the
2D vision selective scan mechanism, and a pretrained Mamba Large
Language Model (Mamba LLM), as illustrated in
Fig. <a href="#fig:vl-mamba" data-reference-type="ref"
data-reference="fig:vl-mamba">1</a>. Taking an image as input, we first
obtain visual features through the visual encoder, then feed the visual
sequences into MMC, and then this output vector combined with a
tokenized text query is fed into Mamba LLM to generate the corresponding
response.

### Vision Encoder

The vision encoder of VL-Mamba uses the Vision Transformer
(ViT) `\cite{vit}`{=latex} architecture that generates a sequence of
patch features from raw images. The vision encoder ${f_V}$, takes an
image $I$ as input and produces a sequence of the visual patch features
$V_{img}$, as follows:

$$\begin{aligned}
\label{eq:vit}
V_{img} = {f_V}(I).
\end{aligned}$$

<figure id="fig:mmp">
<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/pic/mmc.png" style="width:99.0%" />
<figcaption><span>Three architectures of MultiModal Connector: (a) MLP;
(b) MLP-VSS; (c) VSS-2 Linear Layer. </span> </figcaption>
</figure>

<figure id="fig:2D scan">
<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/pic/2Dscan.png" style="width:99.0%" />
<figcaption>Illustration of two different Vision Selective Scan (VSS)
Mechanisms: Bidirectional-Scan Mechanism (BSM) (top) and Cross-Scan
Mechanism (CSM) (bottom). </figcaption>
</figure>

### MultiModal Connector (MMC)

Since the state space models are designed to process 1D sequential data
such as language sequences that have causal relationships, but the
visual sequences generated by the vision encoder are non-causal data, 2D
vision selective scan mechanisms are proposed to solve computer vision
tasks. In this work, we try to apply the 2D vision selective scan
mechanisms for multimodal learning by ensembling them in the multimodal
connector of VL-Mamba. Specifically, we explore three variants of
multimodal connectors:

-   **MLP**: a two-layer Multi-Layer Perceptron (MLP), which is depicted
    in Fig. <a href="#fig:mmp" data-reference-type="ref"
    data-reference="fig:mmp">2</a>(a).

-   **VSS-MLP**: a Vision Selective Scan (VSS) module combined with an
    MLP. The architecture is shown in
    Fig. <a href="#fig:mmp" data-reference-type="ref"
    data-reference="fig:mmp">2</a>(b).

-   **VSS-L2**: a VSS module combined with two linear layers, which is
    depicted in Fig. <a href="#fig:mmp" data-reference-type="ref"
    data-reference="fig:mmp">2</a>(c).

The VSS module aims to bridge the gap between the 1D sequential
processing capabilities inherent in the SSM and the 2D non-causal visual
information. Specifically, the VSS module consists of a 2D vision scan
mechanism and one mamba layer. In this work, we utilize two 2D scan
mechanisms: Bidirectional-Scan Mechanism and Cross-Scan Mechanism, as
follows:

-   **Bidirectional-Scan Mechanism (BSM)** scans the image patch
    features in both forward and backward directions, which aims to
    capture a broader context without increasing computational
    complexity, as illustrated in the top of
    Fig. <a href="#fig:2D scan" data-reference-type="ref"
    data-reference="fig:2D scan">3</a>.

-   **Cross-Scan Mechanism (CSM)** unfolds image patch features into
    sequences along rows and columns and scans them in four directions
    (diagonally across the image), as shown in the bottom of
    Fig. <a href="#fig:2D scan" data-reference-type="ref"
    data-reference="fig:2D scan">3</a>.

After the scan process, these sequences are passed through the mamba
layer and reshaped back into the original image patch order, and all
such features are merged to form a comprehensive representation.

As shown in Fig. <a href="#fig:mmp" data-reference-type="ref"
data-reference="fig:mmp">2</a>(b), the input of the multimodal connector
is the sequential image patch features $V_{img}$ extracted from the
input images via the transformer-based vision encoder. These feature
vectors are then passed through a Vision Selective Scan (VSS) module to
obtain the visual scanned feature $V_{scan}$. After the VSS module, the
output vectors $V_{scan}$ are combined with the original image patch
features $V_{img}$ through a skip connection. The combined vector is
then passed into a norm layer and a two-layer Mult-Layer (MLP):

$$\begin{aligned}
\label{eq:mmc}
V_{scan} &= \mathbf{VSS}(V_{img}), \\
V_{out} &= \mathbf{MLP}(\mathbf{Norm}(V_{scan} + V_{img})).
\end{aligned}$$

And for the variant MMC in
Fig. <a href="#fig:mmp" data-reference-type="ref"
data-reference="fig:mmp">2</a>(c), the feed-forward pass progress can be
formulated as follows:

$$\begin{aligned}
\label{eq:mmc}
V_{img}^{'} &= \mathbf{Linear}(V_{img}), \\
V_{scan} &= \mathbf{VSS}(\mathbf{GELU}(V_{img}^{'})), \\
V_{out} &= \mathbf{Linear}(\mathbf{Norm}(V_{scan} + V_{img}^{'})).
\end{aligned}$$

### Mamba LLM

We use the pre-trained Mamba Large Language Model (Mamba
LLM) `\cite{gu2023mamba}`{=latex} ${f_{L}}$ as our language model. Given
a natural language query $Q$, we utilize the tokenizer and embedding
module $f_T$ to map the text input into the embedding space. Then the
visual vector $V_{out}$ and textual $T$ are concatenated and put into
the MambaLLM to obtain the response $R$.

$$\begin{aligned}
\label{eq:llm}
R = {f_{L}}(V_{out}, f_T(Q)).
\end{aligned}$$

# Experiment [sec:expri]

In this section, we first introduce our experimental setup including
implementation details and MLLM benchmarks in
Sec. <a href="#subsec:setup" data-reference-type="ref"
data-reference="subsec:setup">1.1</a>. Then we present the quantitative
comparison and qualitative results in
Sec. <a href="#subsec:sota" data-reference-type="ref"
data-reference="subsec:sota">1.2</a> and
Sec. <a href="#subsec:vis" data-reference-type="ref"
data-reference="subsec:vis">1.3</a>. Finally, we conduct ablation
studies in Sec. <a href="#subsec:abla" data-reference-type="ref"
data-reference="subsec:abla">1.4</a>.

## Experimental Setup [subsec:setup]

### Implementation details

Following `\cite{liu2023llava,liu2023improvedllava}`{=latex}, the
training process contains two stages: vision-and-language alignment
pre-training and multimodal instruction tuning. During the pretraining
stage, we freeze the vision encoder and Mamba LLM and only keep the
multimodal connector updated. Then we finetune both the multimodal
connector and the Mamba LLM in the instruction tuning stage. Our model
is trained on 8 NVIDIA Tesla A800 GPUs.

### MLLM Benchmarks

We evaluate our model on a diverse set of 8 benchmarks:
VQA-v2 `\cite{goyal2017vqav2}`{=latex},
GQA `\cite{hudson2019gqa}`{=latex},
ScienceQA-IMG `\cite{lu2022learn}`{=latex},
TextVQA `\cite{singh2019textvqa}`{=latex},
POPE `\cite{li2023pope}`{=latex}, MME `\cite{fu2023mme}`{=latex},
MMBench `\cite{Liu2023MMBenchIY}`{=latex},
MM-Vet `\cite{yu2023mmvet}`{=latex}.
VQA-v2 `\cite{goyal2017vqav2}`{=latex} evaluates models’ ability to
understand and reason about images and questions.
GQA `\cite{hudson2019gqa}`{=latex} assesses spatial understanding and
multi-step inference in real-world images.
ScienceQA `\cite{lu2022learn}`{=latex} offers multimodal multiple-choice
questions on scientific topics, requiring common sense reasoning. The
questions in TextVQA `\cite{singh2019textvqa}`{=latex} are related to
the text in an image, it evaluates the model’s optical character
recognition (OCR) and inference capabilities.
POPE `\cite{li2023pope}`{=latex} provides a benchmark for evaluating
object hallucinations, which is a binary classification task that
prompts the model to answer whether an object exists.
MME `\cite{fu2023mme}`{=latex} evaluates perceptual and cognitive
abilities, including OCR, object recognition, common sense reasoning,
numerical calculations, text translation, and code reasoning.
MMBench `\cite{Liu2023MMBenchIY}`{=latex} features 3,000 single-choice
questions across 20 dimensions, using a CircularEval strategy for robust
evaluation, with ChatGPT matching model predictions to choices.
MM-Vet `\cite{yu2023mmvet}`{=latex} identifies 16 emergent tasks from
core visual and linguistic (VL) capabilities, including Recognition,
Knowledge, OCR, Spatial awareness, Language generation, and Math.

<figure id="fig:vis">
<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/pic/vis-arxiv.png" style="width:99.0%" />
<figcaption>Examples of response generated by VL-Mamba. </figcaption>
</figure>

## Quantitative Evaluation [subsec:sota]

As is shown in Table <a href="#tab:results" data-reference-type="ref"
data-reference="tab:results">[tab:results]</a>, we compare our proposed
model VL-Mamba with some SoTA multimodal large language models. Compared
with the MobileVLM-3B `\cite{Chu2023MobileVLMA}`{=latex} model with
similar scale parameters and the same amount of multimodal training
data, our model surpasses the performance on SQA$^\text{I}$ (65.4 v.s.
61.2), VQA$^\text{T}$ (48.9 v.s. 47.5), and MME (1369.6 v.s. 1288.9),
though the Mamba LLM uses much less pretrained tokens (627B) than the
backbone MobileLLaMA (1.3T) of MobileVLM. Compared with the
LLaVA-Phi `\cite{zhu2024llavaphi}`{=latex} model with a SoTA language
model Phi-2-2.7B with 1.4T pretrained tokens, our performance shows
superior on VQA-v2 (76.6 v.s. 71.4), MME (1369 v.s. 1335.1), and MM-Vet
(32.6 v.s. 28.9). It is worth noting that though our proposed model has
fewer parameters and limited training data, it also achieves comparable
performance compared with some models with a larger number of
parameters. Its performance on the POPE benchmark is similar to
LLaVA-1.5 `\cite{liu2023improvedllava}`{=latex}, where the LLM
parameters are 13B, which is approximately 4.6 times larger than the
Mamba LLM. These promising results demonstrate the effectiveness of our
proposed VL-Mamba and show the potential of utilizing the state space
models in multimodal learning tasks.

## Qualitative Result [subsec:vis]

We present some examples to see the qualitative results of the VL-Mamba.
As shown in Fig. <a href="#fig:vis" data-reference-type="ref"
data-reference="fig:vis">1</a>, the VL-Mamba could well understand the
user’s question and respond accurately.

## Ablation Study [subsec:abla]

### The Effect of Variants of Language Model

Table <a href="#tab:lang" data-reference-type="ref"
data-reference="tab:lang">[tab:lang]</a> shows the ablation experiment
of evaluating the effectiveness of different variants of the language
model. We conduct experiments on three different variants, Mamba-1.4B
which has 1.4B parameters and is trained on
Pile `\cite{gao2020pile}`{=latex} with 300B tokens, Mamba-2.8B-Pile with
2.8B parameters and trained on Pile 300B tokens and Mamba-2.8B-Slimpj
trained on SlimPajama with 627B tokens. Specifically, we construct the
baseline models by using the same variant CLIP-ViT as the vision
encoder, Mamba language models as backbone large language models, and
vanilla MLP MultiModal Connectors without 2D vision selective scan
modules. We can see with the increase of model scale and training
tokens, Mamba-2.8B-Slimpj outperforms the other two variants on all
benchmarks. Thus, we choose Mamba-2.8B-Slimpj for other experiments.

<div class="table*" markdown="1">

</div>

### The Effect of Different Vision Encoders

To evaluate the effectiveness of different vision encoders, we conduct
an ablation study which is shown in
Table <a href="#tab:visenc" data-reference-type="ref"
data-reference="tab:visenc">[tab:visenc]</a>. We study two different
vision encoders, CLIP-ViT-L `\cite{Radford2021LearningTV}`{=latex} and
SigLIP-SO `\cite{Zhai2023SigmoidLF}`{=latex}. The baseline models
utilize Mamba-2.8B-Slimpj as LLM and vanilla MLP multimodal connectors.
We can see that the CLIP-based model falls behind the SigLIP-based model
in most benchmarks except the MME benchmark, where the CLIP-based model
surpasses the SigLIP-based model by a large margin. Considering the
comprehensive performance, we choose SigLIP-SO as the vision encoder to
build the final VL-Mamba.

<div class="table*" markdown="1">

</div>

### Ablation on Different MMC Architectures

We also explore the impact of different architectures of Multimodal
Connector (MMC). We evaluate three different MMC variants: MLP, VSS-MLP,
and VSS-L2. As shown in
Table <a href="#tab:arch-mmc" data-reference-type="ref"
data-reference="tab:arch-mmc">[tab:arch-mmc]</a>, by comparing the three
architectures, we observe that VSS-L2 shows relatively better
performance on most benchmarks, especially on the VQA$^\text{T}$, MME,
MMB, and MM-Vet. The scores are 48.9, 1369.6, and 32.6 respectively,
which proves the effectiveness of the VSS module combined with linear
layers. Note that these models utilize SigLIP-SO as the vision encoder,
Mamba-2.8B-Slimpj as the language model and Bi-directional selective
scan mechanism.

<div class="table*" markdown="1">

</div>

### Ablation on Different Scan Mechanisms

We compare two scan mechanisms Bidirectional-Scan Mechanism (BSM) and
Cross-Scan Mechanism (CSM) in the MMC module. As shown in
Table <a href="#tab:scan" data-reference-type="ref"
data-reference="tab:scan">[tab:scan]</a>, although BSM and CSM perform
similarly in some benchmarks, such as they all score 76.6 in the
VQA$^\text{v2}$, BSM exhibits superior performance in most benchmarks.
Especially on the MMB benchmark, BSM scored 1369.6, 5.6 points higher
than CSM, highlighting its strength in processing 2D vision information
for multimodal learning tasks.

<div class="table*" markdown="1">

</div>

# Limitation

In this paper, we are focusing on effectively applying the 2D selective
scan for multi-modal connector in the VL-Mamba, without exploring the
training data that would significantly affect the benchmark performance.
In the future, we will study how to utilize higher-quality training data
to further improve the performance of VL-Mamba.

# Conclusion

In this paper, we introduce VL-Mamba, the first work that explores the
state space model Mamba to solve multimodal learning tasks. The VL-Mamba
consists of a language model, a vision encoder, and a multimodal
connector. To be specific, we utilize the pre-trained Mamba Large
Language Model (Mamba LLM) as the language model. Then, we study three
architectures of MultiModal Connector (MMC) and introduce a Vision
Selective Scan (VSS) module in MMC to bridge the gap between 2D
non-causal image information and the inherent causal modeling
capabilities of state space models (SSMs). In the VSS module, we propose
two 2D scan mechanisms: the Bidirectional Scanning Mechanism (BSM) and
Cross Scanning Mechanism (CSM). We conduct extensive experiments on
eight multimodal benchmarks and achieve comparable performance with some
SoTA MLLMs, and we also conduct ablation studies to evaluate the
effectiveness of language variants, different vision encoders, different
MMC architectures, and different scan mechanisms. The results
demonstrate the effectiveness of our proposed model and prove the
potential of the SSMs applied to multimodal learning.