<div class="center" markdown="1">

<https://github.com/thunlp/LLaVA-UHD>

</div>

# Introduction

Recent progress in Large Multimodal Models
(LMMs) [2023llava1.6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [instructblip2023](None), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [liu2024llava](http://arxiv.org/pdf/2402.11690v1), [bai2023qwen](None)
has witnessed a significant surge in vision-language understanding,
reasoning, and interaction capabilities. This is achieved by projecting
visual signals into Large Language Models (LLMs) to enable their visual
perception of the world, where visual encoding strategy plays a
fundamental
role [li2023blip2](None), [Alayrac2023Flamingo](http://arxiv.org/pdf/2205.07065v1), [liu2024llava](http://arxiv.org/pdf/2402.11690v1).
Real-world images are known to reside in a wide range of aspect ratios
and resolutions, presenting significant challenges for LMMs in various
applications.

However, most existing
LMMs [chen2023shikra](http://arxiv.org/pdf/2306.15195v2), [instructblip2023](None), [liu2024llava](http://arxiv.org/pdf/2402.11690v1)
perceive images in a fixed aspect ratio (i.e., 1:1) and a low resolution
(i.e., 224$\times$`<!-- -->`{=html}224). The compromise to this
simplified setting typically leads to severe shape distortion and blur
of image contents. The problem significantly hurts the capabilities of
LMMs, especially for fine-grained capabilities, such as small object
understanding [li2023otterhd](None) and optical character
recognition [ye2023ureader](None), [bai2023qwen](None), [hong2023cogagent](None).
Moreover, the issue also exacerbates hallucination problems (i.e.,
producing textual responses not factually grounded in images), since
models can only learn to make best guesses to blurred
images [sun2023aligning](None), [yu2023rlhf](None).

To achieve image perception in varied aspect ratios and high resolutions
for LMMs, there are two main challenges: (1) Adaptivity. Since visual
encoders (e.g., CLIP-ViT [radford2021clip](http://arxiv.org/pdf/2404.19696v1)) are
pretrained in fixed resolutions, it can be difficult to deal with images
in a wide range of aspect ratios and resolutions. Simple image
interpolation that deviates far from the pretraining scenarios can
result in out-of-distribution issues. (2) Efficiency. Directly encoding
high-resolution images using vision
Transformers [dosovitskiy2020vit](http://arxiv.org/pdf/2105.15075v2) requires quadratic
computation cost with respect to image sizes. In addition, it can be
even more costly for LLMs to process the large number of visual tokens
from high-resolution images (e.g., 4096 tokens for
896$\times$`<!-- -->`{=html}896 images in ViT-L/14).

Moreover, careless visual encoding strategies can even result in
systematic flaws in correctness. For example, despite its powerful
capabilities in various aspects, it has been commonly reported that
GPT-4V [achiam2023gpt4](None) can surprisingly struggle in
some basic capabilities, such as identifying the number of
objects [yang2023dawn](None). The mechanistic cause for such
embarrassment remains largely unknown. In this work, we perform the
first mechanistic investigation of GPT-4V flaws from the perspective of
visual encoding strategy. Our controlled experiments in probing GPT-4V
show that the problem can be partially rooted in its visual encoding
strategy in dealing with high-resolution images. Investigation on
LLaVA-1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1), a representative
open-source LMM also shows systematic issues in correctness, indicating
their potential vulnerability for adversarial attacks.

To address the challenges, we present LLaVA-UHD, a large multimodal
model that efficiently perceives any aspect ratio and high-resolution
images. The model has three key components: (1) At the core of LLaVA-UHD
is an image modularization strategy that divides native-resolution
images into smaller variable-sized slices for efficient and extensible
encoding. In comparison to recent works that fit images into several
fixed aspect ratios and
resolutions [SPHINX2023](None), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), the
variable-sized slices in LLaVA-UHD enable full adaptivity to
native-resolution images without padding or shape-distorting resizing.
This is in analogy to the better adaptivity of using water drops vs. ice
cubes in full-filling variable-sized glasses. We also show that the
strategy guarantees minor deviation from the pretraining setting of
visual encoders to maximally retain their capabilities. (2) The visual
tokens are condensed by a compression layer to modest lengths, largely
reducing the computation for LLMs. (3) Finally, the compressed slice
tokens are organized in a spatial schema to inform LLMs about the slice
positions in the image.

Comprehensive experiments on 9 benchmarks show that LLaVA-UHD
significantly improves the capabilities of LMMs, outperforming
established counterparts trained with 2-3 orders of magnitude more data.
Notably, our model built on LLaVA-1.5$_{336\times336}$ supports
672$\times$`<!-- -->`{=html}1088 resolution images using only 94%
inference computation, and achieves 6.4 accuracy improvement on TextVQA
and 3.2 accuracy improvement on POPE. The advantage enlarges with more
extreme aspect ratios. We also show that instruction tuning on ViT
parameters is sufficient for adaptation to a broad range of images.
Moreover, the model can be efficiently trained in academic settings,
within 23 hours (vs. 26 hours of LLaVA-1.5) on 8 A100 GPUs.

The contribution of this work can be summarized as threefold: (1) We
perform the first mechanistic investigation of GPT-4V from the
perspective of visual encoding strategy and expose systematic flaws. (2)
We present LLaVA-UHD, a large multimodal model that can efficiently
perceive any aspect ratio and high-resolution images. (3) We conduct
comprehensive experiments to demonstrate the effectiveness of LLaVA-UHD
on 9 popular benchmarks, and also provide analysis for deeper
understanding of the model.

# Pilot Experiments [sec:pilot_exp]

We start with a pilot experiment on the visual encoding strategies of
existing LMMs, taking GPT-4V [achiam2023gpt4](None) and
LLaVA-1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1) as representative examples.
GPT-4V is a powerful and most recognized proprietary LMM, while
LLaVA-1.5 is one of the most influential open-source LMMs. Despite their
strong performance in many aspects, it has been commonly reported that
dilemmas can be encountered in some basic
capabilities [yang2023dawn](None). For example, GPT-4V is
prone to miscounting the object numbers in images, whereas the causes
remain largely unknown.

In this work, we perform the first mechanistic investigation of GPT-4V
flaws from the perspective of visual encoding strategy. The key idea is
that by using synthetic images as continuous probes, we can evaluate the
behaviors of GPT-4V in a highly controlled manner, thereby identifying
the underlying causes. Our experimental results indicate that, some
systematic flaws of GPT-4V are likely to be rooted in its visual
encoding strategy, which can be potentially exploited for adversarial
attacks.

## GPT-4V Experiments

**Preliminary.** According to the publicly available information from
OpenAI,[^2] GPT-4V employs two image processing modes: low resolution
and high resolution. (1) In low-resolution mode, for an original image
with dimensions W and H, the model processes only a low-resolution
overview image. (2) In high-resolution mode, besides the overview image,
GPT-4V processes additional slices of the original high-resolution
image, where each slice has $512\times512$ resolution, resulting in
$\lceil \frac{W}{512} \rceil \times \lceil \frac{H}{512} \rceil$ slices
in total. In our experiments on GPT-4V’s new high-resolution mode,
interesting error patterns are observed, prompting an exploration into
GPT-4V’s underlying visual encoding logic.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.11703v1_md/figures/gpt4_exp1.png" style="width:100.0%" />

</div>

**How do positions in images influence GPT-4V’s behavior?** Our
experiments start with a simple instance: Given the image as shown in
Fig. <a href="#fig:gpt4v_exp1" data-reference-type="ref"
data-reference="fig:gpt4v_exp1">[fig:gpt4v_exp1]</a>(a), we ask GPT-4V:
“How many circles are there in the image?” We synthesize a series of
image variants by changing the positions of circles in the image, and
keep the text prompt unchanged. For better reliability, we also
synthesize images using other colors and shapes as well, in
$\{\text{red}, \text{green}, \text{white}\} \times\{ \text{circle}, \text{triangle}, \text{square}\}$.
For each instance, we query 15 times to better approximate the true
response distribution.

We calculate the average number answered by GPT-4V for each position in
the image, and report the heatmap in
Fig. <a href="#fig:gpt4v_exp1" data-reference-type="ref"
data-reference="fig:gpt4v_exp1">[fig:gpt4v_exp1]</a>(b). We can observe
that the result is highly correlated with object positions in images.
Specifically, the patterns are split by $256\times256$ squares, and
three interesting patterns can be identified: (1) The central square
exhibits the highest response number, (2) the middle edges show a lower
number, and (3) the corners are the closest to ground truth.

To investigate the cause, we further separate the model responses by
number, and report the distribution across positions for each response
in Fig. <a href="#fig:gpt4v_exp1" data-reference-type="ref"
data-reference="fig:gpt4v_exp1">[fig:gpt4v_exp1]</a>(c), (d), (f), (g)
and (h). Interestingly, besides the correct answers (4: 66.1%) and close
answers (5: 16.6%, 3: 10.2%), it turns out that the remaining two
abnormal answers (8: 5.2%, 16: 1.9%), which doubles and quadruples the
ground truth, account for the error pattern in
Fig. <a href="#fig:gpt4v_exp1" data-reference-type="ref"
data-reference="fig:gpt4v_exp1">[fig:gpt4v_exp1]</a>(b). Combining the
results with the public information from OpenAI, we hypothesize the most
likely cause is that, there are overlaps in the slices of GPT-4V when
the image resolution is not divisible by 512.[^3] As illustrated in
Fig. <a href="#fig:gpt4v_exp1" data-reference-type="ref"
data-reference="fig:gpt4v_exp1">[fig:gpt4v_exp1]</a>(e), the overlapping
areas between two slices will double the number, and the overlapping
areas between four slices will quadruple the number.[^4]

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.11703v1_md/figures/gpt4_exp2.png" style="width:100.0%" />

</div>

**How do image resolutions influence GPT-4V’s behavior?** To verify the
hypothesis, we further probe GPT-4V through continuously changing image
resolutions. Specifically, we proportionally resize the image in
Fig. <a href="#fig:gpt4v_exp2" data-reference-type="ref"
data-reference="fig:gpt4v_exp2">[fig:gpt4v_exp2]</a>(a) into different
resolutions, and query about the object number in the same way. For each
resolution, we repeatedly query 30 times for better reliability.

We report the experimental results in
Fig. <a href="#fig:gpt4v_exp2" data-reference-type="ref"
data-reference="fig:gpt4v_exp2">[fig:gpt4v_exp2]</a>(b). We observe that
the model responses show a significant phase change with image
resolutions: (1) In phase 1, since there are no image slices, most
answers are correct; (2) In phase 2, answer 12 dominates the responses
possibly due to the incomplete circles in each slice. (3) Phase 3 shows
mixed answers of 9, 12 and 16. Note that 16 can be well explained by the
error pattern in
Fig. <a href="#fig:gpt4v_exp1" data-reference-type="ref"
data-reference="fig:gpt4v_exp1">[fig:gpt4v_exp1]</a>(e). We refer
readers to
Section <a href="#sec:GPT-4V-illustration" data-reference-type="ref"
data-reference="sec:GPT-4V-illustration">7</a> for a more detailed
illustration of each phase. Besides, we also notice that many abnormal
phenomenons in Fig. <a href="#fig:gpt4v_exp2" data-reference-type="ref"
data-reference="fig:gpt4v_exp2">[fig:gpt4v_exp2]</a>(b) cannot be
perfectly explained yet, which we leave for future work.

In conclusion, these experimental findings shed light on GPT-4V’s
potential vulnerabilities in high-resolution image processing,
warranting further investigation into the implications of these
weaknesses and the development of strategies to counter potential
adversarial attacks on LMMs.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.11703v1_md/figures/padding_hallucination.png" style="width:100.0%" />

</div>

## LLaVA-1.5 Experiments

To deal with images with varied aspect ratios, LLaVA-1.5 pads the input
images into squares before feeding them into the visual encoder. This
encoding method results in a waste of computation for non-square images.
For example, a 1:4 image has only 25% effective computation after
padding into squares. To quantify the influence, we train an unpadded
version of LLaVA-1.5, by fitting the ViT position embedding into the
aspect ratio of input images using 2D interpolation. The resultant image
tokens remain no more than 576 as in LLaVA-1.5 (see
Section <a href="#sec:encoding" data-reference-type="ref"
data-reference="sec:encoding">3.1</a>). From the experimental results in
Table <a href="#tab:module_ablations" data-reference-type="ref"
data-reference="tab:module_ablations">[tab:module_ablations]</a>, we
observe that adaptive aspect ratio encoding without padding consistently
improves the performance of LLaVA-1.5.

Another issue of padding is that, the model essentially cannot know
whether the padding-like pixels come from image pre-processing or an
actual part of the original input image. To demonstrate this issue, we
synthesize a series of input images as in
Fig. <a href="#fig:llava_exp" data-reference-type="ref"
data-reference="fig:llava_exp">[fig:llava_exp]</a>(right), where
blue/green/red rectangles in various aspect ratios are surrounded by
grey (i.e., the color of LLaVA-1.5’s padding RGB value). Given the input
image, we prompt: “What is the color of the left/right/top/bottom most
area?” From the results in
Fig. <a href="#fig:llava_exp" data-reference-type="ref"
data-reference="fig:llava_exp">[fig:llava_exp]</a>(left), we observe
that LLaVA-1.5 neglects the grey input areas (considering them as
padding), and faithfully responds with the color of the central
rectangle.

## Conclusions on Pilot Experiments

In summary, both powerful proprietary LMMs such as GPT-4V and
open-source LLaVA-1.5 have systematic issues in their underlying visual
encoding strategies. The results show that visual strategies must be
designed with caution. Common practices such as padding,
shape-distorting resizing, and repetitive slicing can result in a waste
of computation, a loss of model capability, and even vulnerability to
adversarial attacks. Therefore, there is an urgent need for more
adaptive and efficient visual encoding methods.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.11703v1_md/figures/framework.png" style="width:100.0%" />

</div>

# Method

Based on the principles learned from the pilot experiments, we propose
LLaVA-UHD, a large multimodal model that can efficiently perceive any
aspect ratio and high-resolution images. As shown in
Fig. <a href="#fig:framework" data-reference-type="ref"
data-reference="fig:framework">[fig:framework]</a>, the model includes
three key components: (1) An image modularization strategy that divides
native-resolution images into smaller variable-sized slices for
efficient and extensible encoding, (2) a compression module that further
condenses image tokens from visual encoders, and (3) a spatial
decoration schema to organize slice tokens for LLMs.

## Modularized Visual Encoding [sec:encoding]

To deal with high-resolution images with varied aspect ratios, a naive
approach is to interpolate the position embeddings of ViT to the target
shape for direct encoding as a whole. However, this approach is
sub-optimal due to the quadratic computation cost and the performance
degradation from out-of-distribution issues. To address the challenge,
we present a modularized visual encoding strategy. The basic idea is to
divide native-resolution images into smaller variable-sized slice
slices, where the shape of each slice does not deviate too far from the
standard pretraining setting of ViT. With variable-sized slice slices,
LLaVA-UHD can achieve full adaptivity to native-resolution images
without padding or shape-distorting reshaping.

**High-Resolution Image Partition Strategy.** The goal of image slicing
strategy is to determine a split of high-resolution images, with minimal
changes to the resolutions of each slice. Given an image in resolution
$(W_I, H_I)$ and a ViT pretrained in resolution $(W_v, H_v)$, we first
determine the number of slices (i.e., the ideal computation) needed to
process the image:
$N=\lceil \frac{W_I\times H_I}{W_v\times H_v} \rceil$. Then we factorize
the slice number $N$ into $m$ columns and $n$ rows:
$\mathbb{C}_N= \{(m, n)| m\times n = N, m\in \mathbb{N}, n\in \mathbb{N} \}$.
To select the most appropriate partition, we define a score function to
measure the deviation from the standard pretraining setting of ViT:

$$\small
    S(W_I, H_I, W_v, H_v, m, n)= -\left| \log \frac{W_I \times n}{H_I \times m} - \log \frac{W_v}{H_v}\right|,$$
where higher score $S(\cdot)$ indicates a smaller deviation from the
standard setting of ViT, and is thus preferred. Therefore the partition
can be obtained as follows:

$$\small
    m^*, n^* = \mathop{\mathrm{arg\,max}}_{(m,n)\in \bar{\mathbb{C}}} S(W_I, H_I, W_v, H_v, m, n),
\label{equ:partition}$$ where the candidate set
$\bar{\mathbb{C}} = \mathbb{C_N}$. In practice, we notice that in some
cases, there might be only a few possible factorization schemes for $N$,
especially for prime numbers, which can lead to limited choices and
therefore extreme partitions of images. For example, $N=7$ has only two
extreme partition choices, 1:7 and 7:1. To address the issue, in
addition to the ideal slice number $N$, we also allow a modest change of
slice numbers $N-1, N+1$ to incorporate more plausible partition
choices. Therefore, the final partition is given by
Equation <a href="#equ:partition" data-reference-type="ref"
data-reference="equ:partition">[equ:partition]</a>, where
$\bar{\mathbb{C}} = \mathbb{C}_{N-1} \cup \mathbb{C}_{N} \cup \mathbb{C}_{N+1}$.

Theoretically, we show that the partition strategy guarantees minor
expected changes and modest worst-case changes with respect to standard
pretraining resolution $(W_v, H_v)$ for each slice. Specifically, we
show that for input images where $N \leq 20$ and aspect ratio in
$[1:6, 6:1]$, the aspect ratio of each slice resides within
$[1:2, 2:1]$, and the area of each slice resides within
$[0.33W_IH_I, 1.5W_IH_I]$. We refer readers to
Section <a href="#sec:proofs" data-reference-type="ref"
data-reference="sec:proofs">8</a> for full proof details.

**Arbitrary Aspect Ratio Slice Encoding.** Most existing LMMs utilize a
static resolution for image slice
encoding [bai2023qwen](None), [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1), [instructblip2023](None).
This essentially prevents full adaptivity to native resolutions, since
only several predefined fixed-shape slices are available. Moreover, the
static slice resolution inevitably incurs padding or shape-distorting
resizing, which hurts the performance, efficiency, and even correctness
as discussed in
Section <a href="#sec:pilot_exp" data-reference-type="ref"
data-reference="sec:pilot_exp">2</a>.

To address the problem, we propose to encode image slices in aspect
ratios given by the partition strategy as is. Specifically, we
proportionally resize the original image following the aspect ratio,
such that the number of patches maximally fits within the pretraining
budget $M$ (i.e., the number of position embeddings in ViT). Then we
reshape the pretrained 1D position embedding sequence of ViT into 2D
format $P \in \mathbb{R}^{q\times q \times l}$ following its pretraining
setting, where $M=q\times q$, and $l$ is the dimension of position
embeddings. After that, we 2D-interpolate $P$ to fit the slice
resolution given by the partition strategy for visual encoding. In our
experiments, we show that ViT and position embedding parameters can be
kept frozen during pretraining, and updating these parameters during the
instruction-tuning stage is sufficient for good performance. In addition
to slices, we also provide a low-resolution overview image in native
aspect ratio. The overview image can provide coarse-grained information
and global semantic connections in images.

## Compression Layer

High-resolution images require LLMs to process significantly more visual
tokens, which accounts for a major part of the computation. For example,
a $672\times 1008$ resolution image will produce 3,456 visual tokens for
LLaVA-1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1). To address the issue, we
compress the visual tokens of each image slice using a shared perceiver
resampler layer [Alayrac2023Flamingo](http://arxiv.org/pdf/2205.07065v1). Specifically,
image tokens output by the visual encoders are resampled to a lower
number using a set of query vectors via cross-attention (from $576$ to
$64$ in our experiments). Compared with the prevalent MLP-based visual
projection
approaches [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1), [2023llava1.6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [wang2023cogvlm](None),
perceiver resampler maintains a fixed and affordable number of visual
tokens regardless of image resolutions, and is therefore more compatible
with high-resolution image understanding. As a result, LLaVA-UHD can
encode $672\times1008$ resolution images using an even lower computation
cost than LLaVA-1.5 in encoding $336\times336$ resolution images.

## Spatial Schema for Image Slices

Since the image partition is dynamic across different images, it is
necessary to inform LLM of the spatial organizations of image slices.
Inspired by [fuyu2023](adept.ai/blog/fuyu-8b), we design a spatial schema to
inform the relative positions of image slices using two special tokens.
Specifically, we use “,” to separate the slice representations in a row,
and use “\n” to separate different rows. In our experiments, we find
that the simple schema can effectively inform the dynamic partition to
yield good performance.

# Experiments

In this section, we empirically investigate the effectiveness of
LLaVA-UHD. We first provide the implementation details, and report the
evaluation results on 9 common benchmarks compared with strong
baselines. Then we provide analytic results for better understanding of
the model.

## Implementation Details

**Model Configuration.** In this work, we built LLaVA-UHD following the
implementation of LLaVA-1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1).
Specially, we use the CLIP-ViT-L/14 as visual encoder (default
resolution ${336\times336}$),
Vicuna-13B [chiang2023vicuna](None) as LLM, and a shared
visual resampler [bai2023qwen](None) as the projector to
connect the visual encoder and LLM. During the encoding of image slices,
a minor reshape within half patches (maximum 7-8 pixels) could be
performed to fit the slice into patches. The number of learnable queries
in resampler is set to 64. For the image partitioned as $N$ sub-patches,
the number of visual tokens fed into LLM is $64\times(N+1)$, with tokens
of the low-resolution overview image. We set the maximum $N$ to be 6 in
experiments, which supports a maximum of $672\times1008$ resolution
images. Following LLaVA-1.5, we perform a two-stage training as follows.

**Stage 1: Pretraining details.** During this stage, only the perceiver
resampler is tuned, with the CC-595K
dataset [liu2024llava](http://arxiv.org/pdf/2402.11690v1) for 1 epoch, using AdamW
optimizer with a learning rate of $1e^{-3}$ and the cosine learning rate
schedule. The global batch size is set to 256. The training cost of this
stage is $\sim$`<!-- -->`{=html}5 hours using 8$\times$A100 GPUs.

**Stage 2: Instruction-tuning details.** During this stage, the visual
encoder is frozen and we fine-tune the visual resampler and LLM, with a
656K mixture dataset [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1) which contains
LLaVA-Instruct [liu2024llava](http://arxiv.org/pdf/2402.11690v1),
TextVQA [singh2019textqa](None),
GQA [hudson2019gqa](None),
OCR-VQA [mishra2019ocrvqa](None), and Visual
Genome [krishna2017vg](None). The learning rate is 2$e^{-5}$
and batch size is 128. Other settings are the same as stage 1. The
training cost of this stage is $\sim$`<!-- -->`{=html}18 hours using
8$\times$A100 GPUs.

## Experimental Setting

We introduce experimental settings, including the benchmarks, evaluation
metrics, and baselines.

**Benchmarks.** We adopt 9 popular benchmarks to evaluate our model,
including: (1) General visual question answering benchmarks such as
VQA-V2 [antol2015vqa](None),
GQA [hudson2019gqa](None),
ScienceQA [lu2022scienceqa](http://arxiv.org/pdf/2209.09513v2), and
VizWiz [gurari2018vizwiz](None); (2) Optical character based
visual question answering benchmark such as
TextVQA [singh2019textqa](None); (3) Hallucination benchmark
such as POPE [li2023pope](http://arxiv.org/pdf/2402.15721v1); (4) Comprehensive benchmarks
such as MME [fu2023mme](None),
MMBench [liu2023mmbench](None), and
MMBench-CN [liu2023mmbench](None).

**Evaluation Metrics.** In addition to the performance on popular
benchmarks, we also report the computation cost (TFLOPs) in processing
an image in the maximum supported resolution. The computation cost is
aggregated from the visual encoder, projector, and LLM. We also report
the accumulated multimodal training data volume for reference, which
includes image-text pairs used during pertaining and instruction tuning.
For models post-trained on existing multimodal models as backbones, this
also includes the training data of the backbones.

**Baselines.** We compare our model with strong baselines. (1) General
baselines. We adopt Qwen-VL [bai2023qwen](None),
LLaVA-1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1),
MiniGPT-v2 [chen2023minigptv2](None),
Shikra [chen2023shikra](http://arxiv.org/pdf/2306.15195v2),
BLIP-2 [li2023blip2](None) and
InstructBLIP [instructblip2023](None) as representative
general baselines. Since the implementation of LLaVA-UHD is highly
aligned with LLaVA-1.5, it serves as the most direct baseline. (2)
High-resolution LMMs. SPHINX [SPHINX2023](None) and
mPLUG-Owl2 [ye2023owl2](http://arxiv.org/pdf/2311.04257v2) encode images in fixed
resolutions; Ureader [ye2023ureader](None) and
Monkey [li2023monkey](http://arxiv.org/pdf/2103.15488v1) support enumerated resolution
types (several predefined fixed-shape slices);
Fuyu-8B [fuyu2023](adept.ai/blog/fuyu-8b) and
OtterHD-8B [li2023otterhd](None) can encode images in any
resolutions.

<span id="tab:sota" label="tab:sota"></span>

## Main Results

We report the main experimental results in
Table <a href="#tab:sota" data-reference-type="ref"
data-reference="tab:sota">[tab:sota]</a>, from which we have the
following observations: (1) LLaVA-UHD outperforms strong baselines on
popular benchmarks. This includes strong general baselines trained on
2-3 orders of magnitude more data such as Qwen-VL and InstructBLIP, and
also high-resolution LMMs that require significantly more computation
such as Fuyu-8B, OtterHD-8B, Monkey and SPHINX-2k. The results show that
LLaVA-UHD can properly deal with native-resolution images for strong
performance, as well as good data and computation efficiency. (2)
LLaVA-UHD achieves significant improvements over the LLaVA-1.5 backbone.
Notably, by simply perceiving images in native high-resolution,
LLaVA-UHD achieves 6.4 accuracy improvement on TextVQA and 3.2 accuracy
improvement on POPE. The reason is that the blurred content in
low-resolution images can prevent LMMs from accurately identifying the
challenging fine-grained objects and optical characters. The results
demonstrate the fundamental role of perceiving native high-resolution
images in various multimodal tasks, and the effectiveness of LLaVA-UHD
in addressing the problem. (3) In terms of resolution and efficiency,
compared with LLaVA-1.5 associated fixed $336\times336$ resolution,
LLaVA-UHD supports 672$\times$`<!-- -->`{=html}1088 resolution images in
any aspect ratio using only 94% inference computation. The results
indicate promising scalability of LLaVA-UHD to potentially larger
resolutions in future.

## Analytic Results

We provide further analytic results, including ablation on alternative
components, evaluation on images with more extreme aspect ratios, best
practice for frozen/trainable parameters, and case study.

**Ablation Study.** In
Table <a href="#tab:module_ablations" data-reference-type="ref"
data-reference="tab:module_ablations">[tab:module_ablations]</a>, we
conduct ablation studies on alternative components. (1) We replace the
padding strategy of LLaVA-1.5 with the adaptive encoding strategy of
LLaVA-UHD, supporting arbitrary aspect ratios while maintaining
identical maximum resolutions. We can observe consistent improvement
since wasted computation from padding is avoided. (2) We replace the
perceiver resampler of LLaVA-UHD with the 2-layer MLP of LLaVA-1.5. We
observe that perceiver resampler achieves comparable or better
performance than MLP, using only 12.9% computation cost. (3) We further
replace the LLaVA-UHD image partition strategy with the naive partition
strategy [SPHINX2023](None) (i.e., fixed $2\times2$ slices).
Results show that LLaVA-UHD can more properly divide images into slices
for better performance. (4) We remove the spatial schema from LLaVA-UHD.
The performance degradation demonstrates the effectiveness and necessity
of spatial schema in informing the dynamic slice positions for LMMs.

**LLaVA-UHD generalizes to images with extreme aspect ratios.** We
further investigate the generalization capability of LLaVA-UHD by
constructing an extended version of existing benchmarks. Specifically,
we expand the aspect ratio of an image by doubling the length of its
longer side through padding. From the results in
Table <a href="#tab:padding_evaluation" data-reference-type="ref"
data-reference="tab:padding_evaluation">[tab:padding_evaluation]</a>, we
can see that the advantage of LLaVA-UHD increases as compared with
LLaVA-1.5 and alternatives. The reason is that LLaVA-UHD perceives
images in native aspect ratios. In comparison, LMMs that encode images
in fixed aspect ratios will suffer from significant distortion in the
content shapes. Moreover, this also causes the computation to be
unevenly distributed along the width and height of the image content.

<div id="tab:frozen_stage" markdown="1">

| Update ViT |  | VQA$^\mathrm{v2}$ | GQA | VQA$^\mathrm{T}$ | POPE | SQA | VizWiz |  |  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1-2 pre-training | Fine-tuning |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  | **81.7** | **65.2** | **67.7** | **89.1** | **72.0** | **56.1** |  |  |  |  |  |  |  |
|  |  | 78.2 | 61.1 | 58.9 | 83.9 | 68.6 | 51.4 |  |  |  |  |  |  |  |
|  |  | 79.4 | 64.5 | 65.7 | 87.3 | 71.9 | 55.4 |  |  |  |  |  |  |  |
|  |  | 80.2 | 63.7 | 62.6 | 87.2 | 71.6 | 55.1 |  |  |  |  |  |  |  |
| LLaVA-1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1) |  | 80.0 | 63.3 | 61.3 | 85.9 | 71.6 | 53.6 |  |  |  |  |  |  |  |

The effect of tuning visual encoder at different training stages.

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.11703v1_md/figures/case.png" style="width:100.0%" />

</div>

**Instruction-tuning ViT parameters is sufficient for adaptation.** We
investigate the effect of tuning ViT parameters at different training
stages, including pretraining and instruction-tuning. From the results
in Table <a href="#tab:frozen_stage" data-reference-type="ref"
data-reference="tab:frozen_stage">1</a>, we observe that: (1) Updating
ViT during instruction-tuning is sufficient to achieve good performance.
In fact, we find that LLaVA-UHD can improve over LLaVA-1.5 even when ViT
parameters are frozen in both pretraining and instruction tuning. (2)
Further updating ViT during pretraining does not lead to better results.
We hypothesize the reason is that jointly training ViT and resampler
(from scratch) on limited pretraining data can lead to instability
issues.

**Case Study.** To provide a more intuitive understanding of the
capabilities of LMMs in dealing with high-resolution images, we provide
qualitative results for LLaVA-UHD and LLaVA-1.5 in
Fig. <a href="#fig:case" data-reference-type="ref"
data-reference="fig:case">[fig:case]</a>. We can see that LLaVA-UHD can
correctly identify the dense content in the timetable (Case 1), the text
on the small poster (Case 2), and icons and text on the phone (Case 3)
for fine-grained recognition and reasoning. In comparison, LLaVA-1.5 can
only perceive coarse-grained information, and therefore tends to provide
either uninformative (Cases 1 and 2) or incorrect/hallucinated answers
(Case 3) in these challenging scenarios. The results demonstrate the
effectiveness and advantage of LLaVA-UHD in perceiving native aspect
ratio and high-resolution images for fine-grained multimodal
capabilities.

# Related Work

**Visual Encoding in LMMs.** The advent of
ChatGPT [ChatGPT2022](None) and
GPT-4 [achiam2023gpt4](None) has spurred the development of
numerous open-source large language models
(LLMs) [chiang2023vicuna](None), [touvron2023llama](None), [Chung2022Flan5](http://arxiv.org/pdf/2202.03371v1).
Utilizing an LLM as a language encoder and decoder, there springs up
plenty of
LMMs [li2023blip2](None), [instructblip2023](None), [Alayrac2023Flamingo](http://arxiv.org/pdf/2205.07065v1), [liu2024llava](http://arxiv.org/pdf/2402.11690v1), [bai2023qwen](None), [hong2023cogagent](None),
with aim at understanding visual image. Therefore, how to encode vision
features into LLMs becomes the core problem in the community.
Fortunately, CLIP [radford2021clip](http://arxiv.org/pdf/2404.19696v1) proposes to
respectively extract language embeddings using language models like
BERT [Devlin2018BERT](None) and visual features using vision
models like ViT [dosovitskiy2020vit](http://arxiv.org/pdf/2105.15075v2) and
CNN [he2016resnet](http://arxiv.org/pdf/1608.05895v1), and align them in contrastive
learning fashion using considerable image-text
pairs [schuhmann2022laion](None), so that visual embeddings
are well aligned towards the language.

Existing visual projection approaches towards LLMs can be divided into
three categories. (1) Flamingo [Alayrac2023Flamingo](http://arxiv.org/pdf/2205.07065v1)
proposes perceiver resampler, which utilizes a fixed number of queries
to capture visual features by cross-attention operation and feeds them
into LLMs for image/video understanding. (2)
BLIP-2 [li2023blip2](None) pretrains Q-Former to bridge the
image encoder and LLMs. (3) LLaVA [liu2024llava](http://arxiv.org/pdf/2402.11690v1) just
leverages an MLP module to connect language and vision feature space.
Beyond them, SPHINX [SPHINX2023](None) mixes many kinds of
visual features, including DINO-V2 [oquab2023dinov2](None),
CLIP-ViT [radford2021clip](http://arxiv.org/pdf/2404.19696v1) and
CLIP-CNN [radford2021clip](http://arxiv.org/pdf/2404.19696v1), and Q-Former to augment
visual representation. Vary [wei2023vary](http://arxiv.org/pdf/2312.06109v1) pretrains a
visual model tailored for document/chart recognition and understanding,
and integrates it with visual features of
LLaVA [liu2024llava](http://arxiv.org/pdf/2402.11690v1) for further feature enhancement.

However, since these LMMs rely on CLIP-ViT that requires fixed
resolution image as input, it hinders LMMs from handling images with
higher resolution or any aspect ratio, and undermines fine-grained
downstream tasks like optical character recognition or small object
understanding.

**High-resolution LMMs.** To perceive images with higher resolutions,
recent work can be divided into four categories. (1) Up-Resize.
Qwen-VL [bai2023qwen](None) interpolates the positional
embedding of ViT from 224$\times$`<!-- -->`{=html}224 to
448$\times$`<!-- -->`{=html}448 and additionally executes a training
stage to fine-tune the ViT. CogAgent [hong2023cogagent](None)
and LLaVA-HR [luo2024feast](http://arxiv.org/pdf/2403.03003v1) marries a large
low-resolution encoder with a small high-resolution image.
MiniGPT-v2 [chen2023minigptv2](None) only resizes the
positional embeddings without fine-tuning the visual encoder during
instruction tuning. These methods dramatically change the original
visual position encoding of CLIP-ViT [radford2021clip](http://arxiv.org/pdf/2404.19696v1),
which can cause sub-optimal visual representation. (2) Fix+Crop. To
address the above issue, SPHINX [SPHINX2023](None) utilizes a
fixed window size (224$\times$`<!-- -->`{=html}224) to crop a padded
image (448$\times$`<!-- -->`{=html}448) into four slices, and
concatenates them with a down-sampled 224$\times$`<!-- -->`{=html}224
image as visual inputs. Monkey [li2023monkey](http://arxiv.org/pdf/2103.15488v1) follows
this idea yet increases the accessible image size to
896$\times$`<!-- -->`{=html}1344, and converts each slice using a shared
resampler. (3) Fix+Enumerated-Crop.
UReader [ye2023ureader](None),
LLaVA-1.6 [2023llava1.6](https://llava-vl.github.io/blog/2024-01-30-llava-next/) and
infiMM-HD [liu2024infimm](None) enumerate a similar aspect
ratio to resize, rather than using a fixed square ratio (e.g.,
2$\times$`<!-- -->`{=html}2 as in SPHINX [SPHINX2023](None)).
The unavoidable image resizing and padding operation might cause image
deformation and waste of computation, respectively. (4) Any.
Fuyu-8B [fuyu2023](adept.ai/blog/fuyu-8b) and
Otter-HD [li2023otterhd](None) directly utilize LLMs to encode
visual features instead of vision transformers. They just split images
into patches and project them using linear layers before feeding into
the LLM. Regarding image patches as a sequence enables itself to process
images with continuous resolution. However, the removal of an image
encoder means insufficient visual representation, which makes these
methods limited in unsatisfactory performance.

In comparison, LLaVA-UHD supports images in any aspect ratios and high
resolutions. By integrating the advantages of modularized and adaptive
image encoding, as well as perceiver resampler, LLaVA-UHD can achieve
strong performance with improved computation efficiency.

# Conclusion

In this work, we present LLaVA-UHD, a large multimodal model that
efficiently perceives any aspect ratio and high-resolution images.
Comprehensive experimental results on 9 popular benchmarks demonstrate
the effectiveness of LLaVA-UHD, especially in fine-grained multimodal
capabilities. Analytical evaluation results are provided for deeper
understanding of the model. In this work, we limit the resolution of
LLaVA-UHD to maximum $672\times1008$. In future, considering the
promising efficiency and scalability, we will explore higher-resolution
images and more challenging tasks such as small object detection and
segmentation. Besides, image slices are currently independently encoded,
with interactions only in LLMs. We plan to establish efficient
connections between image slices via improved visual encoding strategies
for fine-grained global information interaction.

# Detailed Illustration on GPT-4V Phases [sec:GPT-4V-illustration]

From the pilot experimental results in
Fig. <a href="#fig:gpt4v_exp2_appendix" data-reference-type="ref"
data-reference="fig:gpt4v_exp2_appendix">[fig:gpt4v_exp2_appendix]</a>,
we observe that the GPT-4V responses show a significant phase change
with image resolutions. Here we provide detailed illustrations of the
hypothesized cause from the perspective of visual encoding:

\(1\) In phase 1, since there is only one image slice, most answers are
correct. More specifically, when dealing with input images under 512
resolution, if the images are resized to 512, the behavior will be the
same within phase 1. However, since the behavior changes significantly
within phase 1, we suspect that the input images are most likely to be
padded into 512 resolutions, as shown in
Fig. <a href="#fig:illustration" data-reference-type="ref"
data-reference="fig:illustration">[fig:illustration]</a>(a).

\(2\) In phase 2, answer 12 dominates the responses possibly due to the
incomplete circles in each slice, as shown in
Fig. <a href="#fig:illustration" data-reference-type="ref"
data-reference="fig:illustration">[fig:illustration]</a>(b).

\(3\) Phase 3 shows mixed answers of 9, 12 and 16. Among these
responses, answer 16 can be well explained by the slice strategy in
Fig. <a href="#fig:illustration" data-reference-type="ref"
data-reference="fig:illustration">[fig:illustration]</a>(c). Besides, we
also notice that many abnormal phenomenons in
Fig. <a href="#fig:gpt4v_exp2" data-reference-type="ref"
data-reference="fig:gpt4v_exp2">[fig:gpt4v_exp2]</a>(b) cannot be
perfectly explained yet, which we leave for future work.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.11703v1_md/figures/gpt4_exp2.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.11703v1_md/figures/illustration_of_gpt.png" style="width:85.0%" />

</div>

# Proofs [sec:proofs]

In this section, we provide proofs for the image partition strategy. We
show that the slice resolution exhibits modest changes to the original
resolution of ViT.

**Range of Slice Aspect Ratios.** The aspect ratio of the slice can be
represented by: $$\frac{W_v}{H_v} = \frac{W_I}{m} : \frac{H_I}{n},$$
where $W_v$, $H_v$ are the width and height of the slice, $W_I$, $H_I$
are the sizes of the original image, and (m, n) is the best partition.
Restricting the aspect ratio $r = \frac{W_v}{H_v} \in [\frac{1}{2} , 2]$
is equivalent to
$\left|\log(\text{r})\right| \leq \left| \log 2 \right|$, which is also
equivalent to
$\left| \log\left(\frac{W_I}{H_I}\right) - \log(\frac{n}{m}) \right| \leq \left| \log(2) \right|$.
We need to prove:
$$\forall \frac{W_{I}}{H_{I}} \in [\frac{1}{6}, 6], N \leq 20$$
$$\exists (\mbox{m, n}) \in \bar{\mathbb{C}}, \left| \log\left(\frac{W_{I}}{H_{I}}\right) - \log(\frac{n}{m}) \right| \leq |\log(2)|,$$
which is equivalent to
$$\forall N \leq 20, (n_{i}, m_{i}) \in \bar{\mathbb{C}}$$
$$\exists (n_{j}, m_{j}) \in \bar{\mathbb{C}}, \left| \left(\log\left(\frac{n_{i}}{m_{i}}\right) - \log\left(\frac{n_{j}}{m_{j}}\right) \right) \right| \leq 2 \cdot \left|\log(2)\right|,$$
which can be verified by enumerating all possible factorizations of
$\bar{\mathbb{C}} =  \mathbb{C}_{N-1} \cup \mathbb{C}_{N} \cup \mathbb{C}_{N+1}$
for $N \leq 20$. The results show that the aspect ratio of each slice
resides within $[\frac{1}{2}, 2]$.

**Expected Aspect Ratio.** We assume that the ratio of the original
image is greater than 1 (i.e., $H_I > W_I$). The situation is the same
for $H_I < W_I$. Assuming that the sizes of the images are uniformly
distributed for $N\in [0, 20]$, while the aspect ratio of the original
images $\frac{W_I}{H_I} \in [1, 6]$, we have
$P(W_I,W_H,n,m) = \frac{1}{20} \cdot \frac{1}{5}$. The expected aspect
ratio can be obtained by:

$$\small
{\textrm{E}}(\frac{m \times W_I}{n \times H_I}) = \iint_{{\begin{array}{c}
    \frac{W_I}{H_I} \in [1, 6] \\
    W_I \cdot H_I \in [0, 20s] \\
    n,m = \mathop{\mathrm{arg\,max}}S(\cdot) 
  \end{array}}} (\frac{m \times W_I}{n \times H_I}) \cdot P(W_I,H_I,n,m) \ dW_I dH_I,$$
where $s$ is the area of a standard resolution of ViT. After
calculation, we obtain ${\textrm{E}}(r) = 1.258$,
${\textrm{Var}}(r) = 0.048$. The results show that the expected aspect
ratio of the slices is 1:1.258, which is close to the standard
pertaining setting of ViT. More commonly assuming that images are
uniformly distributed between $[1, 3]$, and the aspect ratio is
uniformly distributed between $[1, 2]$, we have
${\textrm{E}}(r) = 1.147$, ${\textrm{Var}}(r) = 0.011$, indicating even
smaller changes.

**Range of Slice Area.** Let
$n = \frac{W_I}{W_v} \times \frac{H_I}{H_v}$, which leads to
$N= \lceil n \rceil$. We consider dividing the image into
$\{N-1, N, N+1\}$ slices. Therefore, the maximum value of each slice
$\text{S}_\text{max} = \frac{n}{N-1}$ (when $N \neq 2$), and
$\text{S}_\text{max} = \frac{n}{N}$ (when $N = 2$). The minimum value
$\text{S}_\text{min} = \frac{n}{N+1}$. As $n$ approaches $3^-$, where
$N = 3$, $\text{S}_\text{max}$ achieves the maximum value of $1.5$.
Similarly, as $n$ approaches $1^+$, where $N = 2$, $\text{S}_\text{min}$
achieves the minimum value of $0.33$.

**Expected Slice Area.** Still assuming that the sizes of the images are
uniformly distributed within $N \in [0, 20]$, while the aspect ratio of
the images $\frac{W_{I}}{H_{I}} \in [\frac{1}{6}, 6]$. The expected area
of slice can be obtained by:
$${\textrm{E}}(\frac{W_I \times H_I}{n \times m}) = \iint_{{\begin{array}{c}
    \frac{W_I}{H_I} \in [1, 6] \\
    W_I \cdot H_I \in [0, 20s] \\
    n,m = \mathop{\mathrm{arg\,max}}S(\cdot) 
  \end{array}}} (\frac{W_I \times H_I}{n \times m}) \cdot P(W_I,H_I,n,m) d W_I d H_I.$$
After calculation, we obtain
${\textrm{E}}(\frac{W_I \times H_I}{n \times m})= 1.057$,
${\textrm{Var}}(\frac{W_I \times H_I}{n \times m})= 0.016$. This shows
that our slice areas are relatively concentrated, similar to the
original resolution of ViT.

# Discussions

We provide discussions on limitations and potential negative impact of
this work.

**Limitations and Future Work.** (1) Higher resolutions. In this work,
we limit the resolution of LLaVA-UHD to maximum $672\times1008$.
Although this resolution increases the standard LLaVA-1.5 resolution by
6 times, higher-resolution images such as 4K images and remote sensing
images are still out of reach. In future, considering the promising
efficiency and scalability, we will explore higher-resolution images and
more challenging tasks such as small object detection and segmentation.
(2) Joint slice encoding. Currently image slices are currently
independently encoded, with interactions only in LLMs. We plan to
establish efficient connections between image slices via improved visual
encoding strategies for fine-grained global information interaction.

**Potential Negative Impact.** In this work, we investigate the failure
pattern and the underlying cause for GPT-4V and LLaVA-1.5. The mechanism
can be potentially used for adversarial attacks on these models. It is
worth noting that the goal of this work is to raise attention to the
vulnerability of LMMs and provide a deeper understanding of the
importance of visual encoding strategies. This work calls for further
efforts to mitigate the revealed issues to ensure the robustness and
safety of LMMs.

[^1]: Corresponding Authors

[^2]:  <https://platform.openai.com/docs/guides/vision>

[^3]: Note that the issue is different from the overlapping sliding
    windows in CNNs, since the overlaps in GPT-4V is inconsistent across
    different resolution images.

[^4]: Note that besides visual encoding strategies, model behaviors are
    also influenced by the accumulated training dynamics and RLHF.
    Therefore the double/quadruple effect does not dominate the results.
    All results are from GPT-4V on 03-05-2024.