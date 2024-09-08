<figure id="fig:qualitative_gen_0">
<embed src="/papers/text_rich/arXiv-2405.02246v1_md/images/qualitative_gen_0.png" style="width:100.0%" />
<figcaption>Idefics2-chatty analyzes the table to compute and answer the
query.</figcaption>
</figure>

# Introduction

Vision-language models (VLMs) that take images and texts as inputs and
output texts, are useful for many tasks, like retrieving information in
a scanned PDF [mPLUG-DocOwl-1.5](https://arxiv.org/pdf/2403.12895), explaining charts or
diagrams [Chart-PaLI](https://arxiv.org/pdf/2403.12596), transcribing the text in an image
[Nougat](https://arxiv.org/pdf/2308.13418), counting objects in a picture
[VQAv2](https://doi.org/10.1109/CVPR.2017.670) or turning screenshots of webpages into code
[WebSight](https://arxiv.org/pdf/2403.09029). The development of powerful open large
language models [Llama2](https://arxiv.org/pdf/2307.09288), [Mistral7B](https://arxiv.org/pdf/2310.06825), [Gemma](https://arxiv.org/pdf/2403.08295) and image
encoders [SigLIP](https://arxiv.org/pdf/2303.15343), [EVA-CLIP](https://arxiv.org/pdf/2303.15389), [CLIP](http://arxiv.org/pdf/2404.19696v1) enables researchers to
build upon these unimodal pre-trained models to create advanced VLMs
that solve these problems with increasing accuracy
[InstructBLIP](https://openreview.net/forum?id=vvoWPYqZJA), [LLaVA](https://openreview.net/forum?id=w0H2xGHlkw), [Qwen-VL](https://arxiv.org/pdf/2308.12966), [VILA](https://arxiv.org/pdf/2312.07533), [SPHINX](https://arxiv.org/pdf/2311.07575), [Monkey](https://arxiv.org/pdf/2311.06607), [CogVLM](https://arxiv.org/pdf/2311.03079).
Despite the progress in the field, the literature reveals many disparate
design choices which are often not justified experimentally, or very
briefly.

This situation makes it challenging to distinguish which decisions truly
account for model performance, thereby making it difficult for the
community to make meaningful and grounded progress. For instance,
[Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf), [OBELICS](https://openreview.net/forum?id=SKN2hflBIZ) use interleaved Transformer-based
cross-attentions to fuse the image information into the language model,
while [BLIP-2](http://arxiv.org/pdf/2301.12597v3), [LLaVA](https://openreview.net/forum?id=w0H2xGHlkw) concatenate the sequence of image
hidden states with the sequence of text embeddings, and feed the
concatenated sequence to the language model. To our knowledge, this
choice has not been properly ablated, and trade-offs in terms of
compute, data efficiency and performance are poorly understood. In this
work, we aim to bring experimental clarity to some of these core design
choices and pose the question: **What matters when building
vision-language models?**

We identify two areas where various works adopt different design
choices: (a) model architecture, and in particular, connector modules
that fuse the vision and text modalities and their impact on inference
efficiency, (b) multimodal training procedure and its impact on training
stability. For each of these areas, we rigorously compare different
design choices in a controlled environment and extract experimental
findings. Notably, we find that (a) the progress of vision-language
models is in large part driven by the progress of pre-trained unimodal
backbones, (b) the more recent fully autoregressive architecture
outperforms the cross-attention architecture, although it requires
modifications to the optimization procedure to ensure a stable training,
(c) adaptation of the pre-trained vision backbone and the modules
connecting the text and vision modalities allow for more efficiency at
inference time on one side, and handling images in their original ratio
and size without harming downstream performance on the other side, and
(d) modifications to the image processing enables trading inference cost
for downstream performance.

Our results are complementary with those presented in
[prismatic](https://arxiv.org/pdf/2402.07865), [MM1](https://arxiv.org/pdf/2403.09611), [VILA](https://arxiv.org/pdf/2312.07533) which derive insights about
multi-stage training, selective unfreezing of the pre-trained backbones,
data repetition, and impact of training mixture on zero and few-shot
performance. We specifically delve into unexplored aspects such as model
architecture, training methods, stability, and efficiency improvements
at inference.

Learning from these insights, we train Idefics2, a foundational VLM with
8 billion parameters. Idefics2 achieves state-of-the-art performance in
its size category on various benchmarks while being more efficient at
inference, for both the base and the fine-tuned version. It is on par
with state-of-the-art models 4 times larger on some vision-language
benchmarks and matches the performance of Gemini 1.5 Pro on some
challenging benchmarks. We release the base, instructed, and chat
versions of Idefics2[^1] as resources for the VLM community along with
the data created to train the model.

[^1]: <https://huggingface.co/collections/HuggingFaceM4/idefics2-661d1971b7c50831dd3ce0fe>



# Terminology [section:terminology]

We first establish shared terminology for discussing the different
design choices. Training VLMs typically requires gluing together a
pre-trained vision backbone and a pre-trained language backbone by
initializing new parameters to connect the two modalities. Training
these new parameters is done during the *pre-training phase*. This stage
commonly leverages a large multimodal dataset such as image-caption
pairs. We note that even though it is most common to start from two
separate unimodal pre-trained backbones, the parameters of these two
backbones can be optionally shared and initialized from scratch as done
in [fuyu](https://www.adept.ai/blog/fuyu-8b). As in the large language models literature,
the pre-training stage is followed by an instruction fine-tuning stage,
in which the model learns from task-oriented samples.

Recent works explore two main choices to combine the visual inputs and
the text inputs. In the *cross-attention architecture*
[Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf), [OBELICS](https://openreview.net/forum?id=SKN2hflBIZ), [OpenFlamingo](https://arxiv.org/pdf/2308.01390), the images encoded
through the vision backbone are injected at different layers within the
language model by interleaving cross-attention blocks in which the text
cross-attends to the image hidden states. In contrast, in the *fully
autoregressive architecture* [FROMAGe](http://arxiv.org/pdf/2301.13823v4), [PaLM-E](http://arxiv.org/pdf/2302.14030v3), [LLaVA](https://openreview.net/forum?id=w0H2xGHlkw),
the output of the vision encoder is directly concatenated to the
sequence of text embeddings, and the entire sequence is passed as input
to the language model. The input sequence of the language model is thus
the concatenation of *visual tokens* and text tokens. The sequence of
visual tokens can be optionally pooled into a shorter sequence,
providing more compute efficiency. We refer to the layers that maps the
vision hidden space to the text hidden space as *modality projection*
layers.
Figure <a href="#fig:architecture_idefics2" data-reference-type="ref"
data-reference="fig:architecture_idefics2">1</a> highlights the
fully-autoregressive architecture we ultimately use for Idefics2.

<figure id="fig:architecture_idefics2">
<embed src="/papers/text_rich/arXiv-2405.02246v1_md/images/architecture_idefics2.png" style="width:90.0%" />
<figcaption>Idefics2 fully-autoregressive architecture: Input images are
processed by the Vision encoder. The resulting visual features are
mapped (and optionally pooled) to the <span
class="math inline"><em>L</em><em>L</em><em>M</em></span> input space to
get the visual tokens (<span class="math inline">64</span> in our
standard configuration). They are concatenated (and potentially
interleaved) with the input sequence of text embeddings (green and red
column). The concatenated sequence is fed to the language model (<span
class="math inline"><em>L</em><em>L</em><em>M</em></span>), which
predicts the text tokens output.</figcaption>
</figure>

# Exploring the design space of vision-language models

In this section, we compare recurrent design choices in the
vision-language model literature and highlight findings. Unless
specified otherwise, we run the ablations for 6’000 steps and report the
average score of the 4-shot performance on 4 downstream benchmarks
measuring different capabilities: VQAv2 [VQAv2](https://doi.org/10.1109/CVPR.2017.670) for
general visual question answering, TextVQA [textvqa](http://arxiv.org/pdf/1811.11903v1) for
OCR abilities, OKVQA [okvqa](http://arxiv.org/pdf/1906.00067v2) for external knowledge, and
COCO [coco](http://arxiv.org/pdf/2012.01295v1) for captioning.

## Are all pre-trained backbones equivalent for VLMs?

Most recent VLMs start from pre-trained unimodal backbones. How does the
choice of the backbones (vision and text) influence the performance of
the resulting VLM?

<div class="wraptable" markdown="1">

r4.4cm

</div>

We fix the size of the pretrained backbones, the data used for
multimodal pre-training, and the number of training updates. Under the
cross-attention architecture, we observe that the greatest improvement
in the performance on vision-language benchmarks comes from changing the
language model to a better one. More specifically, replacing LLaMA-1-7B
[LLaMA](https://arxiv.org/pdf/2302.13971) (35.1% on MMLU [MMLU](https://openreview.net/forum?id=d7KBjmI3GmQ)) by
Mistral-7B [Mistral7B](https://arxiv.org/pdf/2310.06825) (60.1% on MMLU) yields a boost of
5.1 (see Table
<a href="#tab:ablations_archi_lm_backbone" data-reference-type="ref"
data-reference="tab:ablations_archi_lm_backbone">[tab:ablations_archi_lm_backbone]</a>).
Additionally, switching the vision encoder from CLIP-ViT-H
[CLIP](http://arxiv.org/pdf/2404.19696v1) (78.0% on ImageNet[ImageNet](https://doi.org/10.1109/CVPR.2009.5206848)) to
SigLIP-SO400M [SigLIP](https://arxiv.org/pdf/2303.15343) (83.2% on ImageNet) yields a 3.3
increase in performance on the benchmarks (see Table
<a href="#tab:ablations_archi_vision_encode_backbone"
data-reference-type="ref"
data-reference="tab:ablations_archi_vision_encode_backbone">[tab:ablations_archi_vision_encode_backbone]</a>).
This result on better vision backbones corroborates observations from
[prismatic](https://arxiv.org/pdf/2402.07865).

<div class="wraptable" markdown="1">

r5cm

</div>

We note that [PaLI-17B](http://arxiv.org/pdf/2402.18932v1) reports a stronger increase in
performance by scaling the size of the vision encoder compared to
scaling the size of the language model even though scaling the vision
encoder leads to a smaller parameter count increase. Although
EVA-CLIP-5B [EVA-CLIP](https://arxiv.org/pdf/2303.15389) is ten times bigger in parameter
counts than SigLIP-SO400M [SigLIP](https://arxiv.org/pdf/2303.15343), we obtain similar
performance across 4 benchmarks, suggesting that EVA-CLIP-5B could be
heavily under-trained, and we acknowledge that the open VLM community is
missing a large well-trained vision encoder.

<div class="tcolorbox" markdown="1">

#### ***Finding* 1.**

For a fixed number of parameters, the quality of the language model
backbone has a higher impact on the performance of the final VLM than
the quality of the vision backbone.

</div>

## How does the fully autoregressive architecture compare to the cross-attention architecture?

To our knowledge, there is no proper comparison between the fully
autoregressive and the cross-attention architecture. We aim to fill this
gap by considering their trade-offs, namely performance, parameter
count, and inference cost.

<div class="wraptable" markdown="1">

r7.1cm

</div>

Following [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf), we first compare the two
architectures by freezing the unimodal backbones and training only the
newly initialized parameters (cross-attention on one side, and modality
projection along with learned pooling on the other side), while fixing
the amount of training data. [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) shows that the
more frequently the cross-attention blocks are interleaved with the
language model layers, and the higher the vision-language performance.
As such, we note that under this setup, the cross-attention architecture
has 1.3B more trainable parameters (2B trainable parameters in total)
than the fully autoregressive architecture. Additionally, at inference
time, the former uses 10% more flops than the latter. Under these
conditions, we observe that the cross-attention architecture performs 7
points better in Table
<a href="#tab:ablations_archi_type_archi_method_training"
data-reference-type="ref"
data-reference="tab:ablations_archi_type_archi_method_training">[tab:ablations_archi_type_archi_method_training]</a>.

Out of the total number of parameters, approximately 15% for the fully
autoregressive architecture and 25% for the cross-attention are trained.
We hypothesize that this low proportion limits the expressivity of the
training and hinders performance. To test that hypothesis, we compare
the two architectures by unfreezing all parameters (newly initialized
parameters and parameters of the pre-trained unimodal backbones). Under
these conditions, training the fully autoregressive architecture would
yield loss divergences, and we were not successful in stabilizing the
training even by aggressively lowering the learning rate or gradually
unfreezing various components. To overcome this stability challenge, we
leverage Low-Rank Adaptation [LoRA](https://openreview.net/forum?id=nZeVKeeFYf9) to adapt the
pre-trained parameters while using standard full fine-tuning for the
newly initialized ones.

This setup yields significantly more stable trainings, and more
importantly, we observe a 12.9 points increase under the fully
autoregressive architecture, and 0.6 point under the cross-attention
architecture. While the cross-attention architecture performs better
than the fully autoregressive architecture with frozen backbones, it is
worse when we add degrees of liberty for the pre-trained backbones.
Besides, using LoRA allows training the unimodal backbones at a fraction
of the GPU memory cost of full fine-tuning, and LoRA layers can be
merged back into the original linear layers yielding no additional cost
at inference. We therefore choose the fully autoregressive architecture
in the rest of this work.

It is interesting to note that this finding contradicts
[prismatic](https://arxiv.org/pdf/2402.07865) in which the authors observed that
unfreezing the pre-trained visual backbone would significantly degrade
the performance. We hypothesize that using parameter-efficient
fine-tuning methods is a key difference.

<div class="tcolorbox" markdown="1">

#### ***Finding* 2.**

The cross-attention architecture performs better than the fully
autoregressive one when unimodal pre-trained backbones are kept frozen.
However, when training the unimodal backbones, the fully autoregressive
architecture outperforms the cross-attention one, even though the latter
has more parameters.

</div>

<div class="tcolorbox" markdown="1">

#### ***Finding* 3.**

Unfreezing the pre-trained backbones under the fully autoregressive
architecture can lead to training divergences. Leveraging LoRA still
adds expressivity to the training and stabilizes it.

</div>

## Where are the efficiency gains?

#### Number of visual tokens

Recent VLMs typically route the entire sequence of the vision encoder’s
hidden states directly into the modality projection layer, which
subsequently inputs into the language model, without no pooling. This is
motivated by previous works in which adding a pooling strategy, like
average pooling, was found to deteriorate the performance
[DePALM](https://arxiv.org/pdf/2403.13499). This results in a high number of visual tokens
for each image ranging from 576 for DeepSeek-VL
[DeepSeek-VL](https://arxiv.org/pdf/2403.05525) to 2890 for SPHINX-2k
[SPHINX](https://arxiv.org/pdf/2311.07575). With the resulting sequence lengths, training
is computationally costly, and in-context learning with interleaved
images and texts is challenging because it requires modifications to the
language models to handle very large context windows.

We reduce the sequence length of each image’s hidden states by using a
perceiver resampler [perceiver](https://proceedings.mlr.press/v139/jaegle21a.html), [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf), [Qwen-VL](https://arxiv.org/pdf/2308.12966) as a
form of trainable Transformer-based pooling. The number of queries (also
referred to as latents) corresponds to the number of resulting visual
tokens after the pooling. We observe that the learned pooling is
effective in two ways: it increases the performance by 8.5 points on
average and reduces the number of visual tokens necessary for each image
from 729 to 64 (see Table
<a href="#tab:ablations_archi_type_archi_method_training"
data-reference-type="ref"
data-reference="tab:ablations_archi_type_archi_method_training">[tab:ablations_archi_type_archi_method_training]</a>).

<div class="wraptable" markdown="1">

r4.6cm

</div>

In contrast to [DePALM](https://arxiv.org/pdf/2403.13499), [MM1](https://arxiv.org/pdf/2403.09611) which find that the more
visual tokens the higher the performance, we observe no gains when using
more than 64 visual tokens. We hypothesize that in a hypothetical
scenario of infinite training on unlimited data, performance might
eventually improve, at the cost of a longer training time. Other
variations over the Perceiver architecture
[MAPL](https://doi.org/10.18653/v1/2023.eacl-main.185), [register-tokens](https://openreview.net/forum?id=2dnO3LLiJ1), [DePALM](https://arxiv.org/pdf/2403.13499) resulted in decreased
performance.

<div class="tcolorbox" markdown="1">

#### ***Finding* 4.**

Reducing the number of visual tokens with learned pooling significantly
improves compute efficiency at training and inference while improving
performance on downstream tasks.

</div>

#### Preserving the original aspect ratio and image resolution

Vision encoders, such as SigLIP, are typically trained on fixed-size
square images. Resizing images alters their original aspect ratio, which
is problematic, for instance, for tasks requiring reading long texts.
Furthermore, conditioning the training on a single resolution size
inherently introduces limitations: a low resolution omits crucial visual
details, while a high resolution leads to inefficiency in training and
inference. Allowing the model to encode images at various resolutions
allows users to decide how much compute is spent on each image.

<div class="wraptable" markdown="1">

r5.2cm

</div>

Following [pix2struct](http://arxiv.org/pdf/2210.03347v2), [PatchNPack](https://openreview.net/forum?id=VpGFHmI7e5), we pass the image
patches to the vision encoder without resizing the image or modifying
its aspect ratio. Given that SigLIP was trained on fixed-size
low-resolution square images, we interpolate the pre-trained positional
embeddings to allow for a higher resolution and train the vision encoder
with LoRA parameters to adapt to these modifications.[^1] Our findings
indicate that the aspect ratio preserving strategy maintains performance
levels on downstream tasks while unlocking computational flexibility
during both training and inference (see Table
<a href="#tab:ablations_archi_aspect_ratio_preserving"
data-reference-type="ref"
data-reference="tab:ablations_archi_aspect_ratio_preserving">[tab:ablations_archi_aspect_ratio_preserving]</a>).
In particular, not having to resize images to the same high resolution
allows for saving GPU memory and handling images at the resolution they
require.

<div class="tcolorbox" markdown="1">

#### ***Finding* 5.**

Adapting a vision encoder pre-trained on fixed-size square images to
preserve images’ original aspect ratio and resolution does not degrade
performance while speeding up training and inference and reducing
memory.

</div>

## How can one trade compute for performance?

[SPHINX](https://arxiv.org/pdf/2311.07575), [Monkey](https://arxiv.org/pdf/2311.06607), [LLAVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [MM1](https://arxiv.org/pdf/2403.09611) show that splitting an
image into sub-images allows boosting the downstream performance with no
changes to the model’s signature. An image is decomposed into sub-images
(for instance 4 equal sub-images), which are then concatenated to the
original image to form a sequence of 5 images. Additionally, the
sub-images are resized to the original image’s size. This strategy
however comes at the cost of a much higher number of tokens to encode
the images.

We adopt this strategy during the instruction fine-tuning stage. Each
single image becomes a list of 5 images: 4 crops and the original image.
This way, at inference, the model is able to deal with standalone images
(64 visual tokens per image), as well as artificially augmented images
(320 visual tokens in total per image). We notice that this strategy is
particularly useful for benchmarks like TextVQA and DocVQA, which
require a sufficiently high resolution to extract the text in an image
(see Table <a href="#table:perf_sft" data-reference-type="ref"
data-reference="table:perf_sft">[table:perf_sft]</a>).

Moreover, when we apply image spitting to only 50% of the training
samples (instead of 100% of the samples), we observe that it does not
impair the performance increase that image splitting provides.
Surprisingly, we find at evaluation time that increasing the resolution
of the sub-images (and the standalone image) provides only a minor boost
in performance compared to the improvement yielded by sole image
splitting: 73.6% when increasing the resolution of the sub-images to the
maximum vs 73.0% accuracy on our validation set of TextVQA, and
respectively 72.7 vs 72.9 ANLS on the validation set of DocVQA.

<div class="tcolorbox" markdown="1">

#### ***Finding* 6.**

Splitting images into sub-images during training allow trading compute
efficiency for more performance during inference. The increase in
performance is particularly noticeable in tasks involving reading text
in an image.

</div>

[^1]: Since SigLIP is trained with a fixed resolution, the positional
    embeddings can be interpreted both as absolute or relative
    positions. With the aspect ratio and resolution preserving, these
    positions become relative positional embeddings.

# Idefics2 - an open state-of-the-art vision-language foundation model

With these learnings in hand, we train an open 8B parameters
vision-language model: Idefics2. This section describes the construction
of the model, the choice of the dataset, the sequence of training phases
and compares the resulting model against VLMs baselines.

## Multi-stage pre-training

We start from SigLIP-SO400M and Mistral-7B-v0.1 and pre-train Idefics2
on 3 types of data.

**Interleaved image-text documents** We use OBELICS
[OBELICS](https://openreview.net/forum?id=SKN2hflBIZ), an open web-scale dataset of interleaved
image-text documents with 350 million images and 115 billion text
tokens. As shown by the authors, the long documents of OBELICS allow for
preserving the performance of the language model while learning to deal
with an arbitrary number of interleaved images and texts and long
context. Additionally, the authors show that interleaved image-text
documents are the biggest driving factor in increasing the performance
on visual question answering (VQA) tasks, in particular in the
in-context learning setup. We perform an additional removal of newly
opted-out content in January 2024 using the Spawning API[^1] even though
OBELICS had already been filtered to exclude opted-out content as of
September 2023. We also removed the 5% of documents with the highest
perplexity scores, as computed by Falcon-1B
[RefinedWeb](https://openreview.net/forum?id=kM5eGcdCzq).

<div class="wraptable" markdown="1">

r3.5cm

</div>

**Image-text pairs** Training on image-text pairs allows the model to
learn the alignment between images and their associated texts. We use a
combination of high-quality human-annotated image-text pairs from PMD
[flava](https://doi.org/10.1109/CVPR52688.2022.01519) and higher-noise web-scale image-text pairs from
[LAION-5B](https://proceedings.neurips.cc/paper_files/paper/2022/file/a1859debfb3b59d094f3504d5ebb6c25-Paper-Datasets_and_Benchmarks.pdf). To limit the amount of poor-quality data, we
opt for the synthetic captions obtained through the LAION COCO[^2]
version of the dataset where images have been captioned with a model
trained on COCO. This improves the quality of the training samples and
thus the quality of the resulting model (see Table
<a href="#tab:ablations_pretraining_type_captions"
data-reference-type="ref"
data-reference="tab:ablations_pretraining_type_captions">[tab:ablations_pretraining_type_captions]</a>).
We use a NSFW classifier[^3] with a high recall and remove 7% of the
samples in LAION COCO. We manually inspect 5’000 examples and found 28
pornographic images in the original LAION COCO and only 1 after
filtering. This filtering does not negatively impact the downstream
performance.

<div class="wraptable" markdown="1">

r5cm

</div>

**PDF documents** [multimodal-rlhf](https://arxiv.org/pdf/2309.14525) shows that a large
proportion of mistakes of state-of-the art VLMs stem from their failure
to accurately extract text in images or documents. In order to obtain
strong OCR and document understanding abilities, we train Idefics2 on
different sources of PDF documents: 19 million industry documents from
OCR-IDL [OCRIDL](https://arxiv.org/pdf/2202.12985) and 18 million pages from PDFA[^4].
Moreover, we add Rendered Text[^5] to complement the dataset with texts
written with a wide variety of fonts and colors and on diverse
backgrounds. These integrations significantly boost the performance on
benchmarks that require reading text without decreasing the performance
on other benchmarks (see Table
<a href="#tab:ablations_finetuning_ocr" data-reference-type="ref"
data-reference="tab:ablations_finetuning_ocr">[tab:ablations_finetuning_ocr]</a>).

To maximize compute efficiency, we decompose the pre-training in two
stages. In the first stage, we limit the max image resolution to 384
pixels, which allows us to use a large global batch size of 2’048 (17k
images and 2.5M text tokens on average). We sample OBELICS for 70% of
the examples with a maximum sequence length of 2’048, and the image-text
pairs datasets for 30% of the examples with a maximum sequence length of
1’536. In the second stage, we introduce PDF documents. Since they
require a higher image resolution for the text to be legible, we
increase the resolution to a maximum of 980 pixels. We use the same
global batch size, but have to decrease the per-device batch size and
use gradient accumulation to compensate for the additional memory cost.
OBELICS represents 45% of the examples with a maximum sequence length of
2’048, image-text pairs represent 35% of the examples with a maximum
sequence length of 1’536, and PDF documents represent the remaining 20%
of the examples with a maximum sequence length of 1’024. Additionally,
we randomly scale up images to adequately cover the distribution of
potential image sizes. We emphasize that the training stages are
different than the ones ablated in [prismatic](https://arxiv.org/pdf/2402.07865): instead
of selectively freezing/unfreezing parts of the model, we train the
entire model during both stages (some parameters are trained with LoRA)
and increase the image resolution from one stage to the other.

We use a learning rate of $10^{-4}$ and do around 2 epochs on our
training data. It corresponds to approximately 1.5 billion images and
225 billion text tokens. We note that this is orders of magnitude more
training data than other open VLMs. For example, ShareGPT
[ShareGPT4V](https://arxiv.org/pdf/2311.12793) uses 1.2 million images, while Monkey
[Monkey](https://arxiv.org/pdf/2311.06607) uses 1.4 million for training.

<div id="table:perf_base" markdown="1">

|  |  |  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Model** | **Size** | **Archi.** |  |  |  |  |  |
| per image | **VQAv2** | **TextVQA** | **OKVQA** | **COCO** |  |  |  |
| OpenFlamingo | 9B | CA | \- | 54.8 | 29.1 | 41.1 | 96.3 |
| Idefics1 | 9B | CA | \- | 56.4 | 27.5 | 47.7 | 97.0 |
| Flamingo | 9B | CA | \- | 58.0 | 33.6 | 50.0 | 99.0 |
| MM1 | 7B | FA | 144 | 63.6 | 46.3 | 51.4 | **116.3** |
| Idefics2-base | 8B | FA | **64** | **70.3** | **57.9** | **54.6** | 116.0 |

Performance of Idefics2-base against state-of-the-art base VLMs. The
evaluations were done with 8 random in-context examples, and in an
open-ended setting for VQA tasks.  
*FA: fully autoregressive architecture. CA: cross-attention
architecture.*  
*(Task, Metric, Split): (VQAv2, VQA acc., testdev), (TextVQA, VQA acc.,
val), (OKVQA, VQA acc., val), (COCO, CIDEr, test)*

</div>

To evaluate the base model, we consider VQAv2 [VQAv2](https://doi.org/10.1109/CVPR.2017.670),
TextVQA [textvqa](http://arxiv.org/pdf/1811.11903v1), OKVQA [okvqa](http://arxiv.org/pdf/1906.00067v2), and
COCO [coco](http://arxiv.org/pdf/2012.01295v1). Table
<a href="#table:perf_base" data-reference-type="ref"
data-reference="table:perf_base">1</a> presents the results. While
having fewer tokens per image, and thus being more efficient, Idefics2
performs favorably compared to the other current best base VLMs
(OpenFlamingo [OpenFlamingo](https://arxiv.org/pdf/2308.01390), Idefics1
[OBELICS](https://openreview.net/forum?id=SKN2hflBIZ), Flamingo [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf), and MM1
[MM1](https://arxiv.org/pdf/2403.09611)). It is notably much better at reading texts in an
image. Figure
<a href="#fig:text_transcription_base" data-reference-type="ref"
data-reference="fig:text_transcription_base">1</a> shows an example of
an output from the base model on a task similar to the pre-training.

<figure id="fig:text_transcription_base">
<embed src="/papers/text_rich/arXiv-2405.02246v1_md/images/text_transcription_base.png" style="width:90.0%" />
<figcaption>An example of text transcription with
Idefics2-base.</figcaption>
</figure>

[^1]: <https://spawning.ai/>

[^2]: <https://laion.ai/blog/laion-coco/>

[^3]: <https://github.com/LAION-AI/LAION-SAFETY>

[^4]: <https://huggingface.co/datasets/pixparse/pdfa-eng-wds>

[^5]: <https://huggingface.co/datasets/wendlerc/RenderedText>

## Instruction fine-tuning

We continue the training with an instruction fine-tuning phase.

To do so, we create and release The Cauldron[^1], a massive collection
of 50 vision-language datasets, covering a wide range of tasks: general
visual question answering, counting, captioning, text transcription,
document understanding, chart/figure understanding, table understanding,
visual reasoning, geometry, spotting differences between 2 images or
converting a screenshot to a functional code. Similarly to
[T0](https://openreview.net/forum?id=9Vrb9D0WI4), [flan](https://openreview.net/forum?id=gEZrGCozdqR), [promptsource](https://doi.org/10.18653/v1/2022.acl-demo.9), [InstructBLIP](https://openreview.net/forum?id=vvoWPYqZJA), [m3it](https://arxiv.org/pdf/2306.04387), each
dataset is prompted into a shared question/answer format. When there are
multiple question/answer pairs per image, we concatenate the pairs into
a multi-turn conversation. We deduplicate the training set against the
evaluation sets, ensuring that there is minimum contamination from the
training to the evaluation.

In addition to these vision-language datasets and following insights
from [MM1](https://arxiv.org/pdf/2403.09611), we add text-only instruction datasets to the
mixture. The datasets aim at teaching the model to follow complex
instructions, solve mathematical problems, or do arithmetic
calculations. We give more details about the chosen datasets, the number
of images, question-answer pairs, and size of each of the subsets, as
well as our selected mixture proportion in Table
<a href="#table:mixture_sft" data-reference-type="ref"
data-reference="table:mixture_sft">[table:mixture_sft]</a> in Appendix
<a href="#subsection:details_the_cauldron" data-reference-type="ref"
data-reference="subsection:details_the_cauldron">[subsection:details_the_cauldron]</a>.

<div id="table:perf_sft" markdown="1">

|  |  |  |  |  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Model** | **Size** |  |  |  |  |  |  |  |  |
| per image | **MMMU** | **MathVista** | **TextVQA** | **MMBench** |  |  |  |  |  |
| LLaVA-NeXT | 13B | 2880 | 36.2/- | 35.3 | 67.1 | 70.0 |  |  |  |
| DeepSeek-VL | 7B | 576 | 36.6/- | 36.1 | 64.4 | 73.2 |  |  |  |
| MM1-Chat | 7B | 720 | 37.0/35.6 | 35.9 | 72.8 | 72.3 |  |  |  |
| Idefics2 | 8B | **64** | **43.5**/**37.9** | **51.6** | 70.4 | **76.8** |  |  |  |
| Idefics2 | 8B | 320 | 43.0/37.7 | 51.4 | **73.0** | 76.7 |  |  |  |

Performance of Idefics2 against state-of-the-art VLMs up to a size of
14B parameters. The evaluations are done in zero shot. Idefics2 with 64
or 320 tokens per image is the same model (same weights), only the
inference differs. The full table is present in Appendix
<a href="#subsection:expanded_evals" data-reference-type="ref"
data-reference="subsection:expanded_evals">[subsection:expanded_evals]</a>.  
*(Benchmark, Split, Metric): (MMMU, val/test, MMMU score), (MathVista,
testmini, MMMU score), (TextVQA, val, VQA acc.), (MMBench, test,
accuracy).*

</div>

We instruction-tune the base model using DoRA [DoRA](https://arxiv.org/pdf/2402.09353) (a
variant of LoRA). During the fine-tuning, we only compute the loss on
the tokens of the answers in the Q/A pairs. Since we are doing many
epochs over some of the datasets, we employ several strategies to lower
the risk of overfitting. First, we add noise to the embeddings with the
NEFTune [NEFTune](https://openreview.net/forum?id=0bMmZ3fkCk) technique. Then, we scale up randomly
the resolution of the images during the training. Finally, when
applicable, we shuffle the multiple user/assistant turns randomly before
feeding the example to the model.

We evaluate Idefics2 on commonly adopted benchmarks: MMMU
[MMMU](http://arxiv.org/pdf/2311.16502v3) for multidiscipline college-level problems,
MathVista [mathvista](http://arxiv.org/pdf/2310.02255v3) for mathematical reasoning,
TextVQA [textvqa](http://arxiv.org/pdf/1811.11903v1) for text reading on natural images,
and MMBench [MMBench](https://arxiv.org/pdf/2307.06281) for various perception and
reasoning tasks. Table
<a href="#table:perf_sft" data-reference-type="ref"
data-reference="table:perf_sft">1</a> presents the results (see Table
<a href="#table:perf_sft_full" data-reference-type="ref"
data-reference="table:perf_sft_full">[table:perf_sft_full]</a> for the
complete result table) of Idefics2 against the current strongest VLMs in
its class size: LLaVA-Next [LLAVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/), DeepSeek-VL
[DeepSeek-VL](https://arxiv.org/pdf/2403.05525) and MM1-Chat [MM1](https://arxiv.org/pdf/2403.09611). While
being computationally much more efficient at inference, Idefics2
exhibits strong performance on various benchmarks, outperforming the
current best foundation VLMs in its size category. It is on par with
state-of-the-art models 4x its size, or with closed-source models like
Gemini 1.5 Pro on several benchmarks like MathVista or TextVQA.

## Optimizing for chat scenarios

The evaluation benchmarks expect very short answers, but humans prefer
long generations when interacting with a model. We find that Idefics2
can exhibit difficulties in precisely following instructions about the
expected format, making it difficult to reconcile “chattiness“ and
downstream performance. As such, after instruction fine-tuning, we
further train Idefics2 on dialogue data. We fine-tune Idefics2 for a few
hundred steps on LLaVA-Conv [LLaVA](https://openreview.net/forum?id=w0H2xGHlkw) and ShareGPT4V
[ShareGPT4V](https://arxiv.org/pdf/2311.12793), with a large batch size. Our blind human
evaluations reveal that Idefics2-chatty is overwhelmingly preferred over
its instruction fine-tuned version in many user interactions. We also
adversarially stress-tested the model to generate inaccurate, biased, or
offensive responses and reported the findings in
Appendix <a href="#sec:red_teaming" data-reference-type="ref"
data-reference="sec:red_teaming">[sec:red_teaming]</a>. We show examples
of generations with Idefics2-chatty in Figure
<a href="#fig:qualitative_gen_0" data-reference-type="ref"
data-reference="fig:qualitative_gen_0">[fig:qualitative_gen_0]</a>, and
in Appendix in Figures
<a href="#fig:qualitative_gen_1" data-reference-type="ref"
data-reference="fig:qualitative_gen_1">[fig:qualitative_gen_1]</a>,
<a href="#fig:qualitative_gen_2" data-reference-type="ref"
data-reference="fig:qualitative_gen_2">[fig:qualitative_gen_2]</a> and
<a href="#fig:qualitative_gen_3" data-reference-type="ref"
data-reference="fig:qualitative_gen_3">[fig:qualitative_gen_3]</a>.

[^1]: <https://huggingface.co/datasets/HuggingFaceM4/the_cauldron>

# Conclusion

In this work, we re-examine common choices made in the VLM literature
and rigorously compare these choices in controlled experiments. Our
findings touch upon the effectiveness of different architectures, their
performance/inference cost trade-offs as well as training stability.
With these learnings at hand, we train Idefics2, an open 8B parameters
vision-language model. Idefics2 is state-of-the-art on various
benchmarks in its category size and is much more efficient at inference.
By releasing our findings, as well as our models and our training
dataset, we aim to contribute to the ongoing evolution of VLMs and their
applications in solving complex real-world problems.

# Acknowledgement [acknowledgement]

We thank Mustafa Shukor for helpful suggestions on the paper, and Yacine
Jernite, Sasha Luccioni, Margaret Mitchell, Giada Pistilli, Lucie-Aimée
Kaffee, and Jack Kumar for red-teaming the model.



# Appendix

## Further experimental details of the ablations

### Cross-attention vs. fully autoregressive architectures

We apply LoRA modules to the LLM for the fully autoregressive
architecture and to the cross-attention modules and the LLM for the
cross-attention architecture. In
Figure <a href="#fig:autoregressive_lora_vs_flamingo_lora"
data-reference-type="ref"
data-reference="fig:autoregressive_lora_vs_flamingo_lora">1</a>, we
report the average performance with respect to the number of steps, the
number of images, as well as the number of text tokens. We see an
improvement across the board with the fully autoregressive architecture.
Comparing the average score with these different axes is essential
because the cross-attention architecture feeds a single token per image
to the language model, against 64 for the fully autoregressive
architecture with perceiver pooling. This implies that for the same
training sequence length, the number of images and text tokens is
different for the two architectures. Equivalently, the same multimodal
document will yield different sequence lengths. Even though we fix the
batch size in the comparison, the number of text tokens and number of
images grow at different paces under the two architectures.

<figure id="fig:autoregressive_lora_vs_flamingo_lora">
<img src="/papers/text_rich/arXiv-2405.02246v1_md/images/autoregressive_lora_vs_flamingo_lora.png"
style="width:100.0%" />
<figcaption>Comparison of the cross-attention and fully autoregressive
architectures through the number of steps, the number of images and the
number of text tokens.</figcaption>
</figure>

### Comparing various vision backbones

We present in
Table <a href="#tab:ablations_archi_vision_encode_backbone_detailed"
data-reference-type="ref"
data-reference="tab:ablations_archi_vision_encode_backbone_detailed">[tab:ablations_archi_vision_encode_backbone_detailed]</a>
the detailed results of comparing multiple vision backbones. While
EVA-CLIP-5B performs similarly to SigLIP-SO400M, we emphasize that it
has 11 times more parameters. We also noticed in early experiments that
TextVQA is the most sensitive benchmark to image resolution, which
accounts for the performance increase.

### Comparing various pooling strategies

We compare multiple pooling strategies: a simple linear layer that takes
the flattened sequence of vision hidden states and projects it into a
shorter sequence of visual tokens, as well as a Mapping Network
[MAPL](https://doi.org/10.18653/v1/2023.eacl-main.185). The perceiver resampler significantly
outperforms these two options (see
Table <a href="#tab:vision_language_adaptor_ablation"
data-reference-type="ref"
data-reference="tab:vision_language_adaptor_ablation">[tab:vision_language_adaptor_ablation]</a>).

We also ablate the number of layers in the perceiver resampler, and find
no statistically significant differences when increasing the number of
layers, similarly to results from [palm2vadapter](https://arxiv.org/pdf/2402.10896). We
settle on 3 layers out of caution to avoid any potential capacity
bottleneck.

Finally, we add a 2-layer modality projection MLP on top of the vision
encoder hidden states to project the vision hidden dimension to the
language model hidden dimension prior to the perceiver resampler. These
changes yield better performance as well (see
Table <a href="#tab:modality_projection_prior_to_perceiver"
data-reference-type="ref"
data-reference="tab:modality_projection_prior_to_perceiver">[tab:modality_projection_prior_to_perceiver]</a>).

### Ablations on OCR data

We hypothesize that adding PDF documents helps the model learn to read
text from images. In
Table <a href="#tab:ablations_finetuning_ocr" data-reference-type="ref"
data-reference="tab:ablations_finetuning_ocr">[tab:ablations_finetuning_ocr]</a>,
we compare checkpoints trained with and without OCR documents, along
with image resolution increase to ensure that the text is legible. We do
not observe statistically significant differences when evaluating
checkpoints in zero or few shot. Instead, we fine-tune the checkpoints
on DocVQA for 500 steps with a learning rate of $1e-5$, leading to
checkpoints showing much stronger differences.

## Details of the instruction fine-tuning

### Statistics of The Cauldron [subsection:details_the_cauldron]

In Table <a href="#table:mixture_sft" data-reference-type="ref"
data-reference="table:mixture_sft">1</a>, we present the statistics of
the datasets included in The Cauldron, as well as the text-only
instruction datasets used for the supervised fine-tuning. For each
dataset, we give the number of different images it contains, the number
of question-answer pairs, the total number of tokens for the answers in
the question-answer pairs, and the selected percentage of tokens it
represents in our final mixture after upsampling or downsampling.

<div id="table:mixture_sft" markdown="1">

| **Dataset** |  | **\# Q/A pairs** | **\# tokens** | **% mixture** |
|:---|:--:|:--:|:--:|:--:|
|  |  |  |  |  |
| *General visual question answering* |  |  |  |  |
| VQAv2 [VQAv2](https://doi.org/10.1109/CVPR.2017.670) | 82,772 | 443,757 | 1,595,929 | 5.72% |
| COCO-QA [CocoQA](https://proceedings.neurips.cc/paper_files/paper/2015/file/831c2f88a604a07ca94314b56a4921b8-Paper.pdf) | 46,287 | 78,736 | 286,982 | 1.47% |
| Visual7W [Visual7w](None) | 14,366 | 69,817 | 279,268 | 1.43% |
| A-OKVQA [A-OKVQA](https://doi.org/10.1007/978-3-031-20074-8_9) | 16,539 | 17,056 | 236,492 | 1.21% |
| TallyQA [TallyQA](http://arxiv.org/pdf/1810.12440v2) | 98,680 | 183,986 | 738,254 | 0.57% |
| OK-VQA [okvqa](http://arxiv.org/pdf/1906.00067v2) | 8,998 | 9,009 | 38,853 | 0.40% |
| HatefulMemes [hatefulmeme](https://proceedings.neurips.cc/paper_files/paper/2020/file/1b84c4cee2b8b3d823b30e2d604b1878-Paper.pdf) | 8,500 | 8,500 | 25,500 | 0.13% |
| VQA-RAD [VQA-RAD](https://doi.org/10.1038/sdata.2018.251) | 313 | 1,793 | 8,418 | 0.09% |
|  |  |  |  |  |
| *Captioning* |  |  |  |  |
| LNarratives [LocalizedNarratives](http://arxiv.org/pdf/2302.11217v2) | 507,444 | 507,444 | 21,328,731 | 4.56% |
| Screen2Words [screen2words](https://doi.org/10.1145/3472749.3474765) | 15,730 | 15,743 | 143,103 | 0.37% |
| VSR [VSR](https://doi.org/10.1162/tacl_a_00566) | 2,157 | 3,354 | 10,062 | 0.21% |
|  |  |  |  |  |
| *OCR, document understanding, text transcription* |  |  |  |  |
| RenderedText[^1] | 999,000 | 999,000 | 27,207,774 | 5.57% |
| DocVQA [DocVQA](https://doi.org/10.1109/WACV48630.2021.00225) | 10,189 | 39,463 | 337,829 | 3.46% |
| TextCaps [textcaps](http://arxiv.org/pdf/2003.12462v2) | 21,953 | 21,953 | 389,658 | 2.00% |
| TextVQA [textvqa](http://arxiv.org/pdf/1811.11903v1) | 21,953 | 34,602 | 181,918 | 1.86% |
| ST-VQA [STVQA](https://doi.org/10.1109/ICCV.2019.00439) | 17,247 | 23,121 | 127,846 | 1.31% |
| OCR-VQA [OCR-VQA](https://doi.org/10.1109/ICDAR.2019.00156) | 165,746 | 801,579 | 6,073,824 | 0.93% |
| VisualMRC [VisualMRC](http://arxiv.org/pdf/2101.11272v2) | 3,027 | 11,988 | 168,828 | 0.86% |
| IAM [IAM](https://doi.org/10.1007/s100320200071) | 5,663 | 5,663 | 144,216 | 0.74% |
| InfoVQA [InfographicVQA](https://doi.org/10.1109/WACV51458.2022.00264) | 2,118 | 10,074 | 61,048 | 0.63% |
| Diagram image-to-text[^2] | 300 | 300 | 22,196 | 0.11% |
|  |  |  |  |  |
| *Chart/figure understanding* |  |  |  |  |
| Chart2Text [Chart2Text](https://doi.org/10.18653/v1/2020.inlg-1.20) | 26,985 | 30,242 | 2,852,827 | 4.38% |
| DVQA [DVQA](http://arxiv.org/pdf/1810.02358v2) | 200,000 | 2,325,316 | 8,346,234 | 4.27% |
| VisText [VisText](http://vis.csail.mit.edu/pubs/vistext) | 7,057 | 9,969 | 1,245,485 | 1.91% |
| ChartQA [ChartQA](https://doi.org/10.18653/v1/2022.findings-acl.177) | 18,271 | 28,299 | 185,835 | 1.90% |
| PlotQA [PlotQA](http://arxiv.org/pdf/1906.04124v2) | 157,070 | 20,249,479 | 8478299.278 | 0.65% |
| FigureQA [FigureQA](https://arxiv.org/pdf/1710.07300) | 100,000 | 1,327,368 | 3,982,104 | 0.61% |
| MapQA [MapQA](https://openreview.net/forum?id=znKbVjeR0yI) | 37,417 | 483,416 | 6,470,485 | 0.33% |
|  |  |  |  |  |
| *Table understanding* |  |  |  |  |
| TabMWP [TabMWP](http://arxiv.org/pdf/2209.14610v3) | 22,729 | 23,059 | 1,948,166 | 2.49% |
| TAT-QA [TAT-QA](https://doi.org/10.18653/v1/2021.acl-long.254) | 2,199 | 13,215 | 283,776 | 2.18% |
| HiTab [Hitab](https://doi.org/10.18653/v1/2022.acl-long.78) | 2,500 | 7,782 | 351,299 | 1.80% |
| MultiHiertt [Multihiertt](https://aclanthology.org/2022.acl-long.454) | 7,619 | 7,830 | 267,615 | 1.37% |
| FinQA [FinQA](https://doi.org/10.18653/v1/2021.emnlp-main.300) | 5,276 | 6,251 | 242,561 | 0.99% |
| WikiSQL [WikiSQL](https://arxiv.org/pdf/1709.00103) | 74,989 | 86,202 | 9,680,673 | 0.99% |
| SQA [SQA](https://doi.org/10.18653/v1/P17-1167) | 8,514 | 34,141 | 1,894,824 | 0.97% |
| WTQ [WTQ](https://doi.org/10.3115/v1/P15-1142) | 38,246 | 44,096 | 6,677,013 | 0.51% |
|  |  |  |  |  |
| *Reasoning, logic, maths* |  |  |  |  |
| GeomVerse [GeomVerse](https://openreview.net/forum?id=A9NOAS0hn1) | 9,303 | 9,339 | 2,489,459 | 3.83% |
| CLEVR-Math [CLEVR-Math](https://doi.org/10.48550/ARXIV.2208.05358) | 70,000 | 788,650 | 3,184,656 | 3.26% |
| CLEVR [CLEVR](https://doi.org/10.1109/CVPR.2017.215) | 70,000 | 699,989 | 2,396,781 | 1.23% |
| IconQA [IconQA](http://arxiv.org/pdf/2110.13214v4) | 27,315 | 29,859 | 112,969 | 1.16% |
| RAVEN [RAVEN](http://arxiv.org/pdf/2207.00590v1) | 42,000 | 42,000 | 105,081 | 0.67% |
| Inter-GPs [Inter-GPS](http://arxiv.org/pdf/2105.04165v3) | 1,451 | 2,101 | 8,404 | 0.17% |
|  |  |  |  |  |
| *Textbook/academic questions* |  |  |  |  |
| AI2D [AI2D](http://arxiv.org/pdf/1603.07396v1) | 3,099 | 9,708 | 38,832 | 0.80% |
| TQA [TQA](https://doi.org/10.1109/CVPR.2017.571) | 1,496 | 6,501 | 26,004 | 0.53% |
| ScienceQA [ScienceQA](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf) | 4,985 | 6,218 | 24,872 | 0.25% |
|  |  |  |  |  |
| *Differences between 2 images* |  |  |  |  |
| NLVR2 [NLVR2](https://doi.org/10.18653/v1/P19-1644) | 50,426 | 86,373 | 259,119 | 1.33% |
| GSD [MIMIC-IT-General-Scene-Difference](https://arxiv.org/pdf/2306.05425) | 70,939 | 141,869 | 4,637,229 | 0.48% |
| Spot the diff [SpotTheDiff](https://doi.org/10.18653/v1/D18-1436) | 8,566 | 9,524 | 221,477 | 0.57% |
|  |  |  |  |  |
| *Screenshot to code* |  |  |  |  |
| WebSight [WebSight](https://arxiv.org/pdf/2403.09029) | 500,000 | 500,000 | 276,743,299 | 0.28% |
| DaTikz [DaTikz](https://arxiv.org/pdf/2310.00367) | 47,974 | 48,296 | 59,556,252 | 0.03% |
|  |  |  |  |  |
|  |  |  |  |  |
| *Text-only general instructions, math problems, arithmetic calculations* |  |  |  |  |
| OpenHermes-2.5 [OpenHermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 0 | 1,006,223 | 248,553,747 | 12.73% |
| LIMA [LIMA](https://openreview.net/forum?id=KBMOKmX2he) | 0 | 1,052 | 633,867 | 0.81% |
| Dolly [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) | 0 | 14,972 | 1,329,999 | 0.68% |
| MetaMathQA [MetaMathQA](https://openreview.net/forum?id=N8N0hgNDRt) | 0 | 395,000 | 74,328,255 | 3.81% |
| MathInstruct [MathInstruct](https://openreview.net/forum?id=yLClGs770I) | 0 | 261,781 | 45,393,559 | 2.33% |
| OrcaMath [Orca-Math](https://arxiv.org/pdf/2402.14830) | 0 | 200,031 | 63,780,702 | 1.63% |
| CamelAIMath [CamelAIMath](https://openreview.net/forum?id=3IyL2XWDkG) | 0 | 49,744 | 21,873,629 | 0.06% |
| AtlasMathSets[^3] | 0 | 17,807,579 | 455,411,624 | 3.50% |
| Goat [Goat](https://arxiv.org/pdf/2305.14201) | 0 | 1,746,300 | 167,695,693 | 0.86% |
|  |  |  |  |  |
|  |  |  |  |  |

The statistics of datasets used for instruction fine-tuning. \# tokens
is the total number of tokens for each dataset for the answers only. %
mixture is our selected percentage of answer tokens for each dataset in
the final mixture.

</div>

## Details of the evaluations

### Evaluation setup

We perform all evaluations with a batch size of 1 and greedy decoding.

For the multi-choice questions in MMMU, MathVista, MMBench, we evaluate
with the same prompt used for similar types of datasets during the
instruction fine-tuning:

<div class="tcolorbox" markdown="1">

Question: {question} Choices: A. {choice_a} B. {choice_b} C. {choice_c}
... Answer with the letter.

</div>

For the open-ended questions in TextVQA, DocVQA, and VQAv2, we evaluate
with the prompt:

<div class="tcolorbox" markdown="1">

Question: {question} Give a very brief answer.

</div>

We use the stop words `Question`, `User`, `<end_of_utterance>` and
`<eos>` to stop a generation.

### Expanded evaluation table [subsection:expanded_evals]

We report the expanded evaluation of Idefics2 and the comparison to
other models in Table
<a href="#table:perf_sft_full" data-reference-type="ref"
data-reference="table:perf_sft_full">2</a>. This includes scores on
VQAv2 [VQAv2](https://doi.org/10.1109/CVPR.2017.670), which is widely adopted for evaluation.
We acknowledge, though, that the metric used for the open-ended visual
question answering benchmarks strongly penalizes models that do not
generate in the same format as the ground truth. For example, answering
"large" when the ground truth is "big" or more verbose reformulations
will be counted as incorrect. Our manual qualitative analysis reveals
that on benchmarks like VQAv2, the generations of two models differing
by 5 points would be barely noticeable. This problem is less concerning
for other open-ended benchmarks like TextVQA or DocVQA which require
finding a text in an image, making the expected answer less prone to
ambiguity.

<div id="table:perf_sft_full" markdown="1">

|  |  |  |  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Model** | **Size** |  |  |  |  |  |  |  |
| per image |  |  |  |  |  |  |  |  |
| *7B-14B models* |  |  |  |  |  |  |  |  |
| LLaVA-NeXT | 13B | 2880 | 36.2/- | 35.3 | 67.1 | 70.0 | \- | 82.8 |
| DeepSeek-VL | 7B | 576 | 36.6/- | 36.1 | 64.4 | 73.2 | 49.6 | \- |
| MM1-Chat | 7B | 720 | 37.0/35.6 | 35.9 | 72.8 | 72.3 | \- | 82.8 |
| Idefics2 | 8B | 64 | 43.5/37.9 | 51.6 | 70.4 | 76.8 | 67.3 | 80.8 |
| Idefics2 | 8B | 320 | 43.0/37.7 | 51.4 | 73.0 | 76.7 | 74.0 | 81.2 |
| *$\geq$`<!-- -->`{=html}30B models* |  |  |  |  |  |  |  |  |
| Mini-Gemini-HD | 34B | 2880 | 48.0/44.9 | 43.3 | 74.1 | 80.6 | \- | \- |
| MM1-Chat | 30B | 720 | 44.7/40.3 | 39.4 | 73.5 | 75.1 | \- | 83.7 |
| LLaVA-NeXT | 34B | 2880 | 51.1/44.7 | 46.5 | 69.5 | 79.3 | \- | 83.7 |
| *Proprietary* |  |  |  |  |  |  |  |  |
| Gemini 1.0 Pro | \- | \- | 47.9/- | 45.2 | 74.6 | \- | 88.1 | 71.2 |
| Claude 3 Haiku | \- | \- | 50.2/- | 46.4 | \- | \- | 88.8 | \- |
| Claude 3 Sonnet | \- | \- | 53.1/- | 47.9 | \- | \- | 89.5 | \- |
| Gemini 1.5 Pro | \- | \- | 58.5/- | 52.1 | 73.5 | \- | 86.5 | 73.2 |

Performance of Idefics2 against state-of-the-art VLMs across different
sizes. The evaluations are done in zero shot. Idefics2 with 64 or 320
tokens per image only differs by the image splitting.  
*(Benchmark, Split, Metric): (MMMU, val/test, MMMU score), (MathVista,
testmini/test, MMMU score), (TextVQA, val, VQA acc.), (MMBench, test,
accuracy), (DocVQA, test, ANLS score), (VQAv2, testdev, VQA acc.).*

</div>

### Qualitative evaluation

We show in Figures
<a href="#fig:qualitative_gen_1" data-reference-type="ref"
data-reference="fig:qualitative_gen_1">2</a>,
<a href="#fig:qualitative_gen_2" data-reference-type="ref"
data-reference="fig:qualitative_gen_2">3</a>, and
<a href="#fig:qualitative_gen_3" data-reference-type="ref"
data-reference="fig:qualitative_gen_3">4</a>, examples of generations
with Idefics2-chatty.

<figure id="fig:qualitative_gen_1">
<embed src="/papers/text_rich/arXiv-2405.02246v1_md/images/qualitative_gen_1.png" style="width:100.0%" />
<figcaption>Idefics2-chatty finds the requested information in the
resume, and organizes it in JSON format.</figcaption>
</figure>

<figure id="fig:qualitative_gen_2">
<embed src="/papers/text_rich/arXiv-2405.02246v1_md/images/qualitative_gen_2.png" style="width:70.0%" />
<figcaption>Idefics2-chatty describes an AI-generated
image.</figcaption>
</figure>

<figure id="fig:qualitative_gen_3">
<embed src="/papers/text_rich/arXiv-2405.02246v1_md/images/qualitative_gen_3.png" style="width:70.0%" />
<figcaption>Idefics2-chatty answers a question on a scientific
diagram.</figcaption>
</figure>

## Red-teaming [sec:red_teaming]

In the context of a red-teaming exercise, our objective is to evaluate
the propensity of the model to generate inaccurate, biased, or offensive
responses. We evaluate more specifically the chat-optimized
checkpoint[^4].

While the model typically refrains from responding to offensive inputs,
we observe that through repeated trials or guided interactions, it tends
to hastily form judgments in situations necessitating nuanced contextual
understanding, often perpetuating harmful stereotypes. Noteworthy
instances include:

-   Speculating or passing judgments, or perpetuating historical
    disparities on individuals’ professions, social status, or insurance
    eligibility based solely on visual cues (e.g., age, attire, gender,
    facial expressions).

-   Generating content that promotes online harassment or offensive
    memes reinforcing harmful associations from a portrait, or from a
    benign image.

-   Assuming emotional states or mental conditions based on outward
    appearances.

-   Evaluating individuals’ attractiveness solely based on their visual
    appearance.

Additionally, we identify behaviors that increase security risks that
already exist:

-   Successfully solving CAPTCHAs featuring distorted text within
    images.

-   Developing phishing schemes from screenshots of legitimate websites
    to deceive users into divulging their credentials.

-   Crafting step-by-step guides on constructing small-scale explosives
    using readily available chemicals from common supermarkets or
    manipulating firearms to do maximum damage.

It’s important to note that these security concerns are currently
limited by the model’s occasional inability to accurately read text
within images.

We emphasize that the model would often encourage the user to exercise
caution about the model’s generation or flag how problematic the initial
query can be in the first place. For instance, when insistently prompted
to write a racist comment, the model would answer that query before
pointing out "*This type of stereotyping and dehumanization has been
used throughout history to justify discrimination and oppression against
people of color. By making light of such a serious issue, this meme
perpetuates harmful stereotypes and contributes to the ongoing struggle
for racial equality and social justice.*".

However, certain formulations can circumvent (i.e. "jailbreak") these
cautionary prompts, emphasizing the need for critical thinking and
discretion when engaging with the model’s outputs. While jail-breaking
text LLMs is an active research area, jail-breaking vision-language
models have recently emerged as a new challenge as vision-language
models become more capable and prominent [jailbreak](https://openreview.net/forum?id=plmBsXHxgR).
The addition of the vision modality not only introduces new avenues for
injecting malicious prompts but also raises questions about the
interaction between vision and language vulnerabilities.

[^1]: <https://huggingface.co/datasets/wendlerc/RenderedText>

[^2]: <https://huggingface.co/datasets/Kamizuru00/diagram_image_to_text>

[^3]: <https://huggingface.co/datasets/AtlasUnified/atlas-math-sets>

[^4]: <https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty>