# Introduction [sec:intro]

<figure id="fig:teaser">
<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/figures/radar1.png" style="width:100.0%" />
</figure>

In recent years, the progress in Large Language Models
(LLMs) [openai2020chatgpt](https://openai.com/blog/chatgpt), [touvron2023llama](http://arxiv.org/pdf/2402.08075v1), [touvron2023llama2](https://arxiv.org/pdf/2307.09288), [jiang2023mistral](https://arxiv.org/pdf/2310.06825), [2023internlm](https://github.com/InternLM/InternLM), [cai2024internlm2](http://arxiv.org/pdf/2403.17297v1), [qwen7b](http://arxiv.org/pdf/2305.05352v6), [du2022glm](http://arxiv.org/pdf/2103.10360v2), [vicuna2023](https://lmsys.org/blog/2023-03-30-vicuna/)
has provoked the development of Large Vision-Language Models (LVLMs).
These models have demonstrated proficiency in tasks such as image
captioning [chen2015microsoft](https://arxiv.org/pdf/1504.00325), [chen2023sharegpt4v](http://arxiv.org/pdf/1809.10312v1) and
visual-question-answering
(VQA) [MMBench](http://arxiv.org/pdf/2005.12661v2), [fu2023mme](http://arxiv.org/pdf/2306.05179v2), [seed_2023](http://arxiv.org/pdf/2307.08041v2), [yue2023mmmu](http://arxiv.org/pdf/2311.16502v3).
Nevertheless, due to their limited resolution, they struggle with
processing images containing fine details, such as
charts [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1),
tables [textvqa](http://arxiv.org/pdf/1811.11903v1), documents [docvqa](http://arxiv.org/pdf/2111.05547v1), and
infographics [infovqa](http://arxiv.org/pdf/2104.12756v2). This limitation constrains their
practical applicability in real-world scenarios.

Recent advancements have aimed at enhancing the resolution of Large
Vision-Language Models (LVLMs). Some
approaches [lv2023kosmos25](https://arxiv.org/pdf/2309.11419), [cogagent](http://arxiv.org/pdf/2402.11941v2), [wei2023vary](None), [li2024mini](None)
involve adapting high-resolution vision encoders directly. However, the
Vision Transformer (ViT) architecture falls short when dealing with
images of varying resolutions and aspect ratios, thereby restricting its
ability to handle diverse inputs effectively. Alternatively, some
methods [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [monkeytext](http://arxiv.org/pdf/2403.14252v1), [docowl](http://arxiv.org/pdf/2307.02499v1), [lin2023sphinx](http://arxiv.org/pdf/2311.07575v1), [llavauhd](http://arxiv.org/pdf/2403.11703v1), [llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [li2023otterhd](https://arxiv.org/pdf/2311.04219)
maintain the vision encoder’s resolution, segmenting high-resolution
images into multiple low-resolution patches. Yet, these methods are
constrained by an inadequate resolution, typically around 1500 $\times$
1500, which does not satisfy the demands of daily content, , website
screenshots [si2024design2code](https://arxiv.org/pdf/2403.03163), document
pages [docvqa](http://arxiv.org/pdf/2111.05547v1), and blueprints [infovqa](http://arxiv.org/pdf/2104.12756v2).
Furthermore, they are confined to either a few predefined
high-resolution
settings [cogagent](http://arxiv.org/pdf/2402.11941v2), [wei2023vary](None), [li2024mini](None), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [lin2023sphinx](http://arxiv.org/pdf/2311.07575v1), [llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [li2023otterhd](https://arxiv.org/pdf/2311.04219), [lv2023kosmos25](https://arxiv.org/pdf/2309.11419), [monkeytext](http://arxiv.org/pdf/2403.14252v1)
or a limited range of resolutions [docowl](http://arxiv.org/pdf/2307.02499v1), [llavauhd](http://arxiv.org/pdf/2403.11703v1),
thereby restricting their utility across a variety of applications.

In this work, we introduce InternLM-XComposer2-4KHD, a pioneering model
that for the first time expands the resolution capabilities of Large
Vision-Language Models (LVLMs) to 4K HD and even higher, thereby setting
a new standard in high-resolution vision-language understanding.
Designed to handle a broad range of resolutions,
InternLM-XComposer2-4KHD supports images with any aspect ratio from 336
pixels up to 4K HD, facilitating its deployment in real-world contexts.

InternLM-XComposer2-4KHD follows patch
division [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [li2023otterhd](https://arxiv.org/pdf/2311.04219) paradigm and
enhances it by incorporating an innovative extension: dynamic resolution
with automatic patch configuration. To be specific, scaling the
resolution of Large Vision-Language Models (LVLMs) to 4K HD and even
higher standard is far beyond merely increasing the number of patches.
It involves a nuanced approach to overcoming specific challenges: (1)
**Dynamic Resolution and Automatic Patch Configuration**: Addressing the
scarcity of high-resolution training data, our framework introduces a
strategy that dynamically adjusts resolution alongside an automatic
layout configuration. During training, it maintains the original aspect
ratios of images while adaptively altering patch (336 $\times$ 336)
layouts and counts. This results in a training resolution that exceeds
the original image resolutions, reaching up to 4KHD, addressing the
shortfall of high-resolution data. (2) **Handling Variability in Patch
Configurations**: Despite the apparent simplicity of dynamic resolution
training, the variability in patch configurations can heavily confuse
LVLMs. To mitigate this, we introduce a newline token after each row of
patch tokens to clearly delineate patch layouts, reducing training
ambiguity and significantly boosting performance. (3) **Inference Beyond
4K Resolution:** Our observations reveal that, even when trained on
images up to 4K resolution, the model can achieve additional performance
improvements during inference by processing images at higher
resolutions.

Furthermore, scaling the training resolution up to 4K standard results
in a consistent improvement in performance, highlighting the potential
for training even beyond 4K resolution. This underscores the capacity
for further enhancing model capabilities and suggests a promising
trajectory for advancing the frontiers of high-resolution image
processing within the domain of large vision-language models.

We evaluate our InternLM-XComposer2-4KHD on 16 diverse benchmarks
spanning various domains, including 5 challenging HD-OCR datasets
(DocVQA[docvqa](http://arxiv.org/pdf/2111.05547v1),
ChartQA[masry2022chartqa](http://arxiv.org/pdf/2203.10244v1),
InfographicVQA[infovqa](http://arxiv.org/pdf/2104.12756v2), TextVQA[textvqa](http://arxiv.org/pdf/1811.11903v1)
and OCRBench[ocrbench](https://arxiv.org/pdf/2305.07895)). Compared to previous open-source
LVLM models and closed-source APIs, our approach achieves SOTA results
in 6 of 16 benchmarks, demonstrating competitive performance despite
only 7B parameters. As shown in
Figure <a href="#fig:teaser" data-reference-type="ref"
data-reference="fig:teaser">1</a>, InternLM-XComposer2-4KHD even
surpasses the performance of GPT4V [openai2023gpt4](https://arxiv.org/pdf/2303.08774) and
Gemini Pro [geminiteam2023gemini](https://arxiv.org/pdf/2312.11805) across ten benchmarks.
Notably, our method exhibits excellent performance on 5 HD-OCR datasets,
over existing open-source LVLMs by a substantial margin.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/figures/teaser_cases.png" style="width:94.0%" />

<span id="fig:teaser1" label="fig:teaser1"></span>

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/figures/teaser_cases_2_1.png" style="width:90.0%" />

<span id="fig:teaser2" label="fig:teaser2"></span>

</div>

# Related Works [sec:related]

Large Language Models
(LLMs) [brown2020language](http://arxiv.org/pdf/2112.07522v2), [ouyang2022training](http://arxiv.org/pdf/2302.05206v1), [openai2020chatgpt](https://openai.com/blog/chatgpt), [chowdhery2022palm](http://arxiv.org/pdf/2209.05735v4), [kaplan2020scaling](http://arxiv.org/pdf/1906.09379v1), [touvron2023llama](http://arxiv.org/pdf/2402.08075v1), [touvron2023llama2](https://arxiv.org/pdf/2307.09288), [jiang2023mistral](https://arxiv.org/pdf/2310.06825), [2023internlm](https://github.com/InternLM/InternLM), [zeng2023glm-130b](https://openreview.net/forum?id=-Aw0rrrPUF), [baichuan2023baichuan2](https://arxiv.org/abs/2309.10305), [qwen7b](http://arxiv.org/pdf/2305.05352v6), [cai2024internlm2](http://arxiv.org/pdf/2403.17297v1)
have gained significant attention due to their impressive performance in
various language-related tasks such as text generation and question
answering. Following this enthusiasm, recent Large Vision-Language
Models (LVLMs) have
emerged[openai2023gpt4](https://arxiv.org/pdf/2303.08774), [chen2023pali](https://arxiv.org/pdf/2209.06794), [chen2023palix](https://arxiv.org/pdf/2305.18565), [chen2023pali3](https://arxiv.org/pdf/2310.09199), [driess2023palme](http://arxiv.org/pdf/2302.14030v3), [fu2023gemini](http://arxiv.org/pdf/2312.12436v2), [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1), [dai2023instructblip](https://arxiv.org/pdf/2305.06500), [zhang2023internlm](http://arxiv.org/pdf/2309.15112v5), [fuyu-8b](https://www.adept.ai/blog/fuyu-8b), [li2023otter](http://arxiv.org/pdf/2311.00233v2), [peng2023kosmos](http://arxiv.org/pdf/2305.16103v1), [ye2023mplug](http://arxiv.org/pdf/2405.00390v2), [awadalla2023openflamingo](http://arxiv.org/pdf/2402.17510v1),
combining LLMs with vision
encoders [radford2021learning](http://arxiv.org/pdf/2404.19696v1), [zhang2024long](None), [sun2023alpha](None)
to leverage the complementary strengths of language and vision
modalities. By fusing textual and visual representations, LVLMs can
ground language in visual contexts, enabling a more comprehensive
understanding and generation of multimodal
content [chen2023sharegpt4v](http://arxiv.org/pdf/1809.10312v1), [chen2023internvl](http://arxiv.org/pdf/2312.14238v3), [lin2023sphinx](http://arxiv.org/pdf/2311.07575v1), [bai2023qwen](http://arxiv.org/pdf/1412.3919v1), [wang2023cogvlm](https://arxiv.org/pdf/2311.03079), [internlmxcomposer2](http://arxiv.org/pdf/2402.17510v1), [cao2024dualfocus](None), [liu2024rar](None).

**LVLMs for High-Resolution Understanding.** Large Vision-Language
Models (LVLMs) often employ CLIP-ViT as the visual encoder for
vision-dependent tasks. However, the visual encoder’s reliance on low
resolutions, such as 224 $\times$ 224 or 336 $\times$ 336 pixels, limits
its effectiveness for high-resolution tasks like OCR and document/chart
perception. To enhance high-resolution understanding, recent works have
primarily employed the following strategies: (1) High-resolution (HR)
visual encoders or dual encoders catering to HR and low-resolution (LR)
inputs [lv2023kosmos25](https://arxiv.org/pdf/2309.11419), [wei2023vary](None), [cogagent](http://arxiv.org/pdf/2402.11941v2), [li2024mini](None).
For instance, Vary [wei2023vary](None) introduces a new image
encoder supporting HR inputs, which are then concatenated with LR
embeddings from the original CLIP visual encoder. Similarly,
CogAgent [cogagent](http://arxiv.org/pdf/2402.11941v2) and
Mini-Gemini [li2024mini](None) also separate HR and LR images
using distinct vision encoders, subsequently merging their features
using a cross-attention module. In contrast, our approach offers a more
simplified solution and shows advantages for varying resolutions and
aspect ratio inputs. (2) Cropped image
patches [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [monkeytext](http://arxiv.org/pdf/2403.14252v1), [llavauhd](http://arxiv.org/pdf/2403.11703v1), [ureader](http://arxiv.org/pdf/2311.13165v1), [docowl](http://arxiv.org/pdf/2307.02499v1), [lin2023sphinx](http://arxiv.org/pdf/2311.07575v1), [li2023otterhd](https://arxiv.org/pdf/2311.04219).
For example, Monkey [li2023monkey](http://arxiv.org/pdf/2103.15488v1) employs sliding
windows to segment images into patches, subsequently processing them
with LoRA fine-tuning. TextMonkey [monkeytext](http://arxiv.org/pdf/2403.14252v1) further
proposes shifted window attention and token resampler to consider the
connections among different patches. These approaches are confined to
either a few predefined high-resolution
settings [cogagent](http://arxiv.org/pdf/2402.11941v2), [wei2023vary](None), [li2024mini](None), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [lin2023sphinx](http://arxiv.org/pdf/2311.07575v1), [llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [li2023otterhd](https://arxiv.org/pdf/2311.04219), [lv2023kosmos25](https://arxiv.org/pdf/2309.11419), [monkeytext](http://arxiv.org/pdf/2403.14252v1)
or a limited range of resolutions [docowl](http://arxiv.org/pdf/2307.02499v1), [llavauhd](http://arxiv.org/pdf/2403.11703v1).
Conversely, our method devises a dynamic image partition strategy to
support the scaling from 336 pixels to 4K resolution, and the maximum
resolution is larger than previous approaches (, 1.5k for
Monkey [li2023monkey](http://arxiv.org/pdf/2103.15488v1) and 2k for
UReader [ureader](http://arxiv.org/pdf/2311.13165v1)).

**LVLMs for Document Understanding.** Document understanding involves
analyzing and comprehending various digital documents, such as figures,
tables, and academic papers. Many document understanding tasks require
models to handle high-resolution inputs, complex layouts, various aspect
ratios, and diverse document formats. To enhance the capabilities of
LVLMs for document understanding, several works have collected and
constructed high-quality document instruction tuning data, including
LLaVAR [zhang2023llavar](None),
mPLUG-DocOwl [ye2023mplug-doc](None) and
TGDoc [wang2023towards](http://arxiv.org/pdf/2311.13194v2).
DocPediaDocPedia [feng2023docpedia](None) processes document
inputs in the frequency domain. Some previous works have improved
document understanding ability by designing special modules for
high-resolution inputs, such as HR and LR
encoders [cogagent](http://arxiv.org/pdf/2402.11941v2), [wei2023vary](None) or cropped image
patches [ureader](http://arxiv.org/pdf/2311.13165v1), [monkeytext](http://arxiv.org/pdf/2403.14252v1), [llavauhd](http://arxiv.org/pdf/2403.11703v1). Our
InternLM-XComposer2-4KHD first scales to 4K resolution inputs and
demonstrates strong document understanding ability on OCR-related
benchmarks. Also, our approach also achieves comparable results on other
general LVLM benchmarks like perception and
reasoning [lu2024mathvista](http://arxiv.org/pdf/2310.02255v3), [MMBench](http://arxiv.org/pdf/2005.12661v2), [seed_2023](http://arxiv.org/pdf/2307.08041v2), [mmstar](http://arxiv.org/pdf/2006.11910v3).

# Method

<div class="table*" markdown="1">

</div>

## Model Architecture.

The model architecture of InternLM-XComposer2-4KHD mainly follows the
design of InternLM-XComposer2[internlmxcomposer2](http://arxiv.org/pdf/2402.17510v1)
(XComposer2 in the following for simplicity.), including a light-weight
Vision Encoder OpenAI ViT-Large/14, Large Language Model InternLM2-7B,
and Partial LoRA for efficient alignment. We recommend the readers to
the XComposer2 paper for more details.

<figure id="fig:framework">
<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/figures/partition1.png" />
<figcaption><strong>The illustration of processing high-resolution
input.</strong> </figcaption>
</figure>

## High-Resolution Input.

**Dynamic Image Partition.** Utilizing a static input image size for
processing high-resolution images, particularly those with varying
aspect ratios, is neither efficient nor effective. To overcome this
limitation, we introduce a dynamic image partitioning approach, as shown
in Figure <a href="#fig:framework" data-reference-type="ref"
data-reference="fig:framework">1</a>. Our method strategically segments
the image into smaller patches, while maintaining the integrity of the
original image’s aspect ratio.

Given a maximum partition number $\mathcal{H}$, the image $x$ with size
$[h,w]$ is resized and padded to the new image $\hat{x}$ with size
$[p_h \times 336, p_w \times 336 ]$. This process is subject to the
following constraints:
$$p_w \times p_h \leq \mathcal{H}; \;   p_h = \lceil p_w \times h / w \rceil$$
here $p_w$ and $p_h$ represent the number of patches in each row and
column, respectively. We then split the $\hat{x}$ into $p_h \times p_w$
non-overlapped patches. Each patch is a small image with $336\times336$
size and we treat these patches as individual inputs for the ViT.

In the following, we use ‘HD-$\mathcal{H}$’ to represent our
high-resolution setting with the constraint of $\mathcal{H}$ patches.
For example, the ’HD-9’ allows up to 9 patches, including a range of
resolutions such as $1008\times1008$, $672\times1344$, $336\times3024$,
.

**Global-Local Format.** For each input image, we present it to the
model with two views. The first is the global view, where the image is
resized to a fixed size (in our case, 336 × 336). This provides a macro
understanding of the image. Empirically, we have found this to be
crucial for the LVLM to correctly understand the image. The second view
is the local view. We divide the image into patches using the previously
mentioned Dynamic Image Partition strategy and extract features from
each patch. Following feature extraction, the patches are reassembled
into a large feature map. The feature map is then flattened to the final
local features after a straightforward token merging process.

**Image 2D Structure Newline Indicator.** Given that an image has a 2D
structure and the image ratio is dynamic, the number of tokens for each
row can vary across different images. This variation can potentially
confuse the LVLM, making it difficult to determine which tokens belong
to the same row of the image and which ones belong to the next row. This
confusion may hinder the LVLM’s ability to understand the 2D structure
of the image, which is crucial for comprehending structural image
content such as documents, charts, and tables. To address this issue, we
introduce a learnable newline (‘$\backslash$n’) token at the end of each
row of the image features before the flattening. Finally, we concatenate
the global and local views, inserting a special ‘separate’ token between
them to distinguish the two views.

## Pre-Training

During the pre-training phase, the LLM is frozen while both the vision
encoder and Partial LoRA are fine-tuned to align the visual tokens with
the LLM. The pre-training data mainly follow the design in XComposer2
which is curated with **three objectives** in mind: 1) general semantic
alignment, 2) world knowledge alignment, 3) vision capability
enhancement. In this paper, we focus on high-resolution and structural
image understanding. Therefore, we have collected more related data to
enhance this specific capability. As shown in
Table.<a href="#tab:pretrain_data" data-reference-type="ref"
data-reference="tab:pretrain_data">[tab:pretrain_data]</a>, we have
utilized a diverse OCR dataset for this purpose.

In practice, we employ the OpenAI CLIP ViT-L-14-336 as the vision
encoder. Different from XComposer2, We keep the ViT resolution as
$336\times336$ and increase the input resolution with more patches. For
the Dynamic Image Partition strategy, we use ‘HD-25’ for the pertaining.
For each image or patch, the image token number is decreased to $1/4$
with a simple **merge operation**. We concatenate the nearby 4 tokens
into a new token through the channel dimension, then align it with the
LLM by an MLP. The ‘separate’ and ‘$\backslash$n’ token are randomly
initialized. For the Partial LoRA, we set a rank of $256$ for all the
linear layers in the LLM decoder block. Our training process involves a
batch size of 4096 and spans across 2 epochs. The learning rate linearly
increases to $2 \times 10^{-4}$ within the first $1\%$ of the training
steps. Following this, it decreases to $0$ according to a cosine decay
strategy. To preserve the pre-existing knowledge of the vision encoder,
we apply a layer-wise learning rate (LLDR) decay strategy, and the decay
factor is set to $0.90$.

<div class="table*" markdown="1">

<span id="tab:sota_comp" label="tab:sota_comp"></span>

</div>

## 4KHD Supervised Fine-tuning

After the pre-training, we empower the model to understand
high-resolution images and solve diverse challenges. Different from
previous perception tasks (, VQAv2, GQA) which typically answer
questions based on the noticeable object in the image. OCR-related tasks
depend on a detailed understanding of text within a high-resolution
image. For instance, in InfoVQA, the length of the longer side of 50% of
the images exceeds 2000 pixels. Low-resolution inputs can distort the
dense text information, causing the model to fail in its understanding.
However, we have observed a resolution saturation problem with the
aforementioned perception tasks, where the influence of resolution
becomes negligible.

<div class="table*" markdown="1">

</div>

To address this, we introduce a mixed-resolution training strategy for
more efficient training. For tasks requiring high resolution, we employ
the ‘HD-55’ setting during training. This allows for the input of 4K
($3840\times1600$) images without necessitating additional image
compression. These tasks are referred to as the HD-OCR QA tasks in
Table <a href="#tab:sft data" data-reference-type="ref"
data-reference="tab:sft data">[tab:sft data]</a>. For other tasks, we
implement a dynamic-resolution strategy. Images are resized to fall
within a range between their original size and the size specified by the
‘HD25’ setting. This dynamic approach enhances the robustness of the
LVLM against differences in input resolution, thereby enabling the LVLM
to utilize a larger resolution during inference. For instance, we have
observed that using the ‘HD30’ setting yields better results on most
OCR-related tasks when the LVLM is trained under the ‘HD25’ setting.

In practice, we jointly train all the components with a batch size of
2048 over 3500 steps. Data from multiple sources are sampled in a
weighted manner, with the weights based on the number of data from each
source. As the ‘HD-55’ setting has double image tokens than the ‘HD-25’,
we adjust the data loader to enable different batch sizes for them and
adjust their weight accordingly. The maximum learning rate is set to
$5 \times 10^{-5}$, and each component has its own unique learning
strategy. For the vision encoder, we set the LLDR to $0.9$, which aligns
with the pretraining strategy. For the LLM, we employ a fixed learning
rate scale factor of $0.2$. This slows down the update of the LLM,
achieving a balance between preserving its original capabilities and
aligning it with vision knowledge.

# Experiments

In this section, we validate the benchmark performance of our
InternLM-XComposer2-4KHD (IXC2-4KHD in the following for simplicity)
after supervised fine-tuning.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2404.06512v1_md/figures/reso_ablation3.png" />

</div>

<div class="table*" markdown="1">

<span id="tab:high-reso" label="tab:high-reso"></span>

</div>

## LVLM Benchmark results.

In Table <a href="#tab:sota_comp" data-reference-type="ref"
data-reference="tab:sota_comp">[tab:sota_comp]</a> and Table
<a href="#tab:entire_comp" data-reference-type="ref"
data-reference="tab:entire_comp">[tab:entire_comp]</a>, we compare our
IXC2-4KHD on a list of benchmarks with both SOTA open-source LVLMs and
closed-source APIs. Here we report results in
DocVQA[docvqa](http://arxiv.org/pdf/2111.05547v1), ChartQA[masry2022chartqa](http://arxiv.org/pdf/2203.10244v1),
InfographicVQA[infovqa](http://arxiv.org/pdf/2104.12756v2), TextVQA[textvqa](http://arxiv.org/pdf/1811.11903v1),
OCRBench[ocrbench](https://arxiv.org/pdf/2305.07895), MMStar[mmstar](http://arxiv.org/pdf/2006.11910v3),
MathVista[lu2024mathvista](http://arxiv.org/pdf/2310.02255v3),
MMMU[yue2023mmmu](http://arxiv.org/pdf/2311.16502v3),
AI2D[kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1), MME
[fu2023mme](http://arxiv.org/pdf/2306.05179v2), MMBench (MMB) [MMBench](http://arxiv.org/pdf/2005.12661v2),
MMBench-Chinese (MMB$^{CN}$) [MMBench](http://arxiv.org/pdf/2005.12661v2), SEED-Bench Image
Part (SEED$^{I}$)[li2023seedbench](https://arxiv.org/pdf/2307.16125), QBench-Testset
(QBench$^{T}$)[wu2023q](http://arxiv.org/pdf/2301.05065v2), MM-Vet
[yu2023mmvet](http://arxiv.org/pdf/2402.15896v1), HallusionBench
(HallB)[guan2023hallusionbench](https://arxiv.org/pdf/2310.14566). The evaluation is mainly
conducted on the OpenCompass VLMEvalKit[2023opencompass](https://github.com/open-compass/opencompass)
for the unified reproduction of the results.

**Comparison with Closed-Source APIs.** As demonstrated in Table
<a href="#tab:sota_comp" data-reference-type="ref"
data-reference="tab:sota_comp">[tab:sota_comp]</a>, IXC2-4KHD exhibits
competitive performance across a variety of benchmarks, rivaling that of
Closed-Source APIs. Owing to its high-resolution input, IXC2-4KHD
achieves a score of $90.0\%$ on DocVQA and $81.0\%$ on ChartQA, thereby
surpassing GPT-4V and Gemini-Pro with a non-trivial margin. In the
challenging InfographicVQA task, our model is the first open-source
model that is close to the performance of Closed-Source APIs, exceeding
the performance of previous open-source models by nearly $20\%$. In
addition to OCR-related tasks, IXC2-4KHD is a general-purpose Large
Vision-Language Modal that excels in semantic-level tasks, demonstrating
competitive results.

**Comparison with Open-Source Models.** We also conduct a comprehensive
comparison with open-source LVLMs under a similar model scale. As shown
in Table <a href="#tab:entire_comp" data-reference-type="ref"
data-reference="tab:entire_comp">[tab:entire_comp]</a>, our model
significantly outperforms existing open-source models, achieving
competitive results across all benchmarks. Notably, the
InternLM-XComposer2 series is the only method that achieves a higher
than $50\%$ score on the challenging MMStar benchmark.

**High-resolution Understanding Evaluation.** Then we compare IXC2-4KHD
with models that are specifically designed for high-resolution
understanding tasks. We report the results of 5 high-resolution
benchmarks in Table <a href="#tab:high-reso" data-reference-type="ref"
data-reference="tab:high-reso">[tab:high-reso]</a>, as a general LVLM,
IXC2-4KHD shows superb performance on these tasks and outperforms
competitors with a large margin. For example, IXC2-4KHD gets $68.6\%$ on
InfographicVQA, surpassing recent DocOwl 1.5 with $+17.9\%$. For the
OCRBench, IXC2-4KHD gets $67.5\%$, outperforms CogAgent with $+8.5\%$.

## Dive into Resolution

**High-Resolution Training is Critical for HD-OCR tasks.** We study four
resolution settings: HD-9 (1561 image tokens at most, we simply the
statement if the following), HD-16 (2653 tokens), HD-25 (4057 tokens),
and 4KHD (8737 tokens). Here we report the validation set of InfoVQA,
DocVQA, and TextVQA, test set of ChartQA and AI2D, MMBench EN-Test, and
a 2k subset of SEEDBench (we denote it as SEED$^*$). In the following
experiments, we report results on the above benchmarks by default.

As illustrated in Fig.<a href="#fig:reso" data-reference-type="ref"
data-reference="fig:reso">[fig:reso]</a>, we note a significant
improvement in the HD-OCR tasks as the resolution increases. For
instance, the model achieves only a $50.5\%$ score on the InfographicVQA
with the HD-9 setting. However, when we switch to the HD-16 setting, we
observe a performance gain of $+10.2\%$. The performance continues to
improve as the resolution increases, with saturation not observed even
for the 4KHD setting. Due to computational constraints, we defer the
exploration of the upper bound of improvement to future work. In terms
of other OCR-related tasks, the performance gain attributable to
increased resolution is relatively minor. For the perception-related
benchmarks, performance is saturated on the resolution that only has
negligible difference between the four settings.

<span id="tab:eval_resolution" label="tab:eval_resolution"></span>

**Higher Inference Resolution Leads to better results on Text-related
Tasks.** An intriguing observation from our experiments is that our
model, when inferring with a slightly higher resolution, tends to yield
improved results on text-related tasks. We present the results of HD-9,
HD-16, and HD-25 in Table
<a href="#tab:eval_resolution" data-reference-type="ref"
data-reference="tab:eval_resolution">[tab:eval_resolution]</a>. For
instance, IXC2-HD9 achieves a $50.5\%$ score on InfographicVQA. When we
infer with HD16, we see a performance gain of $+8.1\%$, without
additional training. Similar improvements are also observed with
IXC2-HD16 and IXC2-HD25. We posit that the dynamic image token length
used in training enhances the robustness of the LVLM, leading to better
results when the text in the image is more ‘clear’ in the higher
resolution input. Conversely, the results on ChartQA consistently
degrade under this setting. This could be due to the model becoming
confused about the chart structure when the resolution is altered.
Additionally, similar to the observation from Figure
<a href="#fig:reso" data-reference-type="ref"
data-reference="fig:reso">[fig:reso]</a>, the impact of resolution on
perception-related benchmarks appears to be quite minor.

**Visualization Results.** We provide the visualization results on
ultra-high HD images in Figure
<a href="#fig:teaser1" data-reference-type="ref"
data-reference="fig:teaser1">[fig:teaser1]</a> and Figure
<a href="#fig:teaser2" data-reference-type="ref"
data-reference="fig:teaser2">[fig:teaser2]</a>. Please refer to the
appendix for more results.

## High-Resolution Strategy Ablation

**The Role of Global-View.** We first examine the impact of the global
view in our Global-Local Format. As indicated in Table
<a href="#tab:global_view" data-reference-type="ref"
data-reference="tab:global_view">[tab:global_view]</a>, we find that the
global view is essential for the LVLM to accurately comprehend the input
image. When it is removed, the model performs worse across all
benchmarks. For instance, the model experiences a $-4.4\%$ drop in
performance on the MMBench EN-Test without the global view. We contend
that the global view offers a general macro understanding of the image,
which the model struggled to derive from the large number of tokens in
the local view.

**The Role of the Newline Token.** We incorporate a special newline
token at the end of each row of the image features before the flattening
operation. This token serves as an indicator of the image’s 2D
structure. We examine its impact on both the HD-9 and 4KHD strategies in
Table <a href="#tab:gang_n" data-reference-type="ref"
data-reference="tab:gang_n">[tab:gang_n]</a>. When a fixed
high-resolution strategy HD-9 is employed, we observe that the benefit
derived from the newline token is minor. This could be attributed to the
LVLM’s ability to handle limited differences in image ratios after
training. However, when we implement a more challenging 4KHD (HD-25 +
HD-55) strategy, which exhibits significant diversity in both image
ratio and token number, the LVLM demonstrates a notable decline in
performance on OCR-related tasks without the newline indicator. This
finding supports our hypothesis that the LVLM struggles to comprehend
the shape of the image when the image tokens are directly flattened into
a 1D sequence. The newline token can assist the model in better
understanding the structure of the image.

<span id="tab:gang_n" label="tab:gang_n"></span>

**Influence of Token Merging Strategy.** In practice, we employ a simple
merging strategy that concatenates four adjacent tokens along the
channel dimension. We have found this approach to be effective in
reducing the number of image tokens efficiently. Here we study the
influence of different token-merging strategies under the 4KHD setting.
In Table <a href="#tab:merge" data-reference-type="ref"
data-reference="tab:merge">[tab:merge]</a>, we study two additional
strategies: Re-Sampler[bai2023qwen](http://arxiv.org/pdf/1412.3919v1) and
C-Abstractor[cha2023honeybee](http://arxiv.org/pdf/2312.06742v2), with their default setting
and the same compressing rate $0.25$, , reducing an image with 576
tokens to 144 tokens. Results show that both concatenation and
C-Abstractor work well and get similar results on most benchmarks, this
observation is also consistent with the study in
MM-1[mckinzie2024mm1](http://arxiv.org/pdf/2403.01757v1) that the influence of the connector
is minor. However, the Re-Sampler performs worse than the other methods
with a noticeable margin. We argue this is caused by the learnable
queries used for gathering information requiring a great number of data
for training, our pre-training data is somewhat lightweight for it to
converge fully.

<span id="tab:merge" label="tab:merge"></span>

# Conclusion

In this paper, we propose the InternLM-Xcomposer2-4KHD that exceeds the
performance of previous open-source models on OCR-related tasks and also
achieves competitive results on general-purpose LVLM benchmarks. Thanks
to our dynamic resolution and automatic patch configuration, our model
supports a maximum training resolution of up to 4K HD. We also integrate
a global view patch to support the macro understanding and a learnable
newline token to handle the various input image resolutions. Our model’s
performance continues to improve as the training resolution increases
for HD-OCR tasks. Notably, we do not observe any performance saturation
even for the 4KHD setting, and we have not explored the upper bound due
to the computational burden increasing with higher-resolution inputs. In
future work, we plan to explore efficient solutions for accurate LVLM
training and inference, enabling our model to handle even higher
resolutions while maintaining computational efficiency.