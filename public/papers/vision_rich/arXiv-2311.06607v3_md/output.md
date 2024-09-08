# Introduction [sec:intro]

The field of large multimodal models (LMMs) is advancing quickly because
of their skill in handling different types of data, like images and
text. Their success in various tasks, including image captioning and
visual question answering, is attracting attention in the academic
community.

<div class="figure*" markdown="1">

<img src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/model.png" style="width:100.0%" alt="image" />

</div>

Training LMMs benefits greatly from high-resolution images
[bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1), because higher resolution allows these
models to detect more nuanced visual details, leading to accurate
recognition of objects, their interrelationships, and the broader
context within the image. Additionally, the improved visual clarity of
high-resolution images aids in effectively capturing and representing
complex details essential for detailed captioning. Despite advancements,
handling the wide range of image resolutions and training data quality
is still challenging, especially in complex situations. Solutions
include using pre-trained visual modules with larger input resolution
(like LLaVA1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1) ) and gradually
increasing the resolution of the training process through curriculum
learning (like Qwen-VL [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1),
PaLI-3 [chen2023pali-3](http://arxiv.org/pdf/2310.09199v2) and
PaLI-X [chen2023pali-x](http://arxiv.org/pdf/2109.04653v1)) have been explored, but they
demand significant training resources and still face challenges in
handling larger image sizes. To fully leverage the benefits of large
input resolution, it is crucial to have more detailed image
descriptions, which can enhance the understanding of image-text
relationships. However, the short captions in widely used datasets such
as COYO [kakaobrain2022coyo-700m](https://github.com/kakaobrain/coyo-dataset) and
LAION [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1) are usually intuitively
insufficient.

We introduce Monkey, a resource-efficient approach to increase input
resolution within the Large Multimodal Model frameworks. Compared to the
approach of directly interpolating the ViT to increase input resolution,
Monkey utilizes a new module that divides high-resolution images into
smaller patches using a sliding window method. Each patch is processed
independently by a static visual encoder, enhanced with
LoRA [hu2021lora](http://arxiv.org/pdf/2402.11485v1) adjustments and a trainable visual
resampler. This technique leverages existing LMMs while circumventing
the need for extensive pre-training. The key idea is that these encoders
are typically trained on smaller resolutions (like
448$\times$`<!-- -->`{=html}448), which is costly to train from scratch.
By resizing each patch to its supported resolution, we maintain the
training data distribution for the encoder. Our method, which uses
various trainable patches to enhance resolution, shows a clear advantage
over traditional interpolation techniques for positional embedding, as
demonstrated by our quantitative analysis.

To further leverage the advantage of large resolution, we have also
proposed an automatic multi-level description generation method. This
method is designed to produce high-quality, abundant caption data by
seamlessly combining insights from multiple generators. It utilizes the
strengths of a diverse array of advanced systems:
BLIP2 [li2023blip2](http://arxiv.org/pdf/2301.12597v3), known for its nuanced image-text
understanding; PPOCR [du2020pp](http://arxiv.org/pdf/2109.03144v2), a robust optical
character recognition system; GRIT [wu2022grit](https://arxiv.org/pdf/2212.00280), which
excels in granular image-text alignments; SAM [sam](http://arxiv.org/pdf/2305.01275v1), a
dynamic model for semantic alignment; and
ChatGPT [chatgpt](https://openai.com/blog/chatgpt/), an AI renowned for its contextual
understanding and language generation capabilities. By integrating the
unique capabilities of these systems, our method offers a comprehensive
and layered approach to caption generation, capturing a wide spectrum of
visual details.

We summarize the advantages of the Monkey as follows:

1.  **Support resolution up to 1344$\times$`<!-- -->`{=html}896 without
    pretraining**. By going beyond the usual
    448$\times$`<!-- -->`{=html}448 resolution used in LMMs, the higher
    resolution helps to better identify and understand small or closely
    grouped objects and dense text.

2.  **Contextual associations**. We introduce a multi-level description
    generation method that improves the model’s ability to grasp the
    relationships among multiple targets and more effectively utilize
    common knowledge in generating text descriptions.

3.  **Performance enhancements on many evaluation datasets**. As shown
    in Fig. <a href="#fig:onecol" data-reference-type="ref"
    data-reference="fig:onecol">1</a>, we carried out testing across 18
    diverse datasets, leading to a very competitive performance by our
    Monkey model in tasks such as Image Captioning, General Visual
    Question Answering, Scene Text-centric Visual Question Answering,
    and Document-oriented Visual Question Answering. In particular,
    during qualitative evaluations centered on dense text question
    answering, Monkey has shown promising results, comparing with GPT4V.

<div class="figure*" markdown="1">

<img src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/multi_level_des.png" style="width:95.0%" alt="image" />

</div>

# Related Work [sec:related]

The Large Multimodal Models (LMMs) field has seen significant progress,
particularly in enhancing visual and language processing. Methods like
Flamingo [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1) and
OpenFlamingo [awadalla2023openflamingo](http://arxiv.org/pdf/2402.17510v1) have advanced
visual representation by integrating a Perceiver Resampler with vision
encoders. BLIP2 [li2023blip2](http://arxiv.org/pdf/2301.12597v3) employs a Q-Former to link
the frozen LLM and vision encoder.
Unified-IO [lu2022unified](http://arxiv.org/pdf/2309.13885v1) demonstrates versatility by
training across over 80 diverse datasets, widening its domain
applicability. PaLM-E [driess2023palm-e](http://arxiv.org/pdf/2302.14030v3) adopts a unique
approach by treating images and text as “multimodal sentences” to
improve visual-language tasks. MiniGPT4 [zhu2023minigpt4](http://arxiv.org/pdf/2402.17510v1)
bridges visual modules and LLMs, enhancing multimodal capabilities.
InstructBLIP [dai2023instructblip](None), starting from BLIP2,
adds instructional inputs to the Q-Former for task-relevant visual
features. MME [fu2023mme](http://arxiv.org/pdf/2306.05179v2) introduces a benchmark for
evaluating LMMs’ perception and cognition.

Additionally, there has been significant progress in leveraging large
language models. The LLaVA series, including
LLaVA [liu2023llava](http://arxiv.org/pdf/2402.11690v1) and
LLaVA1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1), align vision encoders and
LLMs for better image-text understanding.
mPLUG-Owl [ye2023mplug](http://arxiv.org/pdf/2405.00390v2) focuses on fine-tuning with mixed
text and visual-text data. mPLUG-Owl2 [ye2023mplugowl2](https://arxiv.org/pdf/2311.04257)
introduces shared modules for better modality collaboration.
KOSMOS-2 [peng2023kosmos2](http://arxiv.org/pdf/2305.16103v1) enables visual answers like
detection boxes. Shikra [chen2023shikra](http://arxiv.org/pdf/2306.15195v2) specializes in
Referential Dialogue, adept at processing positional inputs and outputs.
BLiVA [hu2023bliva](http://arxiv.org/pdf/2308.09936v3) combines task-related and global
features for enhanced multimodal task processing.
Qwen-VL [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1) improves visual module
resolution to 448. OtterHD [li2023otterhd](https://arxiv.org/pdf/2311.04219) fine-tunes
Fuyu-8B [fuyu-8b](https://www.adept.ai/blog/fuyu-8b) with instruction/response pairs,
maintaining the original image size during inference.

Despite these advancements, challenges remain in extracting finer image
features, as noted by [liu2023hidden](http://arxiv.org/pdf/2305.07895v5), [xu2023lvlm](http://arxiv.org/pdf/2308.14353v1), which
indicate the need for ongoing development in the field.

# Methods

Fig. <a href="#fig:architecture" data-reference-type="ref"
data-reference="fig:architecture">[fig:architecture]</a> illustrates the
comprehensive architecture of Monkey. Initially, the input image is
segmented into patches. These patches are then processed through a
shared Vision Transformer (ViT) equipped with distinct adapters.
Subsequently, both local and global features, along with the question,
are processed using the shared resampler and the Large Language Model
(LLM), resulting in the generation of the desired answers.

## Enhancing Input Resolution

Input resolution is crucial for accurately interpreting text and
detailed image features. Previous
studies [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1), [chen2023pali-3](http://arxiv.org/pdf/2310.09199v2) have shown the
effectiveness of starting with smaller resolutions and progressively
advancing to larger ones through curriculum learning. However, this
approach can be highly resource-demanding, often necessitating
comprehensive pretraining with large-scale data (as seen in Qwen-VL,
which supports resolutions up to 448$\times$`<!-- -->`{=html}448). To
address these issues and efficiently enhance resolution, we introduce a
simple yet more effective technique.

Given an image $I \in \mathbb{R}^{H\times W \times 3}$, we employ a
sliding window $W \in \mathbb{R}^{H_v\times W_v}$ (where $H_v, W_v$
denote the supported resolution of the original LMM) to partition the
image into smaller, local sections. We also leverage
LoRA [hu2021lora](http://arxiv.org/pdf/2402.11485v1) within each shared encoder to address
the varied visual elements in different parts of an image. This
integration of LoRA is to help our encoders to recognize and assimilate
detail-sensitive features from each image area effectively, which
enhances the understanding of spatial and contextual relationships
without a substantial increase in parameters or computational demand.

To preserve the overall structural information of input image, we resize
the original image to dimensions ($H_v, W_v$), maintaining it as a
global image. Following this, both the individual patches and the global
image are processed through the visual encoder and resampler
concurrently. Here, the visual resampler, inspired by
Flamingo [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1), is a mechanism that
performs two main functions: summarizing visual information and
obtaining higher semantic visual representations in a language feature
space. It achieves this by leveraging a cross-attention module. The
module employs trainable vectors (embeddings) as query vectors, along
with image features from the visual encoder serving as keys for
cross-attention operations.

This approach strikes a balance between detailed and holistic
perspectives of the images, thereby enhancing the model performance
while avoiding a substantial increase in computational demand.

## Multi-level Description Generation

Previous models such as LLaVA [liu2023llava](http://arxiv.org/pdf/2402.11690v1) and
Qwen-VL [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1) used large datasets like
LAION [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1),
COYO [kakaobrain2022coyo-700m](https://github.com/kakaobrain/coyo-dataset), and
CC3M [sharma-etal-2018-conceptual](https://doi.org/10.18653/v1/P18-1238) for their initial
training. However, these datasets often offer image-text pairs that are
too simple (e.g., one short sentence to describe a complicated image),
lacking in detailed imagery. As a result, even when these models are
trained with high-resolution images, they struggle to accurately link
visual features with basic captions. This limitation affects the models
to effectively combine visual processing with language understanding.

To bridge this gap, we develop a novel approach for generating
multi-level descriptions automatically. This technique is designed to
create rich and high-quality caption data by effectively blending the
outputs from various generators. We utilize a combination of several
advanced systems, each bringing its own strength to the process:
BLIP2 [li2023blip2](http://arxiv.org/pdf/2301.12597v3), which provides a deep understanding
of the relationship between images and text;
PPOCR [du2020pp](http://arxiv.org/pdf/2109.03144v2), a strong performer in optical character
recognition; GRIT [wu2022grit](https://arxiv.org/pdf/2212.00280), specializing in detailed
image-text matching; SAM [sam](http://arxiv.org/pdf/2305.01275v1), focused on semantic
alignment; and ChatGPT [chatgpt](https://openai.com/blog/chatgpt/), known for its
exceptional ability in contextual language generation.

As shown in Fig. <a href="#fig:generation" data-reference-type="ref"
data-reference="fig:generation">[fig:generation]</a>, the image
description process begins with BLIP2 creating overall captions using a
Q-former for tight integration with the vision encoder and LLM, while
retaining original CC3M annotations for context. Next, GRIT, a
region-to-text model, generates detailed descriptions of specific
regions, objects, and their characteristics. PPOCR extracts text from
the images, and SAM segments and identifies objects and their parts.
These objects are then individually described by BLIP2. However, to
counter potential inaccuracies from these tools, especially in zero-shot
settings, we find it essential to further use BLIP2 to check for
consistency between image areas, objects, and their descriptions,
filtering out low-scoring matches. Finally, all data, including global
captions, localized descriptions, text extracts, and object details with
spatial coordinates, are fed into the ChatGPT API for fine-tuning,
enabling ChatGPT to generate accurate and contextually rich image
descriptions.

By merging the unique features of these systems, our approach achieves a
layered and comprehensive style of caption creation. It captures an
extensive range of visual and textual nuances, resulting in captions
that are not just elaborate, but also contextually diverse and engaging.

<div class="table*" markdown="1">

|  |  |  |  |  |  |  |  |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Model | Image Caption |  | General VQA |  |  |  |  |
|  | Flickr30K | TextCaps | VQAv2 | OKVQA | GQA | ScienceQA | VizWiz |
| Flamingo-80B  [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1) | 67.2 | \- | 56.3 | 50.6 | \- | \- | 31.6 |
| Palm-E-12B  [driess2023palm-e](http://arxiv.org/pdf/2302.14030v3) | \- | \- | 77.7 | <u>60.1</u> | \- | \- | \- |
| BLIP-2 (Vicuna-13B)  [li2023blip2](http://arxiv.org/pdf/2301.12597v3) | 71.6 | \- | 65.0 | 45.9 | 32.3 | 61.0 | 19.6 |
| InstructBLIP (Vicuna-13B)  [dai2023instructblip](None) | 82.8 | \- | \- | \- | 49.5 | 63.1 | 33.4 |
| Shikra (Vicuna-13B)  [chen2023shikra](http://arxiv.org/pdf/2306.15195v2) | 73.9 | \- | 77.4 | 47.2 | \- | \- | \- |
| mPLUG-Owl2  [ye2023mplugowl2](https://arxiv.org/pdf/2311.04257) | 85.1 | \- | 79.4 | 57.7 | 56.1 | <u>68.7</u> | <u>54.5</u> |
| LLaVA1.5 (Vicuna-7B)  [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1) | \- | \- | 78.5 | \- | **62.0** | 66.8 | 50.0 |
| Qwen-VL(Qwen-7B)  [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1) | <u>85.8</u> | <u>65.1</u> | <u>79.5</u> | 58.6 | 59.3 | 67.1 | 35.2 |
| Qwen-VL-Chat  [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1) | 81.0 | \- | 78.2 | 56.6 | 57.5 | 68.2 | 38.9 |
| Monkey | **86.1** | **93.2** | **80.3** | **61.3** | <u>60.7</u> | **69.4** | **61.2** |

</div>

<div id="TextVQA" markdown="1">

| Model | TextVQA | AI2D | STVQA | ESTVQA |
|:---|:--:|:--:|:--:|:--:|
| Pix2Struct-Large  [lee2023pix2struct](http://arxiv.org/pdf/2210.03347v2) | \- | 42.1 | \- | \- |
| BLIP-2 [li2023blip2](http://arxiv.org/pdf/2301.12597v3) | 42.4 | \- | \- | \- |
| InstructBLIP [dai2023instructblip](None) | 50.7 | \- | \- | \- |
| mPLUG-DocOwl  [ye2023mplug](http://arxiv.org/pdf/2405.00390v2) | 52.6 | \- | \- | \- |
| mPLUG-Owl2 [ye2023mplugowl2](https://arxiv.org/pdf/2311.04257) | 54.3 | \- | \- | \- |
| Qwen-VL [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1) | <u>63.8</u> | 55.9 | <u>59.1</u> | <u>77.8</u> |
| Qwen-VL-Chat [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1) | 61.5 | <u>57.7</u> | \- | \- |
| LLaVA-1.5 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1) | 58.2 | \- | \- | \- |
| Monkey | **67.6** | **57.9** | **67.7** | **82.6** |

Results on Scene Text-centric VQA.

</div>

<div class="table*" markdown="1">

</div>

## Multi-task Training

Our goal is to train a model that is both cost-effective and capable of
understanding different types of images for various tasks. By
integrating various datasets and employing uniform instructions for all
tasks, as guided by [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1), we enhance the
model’s learning ability and training efficiency.

We focus on tasks such as creating image captions, responding to
image-based questions, and other activities requiring the model to
process both text and images. For captioning, we instruct the model with
“Generate the caption in English:” for basic captions, and “Generate the
detailed caption in English:” for more intricate ones. When it comes to
answering questions about images, we use a straightforward format:
“{question} Answer: {answer}.”

In our training process, we use a variety of public datasets tailored to
specific tasks. For image captioning, we include both our own detailed
captions and established datasets like COCO
caption [karpathy2015coco](http://arxiv.org/pdf/1412.2306v2) and
TextCaps [textcaps](https://arxiv.org/pdf/2003.12462). For general Visual Question
Answering (VQA), we utilize datasets such as
VQAV2 [goyal2017making](http://arxiv.org/pdf/1612.00837v3),
OKVQA [marino2019ok](http://arxiv.org/pdf/1906.00067v2), GQA [hudson2019gqa](http://arxiv.org/pdf/2112.05136v1),
ScienceQA [lu2022learn](http://arxiv.org/pdf/2209.09513v2), and
VizWiz [gurari2018vizwiz](http://arxiv.org/pdf/1802.08218v4). For Text-centric VQA tasks, we
select TextVQA [singh2019towards](http://arxiv.org/pdf/1811.11903v1),
OCRVQA [mishra2019ocr](http://arxiv.org/pdf/2010.02582v1), and
AI2D [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1); while for document-related
VQA, we employ datasets like DocVQA [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1),
ChartQA [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1),
InfoVQA [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2),
DeepForm [deepform](http://arxiv.org/pdf/2303.13839v1), Kleister Charity
(KLC) [stanislawek2021kleister](http://arxiv.org/pdf/2003.02356v2), WikiTableQuestions
(WTQ) [pasupat2015compositional](http://arxiv.org/pdf/2009.13845v2),
TableFact [chen2019tabfact](http://arxiv.org/pdf/2311.06592v1), and
VisualMRC [tanaka2021visualmrc](http://arxiv.org/pdf/2101.11272v2). To ensure balanced
training, we control the image count for each task as detailed in
Tab. <a href="#tab:data" data-reference-type="ref"
data-reference="tab:data">[tab:data]</a>. Our compiled dataset, with
around 1.44 million examples, is designed to train our model effectively
in understanding and executing various instructions.

# Experiment

We evaluate our model by testing it across a spectrum of standard
vision-language tasks, including the generation of image descriptions,
answering diverse visual questions, and comprehending targeted phrases
in images.

## Implementation Details

**Model Configuration.** We conduct experiments based on the
well-trained Vit-BigG [ilharco_gabriel_2021_5143773](ilharco_gabriel_2021_5143773) and
LLM from Qwen-VL [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1), the pre-trained large
multimodal model. Since the vision encoder has already been well
pretrained, we proceed directly to the instruction-tuning stage. During
instruction tuning, $H_v$, $W_v$ are set to 448 to match the encoder of
Qwen-VL. We employ a consistent resampler across all crops. The
learnable queries engage with local features, utilizing the same set of
256 learnable queries for each crop. Due to limitations in training
time, our main experiments were mainly conducted using images of size
896$\times$`<!-- -->`{=html}896 unless specify. For LoRA, we set the
rank to 16 for the attention module and 32 for MLP in the encoder.
Monkey includes 7.7B parameters for a large language model, with 90M
parameters for the resampling module, an encoder with 1.9B parameters,
and 117M parameters for LoRA. The overall parameters for Monkey is 9.8B.

**Training.** We use our multi-level description generation method to
regenerate around 427k image-text pairs from the CC3M dataset,
previously used in LLaVA’s pretraining. During the training process, we
utilize the AdamW optimizer [adamw](http://arxiv.org/pdf/2311.11446v2) with a learning rate
of 1e-5 and the cosine learning rate schedule. Additionally, we set the
values of $\beta_1$ and $\beta_2$ to 0.9 and 0.95, respectively. We
incorporate a warmup period of 100 steps and employ a batch size of
1024. To control overfitting, we apply a weight decay of 0.1. The whole
training process takes 40 A800 days for one epoch.

## Results

We report the results on Image Caption, General VQA, Scene Text-centric
VQA, and Document-oriented VQA. We also conduct testing on the MME
benchmark and achieve a perception score of 1505.3, ranking second, as
shown in Fig. <a href="#fig:onecol" data-reference-type="ref"
data-reference="fig:onecol">1</a>. The details of each dataset can be
found in Appendix  <a href="#append:details" data-reference-type="ref"
data-reference="append:details">6</a>.

**Image Caption.** Image captioning is vital for connecting visual
content with the understanding of natural language. In our study, we
select Flickr30K [young2014image](http://arxiv.org/pdf/2208.09596v1) and
TextCaps [textcaps](https://arxiv.org/pdf/2003.12462) as the benchmark for testing the
image captioning task. TextCaps challenges the model to interpret and
reason text within images effectively. We present our model’s
performance on Flickr30K and TextCaps in
Tab. <a href="#General VQA" data-reference-type="ref"
data-reference="General VQA">[General VQA]</a>, where the results
indicate that Monkey demonstrates enhanced performance on these
datasets. We also qualitatively show effectiveness of our method in
offering detailed image descriptions in
Sec. <a href="#subsec:vis" data-reference-type="ref"
data-reference="subsec:vis">4.4</a> and Appendix
 <a href="#append:visualization" data-reference-type="ref"
data-reference="append:visualization">7</a>
 <a href="#append:comparison" data-reference-type="ref"
data-reference="append:comparison">9</a>.

**General VQA.** General visual question answering (VQA) requires
ability to learn visual and textual information, showing a deep
understanding of how they interrelate. For General VQA, we validate on
five benchmarks: VQAv2 [goyal2017making](http://arxiv.org/pdf/1612.00837v3),
OKVQA [marino2019ok](http://arxiv.org/pdf/1906.00067v2), GQA [hudson2019gqa](http://arxiv.org/pdf/2112.05136v1),
ScienceQA [lu2022learn](http://arxiv.org/pdf/2209.09513v2), and
VizWiz [gurari2018vizwiz](http://arxiv.org/pdf/1802.08218v4). The performance results are
shown in Tab. <a href="#General VQA" data-reference-type="ref"
data-reference="General VQA">[General VQA]</a>. Our model shows
remarkable proficiency in VQAV2, OKVQA, ScienceQA, and VizViz,
surpassing the nearest competing method by an average of 1.62%. These
results highlight the effectiveness of our method, emphasizing its use
of high input resolution and detailed data.

**Scene Text-centric VQA.** Text information is commonly found in
real-world scenes, making the ability to answer questions about text in
images a crucial aspect of question-answering tasks. For our evaluation,
we employ four datasets: TextVQA [singh2019towards](http://arxiv.org/pdf/1811.11903v1),
AI2D [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1), STVQA [STVQA](http://arxiv.org/pdf/2304.01603v1),
and ESTVQA [ESTVQA](http://arxiv.org/pdf/2002.10215v2). The results, shown in
Tab. <a href="#TextVQA" data-reference-type="ref"
data-reference="TextVQA">1</a>, indicate that our model leads in
performance on these datasets, outperforming the nearest competitor by
an average of 4.35%. Based on our observation, this enhanced performance
is mainly attributed to the increased image resolution, which brings
smaller text and finer details into clearer view. Moreover, the
inclusion of detailed caption data during training provides valuable
textual context, further boosting the robustness of the model.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/caption_new.png" style="width:95.0%" />

</div>

**Document-oriented VQA.** Despite the clean backgrounds of documents,
their densely packed text poses distinct challenges. To effectively
evaluate our model, we select representative benchmarks including
DocVQA [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1),
ChartQA [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1),
InfographicVQA [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2),
DeepForm [deepform](http://arxiv.org/pdf/2303.13839v1),
KLC [stanislawek2021kleister](http://arxiv.org/pdf/2003.02356v2), and
WTQ [pasupat2015compositional](http://arxiv.org/pdf/2009.13845v2). The results, as detailed
in Tab. <a href="#DocVQA" data-reference-type="ref"
data-reference="DocVQA">[DocVQA]</a>, show that Monkey surpasses Qwen-VL
in most Document-oriented VQA tasks, achieving an averagely significant
improvement of 9.77%. The higher resolution of documents reveals more
intricate details and a denser concentration of information. Monkey’s
capability to process larger input resolutions enhances its spatial
perception, thereby improving its recognition and comprehension of
various document elements like text, charts, infographics, and forms.

## Ablation Study [subsec:ab]

We conduct thorough experiments to validate the effectiveness of our
designs.

**Ablation study on strategies of enhancing input resolution.** We first
evaluate the existing technique of improving input resolution, as
illustrated in Tab. <a href="#SizeAblation" data-reference-type="ref"
data-reference="SizeAblation">[SizeAblation]</a>. Resizing the visual
encoder using traditional positional position interpolation to a size of
896 results in worse performance compared with our method under the same
settings (r1 vs. r9). Interestingly, applying LoRA to the encoder for
this traditional interpolation method appears to be less effective than
not using it (r1 vs. r2). This may due to the inherited parameters from
the previous encoder are specifically tuned by lower resolution,
changing it by force may necessitate more training resources.

For our method (r3-r9), as we increase the input size, there is a
noticeable boost in performance, especially demonstrated in the DeepForm
dataset. It can be observed that adding LORA does not significantly
increase FLOPs and the use of one LORA or four LORAs results in a
minimal difference in throughput (r7-r9). The model’s ability to discern
intricate details and sharper images enhances its understanding of
visual aspects such as objects, shapes, and textures, thereby improving
its overall visual perception. When we further push the input resolution
to 1344$\times$`<!-- -->`{=html}896, which is the highest resolution the
device can support, the model shows further improvements on
high-resolution datasets like DeepForm, InfoVQA, and WTQ, as detailed in
Tab. <a href="#SizeAblation" data-reference-type="ref"
data-reference="SizeAblation">[SizeAblation]</a>. However, we can note
that for some datasets, such as TextVQA, using the largest resolution
results in a slight decline in performance; nevertheless, the original
average resolution in the TextVQA dataset is around 950 pixels in width
and 811 pixels in height, further increasing its input resolution seems
unnecessary for these images.

Furthermore, as shown in
Tab. <a href="#Tab:llava15_ablation" data-reference-type="ref"
data-reference="Tab:llava15_ablation">[Tab:llava15_ablation]</a>, we
consistently demonstrate the effectiveness of our method on LLaVA1.5.
Impressively, we noticed significant improvements when we increased the
input resolution from 224 to 448, demonstrating the efficiency of our
approach.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/compare_new.png" style="width:98.0%" />

</div>

**Trainable Adapters.** As shown in
Tab. <a href="#SizeAblation" data-reference-type="ref"
data-reference="SizeAblation">[SizeAblation]</a>, reducing the LoRA
number causes a performance decrease. Using one LoRA for all patches
compared to not using LoRA provides a better perception of local details
(r7 vs. r8), especially with a significant improvement in STVQA.
Utilizing four LoRA modules leads to a better performance, which may
because this approach enables the model to learn a better understanding
of the spatial relationships and contextual information within distinct
image regions.

**Collaboration between High Resolution and Multi-level Description.**
To validate the collaboration between High Resolution and Multi-level
Description, we conduct ablation studies on LLaVA1.5. We employ a ViT-L
as our vision encoder and Vicuna13B [vicuna2023](https://lmsys.org/blog/2023-03-30-vicuna/) as the
language model. By replacing the original annotation from CC3M with our
generated annotations in the pretraining, we consistently achieved
better results on GQA, TextVQA and MMVet [yu2023mm](http://arxiv.org/pdf/2402.15896v1), as
demonstrated in
Tab. <a href="#Tab:llava15_ablation" data-reference-type="ref"
data-reference="Tab:llava15_ablation">[Tab:llava15_ablation]</a>.
Furthermore, we have observed that detailed descriptions consistently
yield greater performance enhancements at resolutions of 336 and 448,
compared to a resolution of 224. In Appendix
 <a href="#sec:resolutions" data-reference-type="ref"
data-reference="sec:resolutions">10</a>, we provide visualization
results for Monkey at different resolutions. These results show that
models with high resolution shines when trained with more comprehensive
descriptions.

## Visualization [subsec:vis]

In a side-by-side qualitative analysis, we compared Monkey with GPT4V
and other LMMs on a task of generating detailed captions. The results,
illustrated in
Fig. <a href="#Densecap_vs_GPT4V" data-reference-type="ref"
data-reference="Densecap_vs_GPT4V">[Densecap_vs_GPT4V]</a>, demonstrate
Monkey’s superior capability in providing exhaustive descriptions of
images. For instance, in the image from
Fig. <a href="#Densecap_vs_GPT4V" data-reference-type="ref"
data-reference="Densecap_vs_GPT4V">[Densecap_vs_GPT4V]</a>, both Monkey
and GPT4V successfully identified an “Emporio Armani” store in the
background. Moreover, Monkey went further in detailing various elements
in the scene, such as describing “another woman in a red coat and black
pants carrying a black purse”.

Additionally, as shown in
Fig. <a href="#Doc_Chart" data-reference-type="ref"
data-reference="Doc_Chart">[Doc_Chart]</a>, we qualitatively observe
that in many cases for understanding complex text-based inquiries,
Monkey has shown impressive performance when compared to GPT4V. More
visualization results of Monkey can be found in Appendix.

## Limitation

The capability of our method to process input images is constrained to a
maximum of six patches due to the limited input length of the language
model. This restriction hampers the further expansion of input
resolution.

Moreover, for the multi-level description generation approach, it is
capable of describing only the scene presented in the image and its
scope is bound by the world knowledge encapsulated in BLIP2 and the
original CC3M annotations. For instance, when provided with a photo of a
location in a country, the method can describe the visual aspects of the
scene, but it lacks the ability to identify and specify that the scene
is indeed in which country.

# Conclusion

This paper proposes a training-efficient approach to effectively improve
the input resolution capacity up to 1344$\times$`<!-- -->`{=html}896
pixels without pretraining from the start. To bridge the gap between
simple text labels and high input resolution, we propose a multi-level
description generation method, which automatically provides rich
information that can guide the model to learn the contextual association
between scenes and objects. With the synergy of these two designs, our
model achieved excellent results on multiple benchmarks. By comparing
our model with various LMMs, including GPT4V, our model demonstrates
promising performance in image captioning by paying attention to textual
information and capturing fine details within the images; its improved
input resolution also enables remarkable performance in document images
with dense text.

# Acknowlegement [acknowlegement]

This research is supported by NSFC (No. 62225603).

# Summary of the Evaluation Benchmarks [append:details]

We present a comprehensive overview of the evaluation benchmarks
utilized, along with their corresponding metrics Tab.
 <a href="#tab:benchmark" data-reference-type="ref"
data-reference="tab:benchmark">[tab:benchmark]</a>. For the Image
Caption task, we selected two datasets:
Flickr30K [young2014image](http://arxiv.org/pdf/2208.09596v1), which is an image caption
dataset for natural images, and TextCaps [textcaps](https://arxiv.org/pdf/2003.12462),
which is an image caption dataset for natural images with text. For
general Visual Question Answering (VQA), we chose five commonly used
datasets. VQAV2 [goyal2017making](http://arxiv.org/pdf/1612.00837v3) is an open-ended VQA
dataset focused on natural images, while
OKVQA [marino2019ok](http://arxiv.org/pdf/1906.00067v2) requires additional world knowledge.
GQA [hudson2019gqa](http://arxiv.org/pdf/2112.05136v1) is a dataset designed for real-world
visual reasoning and compositional question answering.
ScienceQA [lu2022learn](http://arxiv.org/pdf/2209.09513v2) involves multimodal
multiple-choice VQA on science topics, and
VizWiz [gurari2018vizwiz](http://arxiv.org/pdf/1802.08218v4) aims to answer questions from
blind individuals. In the domain of Scene Text-centric VQA, our
selection includes TextVQA [singh2019towards](http://arxiv.org/pdf/1811.11903v1),
AI2Diagram [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1),
STVQA [STVQA](http://arxiv.org/pdf/2304.01603v1), and ESTVQA [ESTVQA](http://arxiv.org/pdf/2002.10215v2). AI2D
is a multiple-choice VQA dataset that focuses on science diagrams, while
the others involve reading and reasoning about text in natural images.
For the STVQA and ESTVQA datasets, we followed the split provided by
 [liu2023hidden](http://arxiv.org/pdf/2305.07895v5). Regarding Doc-oriented VQA, we
encompass various document images, including documents, charts,
infographics, reports, and HTML tables. In the case of
DeepForm [deepform](http://arxiv.org/pdf/2303.13839v1) and
KLC [stanislawek2021kleister](http://arxiv.org/pdf/2003.02356v2), we transform the Key
Information Extraction task into a Visual Question Answering (VQA) task.
Additionally, we evaluate Monkey on the MME
benchmark [fu2023mme](http://arxiv.org/pdf/2306.05179v2), which measures perception and
cognition abilities. Furthermore, for the ablation study on LLaVA1.5
 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1), we adhere to the evaluation settings
specified by LLaVA1.5.

<div class="table*" markdown="1">

</div>

# More Visualization Results [append:visualization]

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/QA_caption.png" style="width:95.0%" />

</div>

We presented additional visualization results, where
Fig. <a href="#QA_ability" data-reference-type="ref"
data-reference="QA_ability">[QA_ability]</a> demonstrates Monkey’s
capabilities in various VQA tasks. Monkey analyzes the question,
identifies the key elements in the image relevant to answering the
question, and exhibits the ability to perceive even minute text within
the image. Moreover, Monkey can reason about the objects present in the
scene and possesses a strong understanding of visual charts. In
addition, Fig. <a href="#QA_ability" data-reference-type="ref"
data-reference="QA_ability">[QA_ability]</a> also showcases Monkey’s
impressive captioning ability, accurately describing various objects in
the image and providing appropriate summaries.

# More Examples of our Generated Data 

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/densetext.png" style="width:95.0%" />

</div>

In Fig. <a href="#dense_text" data-reference-type="ref"
data-reference="dense_text">[dense_text]</a>, we present the detailed
captions generated by our method. Compared to the original annotations
from the CC3M  [sharma-etal-2018-conceptual](https://doi.org/10.18653/v1/P18-1238), our
generated descriptions cover many more details of the image, providing a
more detailed description of the image.

# Comparison with other LMMs. [append:comparison]

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/QA_compare.png" style="width:95.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/caption_compare.png" style="width:95.0%" />

</div>

The comparison results of the VQA task in Fig.
<a href="#QA_compare" data-reference-type="ref"
data-reference="QA_compare">[QA_compare]</a> indicate that after
applying our method of scaling up the model size, Monkey has achieved
significant performance advantages in tasks related to dense text. It
not only surpasses the performance of QwenVL-Chat
 [bai2023qwen-vl](http://arxiv.org/pdf/1412.3919v1), LLaVA-1.5
 [liu2023llava1.5](http://arxiv.org/pdf/2310.19145v1), and mPLUG-Owl2
 [ye2023mplugowl2](https://arxiv.org/pdf/2311.04257) but also achieves promising results
compared to GPT-4V  [openai2023gpt4](https://arxiv.org/pdf/2303.08774) in tasks related to
dense text. This clearly demonstrates the importance of scaling up the
model size for performance improvement in multimodal large models. It
further validates the effectiveness of our method in enhancing the
performance of multimodal large models.

In Fig. <a href="#Caption_compare" data-reference-type="ref"
data-reference="Caption_compare">[Caption_compare]</a>, the comparison
between Monkey and GPT-4V, QwenVL-Chat, LLaVA-1.5, and mPLUG-Owl2 on
Detailed Caption task is shown. It can be observed that Monkey
accurately describes the content of the image and exhibits high
sensitivity to the text within the image. It provides detailed
descriptions of the image while ensuring accuracy.

# Visualization results for models at different resolutions. [sec:resolutions]

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/QA_res.png" style="width:95.0%" />

</div>

In Fig. <a href="#QA_res" data-reference-type="ref"
data-reference="QA_res">[QA_res]</a>, we performed VQA tasks testing at
three different resolutions: 896, 784, and 672. The visual results
obtained further validate the importance of our size expansion method
for improving the performance of LMMs. While using a resolution of 896
for VQA tasks testing yielded correct results, using resolutions of 784
and 672 resulted in errors, with the smallest size of 672 showing more
errors.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2311.06607v3_md/figs/caption_res.png" style="width:95.0%" />

</div>

In Fig. <a href="#Caption_res" data-reference-type="ref"
data-reference="Caption_res">[Caption_res]</a>, we conducted tests at
three different resolutions: 896, 784, and 672. It can be observed that
as the resolution decreases, the details in the images gradually become
less visible to the model.

# Data Generation.

**Hyperparameter Control in Data Generation Pipeline.** The appropriate
selection of hyperparameters is crucial. We empirically selected them
based on qualitative results, finding SAM’s default threshold and a 0.5
Image-Text Matching Score to be effective. We conducted a quantitative
validation on 80 samples using the GPT-4V evaluation. The results shown
in Tab. <a href="#Tab:hyper" data-reference-type="ref"
data-reference="Tab:hyper">[Tab:hyper]</a> reveal that SAM’s threshold
is relatively robust, and the 0.5 threshold for Image-Text Matching
Score offers a better performance.

**Comparison with LLaVA’s GPT4 Method.** While the GPT4 method in LLaVA
is dependent on using manually annotated captions from the COCO dataset
as a foundational basis for data generation, our approach focuses on
generating original, detailed captions autonomously. Additionally, our
detectors are skilled in revealing a spectrum of details in images, from
text to nuanced object characteristics, which enables to enrich
unlabeled data by extracting complex, multi-level details, paving the
way for the creation of both cost-effective and accurate image
descriptions.

**Why choose BLIP2?** We found that the performance is very similar in
the GPT-4V evaluation when utilizing brief descriptions of local areas
from other VLMs, as shown in
Tab. <a href="#Tab:othervlm" data-reference-type="ref"
data-reference="Tab:othervlm">[Tab:othervlm]</a>. However, for
generating approximately 5M descriptions, using BLIP2 takes
approximately 3 days, while LLaVA and mPLUG-Owl require about 21 days
and 32 days, respectively. For the sake of saving time, we choose BLIP2.

# Ablation study on Global Feature.

We conducted experiments on the presence or absence of global features
at a resolution of 896. By adding global features, the results showed a
7.5% performance gain on TextVQA, a 0.6% performance gain on GQA, and a
6.2% performance gain on DocVQA. This demonstrated that global features
contribute to enhancing the overall performance.

[^1]: $^\dagger$equal contribution; $^*$corresponding authors