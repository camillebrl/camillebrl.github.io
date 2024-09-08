# Introduction

Instruction tuning
[ouyang2022training](https://arxiv.org/pdf/2203.02155), [chung2022scaling](https://arxiv.org/pdf/2210.11416) improves
generalization to unseen tasks by formulating various tasks into
instructions. Such open-ended question-answering capability fosters the
recent chatbot boom since ChatGPT. Recently, visual instruction-tuned
models [liu2023visual](https://arxiv.org/pdf/2304.08485), [li2023otter](http://arxiv.org/pdf/2311.00233v2), [Li2023LargeMM](http://arxiv.org/pdf/2306.14895v1)
further augment conversation agents with visual encoders such as
CLIP-ViT [dosovitskiy2020image](https://arxiv.org/pdf/2010.11929), [radford2021learning](https://arxiv.org/pdf/2103.00020),
enabling human-agent interaction based on images. However, possibly due
to the dominance of natural images in training data (e.g., Conceptual
Captions [changpinyo2021conceptual](https://arxiv.org/pdf/2102.08981) and COCO
[lin2015microsoft](https://arxiv.org/pdf/1405.0312)), they struggle with understanding
texts within images [liu2023hidden](https://arxiv.org/pdf/2305.07895). However, textual
understanding is integral to visual perception in everyday life.

Fortunately, tools such as Optical Character Recognition (OCR,
`\citealp{156468}`{=latex}) allow us to recognize text in images. One
naive way to utilize this is to add recognized texts to the input of
visual instruction-tuned models [gao2023llamaadapterv2](https://arxiv.org/pdf/2304.15010).
However, such approach significantly increases the computation (longer
context lengths), and might not fully leverage the encoding capability
of visual encoders. To this end, we propose to enhance the end-to-end
visual instruction-tuned model by collecting instruction-following data
that require understanding texts within images.

Specifically, we first collect 422K noisy instruction-following data
using text-rich[^2] images by combining manually written instructions
(e.g., “Identify any text visible in the provided image.”) and the OCR
results. Such large-scale noisy-aligned data effectively improve feature
alignment between visual features and the language decoder. Furthermore,
we prompt text-only GPT-4 [openai2023gpt4](https://arxiv.org/pdf/2303.08774) with OCR
results and image captions to generate 16K conversations, where each
conversation can be multiple turns of question & answer pairs, as
high-quality instruction-following examples. This process requires GPT-4
to de-noise the OCR results and develop specific questions to create
complex instructions based on the input (Figure
<a href="#fig:highquality" data-reference-type="ref"
data-reference="fig:highquality">[fig:highquality]</a>).

To evaluate the effectiveness of the collected data, we use noisy and
high-quality examples to augment the pretraining and fine-tuning stages
of LLaVA [liu2023visual](https://arxiv.org/pdf/2304.08485) accordingly. We name our model
**LLaVAR**, signifying the LLaVA (Large Language and Vision Assistant)
that can **R**ead. Compared to the original LLaVA, we also conducted
experiments scaling the input resolution from $224^2$ to $336^2$ to
better encode small textual details. Empirically, we report the results
on four text-based VQA datasets following the evaluation protocol from
[liu2023hidden](https://arxiv.org/pdf/2305.07895). Moreover, we apply GPT-4-based
instruction-following evaluation to 30 natural images from COCO
[lin2015microsoft](https://arxiv.org/pdf/1405.0312), [liu2023visual](https://arxiv.org/pdf/2304.08485) and 50 text-rich
images from LAION [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1). We also provide
qualitative analysis (e.g., on posters, website screenshots, and tweets)
to test more complex instruction-following skills.

To sum up, our contributions are as follows:

-   We collect 422K noisy instruction-following data and 16K
    high-quality instruction-following data. Both are shown to be
    effective in augmenting visual instruction tuning.

-   Our model, LLaVAR, significantly enhances text understanding within
    images while slightly improving the model’s performance on natural
    images.

-   The enhanced capability enables our model to provide end-to-end
    interactions based on various forms of online content that combine
    text and images.

-   We open source the training and evaluation data together with the
    model checkpoints.

<div class="figure*" markdown="1">

</div>

# Related Work

#### Instruction Tuning

Following natural language instructions is the key capability for an
agent to interact with real-world users. Instruction tuning starts from
collecting human-preferred feedback for human written instructions
[ouyang2022training](https://arxiv.org/pdf/2203.02155) or formulating multi-task training
in a multi-task instruction-following manner
[chung2022scaling](https://arxiv.org/pdf/2210.11416), [wang2022supernaturalinstructions](https://arxiv.org/pdf/2204.07705).
However, large, capable instruction-tuned models are usually
closed-sourced and serve as commercial APIs only. Recently, Alpaca
[wang2022selfinstruct](https://arxiv.org/pdf/2212.10560), [alpaca](https://github.com/tatsu-lab/stanford_alpaca), Vicuna
[vicuna2023](https://lmsys.org/blog/2023-03-30-vicuna/), and Baize [xu2023baize](https://arxiv.org/pdf/2304.01196)
start the trend of generating high-quality instruction-following data
based on LLMs such as GPT-3.5 / ChatGPT / GPT-4 and finetuning the open
source LLaMA model [touvron2023llama](https://arxiv.org/pdf/2302.13971). However,
evaluating the ability to follow instructions remains a challenge. While
GPT-4 has demonstrated superior evaluation capabilities
[liu2023geval](https://arxiv.org/pdf/2303.16634), there are still a number of concerns,
such as bias toward response length [xu2023baize](https://arxiv.org/pdf/2304.01196) and
lack of robustness to the order of examples
[wang2023large](https://arxiv.org/pdf/2305.17926). Following
[vicuna2023](https://lmsys.org/blog/2023-03-30-vicuna/), [liu2023visual](https://arxiv.org/pdf/2304.08485), [dubois2023alpacafarm](https://arxiv.org/pdf/2305.14387), we
use GPT-4-based instruction-following evaluation in this work.

#### Multimodal Instruction Tuning

Recently, instruction tuning has been expanded to the multimodal
setting, including image, video
[zhang2023video](http://arxiv.org/pdf/2311.12919v2), [maaz2023videochatgpt](https://arxiv.org/pdf/2306.05424), and audio
[Huang2023AudioGPTUA](http://arxiv.org/pdf/2108.04325v2), [zhang2023speechgpt](https://arxiv.org/pdf/2305.11000). For
image-based instruction tuning, MiniGPT-4
[zhu2023minigpt4](https://arxiv.org/pdf/2304.10592) employs ChatGPT to curate and improve
detailed captions for high-quality instruction-following data. LLaVA
[liu2023visual](https://arxiv.org/pdf/2304.08485) generates multimodal
instruction-following data by prompting text-only GPT-4 with captions
and object’s bounding boxes. LLaMA-Adapter
[zhang2023llamaadapter](https://arxiv.org/pdf/2303.16199), [gao2023llamaadapterv2](https://arxiv.org/pdf/2304.15010) uses COCO
data for text-image feature alignment and utilizes textual data only for
instruction tuning. mPLUG-owl [ye2023mplugowl](https://arxiv.org/pdf/2304.14178) combines
more than 1000M image-text pairs for pretraining and a 400K mixture of
text-only/multimodal instruction-following data for finetuning. However,
according to [liu2023hidden](https://arxiv.org/pdf/2305.07895), most of these models
struggle to accomplish tasks requiring OCR capability. InstructBLIP
[dai2023instructblip](https://arxiv.org/pdf/2305.06500) transforms 13 vision-language
tasks (including OCR-VQA [mishra2019ocrvqa](http://arxiv.org/pdf/2010.02582v1)) into the
instruction-following format for instruction tuning. Cream
[kim2023cream](https://arxiv.org/pdf/2305.15080) applies multi-task learning that includes
predicting masked texts in images. A more comprehensive survey can be
found in [li2023multimodal](li2023multimodal). In this work, we select
LLaVA as our baseline, which is the most data-efficient and powerful
model, and demonstrate the effectiveness of our proposed pipeline.

# Data Collection

Starting from the LAION-5B [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1) dataset
[^3], our goal is only to keep images that are text-rich. Considering
that documents usually contain plenty of text, we first obtained a
binary classification dataset by combining natural images and document
data. Subsequently, we trained an image classifier using a DiT
[2022DIT](https://doi.org/10.1145/3503161.3547911)-base backbone, which was fine-tuned on the
RVL-CDIP dataset [harley2015evaluation](https://arxiv.org/pdf/1502.07058). Hopefully, such
a classifier can predict whether an image contains text or not. We first
build a subset by selecting images with a predicted probability greater
than 0.8 while also satisfying $p($watermark$) < 0.8$ and
$p($unsafe$) < 0.5$ [^4]. The derived subset is noisy due to the
limitation of the classifier. To further clean up the data and
incorporate human judgment,

<div class="wrapfigure" markdown="1">

r0.59

<div class="center" markdown="1">

<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/data_intro.png" style="width:59.0%" alt="image" />

</div>

<span id="fig:Data Collection" label="fig:Data Collection"></span>

</div>

we randomly sampled 50K images and clustered them into 100 clusters
based on `CLIP-ViT-B/32` visual features. After inspecting the
clustering results, we carefully select 14 clusters (see Figure
<a href="#clusters" data-reference-type="ref"
data-reference="clusters">[clusters]</a> in the Appendix for examples)
containing diverse text-rich images ranging from posters, covers,
advertisements, infographics, educational materials, and logos. The
cluster model is then used as the filter to collect images for
constructing our instruction-following examples. As a reference, we
provide a CLIP [radford2021learning](https://arxiv.org/pdf/2103.00020)-based
categorization (see Appendix for details.) to illustrate the
distribution of images for both two types of data we collected in Figure
<a href="#fig:Data Collection" data-reference-type="ref"
data-reference="fig:Data Collection">[fig:Data Collection]</a>. We
summarize our collected data and LLaVA’s data in Table
<a href="#data stats" data-reference-type="ref"
data-reference="data stats">1</a>.

#### Noisy Instruction-following Data [para:Noisy Instruction-following Data]

Using the clustering model as a filter, we collect 422K deduplicated
images that belong to the 14 preferred clusters. To balance the examples
from different categories, we keep at most 52K examples for one cluster.
We run all images through PaddleOCR [^5]. Note that running OCR at the
original resolution (e.g.,$1024^2$) might recognize small fonts that are
not visible by visual encoders like CLIP ViT
(`\citealp{dosovitskiy2020image, radford2021learning}`{=latex},
resolution up to $336^2$). To ensure the recognition of visible fonts
while maintaining OCR accuracy, we perform OCR on the image after
downsampling (the short edge is resized to 384 pixels if longer than
that.) to extract the text. Then, based on the geometric relationships
between the recognized words, we merge them into paragraphs before
concatenating them. As a robust instruction-following model should react
similarly to instructions with similar meanings, we reword “Identify any
text visible in the provided image.” into ten distinct instructions
(Table <a href="#Instructions" data-reference-type="ref"
data-reference="Instructions">3</a> in Appendix). We then create a
single-turn conversation for a given image by **(i)** randomly sampling
an ***input instruction*** and **(ii)** using recognized texts as the
desired ***output response***. Such instruction-following data is noisy
because of the relatively limited performance of OCR tools on diverse
fonts and colorful backgrounds.

<div id="data stats" markdown="1">

| **Data** | **Image** | **Instruction** | **\# Conv** | **Avg Ins Len** | **Avg Res Len** |
|:---|:--:|:--:|:--:|:--:|:--:|
| LLaVA pretraining | CC3M | CC3M | 595K | 15.9 | 15.4 |
| R~pretraining~ (Ours) | LAION | PaddleOCR | 422K | 17.2 | 48.8 |
| LLaVA finetuning | COCO | GPT-4 | 158K | 15.9 | 93.1 |
| R~finetuning~ (Ours) | LAION | GPT-4 | 16K | 15.1 | 40.5 |

Summary of data statistics. R~pretraining~ and R~finetuning~ denote the
additional pre-training / finetuning data we collected. The average
instruction and response length are calculated after LLaMA tokenization.

</div>

#### GPT-4-based Instruction-following Data

Compared to high-quality instruction-following data, there are mainly
two issues for the noisy data collected above. **(i)** Responses should
contain organized sentences instead of raw OCR results with missing
words and grammar errors. **(ii)** Instructions should be diverse,
suitable and specific to the given image instead of monotonously asking
for all visible texts. To address these issues, we follow
[liu2023visual](https://arxiv.org/pdf/2304.08485) to generate instruction-following data
by prompting text-only GPT-4 [openai2023gpt4](https://arxiv.org/pdf/2303.08774) with OCR
results and captions.

It is challenging to prompt GPT-4 with fragmented OCR results in a few
words to generate non-trivial instructions. To this end, we carefully
select 4 of the 14 previously mentioned clusters (the 3rd, 4th, 6th and
9th clusters in Figure <a href="#clusters" data-reference-type="ref"
data-reference="clusters">[clusters]</a>) to collect images with enough
visible and coherent sentences. As shown in Figure
<a href="#fig:Data Collection" data-reference-type="ref"
data-reference="fig:Data Collection">[fig:Data Collection]</a>, such
filtering dramatically increases the percentage of book covers and quote
images. We randomly selected 4K examples from each cluster (no overlap
with images used for noisy instruction-following data), yielding a total
of 16K images. Following prior work
[wang2022selfinstruct](https://arxiv.org/pdf/2212.10560), [alpaca](https://github.com/tatsu-lab/stanford_alpaca), [liu2023visual](https://arxiv.org/pdf/2304.08485), we
provide the visualization of verb-noun pairs for instructions generated
by GPT-4 in Appendix Figure
<a href="#fig:instruction1" data-reference-type="ref"
data-reference="fig:instruction1">7</a>. For those instructions without
a verb-noun pair, we demonstrate the frequency of objects being asked in
Appendix Figure <a href="#fig:instruction2" data-reference-type="ref"
data-reference="fig:instruction2">8</a>.

Furthermore, based on the system message and two in-context few-shot
examples (shown in Appendix ), we ask GPT-4 to generate conversational
data based on OCR results and image captions (Figure
<a href="#fig:highquality" data-reference-type="ref"
data-reference="fig:highquality">[fig:highquality]</a>). The generated
questions are used as ***input instructions***, and answers are used as
***output responses***. Concretely, for a given image, we first provide
two OCR results from EasyOCR and PaddleOCR, which can complement each
other. To illustrate visual elements other than texts within the image,
we also provide the result of BLIP-2 image captioning
[li2023blip2](https://arxiv.org/pdf/2301.12597). To prevent the caption from focusing on
the text, we use OCR bounding boxes to mask the text and then use the
inpainting [telea2004image](telea2004image) to refill the mask before
using generation captions. Note that captioning models might suffer from
hallucinations [rohrbach2018object](rohrbach2018object). We mention this
unreliability in our system message and ask GPT-4 only to generate
questions with sure answers. We leave the generation of more detailed
captions [rotstein2023fusecap](https://arxiv.org/pdf/2305.17718), [hu2022promptcap](https://arxiv.org/pdf/2211.09699) for
future work.

<div class="figure*" markdown="1">

</div>

# Model Architecture and Training

#### Architecture

In most of our study, we use the same model architecture as LLaVA. For
the visual encoder $V$, we use `CLIP-ViT-L/14` for $224^2$ resolution
and `CLIP-ViT-L/14-336` for $336^2$ resolution. The grid features before
the last transformer layer are then transformed into the word embedding
space of the language decoder through a trainable projection matrix $W$.
We use Vicuna-13B [vicuna2023](https://lmsys.org/blog/2023-03-30-vicuna/), a LLaMA-based
[touvron2023llama](https://arxiv.org/pdf/2302.13971) instruction-tuned language model, as
the language decoder $D$ except the ablation study in Table
<a href="#table: ablation on encoder/image" data-reference-type="ref"
data-reference="table: ablation on encoder/image">[table: ablation on
encoder/image]</a>.

In Section <a href="#Ablation: encoders/res" data-reference-type="ref"
data-reference="Ablation: encoders/res">5.1.0.3</a> and Appendix , we
extend the current architecture by adding an extra high-resolution
(high-res) visual encoder. Such a high-res encoder outputs thousands of
patch features, which means that the transformed features and
instruction tokens cannot fit in the context length of the language
decoder. To this end, we propose to add cross-attention modules to the
decoder, which attend to key-value pairs transformed from the high-res
patch features.

#### Training

We follow the two-stage training design of LLaVA (Figure
<a href="#fig:training" data-reference-type="ref"
data-reference="fig:training">[fig:training]</a>). The training
objectives of both stages are the same: generate ***output responses***
(*$<$res$>$*) for the ***input instructions*** (*$<$ins$>$*). The
transformed image tokens (*$<$img$>$*) are added before or after the
first input instruction. **(i)** During the first pre-training stage,
only the projection matrix $W$ is trained for feature alignment. Since
the decoder $D$ is frozen, training tolerates noisy data. In the
pre-training stage, we combine the 595K pre-training data from LLaVA
with our 422K noisy instruction-following data. **(ii)** Both the
projection matrix $W$ and the language decoder $D$ are trained during
the finetuning stage, where we merge our 16K instruction-following data
into the 158K instruction-following data from LLaVA as the training set.
Note that the visual encoder is frozen throughout the training period,
which might restrict text recognition performance, as CLIP is trained
for general-purpose text-image alignment. Better choices of the visual
encoder [tschannen2022clippo](https://arxiv.org/pdf/2212.08045) or CLIP-ViT finetuning
[ye2023mplugowl](https://arxiv.org/pdf/2304.14178) may further benefit the visual
understanding capability, which we leave for future work.

# Experiments

We use the same training hyperparameters as LLaVA[^6], except that
**(i)** We set the maximum sequence length to 1024 during pre-training.
**(ii)** We first pad any given image to a square shape before resizing
it to the desired input size, preventing some image content from
cropping during preprocessing. For both resolutions ($224^2$, $336^2$),
we reproduce the original LLaVA for a fair comparison. The GPT-4 model
used in this work refers to the `gpt-4-0314` version, while the cost to
collect finetuning data is around \$300. The temperature used to sample
GPT-4 is set to $1.0$ for the generation of training data, $0.7$ for the
generation of evaluation data, and $0.2$ for the evaluation based on
GPT-4. All experiments are run on NVIDIA A100 80GB GPUs. During the
evaluation, the temperature used to sample from our model is set at
$0.9$ for text-based VQA, $0.7$ for GPT-4-based instruction-following
evaluation, and $0.2$ for other qualitative demonstrations.

<div class="tabular" markdown="1">

lc S@ S @ S @ S & **Res.** & **ST-VQA** & **OCR-VQA** & **TextVQA** &
**DocVQA**  
BLIP-2 `\citeyearpar{li2023blip2}`{=latex} $\dagger$ & & 21.7 & 30.7 &
32.2 & 4.9  
OpenFlamingo `\citeyearpar{anas_awadalla_2023_7733589}`{=latex}
$\dagger$ & & 19.3 & 27.8 & 29.1 & 5.1  
MiniGPT4 `\citeyearpar{zhu2023minigpt4}`{=latex} $\dagger$ & & 14.0 &
11.5 & 18.7 & 3.0  
LLaVA `\citeyearpar{liu2023visual}`{=latex} $\dagger$ & & 22.1 & 11.4 &
28.9 & 4.5  
mPLUG-Owl `\citeyearpar{ye2023mplugowl}`{=latex} $\dagger$ & & 29.3 &
28.6 & 40.3 & 6.9  
LLaVA $\ddagger$ & & 24.3 & 10.8 & 31.0 & 5.2  
LLaVAR & & 30.2 & 23.4 & 39.5 & 6.2    
LLaVA $\ddagger$ & & 28.9 & 11.0 & 36.7 & 6.9  
LLaVAR & & 39.2 & 23.8 & 48.5 & 11.6  

</div>

<div class="tabular" markdown="1">

lSSSS & **ST-VQA** & **OCR-VQA** & **TextVQA** & **DocVQA**  
(1) LLaVA & 28.9 & 11.0 & 36.7 & 6.9  
(2) LLaVA + R~pretraining~ & 36.7 & 26.1 & 46.5 & 9.6  
(3) LLaVA + R~finetuning~ & 34.1 & 21.6 & 43.6 & 9.5  
(4) LLaVA + C~pretraining~ & 35.4 & 27.0 & 45.6 & 9.2  
(5) LLaVA + N~finetuning~ & 34.1 & 25.9 & 43.3 & 10.2  
(6) LLaVAR & 39.2 & 23.8 & 48.5 & 11.6  

</div>

<div class="tabular" markdown="1">

ccccSSSS & **CLIP Res.** & **Extra Enc.** & **R~pretraining~** &
**ST-VQA** & **OCR-VQA** & **TextVQA** & **DocVQA**  
(a) & $224^2$ & & Low & 28.9 & 25.6 & 37.8 & 6.2  
(b) & $336^2$ & & Low & 37.4 & 31.0 & 45.7 & 11.4  
(c) & $224^2$ & & High & 28.9 & 24.9 & 35.8 & 6.2  
(d) & $336^2$ & & High & 36.9 & 30.6 & 46.5 & 11.7  
(e) & $224^2$ & & Low & 31.8 & 30.9 & 41.2 & 8.1  
(f) & $224^2$ & & High & 33.5 & 29.8 & 42.2 & 10.2  
(g) & $224^2$ & & High ($2\times$) & 35.8 & 30.7 & 45.6 & 15.3  

</div>

<figure id="fig:qualitative-ablation">
<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/calendar.png" />
<table>
<thead>
<tr class="header">
<th colspan="2" style="text-align: left;"><strong>Question</strong>: Who
wrote this book?</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;">GT</td>
<td style="text-align: left;">Sandra Boynton</td>
</tr>
<tr class="even">
<td style="text-align: center;">(1)</td>
<td style="text-align: left;">The book was written by Sandra
Boynton.</td>
</tr>
<tr class="odd">
<td style="text-align: center;">(2)</td>
<td style="text-align: left;">The book was written by Sandra Boyton. It
is a children’s book that features a family calendar with an
accompanying activity guide. The book provides information (...) The
cover of the book displays a picture of the family from the calendar,
which helps to visually represent the topic of the book.</td>
</tr>
<tr class="even">
<td style="text-align: center;">(3)</td>
<td style="text-align: left;">The book is written by Sandra
Byington.</td>
</tr>
<tr class="odd">
<td style="text-align: center;">(4)</td>
<td style="text-align: left;">The book was written by Sandra
Boynton.</td>
</tr>
<tr class="even">
<td style="text-align: center;">(5)</td>
<td style="text-align: left;">With 500+ Nity StickerS
&amp;aSwEllMagnetic Phone List! Moms 291573016171800 FAMII (Pocket)
Calendar (...) 2016 SANDRA BOYNTON</td>
</tr>
<tr class="odd">
<td style="text-align: center;">(6)</td>
<td style="text-align: left;">The book was written by Sandra
Boynton.</td>
</tr>
</tbody>
</table>
<figcaption>Ablation study based an example from OCR-VQA. GT refers to
ground truth, and (1) - (6) are different model variants from Table <a
href="#table: VQA ablation" data-reference-type="ref"
data-reference="table: VQA ablation">[table: VQA ablation]</a>. We
replace the excessively long response with (...).</figcaption>
</figure>

## Quantitative Analysis

#### Text-based VQA

Following the evaluation protocol in [liu2023hidden](https://arxiv.org/pdf/2305.07895), we
evaluate the performance of LLaVAR on four text-based VQA datasets:
ST-VQA [STVQA](https://doi.org/10.1109/icdar.2019.00251), OCR-VQA
[mishra2019ocrvqa](http://arxiv.org/pdf/2010.02582v1), TextVQA [textvqa](https://doi.org/10.1109/cvpr.2019.00851),
and DocVQA [mathew2020docvqa](https://arxiv.org/pdf/2007.00398), representing various
domains (see Appendix for more details and Appendix for more datasets).
We present the results of the baseline models and our models in Table
<a href="#table: VQA result" data-reference-type="ref"
data-reference="table: VQA result">[table: VQA result]</a>. Note that
InstructBLIP includes OCR-VQA in its training sets, making it
incomparable with our settings. In both resolution settings and all four
datasets, LLaVAR substantially improves the LLaVA baseline,
demonstrating that our collected data can bring about a robust
improvement. Furthermore, the improvement is more significant in $336^2$
resolution compared to $224^2$, indicating that the collected data might
bring a greater improvement at even higher resolutions. Our best model,
$336^2$-based LLaVAR, performs best in 3 out of 4 evaluated datasets.
Note that this is not a fair comparison. Some key factors include
different language decoders, resolutions, and magnitudes of text-image
training data. We provide more discussions on the comparison with
mPLUG-Owl and the result of finetuning mPLUG-Owl using our data in
Appendix .

#### Ablation Study on pretraining/finetuning data

We report the result in Table
<a href="#table: VQA ablation" data-reference-type="ref"
data-reference="table: VQA ablation">[table: VQA ablation]</a> and
Figure <a href="#fig:qualitative-ablation" data-reference-type="ref"
data-reference="fig:qualitative-ablation">1</a>. **(i)** Based on
variants (2) and (3), we find that the collected data can benefit the
pretraining stage (R~pretraining~) and finetuning stage (R~finetuning~)
separately while being complementary to each other in most cases [^7].
More importantly, enhancing the pretraining stage alone achieves the
second-best overall performance, indicating the potential to boost
textual detail understanding without dependence on GPT-4-generated
high-quality data. **(ii)** Using pretraining images, we obtain
C~pretraining~ by replacing the pretraining instructions with questions
& captions, the same pattern as LLaVA. As variant (4) is not as good as
(2), we can conclude that OCR is more advantageous than captions.
**(iii)** We further validate the value of GPT-4 generated data by
generating noisy finetuning data (N~finetuning~), similar to pretraining
data. Variant (5) achieves comparable accuracy as variant (3). However,
as shown in Figure
<a href="#fig:qualitative-ablation" data-reference-type="ref"
data-reference="fig:qualitative-ablation">1</a>, such noisy finetuning
data hurts the instruction-following capability: (5) responds with all
recognized texts while ignoring the questions.

#### Ablation Study on encoders/image resolution [Ablation: encoders/res]

While keeping finetuning data the same, we report the quantitative
results of adding an extra visual encoder and varying the pretraining
data in Table
<a href="#table: ablation on encoder/image" data-reference-type="ref"
data-reference="table: ablation on encoder/image">[table: ablation on
encoder/image]</a>. **(i)** Take `Pix2Struct-base` as an example, we
find that adding an extra high-res visual encoder with cross-attention
indeed improves the performance ((g) vs. (a)), especially achieving the
best zero-shot performance on DocVQA (15.3% accuracy). The performance
gain on other datasets is relatively limited, probably due to the extra
encoder we use being pretrained on web pages instead of natural images.
On the other hand, the performance of (e) and (f) remains poor, without
doubling the number of high-res examples in R~pretraining~. Given the
larger number of parameters initialized in the cross-attention module,
they may be underfitting when trained on the same data as the projection
matrix $W$ (e.g., (e) vs. (b)), similar to the finding in
[zeng2023matters](zeng2023matters). **(ii)** Considering (c) vs. (a) and
(d) vs. (b), while the images are resized to the same size after
preprocessing, high-res OCR results turn out to be not necessarily
better than the low-resolution version, suggesting the capability of the
visual encoder is almost saturated in (a) and (b). For more details and
results on the extra high-res encoder, please refer to Appendix .

#### GPT-4-based instruction-following evaluation

We also report the GPT-4 evaluation results on instruction-following
questions in Table <a href="#table: GPT4 eval" data-reference-type="ref"
data-reference="table: GPT4 eval">2</a>. **(i)** **Natural Images**: 90
questions based on 30 COCO validation images from
[liu2023visual](https://arxiv.org/pdf/2304.08485), including three aspects: conversation,
detail description, and complex reasoning. This aims to test whether our
collected data will hurt, maintain, or improve the model’s performance
on natural images. First of all, using a higher resolution brings
improvement (+2.9) in the performance of detail description, which is
intuitive. Furthermore, LLaVAR achieves a better trade-off and increases
the performance of all three aspects (+1.6 on average). More details are
in Appendix . **(ii)** **Text-Rich Images**: Similar to collecting the
finetuning data, we leverage 50 text-rich images from LAION to collect
instruction-following questions based on OCR results and human-annotated
captions. We then collect responses from our trained model and use GPT-4
to calculate the relative score w.r.t GPT-4 responses. We add this as an
extra dimension “**Read**” to Table
<a href="#table: GPT4 eval" data-reference-type="ref"
data-reference="table: GPT4 eval">2</a>, where our model demonstrates a
significant (+3.8) improvement. The Appendix provides an example in
Table <a href="#table:read" data-reference-type="ref"
data-reference="table:read">11</a>.

## Qualitative Analysis

We use a recent movie poster [^8] to demonstrate the difference between
LLaVA and LLaVAR when interacting with humans based on text-rich images.
LLaVA, without augmenting textual understanding within images, suffers
from hallucination when answering these questions. Some mentioned
movies, like “A Man Called Ove” and “The Ugly Truth”, are real movies,
suggesting that the language decoder is hallucinating its internal
knowledge, while the visual encoder cannot encode helpful information.
Alternatively, LLaVAR can correctly answer many of the provided
questions with **faithful** information, which is clearly grounded in
the image. However, some limitations remain, such as the spelling error
“ottol” (We provide more statistics related to such spelling errors in
Appendix ). Also, the final question asks for information that is not
observable from the given poster, where an expected response should
express such uncertainty instead of giving concrete answers.
Nevertheless, neither model correctly answers the question.

<div id="table: GPT4 eval" markdown="1">

|                  | **Res** | **Conversation** | **Detail** | **Complex** | **Read** |
|:-----------------|:-------:|:----------------:|:----------:|:-----------:|:--------:|
| LLaVA (Original) | $224^2$ |       83.1       |    75.3    |    96.5     |    \-    |
| LLaVA            | $336^2$ |       83.9       |    78.2    |    95.3     |   87.9   |
| LLaVAR           | $336^2$ |       84.5       |    78.9    |    96.5     |   91.7   |

Relative scores (w.r.t. text-only GPT-4) for instruction-following
questions, where the first three dimensions are based on natural images,
the last dimension (“Read”) is based on text-rich images. In the first
row, we show the original results ($224^2$-based) fetched from
[liu2023visual](https://arxiv.org/pdf/2304.08485). We report our reproduced LLaVA on
$336^2$ resolution for a fair comparison.

</div>

<figure id="fig:qualitative">
<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/example.png" />
<table>
<tbody>
<tr class="odd">
<td style="text-align: left;"><strong><em>H:</em></strong> When will
this movie be released?</td>
</tr>
<tr class="even">
<td style="text-align: left;"><strong><em>L:</em></strong> The movie
"The Call of the Ottoman" starring Daniel Craig is expected to be
released on February 14th, 2023.</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><strong><em>R:</em></strong> The movie
will be released on January 13.</td>
</tr>
<tr class="even">
<td style="text-align: left;"><strong><em>H:</em></strong> Who starred
in the movie?</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><strong><em>L:</em></strong> The movie
stars Tom Cruise and Cobie Smulders.</td>
</tr>
<tr class="even">
<td style="text-align: left;"><strong><em>R:</em></strong> Tom Hanks
starred in the movie.</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><strong><em>H:</em></strong> What is this
movie about?</td>
</tr>
<tr class="even">
<td style="text-align: left;"><strong><em>L:</em></strong> The movie "A
Man Called Ove" is a comedy-drama film directed by Hannes Holm.
(...)</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><strong><em>R:</em></strong> The movie is
about Tom Hanks, who is called Ottol, and it is an international
bestseller based on a true story. (...)</td>
</tr>
<tr class="even">
<td style="text-align: left;"><strong><em>H:</em></strong> Can you
describe the given image?</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><strong><em>L:</em></strong> The image is
a movie poster featuring a man and a cat. The man, who appears to be a
character from the movie “The Ugly Truth,” (...)</td>
</tr>
<tr class="even">
<td style="text-align: left;"><strong><em>R:</em></strong> The image
features a man standing in front of a house with a cat beside him.</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><strong><em>H:</em></strong> Who directed
the film?</td>
</tr>
<tr class="even">
<td style="text-align: left;"><strong><em>L:</em></strong> The film was
directed by Florian Henckel von Donnersmarck.</td>
</tr>
<tr class="odd">
<td style="text-align: left;"><strong><em>R:</em></strong> The film was
directed by Tom Hanks, who made his directorial debut with this
movie.</td>
</tr>
</tbody>
</table>
<figcaption>Qualitative Comparison. We color the human inputs
(<strong><em>H</em></strong>) in blue, LLaVA’s responses
(<strong><em>L</em></strong>) in pink, and LLaVAR’s responses
(<strong><em>R</em></strong>) in green. Both are <span
class="math inline">336<sup>2</sup></span>-based models. We replace the
excessively long response with (...).</figcaption>
</figure>

<figure id="fig:Case Study">
<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/casestudyv4.png" style="width:100.0%" />
<figcaption>Case study of the recognizable font size, in which the
x-axis refers to the height of ground truth answers in the image and the
y-axis stands for the answer accuracy of models. We plot the results for
both <span class="math inline">224<sup>2</sup></span>-based models and
<span class="math inline">336<sup>2</sup></span>-based
models.</figcaption>
</figure>

## Case Study: Recognizable Font Size

We first collect 825 examples from OCR-VQA, which have answers directly
presented in the image and are detectable by the OCR tool. By rescaling
the images, we test the model’s performance in answering these questions
while the vertical heights of answers range from 3 pixels to 19 pixels.
We report the result in Fig
<a href="#fig:Case Study" data-reference-type="ref"
data-reference="fig:Case Study">3</a>. **(i)** For the baseline model
LLaVA, it struggles to provide correct answers in all scenarios, for
both $224^2$-based and $336^2$-based versions. **(ii)** Our model LLaVAR
achieves significantly better results in all scales. We observe a
threshold for recognizable texts for both $224^2$-based and
$336^2$-based versions as the accuracy sharply decreases when the height
is smaller than 7 pixels. More interestingly, the $224^2$-based version
achieves better performance on small texts with 3 pixels height while
the $336^2$-based achieves better performance on large texts with more
than 7 pixels height. We assume the extra training stage of CLIP $336^2$
makes it better on the larger scale but worse on the smaller scale.

## Transferred Instruction-following Capability

According to the dataset statistics (Table
<a href="#data stats" data-reference-type="ref"
data-reference="data stats">1</a>) and the visualization (Figure
<a href="#fig:instruction1" data-reference-type="ref"
data-reference="fig:instruction1">7</a>), our collected
instruction-following data is not as diverse and substantial as LLaVA.
This can be attributed to the relatively limited information given GPT-4
compared to five different human-written captions used in LLaVA. The
content of text-rich images is also less diverse than that of natural
images. While using more complex in-context examples can definitely
stimulate generating more complicated instruction-following examples, it
can also multiply the cost. In Appendix Figure
<a href="#fig:Demo" data-reference-type="ref"
data-reference="fig:Demo">6</a>, we demonstrate the transferred
instruction-following capability of LLaVA, potentially from both the
LLaVA data and the Vicuna backbone. While the extra data we add mainly
focuses on understanding the visible texts within images, LLaVAR manages
to build its reasoning, writing, and elaboration skills based on the top
of its text recognition capability in an end-to-end manner. This allows
users to interact with various online content based on simple
screenshots.

# Conclusion

In this work, we enhance visual instruction-tuned models in terms of
their capability to read texts in images. Using text-rich images from
the LAION dataset, we collect 422K noisy instruction-following examples
using OCR results only and 16K high-quality instruction-following data
based on text-only GPT-4. These two sets of data are leveraged to
augment the pretraining stage and finetuning stage of LLaVA accordingly.
Our model, LLaVAR, demonstrates superior performance in understanding
texts within images and following human instructions on both prior
benchmarks and real-world online content. Moreover, our analysis shows
that the same augmented data is more effective with higher resolution.
Additionally, using noisy instruction-following examples to augment
pretraining essentially boosts the model performance without prompting
GPT-4. For future work, we encourage exploration of **(i)** better image
selection criteria or domain reweighting strategy
[xie2023doremi](https://arxiv.org/pdf/2305.10429) and **(ii)** more data-efficient and
computation-efficient ways to augment instruction-following models with
multimodal capability, especially in the high-res scenario.

<div id="Instructions" markdown="1">

| **Instructions**                                                       |
|:-----------------------------------------------------------------------|
| Identify any text visible in the image provided.                       |
| List all the text you can see in the given image.                      |
| Enumerate the words or sentences visible in the picture.               |
| Describe any readable text present in the image.                       |
| Report any discernible text you see in the image.                      |
| Share any legible words or sentences visible in the picture.           |
| Provide a list of texts observed in the provided image.                |
| Note down any readable words or phrases shown in the photo.            |
| Report on any text that can be clearly read in the image.              |
| Mention any discernable and legible text present in the given picture. |

Ten instructions asking for OCR results.

</div>

# A [CLIP-based categorization]

#### CLIP-based categorization

Based on the observation of selected clusters, we divide the images used
into 8 categories. For each category, we use one or multiple words as
labels.

-   **Quote & Meme**: “quote”, “internet meme”.

-   **Poster**: “movie poster”, “podcast poster”, “TV show poster”,
    “event poster”, “poster”,

-   **Book Cover**: “book cover”, “magazine cover”.

-   **Game Cover**: “game cover”.

-   **Ad & Product Packaging**: “ad”, “advertisement”, “food packaging”,
    “product packaging”.

-   **Infographic**: “chart”, “bar chart”, “pie chart”, “scatter plot”.

-   **Educational Material**: “exam paper”, “quiz”, “certificate”, “book
    page”.

-   **Logo**: “logo”.

For each word, we use the following templates to achieve embedding-space
ensembling [radford2021learning](https://arxiv.org/pdf/2103.00020):

-   “a photo of a {}.”

-   “a blurry photo of a {}.”

-   “a black and white photo of a {}.”

-   “a low contrast photo of a {}.”

-   “a high contrast photo of a {}.”

-   “a bad photo of a {}.”

-   “a good photo of a {}.”

-   “a photo of a small {}.”

-   “a photo of a big {}.”

For each image, we calculate the similarity between the image and all
words mentioned above using `CLIP-ViT-L/14`. If the highest similarity
is less than $0.15$, we then classify the image into **Other**,
otherwise we classify into the “super class” (e.g., **Poster**) of the
word (e.g., “movie poster”) with the highest similarity.

# B [context prompt]

#### System Message

adapted from [liu2023visual](https://arxiv.org/pdf/2304.08485).

<div class="center" markdown="1">

<div class="myquote" markdown="1">

You are an AI visual assistant, and you are seeing a single image. What
you see is provided with two OCR results and one image caption
describing the information within the same image you are looking at.
Image captions might include hallucinations, while OCR results are more
accurate. Answer all questions with definite answers as you are seeing
the image.

Design a conversation between you and a person asking about this photo.
The answers should be in a tone that a visual AI assistant is seeing the
image and answering the question. Ask diverse questions and give
corresponding answers.

Include questions asking about the visual content of the image (e.g.,
the man, the sunset, the ocean.) and the texts contained in the image.
Only include questions that have definite answers:

\(1\) one can see the content in the image that the question asks about
and can answer confidently;

\(2\) one can determine confidently from the image that it is not in the
image. Do not ask any questions that cannot be answered confidently;

\(3\) DO NOT mention OCR or image caption in your questions and answers;

\(4\) DO NOT ask about information from captions while it looks
unrelated to or contradicts OCR results.

Also include complex questions that are relevant to the content in the
image, for example, asking about background knowledge of the texts in
the image, asking to discuss about the design of the image, etc. Again,
do not ask about uncertain details. Provide detailed answers when
answering complex questions. For example, give detailed examples or
reasoning steps to make the content more convincing and well-organized.
You can include multiple paragraphs if necessary.

</div>

</div>

#### Few-shot Examples

<div class="center" markdown="1">

<div class="myquote" markdown="1">

**Human:**

WHAT F YOUR BLESSINGS COME THROUGH RAINDROPS PDf Induded LAURASTOR Y
DEVOHONAL BASED ON THE GRAAAMY VARD WIN ISONG mAT HAS INSEIRED MILLION

WHATIFYOUR BLESSINGS COMETHROUGHRAINDROPS PDF Included LAURASTORY A
DEVOTIONAL BASEDON THE GRAMMY AWARD WINNI PISONGTHATHASINSPIREDMILLIONSE

a girl is standing in a field with a rainbow

**GPT:**

Question: What is the name of the devotional mentioned in the image?

Answer: The devotional is called “What If Your Blessings Come Through
Raindrops” by Laura Story.

Question: What is special about the song mentioned in the image?

Answer: The song is special because it’s a Grammy Award-winning song
that has inspired millions of people."

**Human:**

One of the hardest things in life to accept is a called third strike
Robert Frost te

One of the hardest things in life to accept is a called third strike
Robert Frost quotefancy

a close up of a baseball glove

**GPT:**

Question: Why is the third strike hard to accept? Explain the quote to
me.

Answer: The quote Öne of the hardest things in life to accept is a
called third strikeättributed to Robert Frost is a metaphor for life
situations. In baseball, a called third strike is when the umpire
determines that a pitch was a strike, but the batter did not swing. This
can be frustrating for the batter because they missed an opportunity to
hit the ball due to either a lack of confidence or misjudgment.

In life, this metaphor refers to difficult moments where we might miss
opportunities or face setbacks because of our own inaction or
hesitation. The quote suggests that it is hard to accept these missed
opportunities or challenges, especially when they result from our own
choices or lack of action.

</div>

</div>

# C [Evaluation Dataset]

Details of evaluation datasets used in the main paper:

-   ST-VQA [STVQA](https://doi.org/10.1109/icdar.2019.00251) contains 31791 questions that require
    understanding the scene text, based on images from COCO
    [lin2015microsoft](https://arxiv.org/pdf/1405.0312), Visual Genome
    [krishna2016visual](https://arxiv.org/pdf/1602.07332), ImageNet
    [imagenet](https://doi.org/10.1109/CVPR.2009.5206848), etc.

-   TextVQA [textvqa](https://doi.org/10.1109/cvpr.2019.00851) contains 45,336 questions that
    need reading and reasoning about the text in images to answer, based
    on images from OpenImages [openimages](http://arxiv.org/pdf/1809.05929v7).

-   OCR-VQA [mishra2019ocrvqa](http://arxiv.org/pdf/2010.02582v1) contains more than 1
    million questions asking about information from book cover images
    [iwana2016judging](https://arxiv.org/pdf/1610.09204).

-   DocVQA [mathew2020docvqa](https://arxiv.org/pdf/2007.00398) contains 50000 questions
    based on document images.

Details of extra datasets in Appendix:

-   CT80 [risnumawan2014robust](risnumawan2014robust) contains 80 images for
    curved text OCR evaluation. The formats of questions are: (1) “What
    is written in the image?" for English words. (2) “What is the number
    in the image?" for digit string.

-   POIE [textvqa](https://doi.org/10.1109/cvpr.2019.00851) contains 3000 camera images collected
    from the Nutrition Facts label of products, together with 111,155
    text instances. The format of questions is “What is {entity name} in
    the image?".

-   ChartQA [masry2022chartqa](masry2022chartqa) includes 4,804 charts
    with 9608 human-written questions.

# D [other metrics]

#### Results of other metrics

<div id="table: STVQA metrics" markdown="1">

|        | Res.    | METEOR | ROUGE-L | CIDEr |
|:-------|:--------|:------:|:-------:|:-----:|
| LLaVA  | $224^2$ |  7.0   |   8.2   | 15.3  |
| LLaVAR |         |  10.0  |  11.4   | 24.5  |
| LLaVA  | $336^2$ |  8.4   |   9.9   | 19.1  |
| LLaVAR |         |  12.8  |  14.3   | 30.9  |

Results on ST-VQA using text-matching metrics.

</div>

<div id="table: textVQA metrics" markdown="1">

|        | Res.    | METEOR | ROUGE-L | CIDEr |
|:-------|:--------|:------:|:-------:|:-----:|
| LLaVA  | $224^2$ |  8.7   |  10.5   | 12.2  |
| LLaVAR |         |  12.5  |  14.9   | 21.4  |
| LLaVA  | $336^2$ |  9.9   |  12.1   | 15.3  |
| LLaVAR |         |  14.8  |  17.4   | 27.0  |

Results on textVQA using text-matching metrics.

</div>

<div id="table: OCR-VQA metrics" markdown="1">

|        | Res.    | METEOR | ROUGE-L | CIDEr |
|:-------|:--------|:------:|:-------:|:-----:|
| LLaVA  | $224^2$ |  0.2   |   0.1   |  0.0  |
| LLaVAR |         |  0.3   |   0.1   |  0.0  |
| LLaVA  | $336^2$ |  0.3   |   0.1   |  0.0  |
| LLaVAR |         |  0.2   |   0.1   |  0.0  |

Results on OCR-VQA using text-matching metrics.

</div>

<div id="table: DocVQA metrics" markdown="1">

|        | Res.    | METEOR | ROUGE-L | CIDEr |
|:-------|:--------|:------:|:-------:|:-----:|
| LLaVA  | $224^2$ |  3.8   |   4.8   |  6.3  |
| LLaVAR |         |  5.6   |   6.9   | 12.7  |
| LLaVA  | $336^2$ |  4.6   |   5.6   |  8.7  |
| LLaVAR |         |  8.6   |  10.0   | 21.5  |

Results on DocVQA using text-matching metrics.

</div>

The metric used for text-based VQA in the main paper is the standard
practice in VQA benchmarks [VQA](VQA). For STVQA and DocVQA,
previous works use ANLS (Average Normalized Levenshtein Similarity) as
the metric [STVQA](https://doi.org/10.1109/icdar.2019.00251), [mathew2020docvqa](https://arxiv.org/pdf/2007.00398), which calculates
the average normalized edit distance and only works for supervised
models trained to output short and precise answers. It works badly for
instruction-following models that usually output long sequences instead
of brief answers. For reference, we provide more text-matching metrics
(METEOR, `\citealp[]{banerjee-lavie-2005-meteor}`{=latex}, ROUGE-L,
`\citealp[]{lin-2004-rouge}`{=latex}, CIDEr,
`\citealp[]{vedantam2014cider}`{=latex}) to demonstrate the improvement
of our model (Table
<a href="#table: STVQA metrics" data-reference-type="ref"
data-reference="table: STVQA metrics">4</a>,
<a href="#table: textVQA metrics" data-reference-type="ref"
data-reference="table: textVQA metrics">5</a>,
<a href="#table: OCR-VQA metrics" data-reference-type="ref"
data-reference="table: OCR-VQA metrics">6</a>,
<a href="#table: DocVQA metrics" data-reference-type="ref"
data-reference="table: DocVQA metrics">7</a>), which works well except
for OCR-VQA. We assume these metrics are not valuable for OCR-VQA since
the ground truth answers are usually too short.

# E [Extra datasets]

#### Results on extra datasets

<div class="tabular" markdown="1">

lc S@ S@ S & **Res.** & **CT80** & **POIE** & **ChartQA**  
BLIP-2 `\citeyearpar{li2023blip2}`{=latex} $\dagger$ & & 80.9 & 2.5 &
7.2  
OpenFlamingo `\citeyearpar{anas_awadalla_2023_7733589}`{=latex}
$\dagger$ & & 67.7 & 2.1 & 9.1  
MiniGPT4 `\citeyearpar{zhu2023minigpt4}`{=latex} $\dagger$ & & 57.3 &
1.3 & 4.3  
LLaVA `\citeyearpar{liu2023visual}`{=latex} $\dagger$ & & 61.1 & 2.1 &
7.3  
mPLUG-Owl `\citeyearpar{ye2023mplugowl}`{=latex} $\dagger$ & & 81.9 &
3.3 & 9.5  
LLaVA $\ddagger$ & & 61.5 & 1.9 & 9.2  
LLaVAR & & 81.6 & 5.7 & 10.2  
LLaVA $\ddagger$ & & 64.9 & 2.5 & 10.2  
LLaVAR & & 83.0 & 8.7 & 13.5  

</div>

In Table <a href="#table: extra VQA result" data-reference-type="ref"
data-reference="table: extra VQA result">[table: extra VQA result]</a>,
we provide results on three extra datasets: CT80 (OCR,
`\citealp[]{risnumawan2014robust}`{=latex}), POIE (Information
Extraction, `\citealp[]{kuang2023visual}`{=latex}), and ChartQA
[masry2022chartqa](masry2022chartqa). We use the same VQA metric as other
text-based VQA datasets. We observe similar trends as the main paper
results: LLaVAR data significantly improves over the LLaVA baseline,
usually more considerably in a higher resolution.

# F [Finetune mPLUG]

#### Comparison with mPLUG-Owl

We find that LLaVAR usually performs similarly well with mPLUG-Owl in
the same $224^2$ resolution.We further clarify the setting differences
between mPLUG-Owl and ours: mPLUG-Owl is trained on 1000M+ text-image
pairs, while the original LLaVA is trained on about 0.6M text-image
pairs. Our model, LLaVAR, is trained on about 1M text-image pairs.
Within the same resolution, LLaVAR demonstrates a good performance with
decent data efficiency.

We presume that training on large-scale non-OCR data improves OCR
performance, as many of the captions in LAION datasets are equivalent to
incomplete OCR results (Texts in an online image will sometimes appear
in the captions). In the scale of our experiment, we observe similar
improvement that just training on captions of text-rich images can help
with text recognition capability: In Table
<a href="#table: VQA ablation" data-reference-type="ref"
data-reference="table: VQA ablation">[table: VQA ablation]</a>, variant
(4) is better than variant (1). However, training on captions only
(variant (4)) is not as good as training on OCR-based data (variant
(2)(6)), at least in the scale of our experiments.

#### Results of finetuning mPLUG-Owl

To further validate the effectiveness of our collected data, we provide
the results of finetuning mPLUG-Owl using our 16K GPT-4-based
instruction-following data in Table
<a href="#table: fine-tuned mPlug" data-reference-type="ref"
data-reference="table: fine-tuned mPlug">8</a>. Though the mPLUG-Owl
checkpoint is extensively trained on 1000M+ text-image pairs, we find
that our data can boost performance in most cases, demonstrating the
effectiveness of our data.

<div id="table: fine-tuned mPlug" markdown="1">

|                 | ST-VQA | OCR-VQA | TextVQA | DocVQA | CT80 | POIE | ChartQA |
|:----------------|:------:|:-------:|:-------:|:------:|:----:|:----:|:-------:|
| mPLUG-Owl       |  29.3  |  28.6   |  40.3   |  6.9   | 81.9 | 3.3  |   9.5   |
| mPLUG-Owl~ours~ |  29.6  |  31.2   |  40.8   |  7.0   | 84.7 | 3.7  |  10.2   |

Results (accuracy %) of finetuning mPLUG-Owl. mPLUG-Owl~ours~ denotes
mPLUG-Owl finetuned on our 16K GPT-4-based instruction-following data.

</div>

# G [ScienceQA section]

#### ScienceQA Results

Starting from our pretrained LLaVAR ($336^2$-based, without finetuning),
we also report the results of further finetuning on the ScienceQA
dataset [lu2022learn](https://arxiv.org/pdf/2209.09513) in Table
<a href="#table:scienceqa" data-reference-type="ref"
data-reference="table:scienceqa">[table:scienceqa]</a>, which is a
multimodal multi-choice QA dataset covering diverse domains. Our
motivation is that some images in this dataset contain text descriptions
and tables that require textual understanding within images. The LLaVAR
model finetuned on ScienceQA achieves an average accuracy of 91.42%,
better than LLaVA (90.92%), while the most considerable improvement
comes from natural science questions (+1.43%).

# H [High-Res section]

<figure id="fig:highres">
<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/highres-archv1.png" style="width:80.0%" />
<figcaption>Illustration of the dual visual encoder system. Given an
image, it is simultaneously processed by visual encoders <span
class="math inline"><em>V</em><sub>1</sub></span> and <span
class="math inline"><em>V</em><sub>2</sub></span>. <span
class="math inline"><em>V</em><sub>1</sub></span> features are
transformed by transformation matrix <span
class="math inline"><em>W</em></span> and directly used as input
embeddings to the language model. For <span
class="math inline"><em>V</em><sub>2</sub></span> features, they are
transformed by transformation matrix <span
class="math inline"><em>K</em></span> and <span
class="math inline"><em>V</em></span> and used as keys and values to
calculate the cross attention in every transformer layer (assume there
are <span class="math inline"><em>N</em></span> layers), which uses the
transformed hidden states (through <span
class="math inline"><em>Q</em></span>) from the self-attention module as
queries. For the language decoder <span
class="math inline"><em>D</em></span>, the input is image tokens
(<em><span class="math inline">&lt;</span>img<span
class="math inline">&gt;</span></em>) and instruction tokens (<em><span
class="math inline">&lt;</span>ins<span
class="math inline">&gt;</span></em>), while the target is response
tokens (<em><span class="math inline">&lt;</span>res<span
class="math inline">&gt;</span></em>).</figcaption>
</figure>

The original version of LLaVAR only supports up to $336^2$ resolution,
while our case study has also shown the threshold for the recognizable
font size. Both suggest the difficulty of processing real-world high-res
images without scaling and cutting. To this end, we test a dual visual
encoder system for the high-res variant of LLaVAR, where a high-res
visual encoder is added to work with the standard one. Ideally, the
standard visual encoder extracts general, high-level information, while
the high-res one specifically helps with detailed information.

#### Architecture

A high-res visual encoder usually outputs thousands of visual features.
Simply following LLaVA to feed the transformed visual features into the
context of the language decoder is impractical, as the maximum sequence
length of the language decoder is usually 2048/4096. To this end, we
propose to handle high-res visual features by cross-attention module and
standard visual features by feature transformation. We depict the
proposed system in Figure
<a href="#fig:highres" data-reference-type="ref"
data-reference="fig:highres">4</a>.

Specifically, given a standard visual encoder $V_1$, the extracted
features are transformed into the word embedding space of the language
decoder through a trainable projection matrix $W$. These transformed
features are then concatenated with the word embeddings to build the
input embeddings of the language decoder $D$. $$\begin{aligned}
\begin{split}
    & \mathrm{emb}(\langle \mathrm{img}_1\rangle), \cdots, \mathrm{emb}(\langle \mathrm{img}_m \rangle) = WV_1(I) \\
\mathrm{input}\_\mathrm{emb} = \mathbf{concat}([ & \mathrm{emb}(\langle \mathrm{img}_1\rangle), \cdots, \mathrm{emb}(\langle \mathrm{img}_m \rangle), \mathrm{emb}(\langle \mathrm{ins}_1\rangle), \cdots, \mathrm{emb}(\langle \mathrm{ins}_n \rangle)])
\end{split}
\end{aligned}$$

where $I$ is the input image, $V_1$ denotes extracting the grid features
before the last transformer layer.

At the same time, we use the high-res visual encoder $V_2$ to extract
high-res visual features, which are then transformed into keys/values as
the inputs of the cross-attention module in transformer layers. Given
$h^j$ as the hidden state before the cross-attention module in layer
$j$, $$\begin{aligned}
\begin{split}
    & \mathrm{CrossAttention}(h, V_2, I) = \mathrm{softmax}(\frac{Q^jh^j(K^jV_2(I))^T}{\sqrt{d}})V^jV_2(I)
\end{split}
\end{aligned}$$ where $Q^j, K^j, V^j$ denotes the query/key/value
projection matrix in the $j$-th transformers layer. In practice, there
is a pre-attention LayerNorm before calculating the attention and
another output projection matrix $O^j$ to project the aggregated values
back to the hidden space.

As the pretrained language decoder $D$ might only have self-attention
modules, we manually add another cross-attention module after the
original self-attention module in every transformer layer. Considering
the random initialization of cross-attention modules might hurt the
original language generation capability, we initialize the value
projection matrix $V^j$ as a zero matrix and the output projection
matrix $O^j$ as an identity matrix.

#### Implementation

We use `CLIP-ViT-L/14` as the standard visual encoder. For the
high-resolution encoder, we test two models: **(i)** `Pix2Struct-base`
[lee2022pix2struct](https://arxiv.org/pdf/2210.03347) is a visual encoder trained on
screenshot to HTML transformation. It supports up to 2048 patches with
size $16^2$, equivalent to $1024 * 512$. **(ii)** `ConcatCLIP` refers to
using 16 `CLIP-ViT-L/14` models to encode the $4 * 4$ grids of images
separately and then concatenate the extracted features together. In
other words, it supports $896^2$ resolution. We use Vicuna-7B as the
language decoder for the high-res version of LLaVAR.

#### Training

Only cross-attention modules and the projection matrix $W$ are trained
during pretraining, while visual encoders and the language decoder are
frozen. Cross-attention modules, the projection matrix $W$, and the
language decoder $D$ are trained during finetuning.

#### Data

To fully unlock the potential of the augmented visual encoder, we also
double the number of pretraining examples using the same criteria
mentioned in Section <a href="#para:Noisy Instruction-following Data"
data-reference-type="ref"
data-reference="para:Noisy Instruction-following Data">3.0.0.1</a>. This
corresponds to the variant (g) in Table
<a href="#table: ablation on encoder/image" data-reference-type="ref"
data-reference="table: ablation on encoder/image">[table: ablation on
encoder/image]</a>.

#### Discussion

<div class="tabular" markdown="1">

lS@ S@ S@ S & **ST-VQA** & **OCR-VQA** & **TextVQA** & **DocVQA**  
`Pix2Struct` + LLaVA & 21.9 & 11.8 & 28.7 & 4.4  
`Pix2Struct` + LLaVAR & 35.8 & 30.7 & 45.6 & 15.3  
`ConcatCLIP` + LLaVA & 23.1 & 14.2 & 30.5 & 5.1  
`ConcatCLIP` + LLaVAR & 42.1 & 30.8 & 52.1 & 18.5  

</div>

We report the performance of augmented architecture, using either LLaVA
or LLaVAR data in Table
<a href="#table: pix2struct and clip" data-reference-type="ref"
data-reference="table: pix2struct and clip">[table: pix2struct and
clip]</a>. By comparing the relative improvement in Table
<a href="#table: VQA result" data-reference-type="ref"
data-reference="table: VQA result">[table: VQA result]</a> and
<a href="#table: pix2struct and clip" data-reference-type="ref"
data-reference="table: pix2struct and clip">[table: pix2struct and
clip]</a>, we find that higher-resolution models benefit more from our
collected data, suggesting our data is underutilized in the original
LLaVA architecture.

# I [OCR error analysis]

#### The impact of OCR spelling errors

<div id="table: OCR errors" markdown="1">

|        | **Res.** | **Correct %** | **Partially Correct%** |
|:-------|:--------:|:-------------:|:----------------------:|
| LLaVA  | $224^2$  |     1.6%      |          8.7%          |
| LLaVAR |          |     6.8%      |         22.8%          |
| LLaVA  | $336^2$  |     2.2%      |         11.2%          |
| LLaVAR |          |     9.0%      |         26.8%          |

Statistics of correct answers and partially correct answers on OCR-VQA.

</div>

We study such OCR errors by studying 1673 examples from OCR-VQA, which
have ground truth answers with more than ten characters. We (i) define
“correct” as the ground truth answers that are exactly in the
predictions, and (ii) define “partially correct” as there exists a
substring in the prediction that has high enough similarity with the
ground truth but not the same. Specifically, we look at all substrings
with the same length of the ground truth in the prediction to calculate
ANLS (Average Normalized Levenshtein Similarity) and regard the
prediction as “partially correct” if the highest ANLS is greater or
equal to 0.5 but smaller than 1.

We find that many predictions can be considered partially correct,
indicating the actual performance of tested models is better than the
reported accuracy numbers. However, the percentage of partially correct
predictions is highly correlated with the percentage of correct
predictions. Therefore, we believe that the current metrics can
effectively compare the performance of different models.

# J [ablation study on LLaVA benchmark]

#### Ablation Study on Instruction-following Evaluation

<div id="table: ablation study on LLaVA benchmark" markdown="1">

|                     | **Conversation** | **Detail** | **Complex** |
|:--------------------|:----------------:|:----------:|:-----------:|
| LLaVA               |       83.6       |    78.1    |    95.2     |
| LLaVA + R~pretrain~ |       86.7       |    79.3    |    95.1     |
| LLaVA + R~finetune~ |       79.0       |    79.4    |    98.0     |
| LLaVAR              |       84.4       |    78.9    |    96.6     |

Ablation Study on GPT-4-based Instruction-following Evaluation from
LLaVA [liu2023visual](https://arxiv.org/pdf/2304.08485).

</div>

To provide a more detailed understanding of LLaVAR’s data, we provide
the ablation study results on 90 instruction-following questions from
LLaVA [liu2023visual](https://arxiv.org/pdf/2304.08485). We find that including
pretraining data improves the conversation capability, probably because
longer training data leads to generating longer responses (Table
<a href="#data stats" data-reference-type="ref"
data-reference="data stats">1</a>), which is favored by GPT-4
[xu2023baize](https://arxiv.org/pdf/2304.01196). On the other hand, including only
finetuning data hurts the conversation capability but increases complex
reasoning. Combining pretraining and finetuning data improves the
trade-off between conversation and complex reasoning. Such results prove
that incorporating our data will at least not harm the performance of
interacting with natural images.

<div class="tabular" markdown="1">

lm8.8cmm1cm & Relative Score  

<div id="table:read" markdown="1">

|             |
|:------------|
| GPT-4       |
| (Text-Only) |

An example for our collected evaluation data, corresponding to the
“**Read**” dimension of GPT-4-based instruction-following evaluation.
The corresponding image is Figure
<a href="#fig:read" data-reference-type="ref"
data-reference="fig:read">5</a> in Appendix.

</div>

& From the title and image, it can be inferred that "Bo’s Lasting
Lessons" is a book about leadership lessons taught by a legendary coach,
likely in the context of sports. The image of men running suggests that
the book might be focused on a specific sport, such as football or track
and field. The target audience could be individuals interested in
coaching, sports, or leadership in general, looking to learn from the
experiences and teachings of a successful coach. & -  
LLaVA & The book "Bo’s Lasting Lessons: The Timeless Teachings of Bo
Schembechler" is written by Bo Schembechler and Jim Harshaw. Given the
title and the image on the cover, which shows a football coach running
across the field with his players in pursuit, it can be inferred that
the content of the book is likely focused on football and the
strategies, techniques, or principles that can help players improve
their performance and achieve success in the sport. The potential target
audience for this book could be football players, coaches, and fans who
are interested in learning more about the game and the experiences and
insights of a seasoned coach like Bo Schembechler. & 75%  
LLaVAR & "Bo’s Lasting Lessons" appears to be a book that shares the
legacy coach Bo Schembechler’s teachings on leadership and timeless
fundamentals. The image on the cover, featuring a football coach leading
his team, further supports this inference. The book might be targeted
towards sports enthusiasts, particularly fans of football, and people
interested in learning about leadership principles and fundamentals from
a renowned coach’s perspective. The audiobook, read by John H. Mayer and
featuring a foreword and conversation with John U. Bacon, suggests that
it could also appeal to audiences who prefer listening to books or want
to gain more insights from the author and other experts. & 129%  

</div>

<figure id="fig:read">
<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/read.png" />
<figcaption>An example for the Read dimension of GPT-4-based
instruction-following evaluation.</figcaption>
</figure>

<figure id="fig:Demo">
<p><img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/demo1.png" alt="image" /> <img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/demo2.png"
alt="image" /> <img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/demo3.png" alt="image" /></p>
<figcaption>Transferred instruction-following capability of
LLaVAR.</figcaption>
</figure>

<div class="figure*" markdown="1">

</div>

<figure id="fig:instruction1">
<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/vb.png" />
<figcaption>Visualization of collected instructions.</figcaption>
</figure>

<figure id="fig:instruction2">
<img src="/papers/text_rich_doc/arXiv-2306.17107v2_md/pron.png" />
<figcaption>Visualization of collected instructions.</figcaption>
</figure>

[^1]: Collaborations through Adobe University Gift Program.

[^2]: In this work, we use the phrase “text-rich images” to describe
    images with text in them, such as posters and book covers. In
    contrast, we refer to images without text as “natural images”.

[^3]: <https://huggingface.co/datasets/laion/laion-high-resolution>

[^4]: Both probabilities are from the LAION dataset’s metadata.

[^5]: <https://github.com/PaddlePaddle/PaddleOCR>

[^6]: <https://github.com/haotian-liu/LLaVA>

[^7]: Since the metric only consider the recall, it might favor variant
    (2)(4)(5) due to their longer outputs.

[^8]: <https://www.imdb.com/title/tt7405458/>