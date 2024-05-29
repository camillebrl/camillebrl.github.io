# Introduction [intro]

Recently, research into vision dialogue
robots [BLIP2](http://arxiv.org/pdf/2301.12597v3), [Flamingo](http://arxiv.org/pdf/2205.07065v1), [llava](http://arxiv.org/pdf/2402.11690v1), [minigpt4](http://arxiv.org/pdf/2402.17510v1), [InstructGPT](http://arxiv.org/pdf/2302.05206v1) has
been gaining significant traction. These human-like models, mainly
relying on two components (large language models
(LLMs) [GPT-2](http://arxiv.org/pdf/2203.12926v1), [GPT3](http://arxiv.org/pdf/2112.07522v2), [OPT](http://arxiv.org/pdf/2405.04515v2), [llama](http://arxiv.org/pdf/2402.08075v1), [GPT4](https://arxiv.org/pdf/arXiv preprint arXiv:2303.08774) and vision vocabulary
networks), can not only converse based on user’s input image but also
perform well on simple downstream tasks, such as
VQA [COCO](None), [TextVQA](http://arxiv.org/pdf/1811.11903v1), Image
caption [coco_text](http://arxiv.org/pdf/1707.08831v1), OCR [OCRVQA](http://arxiv.org/pdf/2010.02582v1), and so
on. Hence, it is undeniable that large vision-language models (LVLMs)
are driving the AI community towards the direction of artificial general
intelligence (AGI).

Popular GPT-4 [GPT4](https://arxiv.org/pdf/arXiv preprint arXiv:2303.08774)-like LVLMs, *e.g.*,
BLIP2 [BLIP2](http://arxiv.org/pdf/2301.12597v3),
MiniGPT4 [minigpt4](http://arxiv.org/pdf/2402.17510v1),LLaVA [llava](http://arxiv.org/pdf/2402.11690v1),
Qwen-VL [Qwen-VL](http://arxiv.org/pdf/2308.12966v3), and
*etc*. [dong2023dreamllm](http://arxiv.org/pdf/2309.11499v2), [zhao2023chatspot](http://arxiv.org/pdf/2307.09474v1), [yu2023merlin](http://arxiv.org/pdf/2312.00589v1)
enjoy a stunning performance in multiple aspects with their own
programming paradigm: Based on an LLM [OPT](http://arxiv.org/pdf/2405.04515v2), [T5](http://arxiv.org/pdf/1910.10683v4), BLIP-2
proposes the Q-former, a BERT [Bert](http://arxiv.org/pdf/1810.04805v2) like network as a
vision input embedding layer, aiming to align the image tokens to a text
special. Inherited the structure of BLIP-2, MiniGPT-4 introduces 3500
high-quality image-text pairs as self-supervised fine-tuning (SFT) data,
allowing it can “talk” like GPT-4. Unlike BLIP-2, LLaVA utilizes a
linear layer as the vision embedding layer, which is similar with the
text input embedding layer in the text tokenizer, ensuring the
consistency in the structure of image and text branches. For Qwen-VL, it
utilizes a cross-attention layer to sample and align the image tokens,
making the model can accept larger input resolution. Although the above
LVLMs’ vision input embedding networks are variable (*e.g.*, MLP,
Qformer, Perceiver [Flamingo](http://arxiv.org/pdf/2205.07065v1)), their vision vocabulary
is almost identical (a CLIP-based [radford2021learning](http://arxiv.org/pdf/2404.19696v1)
VIT) which we argue maybe a bottle-neck.

<figure id="fig1">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vary_1.png" style="width:100.0%" />
<figcaption>Previous method <em>vs.</em> Vary: Unlike other models that
use a ready-made vision vocabulary, the processes of Vary can be divided
into two stages: the generation and fusion of vision vocabulary. In the
first stage, we use a “vocabulary network” along with a tiny
decoder-only network to produce a powerful new vision vocabulary via
auto-regression. In the second stage, we fuse the vision vocabulary with
the original one to provide new features for the LVLMs
efficiently.</figcaption>
</figure>

It is recognized that CLIP-VIT is a tremendous general vision
vocabulary, which is trained via contrastive learning upon more than
400M [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) image-text pairs, covering most
natural images and vision tasks. However, for some special scenarios,
*e.g.*, high-resolution perception, Non-English OCR, Document/Chart
understanding, and so on, the CLIP-VIT may regard them as a “foreign
language”, leading to inefficient tokenizing, *i.e.*, difficulty in
encoding all vision information into a fixed number (usually 256) of
tokens. Although mPlug-Owl [ye2023mplug](http://arxiv.org/pdf/2403.14252v1) and Qwen-VL
alleviate the above issues by unfreeze its vision vocabulary network (a
CLIP-L or CLIP-G), we argue that such manner may not be reasonable due
to three aspects: 1) it may overwrite the knowledge of the original
vocabulary; 2) the training efficiency of updating a vision vocabulary
upon a relative large LLM (7B) is low; 3) it can not allow the vision
vocabulary network to “see” an image multiple times (train a dataset
with multiple epochs) due to the strong memory ability of LLMs.
Therefore, a natural question is: *Is there a strategy that can simplify
and effectively intensify the visual vocabulary?*

In this paper, we propose Vary, an efficient and user-friendly approach,
to answer the above question. Vary is inspired by the text vocabulary
expansion manner in vanilla LLMs [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), *i.e.*, when
transferring an English LLM to another foreign language, such as
Chinese, it’s necessary to expand the text vocabulary to lift the
encoding efficiency and model performance under the new language.
Intuitively, for the vision branch, if we feed the “foreign language”
image to the model, we also need to scale up the vision vocabulary. In
Vary, the process of vocabulary scaling up can be divided into two
steps: 1) generate a new vision vocabulary that can make up the old one
(CLIP); 2) integrate the new and old vocabularies. As shown in
Figure <a href="#fig1" data-reference-type="ref" data-reference="fig1">1</a>,
we build a small-size pipeline which is consisting of a vocabulary
network and a tiny decoder-only transformer in the first step to train
the vocabulary model via predicting the next token. It is worth noting
that the autoregressive-based process of generating a vocabulary is
perhaps more suitable for dense perception tasks than that based on
contrastive learning like CLIP. On the one hand, the next-token way can
allow the vision vocabulary to compress longer texts. On the other hand,
the data formats that can be used in this manner are more diverse, such
as VQA [STVQA](http://arxiv.org/pdf/2309.17133v2), [DocVQA](http://arxiv.org/pdf/2111.05547v1) data with prompt. After preparing
the new vision vocabulary, we add it to the vanilla LVLMs to introduce
new features. In this process, we freeze both the new and old
vocabularies networks to avoid the visual knowledge being overwritten.

Afterward scaling up the vision vocabulary, our LVLM can achieve more
fine-grained vision perception, such as document-level Chinese/English
OCR, book image to markdown or *LaTeX*, Chinese/English chart
understanding, and so on, while ensuring its original capabilities
(conversation, VQA, caption, *etc*.). Besides, we provide methods for
producing synthetic data and validate its importance in document/chart
understanding. More importantly, Vary is a useful strategy to strengthen
the visual vocabulary of LVLMs, which can be utilized at arbitrary
downstream visual tasks that CLIP is not good at. In addition to the
document and chart parsing mentioned in this paper, we believe that Vary
still enjoys more fine-grained tasks and we appeal to researchers to
rethink the design ideas of LVLMs from the perspective of visual
vocabulary construction.

# Related Works

## Large Language Models

Over the past year, significant attention has been drawn to large
language models (LLMs) in the fields of both natural language processing
(NLP) and computer vision (CV). This heightened attention stems from
LLMs’ outstanding performance in diverse aspects, especially the
powerful world knowledge base and universal capabilities. Current LLMs
enjoy a unified transformer architecture which is exemplified by
BERT [Bert](http://arxiv.org/pdf/1810.04805v2), GPT-2 [GPT-2](http://arxiv.org/pdf/2203.12926v1),
T5 [T5](http://arxiv.org/pdf/1910.10683v4), *etc*. Subsequently, researchers have uncovered
the concept of an "emergent ability" [wei2022emergent](http://arxiv.org/pdf/2403.15796v2) in
LLMs. This implies that as language model sizes reach a certain
threshold, there may be a qualitative leap in their capabilities.
Furthermore, InstructGPT [InstructGPT](http://arxiv.org/pdf/2302.05206v1) and
ChatGPT [ChatGPT](https://openai.com/blog/chatgpt/) find that Reinforcement Learning with
Human Feedback (RLHF) [christiano2017deep](http://arxiv.org/pdf/2007.12904v2) can further
lift the performance of the "talk robot”. Motivated by the tremendous
success of the GPT series, a multitude of other open-source LLMs have
emerged, including OPT [OPT](http://arxiv.org/pdf/2405.04515v2),
LLaMA [llama](http://arxiv.org/pdf/2402.08075v1), GLM [GLM](http://arxiv.org/pdf/2004.13270v1), and so on.
Building upon these openly available LLMs, numerous tailored fine-tuned
models have been introduced to develop LLMs for diverse applications,
especially LLaMA-driven models,*e.g.*, Alphaca [alpaca](https://github.com/tatsu-lab/stanford_alpaca),
Vicuna [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), which have become the de-facto component
for a Large Vision-Language Model (LVLM).

## LLM-based Large Vision-Language Models

LLM’s robust zero-shot capabilities and logical reasoning make it play
the central controller role within an LVLM. There are two primary
pipeline styles: plugin-based and end-to-end model. Plugin-based
methods [VisualChatGPT](http://arxiv.org/pdf/2303.04671v1), [MMREACT](http://arxiv.org/pdf/2303.11381v1), [Hugginggpt](http://arxiv.org/pdf/2303.17580v4), [taskmatrix](http://arxiv.org/pdf/2303.16434v1), [yang2023gpt4tools](http://arxiv.org/pdf/2401.15328v2)
typically regard LLMs as an agent to invoke various plugins from other
foundational or expert models, executing specific functions in response
to human instructions. While such methods offer versatility, they have
limitations in terms of plugin invocation efficiency and performance.
Conversely, end-to-end LVLMs usually rely on a single large multimodal
model to facilitate interactions. Following this approach,
Flamingo [Flamingo](http://arxiv.org/pdf/2205.07065v1) introduces a gated cross-attention
mechanism trained on billions of image-text pairs to align vision and
language modalities, demonstrating strong performance in few-shot
learning. BLIP-2 [BLIP2](http://arxiv.org/pdf/2301.12597v3) introduces Q-Former to enhance
the alignment of visual features with the language space. More recently,
LLaVA [llava](http://arxiv.org/pdf/2402.11690v1) proposes using a simple linear layer to
replace Q-Former and designed a two-stage instruction-tuning procedure.

Despite the remarkable performance of existing methods, they are
confined to the same and limited vision vocabulary –
CLIP-VIT [radford2021learning](http://arxiv.org/pdf/2404.19696v1). For an LVLM, CLIP-VIT is
a tremendous general vision vocabulary that is trained via contrastive
learning upon million-level image-texts pairs, which can cover most
nature images and vision tasks, *e.g.*, VQA, Caption, Easy English OCR.
However, some images under special scenarios, *e.g.*, high-resolution
image, Non-English OCR, Document/Chart understanding, and so on, will
still be regarded as a “foreign language” by CLIP-VIT, leading to vision
out-of-vocabulary problem, which will in turn become a bottleneck for
LVLMs.

# Method [methods]

## Architecture

<figure id="fig2">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vary_2.png" style="width:100.0%" />
<figcaption>Overview of the Vary. There are two types of Vary form:
Vary-tiny and Vary-base. Vary-tiny is mainly focused on generating a new
vision vocabulary while Vary-base is our new LVLM aiming to handle
various visual tasks based on the new vision vocabulary. </figcaption>
</figure>

Vary enjoys two conformations: Vary-tiny and Vary-base, as shown in
Figure <a href="#fig2" data-reference-type="ref" data-reference="fig2">2</a>.
We devise the Vary-tiny to “write” a new vision vocabulary and the
Vary-base to make use of the new vocabulary. Specifically, Vary-tiny is
mainly composed of a vocabulary network and a tiny
OPT-125M [OPT](http://arxiv.org/pdf/2405.04515v2). Between the two modules, we add a linear
layer to align the channel dimensions. There is no text input branch in
Vary-tiny due to it is a primary focus on fine-grained perception. We
hope the new vision vocabulary network can excel in processing
artificial images, *i.e.*, documents, and charts, to compensate for
CLIP’s shortcomings. At the same time, we also expect that it will not
be a noise for CLIP when tokenizing natural images. Accordingly, during
generating, we feed the manual document and chart data as positive
samples and natural images as negatives to train Vary-tiny. After
completing the above process, we extract the vocabulary network and add
it to a large model to build the Vary-base. As shown in the lower half
of
Figure <a href="#fig2" data-reference-type="ref" data-reference="fig2">2</a>,
the new and old vocabulary networks enjoy independent input embedding
layers and are integrated before the LLM. In such a stage, we freeze
both weights of new and old vision vocabulary networks and unfreeze the
weights of other modules.

## Towards Generating a New Vision Vocabulary

### The new vocabulary network

We use the SAM [kirillov2023segment](http://arxiv.org/pdf/2305.01275v1) pretrained
ViTDet [li2022exploring](http://arxiv.org/pdf/2203.16527v2) image encoder (base scale) as
the main part of the new vocabulary network of Vary. Due to the input
resolution of the SAM-base is (1024$\times$`<!-- -->`{=html}1024) while
the output stride is 16, the feature shape of the last layer is
(64$\times$`<!-- -->`{=html}64$\times$`<!-- -->`{=html}256 for
H$\times$W$\times$C) that can not be aligned to the output of CLIP-L
(256$\times$`<!-- -->`{=html}1024 for N$\times$C). Hence, we add two
convolution layers, which we found is a good token merging unit, behind
the last layer of the SAM initialized network, as shown in
Figure <a href="#fig3" data-reference-type="ref"
data-reference="fig3">[fig3]</a>. The first convolution layer possesses
a kernel size of 3, aiming to transfer the feature shape to
32$\times$`<!-- -->`{=html}32$\times$`<!-- -->`{=html}512. The setting
of the second conv layer is the same as the first one, which can further
convert the output shape to
16$\times$`<!-- -->`{=html}16$\times$`<!-- -->`{=html}1024. After that,
we flattened the output feature to 256$\times$`<!-- -->`{=html}1024 to
align the image token shape of CLIP-VIT.

<div class="wrapfigure" markdown="1">

r0.5 <embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vary-3.png" style="width:50.0%" />

</div>

### Data engine in the generating phrase [data1]

**Documnet data.** We select the high-resolution document image-text
pairs as the main positive dataset used for the new vision vocabulary
pretrain due to the dense OCR can effectively validate the fine-grained
image perception ability of the model. To our knowledge, there is no
publicly available dataset of English and Chinese documents, so we
create our own. We first collect pdf-style documents from open-access
articles on arXiv and CC-MAIN-2021-31-PDF-UNTRUNCATED for the English
part and collect from e-books on the Internet for the Chinese part. Then
we use *fitz* of PyMuPDF to extract the text information in each pdf
page and convert each page into a PNG image via *pdf2image* at the same
time. During this process, we construct 1M Chinese and 1M English
document image-text pairs for training.

**Chart data.** We find current LVLMs are not good at chart
understanding, especially Chinese charts, so we choose it as another
main knowledge that needs to be “written” into the new vocabulary. For
chart image-text pair, we all follow the rendering way. We select both
the *matplotlib* and *pyecharts* as the rendering tools. For
matplotlib-style chart, we built 250k in both Chinese and English. While
for pyecharts, we build 500k for both Chinese and English. Besides, we
convert the text ground truth of each chart to a python-dict form. The
texts used in the chart, *e.g.*, title, x-axis, and y-axis, are randomly
selected from the Natural Language Processing (NLP) corpus downloaded
from the Internet.

**Negative natural image.** For natural image data that CLIP-VIT is good
at, we need to ensure that the newly introduced vocabulary does not
cause noise. Consequently, we construct negative natural image-text
pairs to enable the new vocabulary network to encode correctly when
seeing natural images. We extract 120k images in the
COCO [COCO](None) dataset with each image corresponding to a
text. The text part is randomly selected from follows sentences: "It’s
an image of nature"; "Here’s a nature picture"; "It’s a nature photo";
"This is a natural image"; "That’s a shot from nature".

### Input format

We train all parameters of the Vary-tiny with image-text pairs by
autoregression. The input format follows popular
LVLMs [KOSMOS](http://arxiv.org/pdf/2302.14045v2), *i.e*, the image tokens are packed with
text tokens in the form of a prefix. Specifically, we use two special
tokens "\<img\>" and "\</img\>" to indicate the position of the image
tokens as the input of an interpolated OPT-125M (4096 tokens). During
training, the output of Vary-tiny is only text, and "\</s\>" is regarded
as the *eos* token.

<figure id="fig4">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vary_4.png" style="width:100.0%" />
<figcaption>Visualization of synthetic data. We use <em>pdflatex</em> to
render documents and utilize <em>pyecharts/matplotlib</em> to render
charts. Document data obtains Chinese/English texts, formulas, and
tables. Chart data includes Chinese/English bar, line, pie, and
composite styles.</figcaption>
</figure>

## Towards Scaling Up the Vision Vocabulary

### The structure of Vary-base

After completing the training of the vocabulary network, we introduce it
to our LVLM – Vary-base. Specifically, we parallelize the new vision
vocabulary with the original CLIP-VIT. Both two vision vocabularies
enjoy an individual input embedding layer, *i.e.*, a simple linear. As
shown in
Figure <a href="#fig2" data-reference-type="ref" data-reference="fig2">2</a>,
the input channel of the linear is 1024 and the output is 2048, ensuring
the channel of image tokens after concatenating is 4096, which exactly
aligns the input of LLM (Qwen-7B [qwen](http://arxiv.org/pdf/2309.16609v1) or
Vicuna-7B [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)).

### Data engine in the scaling up phrase

***LaTeX* rendering document**. Except for the collecting document data
in Section <a href="#data1" data-reference-type="ref"
data-reference="data1">3.2.2</a>, we also need data to enjoy some
format, *e.g.*, supporting formula, and table. To this end, we create
document data through *LaTeX* rendering. Firstly, we collected some
*.tex* source files on arxiv, and then extracted tables, mathematical
formulas, and plain texts using regular expressions. Finally, we
re-render these contents with the new template we prepared by
*pdflatex*. We collect 10+ templates to perform batch rendering.
Besides, we transfer the text ground truth of each document page to a
*mathpix* markdown style to unify the format. By this construction
process, we acquired 0.5 million English pages and 0.4 million Chinese
pages. Some samples are shown in
Figure <a href="#fig4" data-reference-type="ref" data-reference="fig4">3</a>.

**Semantic association chart rendering**. In
Section <a href="#data1" data-reference-type="ref"
data-reference="data1">3.2.2</a>, we batch render chart data to train
the new vocabulary network. However, the texts (title, x-axis values,
and y-axis values) in those rendered charts suffer low correlation
because they are randomly generated. This issue is not a problem in the
vocabulary-generating process as we only hope that the new vocabulary
can efficiently compress visual information. However, in the training
stage of the Vary-base, due to unfreezing the LLM, we hope to use higher
quality (strongly correlated content) data for training. Therefore, we
use GPT-4 [GPT4](https://arxiv.org/pdf/arXiv preprint arXiv:2303.08774) to generate some charts using relevant
corpus and then we utilize the high-quality corpus to addition render
200k chart data for the Vary-base training.

**General data**. The processes of training Vary-base follows popular
LVLMs, *e.g.*, LLaVA [llava](http://arxiv.org/pdf/2402.11690v1), including the pretrain and
SFT phases. Different from the LLaVA, we freeze all the vocabulary
networks and unfreeze both the input embedding layer and LLM, which is
more like the pretrain setting of a pure LLM. We use natural image-text
pair data to introduce the general concepts to the Vary-base. The
image-text pairs are randomly extracted from
LAION-COCO [schuhmann2021laion](http://arxiv.org/pdf/2111.02114v1) with the amount of 4
million. In the SFT stage, we use the LLaVA-80k or
LLaVA-CC665k [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) along with the train
set of DocVQA [DocVQA](http://arxiv.org/pdf/2111.05547v1) and
ChartQA [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1) as the fine-tuning dataset.

### Conversation format

When we use the Vicuna-7B as our LLM, the conversation format follows
the Vicuna v1 [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), *i.e.*, USER:
\<img\>"\<image\>"\</img\> "texts input" ASSITANT: "texts output"
\</s\>. Due to the low efficiency in the text vocabulary of Vicuna to
process Chinese, we choose Qwen-7B [qwen-chat](https://github.com/QwenLM/Qwen-7B) as the LLM
for Chinese processing. When we use the Qwen-7B, we design the
conversation style following the
LLaVA-MPT [team2023introducing](http://arxiv.org/pdf/2311.16429v1), [llava](http://arxiv.org/pdf/2402.11690v1), which can be
described as: \<\|im_start\|\>user: \<img\>"\<image\>"\</img\> "texts
input"\<\|im_end\|\> \<\|im_start\|\>assistant: "texts output"
\<\|im_end\|\>.

# Experiments [exp]

## Datasets and Evaluation Metrics

We evaluate the proposed Vary on multiple datasets, including 1) a
document-level OCR test set we created to explore the performance of
dense visual perception; 2) DocVQA [DocVQA](http://arxiv.org/pdf/2111.05547v1) and
ChartQA [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1) to test the improvement on
downstream tasks; 3) MMVet [yu2023mm](http://arxiv.org/pdf/2402.15896v1) to monitor changes
in the general performance of the model. Our own document test set
contains pure OCR and markdown conversion tasks. In a pure OCR task, the
test split includes 100 pages in both Chinese and English, which are
randomly extracted from arxiv and ebook. In the markdown conversion
task, the test set obtains 200 pages, of which 100 pages contain tables
and another 100 pages have mathematical formulas.

We report Normalized Edit
Distance [levenshtein1966binary](http://arxiv.org/pdf/2007.09075v4), [blecher2023nougat](http://arxiv.org/pdf/2308.13418v1) and
F1-score along with the precision and recall for document parsing. For
DocVQA, ChartQA, and MMVet, we use their vanilla metrics for a fair
comparison with other LVLMs.

## Implementation Details

During the vision vocabulary generating process, we optimize all
parameters of Vary-tiny with a batch size of 512 and train the model for
3 epochs. We utilize the AdamW [AdamW](http://arxiv.org/pdf/2311.11446v2) optimizer and a
cosine annealing scheduler [loshchilov2016sgdr](http://arxiv.org/pdf/1608.03983v5) along
with the learning rate of 5e-5 to train Vary-tiny.

In the training stage of the Vary-base, we freeze the weights of both
new and vanilla (CLIP-L) vision vocabulary networks and optimize the
parameters of input embedding layers and LLM. The initial learning rate
is 5e-5 in pretrain while 1e-5 in SFT. Both the pretrain and SFT enjoy a
batch size of 256 and an epoch of 1. Other settings are the same as
Vary-tiny.

<div class="table*" markdown="1">

| **Method** | **Forms** | **Pure Document OCR** |  | **Markdown Format Conversion** |  |  |
|:---|:---|:--:|:--:|:--:|:--:|:--:|
| 3-4 (rl)5-7 |  | Chinese | English | Formula | Table | Average |
| Nougat [blecher2023nougat](http://arxiv.org/pdf/2308.13418v1) | Edit Distance $\downarrow$ | – | 0.126 | 0.154 | 0.335 | 0.245 |
|  | F1-score $\uparrow$ | – | **89.91** | 83.97 | 75.97 | 79.97 |
|  | Prediction $\uparrow$ | – | 89.12 | 82.47 | 75.21 | 78.84 |
|  | Recall $\uparrow$ | – | **90.71** | **85.53** | **76.74** | 81.14 |
| Vary-tiny | Edit Distance $\downarrow$ | 0.266 | 0.197 | – | – | – |
|  | F1-score $\uparrow$ | 86.00 | 84.25 | – | – | – |
|  | Prediction $\uparrow$ | 86.14 | 89.38 | – | – | – |
|  | Recall $\uparrow$ | 85.86 | 79.67 | – | – | – |
| Vary-base | Edit Distance $\downarrow$ | 0.174 | **0.106** | **0.082** | **0.280** | 0.181 |
|  | F1-score $\uparrow$ | 87.32 | 88.24 | **85.94** | **76.26** | 81.10 |
|  | Prediction $\uparrow$ | 86.59 | **90.08** | **87.06** | **76.81** | 81.94 |
|  | Recall $\uparrow$ | 88.06 | 86.47 | 84.84 | 75.71 | 80.28 |

</div>

## Fine-grained Perception Performance

We measure the fine-grained perception performance of Vary through the
dense text recognition ability. As shown in
Table <a href="#tab:1" data-reference-type="ref"
data-reference="tab:1">[tab:1]</a>, Vary-tiny gathers both Chinese and
English dense OCR ability by the process of vision vocabulary
generating. Specifically, it achieves 0.266 and 0.197 edit distance for
Chinese and English documents (plain texts) OCR respectively, proving
the new vision vocabulary enjoys good fine-grained text encoding
capacity. For Vary-base, it can achieve an on-par performance with
nougat [blecher2023nougat](http://arxiv.org/pdf/2308.13418v1) (a special document parsing
model) on English plain text documents. Besides, with different prompts
(*e.g.*, Convert the image to markdown format.), Vary-base can realize
the document image-markdown format conversion. It is worth noting that
in such a task, Vary-base (with 0.181 edict distance and 81.10% F1 on
math and table average) is better than nougat (with 0.245 edict distance
and 79.97% F1 on average) to some extent, which may be due to the super
strong text correction ability of the 7B LLM (Qwen). All the above
results indicate that by scaling up the vision vocabulary, the new LVLM
can lift its fine-grained perception performance.

<div id="tab:2" markdown="1">

| **Method** | DocVQA |  | ChartQA |  |  |
|:---|:--:|:--:|:--:|:--:|:--:|
| 2-3 (rl)4-6 | **val** | **test** | **human** | **augmented** | **Average** |
| Dessurt [davis2022end](http://arxiv.org/pdf/2203.16618v3) | 46.5 | 63.2 | \- | \- | \- |
| Donut [kim2022ocr](http://arxiv.org/pdf/2305.09520v1) | \- | 67.5 | \- | \- | 41.8 |
| Pix2Sturct [lee2023pix2struct](http://arxiv.org/pdf/2210.03347v2) | \- | 72.1 | 30.5 | 81.6 | 56.0 |
| mPLUG-DocOwl [ye2023mplug](http://arxiv.org/pdf/2403.14252v1) | \- | 62.2 | \- | \- | 57.4 |
| Matcha [liu2022matcha](http://arxiv.org/pdf/2212.09662v2) | \- | \- | 38.2 | <u>90.2</u> | 64.2 |
| Qwen-VL [qwen](http://arxiv.org/pdf/2309.16609v1) | \- | 65.1 | \- | \- | 65.7 |
| Vary-base (80k) | <u>78.2</u> | 76.3 | 43.2 | 87.3 | 65.3 |
| Vary-base (665k) | 78.1 | 76.3 | <u>43.8</u> | 88.3 | <u>66.1</u> |

Comparison with popular methods on DocVQA and ChartQA. 80k represents
that the SFT data is LLaVA-80k while 665k is the LLaVA-CC665k. The
metric of DocVQA is ANLS while the ChartQA is relaxed accuracy following
their vanilla papers.

</div>

<div id="tab:3" markdown="1">

| **Method** | MM-Vet |  |  |  |  |  |  |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 2-8 | **Rec** | **OCR** | **Know** | **Gen** | **Spat** | **Math** | **Total** |
| BLIP-2 [BLIP2](http://arxiv.org/pdf/2301.12597v3) | 27.5 | 11.1 | 11.8 | 7.0 | 16.2 | 5.8 | 22.4 |
| LLaVA-7B [llava](http://arxiv.org/pdf/2402.11690v1) | 28.0 | 17.1 | 16.3 | 18.9 | 21.2 | <u>11.5</u> | 23.8 |
| MiniGPT-4 [minigpt4](http://arxiv.org/pdf/2402.17510v1) | 29.9 | 16.1 | 20.4 | 22.1 | 22.2 | 3.8 | 24.4 |
| Otter [li2023otter](http://arxiv.org/pdf/2311.00233v2) | 27.3 | 17.8 | 14.2 | 13.8 | 24.4 | 3.8 | 24.7 |
| OpenFlamingo [Flamingo](http://arxiv.org/pdf/2205.07065v1) | 28.7 | 16.7 | 16.4 | 13.1 | 21.0 | 7.7 | 24.8 |
| LLaVA-13B [llava](http://arxiv.org/pdf/2402.11690v1) | <u>39.2</u> | 22.7 | <u>26.5</u> | <u>29.3</u> | 29.6 | 7.7 | 32.9 |
| LLaVA1.5-7B [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) | \- | \- | \- | \- | \- | \- | 30.5 |
| Vary-base (vicuna7B) (665k) | 38.7 | 22.0 | 23.6 | 24.1 | 29.6 | 7.7 | 32.9 |
| Vary-base (qwen7B) (80k) | 38.9 | <u>30.1</u> | 22.4 | 21.7 | <u>34.3</u> | 7.7 | <u>36.2</u> |

Comparison with popular methods on MMVet. The abbreviations represent:
Rec: Recognition; Know: Knowledge; Gen: Language generation; Spat:
Spatial awareness.

</div>

## Downstream Task Performance

We test the performance improvement on downstream VQA tasks with
DocVQA [DocVQA](http://arxiv.org/pdf/2111.05547v1) and
ChartQA [masry2022chartqa](http://arxiv.org/pdf/2203.10244v1). We use the addition prompt:
"Answer the following questions using a single word or
phrase:" [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) to allow the model to
output short and precise answers. As shown in
Table <a href="#tab:2" data-reference-type="ref" data-reference="tab:2">1</a>,
Vary-base (with Qwen-7B as LLM) can achieve 78.2% (test) and 76.3% (val)
ANLS on DocVQA upon LLaVA-80k [llava](http://arxiv.org/pdf/2402.11690v1) SFT data. With
LLaVA-665k [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1) data for SFT, Vary-base
can reach 66.1% average performance on ChartQA. The performance on both
two challenging downstream tasks is comparable to or even better than
Qwen-VL [Qwen-VL](http://arxiv.org/pdf/2308.12966v3), demonstrating the proposed vision
vocabulary scaling-up method is also promising for downstream.

## General Performance

We monitor the general performance of Vary through
MMVet [yu2023mm](http://arxiv.org/pdf/2402.15896v1) benchmark. As shown in
table <a href="#tab:3" data-reference-type="ref" data-reference="tab:3">2</a>,
with the same LLM (Vicuna-7B) and SFT data (LLaVA-CC665k), Vary lifts
2.4% (32.9% vs. 30.5%) of the total metric than LLaVA-1.5, proving that
our data and training strategy do not hurt the model’s general ability.
Besides, Vary with Qwen-7B and LLaVA-80k can achieve 36.2% performance,
further demonstrating the effectiveness of our vision vocabulary
scaling-up manner.

# Conclusion [discussion]

This paper highlights that scaling up the vocabulary in the visual
branch for an LVLM is quite significant and we successfully devise a
simple method to prove such a claim. According to the experiments, the
provided model, Vary, achieves promising scores in multiple tasks, which
is mainly profited by the new vocabulary we generated. Despite the
satisfactory performance of Vary, we believe that how to effectively
scale up the visual vocabulary still enjoys much improvement rooms,
especially compared to the mature and relatively simple means of
expanding text vocabulary. We hope that the useful and efficient design
of Vary will attract more research attention to such a direction.

# Appendix

In this appendix, we present the output results of our model to provide
a more intuitive understanding of its performance.

<figure id="figa1">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vis.png" style="width:100.0%" />
<figcaption>Instruction following ability of Vary-base to excel markdown
conversion or pure OCR. Vary-base can control the output format for a
document image input upon the user’s prompts.</figcaption>
</figure>

<figure id="figa2">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vis3.png" style="width:100.0%" />
<figcaption>Fine-grained visual perception ability of Vary-base on
English document dense OCR. This image is the page 3 of  <span
class="citation" data-cites="wei2022humanliker"></span>.</figcaption>
</figure>

<figure id="figa3">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vis4.png" style="width:100.0%" />
<figcaption>Fine-grained visual perception ability of Vary-base on
Chinese book dense OCR. This image is from the Internet.</figcaption>
</figure>

<figure id="figa4">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vis5.png" style="width:100.0%" />
<figcaption>Markdown/Latex format conversion ability (on math formula)
of Vary-base. This image is from the Internet.</figcaption>
</figure>

<figure id="figa5">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vis6.png" style="width:100.0%" />
<figcaption>Markdown/Latex format conversion ability (on the table) of
Vary-base.The images are from the Internet.</figcaption>
</figure>

<figure id="figa5">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vis2.png" style="width:85.0%" />
<figcaption>Chart understanding (Chinese) of Vary-base. The images are
from the Internet.</figcaption>
</figure>

<figure id="figa5">
<embed src="/papers/vision_rich/arXiv-2312.06109v1_md/Figs/vis7.png" style="width:92.0%" />
<figcaption>General performance of Vary-base. The images are from
LLaVA <span class="citation" data-cites="llava"></span>
samples.</figcaption>
</figure>

[^1]: Equal contribution

[^2]: Project leader