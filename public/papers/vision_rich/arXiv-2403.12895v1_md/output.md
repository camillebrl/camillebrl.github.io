<figure id="fig:radar">
<div class="center">
<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/radar.png" style="width:70.0%" />
</div>
<figcaption>Compared with similar-size generalists, our DocOwl
1.5 achieves state-of-the-art OCR-free performance on 10 Visual Document
Understanding benchmarks.</figcaption>
</figure>

[^1]: Corresponding authors

# Introduction

Leveraging the strong language understanding and generation ability of
Large Language Models
(LLM) [gpt3](http://arxiv.org/pdf/2112.07522v2), [llama](http://arxiv.org/pdf/2402.08075v1), [vicuna](https://github.com/lm-sys/FastChat), [llm_survey](http://arxiv.org/pdf/2310.12321v1), some recent
works [mplugowl](http://arxiv.org/pdf/2405.00390v2), [mplug-owl2](None), [llava](http://arxiv.org/pdf/2402.11690v1), [llava1.5](http://arxiv.org/pdf/2310.19145v1), [minigpt4](http://arxiv.org/pdf/2402.17510v1), [blip2](None)
have developed Multimodal Large Language Models (MLLMs) for general
vision-and-language understanding. By aligning a pre-trained visual
encoder (e.g. the ViT/L-14 [vit2021](http://arxiv.org/pdf/2105.15075v2) from
CLIP [clip](http://arxiv.org/pdf/2404.19696v1)) and the LLM with a Vision-to-Text (V2T)
module, these models present promising performance on understanding
general images. However, they still face great challenges with images
with rich text information, such as documents, webpages, tables, and
charts [llmocr](http://arxiv.org/pdf/2305.07895v5). This is mainly because the visual
encoder and V2T module are trained on general image-text pairs and not
specifically optimized to represent the textual and structural
information in text-rich images.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/intro_case_5.png" style="width:100.0%" />

</div>

Textual information in images manifests with a multitude of visual
structures, spanning the simplicity of plain text to the systematic grid
layouts of tables and incorporating a spectrum of graphical
representations such as pie, line, and bar charts. These elements may
appear in isolation or be intricately interwoven within the framework of
documents and webpages, reflecting a rich diversity of informational
architecture across posters, invoices, infographics, scientific reports,
academic and news websites, etc. As shown in
<a href="#fig:intro" data-reference-type="ref+label"
data-reference="fig:intro">[fig:intro]</a>, besides the basic textual
content, structure information also plays a big role in Visual Document
Understanding [layoutlmv2](http://arxiv.org/pdf/2310.16527v1), [layoutlmv3](None), [udop](http://arxiv.org/pdf/2212.02623v3), [pix2struct](None).
With basic abilities to understand general images and comprehend
structured texts through the LLM decoder, MLLM has the potential to
achieve unified structure learning on text-rich images. For better
Visual Document Understanding with MLLMs, some
works [docowl](None), [ureader](None), [qwenvl](http://arxiv.org/pdf/2308.12966v3), [docpedia](http://arxiv.org/pdf/2311.11810v3) attempt to design
text-reading tasks to strengthen the text recognition ability, but
either ignore the structure comprehension or only cover limited domains
of text-rich images, such as just webpages [pix2struct](None)
or documents [docpedia](http://arxiv.org/pdf/2311.11810v3). In this work, we first propose
to perform unified structure learning on text-rich images for MLLMs
across 5 domains: document, webpage, table, chart, and natural image.

For better structural understanding, we first design a simple and
effective vision-to-text module, namely . Unlike the
Resampler [Alayrac2022FlamingoAV](http://arxiv.org/pdf/2205.07065v1) or
Q-former [blip2](None) which fuses visual features with
learnable queries but affects spatial information, the  accumulates
neighborhood visual features through convolution to keep the relative
positional relationships. Compared with V2T modules with only linear
layers [llava](http://arxiv.org/pdf/2402.11690v1), [llava1.5](http://arxiv.org/pdf/2310.19145v1), it produces much fewer visual
features, which is more efficient for LLM to understand high-resolution
document images. Considering texts in document images are most organized
from left to right,  merges visual features at the horizontal level. Our
Unified Structure Learning comprises structure-aware parsing tasks and
multi-grained text localization tasks. To learn the organization of text
contents, the former mainly teaches the model to parse the texts in the
image in a structure-aware style, such as using line feeds and spaces to
represent the structure of documents or webpages, and using extended
Markdown syntax to represent the structure of tables and charts.
Multi-grained text localization tasks further enhance the ability to
correlate visually situated texts and concrete positions in the image.
To support unified structure learning, based on publicly available
datasets, we carefully build a comprehensive training set  by
constructing structure-aware sequences and multi-grained pairs of text
and bounding boxes. The  is trained in a two-stage framework, starting
with the Unified Structure Learning and then followed by the Multi-task
Tuning among downstream tasks. Finally, to trigger the reasoning ability
of MLLM in Visual Document Understanding, we construct a high-quality
instruction tuning dataset . By performing joint training on  and
downstream datasets, -Chat well balance giving a simple answer or
detailed explanations.

Our contributions in this work are four-fold:

-   We first propose Unified Structure Learning on text-rich images for
    MLLMs and design both structure-aware parsing tasks and
    multi-grained text localization tasks across 5 domains. A
    comprehensive dataset  is carefully built to support Unified
    Structure Learning.

-   We design a simple and effective vision-to-text module for structure
    learning and perform extensive experiments to validate its
    effectiveness.

-   We construct a high-quality instruction tuning set to trigger the
    reasoning ability of MLLMs on Visual Document Understanding.

-    and -Chat achieves state-of-the-art OCR-free performance on 10
    Visual Document Understanding tasks, achieving improvement of more
    than 10 points on 5/10 tasks among similar-sized models.

# Related Work

(VDU), also known as Visually-situated Language
Understanding [pix2struct](None), [ureader](None), aims to comprehend
images with rich text information. Such images range from
documents [docvqa](None), [infovqa](http://arxiv.org/pdf/2104.12756v2), [deepform](http://arxiv.org/pdf/2303.13839v1), [klc](None), [mpmqa](None),
tables [wikitableqa](http://arxiv.org/pdf/2009.13845v2), [TabFact](http://arxiv.org/pdf/2311.06592v1), [pubtabnet](http://arxiv.org/pdf/2402.04297v1),
charts [chartqa](None), [dvqa](None), [plotqa](http://arxiv.org/pdf/1906.04124v2), [chart2text](None), [vistext](None), [paperowl](http://arxiv.org/pdf/2311.18248v2),
natural images [textcaps](None), [textvqa](None), [qctextcap](http://arxiv.org/pdf/2302.02124v2) to webpage
screenshots [visualmrc](http://arxiv.org/pdf/2101.11272v2), [websrc](http://arxiv.org/pdf/2004.14797v1), where diverse
composition of text and visual objects contains a wealth of information.
To evaluate the multimodal document understanding performance, the task
formats include low-level recognition, e.g. information
extraction [deepform](http://arxiv.org/pdf/2303.13839v1), [klc](None), and high-level semantic
understanding, such as visual question
answering [docvqa](None), [infovqa](http://arxiv.org/pdf/2104.12756v2), [wikitableqa](http://arxiv.org/pdf/2009.13845v2), [chartqa](None), [visualmrc](http://arxiv.org/pdf/2101.11272v2), [textvqa](None),
image captioning [textcaps](None), [chart2text](None), [vistext](None), and
natural language inference [TabFact](http://arxiv.org/pdf/2311.06592v1). According to
whether relying on an off-the-shelf OCR system to recognize texts in the
image, models for Visual Document Understanding can be categorized into
OCR-dependent models [udop](http://arxiv.org/pdf/2212.02623v3), [layoutlmv2](http://arxiv.org/pdf/2310.16527v1), [layoutlmv3](None), [tap](None)
and OCR-free ones [donut](http://arxiv.org/pdf/2305.09520v1), [pix2struct](None). To leverage
recognized texts from an OCR system, OCR-dependent models are always
trained to align textual and visual inputs. For example,
UDOP [udop](http://arxiv.org/pdf/2212.02623v3) is pre-trained to recover masked text and
layout information given image and retained text as inputs. As for
OCR-free methods, training with tasks about text recognition is
indispensable. Dount [donut](http://arxiv.org/pdf/2305.09520v1) design the text reading
task to output continuous text sequences that ignore structure
information. To leverage structure information,
Pix2Struct [pix2struct](None) designs a Screenshot Parsing
Task to generate the HTML DOM tree for webpage screenshots but is hard
to apply to other types of images. In this work, we first propose
Unified Structure Learning for all image types and carefully build a
comprehensive dataset to support layout learning.

(MLLM) have shown strong vision understanding and open-ended
conversation
abilities [mplugowl](http://arxiv.org/pdf/2405.00390v2), [mplug-owl2](None), [minigpt4](http://arxiv.org/pdf/2402.17510v1), [instructblip](None), [qwenvl](http://arxiv.org/pdf/2308.12966v3), [cogagent](None), [mmllm_survey](http://arxiv.org/pdf/2306.14895v1)
for natural images. They follow the architecture paradigm of connecting
a vision encoder,e.g. ViT [vit2021](http://arxiv.org/pdf/2105.15075v2), [clip](http://arxiv.org/pdf/2404.19696v1), with a Large
Language Model(LLM) [llama](http://arxiv.org/pdf/2402.08075v1), [vicuna](https://github.com/lm-sys/FastChat), [qwen](http://arxiv.org/pdf/2309.16609v1) by a
vision-to-text module, such as simple linear
layers [llava](http://arxiv.org/pdf/2402.11690v1), [llava1.5](http://arxiv.org/pdf/2310.19145v1) or a
Q-Former [blip2](None)/Resampler [Alayrac2022FlamingoAV](http://arxiv.org/pdf/2205.07065v1)/Abstractor [mplugowl](http://arxiv.org/pdf/2405.00390v2), [mplug-owl2](None)
with learnable queries. To enable MLLMs to comprehend images with rich
texts, there are major two challenges: how to encode high-resolution
images and how to understand visually-situated texts. To tackle
high-resolution images, most works choose to further
train [qwenvl](http://arxiv.org/pdf/2308.12966v3), [docpedia](http://arxiv.org/pdf/2311.11810v3) or extraly add a high-resolution
vision encoder [cogagent](None).
UReader [ureader](None) first proposes to keep the
low-resolution vision encoder and use a shape-adaptive cropping module
to crop raw images into multiple sub-images with low resolution. To
enhance the visually-situated text understanding, some work design tasks
of reading texts from top-left to bottom-right without taking into
account the importance of structure [ureader](None), [qwenvl](http://arxiv.org/pdf/2308.12966v3).
CogAgent [cogagent](None) and
DocPedia [docpedia](http://arxiv.org/pdf/2311.11810v3) further try strengthening the layout
understanding for documents, webpages, and natural images with text
grounding tasks. However, the comprehension of the overall structure is
ignored, and tables and charts are not covered. In this work, we follow
UReader to process high-resolution images. To strengthen structure
understanding, we design structure-aware praising and multi-grained text
localization tasks for all types of images, covering documents, tables,
charts, webpages, and natural images. We propose a vision-to-text
architecture to better maintain spatial information of visual features
by convolution. Finally, to support unified structure learning, we build
a comprehensive training dataset  and greatly improve the visual
document understanding performance.

# DocOwl 1.5

 follows the typical architecture of Multimodal Large Language Models,
which consists of a visual encoder, a vision-to-text module, and a large
language model as the decoder. To better keep the textual and layout
information in text-rich images of high resolution, we design an  as the
vision-to-text module to ensemble horizontal visual features. As shown
in <a href="#fig:overall_arch" data-reference-type="ref+label"
data-reference="fig:overall_arch">[fig:overall_arch]</a>(a), to enhance
the text recognition and structure understanding abilities, we first
perform Unified Structure Learning with structure-aware parsing and
multi-grained text localization tasks for all types of images. Then, the
model is jointly tuned on multiple downstream tasks of Visual Document
understanding.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/model_and_traing.png" style="width:100.0%" />

</div>

## Model Architecture

**High-resolution Image Encoding.** As proved by previous
works [donut](http://arxiv.org/pdf/2305.09520v1), [pix2struct](None), [ureader](None), the ability to encode
high-resolution images is critical to ensuring that the decoder can use
rich text information from document images. As shown in
<a href="#fig:overall_arch" data-reference-type="ref+label"
data-reference="fig:overall_arch">[fig:overall_arch]</a>(b), following
UReader [ureader](None) , we utilize a parameter-free
Shape-adaptive Cropping Module to crop a shape-variable high-resolution
image $I$ into multiple fixed-size sub-images $(I_1, I_2,...,I_C)$,
where $C$ is the number of crops. To keep the overall layout
information, the raw image is also resized to a low-resolution one as
the global image $I_0$. Then, each image $I_i$ in $(I_0,I_1,...,I_C)$ is
independently encoded to a sequence of visual features
$V_i = (v_i^1, v_i^2,...,v_i^L), 0 \leq i \leq C$ by a transformer-based
Visual Encoder, where $v_i^j, 1 \leq j \leq L$ is a $D$-dimension
vector, $L$ is the length of visual features for each image.

**Spatial-aware Vision-to-Text Module: .** There are two kinds of
popular vision-to-text modules for Multimodal Large Language Models: a
MLP [llava](http://arxiv.org/pdf/2402.11690v1), [llava1.5](http://arxiv.org/pdf/2310.19145v1), [minigpt4](http://arxiv.org/pdf/2402.17510v1) or a cross-attention
module with learnable
queries [mplugowl](http://arxiv.org/pdf/2405.00390v2), [qwenvl](http://arxiv.org/pdf/2308.12966v3), [Alayrac2022FlamingoAV](http://arxiv.org/pdf/2205.07065v1), [blip2](None).
Both two are not quite suitable for representing high-resolution
text-rich images. The former projects complete visual features into the
language embedding space. It maintains all spatial information in the
document image but keeps the sequence length of raw visual features,
which is too long when processing high-resolution images. For example,
encoding a 1,344x1,344 image with the ViT/L-14 results in 9,216 visual
tokens. The cross-attention module could greatly reduce the length of
the visual sequence to the number of learnable queries, but may lose
spatial information during semantic fusion.

In this work, we design a more appropriate vision-to-text module for
Visual Document Understanding, namely , which not only reduces visual
sequence length but also keeps the spatial information. As shown in
<a href="#fig:overall_arch" data-reference-type="ref+label"
data-reference="fig:overall_arch">[fig:overall_arch]</a>(b), the  is
comprised of a convolution layer to reduce sequence length and a
fully-connected layer to project visual features to language embedding
space. Since most textual information in document images is arranged
from left to right, the horizontal text information is usually
semantically coherent. Thus, the kernel size and stride size in the
convolution layer are set as 1x4 to ensemble horizontal 4 visual
features. The output channel is set equal to the input channel $D$. The
convolution calculation is as follows: $$\begin{gathered}
    V_i = (v_i^1, v_i^2,...,v_i^L)\\
    \overline{v}_i^j = f(v_i^{4j-3},v_i^{4j-2},v_i^{4j-1},v_i^{4j}), 1 \leq j \leq L/4, \\
    \overline{V}_i = (\overline{v}_i^1, \overline{v}_i^2,...,\overline{v}_i^{L/4}),
\end{gathered}$$ where $f$ represents the dot product with kernel
weights on multiple channels. After the convolution layer, the visual
features of image $I_i$ are converted to the $\overline{V}_i$, the
feature length of which is $L/4$.

Then, with a fully connected layer to align visual features to the
language embedding space, the $\overline{V}_i$ are transferred to
$\hat{V}_i = (\hat{v}_i^1, \hat{v}_i^2,...,\hat{v}_i^{L/4})$.

**Multimodal Modeling with LLM.** As the decoder of MLLM, large language
models should understand both the visual features of images and the
textual features of language instructions. Following
mPLUG-Owl2 [mplug-owl2](None), we apply the Modality-adaptive
Module(MAM) in LLM to better distinguish visual and textual inputs.
During self-attention, MAM utilizes two sets of linear projection layers
to separately perform the key/value projection for visual features and
textual features. To help the LLM correlate multiple cropped sub-images,
UReader [ureader](None) designs learnable crop position
embeddings to denote the row and column position in the raw image. In
this work, we simply add special textual tokens `‘<rowx_coly>’` before
the visual features of each cropped image, where $x$ and $y$ refer to
the row and column index respectively. For the global image, the textual
indicator token is `‘<global_img>’`. This design eliminates the need to
introduce additional parameters and is more friendly to the LLM decoder.
Our experiments validate that it achieves comparable effects as the crop
position embedding. Overall, the decoding of the LLM is as follows:
$$\begin{gathered}
    Y = \rm{LLM}([T_0;\hat{V}_0, T_1;\hat{V}_1, ...,T_C; \hat{V}_C;X])
\end{gathered}$$ where $[;]$ means the concatenation operation, $C$ is
the crop number of the image, $T_j, 0 \leq j \leq C$ is the textual
embeddings of the special textual indicator for the global image or
positions of cropped images, $\hat{V}_j$ is the visual features of a
global or cropped image, $X$ is the textual embeddings of the
instruction, $Y$ is the predicted answer.

## Unified Structure Learning

Most Multimodal Large Language
Models [llava](http://arxiv.org/pdf/2402.11690v1), [mplug-owl2](None), [cogvlm](http://arxiv.org/pdf/2210.00066v1) are trained with
image-text pairs of natural images to align the visual encoder with the
LLM, such as Conceptual Captions [ConceptualCaption](None),
LAION [laion](None) and COYO [coyo](https://github.com/kakaobrain/coyo-dataset).
Initializing from such models could inherit the shallow text recognition
ability, but is far from understanding complex textual and structural
information in various text-rich images. In this work, to empower the
comprehensive document understanding abilities of MLLM, we design a
Unified Structure Learning across 5 domains, including natural images,
documents, tables, charts, and webpages. It involves both
structure-aware parsing tasks and multi-grained text localization tasks,
as shown in <a href="#fig:layout_tasks" data-reference-type="ref+label"
data-reference="fig:layout_tasks">[fig:layout_tasks]</a>.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/layout_training.png" style="width:100.0%" />

</div>

**Document Parsing.** For representing the structure information,
Pix2Struct [pix2struct](None) parses webpage screenshots with
condensed HTML DOM trees, which are built based on the HTML source codes
and are not available for other formats of documents or webpage
screenshots, e.g. PDF. In documents or webpages, horizontal and vertical
distances between texts form the main layout information. Therefore, to
make the structure-aware parsing task applicable to most documents and
webpage screenshots, we choose to add extra line
feeds(`‘\textbackslash n’`) and spaces into the text sequence to denote
different lines and horizontal distances. The greater the horizontal
distance, the more space characters.

We choose CCpdf [ccpdf](http://arxiv.org/pdf/2304.14953v2),
RVL-CDIP [rvlcdip](http://arxiv.org/pdf/1502.07058v1),
VisualMRC [visualmrc](http://arxiv.org/pdf/2101.11272v2) and datasets encapsulated in
DUE-Benchmark [due](None) (DocVQA [docvqa](None),
InfoVQA [infovqa](http://arxiv.org/pdf/2104.12756v2), DeepForm [deepform](http://arxiv.org/pdf/2303.13839v1),
KLC [klc](None), WTQ [wikitableqa](http://arxiv.org/pdf/2009.13845v2),
TabFact [TabFact](http://arxiv.org/pdf/2311.06592v1)) to support the Document Parsing task.
CCpdf [ccpdf](http://arxiv.org/pdf/2304.14953v2) is a multi-lingual PDF dataset built upon
webpages from Common Cramwl[^1], covering diverse domains of documents,
such as industry, academic, and medical. In this work, we mainly focus
on English Document Understanding and drop PDFs detected as other
languages. RVL-CDIP contains 16 categories of industry documents, such
as ‘letter’, ‘email’, and ‘scientific reports’. We further remove some
categories with flipping and blurring texts, such as ‘handwritten’ and
‘form’. DUE-Benchmark is a collection of available and reformulated
datasets over various document domains and layouts featuring tables,
graphs, lists, and infographics. VisualMRC is a webpage screenshot
dataset across 35 websites. OCR annotations in VisualMRC are aligned
with local regions, thus, we follow them to utilize crops of a
screenshot as input for this parsing task. For CCpdf and DUE-Benchmark,
a PDF-parsing tool pdfplumber[^2] can be directly used to generate
structure-aware text sequence with a PDF page as the input. For RVL-CDIP
and VisualMRC, there are no PDF files, just annotations of bounding
boxes of texts. As an alternative, akin to the
LATIN-Prompt [latin](None), we insert the line feeds and
spaces by calculating and comparing the horizontal and vertical
distances of bounding boxes. To avoid too many space characters
resulting in sparse texts, we further limit the maximum number of
consecutive spaces to 4. This strategy allows us to construct
structure-aware text sequences in the same style as pdfplumber.

**Table Parsing.** Different from documents or webpages, tables are
structured in a more standardized way, where row and column
correspondences represent key-value pairs. HTML and Markdown codes are
mainly two kinds of text sequences used to represent a table. HTML codes
can represent all kinds of tables, with or without cells spanning
multiple rows and grids, but they contain too many paired labels (e.g.
`‘<tr></tr>’` and `‘<td></td>’`), causing text sequences to be too long.
Markdown codes can represent a table with concise text sequence, but
they cannot represent cells spanning multiple rows and columns. To
represent all tables with concise text sequence, we follow the main
grammar of Markdown to represent table structure with `‘|’` and line
feeds(`‘\textbackslash n’`). To represent cells spanning multiple rows
and columns, we add special text tokens `‘<COLSPAN=x>’` and
`‘<ROWSPAN=y>’` before the value, as shown in
<a href="#fig:layout_tasks" data-reference-type="ref+label"
data-reference="fig:layout_tasks">[fig:layout_tasks]</a>.

We choose TURL [turl](None) and
PubTabNet [pubtabnet](http://arxiv.org/pdf/2402.04297v1) to do the structure-aware table
parsing task, where tables are collected from Wikipedia pages and
scientific articles, respectively. Without cells across rows and
columns, tables in TURL can be directly represented with Markdown codes.
Due to lacking table images in TURL, we transfer tables into HTML codes
and render table images with variations in background color and font
size. PubTabNet contains pairs of table images and HTML codes. We
convert HTML codes into Markdown style and add `‘<ROWSPAN=x>’` or
`‘<COLSPAN=y>’` before the value when attributes `‘rowspan=x’` or
`‘colspan=y’` are set in the `‘<td>’` label.

**Chart Parsing.** Unlike documents and tables, organizing texts in
reading order cannot represent the structure of charts. Considering that
the chart is a visualization form of the table, parsing charts to tables
could best maintain the mathematical characteristics of the chart. This
requires the model to understand the structure of the chart and the
alignment of the x/y axis. Besides, to keep consistent with the Table
Parsing task, we also use Markdown codes to represent the data tables of
charts, as shown in
<a href="#fig:layout_tasks" data-reference-type="ref+label"
data-reference="fig:layout_tasks">[fig:layout_tasks]</a>.

We adopt PlotQA [plotqa](http://arxiv.org/pdf/1906.04124v2),
FigureQA [figureqa](http://arxiv.org/pdf/2109.02226v1), DVQA [dvqa](None), and
ChartQA [chartqa](None) to support the structure-aware chart
parsing task. These datasets cover charts on both
synthetic [figureqa](http://arxiv.org/pdf/2109.02226v1), [dvqa](None) data and data from real-world
sources [plotqa](http://arxiv.org/pdf/1906.04124v2), [chartqa](None). Chart types include vertical
bar, horizontal bar, line, dot line, and pie chart. Source data of the
chart is provided in the JSON [plotqa](http://arxiv.org/pdf/1906.04124v2), [figureqa](http://arxiv.org/pdf/2109.02226v1), [plotqa](http://arxiv.org/pdf/1906.04124v2)
or CSV format [chartqa](None), both can be conveniently
converted to Markdown codes. However, some raw values are not suitable
as standard answers for parsing because there are too many significant
digits to be represented on the chart. Therefore, to reduce the
difficulty of estimating values and make the model focus more on
structural understanding, we keep 4 significant digits for all values.

**Natural Image Parsing.** Quite different from text-dominant images
mentioned above, the semantics of natural images is a combination of
natural objects and scene texts. Thus, parsing natural images is
necessary to organize scene texts and mention the main image content.
Manually annotating captions to describe the relationship between
objects and scene texts is labour- and financial-intensive. Like
TAP [tap](None), we concatenate the general caption with OCR
texts to form the target parsing sequence.

We utilize OCR-CC [tap](None) to support the Natural Image
Parsing task. OCR-CC is a subset of Conceptual
Caption [cc2018](None), which contains images with scene texts
detected by the Microsoft Azure OCR system.

**Multi-grained Text Localization.** As proved in previous
works [e2evlp](None), [ofa](None), [kosmos2](http://arxiv.org/pdf/2305.16103v1) on general image
understanding, semantic comprehension and object grounding tasks can be
well unified in a single model. For Visual Document Understanding,
structure-aware parsing tasks mainly focus on organizing texts according
to the overall structure, while neglecting the correspondence between
specific texts and local positions. Correlating texts with the concrete
position in images is another basic structure understanding ability for
visual documents. To support text position learning, we design two
symmetrical tasks, namely Multi-grained Text Grounding and Multi-grained
Text Recognition. The former aims to predict the bounding box given the
visually-situated texts, while the latter does the opposite. We set four
granularities of texts for these two tasks: word, phrase, line, and
block. The ‘word’ is the smallest granularity of the bounding box,
referring to only 1 word. To ensure that the word is visible and the
answer is unique, words that are too small (normalized area \< 0.001)
and words that appear multiple times in the same image are excluded from
candidates. The ‘line’ consists of texts that are judged to be
horizontally parallel by vertical distance, and the ‘phrase’ is
comprised of multiple adjacent words within the same line. The ‘block’
is a combination of multiple successive lines, ranging from 2 to half of
the total lines. The text sequences of word-level and phrase-level
question answering are much shorter than the other two. Therefore, in
order to learn localization more efficiently, each word-level or
phrase-level sample consists of up to 5 question-answer pairs for the
same image. As for the representation of bounding boxes, we transfer
each continuous value in the normalized bounding box into a discrete
position token, ranging from 0 to 999.

The bounding box annotation is necessary for constructing samples for
Multi-grained Text Localization tasks. Therefore, we take DocVQA,
InfoVQA, WTQ, TabFact, DeepForm, KLC, ChartQA, VisualMRC, and
TextVQA [textvqa](None) for this task, across domains of the
document, table, chart, webpage, and natural image.

Overall, to support the unified structure learning for text-rich images,
we build a  dataset by ensembling multiple training sets of publicly
available datasets and constructing structure-aware text sequences or
text-position pairs as the targets. The form of instructions for each
task is very diverse for developing the general instruction-following
ability of the model.
<a href="#fig:data_distri" data-reference-type="ref+label"
data-reference="fig:data_distri">[fig:data_distri]</a> shows the
detailed statistics of .

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/data_distribution.png" style="width:90.0%" />

</div>

## Multi-task Fine-tuning

Through Unified Structure Learning, models could well understand the
structure of diverse document images but cannot follow users’
instructions to do different types of tasks, such as information
extraction or image captioning. So, we further perform multi-task
fine-tuning to train a generalist of visual document understanding as
UReader [ureader](None).

## Training Paradigm

As shown in <a href="#fig:overall_arch" data-reference-type="ref+label"
data-reference="fig:overall_arch">[fig:overall_arch]</a>(a),  is trained
in a two-stage framework. Considering the LLM has strong comprehension
abilities for structured text [latin](None), [tablellama](http://arxiv.org/pdf/2311.09206v3), we
argue that the main limitation of MLLM in visual document understanding
is the representation ability of the Visual Encoder and Vision-to-Text
module for visually-situated text and structure information. Thus,
during the Unified Structure Learning, we freeze the LLM parameters and
tune the Visual Encoder and . The MAM is also optimized to help the LLM
better distinguish visual features and texts parsed from the image.
During the stage of Multi-task Fine-tuning, the model mainly learns how
to follow the user’s instructions to give answers based on
visually-situated text and structure understanding capabilities acquired
in the first stage. Therefore, the Visual Encoder is frozen and other
modules are tuned.

# -Chat

Existing benchmarks mainly evaluate the document understanding ability
by answering the question with simple phrases and neglect detailed
explanations. In this work, to better leverage the strong language
reasoning ability of Large Language Models on Visual Document
Understanding, we build a small instruction-tuning set with detailed
explanations on text-rich image understanding, namely . Based on raw
questions from DocVQA [docvqa](None),
InfoVQA [infovqa](http://arxiv.org/pdf/2104.12756v2), WTQ [wikitableqa](http://arxiv.org/pdf/2009.13845v2),
VisualMRC [visualmrc](http://arxiv.org/pdf/2101.11272v2), ChartQA [chartqa](None)
and TextVQA [textvqa](None), we collect detailed explanations
with ChatGPT[^3]. Text contents are dominant information on documents,
tables or webpage screenshots. Therefore, for DocVQA, InfoVQA, WTQ, and
VisualMRC, we take the structure-aware text sequence of the image as the
input to `gpt-3.5-turbo-0301` and prompt it to answer the question with
simple answers and detailed explanations. As for ChartQA and TextVQA, we
take the image as the input and utilize the `gpt-4-vision-preview` to
answer the question with detailed explanations. In order to filter out
samples where ChartGPT answers incorrectly, we further prompt
`gpt-3.5-turbo-0301` to judge whether the answer given by ChartGPT is
consistent with the concise human-annotated ground-truth answer.
Compared with raw questions in benchmark datasets, questions in  are
added with a prompt `‘Answer the question with detailed explanation’`.
Detailed statistics of  are presented in
<a href="#tab:instruct_set" data-reference-type="ref+label"
data-reference="tab:instruct_set">[tab:instruct_set]</a>. -Chat is
trained by combining downstream datasets with  and performing multi-task
tuning after Unified Structure Learning.

<div class="table*" markdown="1">

|            | DocVQA | InfoVQA |  WTQ  | VisualMRC | ChartQA | TextVQA |  ALL   |
|:----------:|:------:|:-------:|:-----:|:---------:|:-------:|:-------:|:------:|
|   Image    | 1,491  |  1,614  |  850  |   1,927   |  1,252  |  1,612  | 8,746  |
|   Sample   | 5,119  |  5,421  | 5,994 |   5,263   |  1,827  |  2,253  | 25,877 |
| Avg Length |  79.2  |  95.4   | 77.7  |   103.4   |  106.9  |  88.0   |  89.9  |

</div>

[^1]: <https://commoncrawl.org>

[^2]: <https://github.com/jsvine/pdfplumber>

[^3]: <https://openai.com/chatgpt>

# Experiments

## Implementation Details

 is initialized from mPLUG-Owl2 [mplug-owl2](None), which
utilizes the ViT/L-14 [vit2021](http://arxiv.org/pdf/2105.15075v2) as the Visual Encoder
and a 7B Large Langauge Model with the Modality Adaptive Module as the
language decoder. According to the aspect ratio and resolution, each
image is cropped into up to 9 sub-images with a fixed resolution of
448x448. Each sub-image is encoded to 1,024 features by the ViT/L-14 and
then reduced to 256 features by the . The model is trained with 12,000
iterations on , with the learning rate and batch size set as 1e-4 and
1,024. It costs about 128 A100 days. During the Multi-task finetuning,
the model is trained for 6,500 iterations with the batch size set as 256
and the learning rate set as 2e-5. This further costs about 24 A100
days.

<div class="table*" markdown="1">

</div>

<div class="table*" markdown="1">

</div>

## Main Results

We evaluate the Visual Document Understanding performance on 10
text-rich image benchmarks, covering documents
(DocVQA [docvqa](None), InfoVQA [infovqa](http://arxiv.org/pdf/2104.12756v2),
DeepForm [deepform](http://arxiv.org/pdf/2303.13839v1), KLC [klc](None)), tables
(WTQ [wikitableqa](http://arxiv.org/pdf/2009.13845v2), TabFact [TabFact](http://arxiv.org/pdf/2311.06592v1)),
charts (ChartQA [chartqa](None)), natural images
(TextVQA [textvqa](None),
TextCaps [textcaps](None)), and webpage screenshots
(VisualMRC [visualmrc](http://arxiv.org/pdf/2101.11272v2)). We compare  with
state-of-the-art OCR-free models, including both Multimodal Large
Language Models adapted for recognizing texts and much smaller models
trained only for document understanding. The detailed comparison of
model settings can be found in
<a href="#tab:model_setting" data-reference-type="ref+label"
data-reference="tab:model_setting">[tab:model_setting]</a>. As shown in
<a href="#tab:main" data-reference-type="ref+label"
data-reference="tab:main">[tab:main]</a>, previous MLLMs with more than
7B parameters underperform domain-specific models with less than 1B
parameters, showing that the document understanding is still a
shortcoming for existing MLLMs. Our  outperforms both domain-specific
models and MLLMs with similar sizes on all 10 benchmarks. This validates
that  is much stronger on visual document understanding across 5
domains, covering visual question answering, information retrieval,
natural language inference, and image captioning tasks. Besides, with
much fewer unnatural data (3M vs 9M) and parameters (8.1B vs 17.3B),
 outperforms CogAgent [cogagent](None) on InfoVQA and ChartQA,
and achieves comparable performance on DocVQA. This suggests that our
unified structure learning with  is more efficient in learning printed
text recognition and how to analyze documents. However, our model still
underperforms CogAgent on TextVQA, which requires the ability of scene
text recognition and general knowledge about natural objects. The
primary reason is that scene texts are more diverse in shapes than
printed texts and CogAgent is trained on 98M samples of scene text
recognition from LAION-2B [laion](None) and
COYO-700M [coyo](https://github.com/kakaobrain/coyo-dataset), much more than the natural images (1M)
in . In this work, we mainly focus on improving the unified structure
comprehension of visual documents and leave further scaling up data on
natural scenes as future work. Finally, -Chat can also be evaluated on
these concise-answer benchmarks by removing the prompt of detailed
explanation. It achieves comparable or slightly better performance than
, showing that a small amount of detailed explanatory data may better
help the model understand the semantics of text-rich images.

<div class="table*" markdown="1">

</div>

## Ablation Study

As shown in <a href="#tab:ablation" data-reference-type="ref+label"
data-reference="tab:ablation">[tab:ablation]</a>, we further perform a
comprehensive ablation study to validate the effectiveness of our  and
Unified Structure Learning.

Firstly, initializing from a stronger general MLLMs brings better
performance on text-rich images (r2 vs r1), showing general
vision-and-language knowledge benefits visual document understanding.
Tuning the visual encoder during multi-task fine-tuning significantly
improves the document understanding performance (r3 vs r2). This
suggests that the visual representation of document images may be the
main shortcoming of MLLMs and inspires us to design Unified Structure
Learning to enhance the representation ability of the visual encoder for
visually situated texts and structure.

**Effectiveness of .** When using the Shape-adaptive Cropping Module,
the image resolution supported by the MLLM is the product of the
cropping number and basic resolution of each crop. With the Abstractor
as the vision-to-text module, reducing the cropping number causes an
obvious performance decrease (r4 vs r3) on documents. However, with a
smaller cropping number, the  achieves better performance than the
Abstractor (r5 vs r3), showing that $448^2\times9\approx2^{21}$ is an
acceptable resolution for existing benchmarks and the  is stronger on
maintaining rich text information during vision-and-language feature
alignment. Besides, we further compare different settings of the merging
shape in the convolution layer. With the same number of merged tokens,
the model with the 1x4 merging shape achieves better performance than
the one with the 2x2 merging shape on document and table datasets but
slightly worse performance on chart understanding (r6 vs r5). This is
consistent with the common sense that documents and tables mainly
organize texts in the left-to-right order while the semantic structures
of charts are much more flexible. A square merging shape is more suited
to encode visual features in the form of bars, lines, or pies while the
1x4 merging shape is more appropriate for general document
understanding. As shown in r7-r9, further extending the 1x4 merging
shape horizontally and vertically decreases the length of visual
features but at the cost of performance degradation. Considering the
overall performance on all text-rich images, we finally choose the 1x4
as the merging shape in .

**Effectiveness of Unified Structure Learning.** After determining the
vision-to-text module, we perform two-stage training with Unified
Structure Learning. With only the structure-aware parsing tasks, there
is significant improvement across different domains (r10 vs r5). This
validates that fine-tuning the visual encoder and  with structure-aware
parsing tasks greatly helps MLLMs understand text-rich images. Further
tuning the parameters of LLM brings slight improvement (r11 vs r10),
suggesting that general language knowledge is not the main obstacle to
visual document understanding. By replacing the learnable crop position
embeddings with special textual tokens, the model achieves better
performance (r12 vs r11), showing that the LLM can well understand the
relative positions of multiple cropped images with just simple textual
indicators. Finally, by introducing Multi-grained Text Localization
tasks,  achieves the best performance, validating that correlating
visually situated texts with concrete positions helps comprehend
documents more accurately.

<div class="table*" markdown="1">

|                   | **One-Stage** |       |       |       |       | **Two-Stage**  |
|:------------------|:-------------:|:-----:|:-----:|:-----:|:-----:|:--------------:|
|  samples          |     0.0M      | 0.5M  | 1.0M  | 2.0M  | 4.0M  |      4.0M      |
| Benchmark samples |     0.6M      | 0.6M  | 0.6M  | 0.6M  | 0.6M  |      0.6M      |
| Epoch/iteration   |     7/18k     | 6/25k | 6/37k | 4/40k | 3/54k | 3/12k + 3/6.5k |
| Cost (A100 days)  |     60.0      | 83.3  | 123.3 | 133.3 | 180.0 |     144.8      |
| DocVQA            |     72.8      | 75.5  | 78.6  | 78.8  | 78.9  |      79.9      |

</div>

**Effectiveness of the Two-stage Training.** As shown in
<a href="#tab:two_stage" data-reference-type="ref+label"
data-reference="tab:two_stage">[tab:two_stage]</a>, instead of two-stage
training, we also try one-stage joint training of the structure learning
and downstream tasks and gradually increase the samples from . The epoch
is gradually reduced because we didn’t observe performance improvements
with more iterations. For joint training, the model improves
significantly on DocVQA as the samples of Unified Structure Learning
increase when it is below 1M. However, as the Unified Structure Learning
samples are further increased, the improvement of the model becomes
subtle and its performance is not as good as the one using two-stage
training. This shows that the two-stage training could better enhance
basic text recognition and structure parsing abilities and is more
beneficial and efficient for downstream document understanding.

<div class="table*" markdown="1">

<div class="tabular" markdown="1">

c\|cccc\|ccccc \***Task** & &  
  & **Word** & **Phrase** & **Line** &**Block** & **Doc** & **Table** &
**Chart** &**Web** & **Natural**  
Text Recognition & 622 & 499 & 522 & 482 & 1,004 & 491 & 229 & 267 &
134  
Text Grounding & 595 & 542 & 503 & 485 & 1,011 & 524 & 240 & 242 & 108  

</div>

</div>

<div class="table*" markdown="1">

</div>

## Text Localization Evaluation

Besides proving the effectiveness of  through downstream text-rich image
understanding performance in
<a href="#tab:ablation" data-reference-type="ref+label"
data-reference="tab:ablation">[tab:ablation]</a>, we further directly
compare the text localization performance after the Unified Structure
Learning to validate its superiority in preserving spatial features. We
build a text localization evaluation set  with 4,250 samples balanced on
4 granularities and covering both text recognition and text grounding
tasks. The detailed statistics of  are shown in
<a href="#tab:eval_set" data-reference-type="ref+label"
data-reference="tab:eval_set">[tab:eval_set]</a>. Considering that
document images are much more diverse and complex than other images,
there are more samples in this domain than others. The IOU@0.5 is used
to evaluate the text grounding performance. As for text recognition, the
word, phrase, line, and block granularity is evaluated with BLEU1,
BLEU2, BLEU3, and BLEU4 [bleu](http://arxiv.org/pdf/2202.11027v1), respectively. As shown
in <a href="#tab:grounding" data-reference-type="ref+label"
data-reference="tab:grounding">[tab:grounding]</a>, when trained with
the same iterations, the  achieves much better performance on both Text
Recognition and Text Grounding tasks, showing that  with the 1x4 merging
shape helps the LLM better understand concrete positions in images.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/qa_case.png" style="width:100.0%" />

</div>

## Qualitative Results

Besides quantitative results, we further present some qualitative
results of visual document understanding on different domains of images.
As shown in <a href="#fig:qa_case" data-reference-type="ref+label"
data-reference="fig:qa_case">[fig:qa_case]</a>(a) and (b), both models
answer the question with texts in the image.  can better understand the
structure of two documents and give correct answers. In
<a href="#fig:qa_case" data-reference-type="ref+label"
data-reference="fig:qa_case">[fig:qa_case]</a>(c), due to the learning
of parsing chart with Markdown codes,  can better understand the chart
and successfully correlate the x/y axis.
<a href="#fig:qa_case" data-reference-type="ref+label"
data-reference="fig:qa_case">[fig:qa_case]</a>(d) shows that although
inconsistent with the ground truth,  gives another correct answer with
the help of stronger structure understanding on tables.

<a href="#fig:instruct_case_1" data-reference-type="ref+label"
data-reference="fig:instruct_case_1">[fig:instruct_case_1]</a> and
<a href="#fig:instruct_case_2" data-reference-type="ref+label"
data-reference="fig:instruct_case_2">[fig:instruct_case_2]</a> present
qualitative results of detailed explanations. Through a small amount of
reasoning training, -Chat can well inherit the reasoning ability of LLM
and provide detailed explanations about the answer. However, as
presented in
<a href="#fig:instruct_case_2" data-reference-type="ref+label"
data-reference="fig:instruct_case_2">[fig:instruct_case_2]</a>(c), like
most general Multimoal large Language
Models [mplugowl](http://arxiv.org/pdf/2405.00390v2), [mplug-owl2](None), [qwenvl](http://arxiv.org/pdf/2308.12966v3), -Chat may also
suffer from the hallucination problem in Visual Document Understanding.
In this work, we mainly focus on enhancing the unified structure
understanding ability of MLLMs and leave how to resolve the
hallucination problem in OCR-free document understanding as future work.

**Structure-aware Parsing.** As shown in
<a href="#fig:doc_parse" data-reference-type="ref+label"
data-reference="fig:doc_parse">[fig:doc_parse]</a>,  could parse a
document image by using line feeds and spaces to represent the structure
of text contents. Besides parsing the whole document, as shown in
<a href="#fig:doc_parse2" data-reference-type="ref+label"
data-reference="fig:doc_parse2">[fig:doc_parse2]</a>, it could also
parse texts from the middle of the image according to human instruction.
<a href="#fig:table_parse1" data-reference-type="ref+label"
data-reference="fig:table_parse1">[fig:table_parse1]</a> presents
qualitative results of structure-aware table parsing through extended
Markdown syntax on tables with cells spanning multiple columns or not.
Furthermore, <a href="#fig:chart_parse1" data-reference-type="ref+label"
data-reference="fig:chart_parse1">[fig:chart_parse1]</a> shows some
cases of parsing different types of charts into Markdown codes,
including vertical bar, horizontal bar, pie, and line charts. When all
data points are presented in the chart,  can accurately align statistic
objects with corresponding numbers. It makes some mistakes in
<a href="#fig:chart_parse1" data-reference-type="ref+label"
data-reference="fig:chart_parse1">[fig:chart_parse1]</a>(d) because
estimating the concrete numbers is quite challenging when no data points
are provided. Finally, as shown in
<a href="#fig:natural_parse1" data-reference-type="ref+label"
data-reference="fig:natural_parse1">[fig:natural_parse1]</a>,  can both
describe the content of natural images and read scene texts.

**Multi-grained Text Localization.**
<a href="#fig:ground" data-reference-type="ref+label"
data-reference="fig:ground">[fig:ground]</a> and
<a href="#fig:recognize" data-reference-type="ref+label"
data-reference="fig:recognize">[fig:recognize]</a> show qualitative
results of text grounding and text recognition at granularities of word,
phrase, line and block. The image domains range from documents,
webpages, charts, and tables to natural images.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/instruct_case_1.png" style="width:100.0%" />

</div>

# Conclusion

To enhance the Visual Document Understanding performance of Multimodal
Large Language Models, we first propose Unified Structure Learning
across 5 domains of text-rich images, including both structure-aware
parsing tasks and multi-grained text localization tasks. To better
maintain structure and spatial information during vision-and-language
feature alignment, we design a simple and effective vision-to-text
module, named . It mainly utilizes a convolution layer to aggregate
horizontally neighboring visual features. To support the Unified
Structure Learning, we build a training dataset  by collecting publicly
available images and carefully constructing structure-aware text
sequences and multi-grained pairs of texts and bounding boxes. With
Unified Structure Learning, our model  achieves state-of-the-art
OCR-free performance on 10 visual document understanding benchmarks.

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/instruct_case_2.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/doc_parse_2.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/doc_parse_3.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/table_parse_1.png" style="width:90.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/chart_parse_1.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/natural_parse_1.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/text_ground_1.png" style="width:100.0%" />

</div>

<div class="figure*" markdown="1">

<embed src="/papers/vision_rich/arXiv-2403.12895v1_md/figures/text_recognition_1.png" style="width:100.0%" />

</div>