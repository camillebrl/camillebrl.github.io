<div class="figure*" markdown="1">

<embed src="/papers/visionrich_small_dec/arXiv-2309.11419v1_md/images/kosmos25-1.png" style="width:100.0%" />

</div>

# Introduction

Over the past several years, large language models (LLMs) have emerged
as a critical area of research in artificial intelligence. These models
are designed to learn from massive amounts of natural language data,
allowing them to perform a wide range of language-related tasks with
impressive accuracy. This development has been fueled by advancements in
model scaling that enabled researchers to create models with
unprecedented complexity. As a result, LLMs have become increasingly
prevalent across various industries and applications, from customer
service chatbots to virtual assistants and automated content creation.
One notable trend in recent years has been the focus on building larger
and more complex models, such as
GPT-3 [https://doi.org/10.48550/arxiv.2005.14165](https://doi.org/10.48550/ARXIV.2005.14165) and
GPT-4 [openai2023gpt4](https://arxiv.org/pdf/2303.08774), which has hundreds/thousands of
billion parameters and can generate compelling language outputs. While
these models require significant computing resources to train and
operate, they hold enormous potential for revolutionizing how we
interact with and understand natural language.

Current LLMs primarily focus on textual information and cannot
understand visual information. However, advancements in the field of
multimodal large language models (MLLMs) aim to address this limitation.
MLLMs combine visual and textual information within a single
Transformer-based model, enabling the model to learn and generate
content based on both modalities. MLLMs have shown promise in a variety
of real-world applications, including natural image understanding and
text image understanding. These models leverage the power of language
modeling as a general interface for multimodal problems, allowing them
to process and generate responses based on textual and visual inputs.
While existing MLLMs have mainly focused on natural images with lower
resolutions, the exploration of text images is an area that requires
further investigation. Taking advantage of large-scale multimodal
pre-training for text images is an important direction for MLLM
research. By incorporating text images into the training process and
developing models based on textual and visual information, we can unlock
new possibilities for multimodal applications involving high-resolution
text-intensive images.

In this study, we present **<span class="smallcaps">Kosmos-2.5</span>**,
a multimodal literate model that takes advantage of
<span class="smallcaps">Kosmos-2</span> [peng2023kosmos](http://arxiv.org/pdf/2305.16103v1)
designed to tackle machine reading of text-intensive images, which is
shown in <a href="#fig:introduction" data-reference-type="ref+label"
data-reference="fig:introduction">[fig:introduction]</a>.
<span class="smallcaps">Kosmos-2.5</span> performs two closely related
transcription tasks in a unified multimodal model. The first task
generates spatially-aware text blocks, assigning text lines their
corresponding spatial coordinates within the original text-rich image.
The second task produces structured text output, capturing styles and
structures in the markdown format. Both tasks are conducted under a
unified framework, leveraging a shared Transformer architecture,
task-specific prompts, and flexible text representations. Specifically,
our model architecture combines a ViT-based vision encoder and a
Transformer-based language decoder linked by a resampler module. Our
model is pre-trained on a large corpus of text-intensive images, whose
text representations include text lines with bounding boxes and plain
markdown texts. By employing this dual-task training strategy,
<span class="smallcaps">Kosmos-2.5</span> enhances its general-purpose
multimodal literate capabilities. We assess the performance of
<span class="smallcaps">Kosmos-2.5</span> on two tasks: end-to-end
document-level text recognition and markdown-formatted image-to-text
generation. Experiment results have demonstrated strong literate
performance on several text-intensive image understanding tasks. In
addition, <span class="smallcaps">Kosmos-2.5</span> also demonstrates
promising capabilities in few-shot and zero-shot learning scenarios,
offering a universal interface for real-world applications that involve
text-rich images.

The contributions of this work are summarized as follows:

-   <span class="smallcaps">Kosmos-2.5</span> represents a significant
    paradigm shift in text image understanding, transitioning from
    encoder-only/encoder-decoder models to a decoder-only model. It is
    pre-trained by incorporating dual transcription tasks
    (spatially-aware text block generation and structured markdown text
    generation) into a single, unified model architecture.

-   This innovative method streamlines the application interface by
    integrating generative multimodal language modeling, simplifying the
    traditionally complex cascaded pipelines used for various downstream
    tasks.

-   Furthermore, <span class="smallcaps">Kosmos-2.5</span> demonstrates
    impressive multimodal literate capabilities, thus setting the stage
    for future scaling of multimodal large language models.

<div class="figure*" markdown="1">

<embed src="/papers/visionrich_small_dec/arXiv-2309.11419v1_md/images/kosmos25-arch5.png" style="width:100.0%" />

</div>

# <span class="smallcaps">Kosmos-2.5</span>

## Model Architecture

The model architecture of <span class="smallcaps">Kosmos-2.5</span>
consists of a pre-trained vision encoder and a language decoder
connected with a resampler module, shown in
<a href="#fig:model_arch" data-reference-type="ref+label"
data-reference="fig:model_arch">[fig:model_arch]</a>. We adopt the
pre-trained vision encoder based on the Vision Transformer
(ViT) [vit](http://arxiv.org/pdf/2105.15075v2). We further adapt a Perceiver Resampler
module with an attentive pooling mechanism to reduce the size of image
embeddings [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1). The language decoder is
built upon the Transformer-based decoder to condition on image and text
context for the next token prediction.

## Image and Text Representations

<span class="smallcaps">Kosmos-2.5</span> takes a composite input
consisting of an image and a text representation. **The image
representation** is uniform across various configurations and leverages
a variable-resolution input strategy following
Pix2Struct [lee2023pix2struct](http://arxiv.org/pdf/2210.03347v2). Precisely, we extract the
maximum number of fixed-size patches ($16 \times 16$) that can fit
within a predefined sequence length $L$. In addition,
Resampler [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1) is used as an attentive
pooling mechanism to reduce the number of image embeddings. **The text
representation**, however, is more versatile and can be one of two
types: text lines with bounding boxes or plain markdown texts.

**Text lines with bounding boxes:** For the layout-based document
representation, text lines and their associated bounding boxes are
extracted. Inspired by
<span class="smallcaps">Kosmos-2</span> [peng2023kosmos](http://arxiv.org/pdf/2305.16103v1),
we ground the text lines to their spatial positions in images by
aligning their representations. The coordinates of these bounding boxes
are then converted into discrete location tokens. Given that $L$ also
represents the maximum length for each image dimension, we introduce a
set of $2L+2$ specialized tokens. These tokens, `<x`$_0$`>`,
`<x`$_1$`>`, …, `<x`$_{L-1}$`>`, `<y`$_0$`>`, …, `<y`$_{L-1}$`>`,
`<bbox>`, and `</bbox>`, correspond to the coordinates and the start and
end of a bounding box. The coordinates are obtained by rounding down the
actual position after resizing images. Consider a document $T$ that
comprises $N$ text lines. Each line is represented as
$\mathbf{T}_n = \{ w_1^{(n)}, w_2^{(n)}, \ldots, w_{M_n}^{(n)} \}$,
where $M_n$ is the number of words in the $n$-th text line. The bounding
box for $\mathbf{T}_n$ is then denoted by
$\mathbf{B}_n = \texttt{<bbox><} x_{\text{tl}}^{(n)} \texttt{><} y_{\text{tl}}^{(n)} \texttt{><} x_{\text{br}}^{(n)} \texttt{><} y_{\text{br}}^{(n)} \texttt{></bbox>}$,
which includes coordinates for its top-left and bottom-right corners.

**Markdown texts:** For the markup-based document representation where
the output text is in the markdown format, the text component captures
both content and formatting markup. Unlike layout-based documents,
markdown text does not require bounding boxes. Instead, the text is
directly tokenized, retaining all special characters and formatting
indicators.

To facilitate these diverse input types, we employ different composite
representations. For image-text pairs with text lines and bounding
boxes, the input is denoted as `<s><image>Image Embedding</image>`
$\bigcup_{n=1}^{N}$ ($\mathbf{B}_n \oplus \mathbf{T}_n)$ `</s>`. The
operator $\oplus$ represents the concatenation of the text line
$\mathbf{T}_n$ and its bounding box $\mathbf{B}_n$. Conversely, when the
text is in the markdown format, the input simplifies to
`<s><image>Image Embedding</image>Markdown Text</s>`. In both cases,
`<s>` and `</s>` signify the sequence boundaries, while `<image>` and
`</image>` indicate the beginning and end of image embeddings. This
flexibility in text representation allows
<span class="smallcaps">Kosmos-2.5</span> to apply to various document
analysis tasks.

## Pre-training Data

The pre-training process enables
<span class="smallcaps">Kosmos-2.5</span> to learn versatile
representations suitable for various text-intensive image understanding
tasks. The model is pre-trained on a rich array of datasets from diverse
sources. Traditional Optical Character Recognition (OCR) task is
primarily geared towards generating text content and its 2D positions
within an image. However, they often neglect the need to maintain the
order and structural integrity of the original document, which is
essential for text-intensive image understanding tasks involving
structured information.

To address this, we steer <span class="smallcaps">Kosmos-2.5</span> to
excel in two distinct yet cooperative transcription tasks: (1)
generating spatially-aware text blocks, where each block of text is
assigned its spatial coordinates within the image, and (2) producing
structured text output that captures styles and structures into the
markdown format. Markdown provides an advantage over plain text by
explicitly distinguishing different structural elements, such as tables
and lists, with specific tokens. For example, table cells can be denoted
with vertical bars (\|) and list items with bullets (\*, -, or +). It
also standardizes the representation of typographic emphases like bold
(\*\*bold\*\*) and italics (\*italics\*), integrating the learning of
document structure with natural language understanding in a unified
model.

For spatially-aware text blocks, we use:

-   **IIT-CDIP:** The IIT-CDIP dataset is a large-scale public
    collection comprising scanned document images. We used approximately
    27.6 million pages to train our model.

-   **arXiv papers:** arXiv, an open-access research-sharing platform,
    provides another significant data source, accounting for roughly
    20.9 million pages. We downloaded a bulk of data, consisting of PDF
    and LaTeX source files, from the official arXiv repository[^2].

-   **PowerPoint slides:** A corpus of 6.2 million pages is collected
    from various web pages containing PowerPoint documents,
    significantly enhancing the diversity of our training data.

-   **General PDF:** Additionally, we crawled the web for diverse
    open-domain digital PDF files, leading to the collection of a large
    corpus comprising approximately 155.2 million pages.

-   **Web screenshots:** A subset of the mC4 webpages is scraped and
    rendered as screenshots containing almost 100 million pages.

For structured text output in markdown format, we use:

-   **README:** We collect 2.9 million “README.md” files from
    open-source GitHub projects, primarily written in markdown format.

-   **DOCX:** We also extract 1.1 million DOCX pages from millions of
    WORD files crawled from the web. The DOCX pages are converted to
    markdown format, and each page corresponds to its markdown
    information.

-   **LaTeX:** A subset of the entire arXiv papers is used to extract
    the mapping of PDF pages and its corresponding markdown information
    converted from the LaTeX code, which contains a total of 3.7 million
    pages.

-   **HTML:** We obtain 6.3 million HTML files from the aforementioned
    mC4 subset and convert them into markdown format.

## Data Processing [section:dp]

The pre-training data has a wide coverage, and each type of data
requires a different processing workflow, which is introduced as
follows:

#### IIT-CDIP

The IIT-CDIP dataset mainly consists of scanned document images. We use
the Microsoft Read API [^3] to extract text and layout information.

#### arXiv papers, PowerPoint slides, General PDF

We first compile and convert arXiv papers and PowerPoint slides into PDF
files. Together with other general PDFs, we employed the PyMuPDF parser
[^4] to extract text and layout information efficiently.

#### Web screenshots

We also include webpage screenshots in the model pre-training to
diversify the layout distribution further. We collect the webpage URLs
from the English portion of the mC4 dataset. Playwright [^5] is used to
access a specified URL and open the webpage. The HTML content of the
page is extracted and parsed using the lxml library [^6] to obtain a
Document Object Model (DOM) tree representation. This DOM tree is
traversed, examining the XPath of each element within it. This traversal
aims to determine whether each element is visible and retrieve
information about its bounding boxes.

#### README (markdown)

In addition to layout-based data, we collect markup-based data for the
pre-training. We collect “README.md” files from many GitHub projects and
convert these files into HTML using Pandoc [^7]. Then, wkhtmltopdf [^8]
is used to obtain the images from the generated HTML content.

#### DOCX (markdown)

The Microsoft Office WORD files have been extensively used in existing
research like TableBank [li2020tablebank](https://arxiv.org/pdf/1903.01949) and
ReadingBank [wang2021layoutreader](http://arxiv.org/pdf/2108.11591v2). We collect WORD DOCX
files and convert them into texts with markdown. First, we use Pandoc to
convert the XML content within the DOCX files into markdown files. As
Pandoc keeps the “\<table\>” tags to represent the tabular cells in the
generated markdown, we further identify all the tables and use
markdownify [^9] to convert them into the markdown formats. Finally, the
original DOCX files are converted into PDF files, and each page is
aligned to the corresponding span of the markdown content based on a
heuristic method.

#### LaTeX (markdown)

LaTeX documents from arXiv have been used to generate PDF files to
obtain texts with bounding boxes. Meanwhile, we also convert the
LaTeX content into the markdown texts. Similar to
Nougat [blecher2023nougat](https://arxiv.org/pdf/2308.13418), LaTeXML [^10] is used to
convert the LaTeX code into the HTML sequence, which is further
transformed into the markdown format. Different from Nougat, we keep all
the tables at the beginning of the page as most LaTeX users prefer to
position tables with “\[t\]” or “\[h\]” instead of “\[b\]”. Meanwhile,
we also convert the table content from the LaTeX format into the
markdown format.

#### HTML (markdown)

The most straightforward way to obtain markdown resources from HTML
webpages is through web scraping. However, webpages are often cluttered
with various layouts and styles, resulting from the misuse of HTML tags.
Moreover, HTML pages may include extraneous elements, such as
advertisements, navigation menus, or formatting elements, making
extracting clean and meaningful content challenging. To overcome these
obstacles, we employ Playwright, a fast and reliable end-to-end testing
framework for the web. The library allows us to navigate the HTML
structure, filter out non-essential elements, and extract the relevant
text content. We also apply custom rules and regular expressions to
further refine the extracted text and format it as markdown, ensuring
that the resulting markdown files are coherent and readable.

<div id="tab:pretraining_data" markdown="1">

| **Task** | **Data Source** | **Number of Pages** | **Sampling Ratio** |
|:---|:---|:---|:---|
| Layout-based (texts+bboxes) | IIT-CDIP | 27.6M | 10% |
|  | arXiv papers | 20.9M | 5% |
|  | PowerPoint slides | 6.2M | 5% |
|  | General PDF | 155.2M | 20% |
|  | Web screenshots | 100.5M | 10% |
| Markup-based (texts+markdown) | README | 2.9M | 15% |
|  | DOCX | 1.1M | 10% |
|  | LaTeX | 3.7M | 15% |
|  | HTML | 6.3M | 10% |
| **Total** |  | 324.4M | 100% |

Summary of Pre-training Data in
<span class="smallcaps">Kosmos-2.5</span>

</div>

## Filtering and Quality Control

We employ fastText for language identification (with a threshold of 0.5)
to filter out non-English documents from the entire pre-training
dataset. To ensure content diversity within each source, we utilize the
MinHash [broder1997resemblance](http://arxiv.org/pdf/2103.07007v1) to identify and remove
redundant pages. We use the same parameters
as [lee2021deduplicating](http://arxiv.org/pdf/2107.06499v2) and a document pair with
similarity 0.8 will be marked as duplicate. A comprehensive breakdown of
the pre-training data, along with their respective sampling ratios, is
provided in
<a href="#tab:pretraining_data" data-reference-type="ref+label"
data-reference="tab:pretraining_data">1</a>. When dealing with
image-to-markdown data from README, DOCX, LaTeX, and HTML sources, we
observe discrepancies between the content in text images and their
corresponding markdown sequences due to conversion issues. Consequently,
we refine the data by evaluating token overlap between images and
markdown files, requiring a token intersection-to-union ratio greater
than 0.95 for inclusion.
Section <a href="#supp:data" data-reference-type="ref"
data-reference="supp:data">6.2</a> shows some of the training samples.

# Experiments

## Evaluation

#### Text Recognition

We utilize word-level *precision* (# or correct matches over the number
of detected words), *recall* (# of correct matches over the number of
ground truth words), and *f1* as the metrics to evaluate the text
recognition performance. If there are repeated words in the ground
truth, they are expected to be repeated in the prediction. Text
recognition is evaluated on three benchmark datasets, including
FUNSD [jaume2019funsd](https://arxiv.org/pdf/1905.13538),
SROIE [huang2019icdar2019](http://arxiv.org/pdf/2103.10213v1) and
CORD [park2019cord](http://arxiv.org/pdf/2103.10213v1). We compare
<span class="smallcaps">Kosmos-2.5</span> to the text recognition
results from Document OCR in Google Document AI [^11].

#### Image-to-markdown Generation

In light of the unique nature of the image-to-markdown conversion task,
assessing the quality of the generated markdown necessitates specialized
metrics. We adopt a two-fold evaluation scheme: Normalized Edit Distance
(NED) and Normalized Tree Edit Distance (NTED), considering both the
lexical accuracy and the preservation of the original structural
elements.

The NED is formulated as
$$\textit{NED} = 1-\frac{1}{N} \sum_{i=1}^N D\left(s_i, \hat{s}_i\right) / \max \left(\mathrm{len}(s_i), \mathrm{len}(\hat{s}_i\right))$$
where $N$, $s$, and $\hat{s}$ denote the number of samples, prediction,
and ground truth, respectively. $D(\cdot,\cdot)$ and
$\mathrm{len}(\cdot)$ represent the edit distance function and the
length of a string. The *NED* value ranges from 0 to 1, with a higher
*NED* value indicating the prediction is closer to the ground truth.

However, given the hierarchical structure inherent to markdown, relying
solely on a string-based comparison metric like NED can be insufficient.
Thus, we adopt NTED as an additional evaluation metric for structural
differences. NTED is a tree edit distance normalized by the number of
nodes in the tree, considering the structural discrepancies between
parse trees. Specifically, the predicted markdown sequence is first
transformed into an HTML tree. Then, the tree edit distance between the
prediction and the ground truth is calculated using the ZSS algorithm
[zhang1989simple](http://arxiv.org/pdf/1703.08940v1). The NTED is formulated as
$$\textit{NTED} = 1-\frac{1}{N} \sum_{i=1}^N \mathrm{TD}\left(t_i, \hat{t}_i\right) / \max \left(\mathrm{node}(t_i), \mathrm{node}(\hat{t}_i\right))$$
where $N$, $t$, and $\hat{t}$ signify the number of samples, the HTML
tree of prediction, and the HTML tree of ground truth, respectively.
Besides, $\mathrm{TD}(\cdot,\cdot)$ and $\mathrm{node}(\cdot)$ stand for
the tree edit distance function and the number of nodes in a tree.

We create three datasets to evaluate the image-to-markdown task from
different data sources, including document-level markdown generation,
README markdown generation and table markdown generation. Each dataset
includes 1,000 $\langle$image, markdown$\rangle$ pairs, which are held
out from the pre-training data. We compare
<span class="smallcaps">Kosmos-2.5</span> to the markdown generated by
the Nougat [blecher2023nougat](https://arxiv.org/pdf/2308.13418) base and small models.

## Implementation Details

We employ the AdamW optimizer [loshchilov2017decoupled](http://arxiv.org/pdf/2311.11446v2)
with $\beta=(0.9,0.98)$ for optimization, setting the weight decay to
0.01 and the dropout rate to 0.1. The learning rate is warmed up to
$2 \times 10^{-4}$ during the initial 375 steps, followed by a linear
decay to zero throughout the remaining training steps. The batch size is
adjustable to align with the available computational resources and
specific training requirements.
<span class="smallcaps">Kosmos-2.5</span> contains a total of 1.3
billion parameters. The vision encoder is initialized from the encoder
of the Pix2Struct-Large model. The language decoder includes 24
Transformer layers with a hidden size of 1,536, an FFN intermediate size
of 6,144, and 16 attention heads.
Section <a href="#supp:para" data-reference-type="ref"
data-reference="supp:para">6.1</a> shows more details of the training
hyperparameters.

Due to the substantially larger quantity of available layout-based data
than markup-based data, we initially trained the model for 100k steps
exclusively using the layout-based dataset. Subsequently, the two
datasets were combined for further training of 140k steps. Additionally,
we incorporate the training split of the evaluation dataset into the
entire pre-training data, extending the process by an additional 10k
steps. For text tokenization, we utilize
SentencePiece [kudo2018sentencepiece](http://arxiv.org/pdf/1808.06226v1) and adopt the
“full-sentence” format [liu2019roberta](http://arxiv.org/pdf/1907.11692v1). This approach
packs each input sequence with full sentences, continuously sampled from
one or multiple documents. Newly added word embeddings of location
tokens are randomly initialized, with all parameters updated during
training. We also leverage the data augmentation approaches from
TrOCR [li2022trocr](https://arxiv.org/pdf/2109.10282) in the training to make models more
robust.

Throughout the evaluation process, model inference is conducted using a
single model checkpoint across various evaluation datasets with the
corresponding task prompt respectively, demonstrating that our approach
does not necessitate individualized model fine-tuning for each dataset.

## Results

<span class="smallcaps">Kosmos-2.5</span> is a flexible framework that
facilitates multitasking, with tasks determined by the provided task
prompts. Experimental results are demonstrated in Table
<a href="#tab:ocr_results" data-reference-type="ref"
data-reference="tab:ocr_results">2</a> and Table
<a href="#tab:md_results" data-reference-type="ref"
data-reference="tab:md_results">3</a>. Specifically, for the text
recognition task, our <span class="smallcaps">Kosmos-2.5</span>
outperforms Google Document OCR by 0.33%, 2.45%, and 1.35% in terms of
the F1 score, showcasing its effectiveness. For the image-to-markdown
task, it is worth noting that our method significantly outperforms the
Nougat [blecher2023nougat](https://arxiv.org/pdf/2308.13418). For example,
<span class="smallcaps">Kosmos-2.5</span> achieves a notable improvement
of 33.68% (95.09% vs 61.41%) over $\text{Nougat}_{\text{\,BASE}}$ in
terms of NED on the README dataset. Besides, regarding NTED,
<span class="smallcaps">Kosmos-2.5</span> also boosts the performance by
33.38% (82.08% vs 48.70%) compared with $\text{Nougat}_{\text{\,BASE}}$
on the Documents dataset. We attribute the performance boost to the
increased diversity of our training data compared to Nougat, which
primarily focuses on the academic paper domain. Notably, the greater
diversity in our training data significantly enhances our model’s
comprehension of different document types and strengthens its
generalization capabilities. In summary, the experimental results
validate the remarkable capabilities of
<span class="smallcaps">Kosmos-2.5</span> in various tasks.

<div id="tab:ocr_results" markdown="1">

| **Dataset** | **FUNSD** | **SROIE** | **CORD** |
|:--:|:--:|:--:|:--:|
| 2-4   | **P / R / F1** | **P / R / F1** | **P / R / F1** |
| Commercial OCR | **85.12** / 80.86 / 82.93 | 89.68 / 89.69 / 89.69 | 81.95 / 86.87 / 84.34 |
| <span class="smallcaps">Kosmos-2.5</span>$^\dagger$ | 83.88 / **82.66** / **83.26** | **91.72 / 92.57 / 92.14** | **83.64 / 87.83 / 85.69** |

Experimental results on text recognition using Precision (%), Recall
(%), F1 (%), where model inference is conducted with the layout task
prompt. $^\dagger$<span class="smallcaps">Kosmos-2.5</span> does not
require task-specific fine-tuning.

</div>

<div id="tab:md_results" markdown="1">

| **Dataset** | **General Documents** | **README** | **Tables** |
|:--:|:--:|:--:|:--:|
| 2-4   | **NED / NTED** | **NED / NTED** | **NED / NTED** |
| $\text{Nougat}_{\text{\,SMALL}}$ [blecher2023nougat](https://arxiv.org/pdf/2308.13418)$^\dag$ | 82.80 / 48.96 | 58.58 / 35.49 | 68.33 / 61.52 |
| $\text{Nougat}_{\text{\,BASE}}$ [blecher2023nougat](https://arxiv.org/pdf/2308.13418)$^\dag$ | 83.75 / 48.70 | 61.41 / 36.41 | 68.53 / 61.60 |
| <span class="smallcaps">Kosmos-2.5</span>$^\ddagger$ | **91.59** / **82.08** | **95.09** / **91.18** | **85.14** / **90.64** |

Experimental results on image-to-markdown using NED (%) and NTED (%),
where model inference is conducted with the markup task prompt.
$^\dag$Nougat [blecher2023nougat](https://arxiv.org/pdf/2308.13418) generates the table
content in the LaTeX format, which is converted to the markdown format
for fair comparison.
$^\ddagger$<span class="smallcaps">Kosmos-2.5</span> does not require
task-specific fine-tuning.

</div>

## Discussion

<figure id="fig:exp">
<figure id="fig:models_in">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:models_ocr">

<figcaption>Using the layout prompt</figcaption>
</figure>
<p> </p>
<figure id="fig:models_md">

<figcaption>Using the markup prompt</figcaption>
</figure>
<figcaption>Model outputs from <span class="smallcaps">Kosmos-2.5</span>
with different task prompts given the same input text
image.</figcaption>
</figure>

We illustrate an example in
<a href="#fig:exp" data-reference-type="ref+label"
data-reference="fig:exp">4</a>, showcasing the model outputs produced by
<span class="smallcaps">Kosmos-2.5</span> with various task prompts when
presented with the same input text image. As shown in the figure, the
model generates distinct outputs depending on the task prompts it
receives. When given the layout task prompt, the model produces the
following text sequence, which includes textual content and
corresponding bounding boxes:

```
[x_52] [y_113] [x_756] [y_145]: NYC Department of Education School Year Calendar 2023-2024
[x_52] [y_159] [x_826] [y_181]: This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private,
[x_52] [y_180] [x_820] [y_202]: parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact
[x_52] [y_201] [x_639] [y_223]: your child's school for information about their calendar. Please note the following:
[x_65] [y_223] [x_77] [y_245]: $\bullet$
[x_92] [y_223] [x_825] [y_245]: On days when school buildings are closed due to inclement weather or other emergencies, all students

... 
```

With the markup task prompt, the model generates another text sequence
that follows the markdown format:

```
# NYC Department of Education School Year Calendar 2023-2024

This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private, parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact your child's school for information about their calendar. Please note the following:
... 
-   On this schedule, **elementary schools** are defined as programs that serve kindergarten (K) through grade 8, including schools with 3-K and Pre-K programs, as well as those that end in grade 5. **Middle schools** are defined as programs that serve grades 6-8, and **high schools** are defined as programs that serve grades 9-12.

...
```

It is apparent that <span class="smallcaps">Kosmos-2.5</span> excels in
precisely identifying text positions and recognizing text content.
Moreover, it adeptly captures the styles and structures present within
the text image, including elements like titles, bullet points, tables,
and bold text. Section <a href="#supp:example" data-reference-type="ref"
data-reference="supp:example">6.3</a> provides the full output sequence
using different task prompts for this example.

<span class="smallcaps">Kosmos-2.5</span> provides a unified
architecture and interface for text image understanding, making it
versatile for various application scenarios. Firstly, it can be
fine-tuned as a single model for a wide range of text image
understanding tasks, including information extraction, layout detection
and analysis, visual question answering, screenshot understanding, UI
automation, and many others. This unified model interface significantly
streamlines downstream task training and enables the model to
effectively follow instructions in real-world applications. Secondly,
our solution is compatible with more powerful LLMs like GPT-3.5 or
GPT-4. The output from our model can serve as contexts for LLMs,
enhancing their capabilities through further prompt engineering. This
approach empowers LLMs with robust text image understanding
capabilities. Thirdly, we have the potential to augment the pre-training
with textual data, transforming it into a general-purpose MLLM. This
expanded model not only processes visual signals but also possesses
strong language understanding capabilities.

# Related Work

## Multimodal Large Language Models

The flourishing blossom of large language models (LLM), represented by
ChatGPT [chatgpt](https://openai.com/blog/chatgpt), has revolutionized artificial
intelligence and significantly impacted numerous downstream tasks such
as text translation, code generation, question answering, etc. Despite
the rapid development, it is significant to recognize that the human
perception of the world is not limited to language alone but encompasses
a wide range of modalities, with particular emphasis on the visual
modality. Many research works attempt to “bring eyes” to LLM and develop
multimodal large language models (MLLM), which can be categorized into
LLM-centric scheduling systems and end-to-end trainable multimodal
systems.

The LLM-centric scheduling system
[wu2023visual](http://arxiv.org/pdf/2303.04671v1), [yang2023mm](http://arxiv.org/pdf/2303.11381v1), [liang2023taskmatrix](http://arxiv.org/pdf/2303.16434v1), [shen2023hugginggpt](http://arxiv.org/pdf/2303.17580v4), [liu2023internchat](http://arxiv.org/pdf/2012.09130v1), [suris2023vipergpt](http://arxiv.org/pdf/1905.11127v1), [chen2023language](http://arxiv.org/pdf/2310.15166v1)
takes advantage of many vision foundation models (e.g., Stable Diffusion
[rombach2022high](http://arxiv.org/pdf/2307.10094v1), ControlNet
[zhang2023adding](http://arxiv.org/pdf/2210.12192v1), BLIP [li2022blip](http://arxiv.org/pdf/2311.01038v2),
etc.), and schedules these models in a language-centric manner. For
example, Visual ChatGPT [wu2023visual](http://arxiv.org/pdf/2303.04671v1) develops a set of
prompts to incorporate visual information into ChatGPT, enabling users
to draw or edit images through chatting. MM-REACT
[yang2023mm](http://arxiv.org/pdf/2303.11381v1) leverages vision experts to augment its
multimodal capabilities by incorporating a textual prompt design that
can effectively represent various visual signals, including text
descriptions, coordinates, and aligned file names, for images and
videos. HuggingGPT [shen2023hugginggpt](http://arxiv.org/pdf/2303.17580v4) connects LLMs
with extensive AI models in machine learning communities, tackling user
requests through ChatGPT’s task planning, model selection, and response
summarization capabilities. Further, TaskMatrix.AI
[liang2023taskmatrix](http://arxiv.org/pdf/2303.16434v1) largely extends the scale and
connects foundation models with millions of APIs for solving tasks in
both digital and physical domains. Differently, InternGPT
[liu2023internchat](http://arxiv.org/pdf/2012.09130v1) incorporates pointing instructions
(e.g., clicking and dragging) for better communication between chatbots
and users, while also improving the accuracy of chatbots in performing
vision-centric tasks. Nevertheless, this approach has several
limitations, such as the expenses associated with API calls or the
storage space required for the pre-trained weights of foundation models.

End-to-end trainable multimodal system
[metalm](http://arxiv.org/pdf/0911.2327v1), [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1), [huang2023language](http://arxiv.org/pdf/2302.14045v2), [peng2023kosmos](http://arxiv.org/pdf/2305.16103v1), [huang2021seeing](http://arxiv.org/pdf/2402.17510v1), [xue2021probing](http://arxiv.org/pdf/1911.03875v3), [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1), [huang2023sparkles](http://arxiv.org/pdf/2308.16463v2), [li2023blip](http://arxiv.org/pdf/2301.12597v3), [dai2023instructblip](https://arxiv.org/pdf/2305.06500), [liu2023visual](http://arxiv.org/pdf/2402.11690v1), [luo2023cheap](http://arxiv.org/pdf/2210.09175v1), [wang2023visionllm](http://arxiv.org/pdf/2312.13503v1), [su2023pandagpt](http://arxiv.org/pdf/1808.10000v1), [zhang2023llama](http://arxiv.org/pdf/2207.10858v1), [gao2023llama](http://arxiv.org/pdf/2303.16199v2), [koh2023grounding](http://arxiv.org/pdf/2401.13388v2), [li2023otter](http://arxiv.org/pdf/2311.00233v2)
integrates vision and language models into a unified model, which are
further trained on multimodal datasets. For instance, Flamingo
[alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1) leverages gated cross-attention to
fuse pre-trained vision and language models, showing impressive ability
in downstream multimodal tasks. Besides, BLIP-2
[li2023blip](http://arxiv.org/pdf/2301.12597v3) utilized Q-Former to align the visual
features with a large language model. Furthermore, Instruct-BLIP
improves the training of Q-Former by introducing a novel
instruction-aware visual feature extraction method. Based on this
design, MiniGPT-4 [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1) uses Vicuna
[vicuna2023](https://lmsys.org/blog/2023-03-30-vicuna/) as the text encoder and fine-tunes detailed
image descriptions to better match user intent. Sparkles unlocks
multimodal instruction-following models’ capabilities in open-ended
dialogues involving multiple images [huang2023sparkles](http://arxiv.org/pdf/2308.16463v2).
LLaVA [liu2023visual](http://arxiv.org/pdf/2402.11690v1) injects visual features into the
language model by treating image tokens as a foreign language, and uses
conversation generated by GPT-4 [gpt4](https://openai.com/gpt-4) for fine-tuning.
<span class="smallcaps">Kosmos-1</span>
[huang2023language](http://arxiv.org/pdf/2302.14045v2) is trained from scratch using
web-scale corpora while showing impressive performance in zero-shot,
few-shot, and multimodal chain-of-thought prompting settings.
Analogously, <span class="smallcaps">Kosmos-2</span>
[peng2023kosmos](http://arxiv.org/pdf/2305.16103v1) incorporates grounding and referring
abilities and can accept image regions users select using bounding boxes
as input. mPLUG-Owl [ye2023mplug](http://arxiv.org/pdf/2405.00390v2) efficiently fine-tunes
the language model using low-rank adaption with multimodal instruction
datasets. Otter [li2023otter](http://arxiv.org/pdf/2311.00233v2) is built using Flamingo and
aims to explore multimodal in-context learning capabilities.

## Text Image Understanding

Text image understanding is a cutting-edge technology that harnesses the
power of artificial intelligence, including natural language processing
and computer vision, to automatically comprehend, categorize, and
extract information from documents [cui2021document](https://arxiv.org/pdf/2111.08609). Any
file containing written or printed characters can be considered a
document, including web pages, slides, posters, and even scene text
images. Documents are ubiquitous in our daily lives, so the research on
documents is significant.

Before the deep learning era, researchers used rule-based heuristic
approaches for document analysis
[wong1982document](http://arxiv.org/pdf/2402.11048v1), [o1993document](http://arxiv.org/pdf/2305.08719v2). They manually observed
layout information and summarized heuristic rules, but these methods are
not scalable and require enormous labour costs. Subsequently, the rise
of deep learning has led to significant advancements in the field of
Document AI
[xu2020layoutlm](http://arxiv.org/pdf/2305.18721v2), [xu-etal-2021-layoutlmv2](https://doi.org/10.18653/v1/2021.acl-long.201), [xu2021layoutxlm](https://arxiv.org/pdf/2104.08836), [huang2022layoutlmv3](http://arxiv.org/pdf/2204.08387v3), [chen2022xdoc](http://arxiv.org/pdf/2310.16527v1), [li2021markuplm](http://arxiv.org/pdf/2110.08518v2), [li2022dit](http://arxiv.org/pdf/2310.16527v1), [li2021selfdoc](http://arxiv.org/pdf/2009.14457v2), [appalaraju2021docformer](http://arxiv.org/pdf/2309.05503v1), [wang2022lilt](http://arxiv.org/pdf/2202.13669v1), [gu2022xylayoutlm](http://arxiv.org/pdf/2203.13530v2), [li2021structurallm](http://arxiv.org/pdf/2311.01038v2), [yu2023structextv2](http://arxiv.org/pdf/2310.16527v1).
For example, LayoutLM series
[xu2020layoutlm](http://arxiv.org/pdf/2305.18721v2), [xu-etal-2021-layoutlmv2](https://doi.org/10.18653/v1/2021.acl-long.201), [huang2022layoutlmv3](http://arxiv.org/pdf/2204.08387v3)
employs large-scale document data for pre-training and incorporates
text, layout, and image information into the model, showing impressive
performance in downstream tasks like key information extraction and
document question answering. Similarly, DocFormer
[appalaraju2021docformer](http://arxiv.org/pdf/2309.05503v1) introduces an additional task
to reconstruct the document image during pre-training.
Donut [kim2021donut](http://arxiv.org/pdf/2202.00470v1) introduces an OCR-free document
understanding Transformer, directly mapping an input document image to
the desired output with OCR. MarkupLM [li2021markuplm](http://arxiv.org/pdf/2110.08518v2)
takes advantage of large-scale webpages from Common Crawl and uses
node-level hierarchical structure information as the pre-training
objective. XDoc [chen2022xdoc](http://arxiv.org/pdf/2310.16527v1) introduces a unified
framework for tackling multiple document formats in one model for
parameter efficiency. UDOP [tang2023unifying](http://arxiv.org/pdf/2212.02623v3) designs a
unified model that integrates text, image, and layout modalities,
showing impressive performance on diverse document understanding tasks.
Pix2Struct [lee2023pix2struct](http://arxiv.org/pdf/2210.03347v2) is a pre-trained
image-to-text model trained to parse masked screenshots of web pages
into simplified HTML.

Despite significant progress in text image understanding, most models
are designed for specific tasks and lack generalizability. On the
contrary, the proposed <span class="smallcaps">Kosmos-2.5</span>
represents an important step forward in this field, demonstrating the
potential of MLLM in achieving robust and generalizable performance
across a wide range of text image types.

# Conclusion and Future Work

We introduced <span class="smallcaps">Kosmos-2.5</span>, a multimodal
literate model built on the strengths of
<span class="smallcaps">Kosmos-2</span>, designed to enhance machine
understanding of text-intensive images. This model shifted from
conventional encoder-only/encoder-decoder models to a more unified,
decoder-only architecture. The shift to generative multimodal language
modeling simplifies task interfaces, eliminating the need for complex,
task-specific pipelines. Moreover,
<span class="smallcaps">Kosmos-2.5</span> demonstrated potential in
few-shot and zero-shot learning capabilities, laying a foundation for
future advances and scalability in multimodal literate models.

Despite these promising results, our current model faces some
limitations, offering valuable future research directions. For instance,
<span class="smallcaps">Kosmos-2.5</span> currently does not support
fine-grained control of document elements’ positions using natural
language instructions, despite being pre-trained on inputs and outputs
involving the spatial coordinates of text. Instruction tuning could
offer a promising route to enhance this aspect of the model, leading to
broader application capabilities. Furthermore, documents spanning
multiple pages pose a challenge as they typically demand holistic
processing and comprehension. Meanwhile, it is also feasible that
<span class="smallcaps">Kosmos-2.5</span> allows for multiple image
pages interleaved with text as input; however, managing long context
windows remains a vital issue we aim to address in future work.

In the broader research landscape, a significant direction lies in
furthering the development of model scaling capabilities. With an
expanding spectrum of tasks and rising complexities, scaling up the
model to handle larger volumes of data is crucial for the progression of
multimodal literate models. Ultimately, our goal is to develop a model
that effectively interprets both visual and textual data, and
generalizes smoothly across an expanded array of text-intensive
multimodal tasks.

# Acknowledgement [acknowledgement]

We would like to acknowledge Zhiliang Peng for the helpful discussions.

# Supplementary Material

## Hyperparameters [supp:para]

The settings of hyperparameters are demonstrated in
<a href="#tab:hyperparameters" data-reference-type="ref+label"
data-reference="tab:hyperparameters">5</a>.

<div class="subtable" markdown="1">

0.45

<div id="tab:hyperparameters" markdown="1">

| **Hyperparameters**   |                                              |
|:----------------------|:--------------------------------------------:|
| Number of layers      |                      24                      |
| Hidden size           |                    1,536                     |
| FFN inner hidden size |                    6,144                     |
| Attention heads       |                      16                      |
| Activation function   | GeLU [hendrycks2016gaussian](http://arxiv.org/pdf/1606.08415v5) |
| Vocabulary size       |                   108,481                    |
| Soft tokens V size    |                    2,048                     |
| Max sequence length   |                    4,096                     |
| Initialization        | Magneto [wang2022foundation](http://arxiv.org/pdf/2304.02263v2) |

Hyperparameters of <span class="smallcaps">Kosmos-2.5</span>

</div>

</div>

<div class="subtable" markdown="1">

0.45

<div id="tab:hyperparameters" markdown="1">

| **Hyperparameters** |             |
|:--------------------|:-----------:|
| Training steps      |   200,000   |
| Warmup steps        |     375     |
| Batch size          |    1,024    |
| Optimizer           |    AdamW    |
| Learning rate       |    2e-4     |
| Learning rate decay |   Linear    |
| Adam $\beta$        | (0.9, 0.98) |
| Weight decay        |    0.01     |
| Dropout             |     0.1     |

Hyperparameters of <span class="smallcaps">Kosmos-2.5</span>

</div>

</div>

## Data Samples [supp:data]

We demonstrate some of the training samples in
<span class="smallcaps">Kosmos-2.5</span>, which include the input and
output from IIT-CDIP, arXiv papers, PowerPoint slides, general PDFs, web
screenshots, README, DOCX, LaTeX  and HTML.

<figure id="fig:ocr_cdip">
<figure id="fig:cdip_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:cdip_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the layout-based task from
IIT-CDIP</figcaption>
</figure>

<figure id="fig:ocr_arxiv1">
<figure id="fig:arxiv1_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:arxiv1_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the layout-based task from arXiv
papers (single-column)</figcaption>
</figure>

<figure id="fig:ocr_arxiv2">
<figure id="fig:arxiv_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:arxiv_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the layout-based task from arXiv
papers (two-column)</figcaption>
</figure>

<figure id="fig:ocr_ppt">
<figure id="fig:ppt_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:ppt_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the layout-based task from PowerPoint
slides</figcaption>
</figure>

<figure id="fig:ocr_pdf">
<figure id="fig:pdf_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:pdf_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the layout-based task from
PDFs</figcaption>
</figure>

<figure id="fig:ocr_screen">
<figure id="fig:screen_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:screen_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the layout-based task from web
screenshots</figcaption>
</figure>

<figure id="fig:md_readme">
<figure id="fig:readme_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:readme_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the markup-based task from
README</figcaption>
</figure>

<figure id="fig:md_docx">
<figure id="fig:docx_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:docx_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the markup-based task from
DOCX</figcaption>
</figure>

<figure id="fig:md_latex1">
<figure id="fig:latex1_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:latex1_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the markup-based task from
LaTeX (single-column)</figcaption>
</figure>

<figure id="fig:md_latex2">
<figure id="fig:latex_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:latex_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the markup-based task from
LaTeX (two-column)</figcaption>
</figure>

<figure id="fig:md_screen">
<figure id="fig:html_sub1">

<figcaption>Input</figcaption>
</figure>
<p> </p>
<figure id="fig:html_sub2">

<figcaption>Rendered output</figcaption>
</figure>
<figcaption>A training sample for the markup-based task from
HTMLs</figcaption>
</figure>

## Examples of Model Inference [supp:example]

```
[x_52] [y_113] [x_756] [y_145]: NYC Department of Education School Year Calendar 2023-2024
[x_52] [y_159] [x_826] [y_181]: This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private,
[x_52] [y_180] [x_820] [y_202]: parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact
[x_52] [y_201] [x_639] [y_223]: your child's school for information about their calendar. Please note the following:
[x_65] [y_223] [x_77] [y_245]: $\bullet$
[x_92] [y_223] [x_825] [y_245]: On days when school buildings are closed due to inclement weather or other emergencies, all students
[x_92] [y_244] [x_525] [y_266]: and families should plan on participating in remote learning.
[x_65] [y_265] [x_77] [y_287]: $\bullet$
[x_92] [y_265] [x_846] [y_287]: Individual schools' Parent-Teacher Conference dates might be different from the dates below. Your child's
[x_92] [y_286] [x_491] [y_308]: teacher will work with you to schedule your conference.
[x_65] [y_308] [x_77] [y_330]: $\bullet$
[x_92] [y_307] [x_845] [y_330]: On this schedule, elementary schools are defined as programs that serve kindergarten (K) through grade
[x_92] [y_329] [x_826] [y_351]: 8, including schools with 3-K and Pre-K programs, as well as those that end in grade 5. Middle schools
[x_92] [y_350] [x_810] [y_372]: are defined as programs that serve grades 6-8, and high schools are defined as programs that serve
[x_92] [y_371] [x_186] [y_393]: grades 9-12.
[x_60] [y_414] [x_106] [y_436]: DATE
[x_318] [y_414] [x_399] [y_436]: WEEKDAY
[x_605] [y_414] [x_659] [y_436]: EVENT
[x_60] [y_437] [x_155] [y_459]: September 7
[x_297] [y_437] [x_366] [y_459]: Thursday
[x_432] [y_437] [x_565] [y_459]: First day of school
[x_60] [y_470] [x_164] [y_492]: September 14
[x_297] [y_470] [x_366] [y_492]: Thursday
[x_432] [y_459] [x_804] [y_481]: Evening Parent-Teacher Conferences for elementary
[x_432] [y_480] [x_622] [y_503]: schools and Pre-K Centers
[x_60] [y_514] [x_164] [y_536]: September 21
[x_297] [y_514] [x_366] [y_536]: Thursday
[x_432] [y_504] [x_832] [y_526]: Evening Parent-Teacher Conferences for middle schools
[x_432] [y_525] [x_553] [y_547]: and D75 schools
[x_60] [y_548] [x_164] [y_570]: September 25
[x_297] [y_548] [x_360] [y_570]: Monday
[x_432] [y_548] [x_630] [y_570]: Yom Kippur, schools closed
[x_60] [y_581] [x_164] [y_603]: September 28
[x_297] [y_581] [x_366] [y_603]: Thursday
[x_432] [y_570] [x_818] [y_593]: Evening Parent-Teacher Conferences for high schools,
[x_432] [y_592] [x_601] [y_614]: K-12, and 6-12 schools
[x_60] [y_625] [x_135] [y_647]: October 9
[x_297] [y_625] [x_360] [y_647]: Monday
[x_432] [y_614] [x_786] [y_636]: Italian Heritage/Indigenous Peoples' Day, schools
[x_432] [y_636] [x_482] [y_658]: closed
[x_60] [y_679] [x_152] [y_701]: November 2
[x_297] [y_679] [x_366] [y_701]: Thursday
[x_432] [y_658] [x_829] [y_680]: Afternoon and Evening Parent-Teacher Conferences for
[x_432] [y_679] [x_833] [y_701]: elementary schools; students in these schools dismissed
[x_432] [y_700] [x_556] [y_723]: three hours early
[x_60] [y_727] [x_152] [y_749]: November 7
[x_297] [y_727] [x_360] [y_749]: Tuesday
[x_432] [y_727] [x_745] [y_749]: Election Day, students do not attend school
[x_60] [y_775] [x_152] [y_797]: November 9
[x_297] [y_775] [x_366] [y_797]: Thursday
[x_432] [y_754] [x_829] [y_776]: Afternoon and Evening Parent-Teacher Conferences for
[x_432] [y_775] [x_793] [y_797]: middle schools and D75 schools; students in these
[x_432] [y_796] [x_687] [y_818]: schools dismissed three hours early
[x_60] [y_829] [x_161] [y_851]: November 16
[x_297] [y_829] [x_366] [y_851]: Thursday
[x_432] [y_819] [x_818] [y_841]: Evening Parent-Teacher Conferences for high schools,
[x_432] [y_840] [x_601] [y_862]: K-12, and 6-12 schools
[x_60] [y_884] [x_161] [y_906]: November 17
[x_297] [y_884] [x_344] [y_906]: Friday
[x_432] [y_863] [x_773] [y_885]: Afternoon Parent-Teacher Conferences for high
[x_432] [y_884] [x_791] [y_906]: schools, K-12, and 6-12 schools; students in these
[x_432] [y_905] [x_687] [y_927]: schools dismissed three hours early
[x_60] [y_928] [x_186] [y_950]: November 23-24
[x_297] [y_928] [x_416] [y_950]: Thursday-Friday
[x_432] [y_928] [x_692] [y_950]: Thanksgiving Recess, schools closed
[x_60] [y_960] [x_234] [y_983]: December 25-January 1
[x_297] [y_950] [x_368] [y_972]: Monday-
[x_297] [y_971] [x_360] [y_994]: Monday
[x_432] [y_960] [x_646] [y_983]: Winter Recess, schools closed
[x_60] [y_999] [x_140] [y_1021]: January 15
[x_297] [y_999] [x_360] [y_1021]: Monday
[x_432] [y_999] [x_789] [y_1021]: Rev. Dr. Martin Luther King Jr. Day, schools closed
[x_60] [y_1027] [x_170] [y_1049]: January 23- 26
[x_297] [y_1027] [x_410] [y_1049]: Tuesday-Friday
[x_432] [y_1027] [x_603] [y_1049]: Regents Administration
[x_52] [y_1099] [x_311] [y_1118]: NYCDOE School Year Calendar 2023-24
```

```
# NYC Department of Education School Year Calendar 2023-2024

This is the 2023-24 school year calendar for all 3K-12 NYCDOE public schools. If your child attends a private, parochial, charter school, NYC Early Education Center (NYCEEC) or Family Childcare Program, please contact your child's school for information about their calendar. Please note the following:

-   On days when school buildings are closed due to inclement weather or other emergencies, all students and families should plan on participating in remote learning.

-   Individual schools' Parent-Teacher Conference dates might be different from the dates below. Your child's teacher will work with you to schedule your conference.

-   On this schedule, **elementary schools** are defined as programs that serve kindergarten (K) through grade 8, including schools with 3-K and Pre-K programs, as well as those that end in grade 5. **Middle schools** are defined as programs that serve grades 6-8, and **high schools** are defined as programs that serve grades 9-12.

| DATE | WEEKDAY | EVENT |
| --- | --- | --- |
| September 7 | Thursday | First day of school |
| September 14 | Thursday | Evening Parent-Teacher Conferences for elementary schools and Pre-K Centers |
| September 21 | Thursday | Evening Parent-Teacher Conferences for middle schools and D75 schools |
| September 25 | Monday | Yom Kippur, schools closed |
| September 28 | Thursday | Evening Parent-Teacher Conferences for high schools, K-12, and 6-12 schools |
| October 9 | Monday | Italian Heritage/Indigenous Peoples' Day, schools closed |
| November 2 | Thursday | Afternoon and Evening Parent-Teacher Conferences for elementary schools; students in these schools dismissed three hours early |
| November 7 | Tuesday | Election Day, students do not attend school |
| November 9 | Thursday | Afternoon and Evening Parent-Teacher Conferences for middle schools and D75 schools; students in these schools dismissed three hours early |
| November 16 | Thursday | Evening Parent-Teacher Conferences for high schools, K-12, and 6-12 schools |
| November 17 | Friday | Afternoon Parent-Teacher Conferences for high schools, K-12, and 6-12 schools; students in these schools dismissed three hours early |
| November 23-24 | Thursday-Friday | Thanksgiving Recess, schools closed |
| December 25-January 1 | Monday- Monday | Winter Recess, schools closed |
| January 15 | Monday | Rev. Dr. Martin Luther King Jr. Day, schools closed |
| January 23- 26 | Tuesday-Friday | Regents Administration |
```

[^1]:  Equal contribution. $\dagger$ Corresponding author.

[^2]: <https://info.arxiv.org/help/bulk_data/index.html>

[^3]: <https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/overview-ocr#read-api>

[^4]: <https://github.com/pymupdf/PyMuPDF>

[^5]: <https://github.com/microsoft/playwright-python>

[^6]: <https://lxml.de/>

[^7]: <https://pandoc.org/>

[^8]: <https://wkhtmltopdf.org/>

[^9]: <https://github.com/matthewwithanm/python-markdownify>

[^10]: <https://math.nist.gov/~BMiller/LaTeXML/>

[^11]: <https://cloud.google.com/document-ai>