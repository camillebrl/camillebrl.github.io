# Introduction

The majority of scientific knowledge is stored in books or published in
scientific journals, most commonly in the Portable Document Format
(PDF). Next to HTML, PDFs are the second most prominent data format on
the internet, making up  2.4% of common crawl
[sebastian_spiegler_statistics_2013](https://docs.google.com/file/d/1_9698uglerxB9nAglvaHkEgU-iZNm1TvVGuCW7245-WGvZq47teNpb_uL5N9). However, the
information stored in these files is very difficult to extract into any
other formats. This is especially true for highly specialized documents,
such as scientific research papers, where the semantic information of
mathematical expressions is lost.  
Existing Optical Character Recognition (OCR) engines, such as Tesseract
OCR [smith_overview_2007](https://doi.org/10.1109/ICDAR.2007.4376991), excel at detecting and
classifying individual characters and words in an image, but fail to
understand the relationship between them due to their line-by-line
approach. This means that they treat superscripts and subscripts in the
same way as the surrounding text, which is a significant drawback for
mathematical expressions. In mathematical notations like fractions,
exponents, and matrices, relative positions of characters are crucial.  
Converting academic research papers into machine-readable text also
enables accessibility and searchability of science as a whole. The
information of millions of academic papers can not be fully accessed
because they are locked behind an unreadable format. Existing corpora,
such as the S2ORC dataset [lo_s2orc_2020](https://doi.org/10.18653/v1/2020.acl-main.447), capture the
text of 12M[^2] papers using GROBID [lopez_grobid_2023](https://github.com/kermitt2/grobid),
but are missing meaningful representations of the mathematical
equations.

To this end, we introduce Nougat, a transformer based model that can
convert images of document pages to formatted markup text.  
The primary contributions in this paper are

-   Release of a pre-trained model capable of converting a PDF to a
    lightweight markup language. We release the code and the model on
    GitHub[^3]

-   We introduce a pipeline to create dataset for pairing PDFs to source
    code

-   Our method is only dependent on the image of a page, allowing access
    to scanned papers and books

# Related Work

Optical Character Recognition (OCR) is an extensively researched field
in computer vision for a variety applications, such as document
digitalization
[moysset_full-page_2017](http://arxiv.org/abs/1704.08628), [smith_overview_2007](https://doi.org/10.1109/ICDAR.2007.4376991),
handwriting recognition and scene text recognition
[bautista_scene_2022](http://arxiv.org/abs/2207.06966), [li_trocr_2022](https://doi.org/10.48550/arXiv.2109.10282), [diaz_rethinking_2021](http://arxiv.org/abs/2104.07787).  
More concretely, recognizing mathematical expressions is a heavily
researched subtopic. Grammar based methods
[maclean_new_2013](https://doi.org/10.1007/s10032-012-0184-x), [awal_global_2014](http://arxiv.org/pdf/1707.03088v2), [alvaro_recognition_2014](https://doi.org/10.1016/j.patrec.2012.09.023)
for handwritten mathematical expressions were improved upon by different
encoder-decoder models. The fully convolutional model
[yan_convmath_2020](http://arxiv.org/abs/2012.12619) was succeeded by various RNN decoder
models
[deng_image--markup_2016](https://doi.org/10.48550/arXiv.1609.04938), [le_training_2017](https://doi.org/10.1109/ICDAR.2017.175), [singh_teaching_2018](http://arxiv.org/abs/1802.05415), [zhang_multi-scale_2018](https://doi.org/10.48550/arXiv.1801.03530), [wang_translating_2019](https://doi.org/10.48550/arXiv.1908.11415),
both for handwritten and printed formulas. Recently, the decoder
[zhao_handwritten_2021](http://arxiv.org/abs/2105.02412), [mahdavi_icdar_2019](https://doi.org/10.1109/ICDAR.2019.00247) as well as the
encoder [blecher_pix2tex_2023](https://github.com/lukas-blecher/LaTeX-OCR) were replaced with the
Transformer [vaswani_attention_2017](https://doi.org/10.48550/arXiv.1706.03762) architecture.

Visual Document Understanding (VDU) is another related topic of deep
learning research and focuses on extracting relevant information of a
variety of document types. Previous works depend on pre-trained models
that learn to extract information by jointly modeling text and layout
information using the Transformer architecture. The LayoutLM model
family
[xu_layoutlm_2020](https://doi.org/10.1145/3394486.3403172), [xu_layoutlmv2_2022](http://arxiv.org/abs/2012.14740), [huang_layoutlmv3_2022](http://arxiv.org/abs/2204.08387)
uses masked layout prediction task to capture the spatial relationships
between different document elements.

Open source solutions with a related goal as ours include GROBID
[lopez_grobid_2023](https://github.com/kermitt2/grobid), which parses digital-born scientific
documents to XML with a focus on the bibliographic data and `pdf2htmlEX`
[lu_wang_online_2013](https://www.tug.org/TUGboat/tb34-3/tb108wang.pdf), that converts digital-born PDFs to
HTML while preserving the layout and appearance of the document.
However, both solutions can not recover the semantic information of
mathematical expressions.

# Model

Previous VDU methods either rely on OCR text from a third party tool
[xu_layoutlm_2020](https://doi.org/10.1145/3394486.3403172), [xu_layoutlmv2_2022](http://arxiv.org/abs/2012.14740), [appalaraju_docformer_2021](https://doi.org/10.48550/arXiv.2106.11539)
or focus on document types such as receipts, invoices or form-like
documents [majumder_representation_2020](https://doi.org/10.18653/v1/2020.acl-main.580). Recent studies
[kim_ocr-free_2022](https://doi.org/10.48550/arXiv.2111.15664), [davis_end--end_2022](http://arxiv.org/abs/2203.16618) show that an
external OCR engine is not necessarily needed to achieve competitive
results in VDU.

The architecture is a encoder-decoder transformer
[vaswani_attention_2017](https://doi.org/10.48550/arXiv.1706.03762) architecture, that allows for an
end-to-end training procedure. We build on the Donut
[kim_ocr-free_2022](https://doi.org/10.48550/arXiv.2111.15664) architecture. The model does not
require any OCR related inputs or modules. The text is recognized
implicitly by the network. See Fig.
<a href="#fig:model" data-reference-type="ref"
data-reference="fig:model">1</a> for an overview of the approach.

**Encoder** The visual encoder receives a document image
$\mathbf x\in \mathbb R^{3\times H_0\times W_0}$, crops the margins and
resizes the image to fit in a fixed rectangle of size $(H,\,W)$. If the
image is smaller than the rectangle, additional padding is added to
ensure each image has the same dimensionality. We use a Swin Transformer
[liu_swin_2021](https://doi.org/10.48550/arXiv.2103.14030), a hierarchical vision transformer
[dosovitskiy_image_2021](https://doi.org/10.48550/arXiv.2010.11929) that splits the image into
non-overlapping windows of fixed size and applies a series of
self-attention layers to aggregate information across these windows. The
model output a sequence of the embedded patches
$\mathbf z\in \mathbb R^{d\times N}$ where $d$ is the latent dimension
and $N$ is the number of patches.

**Decoder** The encoded image $\mathbf z$ is decoded into a sequence of
tokens using a transformer decoder architecture with cross-attention.
The tokens are generated in an auto-regressive manner, using
self-attention and cross-attention to attend to different parts of the
input sequence and encoder output respectively. Finally, the output is
projected to the size of the vocabulary $v$, yielding the logits
$\boldsymbol\ell \in \mathbb R^v$.  
Following Kim et al. [kim_ocr-free_2022](https://doi.org/10.48550/arXiv.2111.15664), we use the
implementation of the mBART [lewis_bart_2019](https://doi.org/10.48550/arXiv.1910.13461) decoder. We
use the same tokenizer as Taylor et al.
[taylor_galactica_2022](https://doi.org/10.48550/arXiv.2211.09085) because their model is also
specialized in the scientific text domain.

<figure id="fig:model">
<embed src="/papers/visionrich_small_dec/arXiv-2308.13418v1_md/figures/model.png" style="width:95.0%" />
<figcaption> Our simple end-to-end architecture followin Donut <span
class="citation" data-cites="kim_ocr-free_2022"></span>. The Swin
Transformer encoder takes a document image and converts it into latent
embeddings, which are subsequently converted to a sequence of tokens in
a auto-regressive manner</figcaption>
</figure>

## Setup

We render the document images at a resolution of 96 DPI. Due to the
restrictive possible input dimensions of the Swin Transformer we choose
the input size $(H,\,W) = (896,\,672)$. The aspect ratio is in between
the US letter and Din A4 format $\frac{22}{17}<\frac43<\sqrt 2$. The
document images are resized and then padded to achieve the desired input
size. This input size allows us to use the Swin base model architecture
[liu_swin_2021](https://doi.org/10.48550/arXiv.2103.14030). We initialize the model with the
pre-trained weights.  
The Transformer decoder has a maximal sequence length of $S=4096$. This
relatively large sizing is due to the fact that the text of academic
research papers can be dense and the syntax for tables in particular is
token intensive. The BART decoder is a decoder-only transformer with 10
layers. The entire architecture has a total of 350M parameters.  
We also test experiment with a smaller model (250M parameters) with a
slightly smaller sequence length of $S=3584$ and only 4 decoder layers,
where we start from the pre-trained base model.  
During inference the text is generated using greedy decoding.  
**Training** We use an AdamW optimizer
[loshchilov_decoupled_2019](http://arxiv.org/abs/1711.05101) to train for 3 epochs with an
effective batch size of 192. Due to training instabilities, we choose a
learning rate of $\mathrm{lr}_{\rm init}=5\cdot10^{-5}$ which is reduced
by a factor of $0.9996$ every 15 updates until it reaches
$\mathrm{lr}_{\rm end}=7.5\cdot10^{-6}$.

## Data Augmentation

In image recognition tasks, it is often beneficial to use data
augmentation to improve generalization. Since we are only using
digital-born academic research papers, we need to employ a number of
transformations to simulate the imperfections and variability of scanned
documents. These transformations include erosion, dilation, gaussian
noise, gaussian blur, bitmap conversion, image compression, grid
distortion and elastic transform [simard_best_2003](https://doi.org/10.1109/ICDAR.2003.1227801). Each
has a fixed probability of being applied to a given image. The
transformations are implemented in the *Albumentations*
[buslaev_albumentations_2020](https://doi.org/10.3390/info11020125) library. For an overview of
the effect of each transformation, see Fig.
<a href="#fig:augmentations" data-reference-type="ref"
data-reference="fig:augmentations">2</a>.  
During training time, we also add perturbations to the ground truth text
by randomly replacing tokens. We found this to reduce the collapse into
a repeating loop significantly. For more details, see Section
<a href="#seq:repeptition" data-reference-type="ref"
data-reference="seq:repeptition">5.4</a>.

<figure id="fig:augmentations">
<embed src="/papers/visionrich_small_dec/arXiv-2308.13418v1_md/figures/aug.png" style="width:95.0%" />
<figcaption>List of the different image augmentation methods used during
training on an example snippet form a sample document.</figcaption>
</figure>

# Datasets

To the best of our knowledge there is no paired dataset of PDF pages and
corresponding source code out there, so we created our own from the open
access articles on arXiv.[^4] For layout diversity we also include a
subset of the *PubMed Central* [^5] (PMC) open access non-commercial
dataset. During the pretraining, a portion of the *Industry Documents
Library* [^6] (IDL) is included. See Table
<a href="#tab:dataset_composition" data-reference-type="ref"
data-reference="tab:dataset_composition">2</a> for the dataset
composition.

**arXiv** We collected the source code and compiled PDFs from 1,748,201
articles released on arXiv. To ensure consistent formatting, we first
process the source files using *LaTeXML*[^7] and convert them into HTML5
files. This step was important as it standardized and removed ambiguity
from the LaTeX source code, especially in mathematical expressions. The
conversion process included replacing user-defined macros, standardizing
whitespace, adding optional brackets, normalizing tables, and replacing
references and citations with their correct numbers.

We then parse the HTML files and convert them into a lightweight markup
language that supports various elements such as headings, bold and
italic text, algorithms, LaTeX inline and display math and LaTeX tables.
This way, we ensure that the source code is properly formatted and ready
for further processing.  
The process is visualized in Fig.
<a href="#fig:dataformat" data-reference-type="ref"
data-reference="fig:dataformat">3</a>.

<figure id="fig:dataformat">
<embed src="/papers/visionrich_small_dec/arXiv-2308.13418v1_md/figures/dataformat.png" />
<figcaption>Data processing. The source file is converted into HTML
which is then converted to Markdown. a) The LaTeX source provided by the
authors. b) The HTML file computed form the LaTeX source using LaTeXML.
c) The Markdown file parsed from the HTML file. d) The PDF file provided
by the authors</figcaption>
</figure>

**PMC** We also processed articles from PMC, where XML files with
semantic information are available in addition to the PDF file. We parse
these files into the same markup language format as the arXiv articles.
We chose to use far fewer articles from PMC because the XML files are
not always as rich in semantic information. Often times equations and
tables are stored as images and these cases are not trivial to detect,
which leads to our decision to limit the use of PMC articles to the
pre-training phase.

The XML files are parsed into the same markup language as described
above.

**IDL** The IDL is a collection of documents produced by industries that
have an impact on public health and is maintained by the University of
California, San Francisco Library. Biten et al.
[biten_ocr-idl_2022](https://doi.org/10.48550/arXiv.2202.12985) provide high quality OCR text for
PDFs from the IDL dataset. This does not include text formatting and is
only used for pre-training to teach the model basic OCR of scanned
documents.  

## Splitting the pages

We split the markdown files according to the page breaks in the PDF file
and rasterize each page as an image to create the final paired dataset.
During the compilation, the LaTeX compiler determines the page breaks of
the PDF file automatically. Since we are not recompiling the LaTeX
sources for each paper, we must heuristically split the source file into
parts, which correspond to different pages. To achieve that we are using
the embedded text on the PDF page and match it to source text.  
However, figures and tables in the PDF may not correspond to their
position in the source code. To address this issue, we remove these
elements in a pre-processing step using `pdffigures2`
[clark_pdffigures_2016](https://doi.org/10.1145/2910896.2910904). The recognized captions are are
then compared to the captions in the XML file and matched based on their
Levenshtein distance [levenshtein_binary_1965](https://www.semanticscholar.org/paper/Binary-codes-capable-of-correcting-deletions%2C-and-Levenshtein/b2f8876482c97e804bb50a5e2433881ae31d0cdd). Once the
source document has been split into individual pages, the removed
figures and tables are reinserted at the end of each page.  
For a better matching we also replaced unicode characters in the PDF
text with corresponding LaTeX commands using the pylatexenc-library[^8].

**Bag of Words matching** First we extract the text lines from the PDF
using MuPDF[^9] and preprocess them to remove page numbers and potential
headers/footers. We then use a *Bag of Words* model
[harris_distributional_1954](https://doi.org/10.1080/00437956.1954.11659520) with TF-IDF vectorizer and a
linear Support Vector Machine classifier. The model is fitted to the PDF
lines with the page number as label. Next we split the LaTeX source into
paragraphs and predict the page number for each of them.

Ideally, the predictions will form a stair case function but in practice
the signal will be noisy. To find the best boundary points we employ a
similar logic as decision trees and minimize a measure based on the
*Gini* impurity
$$G_{[a,\:\!b]}(i) = (b-a) \cdot \left( 1 - p_{[a,\:\!b]}^2(i)- p_{[a,\:\!b]}^2(i+1)\right),
    \label{eq:gini}$$ where $p_{[a,\:\!b]}(i)$ is the probability of
choosing an element with the predicted page number $i$ in the interval
$[a,\, b]$ that describes which paragraphs (elements) were considered
for the split.  
The best splitting position $t$ in the interval $[a,\, b]$ is then
$${\hat t}_i = \mathop{\mathrm{\arg\,\min}}_t \left(G_{[a,\:\!t]}(i)+G_{[t,\:\!b]}(i) \right).
    \label{eq:splitting_position}$$ The search process starts with all
paragraphs and for each subsequent page break, the lower bound of the
search interval is set to the previous split position. See Fig.
<a href="#fig:staircase" data-reference-type="ref"
data-reference="fig:staircase">4</a> for a visualization of an example
page.

<figure id="fig:staircase">
<embed src="/papers/visionrich_small_dec/arXiv-2308.13418v1_md/figures/staircase.png" />
<figcaption>Example for splitting the paragraphs in the source code into
different pages. The points in blue denote the page index predicted by
the SVM. </figcaption>
</figure>

**Fuzzy matching** After this first coarse document splitting we try to
find the exact position within the paragraph. This is done by comparing
the source text within the neighborhood of the predicted splitting
position to the last sentences of the previous page of the embedded PDF
text, and the first sentences of the next page using the `fuzzysearch`
library[^10]. If the two dividing points are at the same location in the
source text, the page break is considered “accurate” and receives a
score of 1. On the other hand, if the splitting positions differ, the
one with the smallest normalized Levenshtein distance is selected and
given a score of 1 minus the distance. To be included in the dataset, a
PDF page must have an average score of at least 0.9 for both page
breaks. This results in an acceptance rate of about $47\%$ of all pages.

## Ground truth artifacts [seq:artifacts]

Because the dataset was pre-processed by LaTeXML, the markup version of
the source code can contain artifacts and commands from unsupported
packages. The HTML file may contain subsection titles with numbering
even though they are not numbered in the PDF. There may also be
instances where figures or tables are missing from the ground truth due
to processing errors.

In addition, the splitting algorithm of the source code will in some
cases include text from the previous page or cut off words from the end.
This is especially true for “invisible” characters used for formatting,
like italic, bold text or section header.

For PMC papers the inline math is written as Unicode or italic text,
while display math equations or tables are often included in image
format and will therefore be ignored.

Each of these issues reduces the overall data quality. However, the
large number of training samples compensates for these small errors.

# Results & Evaluation

<figure id="fig:example_theoretical">
<figure>

</figure>
<figure>

</figure>
<figcaption>Example of a page with many mathematical equations taken
from <span class="citation" data-cites="sorscher_beyond_2022"></span>.
Left: Image of a page in the document, Right: Model output converted to
LaTeX and rendered to back into a PDF. Examples of scanned documents can
be found in the appendix <a href="#seq:examples"
data-reference-type="ref"
data-reference="seq:examples">9</a>.</figcaption>
</figure>

In this section we discuss the results and performance of the model. For
an example see Fig.
<a href="#fig:example_theoretical" data-reference-type="ref"
data-reference="fig:example_theoretical">5</a> or go to Sec.
<a href="#seq:examples" data-reference-type="ref"
data-reference="seq:examples">9</a>. The model focuses only on the
important content relevant features of the page. The box around the
equations is skipped.

## Metrics

We report the following metrics on our test set.

**Edit distance** The edit distance, or Levenshtein distance
[levenshtein_binary_1965](https://www.semanticscholar.org/paper/Binary-codes-capable-of-correcting-deletions%2C-and-Levenshtein/b2f8876482c97e804bb50a5e2433881ae31d0cdd), measures the number of
character manipulations (insertions, deletions, substitutions) it takes
to get from one string to another. In this work we consider the
normalized edit distance, where we divide by the total number of
characters.

**BLEU** The BLEU [papineni_bleu_2002](https://doi.org/10.3115/1073083.1073135) metric was
originally introduced for measuring the quality of text that has been
machine-translated from one language to another. The metric computes a
score based on the number of matching n-grams between the candidate and
reference sentence.

**METEOR** Another machine-translating metric with a focus on recall
instead of precision, introduced in
[banerjee_meteor_2005](https://aclanthology.org/W05-0909).

**F-measure** We also compute the F1-score and report the precision and
recall.

## Text modalities

In a scientific research article, there are three distinct types of
text: 1) plain text, which comprises the majority of the document, 2)
mathematical expressions, and 3) tables. It is important to separately
examine each of these components during the evaluation process. This is
necessary because in LaTeX, there are multiple ways to express the same
mathematical expression. While some variability has been eliminated
during the LaTeXML pre-processing step, there still is a significant
amount of ambiguity present, like ordering of subscript and superscript,
equivalent commands with different notation (`stackrel`, `atop`,
`substack` or `frac`, `over`), situationally interchangeable commands
(`bm`, `mathbf`, `boldsymbol`, `bf` or `\left(`, `\big(`, etc.),
whitespace commands, additional layers of brackets, and more. As a
consequence, there can be a discrepancy between prediction and ground
truth, even if the rendered formulas appear identical.  
In addition, it is not always possible to determine, where a inline math
environment ends and text begins, when writing numbers and punctuation
(Example: `$\mathrm{H}_{0}$1,` vs. `H$_{0}1,$` $\to$
$\mathrm{H}_{0}$`<!-- -->`{=html}1, vs. H$_{0}1,$). This ambiguity
reduces both math and plain text scores.  
The expected score for mathematical expressions is lower than for plain
text.

## Comparison

We present our results in Table
<a href="#tab:results_detailed" data-reference-type="ref"
data-reference="tab:results_detailed">1</a>. As expected, the
mathematical expressions have the worst agreement with the ground truth.
For the plain text, most discrepancies come from formatting ambiguities
and missing text due to inline math, as described above. The output
format of GROBID is an XML file, which we convert into a compatible
markup language, similar to the PMC or arXiv files. To some extent,
GROBID provides support for formulas in its output, but it identifies
and stores them as the Unicode representations embedded in the PDF. We
replace each Unicode symbol with its corresponding LaTeX command to
increase the similarity. Additionally, GROBID mislabels small inline
expressions as text. For identified formulas, GROBID stores the bounding
box coordinates. We modify the program by sending the snippet to the
external formula recognition software LaTeX-OCR
[blecher_pix2tex_2023](https://github.com/lukas-blecher/LaTeX-OCR). This way we can also get a signal
for math modality. The reported results in this section are quite poor,
primarily due to the amount of missed formulas by GROBID and the
equation prediction accuracy is affected by the quality of the bounding
boxes. The performance of the embedded PDF text alone is better than
GROBID, which is due to formatting differences for the title page or
reference section.  
Both Nougat small and base are able to outperform the other approach and
achieve high scores in all metrics. We note that the performance of the
smaller model is on par with the larger base model.

<div id="tab:results_detailed" markdown="1">

| Method | Modality | Edit distance $\downarrow$ | BLEU $\uparrow$ | METEOR $\uparrow$ | Precision $\uparrow$ | Recall $\uparrow$ | F1 $\uparrow$ |
|:---|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| PDF | All | 0.255 | 65.8 | 82.1 | 77.1 | 81.4 | 79.2 |
| GROBID | All | 0.312 | 55.6 | 71.9 | 74.0 | 72.1 | 73.0 |
| 2-8 | Tables | 0.626 | 25.1 | 64.5 | 61.4 | 80.7 | 69.7 |
| \+ LaTeX OCR | Plain text | 0.363 | 57.4 | 69.2 | 82.1 | 70.5 | 75.9 |
|  | Math | 0.727 | 0.3 | 5.0 | 11.0 | 8.6 | 9.7 |
| Nougat small (250M$^\ast$) | All | 0.073 | 88.9 | 92.8 | **93.6** | 92.2 | 92.9 |
|  | Tables | 0.220 | 68.5 | 78.6 | 75.0 | 79.8 | 77.3 |
|  | Plain text | 0.058 | 91.0 | 94.3 | 96.1 | 95.3 | 95.7 |
|  | Math | 0.117 | 56.0 | 74.7 | 77.1 | 76.8 | 76.9 |
| Nougat base (350M$^\ast$) | All | **0.071** | **89.1** | **93.0** | 93.5 | **92.8** | **93.1** |
|  | Tables | 0.211 | 69.7 | 79.1 | 75.4 | 80.7 | 78.0 |
|  | Plain text | 0.058 | 91.2 | 94.6 | 96.2 | 95.3 | 95.7 |
|  | Math | 0.128 | 56.9 | 75.4 | 76.5 | 76.6 | 76.5 |

Results on arXiv test set. PDF is the text embedded in the PDF file. The
modality “All" refers to the output text without any splitting.
$^\ast$Number of parameters.

</div>

## Repetitions during inference [seq:repeptition]

<figure id="fig:repetition">
<figure id="fig:rephal">
<p><embed src="/papers/visionrich_small_dec/arXiv-2308.13418v1_md/figures/repetition_hal.png" style="width:100.0%" /> <span
id="fig:rephal" label="fig:rephal"></span></p>
</figure>
<figure id="fig:repnohal">
<p><embed src="/papers/visionrich_small_dec/arXiv-2308.13418v1_md/figures/repetition_nohal.png" style="width:100.0%" />
<span id="fig:repnohal" label="fig:repnohal"></span></p>
</figure>
<figcaption>Examples for repetition detection on logits. Top: Sample
with repetition, Bottom: Sample without repetition. Left: Highest logit
score for each token in the sequence <span
class="math inline">ℓ(<em>x</em>)</span>, Center: Sliding window
variance of the logits <span
class="math inline">VarWin<sub><em>B</em></sub>[ℓ](<em>x</em>)</span>,
Right: Variance of variance from the position to the end <span
class="math inline">VarEnd<sub><em>B</em></sub>[ℓ](<em>x</em>)</span></figcaption>
</figure>

We notice that the model degenerates into repeating the same sentence
over and over again. The model can not recover from this state by
itself. In its simplest form, the last sentence or paragraph is repeated
over and over again. We observed this behavior in $1.5\%$ of pages in
the test set, but the frequency increases for out-of-domain documents.
Getting stuck in a repetitive loop is a known problem with
Transformer-based models, when sampled with greedy decoding
[holtzman_curious_2020](http://arxiv.org/abs/1904.09751).  
It can also happen that the model alternates between two sentences but
sometimes changes some words, so a strict repetition detection will not
suffice. Even harder to detect are predictions where the model counts
its own repetitions, which sometimes happens in the references
section.  
In general we notice this kind behavior after a mistake by the model.
The model is not able to recover from the collapse.

**Anti-repetition augmentation** Because of that we introduce a random
perturbation during training. This helps the model to learn how to
handle a wrongly predicted token. For each training example, there is a
fixed probability that a random token will be replaced by any other
randomly chosen token. This process continues until the newly sampled
number is greater than a specified threshold (in this case, 10%). We did
not observe a decrease in performance with this approach, but we did
notice a significant reduction in repetitions. Particularly for
out-of-domain documents, where we saw a 32% decline in failed page
conversions.

**Repetition detection** Since we are generating a maximum of $4096$
tokens the model will stop at some point, however it is very inefficient
and resource intensive to wait for a “end of sentence” token, when none
will come. To detect the repetition during inference time we look at the
largest logit value $\ell_i=\max \boldsymbol{ \ell}_i$ of the ith token.
We found that the logits after a collapse can be separated using the
following heuristic. First calculate the variance of the logits for a
sliding window of size $B=15$
$$\operatorname{VarWin}_B[ \boldsymbol\ell](x)=\frac1B\sum_{i=x}^{x+B}\left(\ell_i-\frac1B\sum_{j=x}^{x+B}\ell_j\right)^2.\nonumber
    \label{eq:varwin}$$ Here $\ell$ is the signal of logits and $x$ the
index. Using this new signal we compute variances again but this time
from the point $x$ to the end of the sequence
$$\operatorname{VarEnd}_B[ \boldsymbol\ell](x)=\frac{1}{S-x}\sum_{i=x}^{S}\left(\operatorname{VarWin}_B[ \boldsymbol\ell](i)-\frac{1}{S-x}\sum_{j=x}^{S}\operatorname{VarWin}_B[ \boldsymbol\ell](i) \right)^2.\nonumber
    \label{eq:varend}$$ If this signal drops below a certain threshold
(we choose 6.75) and stays below for the remainder of the sequence, we
classify the sequence to have repetitions.

During inference time, it is obviously not possible to compute the to
the end of the sequence if our goal is to stop generation at an earlier
point in time. So here we work with a subset of the last 200 tokens and
a half the threshold. After the generation is finished, the procedure as
described above is repeated for the full sequence.

## Limitations & Future work

**Utility** The utility of the model is limited by a number of factors.
First, the problem with repetitions outlined in section
<a href="#seq:repeptition" data-reference-type="ref"
data-reference="seq:repeptition">5.4</a>. The model is trained on
research papers, which means it works particularly well on documents
with a similar structure. However, it can still accurately convert other
types of documents.  
Nearly every dataset sample is in English. Initial tests on a small
sample suggest that the model’s performance with other Latin-based
languages is satisfactory, although any special characters from these
languages will be replaced with the closest equivalent from the Latin
alphabet. Non-Latin script languages result in instant repetitions.  
**Generation Speed** On a machine with a NVIDIA A10G graphics card with
24GB VRAM we can process 6 pages in parallel. The generation speed
depends heavily on the amount of text on any given page. With an average
number of tokens of $\approx 1400$ we get an mean generation time of
19.5s per batch for the base model without any inference optimization.
Compared to classical approaches (GROBID 10.6 PDF/s
[lopez_grobid_2023](https://github.com/kermitt2/grobid)) this is very slow, but it is not
limited to digital-born PDFs and can correctly parse mathematical
expressions.  
**Future work** The model is trained on one page at a time without
knowledge about other pages in the document. This results in
inconsistencies across the document. Most notably in the bibliography
where the model was trained on different styles or section titles where
sometimes numbers are skipped or hallucinated. Though handling each page
separately significantly improves parallelization and scalability, it
may diminish the quality of the merged document text.  
The primary challenge to solve is the tendency for the model to collapse
into a repeating loop, which is left for future work.

# Conclusion

In this work, we present Nougat, an end-to-end trainable encoder-decoder
transformer based model for converting document pages to markup. We
apply recent advances in visual document understanding to a novel OCR
task. Distinct from related approaches, our method does not rely on OCR
or embedded text representations, instead relying solely on the
rasterized document page. Moreover, we have illustrated an automatic and
unsupervised dataset generation process that we used to successfully
train the model for scientific document to markup conversion. Overall,
our approach has shown great potential for not only extracting text from
digital-born PDFs but also for converting scanned papers and textbooks.
We hope this work can be a starting point for future research in related
domains.  
All the code for model evaluation, training and dataset generation can
be accessed at <https://github.com/facebookresearch/nougat>.

# Acknowledgments

Thanks to Ross Taylor, Marcin Kardas, Iliyan Zarov, Kevin Stone, Jian
Xiang Kuan, Andrew Poulton and Hugo Touvron for their valuable
discussions and feedback.  
Thanks to Faisal Azhar for the support throughout the project.

# Dataset

<div id="tab:dataset_composition" markdown="1">

| Name      | Number of Pages |
|:----------|----------------:|
| arXiv     |       7,511,745 |
| PMC       |         536,319 |
| IDL       |         446,777 |
| **Total** |   **8,204,754** |

Dataset composition

</div>

The most important data source is arXiv, making up $>91.5\%$ of the
corpus. On arXiv most research documents are paired with the LaTeX
source code provided by the authors. The LaTeX source offers more
information and is left unprocessed, unlike the XML format from PMC
where equations and tables are frequently substituted with images. This
allows us to select exactly which information we need to build the
dataset.

# Examples [seq:examples]

In this section we converted some pages from old text books using the
Nougat base model. The text books from the *Internet Archive*[^11] and
*Project Gutenberg*[^12] and are in public domain.  
The performance for these scanned pages is noticeable worse than for
digital-born documents. However, the model does generate sensible text
for each page with few errors. For example see the first row of Fig.
<a href="#fig:example_calc" data-reference-type="ref"
data-reference="fig:example_calc">9</a>. Here the model mistakes the
almost illegible exponent $n$ for $\ast$. In the second row of the same
figure the model falls into a repetitive loop after predicting another
comma instead of a dot. Similar problems can be seen in Fig.
<a href="#fig:example_phys" data-reference-type="ref"
data-reference="fig:example_phys">10</a>.  
In Fig. <a href="#fig:example_scanthesis" data-reference-type="ref"
data-reference="fig:example_scanthesis">11</a> we present pages, scanned
with a mobile device, from a printed master thesis and the Nougat
output. The model is robust to the artifacts that arise when
hand-scanning a document.  
Explore the examples in this section on the project page:
<https://facebookresearch.github.io/nougat>.

<figure id="fig:example_calc">
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figcaption>Example of an old calculus text book <span class="citation"
data-cites="march_calculus_1917"></span>. </figcaption>
</figure>

<figure id="fig:example_phys">
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figcaption>A selection of pages from a NASA conference from 1970 <span
class="citation"
data-cites="noauthor_kinetics_1970"></span>.</figcaption>
</figure>

<figure id="fig:example_scanthesis">
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figcaption>Scan of a modern thesis with a mobile device camera, with
permission from the author. </figcaption>
</figure>

<figure id="fig:example_tables">
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figure>

</figure>
<figcaption>Pages with tables. Upper: Fan et al. <span class="citation"
data-cites="fan_hierarchical_2018"></span> page 6, Lower: Shah et al.
<span class="citation" data-cites="shah_cycle-consistency_2019"></span>
page 6</figcaption>
</figure>

[^1]: Correspondence to: <lblecher@meta.com>

[^2]: The paper reports 8.1M papers but the authors recently updated the
    numbers on the GitHub page <https://github.com/allenai/s2orc>

[^3]: <https://github.com/facebookresearch/nougat>

[^4]: <https://arxiv.org/>

[^5]: <https://www.ncbi.nlm.nih.gov/pmc/>

[^6]: <https://www.industrydocuments.ucsf.edu/>

[^7]: <http://dlmf.nist.gov/LaTeXML/>

[^8]: <https://github.com/phfaist/pylatexenc>

[^9]: <https://mupdf.com/>

[^10]: <https://github.com/taleinat/fuzzysearch>

[^11]: <https://archive.org/>

[^12]: <https://www.gutenberg.org/>