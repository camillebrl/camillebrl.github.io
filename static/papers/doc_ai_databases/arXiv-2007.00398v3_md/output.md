# Introduction

Research in Document Analysis and Recognition (DAR) is generally focused
on information extraction tasks that aim to convert information in
document images into machine readable form, such as character
recognition [doermann2014handbook](http://arxiv.org/pdf/1509.03456v1), table
extraction [kavasidis2019saliency](http://arxiv.org/pdf/1804.06236v1) or key-value pair
extraction [palm2017cloudscan](http://arxiv.org/pdf/1708.07403v1). Such algorithms tend to
be designed as task specific blocks, blind to the end-purpose the
extracted information will be used for.

Progressing independently in such information extraction processes has
been quite successful, although it is not necessarily true that holistic
document image understanding can be achieved through a simple
constructionist approach, building upon such modules. The scale and
complexity of the task introduce difficulties that require a different
point of view.

In this article we introduce Document Visual Question Answering
(DocVQA), as a high-level task dynamically driving DAR algorithms to
conditionally interpret document images. By doing so, we seek to inspire
a “purpose-driven” point of view in DAR research.

<figure id="fig:firspage_example">
<div class="center">
<table style="width:95%;">
<colgroup>
<col style="width: 95%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/gfbx0227_3.png"
alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> Mention the
ZIP code written? </span></p>
<p><span><strong>A:</strong> 80202</span></p></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p><span><strong>Q:</strong> What date is
seen on the seal at the top of the letter? </span></p>
<p><span><strong>A:</strong> 23 sep 1970</span></p></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> Which company
address is mentioned on the letter? </span></p>
<p><span><strong>A:</strong> Great western sugar Co.</span></p></td>
</tr>
</tbody>
</table>
</div>
<figcaption>Example question-answer pairs from DocVQA. Answering
questions in the new dataset require models not just to read text but
interpret it within the layout/structure of the document.</figcaption>
</figure>

In case of Document VQA, as illustrated in Figure
<a href="#fig:firspage_example" data-reference-type="ref"
data-reference="fig:firspage_example">1</a>, an intelligent reading
system is expected to respond to ad-hoc requests for information,
expressed in natural language questions by human users. To do so,
reading systems should not only extract and interpret the textual
(handwritten, typewritten or printed) content of the document images,
but exploit numerous other visual cues including layout (page structure,
forms, tables), non-textual elements (marks, tick boxes, separators,
diagrams) and style (font, colours, highlighting), to mention just a
few.

Departing from generic VQA [vqa2](https://arxiv.org/pdf/1612.00837) and Scene Text VQA
 [textvqa](http://arxiv.org/pdf/1811.11903v1), [stvqa_iccv](http://arxiv.org/pdf/2304.01603v1) approaches, the document images
warrants a different approach to exploit all the above visual cues,
making use of prior knowledge of the implicit written communication
conventions used, and dealing with the high-density semantic information
conveyed in such images. Answers in case of document VQA cannot be
sourced from a closed dictionary, but they are inherently open ended.

Previous approaches on bringing VQA to the documents domain have either
focused on specific document elements such as data
visualisations [dvqa](http://arxiv.org/pdf/1810.02358v2), [kahou2017figureqa](http://arxiv.org/pdf/2109.02226v1) or on specific
collections such as book covers [mishra2019ocr](None). In
contrast to such approaches, we recast the problem to its generic form,
and put forward a large scale, varied collection of real documents.

Main contributions of this work can be summarized as following:

-   We introduce DocVQA, a large scale dataset of $12,767$ document
    images of varied types and content, over which we have defined
    $50,000$ questions and answers. The questions defined are
    categorised based on their reasoning requirements, allowing us to
    analyze how DocVQA methods fare for different question types.

-   We define and evaluate various baseline methods over the DocVQA
    dataset, ranging from simple heuristic methods and human performance
    analysis that allow us to define upper performance bounds given
    different assumptions, to state of the art Scene Text VQA models and
    NLP models.

# Related Datasets and Tasks

Machine reading comprehension (MRC) and open-domain question answering
(QA) are two problems which are being actively pursued by Natural
Language Processing (NLP) and Information Retrieval (IR) communities. In
MRC the task is to answer a natural language question given a question
and a paragraph (or a single document) as the context. In case of open
domain QA, no specific context is given and answer need to be found from
a large collection (say Wikipedia) or from Web. MRC is often modelled as
an extractive QA problem where answer is defined as a span of the
context on which the question is defined. Examples of datsets for
extractive QA include SQuAD 1.1 [squad](http://arxiv.org/pdf/1606.02270v2),
NewsQA [newsqa](None) and Natural
Questions [naturalquestions](http://arxiv.org/pdf/2105.00811v1). MS
MARCO [ms_marco](http://arxiv.org/pdf/1611.09268v3) is an example of a QA dataset for
abstractive QA where answers need to be generated not extracted.
Recently Transformer based pretraining methods like Bidirectional
Encoder Representations from Transformers (BERT) [bert](None)
and XLNet [xlnet](http://arxiv.org/pdf/1906.08237v2) have helped to build QA models
outperforming Humans on reading comprehension on
SQuAD [squad](http://arxiv.org/pdf/1606.02270v2). In contrast to QA in NLP where context is
given as computer readable strings, contexts in case of DocVQA are
document images.

Visual Question Answering (VQA) aims to provide an accurate natural
language answer given an image and a natural language question. VQA has
attracted an intense research effort over the past few
years [vqa2](https://arxiv.org/pdf/1612.00837), [agrawal2017c](None), [johnson2017clevr](http://arxiv.org/pdf/1612.06890v1). Out of a
large body of work on VQA, scene text VQA branch is the most related to
our work. Scene text VQA refers to VQA systems aiming to deal with cases
where understanding scene text instances is necessary to respond to the
questions posed. The ST-VQA [stvqa_iccv](http://arxiv.org/pdf/2304.01603v1) and
TextVQA [textvqa](http://arxiv.org/pdf/1811.11903v1) datasets were introduced in parallel in
2019 and were quickly followed by more
research [singh2019strings](http://arxiv.org/pdf/1904.08920v2), [gao2020multi](http://arxiv.org/pdf/2003.13962v1), [wang2020general](http://arxiv.org/pdf/2002.10215v2).

The ST-VQA dataset [stvqa_iccv](http://arxiv.org/pdf/2304.01603v1) has $31,000\texttt{+}$
questions over $23,000\texttt{+}$ images collected from different public
data sets. The TextVQA dataset [textvqa](http://arxiv.org/pdf/1811.11903v1) has
$45,000\texttt{+}$ questions over $28,000\texttt{+}$ images sampled from
specific categories of the OpenImages
dataset [OpenImages2](http://arxiv.org/pdf/1809.05929v7) that are expected to contain text.
Another dataset named OCR-VQA [mishra2019ocr](None) comprises
more than 1 million question-answer pairs over 207K+ images of book
covers. The questions in this dataset are domain specific, generated
based on template questions and answers extracted from available
metadata.

<div class="figure*" markdown="1">

<figure id="fig:industry_distr">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/industry_pie.png" />
<figcaption>Industry-wise distribution of the documents.</figcaption>
</figure>

<figure id="fig:doc_year_distr">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/doc_years.png" />
<figcaption>Year wise distribution of the documents.</figcaption>
</figure>

<figure id="fig:doc_type_distr">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/doc_types.png" />
<figcaption>Various types of documents used.</figcaption>
</figure>

</div>

Scene text VQA
methods [m4c](http://arxiv.org/pdf/1911.06258v3), [gao2020multi](http://arxiv.org/pdf/2003.13962v1), [textvqa](http://arxiv.org/pdf/1811.11903v1), [gomez2020multimodal](http://arxiv.org/pdf/2006.00923v2)
typically make use of pointer mechanisms in order to deal with
out-of-vocabulary (OOV) words appearing in the image and provide the
open answer space required. This goes hand in hand with the use of word
embeddings capable of encoding OOV words into a pre-defined semantic
space, such as FastText [bojanowski2017enriching](http://arxiv.org/pdf/2102.02270v2) or
BERT [bert](None). More recent, top-performing methods in this
space include M4C [m4c](http://arxiv.org/pdf/1911.06258v3) and
MM-GNN [gao2020multi](http://arxiv.org/pdf/2003.13962v1) models.

Parallelly there have been works on certain domain specific VQA tasks
which require to read and understand text in the images. The DVQA
dataset presented by Kafle  [kafle2020answering](http://arxiv.org/pdf/1908.01801v2), [dvqa](http://arxiv.org/pdf/1810.02358v2)
comprises synthetically generated images of bar charts and template
questions defined automatically based on the bar chart metadata. The
dataset contains more than three million question-answer pairs over
300,000 images.

FigureQA [kahou2017figureqa](http://arxiv.org/pdf/2109.02226v1) comprises over one million
yes or no questions, grounded on over 100,000 images. Three different
types of charts are used: bar, pie and line charts. Similar to DVQA,
images are synthetically generated and questions are generated from
templates. Another related QA task is Textbook Question Answering
(TQA) [textbookqa](http://arxiv.org/pdf/2010.00562v1) where multiple choice questions are
asked on multimodal context, including text, diagrams and images. Here
textual information is provided in computer readable format.

Compared to these existing datasets either concerning VQA on real word
images, or domain specific VQA for charts or book covers, the proposed
DocVQA comprise document images. The dataset covers a multitude of
different document types that include elements like tables, forms and
figures , as well as a range of different textual, graphical and
structural elements.

# DocVQA

In this section we explain data collection and annotation process and
present statistics and analysis of DocVQA.

## Data Collection

**Document Images:** Images in the dataset are sourced from documents in
UCSF Industry Documents Library[^1]. The documents are organized under
different industries and further under different collections. We
downloaded documents from different collections and hand picked pages
from these documents for use in the dataset. Majority of documents in
the library are binarized and the binarization has taken on a toll on
the image quality. We tried to minimize binarized images in DocVQA since
we did not want poor image quality to be a bottleneck for VQA. We also
prioritized pages with tables, forms, lists and figures over pages which
only have running text.

The final set of images in the dataset are drawn from pages of $6,071$
industry documents. We made use of documents from as early as 1900 to as
recent as 2018.
( <a href="#fig:doc_year_distr" data-reference-type="autoref"
data-reference="fig:doc_year_distr">[fig:doc_year_distr]</a>). Most of
the documents are from the 1960-2000 period and they include
typewritten, printed, handwritten and born-digital text. There are
documents from all 5 major industries for which the library hosts
documents — tobacco, food, drug, fossil fuel and chemical. We use many
documents from food and nutrition related collections, as they have a
good number of non-binarized images. . See
 <a href="#fig:industry_distr" data-reference-type="autoref"
data-reference="fig:industry_distr">[fig:industry_distr]</a> for
industry wise distribution of the $6071$ documents used. The documents
comprise a wide variety of document types as shown
in <a href="#fig:doc_type_distr" data-reference-type="autoref"
data-reference="fig:doc_type_distr">[fig:doc_type_distr]</a>.

**Questions and Answers:** Questions and answers on the selected
document images are collected with the help of remote workers, using a
Web based annotation tool. The annotation process was organized in three
stages. In stage 1, workers were shown a document image and asked to
define at most 10 question-answer pairs on it. We encouraged the workers
to add more than one ground truth answer per question in cases where it
is warranted.

<figure id="fig:question_types">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/question_types_new.png" />
<figcaption>The 9 question types and share of questions in each
type.</figcaption>
</figure>

Workers were instructed to ask questions which can be answered using
text present in the image and to enter the answer verbatim from the
document. This makes VQA on the DocVQA dataset an extractive QA problem
similar to extractive QA tasks in NLP [squad](http://arxiv.org/pdf/1606.02270v2), [newsqa](None) and
VQA in case of ST-VQA [stvqa_iccv](http://arxiv.org/pdf/2304.01603v1).

<div class="figure*" markdown="1">

<figure id="fig:top_questions">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/top_questions_new.png" />
<figcaption>Top 15 most frequent questions.</figcaption>
</figure>

<figure id="fig:top_anwers">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/top_answers_new.png" />
<figcaption>Top 15 most frequent answers.</figcaption>
</figure>

<figure id="fig:top_non_numeric_Answers">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/top_answers_non_numeric.png" />
<figcaption>Top 15 non numeric answers.</figcaption>
</figure>

<figure id="fig:compare_question_lengths">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/question_lenghts_compare.png" />
<figcaption>Questions with a particular length.</figcaption>
</figure>

<figure id="fig:compare_answer_lengths">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/answer_lengths_compare.png" />
<figcaption>Answers with a particular length.</figcaption>
</figure>

<figure id="fig:compare_document_lengths">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/token_lengths_compare.png" />
<figcaption>/papers/doc_ai_databases/arXiv-2007.00398v3_md/Images/contexts with a particular length</figcaption>
</figure>

</div>

The second annotation stage aims to verify the data collected in the
first stage. Here a worker was shown an image and questions defined on
it in the first stage (but not the answers from the first stage), and
was required to enter answers for the questions. In this stage workers
were also required to assign one or more question types to each
question. The different question types in DocVQA are discussed
in <a href="#sec:stats_analysis" data-reference-type="autoref"
data-reference="sec:stats_analysis">[sec:stats_analysis]</a>. During the
second stage, if the worker finds a question inapt owing to language
issues or ambiguity, an option to flag the question was provided. Such
questions are not included in the dataset.

If none of the answers entered in the first stage match exactly with any
of the answers from the second stage, the particular question is sent
for review in a third stage. Here questions and answers are editable and
the reviewer either accepts the question-answer (after editing if
necessary) or ignores it. The third stage review is done by the authors
themselves.

## Statistics and Analysis [sec:stats_analysis]

The DocVQA comprises $50,000$ questions framed on $12,767$ images. The
data is split randomly in an $80-10-10$ ratio to train, validation and
test splits. The train split has $39,463$ questions and $10,194$ images,
the validation split has $5,349$ questions and $1,286$ images and the
test split has $5,188$ questions and $1,287$ images.

As mentioned before, questions are tagged with question type(s) during
the second stage of the annotation process.
 <a href="#fig:question_types" data-reference-type="autoref"
data-reference="fig:question_types">[fig:question_types]</a> shows the 9
question types and percentage of questions under each type. A question
type signifies the type of data where the question is grounded. For
example, ‘table/list’ is assigned if answering the question requires
understanding of a table or a list. If the information is in the form of
a key:value, the ‘form’ type is assigned. ‘Layout’ is assigned for
questions which require spatial/layout information to find the answer.
For example, questions asking for a title or heading, require one to
understand structure of the document. If answer for a question is based
on information in the form of sentences/paragraphs type assigned is
‘running text’. For all questions where answer is based on handwritten
text, ‘handwritten’ type is assigned. Note that a question can have more
than one type associated with it. (Examples from DocVQA for each
question type are given in the supplementary.)

In the following analysis we compare statistics of questions, answers
and OCR tokens with other similar datasets for VQA — VQA
2.0 [vqa2](https://arxiv.org/pdf/1612.00837), TextVQA [textvqa](http://arxiv.org/pdf/1811.11903v1) and
ST-VQA [stvqa_iccv](http://arxiv.org/pdf/2304.01603v1) and SQuAD 1.1 [squad](http://arxiv.org/pdf/1606.02270v2)
dataset for reading comprehension. Statistics for other datasets are
computed based on their publicly available data splits. For statistics
on OCR tokens, for DocVQA we use OCR tokens generated by a commercial
OCR solution. For VQA 2.0, TextVQA and ST-VQA we use OCR tokens made
available by authors of LoRRA [textvqa](http://arxiv.org/pdf/1811.11903v1) and
M4C [m4c](http://arxiv.org/pdf/1911.06258v3) as part of the MMF [mmf](https://github.com/facebookresearch/mmf)
framework.

<figure id="fig:wordcloud">
<p><embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/wordcloud_answer_words.png" /> <embed
src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/wordcloud_ocr_words.png" /></p>
<figcaption>Word clouds of words in answers (left) and words spotted on
the document images in the dataset (right)</figcaption>
</figure>

<a href="#fig:compare_question_lengths" data-reference-type="autoref"
data-reference="fig:compare_question_lengths">[fig:compare_question_lengths]</a>
shows distribution of question lengths for questions in DocVQA compared
to other similar datasets. The average question length is is $8.12$,
which is second highest among the compared datasets. . In DocVQA$35,362$
($70.72\%$) questions are unique.
 <a href="#fig:top_questions" data-reference-type="autoref"
data-reference="fig:top_questions">[fig:top_questions]</a> shows the top
$15$ most frequent questions and their frequencies. There are questions
repeatedly being asked about dates, titles and page numbers. A sunburst
of first 4 words of the questions is shown
in <a href="#fig:sunburst_4grams" data-reference-type="autoref"
data-reference="fig:sunburst_4grams">[fig:sunburst_4grams]</a>.

It can be seen that a large majority of questions start with “what is
the”, asking for date, title, total, amount or name.

Distribution of answer lengths is shown
in <a href="#fig:compare_answer_lengths" data-reference-type="autoref"
data-reference="fig:compare_answer_lengths">[fig:compare_answer_lengths]</a>.
We observe in the figure that both DocVQA and SQuAD 1.1 have a higher
number of longer answers compared to the VQA datasets. The average
answer length is $2.17$.

$63.2\%$ of the answers are unique , which is second only to SQuAD
1.1$(72.5\%)$. The top $15$ answers in the dataset are shown
in <a href="#fig:top_anwers" data-reference-type="autoref"
data-reference="fig:top_anwers">[fig:top_anwers]</a>.

We observe that almost all of the top answers are numeric values, which
is expected since there are a good number of document images of reports
and invoices.
In <a href="#fig:top_non_numeric_Answers" data-reference-type="autoref"
data-reference="fig:top_non_numeric_Answers">[fig:top_non_numeric_Answers]</a>
we show the top $15$ non numeric answers. These include named entities
such as names of people, institutions and places. The word cloud on the
left in <a href="#fig:wordcloud" data-reference-type="autoref"
data-reference="fig:wordcloud">[fig:wordcloud]</a> shows frequent words
in answers. Most common words are names of people and names of calendar
months.

In <a href="#fig:compare_document_lengths" data-reference-type="autoref"
data-reference="fig:compare_document_lengths">[fig:compare_document_lengths]</a>
we show the number of images (or ‘context’s in case of SQuAD 1.1)
containing a particular number of text tokens. Average number of text
tokens in an image or context is the highest in the case of
DocVQA($182.75$). It is considerably higher compared to SQuAD 1.1 where
contexts are usually small paragraphs whose average length is $117.23$.
In case of VQA datasets which comprise real world images average number
of OCR tokens is not more than $13$. Word cloud on the right
in <a href="#fig:wordcloud" data-reference-type="autoref"
data-reference="fig:wordcloud">[fig:wordcloud]</a> shows the most common
words spotted by the OCR on the images in DocVQA. We observe that there
is high overlap between common OCR tokens and words in answers.

<figure id="fig:sunburst_4grams">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/sunburst_all_questions_mincount80_maxdispnum8.png" />
<figcaption>Distribution of questions by their starting 4-grams. Most
questions aim to retrieve common data points in documents such as date,
title, total mount and page number. </figcaption>
</figure>

# Baselines [sec:baselines]

In this section we explain the baselines we use, including heuristics
and trained models.

## Heuristics and Upper Bounds [sec:heuristics]

Heuristics we evaluate are: (i) **Random answer:** measures performance
when we pick a random answer from the answers in the train split. (ii)
**Random OCR token:** performance when a random OCR token from the given
document image is picked as the answer. (iii) **Longest OCR token** is
the case when the longest OCR token in the given document is selected as
the answer. (iv) **Majority answer** measures the performance when the
most frequent answer in the train split is considered as the answer.

We also compute following upper bounds: (i) **Vocab UB:** This upper
bound measures performance upper bound one can get by predicting correct
answers for the questions, provided the correct answer is present in a
vocabulary of answers, comprising all answers which occur more than once
in the train split. (ii) **OCR substring UB:** is the upper bound on
predicting the correct answer provided the answer can be found as a
substring in the sequence of OCR tokens. The sequence is made by
serializing the OCR tokens recognized in the documents as a sequence
separated by space, in top-left to bottom-right order. (iii) **OCR
subsequence UB:** upper bound on predicting the correct answer, provided
the answer is a subsequence of the OCR tokens’ sequence.

## VQA Models [sec: vqa models]

For evaluating performance of existing VQA models on DocVQA we employ
two models which take the text present in the images into consideration
while answering questions – Look, Read, Reason & Answer
(LoRRA) [textvqa](http://arxiv.org/pdf/1811.11903v1) and Multimodal Multi-Copy
Mesh(M4C) [m4c](http://arxiv.org/pdf/1911.06258v3).

**LoRRA:** follows a bottom-up and top-down
attention [topdown_bottomup](https://arxiv.org/pdf/1707.07998) scheme with additional
bottom-up attention over OCR tokens from the images. In LoRRA, tokens in
a question are first embedded using a pre-trained embedding
(GloVe [glove](http://arxiv.org/pdf/1608.02094v1)) and then these tokens are iteratively
encoded using an LSTM [lstm](http://arxiv.org/pdf/2103.15232v1) encoder. The model uses two
types of spatial features to represent the visual information from the
images - (i) grid convolutional features from a
Resnet-152 [resnet](https://arxiv.org/pdf/1512.03385) which is pre-trained on
ImageNet [imagenet](http://arxiv.org/pdf/1903.10412v1) and (ii) features extracted from
bounding box proposals from a Faster R-CNN [faster-r-cnn](http://arxiv.org/pdf/1506.01497v3)
object detection model, pre-trained on Visual
Genome [visual_genome](http://arxiv.org/pdf/1602.07332v1). OCR tokens from the image are
embedded using a pre-trained word embedding
(FastText [fasttext](http://arxiv.org/pdf/2102.02270v2)). An attention mechanism is used to
compute an attention weighed average of the image features as well the
OCR tokens’ embeddings. These averaged features are combined and fed
into an output module. The classification layer of the model, predicts
an answer either from a fixed vocabulary (made from answers in train
set) or copy an answer from a dynamic vocabulary which essentially is
the list of OCR tokens in an image. Here the copy mechanism can copy
only one of the OCR tokens from the image. Consequently it cannot output
an answer which is a combination of two or more OCR tokens.

**M4C**: uses a multimodal transformer and an iterative answer
prediction module. Here tokens in questions are embedded using a BERT
model [bert](None). Images are represented using (i) appearance
features of the objects detected using a Faster-RCNN pretrained on
Visual Genome [visual_genome](http://arxiv.org/pdf/1602.07332v1) and (ii) location
information - bounding box coordinates of the detected objects. Each OCR
token recognized from the image is represented using (i) a pretrained
word embedding (FastText), (ii) appearance feature of the token’s
bounding box from the same Faster R-CNN which is used for appearance
features of objects (iii) PHOC [phoc](http://arxiv.org/pdf/1712.07487v1) representation of
the token and (iv) bounding box coordinates of the token. Then these
feature representations of the three entities (question tokens, objects
and OCR tokens) are projected to a common, learned embedding space. Then
a stack of Transformer [attention_is_all_you_need](http://arxiv.org/pdf/2107.08000v1) layers
are applied over these features in the common embedding space. The
multi-head self attention in transformers enable both inter-entity and
intra-entity attention. Finally, answers are predicted through iterative
decoding in an auto-regressive manner. Here the fixed vocabulary used
for the closed answer space is made up of the most common answer words
in the train split. Note that in this case the fixed vocabulary
comprises of answer words, not answers itself as in the case of LoRRA.
At each step in the decoding, the decoded word is either an OCR token
from the image or a word from the fixed vocabulary of common answer
words.

In our experiments we use original LoRRA and M4C models and few variants
of these models. Document images in DocVQA usually contain higher number
of text tokens compared to images in scene text VQA datasets. Hence we
try out larger dynamic vocabularies (i.e. more OCR tokens are considered
from the images) for both LoRRA and M4C. For both the models we also
evaluate performance when no fixed vocabulary is used.

Since the notion of visual objects in real word images is not directly
applicable in case of document images, we also try out variants of LoRRA
and M4C where features of objects are omitted.

## Reading Comprehension Models [sec:RC_models]

In addition to the VQA models which can read text, we try out extractive
question answering / reading comprehension models from NLP. In
particular, we use BERT [bert](None) question answering models.
BERT is a method of pre-training language representations from
unlabelled text using
transformers [attention_is_all_you_need](http://arxiv.org/pdf/2107.08000v1). These
pretrained models can then be used for downstream tasks with just an
additional output layer. In the case of extractive Question Answering,
this is an output layer to predict start and end indices of the answer
span.

# Experiments [sec:experiments]

In this section we explain evaluation metrics and our experimental
settings and report results of experiments.

## Evaluation Metrics [sec:evaluation]

Two evaluation metrics we use are Average Normalized Levenshtein
Similarity (ANLS) and Accuracy (Acc.). ANLS was originally proposed for
evaluation of VQA on ST-VQA [st-vqa_challenge](None). The
Accuracy metric measures percentage of questions for which the predicted
answer matches exactly with any of the target answers for the question.
Accuracy metric awards a zero score even when the prediction is only a
little different from the target answer. Since no OCR is perfect, we
propose to use ANLS as our primary evaluation metric, so that minor
answer mismatches stemming from OCR errors are not severely penalized.

<div id="tab:human_heuristics" markdown="1">

|                    |       |       |       |       |
|:-------------------|:-----:|:-----:|:-----:|:-----:|
|                    |  val  |       | test  |       |
| Baseline           | ANLS  | Acc.  | ANLS  | Acc.  |
| Human              |  \-   |  \-   | 0.981 | 94.36 |
| Random answer      | 0.003 | 0.00  | 0.003 | 0.00  |
| Rnadom OCR token   | 0.013 | 0.52  | 0.014 | 0.58  |
| Longest OCR token  | 0.002 | 0.05  | 0.003 | 0.07  |
| Majority answer    | 0.017 | 0.90  | 0.017 | 0.89  |
| Vocab UB           |  \-   | 31.31 |  \-   | 33.78 |
| OCR substring UB   |  \-   | 85.64 |  \-   | 87.00 |
| OCR subsequence UB |  \-   | 76.37 |  \-   | 77.00 |

Evaluation of different heuristics and upper bounds. Predicting random
answers or majority answer do not even yield 1% accuracy. Answers are a
substring of the serialized OCR output in more than 85% of the cases.

</div>

<div class="table*" markdown="1">

|                       |                  |              |                     |           |       |           |       |
|:----------------------|:----------------:|:------------:|:-------------------:|:---------:|:-----:|:---------:|:-----:|
|                       |                  |              |                     |    val    |       |   test    |       |
| Method                | Objects’ feature | Fixed vocab. | Dynamic vocab. size |   ANLS    | Acc.  |   ANLS    | Acc.  |
| LoRRA `\cite`{=latex} |                  |              |         50          | **0.110** | 7.22  | **0.112** | 7.63  |
|                       |                  |              |         50          |   0.041   | 2.64  |   0.037   | 2.58  |
|                       |                  |              |         50          |   0.102   | 6.73  |   0.100   | 6.43  |
|                       |                  |              |         150         |   0.101   | 7.09  |   0.102   | 7.22  |
|                       |                  |              |         500         |   0.094   | 6.41  |   0.095   | 6.31  |
| M4C `\cite`{=latex}   |                  |              |         50          |   0.292   | 18.34 |   0.306   | 18.75 |
|                       |                  |              |         50          |   0.216   | 12.44 |   0.219   | 12.15 |
|                       |                  |              |         50          |   0.294   | 18.75 |   0.310   | 18.92 |
|                       |                  |              |         150         |   0.352   | 22.66 |   0.360   | 22.35 |
|                       |                  |              |         300         |   0.367   | 23.99 |   0.375   | 23.90 |
|                       |                  |              |         500         | **0.385** | 24.73 | **0.391** | 24.81 |

</div>

## Experimental setup [sec: experimental setup]

For measuring human performance , we collect answers for all questions
in test split, with help a few volunteers from our institution.

In all our experiments including heuristics and trained baselines, OCR
tokens we use are extracted using a commercial OCR application. For the
heuristics and upper bounds we use a vocabulary $4,341$ answers which
occur more than once in the train split.

For LoRRA and M4C models we use official implementations available as
part of the MMF framework [mmf](https://github.com/facebookresearch/mmf). The training settings
and hyper parameters are same as the ones reported in the original
works. The fixed vocabulary we use for LoRRA is same as the vocabulary
we use for computing vocabulary based heuristics and upper bounds. For
M4C the fixed vocabulary we use is a vocabulary of the $5,000$ most
frequent words from the answers in the train split.

For QA using BERT, three pre-trained BERT models[^2] from the
Transformers library [huggingface](http://arxiv.org/pdf/1910.03771v5) are used. The models
we use are bert-base-uncased, bert-large-uncased-whole-word-masking and
bert-large-uncased-whole-word-masking-finetuned-squad. We abbreviate the
model names as bert-base, bert-large and bert-large-squad respectively.
Among these, bert-large-squad is a pre-trained model which is also
finetuned on SQuAD 1.1 for question answering. In case of extractive
question answering or reading comprehension datasets ‘contexts’ on which
questions are asked are passages of electronic text. But in
DocVQA‘contexts’ are document images. Hence to finetune the BERT QA
models on DocVQA we need to prepare the data in SQuAD style format where
the answer to a question is a ‘span’ of the context, defined by start
and end indices of the answer. To this end we first serialize the OCR
tokens recognized on the document images to a single string, separated
by space, in top-left to bottom-right order. To approximate the answer
spans we follow an approach proposed in
TriviaQA [triviaqa](None), which is to find the first match of
the answer string in the serialized OCR string.

The bert-base model is finetuned on DocVQA on 2 Nvidia GeForce 1080 Ti
GPUs, for 2 epochs, with a batch size of 32. We use Adam
optimizer [adam](None) with a learning rate of $5e-05$. The
bert-large and bert-large-squad models are finetuned on 4 GPUs for 6
epochs with a batch size of 8, and a learning rate of $2e-05$.

<div id="tab:bert_results" markdown="1">

|                  |                 |           |       |           |       |
|:-----------------|:----------------|:---------:|:-----:|:---------:|:-----:|
|                  |                 |    val    |       |   test    |       |
| Pretrained model | DocVQA finetune |   ANLS    | Acc.  |   ANLS    | Acc.  |
| bert-base        |                 |   0.556   | 45.6  |   0.574   | 47.6  |
| bert-large-      |                 |   0.594   | 49.28 |   0.610   | 51.08 |
| bert-large-squad |                 |   0.462   | 36.72 |   0.475   | 38.26 |
| bert-large-squad |                 | **0.655** | 54.48 | **0.665** | 55.77 |

Performance of BERT question answering models. A BERT~LARGE~ model which
is fine tuned on both SQuAD 1.1 [squad](http://arxiv.org/pdf/1606.02270v2) and DocVQA
performs the best.

</div>

## Results [sec:results]

Results of all heuristic approaches and upper bounds are reported
in <a href="#tab:human_heuristics" data-reference-type="autoref"
data-reference="tab:human_heuristics">[tab:human_heuristics]</a>. We can
see that none of the heuristics get even a $1\%$ accuracy on the
validation or test splits.

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:96%;">
<colgroup>
<col style="width: 32%" />
<col style="width: 32%" />
<col style="width: 32%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/ggxn0226_79.png"
alt="image" /></td>
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/ypcx0078_1.png"
alt="image" /></td>
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/frgl0228_1.png"
alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
underlined heading just above the table? </span></p>
<p><span><strong>GT:</strong> Indications for implantation</span></p>
<p><span><strong>M4C best:</strong> indications for
implantation</span></p>
<p><span><strong>BERT best:</strong> total aneurism</span></p>
<p><span><strong>Human:</strong> indications for
implantation</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
Extension Number as per the voucher? </span></p>
<p><span><strong>GT:</strong> (910) 741-0673</span></p>
<p><span><strong>M4C best:</strong> 963.12</span></p>
<p><span><strong>BERT best:</strong> (910) 741-0673</span></p>
<p><span><strong>Human:</strong> (910) 741-0673</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> How many
boxed illustrations are there ? </span></p>
<p><span><strong>GT:</strong> 9</span></p>
<p><span><strong>M4C best:</strong> 4</span></p>
<p><span><strong>BERT best:</strong> 4</span></p>
<p><span><strong>Human:</strong> 9</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

*OCR substring UB* yields more than $85\%$ accuracy on both validation
and test splits. It has a downside that the substring match in all cases
need not be an actual answer match. For example if the answer is “2"
which is the most common answer in the dataset, it will match with a “2"
in “2020" or a “2" in “2pac”. This is the reason why we evaluate the
*OCR subsequence UB*. An answer is a sub sequence of the serialized OCR
output for around $76\%$ of the questions in both validation and test
splits.

Results of our trained VQA baselines are shown
in <a href="#tab:vqa_results" data-reference-type="autoref"
data-reference="tab:vqa_results">[tab:vqa_results]</a>. First rows for
both the methods report results of the original model proposed by the
respective authors. In case of LoRRA the original setting proposed by
the authors yields the best results compared to the variants we try out.
With no fixed vocabulary, the performance of the model drops sharply
suggesting that the model primarily relies on the fixed vocabulary to
output answers. Larger dynamic vocabulary results in a slight
performance drop suggesting that incorporating more OCR tokens from the
document images does little help. Unlike LoRRA, M4C benefits from a
larger dynamic vocabulary. Increasing the size of the dynamic vocabulary
from $50$ to $500$ improves the ANLS by around $50\%$. And in case of
M4C, the setting where features of objects are omitted, performs
slightly better compared to the original setting.

<figure id="fig:performance_question_type">
<embed src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/performance_question_type_new.png" />
<figcaption>Best baselines from VQA space and reading comprehension
space pitted against the human performance for different question types.
We need models which can understand figures and text on photographs
better. We need better handwriting recognizers too!</figcaption>
</figure>

Results of the BERT question answering models are reported
in <a href="#tab:bert_results" data-reference-type="autoref"
data-reference="tab:bert_results">[tab:bert_results]</a>. We observe
that all BERT models perform better than the best VQA baseline using
M4C(last row in <a href="#tab:vqa_results" data-reference-type="ref"
data-reference="tab:vqa_results">[tab:vqa_results]</a>). The best
performing model out of all the baselines analysed is the
bert-large-squad model, finetuned on DocVQA. Answers predicted by this
model match one of the target answers exactly for around $55\%$ of the
questions.

In <a href="#fig:performance_question_type" data-reference-type="autoref"
data-reference="fig:performance_question_type">[fig:performance_question_type]</a>
we show performance by question type. We compare the best models among
VQA models and BERT question answering models against the human
performance on the test split. We observe that the human performance is
uniform while the models’ performance vary for different question types.
In <a href="#fig:qualitative_results" data-reference-type="autoref"
data-reference="fig:qualitative_results">[fig:qualitative_results]</a>
we show a few qualitative results from our experiments.

# Conclusion

We introduce a new data set and an associated VQA task with the aim to
inspire a “purpose-driven” approach in document image analysis and
recognition research. Our baselines and initial results motivate
simultaneous use of visual and textual cues for answering questions
asked on document images. This could drive methods that use the
low-level cues (text, layout, arrangements) and high-level goals
(purpose, relationship, domain knowledge) in solving problems of
practical importance.

**Acknowledgements**

We thank Amazon for supporting the annotation effort, and Dr. R.
Manmatha for many useful discussions and inputs. This work is partly
supported by MeitY, Government of India, the project TIN2017-89779-P, an
Amazon AWS Research Award and the CERCA Programme.

# Screen grabs of Annotation Tool [appendix:screen grabs]

As mentioned in Section 3.1 in the main paper, annotation process
involves three stages. In
 <a href="#fig:ann_stage1" data-reference-type="autoref"
data-reference="fig:ann_stage1">[fig:ann_stage1]</a>,
 <a href="#fig:ann_stage2" data-reference-type="autoref"
data-reference="fig:ann_stage2">[fig:ann_stage2]</a>
and <a href="#fig:ann_stage3" data-reference-type="autoref"
data-reference="fig:ann_stage3">[fig:ann_stage3]</a> we show screen
grabs from stage 1, stage 2 and stage 3 of the annotation process
respectively.

<div class="figure*" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/stage1.jpeg" alt="image" />

</div>

<div class="figure*" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/stage2_new.jpeg" alt="image" />

</div>

<div class="figure*" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/stage3.jpeg" alt="image" />

</div>

# Examples of Question Types [appendix:question_types]

We define 9 question types, based on the kind of reasoning required to
answer a question. Question types are assigned at the second stage of
the annotation. We discuss the question types in Section 3.2. in the
main paper.

Examples for types *form*, *yes/no* and *layout* are shown
in <a href="#fig:question_type_examples yesno and layout"
data-reference-type="autoref"
data-reference="fig:question_type_examples yesno and layout">[fig:question_type_examples
yesno and layout]</a>. Examples for a question based on a handwritten
date in a form (types *form* and *handwritten*) are shown
in <a href="#fig:question_type_examples handwritten date form"
data-reference-type="autoref"
data-reference="fig:question_type_examples handwritten date form">[fig:question_type_examples
handwritten date form]</a>. An example for a question based on
information in the form of sentences or paragraphs ( type *running
text*) is shown
in <a href="#fig:question_type running text" data-reference-type="autoref"
data-reference="fig:question_type running text">[fig:question_type
running text]</a>. Examples for types *photograph* and *table* are shown
in <a href="#fig:question_types photo and table"
data-reference-type="autoref"
data-reference="fig:question_types photo and table">[fig:question_types
photo and table]</a>. An example for a question based on a plot (type
*figure*) is shown in <a href="#fig:question_type_examples figure"
data-reference-type="autoref"
data-reference="fig:question_type_examples figure">[fig:question_type_examples
figure]</a>. In all examples a crop of the original image is shown below
the original image, for better viewing of the image region where the
question is based on.

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:84%;">
<colgroup>
<col style="width: 40%" />
<col style="width: 44%" />
</colgroup>
<tbody>
<tr class="odd">
<td colspan="2" style="text-align: center;"><img
src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/rnly0000_2.png" alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> Is it an
existing item ?</span></p>
<p><span><strong>Question types:</strong> <span
style="color: blue"><em>form</em></span> and <span
style="color: blue"><em>yes/no</em></span></span></p>
<p><span><strong>A:</strong> yes</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
date given at the top left? </span></p>
<p><span><strong>Question types:</strong> <span
style="color: blue"><em>layout</em></span> </span></p>
<p><span><strong>A:</strong> 03/17/98</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:80%;">
<colgroup>
<col style="width: 40%" />
<col style="width: 40%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/rnly0000_2.png"
alt="image" /></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
date written next to RSM approval? </span></p>
<p><span><strong>Question types:</strong> <span
style="color: blue"><em>form</em></span> and <span
style="color: blue"><em>handwritten</em></span></span></p>
<p><span><strong>A:</strong> 3-17-98</span></p></td>
<td style="text-align: left;"></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:80%;">
<colgroup>
<col style="width: 40%" />
<col style="width: 40%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/rnly0000_2.png"
alt="image" /></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> If the
request needs to be warehoused by RJR, what needs to be done
?</span></p>
<p><span><strong>Question types:</strong> <span
style="color: blue"><em>running text</em></span> </span></p>
<p><span><strong>A:</strong> write to RJR</span></p></td>
<td style="text-align: left;"></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:80%;">
<colgroup>
<col style="width: 40%" />
<col style="width: 40%" />
</colgroup>
<tbody>
<tr class="odd">
<td colspan="2" style="text-align: center;"><img
src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/khnk0226_3.jpeg" alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> Whose picture
is given? </span></p>
<p><span><strong>Question types:</strong> <span
style="color: blue"><em>photograph</em></span> and <span
style="color: blue"><em>layout</em></span></span></p>
<p><span><strong>A:</strong> Dr. Dwayne G. Westfall</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
average sucrose % for N level 501+ ?</span></p>
<p><span><strong>Question types:</strong> <span
style="color: blue"><em>table</em></span> </span></p>
<p><span><strong>A:</strong> 15.9</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:80%;">
<colgroup>
<col style="width: 40%" />
<col style="width: 40%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/hslf0227_14.jpeg"
alt="image" /></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
highest value for “Intake, mg/1000kcal" plotted on the ‘X’ axis of the
graph?</span></p>
<p><span><strong>Question types:</strong> <span
style="color: blue"><em>figure</em></span> </span></p>
<p><span><strong>A:</strong> 300</span></p></td>
<td style="text-align: left;"></td>
</tr>
</tbody>
</table>

</div>

</div>

# Additional Qualitative Examples [appendix:Additional Qualitative Examples]

Here we show more qualitative results from our baseline experiments.
These results supplement the Results section (Section 5.3 ) in the main
paper.

Remember that BERT [bert](None) question answering model is
designed to answer questions asked on sentences or paragraphs of text (
reading comprehension).
In <a href="#fig:qual : bert wines" data-reference-type="autoref"
data-reference="fig:qual : bert wines">[fig:qual : bert wines]</a> we
show two examples where the model answers questions outside the ambit of
reading comprehension style question answering.
In <a href="#fig:qual : m4c_wins" data-reference-type="autoref"
data-reference="fig:qual : m4c_wins">[fig:qual : m4c_wins]</a> we show
examples where the M4C [m4c](http://arxiv.org/pdf/1911.06258v3) model outperforms the BERT
model to answer questions based on text seen on pictures or photographs.
Such questions are similar to questions in
TextVQA [textvqa](http://arxiv.org/pdf/1811.11903v1) or ST-VQA [stvqa_iccv](http://arxiv.org/pdf/2304.01603v1)
datasets where M4C model yield state-of-the-art results.
In <a href="#fig:qual : inconsistent" data-reference-type="autoref"
data-reference="fig:qual : inconsistent">[fig:qual : inconsistent]</a>
we show an example where both the models yield inconsistent results when
posed with questions of similar nature, highlighting lack of reasoning
behind answering.
In <a href="#fig: qual: reasoning" data-reference-type="autoref"
data-reference="fig: qual: reasoning">[fig: qual: reasoning]</a> we show
two examples where both the M4C and BERT model fail to answer questions
which require understanding of a figure or a diagram.
In <a href="#fig: qual: ocr error" data-reference-type="autoref"
data-reference="fig: qual: ocr error">[fig: qual: ocr error]</a> we show
how OCR errors have resulted in wrong answers although the models manage
to ground the questions correctly.

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:96%;">
<colgroup>
<col style="width: 48%" />
<col style="width: 48%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/hrfw0227_19.jpeg"
alt="image" /></td>
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/krcy0227_46.jpeg"
alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
total cost for Fat cell size (Mt. SInai) in the -05 year ? </span></p>
<p><span><strong>GT:</strong> $35,864</span></p>
<p><span><strong>M4C best:</strong> 4400</span></p>
<p><span><strong>BERT best:</strong> $35 , 864</span></p>
<p><span><strong>Human:</strong> $35,864</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
first recipe on the page? </span></p>
<p><span><strong>GT:</strong> hawaiian fruit cake</span></p>
<p><span><strong>M4C best:</strong> island desserts (continued from
cake</span></p>
<p><span><strong>BERT best:</strong> hawaiian fruit cake</span></p>
<p><span><strong>Human:</strong> hawaiian fruit cake</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:96%;">
<colgroup>
<col style="width: 48%" />
<col style="width: 48%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/hnnp0227_33.jpeg"
alt="image" /></td>
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/znbx0223_6.png"
alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p><span><strong>Q:</strong> What is
written inside logo in the bottom of the document? </span></p>
<p><span><strong>GT:</strong> let yourself grow!</span></p>
<p><span><strong>M4C best:</strong> yourself grow!</span></p>
<p><span><strong>BERT best:</strong> <span
class="math inline"> &lt; <em>n</em><em>o</em>  <em>p</em><em>r</em><em>e</em><em>d</em><em>i</em><em>c</em><em>t</em><em>i</em><em>o</em><em>n</em>&gt;</span></span></p>
<p><span><strong>Human:</strong> let yourself grow!</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What Tobacco
brand of GPI is shown in the picture? </span></p>
<p><span><strong>GT:</strong> Prince</span></p>
<p><span><strong>M4C best:</strong> prince</span></p>
<p><span><strong>BERT best:</strong> <span
class="math inline"> &lt; <em>n</em><em>o</em>  <em>p</em><em>r</em><em>e</em><em>d</em><em>i</em><em>c</em><em>t</em><em>i</em><em>o</em><em>n</em>&gt;</span></span></p>
<p><span><strong>Human:</strong> prince</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:80%;">
<colgroup>
<col style="width: 40%" />
<col style="width: 40%" />
</colgroup>
<tbody>
<tr class="odd">
<td colspan="2" style="text-align: center;"><img
src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/tnbx0223_15.png" alt="image" /></td>
</tr>
<tr class="even">
<td colspan="2" style="text-align: center;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="even">
<td style="text-align: left;"><p><span><strong>Q:</strong> What was the
committee strength for the first meeting? </span></p>
<p><span><strong>GT:</strong> 6</span></p>
<p><span><strong>M4C best:</strong> 6</span></p>
<p><span><strong>BERT best:</strong> 6</span></p>
<p><span><strong>Human:</strong> 6</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What was the
committee strength for the last meeting? </span></p>
<p><span><strong>GT:</strong> 5</span></p>
<p><span><strong>M4C best:</strong> 6</span></p>
<p><span><strong>BERT best:</strong> 6</span></p>
<p><span><strong>Human:</strong> 5</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:96%;">
<colgroup>
<col style="width: 48%" />
<col style="width: 48%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/lfgw0228_3.png"
alt="image" /></td>
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/skgb0228_39.jpeg"
alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
position above "vice chairman" ? </span></p>
<p><span><strong>GT:</strong> chairman</span></p>
<p><span><strong>M4C best:</strong> legal counsel</span></p>
<p><span><strong>BERT best:</strong> legal counsel</span></p>
<p><span><strong>Human:</strong> chairman</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
highest value shown on the vertical axis? </span></p>
<p><span><strong>GT:</strong> 99.99</span></p>
<p><span><strong>M4C best:</strong> 50</span></p>
<p><span><strong>BERT best:</strong> 32</span></p>
<p><span><strong>Human:</strong> 99.99</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

<div class="figure*" markdown="1">

<div class="center" markdown="1">

<table style="width:96%;">
<colgroup>
<col style="width: 48%" />
<col style="width: 48%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/lxkp0227_5.jpeg"
alt="image" /></td>
<td style="text-align: left;"><img src="/papers/doc_ai_databases/arXiv-2007.00398v3_md/images/xyyv0228_1.jpeg"
alt="image" /></td>
</tr>
<tr class="even">
<td style="text-align: left;"></td>
<td style="text-align: left;"></td>
</tr>
<tr class="odd">
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
name of the passenger? </span></p>
<p><span><strong>GT:</strong> dr. william j. darby</span></p>
<p><span><strong>M4C best:</strong> larry</span></p>
<p><span><strong>BERT best:</strong> larry</span></p>
<p><span><strong>Human:</strong> dr. william j. darry</span></p></td>
<td style="text-align: left;"><p><span><strong>Q:</strong> What is the
date present in the memo ?</span></p>
<p><span><strong>GT:</strong> 1/7/77</span></p>
<p><span><strong>M4C best:</strong> 1 7 77</span></p>
<p><span><strong>BERT best:</strong> 1 / 7</span></p>
<p><span><strong>Human:</strong> 1/7/77</span></p></td>
</tr>
</tbody>
</table>

</div>

</div>

[^1]: <https://www.industrydocuments.ucsf.edu/>

[^2]: <https://huggingface.co/transformers/pretrained_models.html>