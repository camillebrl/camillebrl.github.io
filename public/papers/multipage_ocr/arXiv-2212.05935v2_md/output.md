# Introduction [sec:intro]

Automatically managing document workflows is paramount in various
sectors including Banking, Insurance, Public Administration, and the
running of virtually every business. For example, only in the UK more
than 1 million home insurance claims are processed every year. Document
Image Analysis and Recognition (DIAR) is at the meeting point between
computer vision and NLP. For the past 50 years, DIAR methods have
focused on specific information extraction and conversion tasks.
Recently, the concept of Visual Question Answering was introduced in
DIAR
 [mathew2020document](mathew2020document), [mathew2021docvqa](mathew2021docvqa), [mathew2022infographicvqa](mathew2022infographicvqa).
This resulted in a paradigm shift, giving rise to end-to-end methods
that condition the information extraction pipeline on the
natural-language defined task. DocVQA is a complex task that requires
reasoning over typed or handwritten text, layout, graphical elements
such as diagrams and figures, tabular structures, signatures and the
semantics that these convey.

All existing datasets and methods for DocVQA focus on single page
documents, which is far from real life scenarios. Documents are
typically composed of multiple pages and therefore, in a real document
management workflow all pages of a document need to be processed as a
single set.

In this work we aim at extending single-page DocVQA to the more
realistic multi-page setup. Consequently, we define a new task and
propose a novel dataset, MP-DocVQA, designed for Multi-Page Document
Visual Question Answering. MP-DocVQA is an extension of the
SingleDocVQA [mathew2021docvqa](mathew2021docvqa) dataset where the
questions are posed on documents with between 1 and 20 pages.

Dealing with multiple pages largely increases the amount of input data
to be processed. This is particularly challenging for current
state-of-the-art DocVQA methods
[xu2020layoutlm](xu2020layoutlm), [xu2021layoutlmv2](xu2021layoutlmv2), [huang2022layoutlmv3](huang2022layoutlmv3), [powalski2021going](powalski2021going)
based on the Transformer architecture
[vaswani2017attention](vaswani2017attention) that take as input textual, layout
and visual features obtained from the words recognized by an OCR. As the
complexity of the transformer scales up quadratically with the length of
the input sequence, all these methods fix some limit on the number of
input tokens which, for long multi-page documents, can lead to
truncating a significant part of the input data. We will empirically
show the limitations of current methods in this context.

As an alternative, we propose the Hierarchical Visual T5(Hi-VT5), a
multimodal hierarchical encoder-decoder transformer build on top of
T5 [raffel2020exploring](raffel2020exploring) which is capable to naturally
process multiple pages by extending the input sequence length up to
20480 tokens without increasing the model complexity. In our
architecture, the encoder processes separately each page of the
document, providing a summary of the most relevant information conveyed
by the page conditioned on the question. This information is encoded in
a number of special tokens, inspired in the token of the BERT model
[devlin2018bert](devlin2018bert). Subsequently, the decoder generates the
final answer by taking as input the concatenation of all these summary
tokens for all pages. Furthermore, the model includes an additional head
to predict the index of the page where the answer has been found. This
can be used to locate the context of the answer within long documents,
but also as a measure of explainability, following recent works in the
literature  [wang2020general](wang2020general), [tito2021document](tito2021document). Correct
page identification can be used as a way to distinguish which answers
are the result of reasoning over the input data, and not dictated from
model biases.

To summarize, the key contributions of our work are:

1.  We introduce the novel dataset MP-DocVQA containing questions over
    multi-page documents.

2.  We evaluate state-of-the-art methods on this new dataset and show
    their limitations when facing multi-page documents.

3.  We propose Hi-VT5, a multimodal hierarchical encoder-decoder method
    that can answer questions on multi-page documents and predict the
    page where the answer is found.

4.  We provide extensive experimentation to show the effectiveness of
    each component of our framework and explore the relation between the
    accuracy of the answer and the page identification result.

The dataset, baselines and Hi-VT5 model code and weights are publicly
available through the DocVQA Web portal[^1] and GitHub project[^2].

# Related Work

**Document VQA datasets**:
DocVQA [mathew2020document](mathew2020document), [tito2021icdar](tito2021icdar) has seen
numerous advances and new datasets have been released following the
publication of the SingleDocVQA [mathew2021docvqa](mathew2021docvqa)
dataset. This dataset consists of $50,000$ questions posed over industry
document images, where the answer is always explicitly found in the
text. The questions ask for information in tables, forms and paragraphs
among others, becoming a high-level task that brought to classic DIAR
algorithms an end purpose by conditionally interpreting the document
images. Later on,
InfographicsVQA [mathew2022infographicvqa](mathew2022infographicvqa) proposed
questions on infographic images, with more visually rich elements and
answers that can be either extractive from a set of multiple text spans
in the image, a multiple choice given in the question, or the result of
a discrete operation resulting in a numerical non-extractive answer. In
parallel, VisualMRC [tanaka2021visualmrc](tanaka2021visualmrc) proposed
open-domain questions on webpage screenshots with abstractive answers,
which requires to generate longer answers not explicitly found in the
text. DuReader~Vis~ [qi2022dureadervis](qi2022dureadervis) is a Chinese
dataset for open-domain document visual question answering, where the
questions are queries from the Baidu search engine, and the images are
screenshots of the webpages retrieved by the search engine results.
Although the answers are extractive, $43\%$ of them are non-factual and
much longer on average than the ones in previous DocVQA datasets. In
addition, each image contains on average a bigger number of text
instances. However, due to the big size of the image collection, the
task is posed as a 2-stage retrieval and answering tasks, where the
methods must retrieve the correct page first, and answer the question in
a second step. Similarly, the Document Collection Visual Question
Answering (DocCVQA) [tito2021icdar](tito2021icdar) released a set of
$20$ questions posed over a whole collection of $14,362$ single page
document images. However, due to the limited number of questions and the
low document variability, it is not possible to do training on this
dataset and current approaches need to rely on training on SingleDocVQA.
Finally, TAT-DQA [zhu2022towards](zhu2022towards) contains extractive and
abstractive questions on modern financial reports. Despite that the
documents might be multi-page, only 306 documents have actually more
than one page, with a maximum of 3 pages. Instead, our proposed
MP-DocVQA dataset is much bigger and diverse with $46,176$ questions
posed over $5,928$ multi-page documents with its corresponding $47,952$
page images, which provides enough data for training and evaluating new
methods on the new multi-page setting.

**Methods**: Since the release of the SingleDocVQA dataset, several
methods have tackled this task from different perspectives. From NLP,
Devlin proposed BertQA [mathew2021docvqa](mathew2021docvqa) which consists
of a BERT [devlin2018bert](devlin2018bert) architecture followed by a
classification head that predicts the start and end indices of the
answer span from the given context. While many models have extended BERT
obtaining better results
[liu2019roberta](liu2019roberta), [lan2019albert](lan2019albert), [garncarek2021lambert](garncarek2021lambert), [sanh2019distilbert](sanh2019distilbert)
by changing key hyperparameters during training or proposing new
pre-training tasks, T5 [raffel2020exploring](raffel2020exploring) has become
the backbone of many state-of-the-art
methods [powalski2021going](powalski2021going), [biten2022latr](biten2022latr), [lu2022unified](lu2022unified)
on different NLP and multimodal tasks. T5 relies on the original
Transformer [vaswani2017attention](vaswani2017attention) by performing minimal
modifications on the architecture, but pre-training on the novel
de-noising task on a vast amount of data.

On the other hand, and specifically designed for document tasks,
LayoutLM [xu2020layoutlm](xu2020layoutlm) extended BERT by decoupling the
position embedding into 2 dimensions using the token bounding box from
the OCR and fusing visual and textual features during the downstream
task. Alternatively, LayoutLMv2 [xu2021layoutlmv2](xu2021layoutlmv2) and
TILT [powalski2021going](powalski2021going), included visual information
into a multimodal transformer and introduced a learnable bias into the
self-attention scores to explicitly model relative position. In
addition, TILT used a decoder to dynamically generate the answer instead
of extracting it from the context.
LayoutLMv3 [huang2022layoutlmv3](huang2022layoutlmv3) extended its previous
version by using visual patch embeddings instead of leveraging a CNN
backbone and pre-training with 3 different objectives to align text,
layout position and image context. In contrast, while all the previous
methods utilize the text recognized with an off-the-shelf OCR,
Donut [kim2022ocr](kim2022ocr) and
Dessurt [davis2022end](davis2022end) are end-to-end encoder-decoder
methods where the input is the document image along with the question,
and they implicitly learn to read as well as understand the semantics
and layout of the images.

However, the limited input sequence length of these methods make them
unfeasible for tasks involving long documents such as the ones in
MP-DocVQA. Different
methods[dai2019transformer](dai2019transformer), [beltagy2020longformer](beltagy2020longformer), [zaheer2020big](zaheer2020big)
have been proposed in the NLP domain to improve the modeling of long
sequences without increasing the model complexity.
Longformer [beltagy2020longformer](beltagy2020longformer) replaces the common
self-attention used in transformers where each input attends to every
other input by a combination of global and local attention. The global
attention is used on the question tokens, which attend and are attended
by all the rest of the question and context tokens, while a sliding
window guides the local attention over the context tokens to attend the
other locally close context tokens. While the standard self-attention
has a complexity of $O(n^2)$, the new combination of global and local
attention turns the complexity of the model into $O(n)$. Following this
approach, Big Bird [zaheer2020big](zaheer2020big) also includes
attention on randomly selected tokens that will attend and be attended
by all the rest of the tokens in the sequence, which provides a better
global representation while adding a marginal increase of the complexity
in the attention pattern.

# MP-DocVQA Dataset

The Multi-Page DocVQA (MP-DocVQA) dataset comprises 46K questions posed
over 48K images of scanned pages that belong to 6K industry documents.
The page images contain a rich amount of different layouts including
forms, tables, lists, diagrams and pictures among others as well as text
in handwritten, typewritten and printed fonts.

## Dataset creation [subsec:dataset_creation]

Documents naturally follow a hierarchical structure where content is
structured into blocks (sections, paragraphs, diagrams, tables) that
convey different pieces of information. The information necessary to
respond to a question more often than not lies in one relevant block,
and is not spread over the whole document. This intuition was confirmed
during our annotation process in this multi-page setting. The
information required to answer the questions defined by the annotators
was located in a specific place in the document. On the contrary, when
we forced the annotators to use different pages as a source to answer
the question, those become very unnatural and did not capture the
essence of questions that we can find in the real world.

Consequently, we decided to use the
SingleDocVQA [mathew2021docvqa](mathew2021docvqa) dataset, which already
has very realistic questions defined on single pages. To create the new
MP-DocVQA dataset, we took every image-question pair from
SingleDocVQA [mathew2021docvqa](mathew2021docvqa) and added to every image
the previous and posterior pages of the document downloaded from the
original source UCSF-IDL[^3]. As we show in
<a href="#fig:doc_pages" data-reference-type="ref+label"
data-reference="fig:doc_pages">[fig:doc_pages]</a> most of documents in
the dataset have between $1$ and $20$ pages, followed by a long tail of
documents with up to $793$ pages. We focused on the most common scenario
and limited the number of pages in the dataset to $20$. For longer
documents, we randomly selected a set of $20$ pages that included the
page where the answer is found

Next, we had to analyze and filter the questions since we observed that
some of the questions in the SingleDocVQA dataset became ambiguous when
posed in a multi-page setup(e.g. asking for the page number of the
document). Consequently, we performed an analysis detailed in
<a href="#appendix:construction_details" data-reference-type="ref+label"
data-reference="appendix:construction_details">[appendix:construction_details]</a>
to identify a set of key-words, such as *‘document’*, that when included
in the text of the question, can lead to ambiguous answers in a
multi-page setting, as they originally referred to a specific page and
not to the whole multi-page document.

After removing ambiguous questions, the final dataset comprises $46,176$
questions posed over $47,952$ page images from $5,928$ documents. Notice
that the dataset also includes documents with a single page when this is
the case. Nevertheless, as we show in
<a href="#fig:questions_page_ranges" data-reference-type="ref+label"
data-reference="fig:questions_page_ranges">[fig:questions_page_ranges]</a>,
the questions posed over multi-page documents represent the $85.95\%$ of
the questions in the dataset.

Finally, we split the dataset into train, validation and test sets
keeping the same distribution as in SingleDocVQA. However, following
this distribution some pages would appear in more than one split as they
originate from the same document. To prevent this, we trim the number of
pages used as context for such specific cases to ensure that no
documents are repeated between training and validation/test splits. In
<a href="#fig:questions_page_ranges" data-reference-type="ref+label"
data-reference="fig:questions_page_ranges">[fig:questions_page_ranges]</a>
we show the number of questions according to the final document length.

To facilitate research and fair comparison between different methods on
this dataset, along with the images and questions we also provide the
OCR annotations extracted with Amazon Textract[^4] for all the $47,952$
document images (including page images beyond the $20$ page limit to not
limit future research on longer documents).

## Dataset statistics

As we show in
<a href="#tab:datasets_stats" data-reference-type="ref+label"
data-reference="tab:datasets_stats">[tab:datasets_stats]</a>, given that
MP-DocVQA is an extension of SingleDocVQA, the average question and
answer lengths are very similar to this dataset in contrast to the long
answers that can be found in the open-domain datasets VisualMRC and
DuReader~Vis~. On the contrary, the main difference lies in the number
of OCR tokens per document, which is even superior to the Chinese
DuReader~Vis~. In addition, MP-DocVQA adopts the multi-page concept,
which means that not all documents have the same number of pages
(<a href="#fig:questions_page_ranges" data-reference-type="ref+label"
data-reference="fig:questions_page_ranges">[fig:questions_page_ranges]</a>),
but also that each page of the document may contain a different content
distribution, with varied text density, different layout and visual
elements that raise unique challenges. Moreover, as we show in Figs.
<a href="#fig:questions_page_ranges" data-reference-type="ref"
data-reference="fig:questions_page_ranges">[fig:questions_page_ranges]</a>
and <a href="#fig:words_per_question" data-reference-type="ref"
data-reference="fig:words_per_question">[fig:words_per_question]</a> the
variability between documents is high, with documents comprising between
$1$ and $20$ pages, and between $1$ and $42,313$ recognized OCR words.

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/Hi-VT5.png" style="width:90.0%" />

</div>

# Hi-VT5 [sec:method]

Although documents contain dense information, not all of them is
necessary to answer a given question. Following this idea, we propose
the Hierarchical Visual T5(Hi-VT5), a hierarchical encoder-decoder
multimodal transformer where given a question, the encoder extracts the
most relevant information from each page conditioned to the question and
then, the decoder generates the answer from the summarized relevant
information extracted from the encoder. Figure
<a href="#fig:Hi-LT5" data-reference-type="ref"
data-reference="fig:Hi-LT5">[fig:Hi-LT5]</a> shows an overview of the
model. We can see that each page is independently processed by the
encoder taking as input the sequence of OCR tokens (encoding both text
semantics and layout features), a set of patch-based visual features and
the encoded question tokens. In addition, a number of learnable tokens
are introduced to embed at the output of the encoder the summary of
every page. These tokens are concatenated and passed through the decoder
to get the final answer. Moreover, in parallel to the answer generation,
the answer page identification module predicts the page index where the
information to answer the question is found, which can be used as a kind
of explainability measure. We utilize the T5 architecture as the
backbone for our method since the enormous amount of data and their
novel de-noising task utilized during pretraining makes it an excellent
candidate for the model initialization. In this section, we first
describe each module, then how they are integrated and finally, the
training process followed.

**Textual representation:** Following recent literature on document
understanding [huang2022layoutlmv3](huang2022layoutlmv3), [powalski2021going](powalski2021going)
which demonstrates the importance of layout information when working
with Transformers, we utilize a spatial embedding to better align the
layout information with the semantic representation. Formally, given an
OCR token $O_{i}$, we define the associated word bounding box as
$(x^{i}_{0}, y^{i}_{0}, x^{i}_{1}, y^{i}_{1})$.
Following [biten2022latr](biten2022latr), to embed bounding box
information, we use a lookup table for continuous encoding of one-hot
vectors, and sum up all the spatial and semantic representations
together: $$\small
    \mathcal{E}_{i} = E_{O} (O_{i}) + E_{x}(x^{i}_{0}) + E_{y}(y^{i}_{0})+E_{x}(x^{i}_{1}) + E_{y}(y^{i}_{1})
% \vspace{-5pt}
\label{eq:ocr_emb}$$

**Visual representation:** We leverage the Document Image Transformer
(DIT) [li2022dit](li2022dit) pretrained on Document Intelligence
tasks to represent the page image as a set of patch embeddings.
Formally, given an image I with dimension $H \times W \times C$, is
reshaped into $N$ 2D patches of size $P^{2} \times C$, where $(H, W)$ is
the height and width, $C$ is the number of channels, $(P, P)$ is the
resolution of each image patch, and $N = HW/P^{2}$ is the final number
of patches. We map the flattened patches to $D$ dimensional space, feed
them to DiT, pass the output sequence to a trainable linear projection
layer and then feed it to the transformer encoder. We denote the final
visual output as $V=\{v_{0}, \ldots, v_{N}\}$.

**Hi-VT5 hierarchical paradigm:** Inspired by the
BERT [devlin2018bert](devlin2018bert) token, which is used to represent
the encoded sentence, we use a set of $M$ learnable tokens to represent
the page information required to answer the given question. Hence, we
input the information from the different modalities along with the
question and the learnable tokens to the encoder to represent in the
tokens the most relevant information of the page conditioned by the
question. More formally, for each page
$p_{j} \in P=\{p_{0}, \ldots, p_{K}\}$, let
$V_{j}=\{v_{0}, \ldots, v_{N}\}$ be the patch visual features,
$Q=\{q_{0}, \ldots, q_{m}\}$ the tokenized question,
$O_{j}=\{o_{1}, \ldots, o_{n}\}$ the page OCR tokens and
$K_{j}=\{k_{0}, \ldots, k_{M}\}$ the learnable tokens. Then, we embed
the OCR tokens and question using
<a href="#eq:ocr_emb" data-reference-type="ref+label"
data-reference="eq:ocr_emb">[eq:ocr_emb]</a> to obtain the OCR
$\mathcal{E}_{j}^{o}$ and question $\mathcal{E}^{q}$ encoded features.
And concatenate all the inputs
$[K_{j};V_{j};\mathcal{E}^{q};\mathcal{E}_{j}^{o}]$ to feed to the
transformer encoder. Finally, all the contextualized $K^{'}$ output
tokens of all pages are concatenated to create a holistic representation
of the document $D=[K_{0}^{'}; \ldots; K_{K}{'}]$, which is sent to the
decoder that will generate the answer, and to the answer page prediction
module.

**Answer page identification module**: Following the trend to look for
interpretability of the answers in VQA [wang2020general](wang2020general),
in parallel to the the answer generation in the decoder, the
contextualized tokens $D$ are fed to a classification layer that outputs
the index of the page where the answer is found.

**Pre-training strategy:** Since T5 was trained without layout
information, inspired by [biten2022latr](biten2022latr) we propose a
hierarchical layout-aware pretraining task to align the layout and
semantic textual representations, while providing the tokens with the
ability to attend to the other tokens. Similar to the standard
de-noising task, the layout-aware de-noising task masks a span of tokens
and forces the model to predict the masked tokens. Unlike the normal
de-noising task, the encoder has access to the rough location of the
masked tokens, which encourages the model to fully utilize the layout
information when performing this task. In addition, the masked tokens
must be generated from the contextualized $K^{'}$ tokens created by the
encoder, which forces the model to embed the tokens with relevant
information regarding the proposed task.

**Training strategy:** Even though Hi-VT5 keeps the same model
complexity as the sum of their independent components (T5~BASE~ (223M) +
DiT~BASE~ (85M)) and despite being capable to accept input sequences of
up to 20480 tokens, the amount of gradients computed at training time
scales linearly with the number of pages since each page is passed
separately through the encoder and the gradients are stored in memory.
Consequently, it is similar to have a batch size $P$ times bigger in the
encoder compared to a single page setting. While this could be tackled
by parallelizing the gradients corresponding to a set of pages into
different GPUs, we offer an alternative strategy using limited
resources. We train the model on shortened versions of the documents
with only two pages: the page where the answer is found and the previous
or posterior page. Even though this drops the overall performance of the
model, as we show in
<a href="#appendix:train_doc_pages" data-reference-type="ref+label"
data-reference="appendix:train_doc_pages">[appendix:train_doc_pages]</a>,
training with only 2 pages is enough to learn the hierarchical
representation of the model achieving results close to the ones using
the whole document, and offers a good trade-off in terms of memory
requirements. However, after the training phase the decoder and the
answer page identification module can’t deal with the full version of
the documents of up to 20 pages. For this reason, we perform a final
fine-tuning phase using the full-length documents and freezing the
encoder weights.

# Experiments [sec:experiments]

To evaluate the performance of the methods, we use the standard
evaluation metrics in DocVQA, accuracy and Average Normalized
Levenshtein Similarity (ANLS) [biten2019scene](biten2019scene). To assess
the page identification we use accuracy.

## Baselines

As Multi-Page DocVQA is a new task, we adapt several state-of-the-art
methods as baselines to analyze their limitations in the multi-page
setup and compare their performance against our proposed method. We
choose BERT [devlin2018bert](devlin2018bert) because it was the first
question-answering method based on transformers, and it shows the
performance of such a simple baseline.
Longformer [beltagy2020longformer](beltagy2020longformer) and Big
Bird [zaheer2020big](zaheer2020big) because they are specially designed
to deal with long sequences, which might be beneficial for the
multi-page setting. In the case of Big Bird it can work following two
different strategies. The former, Internal Transformer Construction
(ITC) only sets the global attention over one single token, while the
Extended Transformer Construction (ETC) sets the global attention over a
set of tokens. Although the latter strategy is the desired setup for
question-answering tasks by setting all the question tokens with global
attention, the current released code only supports the ITC strategy and
hence, we limit our experiments to this attention strategy. We also use
LayoutLMv3 [huang2022layoutlmv3](huang2022layoutlmv3) because it is the
current public state-of-the-art method on the SingleDocVQA task and uses
explicit visual features by representing the document in image patches.
Finally, T5 [raffel2020exploring](raffel2020exploring) because it is the only
generative baseline and the backbone of our proposed method.

However, all these methods are not directly applicable to a multi-page
scenario. Consequently, we define three different setups to allow them
to be evaluated on this task. In the *‘oracle’* setup, only the page
that contains the answer is given as input to the transformer model.
Thus, this setup aims at mimicking the Single page DocVQA task. It shows
the raw answering capabilities of each model regardless of the size of
the input sequences they can accept. So, it should be seen as a
theoretical maximum performance, assuming that the method has correctly
identified the page where the information is found. In the *‘concat’*
setup, the context input to the transformer model is the concatenation
of the contexts of all the pages of the document. This can be considered
the most realistic scenario where the whole document is given as a
single input. It is expected that the large amount of input data becomes
challenging for the baselines. The page corresponding to the predicted
start index is used as the predicted page, except for T5, since being a
generative method it does not predict the start index. Finally, max conf
is the third setup, which is inspired in the strategy that the best
performing methods in the DocCVQA challenge
[tito2021document](tito2021document) use to tackle the big collection of
documents. In this case, each page is processed separately by the model,
providing an answer for every page along with a confidence score in the
form of logits. Then, the answer with the highest confidence is selected
as the final answer with the corresponding page as the predicted answer
page.

For BERT, Longformer, Big Bird and T5 baselines we create the context
following the standard practice of concatenating the OCR words in the
image following the reading (top-left to bottom-right) order. For all
the methods, we use the
Huggingface [wolf2020transformers](wolf2020transformers) implementation and
pre-trained weights from the most similar task available. We describe
the specific initialization weights and training hyperparameters in
<a href="#appendix:hyperparameters" data-reference-type="ref+label"
data-reference="appendix:hyperparameters">[appendix:hyperparameters]</a>.

## Baseline results

As we show in
<a href="#tab:methods_results" data-reference-type="ref+label"
data-reference="tab:methods_results">[tab:methods_results]</a>, the
method with the best answering performance in the oracle setup (i.e.
when the answer page is provided) is T5, followed by LayoutLMv3, Big
Bird, Longformer and BERT. This result is expected since this setup is
equivalent to the single page document setting, where T5 has already
demonstrated its superior results. In contrast, in the *‘max conf.’*
setup, when the logits of the model are used as a confidence score to
rank the answers generated for each page, T5 performs the worst because
the softmax layer used across the vocabulary turns the logits unusable
as a confidence to rank the answers. Finally, in the concat setup, when
the context of all pages are concatenated Longformer outperforms the
rest, showing its capability to deal with long sequences as seen in
<a href="#fig:methods_anls_by_answer_page"
data-reference-type="ref+label"
data-reference="fig:methods_anls_by_answer_page">[fig:methods_anls_by_answer_page]</a>,
which shows that the performance gap increases as long as the answer
page is placed at the end of the document. The second best performing
method in this setting is T5, which might seem surprising due to its
reduced sequence length. However, looking at
<a href="#fig:methods_anls_by_answer_page"
data-reference-type="ref+label"
data-reference="fig:methods_anls_by_answer_page">[fig:methods_anls_by_answer_page]</a>
it is possible to see that is good on questions whose answers can fit
into the input sequence, while it is not capable to answer the rest. In
contrast, Big Bird is capable to answer questions that require long
sequences since its maximum input length is 4096 as Longformer.
Nevertheless, it performs worse due to the ITC strategy Big Bird is
using, which do not set global attention to all question tokens and
consequently, as long as the question and the answer tokens become more
distant, it is more difficult to model the attention between the
required information to answer the question.

## Hi-VT5 results

In our experiments we fixed the number of tokens to $M=10$, through
experimental validation explained in detail in
<a href="#appendix:num_page_tokens" data-reference-type="ref+label"
data-reference="appendix:num_page_tokens">[appendix:num_page_tokens]</a>.
We observed no significant improvements beyond this number. We pretrain
Hi-VT5 on hierarchical aware de-noising task on a subset of 200,000
pages of OCR-IDL [biten2022ocr](biten2022ocr) for one epoch. Then, we
Train on MP-DocVQA for 10 epochs with the 2-page shortened version of
the documents and finally, perform the fine-tuning of the decoder and
answer page identification module with the full length version of the
documents for 1 epoch. During training and fine-tuning all layers of the
DiT visual encoder are frozen except a last fully connected projection
layer.

Hi-VT5 outperforms all the other methods both on answering and page
identification in the concat and *‘max conf.’* setups, which are the
most realistic scenarios. In addition, when looking closer at the ANLS
per answer page position (see <a href="#fig:methods_anls_by_answer_page"
data-reference-type="ref+label"
data-reference="fig:methods_anls_by_answer_page">[fig:methods_anls_by_answer_page]</a>),
the performance gap becomes more significant when the answers are
located at the end of the document, even compared with Longformer, which
is specifically designed for long input sequences. In contrast, Hi-VT5
shows a performance drop in the *‘oracle’* setup compared to the
original T5. This is because it must infer the answer from a compact
summarized representation of the page, while T5 has access to the whole
page representation. This shows that the page representation obtained by
the encoder has still margin for improvement.

Finally, identifying the page where the answer is found at the same time
as answering the question allows to better interpret the method’s
results.
In <a href="#tab:methods_results" data-reference-type="ref+label"
data-reference="tab:methods_results">[tab:methods_results]</a> we can
see that Hi-VT5 obtains a better answer page identification performance
than all the other baseline methods. In addition, in
<a href="#fig:ret_answ_matrix" data-reference-type="ref+label"
data-reference="fig:ret_answ_matrix">1</a> we show that it is capable to
predict the correct page even when it cannot provide the correct answer.
Interestingly, it answers correctly some questions for which the
predicted page is wrong, which means that the answer has been inferred
from a prior learned bias instead of the actual input data. We provide
more details by analyzing the attention of Hi-VT5 in
<a href="#appendix:attention_viz" data-reference-type="ref+label"
data-reference="appendix:attention_viz">[appendix:attention_viz]</a>.

<figure id="fig:ret_answ_matrix">
<embed src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/plots/ret_answer_Hi-LT5_smooth_matrix.png"
style="width:100.0%" />
<figcaption>Matrix showing the Hi-VT5 correct and wrong answered
questions depending on the answer page prediction module
result.</figcaption>
</figure>

# Ablation studies [sec:ablation]

To validate the effectiveness of each feature proposed in Hi-VT5, we
perform an ablation study and show results in
<a href="#tab:ablation_results" data-reference-type="ref+label"
data-reference="tab:ablation_results">[tab:ablation_results]</a>.
Without the answer page prediction module the model performs slightly
worse on the answering task, showing that both tasks are complementary
and the correct page prediction helps to answer the question. The most
significant boost comes from the hierarchical de-noising pre-training
task, since it allows the tokens to learn better how to represent the
content of the document. The last fine-tuning phase where the decoder
and the answer page prediction module are adapted to the 20 pages
maximum length of the MP-DocVQA documents, is specially important for
the answer page prediction module because the classification layer
predicts only page indexes seen during training and hence, without
finetuning it can only predict the first or the second page of the
documents as the answer page. Finally, when removing the visual features
the final scores are slightly worse, which has also been show in other
works in the
literature [huang2022layoutlmv3](huang2022layoutlmv3), [biten2022latr](biten2022latr), [powalski2021going](powalski2021going),
the most relevant information is conveyed within the text and its
position, while explicit visual features are not specially useful for
grayscale documents.

# Conclusions [sec:conclusions]

In this work, we propose the task of Visual Question Answering on
multi-page documents and make public the MP-DocVQA dataset. To show the
challenges the task poses to current DocVQA methods, we convey an
analysis of state-of-the-art methods showing that even the ones designed
to accept long sequences are not capable to answer questions posed on
the final pages of a document. In order to address these limitations, we
propose the new method Hi-VT5 that, without increasing the model
complexity, can accept sequences up to 20,480 tokens and answer the
questions regardless of the page in which the answer is placed. Finally,
we show the effectiveness of each of the components in the method, and
perform an analysis of the results showing how the answer page
prediction module can help to identify answers that might be inferred
from prior learned bias instead of the actual input data.

# Acknowledgements [acknowledgements]

This work has been supported by the UAB PIF scholarship B18P0070, the
Consolidated Research Group 2017-SGR-1783 from the Research and
University Department of the Catalan Government, and the project
PID2020-116298GB-I00, from the Spanish Ministry of Science and
Innovation.

[^1]: [<a href="rrc.cvc.uab.es/?ch=17" class="uri">rrc.cvc.uab.es/?ch=17</a>](https://rrc.cvc.uab.es/?ch=17)

[^2]: [<a href="github.com/rubenpt91/MP-DocVQA-Framework"
    class="uri">github.com/rubenpt91/MP-DocVQA-Framework</a>](https://github.com/rubenpt91/MP-DocVQA-Framework)

[^3]: <https://www.industrydocuments.ucsf.edu/>

[^4]: <https://aws.amazon.com/textract/>

<figure id="fig:task">
<figure>
<p><embed src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/MP-DocVQA_task.png" /> <strong>Q:</strong> What
was the gross profit in the year 2009? <strong>A:</strong> $19,902</p>
</figure>
<figcaption>In the <strong>MP-DocVQA task</strong>, questions are posed
over multi-page documents where methods are required to understand the
text, layout and visual elements of each page in the document to
identify the correct page (blue in the figure) and answer the
question.</figcaption>
</figure>

<div class="table*" markdown="1">

<div class="tabular" markdown="1">

lccccSSS & & & & **Avg. pages** &**Question** & **Answer** & **Document
Avg.**  
& & & & **per question** & **Avg. length** & **Avg. length** & **OCR
Tokens**  
SingleDocVQA [mathew2021docvqa](mathew2021docvqa) & 50K & 6K & 12K & 1.00 &
9.49 & 2.43 & 151.46  
VisualMRC [tanaka2021visualmrc](tanaka2021visualmrc) & 30K & 10K & 10K & 1.00
& 10.55 & 9.55 & 182.75  
InfographicsVQA [mathew2022infographicvqa](mathew2022infographicvqa) & 30K & 5.4K &
5.4K & 1.00 & 11.54 & 1.60 & 217.89  
DuReaderVis [qi2022dureadervis](qi2022dureadervis) & 15K & 158K & 158K &
1.3K & 9.87 & 180.54 & 1968.21  
DocCVQA [tito2021document](tito2021document) & 20 & 14K & 14K & 14K & 14.00
& 12.75 & 509.06  
TAT-DQA [zhu2022towards](zhu2022towards) & 16K & 2.7K & 3K & 1.07 & 12.54
& 3.44 & 550.27  
MP-DocVQA (ours) & 46K & 6K & 48K & 8.27 & 9.90 & 2.20 & 2026.59  

</div>

<span id="tab:datasets_stats" label="tab:datasets_stats"></span>

</div>

<div class="figure*" markdown="1">

<figure id="fig:doc_pages">
<embed src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/plots/doc_pages_51.png" />
<figcaption aria-hidden="true"></figcaption>
</figure>

<figure id="fig:questions_page_ranges">
<embed src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/plots/question_page_ranges_21.png" />
<figcaption aria-hidden="true"></figcaption>
</figure>

<figure id="fig:words_per_question">
<embed src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/plots/words_per_question_5000.png" />
<figcaption aria-hidden="true"></figcaption>
</figure>

</div>

<div class="table*" markdown="1">

|  |  |  |  |  |  |  |  |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|  |  |  | **Max Seq.** |  |  |  | **Ans. Page** |
| **Model** | **Size** | **Parameters** | **Length** | **Setup** | **Accuracy** | **ANLS** | **Accuracy** |
|  |  |  |  | Oracle | 39.77 | 0.5904 | 100.00 |
| BERT [devlin2018bert](devlin2018bert) | Large | 334M | 512 |  | 34.78 | 0.5347 | 71.24 |
|  |  |  |  | Concat | 27.41 | 0.4183 | 51.61 |
|  |  |  |  | Oracle | 52.48 | 0.6177 | 100.00 |
|  [beltagy2020longformer](beltagy2020longformer) | Base | 148M | 4096 |  | 45.87 | 0.5506 | 70.37 |
|  |  |  |  | Concat | 43.91 | 0.5287 | 71.17 |
|  |  |  |  | Oracle | 55.31 | 0.6450 | 100.00 |
|  [zaheer2020big](zaheer2020big) | Base | 131M | 4096 |  | **49.57** | 0.5854 | 72.27 |
|  |  |  |  | Concat | 41.06 | 0.4929 | 67.54 |
|  |  |  |  | Oracle | 58.81 | 0.6729 | 100.00 |
| LayoutLMv3 [huang2022layoutlmv3](huang2022layoutlmv3) | Base | 125M | 512 |  | 42.70 | 0.5513 | 74.02 |
|  |  |  |  | Concat | 38.47 | 0.4538 | 51.94 |
|  |  |  |  | Oracle | **59.00** | **0.6814** | 100.00 |
| T5 [raffel2020exploring](raffel2020exploring) | Base | 223M | 512 |  | 32.68 | 0.4028 | 46.05 |
|  |  |  |  | Concat | 41.80 | 0.5050 | – |
|  | Base | 316M | 20480 | Oracle | 50.01 | 0.6572 | 100.00 |
|  |  |  |  | Multipage | 48.28 | **0.6201** | **79.23** |

<span id="tab:methods_results" label="tab:methods_results"></span>

</div>

<div class="figure*" markdown="1">

<embed
src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/plots/ANLS_per_answer_page_position_grouped_by_methods.png" />

</div>

<div id="tab:ablation_results" markdown="1">

| **Method**  | **Accuracy** | **ANLS** | **Ans. Page Acc.** |
|:------------|:------------:|:--------:|:------------------:|
| Hi-VT5      |    48.28     |  0.6201  |       79.23        |
| –2D-pos     |    46.12     |  0.5891  |       78.21        |
| –Vis. Feat. |    46.82     |  0.5999  |       78.22        |
| –APPM       |    47.78     |  0.6130  |       00.00        |
| –Pretrain   |    42.10     |  0.5864  |       81.47        |
| –Fine-tune  |    42.86     |  0.6263  |       55.74        |

**ablation studies**. We study the effect of removing different
components independently from namely the 2D position embedding (2D-pos),
visual features (Vis. Feat.), the answer page prediction module (APPM),
the pretraining (Pretrain) and the last fine-tuning (Fine-tune) phase of
the decoder and answer page prediction module.

</div>

 

# construction process [appendix:construction_details]

As described in
<a href="#subsec:dataset_creation" data-reference-type="ref+label"
data-reference="subsec:dataset_creation">[subsec:dataset_creation]</a>,
the source data of the dataset is the
SingleDocVQA [mathew2021docvqa](mathew2021docvqa) dataset. The first row of
<a href="#tab:construction_process_stats"
data-reference-type="ref+label"
data-reference="tab:construction_process_stats">[tab:construction_process_stats]</a>
shows the number of documents, pages and questions in this dataset. The
first step to create the dataset was to download and append to the
existing documents their previous and posterior pages, increasing the
number of page images from 12,767 to 64,057, as shown in the second row
of <a href="#tab:construction_process_stats"
data-reference-type="ref+label"
data-reference="tab:construction_process_stats">[tab:construction_process_stats]</a>.

However, not all questions are suited to be asked on multi-page
documents. Therefore, we performed an analysis based on manually
selected key-words that appear in the questions, searching for those
questions whose answer becomes ambiguous when they are posed over a
document. Some of the selected key-words are shown in table
<a href="#tab:key-word_analysis" data-reference-type="ref+label"
data-reference="tab:key-word_analysis">[tab:key-word_analysis]</a>,
along with some examples of potentially ambiguous questions containing
those key-words. The most clear example is with the word ’document’.
When looking at each document page separately, we can observe that many
times they start with a big text on the top that can be considered as
the title, which is actually the answer in the single page DocVQA
scenario when the question asks about the title of the document.
However, this pattern is repeated in every page of the document, making
the question impossible to answer when multiple pages are taken into
account. Moreover, even if there is only one page with a title, the
answer can still be considered wrong, since the title of the document is
always found in the first page like in the example in
<a href="#fig:task" data-reference-type="ref+label"
data-reference="fig:task">[fig:task]</a>. On the other hand, when we
analyzed more closely other potentially ambiguous selected key-words
such as ’image’, ’appears’ or ’graphic’ we found out that the answers
were not always ambiguous and also the amount of questions with those
words was negligible compared to the entire dataset. Thus, we decided to
keep those questions in our dataset. Finally, we found that the key-word
’title’ was mostly ambiguous only when it was written along with the
word ’document’. Hence, we decided to remove only the questions with the
word ’document’ in it, while keeping all the rest. This filtered
version, which is represented in the third row of
<a href="#tab:construction_process_stats"
data-reference-type="ref+label"
data-reference="tab:construction_process_stats">[tab:construction_process_stats]</a>
is the dataset version that was released and used in the experiments.

Nevertheless, it is important to notice that not all the questions in
are posed over multi-page documents. We keep the documents with a single
page because they are also a possible case in a real life scenario.
However, as showed in the fourth row of
<a href="#tab:construction_process_stats"
data-reference-type="ref+label"
data-reference="tab:construction_process_stats">[tab:construction_process_stats]</a>,
the questions posed over multiple pages represent the 85.95% of all the
questions in the dataset.

# Number of tokens [appendix:num_page_tokens]

embeds the most relevant information from each page conditioned by a
question into $M$ tokens. However, we hypothesize that contrary to
BERT [devlin2018bert](devlin2018bert), which represents a sentence with a
single token, will require more than one token to represent a whole
page, since it conveys more information. Consequently, we perform an
experimental study to find the optimum number of tokens to use. We start
by defining the maximum number of tokens $M$ that can be used, which is
limited by the decoder input sequence length $S$, and the number of
pages $P$ that must be processed. Formally,
$$M=int\left(\frac{S}{P}\right) \label{eq:page_tokens_tradeoff}
\vspace{-2mm}$$ We can set $M$ as an hyperparameter to select depending
on the number of pages we need to process, where in the extreme cases we
can represent a single page with 1024 tokens, or a 1024 page document
with a single token for each page.

Constraining to the 20 pages documents scenario of , the maximum
possible number of tokens $M$ would be 51. We performed a set of
experiments with different tokens to find the optimal value. As we show
in <a href="#tab:page_tokens_exp" data-reference-type="ref+label"
data-reference="tab:page_tokens_exp">1</a>, the model is able to answer
correctly some questions even when using only one or two tokens.
However, the performance increases significantly when more tokens are
used. Nevertheless, the model does not benefit from using more than 10
tokens, since it performs similarly either with 10 or 25 tokens.
Moreover, the performance decreases when using more. This can be
explained because the information extracted from each page can be fully
represented by 10 tokens, while using more, not only does not provide
any benefit, but also makes the training process harder.

<div id="tab:page_tokens_exp" markdown="1">

|            |              |          |               |
|:----------:|:------------:|:--------:|:-------------:|
|            | **Accuracy** | **ANLS** | **Ans. Page** |
| **Tokens** |              |          | **Accuracy**  |
|     1      |    36.41     |  0.4876  |     79.87     |
|     2      |    37.94     |  0.5282  |     79.88     |
|     5      |    39.31     |  0.5622  |     80.77     |
|     10     |    42.10     |  0.5864  |     81.47     |
|     25     |    42.16     |  0.5896  |     81.35     |
|     50     |    30.63     |  0.5768  |     59.18     |

Results of with different tokens.

</div>

# Document pages during training [appendix:train_doc_pages]

As described in <a href="#sec:method" data-reference-type="ref+label"
data-reference="sec:method">[sec:method]</a>, it is not feasible to
train with 20 page length documents due to training resource
limitations. However, as we show in
<a href="#tab:train_pages" data-reference-type="ref+label"
data-reference="tab:train_pages">[tab:train_pages]</a>, even though the
model performs significantly worse when trained with a single page, the
returns become diminishing when training with more than 2. Thus, as
explained in <a href="#sec:method" data-reference-type="ref+label"
data-reference="sec:method">[sec:method]</a> we decided to use 2 pages
in the first stage of training.

# Hyperparameters [appendix:hyperparameters]

<div class="figure*" markdown="1">

<embed src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/plots/ret_prec_per_answer_page_position.png" />

</div>

# Page identification accuracy by answer page position

In <a href="#fig:methods_ret_prec_by_answer_page"
data-reference-type="ref+label"
data-reference="fig:methods_ret_prec_by_answer_page">[fig:methods_ret_prec_by_answer_page]</a>
we show the answer page identification accuracy of the different
baselines and the proposed method, as a function of the page number of
the answer. The overall performance follows a similar behavior as the
answer scores. is the baseline that performs the best in the concat
setting, and and the performance gap between this and the rest of the
baselines becomes more significant as the answer page is located in the
final pages of the document. However, outperforms all the baselines by a
big margin.

# attention visualization [appendix:attention_viz]

To further explore the information that embeds into the tokens, we show
the attention scores for some examples in . The attention of
<a href="#subfig:att_global" data-reference-type="ref+label"
data-reference="subfig:att_global">1</a>, corresponds to the first
token, which usually performs a global attention over the whole document
with a slight emphasis on the question tokens, which provides a holistic
representation of the page. Other tokens like in
<a href="#subfig:att_question" data-reference-type="ref+label"
data-reference="subfig:att_question">3</a> focuses its attention over
the other , and question tokens. More importantly, there is always a
token that focuses its attention to the provided answer like in Figs.
<a href="#subfig:att_answer1" data-reference-type="ref"
data-reference="subfig:att_answer1">2</a> and
<a href="#subfig:att_answer2" data-reference-type="ref"
data-reference="subfig:att_answer2">4</a>.

<div class="figure*" markdown="1">

<figure id="subfig:att_global">
<embed
src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/attentions/encoder_page_8_layer_11_head_11_token_0_global.png" />
<figcaption>Global attention over all the text in the page</figcaption>
</figure>

<figure id="subfig:att_answer1">
<embed
src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/attentions/encoder_page_8_layer_11_head_11_token_1_answer.png" />
<figcaption>Attention focused over the OCR tokens corresponding to the
answer (7 June, 1988)</figcaption>
</figure>

  

<figure id="subfig:att_question">
<embed
src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/attentions/encoder_page_1_layer_11_head_11_token_0_question.png" />
<figcaption>Attention focused over the rest of the and question
tokens.</figcaption>
</figure>

<figure id="subfig:att_answer2">
<embed
src="/papers/multipage_ocr/arXiv-2212.05935v2_md/images/attentions/encoder_page_1_layer_11_head_11_token_1_answer.png" />
<figcaption>Attention focused over the OCR tokens corresponding to the
answer ($115.872)</figcaption>
</figure>

</div>