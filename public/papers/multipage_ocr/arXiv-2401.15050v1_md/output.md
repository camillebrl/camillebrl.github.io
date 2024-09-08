# Introduction

There has been a noticeable industrial interest surrounding the
automation of data extraction from various documents, including
receipts, reports, and forms to minimize manual efforts and enable
seamless downstream analysis of the extracted data
[zhang2020rapid](https://arxiv.org/pdf/2002.01861), [layoutlm](https://doi.org/10.1145/3394486.3403172). However, the process of
parsing documents poses several challenges, including obscure
information within scanned documents that may result in Optical
Character Recognition (OCR) errors, complex layouts (such as tables),
and intricate content structures.

To investigate and address these challenges, several datasets have been
made available. These datasets encompass a wide range of tasks, such as
classification [rvl-cdip](https://arxiv.org/pdf/2009.14457), semantic entity recognition
[cord](http://arxiv.org/pdf/2103.10213v1), [funsd](http://arxiv.org/pdf/1905.13538v2), relation extraction
[funsd](http://arxiv.org/pdf/1905.13538v2), question answering [docvqa](https://arxiv.org/pdf/2007.00398), and
key information extraction [sroie](https://doi.org/10.1109/icdar.2019.00244). Nonetheless, a
significant limitation shared by these datasets is that they mostly
consist of single-page documents with a limited amount of content. As a
consequence, these datasets fail to capture various challenges inherent
in parsing lengthy documents spanning multiple pages, which are commonly
encountered in the financial industry. Financial reports and documents
can become exceedingly lengthy, necessitating a comprehensive
understanding of the entire context to effectively analyze and extract
pertinent information.

<figure id="models">
<img src="/papers/multipage_ocr/arXiv-2401.15050v1_md/Inline_example.png" style="width:40.0%" />
<figcaption>First page from a 4-page example financial form in the
LongForms dataset. The information in these documents is spread over a
mix of tables and text spanning multiple pages which makes it
challenging for short-context models. </figcaption>
</figure>

The limitations inherent in existing datasets have a direct impact on
the capabilities of the proposed models. In the literature, two primary
lines of work have emerged: *(i)* OCR-dependent architectures
[lilt](https://doi.org/10.18653/v1/2022.acl-long.534), [layoutlm](https://doi.org/10.1145/3394486.3403172), [layoutlmv2](https://doi.org/10.18653/v1/2021.acl-long.201), [layoutlmv3](https://doi.org/10.1145/3503161.3548112), [udop](https://arxiv.org/pdf/2212.02623) *(ii)*
OCR-free models [donut](https://arxiv.org/pdf/2111.15664), [pix2struct](https://arxiv.org/pdf/2210.03347). OCR-dependent models
typically employ transformer-based text encoders and incorporate spatial
information by leveraging the words’ coordinates in the documents as
additional embeddings. One notable exception is UDOP
[udop](https://arxiv.org/pdf/2212.02623) which consists of an encoder-decoder architecture.
Conversely, OCR-free models typically employ a vision encoder to process
the scanned document image and a text decoder to generate the desired
information. Nevertheless, a common limitation shared by most of these
models is their design and pretraining to handle a maximum of 512 tokens
or process a single input image.

In this work, we introduce two main contributions. Firstly, we present
the LongForms dataset, a comprehensive financial dataset primarily
comprising 140 long forms where the task is formulated as named entity
recognition. Due to privacy concerns and proprietary limitations, we
were unable to utilize our internal resources to construct this dataset.
Consequently, we obtained financial statements from the SEC website[^1],
aligning our tasks to encompass the significant challenges encountered
in the financial documents which require a deep understanding of lengthy
contexts. Secondly, we propose LongFin, a multimodal document
understanding model capable of processing up to 4K tokens. Our approach
builds upon LiLT [lilt](https://doi.org/10.18653/v1/2022.acl-long.534), one of the state-of-the-art
multimodal document understanding models. Additionally, we incorporate
techniques that effectively extend the capabilities of text-only models,
such as RoBERTa [roberta](https://arxiv.org/pdf/1907.11692), to handle longer sequences, as
demonstrated by Longformer [longformer](https://arxiv.org/pdf/2004.05150). By leveraging
these techniques, our proposed model exhibits enhanced performance in
processing lengthy financial forms. The efficacy of our approach is
extensively evaluated, showcasing its effectiveness and paving the way
for numerous commercial applications in this domain.

# Related Work [sec:relatedwork]

## Document Datasets

Several recently released datasets in the field of document
understanding have contributed significantly to advancing research in
this area. The RVL-CDIP dataset [rvl-cdip](https://arxiv.org/pdf/2009.14457) introduced a
classification task, encompassing 400K scanned documents categorized
into 16 classes, such as forms and emails. Another notable dataset,
DocVQA [docvqa](https://arxiv.org/pdf/2007.00398), focuses on document question answering
and comprises 50K question-answer pairs aligned with 12K scanned images.
In addition, the CORD dataset [cord](http://arxiv.org/pdf/2103.10213v1) consists of 11K
scanned receipts, challenging models to extract 54 different data
elements (e.g., phone numbers and prices). Furthermore, the FUNSD
dataset [funsd](http://arxiv.org/pdf/1905.13538v2) was proposed, featuring 200 scanned
forms. This dataset primarily revolves around two key tasks: semantic
entity recognition (e.g., header, question, answer) and relation
extraction (question-answer pairs). FUNSD is particularly relevant to
our dataset, LongForms, as it also mainly consist of forms. However,
FUNSD and all the above-mentioned datasets mainly focus on short
contexts, as they typically consist of single-page documents. In
contrast, our LongForms dataset primarily consists of multi-page
documents, presenting unique challenges that demand a comprehensive
understanding of lengthy contexts which is common in the financial
industry.

## Document AI Models

Numerous document understanding models have been developed to tackle the
challenges posed by the aforementioned benchmark datasets. These models
can be broadly categorized into two main groups: OCR-free and
OCR-dependent models. OCR-free models, exemplified by Donut
[donut](https://arxiv.org/pdf/2111.15664) and Pix2Struct [pix2struct](https://arxiv.org/pdf/2210.03347),
typically employ vision transformer-based encoders to process input
images and text decoders to handle output generation. These models are
often pretrained on OCR-related tasks, enabling them to comprehend the
text embedded within scanned documents effectively. On the other hand,
OCR-dependent models, including LayoutLM [layoutlm](https://doi.org/10.1145/3394486.3403172),
LayoutLMv2 [layoutlmv2](https://doi.org/10.18653/v1/2021.acl-long.201), LayoutLMv3
[layoutlmv3](https://doi.org/10.1145/3503161.3548112), LiLT [lilt](https://doi.org/10.18653/v1/2022.acl-long.534), DocFormer
[docformer](https://arxiv.org/pdf/2106.11539) and UDOP [udop](https://arxiv.org/pdf/2212.02623), rely on
external OCR tools to initially extract underlying text from scanned
documents. To incorporate layout information, these models utilize
specialized positional embeddings, encoding the coordinates of each word
in the document. Additionally, some models, such as LayoutLMv2,
LayoutLMv3, DocFormer, and UDOP, employ visual embeddings created by
splitting the image into patches. These visual embeddings, along with
the text and layout embeddings, are fed into the models. While LayoutLM,
LayoutLMv2, LayoutLMv3, DocFormer, and LiLT adopt an encoder-only
architecture, UDOP is based on the T5 model [t5](http://jmlr.org/papers/v21/20-074.html), which
follows an encoder-decoder architecture. Despite the impressive
achievements of these models, they share a common limitation: they are
typically designed to process a single page or a maximum of 512 tokens,
thereby restricting their applicability to multi-page documents.
[longdocument](http://arxiv.org/pdf/2108.09190v2) proposed a multimodal document
understanding model that can process up to 4096 tokens, however their
code is not publicly available and their model performance deteriorates
on the short-context datasets such as FUNSD [funsd](http://arxiv.org/pdf/1905.13538v2). In
contrast, our proposed model, LongFin, works efficiently on both short
and long contexts (to up 4096 tokens), making it particularly
well-suited for a variety of real-world industrial applications.

# LongForms Dataset [sec:longfin]

Due to privacy constraints, we are unable to utilize internal documents
for dataset construction. Instead, we turn to publicly available
financial reports and tailor our dataset, LongForms, to emulate the
challenges encountered in our proprietary datasets. This approach
ensures the task’s alignment with real-world financial contexts without
violating privacy.

## Dataset Collection & Preparation [sec:dataset_collection]

To construct LongForms, we leverage the EDGAR database [^2], a
comprehensive repository of financial filings and reports submitted by
US companies. These filings are based on different financial form types
(e.g., 10-K, 10-Q) which vary in structure and content. Our dataset
primarily centers around the SEC Form 10-Q, which provides a detailed
quarterly report on a company’s finances. This specific form is chosen
due to its similarity in both structure and content to to the documents
we frequently encounter in the financial services industry.

We download 140 10-Q forms that were published between 2018 and 2023.
This deliberate decision to keep the dataset relatively small is
intended to mirror the limited data challenges commonly encountered in
real-world scenarios, particularly in the finance domain, where strict
data confidentiality prevents access to large-scale datasets.
Consequently, it is common practice to construct smaller datasets that
mimic the proprietary datasets [madl2023approximate](https://arxiv.org/pdf/2307.01875).
Furthermore, our dataset size aligns with recently published datasets,
such as the FUNSD dataset [funsd](http://arxiv.org/pdf/1905.13538v2) which primarily
consists of single-page forms. Inspired by the FUNSD dataset, we perform
a random split of the LongForms dataset and divide the dataset into 105
training documents, which account for 75% of the total dataset, and 35
testing documents, representing the remaining 25%.

## Dataset Description & Setup [sec:task_desctiption]

Our dataset, LongForms, is formulated as a Named Entity Recognition
(NER) task. The dataset consists of $N$ examples, denoted as
$D = \{d_i, w_i, b_i, n_i\}_{i=1}^N$, where $d_i$ represents a PDF
document, $w_i$ represents the list of words, $b_i$ represents the list
of bounding boxes, and $n_i$ represents a list of entities present in
the document. To obtain the words ($w_i$) and their bounding boxes
($b_i$), each PDF document is processed using the pdftotext[^3] tool.
Moreover, we define six entity types: *(i)* Total Assets, *(ii)* Cash at
the beginning of the period (Beginning Cash), *(iii)* Cash at the end of
the period (End Cash), *(iv)* Cash provided by financial activities
(Financial Cash), *(v)* Net change in cash (Change in Cash), and *(vi)*
Quarter Keys. As shown in Table
<a href="#tab:data_stats" data-reference-type="ref"
data-reference="tab:data_stats">[tab:data_stats]</a>, our LongForms
dataset contains 140 forms that consist of 685 pages, 168458 words, and
1128 entities in total. The models are trained to predict $n_i$ given
both $w_i$ and $b_i$.

<span id="tab:data_stats" label="tab:data_stats"></span>

# LongFin Model [sec:longlilt]

<div class="figure*" markdown="1">

<figure id="fig:longlilt">
<img src="/papers/multipage_ocr/arXiv-2401.15050v1_md/longfin.png" style="width:100.0%" />
<figcaption><span>LongFin</span></figcaption>
</figure>

<figure id="fig:attention">

<figcaption><span>Local + Global Atention</span></figcaption>
</figure>

<span id="fig:models" label="fig:models"></span>

</div>

## Architecture

Figure <a href="#fig:models" data-reference-type="ref"
data-reference="fig:models">[fig:models]</a> illustrates the overall
architecture of our proposed model, LongFin, which builds upon recently
published models: LiLT [lilt](https://doi.org/10.18653/v1/2022.acl-long.534) and Longformer
[longformer](https://arxiv.org/pdf/2004.05150). Similar to LiLT [lilt](https://doi.org/10.18653/v1/2022.acl-long.534),
LongFin comprises three primary components: a text encoder, a layout
encoder, and the BiACM (bidirectional attention complementation
mechanism) layer [lilt](https://doi.org/10.18653/v1/2022.acl-long.534). However, LongFin introduces
additional mechanisms, namely sliding window local attention and
interval-based global attention, to effectively handle long contexts
within both the text and layout encoders. One key advantage of LongFin
is its ability to scale linearly with the input sequence length, in
contrast to the quadratic scaling ($O(n^2)$) observed in the original
transformers’ [vaswani2017attention](https://arxiv.org/pdf/1706.03762) attention mechanism.
This linear scaling, inspired by the Longformer model
[longformer](https://arxiv.org/pdf/2004.05150), allows LongFin to efficiently handle long
contexts up to 4K tokens.

### Text Encoder

For the text encoder in LongFin, we adopt the Longformer
[longformer](https://arxiv.org/pdf/2004.05150) model, which has been pretrained to handle
long textual contexts of up to 4096 tokens. As depicted in Figure
<a href="#fig:longlilt" data-reference-type="ref"
data-reference="fig:longlilt">2</a>, the input to the text encoder
consists of two types of embeddings: text embeddings ($E_{T}$) and
absolute position embeddings ($E_{P}$). These embeddings are added
together to produce the final embeddings ($E_{final}$). Subsequently, a
layer normalization [layernormalization](https://arxiv.org/pdf/1607.06450) operation is
applied, and the resulting output is fed into the encoder.

The attention mechanism in LongFin incorporates two types of attention:
local attention and global attention. The local attention employs a
sliding window approach, where each token attends to the 512 local
tokens surrounding it. On the other hand, the global attention involves
a set of global tokens, selected at intervals of 100. While other
approaches [longformer](https://arxiv.org/pdf/2004.05150), [longdocument](http://arxiv.org/pdf/2108.09190v2) may employ
different methods for selecting global tokens, such as random selection
or task-specific strategies, we limit our experimentation to
interval-based selection for simplicity and due to limited computational
resources. Each token in the input sequence attends to these global
tokens, in addition to its local context as shown in Figure
<a href="#fig:attention" data-reference-type="ref"
data-reference="fig:attention">3</a>. This combination of local and
global attention mechanisms enhances the model’s ability to capture both
local context and broader global dependencies within the long input
sequences.

### Layout Encoder

For the layout encoder in LongFin, we adopt the layout encoder utilized
in the LiLT model [lilt](https://doi.org/10.18653/v1/2022.acl-long.534). Similar to the text encoder,
the input for the layout encoder comprises two types of embeddings:
absolute position embeddings and layout embeddings. Each word in the
input document is associated with a bounding box that defines its
location within the document layout. This bounding box is represented by
four numbers: $x_0$, $y_0$, $x_1$, and $y_1$, which correspond to the
coordinates of the top-left and bottom-right points of the bounding box.
To normalize these coordinates within the range \[0,1000\], we use the
page’s height and width.

To generate the layout embedding for each word, each coordinate in the
normalized bounding box is used to obtain an embedding vector. The
different coordinates’ embedding vectors are then concatenated and
projected using a linear layer. The resulting layout embeddings are
added to the absolute position embeddings to obtain the final
embeddings. These final embeddings are then fed into the layout encoder.
Similar to the text encoder, we also employ the local & global attention
mechanisms in the layout encoder to process long sequences.

### BiACM

To facilitate communication between the text encoder and layout encoder,
we incorporate the BiACM layer from the LiLT model
[lilt](https://doi.org/10.18653/v1/2022.acl-long.534). As depicted in Figure
<a href="#fig:longlilt" data-reference-type="ref"
data-reference="fig:longlilt">2</a>, the BiACM layer adds the scores
resulting from the multiplication of keys and queries from both
encoders. In LiLT, a detach operation is applied to the scores generated
by the text encoder before passing them to the layout encoder. This
detachment prevents the layout encoder from backpropagating into the
text encoder during pretraining, promoting better generalization when
fine-tuning the model with different language text encoders. However,
since our focus is primarily on the English language for our
applications, we have chosen to remove the detach operation to expedite
pretraining, given our limited computational resources.

## Pretraining [sec:pretraining]

To pretrain LongFin, we utilize the IIT-CDIP [iit](https://doi.org/10.1145/1148170.1148307)
dataset which contains 11M scanned images that make up 6M documents. We
obtain the OCR annotations (words and their bounding boxes) from OCR-IDL
[ocraws](http://arxiv.org/pdf/2202.12985v1) which used the AWS Textract API[^4]. We
initialize our text encoder from Longformer [longformer](https://arxiv.org/pdf/2004.05150)
and our layout encoder from LiLT [lilt](https://doi.org/10.18653/v1/2022.acl-long.534) layout encoder.
Since the LiLT layout encoder was pretrained on inputs with a maximum
length of 512 tokens, we copy LiLT’s pretrained positional embeddings
eight times to initialize our layout encoder positional embeddings,
which consist of 4096 embedding vectors. This enables the layout encoder
to handle longer sequences while leveraging the pretrained positional
information from the LiLT model.

For the pretraining of LongFin, we employ the Masked Visual-Language
Modeling task [bert](https://arxiv.org/pdf/1810.04805), [lilt](https://doi.org/10.18653/v1/2022.acl-long.534). In this task, 15% of the
tokens in the input to the text encoder are masked. In 80% of the cases,
we replace the masked tokens with the
<span class="smallcaps">\[MASK\]</span> token. In 10% of the cases, we
replace the masked tokens with random tokens. In the remaining 10%, we
keep the original token unchanged. Inspired by Longformer
[longformer](https://arxiv.org/pdf/2004.05150), we pretrain the model for 65K steps with a
learning rate of 3e-5 and batch size of 12 on one A100 GPU. We set the
warmup steps to 500 and use the AdaFactor optimizer
[shazeer2018adafactor](https://arxiv.org/pdf/1804.04235). Also, we utilize gradient
checkpointing [gradientcheckpointing](https://arxiv.org/pdf/1604.06174) to enable using a
large batch size. The pretraining loss curve is shown in Figure
<a href="#fig:loss_curve" data-reference-type="ref"
data-reference="fig:loss_curve">4</a>

<figure id="fig:loss_curve">
<img src="/papers/multipage_ocr/arXiv-2401.15050v1_md/loss_curve.png" style="width:45.0%" />
<figcaption>LongFin pretraining loss curve. The loss starts at 2.84 and
oscillated between 1.97 and 1.94 near convergence. </figcaption>
</figure>

# Experiments & Evaluation [sec:evaluation]

## Tasks & Datasets

To assess the generalizability of LongFin on both short and long
contexts, we evaluate LongFin on two existing short (single-page)
datasets: FUNSD [funsd](http://arxiv.org/pdf/1905.13538v2) and CORD [cord](http://arxiv.org/pdf/2103.10213v1) to
show the generalizability of our model on short contexts as well as our
newly created LongForms dataset.

**$\bullet$** : This dataset comprises 200 scanned forms and requires
models to extract four main entities: headers, questions, answers, and
other relevant information. Additionally, it involves linking questions
with their corresponding answers, thereby encompassing named entity
recognition and relation extraction tasks. We mainly focus on the named
entity recognition task and use the entity-level F1 score as our
evaluation metric.

**$\bullet$** : With over 11,000 receipts, this dataset focuses on
extracting 54 different data elements (e.g., phone numbers) from
receipts. The task can be formulated as named entity recognition or
token classification. For evaluation, we use the entity-level F1 score.

## Baselines

To demonstrate the effectiveness of LongFin on our LongForms dataset, we
compare it against a set of publicly available text and text+layout
baselines that are capable of handling both short and long input
sequences. For the text baselines, we select the following models: *(i)*
BERT [bert](https://arxiv.org/pdf/1810.04805) which is a widely used text-based model known
for its strong performance on short context tasks (512 tokens), *(ii)*
Longformer [longformer](https://arxiv.org/pdf/2004.05150) which is specifically designed to
handle text long texts (up to 4096 tokens). For the text+layout
baseline, we utilize LiLT [lilt](https://doi.org/10.18653/v1/2022.acl-long.534), which is one of the
state-of-the-art models for document understanding [^5]. For the short
context models, we split the LongForms documents into chunks that can
fit within 512 tokens. Table
<a href="#tab:finetuningdetails" data-reference-type="ref"
data-reference="tab:finetuningdetails">[tab:finetuningdetails]</a> shows
the hyperparameters of the different models when finetuning on the
LongForms dataset. It also presents the hyperparameters we used when
finetuning LongFin on the previous single-page datasets. All the
finetuning experiments were performed on one A100 and one T4 GPUs.

## Results

<span id="tab:prev_datasets" label="tab:prev_datasets"></span>

## Previous (Single-Page) Datasets

As shown in Table <a href="#tab:prev_datasets" data-reference-type="ref"
data-reference="tab:prev_datasets">[tab:prev_datasets]</a>, LongFin
outperforms other long-context models such as Longformer
[longformer](https://arxiv.org/pdf/2004.05150) and [longdocument](http://arxiv.org/pdf/2108.09190v2) on the
previous datasets that mainly consist of single-page documents. The
performance disparity is particularly pronounced on the FUNSD dataset
[funsd](http://arxiv.org/pdf/1905.13538v2), where all documents have very short textual
content (less than 512 tokens). Notably, LongFin also achieves
comparable performance to the short-context models on these datasets.
This comparison highlights the superior generalization ability of our
model, LongFin, which performs well on both short and long contexts. In
contrast, the performance of [longdocument](http://arxiv.org/pdf/2108.09190v2) model
deteriorates on short-context documents.

<span id="tab:longfin_results" label="tab:longfin_results"></span>

## LongForms Dataset [longforms-dataset]

As presented in Table
<a href="#tab:longfin_results" data-reference-type="ref"
data-reference="tab:longfin_results">[tab:longfin_results]</a>, the
performance results on our LongForms dataset highlight the advantage of
our model, LongFin, compared to the short-context models. This
observation emphasizes the significance of long-context understanding
when working with financial documents. There is also a noticeable
difference in performance between the text models (BERT
[bert](https://arxiv.org/pdf/1810.04805) and Longformer [longformer](https://arxiv.org/pdf/2004.05150)) and
text+layout models (LiLT [lilt](https://doi.org/10.18653/v1/2022.acl-long.534) and LongFin). This is
mainly because the documents in LongForms contain diverse layouts that
might be challenging for text-only models.

To provide a deeper analysis of the results on the LongForms dataset, we
conduct ablations and report metrics by entity for both LiLT
[lilt](https://doi.org/10.18653/v1/2022.acl-long.534) and LongFin, as shown in Table
<a href="#tab:longfin_ablations" data-reference-type="ref"
data-reference="tab:longfin_ablations">[tab:longfin_ablations]</a>. We
notice that the gap in performance is more significant in the entities
that are typically found in long tables such as Beginning Cash, Ending
Cash, Financial Cash, and Change in Cash. To illustrate the challenges
posed by long tables, we present an examples from our test set in Figure
<a href="#fig:test_example_pred" data-reference-type="ref"
data-reference="fig:test_example_pred">[fig:test_example_pred]</a>. In
the example, the table header indicates "Nine Months," implying that the
table includes information for a nine-month period that should not be
extracted as we are only interested in the financial information per
quarter "Three Months". Due to the large number of rows and content in
the table, the short-context models may not be able to include all the
table information in a single forward pass of 512 tokens. Consequently,
when the long documents are split into chunks, such tables might be
divided as well, leading to the short-context models losing important
context when making predictions.

<span id="tab:longfin_ablations" label="tab:longfin_ablations"></span>

<div class="figure*" markdown="1">

<img src="/papers/multipage_ocr/arXiv-2401.15050v1_md/Example_for_paper.png" style="width:60.0%" alt="image" />

<span id="fig:test_example_pred" label="fig:test_example_pred"></span>

</div>

# Limitations

Despite the effectiveness of our model, LongFin, on both short and long
context document understanding datasets, it has a few limitations.
First, LongFin was trained and evaluated on the English language only.
In future, we plan to expand it to support multiple languages. Second,
although LongFin maximum input length (4096 tokens) can accommodate the
multi-page documents in the LongForms dataset as well as most our
proprietary datasets, it might not accommodate certain financial
documents that contain tens of pages. To overcome this limitation, we
may consider further expanding the positional embeddings to accomodate
16K tokens similar to the LED model [longformer](https://arxiv.org/pdf/2004.05150) or
explore utlizing a model architecture that uses relative position
embeddings [shaw-etal-2018-self](https://doi.org/10.18653/v1/N18-2074) such as T5
[t5](http://jmlr.org/papers/v21/20-074.html) instead of the absolute position embeddings. Third,
due to limited computational resources, we have not explored many
different hyperparameters setup. Hence, there might be room for
improvement in our model performance. Finally, while our LongForms shed
the light on long context understanding challenges which are frequent in
the financial industry, it is still limited in size. We encourage the
research community to explore this undercharted area of research since
it has various commercial applications in many industries such as
finance and legal.

# Conclusion

We introduce LongFin, a multimodal document AI model designed to handle
lengthy documents. Additionally, we present the LongForms dataset, which
aims to replicate real-world challenges in understanding long contexts,
specifically in the financial industry. Through our evaluation, we
demonstrate the superior performance of LongFin on the LongForms
dataset, which comprises multi-page documents, while achieving
comparable results on previous datasets consisting of single-page
documents. Moving forward, our plan is to deploy LongFin after training
it on our proprietary datasets in the finance domain. Furthermore, we
are working on extending LongFin to support different languages.

<span id="sec:appendix" label="sec:appendix"></span>

# Ethical Statement

All the documents used in our LongForms dataset is collected from the
EDGAR database which grants the right to use and distribute their data
without permissions [^6]. The dataset annotation process were
accomplished by data annotators who are fairly compensated. We provide
the hyperparameters and experimental setups of our experiments to ensure
the reproducibility of our work. Moreover, the models, LiLT
[lilt](https://doi.org/10.18653/v1/2022.acl-long.534) and Longformer [longformer](https://arxiv.org/pdf/2004.05150), on
which our LongFin model is built are published under permissive licenses
[^7][^8] that allow commercial use.

[^1]: https://www.sec.gov/edgar/

[^2]: https://www.sec.gov/edgar/

[^3]: https://pypi.org/project/pdftotext/

[^4]: https://aws.amazon.com/textract/

[^5]: LayoutLMv3 [layoutlmv3](https://doi.org/10.1145/3503161.3548112) is another state-of-the-art
    document understanding model, but its usage is limited to
    non-commercial applications

[^6]: https://www.sec.gov/privacy#dissemination

[^7]: https://github.com/allenai/longformer

[^8]: https://github.com/jpWang/LiLT