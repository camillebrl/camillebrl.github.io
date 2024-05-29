Cheng *et al.*: Bare Demo of IEEEtran.cls for Computer Society Journals

[^1]: $^*$Z. Cheng, P. Zhang and C. Li contributed equally to this
    research.



visually rich document (VRD) is a traditional yet very important
research topic
[zhang2020trie](None), [katti2018chargrid](None), [zhao2019cutie](None), [palm2017cloudscan](None), [sage2019recurrent](None), [Aslan2016APB](None), [Janssen2012Receipts2GoTB](None), [dengel2002smartfix](None), [schuster2013intellix](None), [Simon1997AFA](None).
This is because automatically understanding VRDs can greatly facilitate
the key information entry, retrieval and compliance check in enormous
and various applications, including file understanding in court trial,
contract checking in the business system, statements analysis in
accounting or financial, case recognition in medical applications,
invoice recognition in reimburses system, resume recognition in
recruitment system, and automatically examining test paper in education
applications, etc.

In general, a VRD system can be divided into two separated parts: text
reading and key information extraction. Text reading module refers to
obtaining text positions as well as their character sequence in document
images, which falls into the computer vision areas related to optical
character recognition (*abbr*. OCR)
[wang2020all](None), [qiao2020textperceptron](None), [feng2019textdragon](None), [liao2017textboxes](None), [jaderberg2016reading](None), [wang2012end](http://arxiv.org/pdf/2207.04651v1), [shi2016end](None), [liao2019mask](None).
Information extraction (IE) module is responsible for mining key
contents (entity, relation) from the captured plain text, related to
natural language processing (NLP) techniques like named entity
recognition (NER)
[nadeau2007survey](None), [lample2016neural](None), [ma2019end](None) and
question-answer
[yang2016stacked](None), [anderson2018bottom](None), [fukui2016multimodal](None).

<figure id="fig:framework">
<embed src="/papers/visionrich_small_dec/arXiv-2207.06744v1_md/images/framework.png" style="width:50.0%" />
<figcaption>Illustration of the proposed end-to-end VRD framework. It
consists of three sub-modules: the text reading part for generating text
layout and character strings, and the information extraction module for
outputting key contents. The multi-modal context block is responsible
for fully assembling visual, textual, layout features, and even language
knowledge, and bridges the text reading and information extraction parts
in an end-to-end trainable manner. Dashed lines denote back-propagation.
</figcaption>
</figure>

Early works  [palm2017cloudscan](None), [sage2019recurrent](None)
implement the VRD frameworks by directly concatenating an offline OCR
engine and the downstream NER-based IE module, which completely discards
the visual features and position/layout[^1] information from images.
However, as appearing in many applications
[palm2017cloudscan](None), [zhang2020trie](None), [dengel2002smartfix](None), [schuster2013intellix](None), [sun2021spatial](None), [wang2021tag](None),
VRDs are usually organized with both semantic text features and flexible
visual structure features in a regular way. For better results,
researchers should consider the key characteristics of documents into
their techniques, such as layout, tabular structure, or even font size
in addition to the plain text. Then recent works begin to incorporate
these characteristics into the IE module by embedding multi-dimensional
information such as text content and their layouts
[katti2018chargrid](None), [denk2019bertgrid](None), [zhao2019cutie](None), [palm2019attend](None), [liu2019graph](None),
and even image features
[xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2), [PICK2020YU](None), [Xu2020LayoutLMv2MP](None).

Unfortunately, all existing methods suffer from two main problems:
First, multi-modality features (like visual, textual and even layout
features) are essential for VRD , but the exploitation of the
multi-modal features is limited in previous methods. Contributions of
different kinds of features should be addressed for the IE part. For
another, text reading and IE modules are highly correlated, but their
contribution and relations have rarely been explored.

<div class="table*" markdown="1">

|  |  |  |
|:--:|:---|:---|
|  |  |  |
| type | Structured | Semi-structured |
| 1-3 Fixed |  |  |
| value-added tax invoice [liu2019graph](None), passport [qin2019eaten](None), fixed-format taxi invoice [zhang2020trie](None), |  |  |
| national ID card [Zhenlong2019TowardsPE](None), train ticket [PICK2020YU](None), [Janssen2012Receipts2GoTB](None), business license [wang2021tag](None) |  |  |
| business email[Harley2015EvaluationOD](http://arxiv.org/pdf/1502.07058v1), |  |  |
| national housing contract |  |  |
| 1-3 Variable |  |  |
| medical invoice [PICK2020YU](None), [dengel2002smartfix](None), paper head[wang2021towards](None), bank card [Zhenlong2019TowardsPE](None), |  |  |
| free-format invoice[palm2017cloudscan](None), [katti2018chargrid](None), [palm2019attend](None), [MajumderPTWZN20](None), [Rusiol2013FieldEF](None), [Ha2018RecognitionOO](None), [zhang2020trie](None), business card [qin2019eaten](None), |  |  |
| purchase receipt [zhao2019cutie](None), [liu2019graph](None), [PICK2020YU](None), [Janssen2012Receipts2GoTB](None), [sun2021spatial](None), purchase orders[sage2019recurrent](None), [xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2) |  |  |
| personal resume [zhang2020trie](None), |  |  |
| financial report [Harley2015EvaluationOD](http://arxiv.org/pdf/1502.07058v1),newspaper[Yang2017LearningTE](None), |  |  |
| free-format sales contract[Gralinski2020KleisterAN](None) |  |  |

</div>

Considering the above issues, in this paper, we propose a novel
end-to-end . The workflow is as shown in
Figure <a href="#fig:framework" data-reference-type="ref"
data-reference="fig:framework">1</a>. Instead of focusing on information
extraction task only, we bridge *text reading* and *information
extraction* tasks via a developed multi-modal context block. In this
way, two separated tasks can reinforce each other amidst a unified
framework. Specifically, the text reading module produces diversiform
features, including layout features, visual features and textual
features. The multi-modal context block fuses multi-modal features with
the following steps: (1) Layout features, visual features and textual
features are first fed into the multi-modal embedding module, obtaining
their embedding representation. (2) Considering the effectiveness of the
language model like BERT [denk2019bertgrid](None), (3) The
embedded features are then correlated with the spatial-aware attention
to learn the instance-level interactions. It means different text
instances may have explicit or implicit interactions, the ‘Total-Key’
and ‘Total-Value’ in receipts are highly correlated.

Consequently, the multi-modal context block can provide robust features
for the information extraction module, and the supervisions in
information extraction also contribute to the optimization of text
reading. Since all the modules in the network are differentiable, the
whole network could be trained in a global optimization way. To the best
of our knowledge, this is the first end-to-end trainable framework.

We also notice that it is difficult to compare existing methods directly
due to the different benchmarks used (most of them are private), the
non-uniform evaluation protocols, and even various experimental
settings. As is known to all, text reading
[Chen2020TextRI](None) is a rapidly growing research area,
attributing to its various applications and its uniform benchmarks and
evaluation protocols. We here reckon that these factors may restrict the
study of document understanding. To remedy this problem, we first
analyze many kinds of documents, and then categorize VRDs into four
groups along the dimensions of *layout* and *text type*. *Layout* refers
to the relative position distribution of texts or text blocks, which
contains two modes: the fixed mode and the variable mode. The former
connotes documents that follow a uniform layout format, such as passport
and the national value-added tax invoice, while the latter means that
documents may appear in different layouts. Referring to
[judd2004apparatus](http://arxiv.org/pdf/2305.19912v1), [soderland1999learning](None), we define
*text type* into two modalities[^2] : the structured and the
semi-structured. In detail, the structured type means that document
information is organized in a predetermined schema, i.e., the key-value
schema of the document is predefined and often tabular in style, which
delimits entities to be extracted directly. For example, taxi invoices
usually have quite a uniform tabular-like layout and information
structure like ‘Invoice Number’, ‘Total’, ‘Date’ etc. The
semi-structured type connotes that document content is usually
ungrammatical, but each portion of the content is not necessarily
organized in a predetermined format. For example, a resume may include
some predefined fields such as job experience and education information.
Within the job experience fields, the document may include free text to
describe the person’s job experience. Then, the user may desire to
search on free text only within the job experience field.
Table <a href="#table:dataset_summary" data-reference-type="ref"
data-reference="table:dataset_summary">[table:dataset_summary]</a>
summarizes the categories of visually rich documents from the previous
research literature. Secondly, we recommend or provide the corresponding
benchmarks for each kind of documents, and also provide the uniform
evaluation protocols, experimental settings and strong baselines,
expecting to promote this research area.

Major contributions are summarized as follows. (1) We propose an
end-to-end trainable framework TRIE++ for , which can be trained from
scratch, with no need for stage-wise training strategies. (2) We
implement the framework by simultaneously learning text reading and
information extraction tasks via a well-designed multi-modal context
block, and also verify the mutual influence of text reading and
information extraction. (3) To make evaluations more comprehensive and
convincing, we define and divide VRDs into four categories, in which
three kinds of real-life benchmarks are collected with full annotations.
For each kind of document, we provide the corresponding benchmarks,
experimental settings, and strong baselines. (4) Extensive evaluations
on four kinds of real-world benchmarks show superior performance
compared with the state-of-the-art. Those benchmarks cover diverse types
of document images, from fixed to variable layouts, from structured to
semi-unstructured text types.

Declaration of major extensions compared to the conference version
[zhang2020trie](None): (1) Instead of modelling context with
only layout and textual features in [zhang2020trie](None), we
here enhance the multi-modal context block by fusing three kinds of
features (layout, visual and textual features) with a spatial-aware
attention mechanism. Besides, we expand the application ranges of our
method, showing the ability to handle with four kinds of VRDs. (2)
Following the suggestions in the conference reviews that the prior
knowledge may be helpful to our method, we also attempt to introduce the
pre-trained language model [denk2019bertgrid](None) into the
framework with a knowledge absorption module for further improving the
information extraction performance. (3) We address the problem of
performance comparison in existing methods, and then define the four
categories of VRDs. To promote the document understanding area, we
recommend the corresponding benchmarks, experimental settings, and
strong baselines for each kind of document. (4) We explore the effects
of the proposed framework with more extensive experimental evaluations ,
which demonstrates its advantages.

[^1]: Note that, terms of ‘position’ and ‘layout’ are two different but
    highly relevant concepts. The former refers to the specific
    coordinate locations of candidate text regions generated by text
    reading module. The later means the abstract spatial information
    (position arrangement of text regions) derived from the generated
    position results via some embedding operations. Thus, layout can be
    treated as the high-level of spatial information in document
    understanding. In the follow-up, we use term ‘layout’ instead of
    term ‘position’ as one kind of modality.

[^2]: Another text type, the unstructured, is also defined in
    [judd2004apparatus](http://arxiv.org/pdf/2305.19912v1), which means that document
    content is grammatically free text without explicit identifiers such
    as books. Since such documents usually lack visually rich elements
    (layout), we exclude it from the concept of VRD.

# Related Works [related_work]

Thanks to the rapid expansion of artificial intelligence techniques
[zhuang2020next](None), advanced progress has been made in many
isolated applications such as document layout analysis
[esser2012automatic](http://arxiv.org/pdf/2312.02941v1), [xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2), scene text spotting
[liu2018fots](None), [Qiao2020MANGOAM](None), video understanding
[xu2019segregated](None), named entities identification
[yadav2019survey](None), question answering
[duan2018temporality](http://arxiv.org/pdf/2103.12876v1), or even causal inference
[kuang2020causal](http://arxiv.org/pdf/acc-phys/9411001v1) etc. However, it is crucial to build
multiple knowledge representations for understanding the complex and
challenging world. VRD is such a real task greatly helping office
automation, which relies on integrating multiple techniques, including
object detection, sequence learning, information extraction and even the
multi-modal knowledge representation. Here, we roughly brief techniques
as follows.

## Text Reading

Text reading belongs to the OCR research field and has been widely
studied for decades. A text reading system usually consists of two
parts: text detection and text recognition.

In *text detection*, methods are usually divided into two categories:
anchor-based methods and segmentation-based methods. Following Faster
R-CNN [RenHG017](None), anchor-based
methods [he2017single](None), [liao2017textboxes](None), [liao2018textboxes++](None), [liao2018rotation](None), [ma2018arbitrary](None), [liu2017deep](None), [shi2017detecting](None), [Rosetta18Borisyuk](None)
predicted the existence of texts and regress their location offsets at
pre-defined grid points of the input image. To localize arbitrary-shaped
text, Mask RCNN [HeGDG17mask](None)-based methods
[xie2018scene](None), [Zhang2019look](None), [liu2019Towards](None) were
developed to capture irregular text and achieve better performance.
Compared to anchor-based methods, segmentation can easily be used to
describe the arbitrary-shaped text. Therefore, many segmentation-based
methods [zhou2017east](None), [long2018textsnake](None), [Wang2019Shape](None), [xu2019textfield](None)
were developed to learn the pixel-level classification tasks to separate
text regions apart from the background. In *text recognition*, the
encoder-decoder architecture
[CRNN](None), [shi2018aster](None), [cheng2017focusing](None) dominates the
research field, including two mainstreaming routes:
CTC[Graves2006](None)-based
[shi2016end](None), [Rosetta18Borisyuk](None), [wang2017gated](None), [R2AM](None) and
attention-based
[cheng2017focusing](None), [shi2018aster](None), [cheng2018aon](None) methods. To
achieve the global optimization between detection and recognition, many
end-to-end trainable
methods [liu2018fots](None), [li2017towards](None), [he2018end](None), [busta2017deep](None), [wang2020all](None), [qiao2020textperceptron](None), [feng2019textdragon](None), [MaskTextspotter18Lyu](None), [Qiao2020MANGOAM](None)
were proposed, and achieved better results than the pipeline approaches.

## Information Extraction

Information extraction is a traditional research topic and has been
studied for many years. Here, we divide existing methods into two
categories as follows.

### Rule-based Methods

Before the advent of learning-based models, rule-based
methods[riloff1993automatically](None), [huffman1995learning](http://arxiv.org/pdf/1904.02634v1), [muslea1999extraction](None), [dengel2002smartfix](None), [schuster2013intellix](None), [esser2012automatic](http://arxiv.org/pdf/2312.02941v1)
dominated this research area. It is intuitive that the key information
can be identified by matching a predefined pattern or template in the
unstructured text. Therefore, expressive pattern matching languages
[riloff1993automatically](None), [huffman1995learning](http://arxiv.org/pdf/1904.02634v1) were
developed to analyze syntactic sentence, and then output one or multiple
target values.

To extract information from general documents such as business
documents, many solutions
[dengel2002smartfix](None), [schuster2013intellix](None), [Rusiol2013FieldEF](None), [esser2012automatic](http://arxiv.org/pdf/2312.02941v1), [Medvet2010APA](http://arxiv.org/pdf/2005.01646v1)
were developed by using the pattern matching approaches. In detail,
[schuster2013intellix](None), [Rusiol2013FieldEF](None), [Cesarini2003AnalysisAU](http://arxiv.org/pdf/2311.11856v1)
required a predefined document template with relevant key fields
annotated, and then automatically generated patterns matching those
fields.
[dengel2002smartfix](None), [esser2012automatic](http://arxiv.org/pdf/2312.02941v1), [Medvet2010APA](http://arxiv.org/pdf/2005.01646v1) all
manually configured patterns based on keywords, parsing rules or
positions. The rule-based methods heavily rely on the predefined
template, and are limited to the documents with unseen templates. As a
result, it usually requires deep expertise and a large time cost to
conduct the templates’ design and maintenance.

### Learning-based Methods

Learning-based methods can automatically extract key information by
applying machine learning techniques to a prepared training dataset.

Traditionally machine learning techniques like logistic regression and
SVM were widely adopted in document analysis tasks.
[Shilman2005LearningNG](http://arxiv.org/pdf/2304.01746v1) proposed a general machine
learning approach for the hierarchical segmentation and labeling of
document layout structures. This approach modeled document layout as
grammar and performed a global search for the optimal parse based on a
grammatical cost function. This method utilized machine learning to
discriminatively select features and set all parameters in the parsing
process.

The early methods often ignore the layout information in the document,
and then the document understanding task is downgraded to the pure NLP
problem. That is, many named entity recognition (NER) based methods
 [lample2016neural](None), [ma2019end](None), [yadav2019survey](None), [devlin2018bert](None), [dai2019transformer](None), [yang2019xlnet](None)
can be applied to extract key information from the one-dimensional plain
text. Inspired by this idea,  [palm2017cloudscan](None)
proposed CloudScan, an invoice analysis system, which used recurrent
neural networks to extract entities of interest from VRDs instead of
templates of invoice layout.  [sage2019recurrent](None)
proposed a token level recurrent neural network for end-to-end table
field extraction that starts with the sequence of document tokens
segmented by an OCR engine and directly tags each token with one of the
possible field types. However, they discard the layout information
during the text serialization, which is crucial for document
understanding.

Observing the rich layout and visual information contained in document
images, researchers tended to incorporate more details from VRDs. Some
works
[katti2018chargrid](None), [denk2019bertgrid](None), [zhao2019cutie](None), [palm2019attend](None), [wang2021tag](None)
took the layout into consideration, and worked on the reconstructed
character or word segmentation of the document. Concretely,
[katti2018chargrid](None) first achieved a new type of text
representation by encoding each document page as a two-dimensional grid
of characters. Then they developed a generic document understanding
pipeline named Chargrid for structured documents by a fully
convolutional encoder-decoder network. As an extension of Chargrid,
[denk2019bertgrid](None) proposed Bertgrid in combination with
a fully convolutional network on a semantic instance segmentation task
for extracting fields from invoices. To further explore the effective
information from both semantic meaning and spatial distribution of texts
in documents, [zhao2019cutie](None) proposed a convolutional
universal text information extractor by applying convolutional neural
networks on gridding texts where texts are embedded as features with
semantic connotations. [palm2019attend](None) proposed the
attend, copy, parse architecture, an end-to-end trainable model
bypassing the need for word-level labels. [wang2021tag](None)
proposed a tag, copy or predict network by first modelling the semantic
and layout information in 2D OCR results, and then learning the
information extraction in a weakly supervised manner. Contemporaneous
with the above-mentioned methods, there are methods
[liu2019graph](None), [MajumderPTWZN20](None), [sun2021spatial](None), [xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2), [Xu2020LayoutLMv2MP](None), [li2021structurallm](None), [li2021structext](None)
which resort to graph modeling to learn relations between multimodal
inputs. [liu2019graph](None) introduced a graph
convolution-based model to combine textual and layout information
presented in VRDs, in which graph embedding was trained to summarize the
context of a text segment in the document, and further combined with
text embedding for entity extraction. [MajumderPTWZN20](None)
presented a representation learning approach to extract structured
information from templatic documents, which worked in the pipeline of
candidate generation, scoring and assignment.
[sun2021spatial](None) modelled document images as
dual-modality graphs by encoding both textual and visual features, then
generated key information with the proposed Spatial Dual-Modality Graph
Reasoning method (SDMG-R). Besides, they also released a new dataset
named WildReceipt.

## End-to-End Information Extraction from VRDs

Two related concurrent works were presented
in [qin2019eaten](None), [carbonell2019treynet](None).
[qin2019eaten](None) proposed an entity-aware attention text
extraction network to extract entities from VRDs. However, it could only
process documents of relatively fixed layout and structured text, like
train tickets, passports and business cards.
[carbonell2019treynet](None) localized, recognized and
classified each word in the document. Since it worked in the word
granularity, it required much more labeling efforts (layouts, content
and category of each word) and had difficulties extracting those
entities which were embedded in word texts (extracting ‘51xxxx@xxx.com’
from ‘153-xxx97$|$`<!-- -->`{=html}51xxxx@xxx.com’). Besides, in its
entity recognition branch, it still worked on the serialized word
features, which were sorted and packed in the left to right and top to
bottom order. The two existing works are strictly limited to documents
of relatively fixed layout and one type of text (structured or
semi-structured). Similar to the conference version
[zhang2020trie](None) of our method,
[wang2021towards](None) recently proposed an end-to-end
framework accompanied by a Chinese examination paper head dataset.
Unlike them, our method acts as a general , and can handle documents of
both fixed and variable layouts, structured and semi-structured text
types.

# Methodology

<figure id="fig:system_architecture">
<img src="images/system_architecture" style="width:49.0%" />
<figcaption>The overall framework. The network predicts text locations,
text contents and key entities in a single forward pass. </figcaption>
</figure>

This section introduces the proposed framework, which has three parts:
text reading, multi-modal context block and information extraction
module, as shown in
Figure <a href="#fig:system_architecture" data-reference-type="ref"
data-reference="fig:system_architecture">1</a>.

## Text Reading

Text reading module commonly includes a shared convolutional backbone, a
text detection branch as well as a text recognition branch. We use
ResNet-D [he2019bag](http://arxiv.org/pdf/2001.03992v1) and Feature Pyramid Network (FPN)
[LinDGHHB17feature](None) as our backbone to extract the shared
convolutional features. For an input image $x$, we denote $\mathcal{I}$
as the shared feature maps.

**Text detection**. The branch takes $\mathcal{I}$ as input and predicts
the locations of all candidate text regions, i.e., $$\label{equa1}
\mathcal{B}=\textit{Detector}(\mathcal{I})$$ where the
$\textit{Detector}$ can be the
anchor-based [he2017single](None), [liao2017textboxes](None), [liu2017deep](None), [shi2017detecting](None)
or segmentation-based
 [zhou2017east](None), [long2018textsnake](None), [Wang2019Shape](None) text
detection heads. $\mathcal{B}=(b_1, b_2,\dots, b_m)$ is a set of $m$
text bounding boxes, and $b_i=(x_{i0}, y_{i0},$ $x_{i1}, y_{i1})$
denotes the top-left and bottom-right positions of the $i$-th text. In
mainstream methods, RoI-like operations (*e.g.*, RoI-Pooling
[RenHG017](None) used in [li2017towards](None),
ROI-Align [HeGDG17mask](None) used in
[he2018end](None), RoI-Rotate used in
[liu2018fots](None), or even RoI-based arbitrary-shaped
transformation [qiao2020textperceptron](None), [wang2020all](None)) are
applied on the shared convolutional features $\mathcal{I}$ to get their
text instance features. Here, the text instance features are denoted as
$\mathcal{C}=(c_1, c_2,\dots, c_m)$. The detailed network architecture
is shown in Section <a href="#sec-impl" data-reference-type="ref"
data-reference="sec-impl">[sec-impl]</a>.

**Text recognition**. The branch predicts a character sequence from each
text region features $c_i$. Firstly, each instance feature $c_i$ is fed
into an encoder (CNN and LSTM [LSTM](None)) to extract a
higher-level feature sequence $\mathcal{H}=(h_1, h_2, \dots, h_l)$,
where $l$ is the length of the extracted feature sequence. Then, a
general sequence decoder (attention-based
[shi2016end](None), [cheng2017focusing](None)) is adopted to generate
the sequence of characters $y=(y_1, y_2,\dots, y_T)$, where $T$ is the
length of label sequence. Details are shown in Section
<a href="#sec-impl" data-reference-type="ref"
data-reference="sec-impl">[sec-impl]</a>.

We choose attention-based sequence decoder as the character recognizer.
It is a recurrent neural network that directly generates the character
sequence $y$ from an input feature sequence $\mathcal{H}$.

## Multi-modal Context Block

We design a multi-modal context block to consider layout features,
visual features and textual features altogether. Different modalities of
information are complementary to each other, and fully fused for
providing robust multi-modal feature representation.

### Multi-modal Feature Generation

Document details such as the apparent color, font, layout and other
informative features also play an important role in document
understanding.

A natural way of capturing the layout and visual features of a text is
to resort to the convolutional neural network. Concretely, the position
information of each text instance is obtained from the detection branch,
i.e., $\mathcal{B}=(b_1, b_2,\dots, b_m)$. For visual feature, different
from [xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2), [Xu2020LayoutLMv2MP](None) which extract
these features from scratch, we directly reuse text instance features
$\mathcal{C}=(c_1, c_2, \dots, c_m)$ by text reading module as the
visual features. Thanks to the deep backbone and lateral connections
introduced by FPN, each $c_i$ summarizes the rich local visual patterns
of the $i$-th text.

In sequence decoder, give the $i$-th text instance, its represented
feature of characters before softmax contain rich semantic information.
For the attention-based decoder, we can directly use
$z_i=(s_1, s_2, \dots, s_T)$ as its textual features.

### Prior Knowledge Absorption

Since pre-trained language model contains general language knowledge
like semantic properties, absorbing knowledge from the language model
may help improve the performance of information extraction. Compared to
the conference paper [zhang2020trie](None), we here attempt to
bring the language model into our framework. However, prior language
information has different contributions on different VRDs. For example,
on Resume scenario that require semantics, prior language information
contributes more, while on Taxi scenario which requires less semantics,
prior language information contributes less. Inspired by the gating
operation in LSTM [LSTM](None), we design a gated knowledge
absorption mechanism to adjust the prior knowledge flows in our
framework, as shown in Figure
<a href="#fig:prior" data-reference-type="ref"
data-reference="fig:prior">2</a>.

In order to dynamically determine the degree of dependency of the
pre-trained model, we use an on-off gate $g^\prime$
$$g^\prime = \sigma(W_{g^\prime}a + U_{g^\prime}z + b_{g^\prime})$$ to
balance the flow of the prior knowledge activation $r^\prime$
$$r^\prime = \delta(W_{r^\prime}a + U_{r^\prime}z + b_{r^\prime}).$$
Here, the gate is used for determining whether general knowledge is
needed. Then the modulated textual feature $o$ is calculated as
$$\label{gating}
o = g^\prime \odot r^\prime + W_oz.$$

<figure id="fig:prior">
<img src="images/prior-fusion" style="width:45.0%" />
<figcaption> </figcaption>
</figure>

### Multi-modal Context Modelling

We first embed each modality information into feature sequences with the
same dimension, and fuse them with a normalization layer. Inspired by
the powerful Transformer
[devlin2018bert](None), [VisualBERTLi](None), [Lu2019ViLBERT](None), [Xu2020LayoutLMv2MP](None),
the self-attention mechanism is used to build deep relations among
different modalities,

**Multi-modal Feature Embedding** Given a document with $m$ text
instance, we can capture the inputs of position
$\mathcal{B}=(b_1,b_2,\dots,b_m)$, the inputs of visual feature
$\mathcal{C}=(c_1,c_2,\dots,c_m)$ and the inputs of modulated textual
feature $o=(o_1,o_2,\dots,o_m)$.

Since position information provides layout information of documents, we
introduce a position embedding layer to preserve layout information, for
the $i$-th text instance in a document,
$$pe_i=\sum_{j=1}^{|b_i|} embedding(b_{ij}),$$ where $embedding$ is a
learnable embedding layer, $b_i=(x_{i0},y_{i0},x_{i1},y_{i1})$ and
$pe_i\in \mathbb{R}^{d_e}$.

For $c_i$ visual feature, we embed it using a convolutional neural
network layer with the same shape of $pe_i$,
$$\widehat{c_i}=ConvNet_c(c_i).$$

For $o_i$ textual feature, a $ConvNet$ of multiple kernels similar
to [zhang2015character](None) is used to aggregate semantic
character features in $o_i$ and outputs
$\widehat{z_i}\in\mathbb{R}^{d_e}$, $$\widehat{z_i}=ConvNet_z(o_i). 
\label{eq:textual}$$

Then, the $i$-th text’s embedding is fused of $\widehat{c_i}$,
$\widehat{z_i}$ and $pe_{i}$, followed by the $LayerNorm$ normalization,
defined as $$emb_i=LayerNorm(\widehat{c_i} + \widehat{z_i} + pe_i).$$
Afterwards, we pack all the texts’ embedding vector together, i.e.,
$emb=(emb_1, emb_2, \dots, emb_m)$, which serves as the $K$, $Q$ and $V$
in the scaled dot-product attention.

**Spatial-Aware Self-Attention** To better learn pair-wise interactions
between text instances, we use the spatial-aware self-attention
mechanism instead of the original self-attention, and the correlative
context features
$\widetilde{\mathcal{C}}=(\widetilde{c_1}, \widetilde{c_2}, \dots, \widetilde{c_m})$
are obtained by, $$\begin{split}
\widetilde{\mathcal{C}}&=Attention(Q,K,V) \\
&=softmax(\frac{QK^\mathsf{T}}{\sqrt{d_{info}}}+pe_{\Delta \mathcal{B}})V
\end{split}$$ where $d_{info}$ is the dimension of text embedding, and
$\sqrt{d_{info}}$ is the scaling factor. $pe_{\Delta \mathcal{B}}$
refers to the spatial-aware information, and is calculated by embedding
features of position relations ${\Delta \mathcal{B}}$ among different
text instances in $\mathcal{B}$, i.e.,
$pe_{\Delta \mathcal{B}}= embedding({\Delta \mathcal{B}})$. Here,
${\Delta \mathcal{B}}$ is defined as $$\Delta \mathcal{B} = 
\left[
\begin{array}{cccc}
0           & b_1-b_2   & \cdots    & b_1-b_m\\
b_2-b_1     & 0             &   \cdots  & b_2-b_m\\
\cdots  & \cdots    & \cdots        &\cdots \\
b_m-b_1     & b_m-b_2   &   \cdots  & 0
\end{array}
\right].$$ To further improve the representation capacity of the
attended feature, multi-head attention is introduced. Each head
corresponds to an independent scaled dot-product attention function and
the text context features $\widetilde{\mathcal{C}}$ is given by:
$$\begin{split}
\widetilde{\mathcal{C}}&=MultiHead(Q,K,V)\\
&=[head_1, head_2, ..., head_n]W^{info}
\end{split}$$ $$head_j=Attention(QW_j^Q, KW_j^K, VW_j^V)$$ where
$W^Q_j$, $W^K_j$ and $W^V_j$ $\in \mathbb{R}^{(d_{info}\times d_n)}$ are
the learned projection matrix for the $j$-th head, $n$ is the number of
heads, and $W^{info}\in \mathbb{R}^{(d_{info} \times d_{info})}$. To
prevent the multi-head attention model from becoming too large, we
usually have $d_n = \frac{d_{info}}{n}$.

**Context Fusion** Both the multi-modal context and textual features
matter in entity extraction. The multi-modal context features
($\widetilde{\mathcal{C}}$) provide necessary information to tell
entities apart while the textual features $o$ enable entity extraction
in the character granularity, as they contain semantic features for each
character in the text. Thus, we need to fuse them further. That is, for
the $i$-the text instance, we pack its multi-modal context vector
$\widetilde{c_i}$ and its modulated textual features $o_i$ together
along the channel dimension, i.e., $(u_{i1}, u_{i2},\dots, u_{iT})$
where $u_{ij}=[o_{i,j}, c_i]$.

## Information Extraction [ie]

Then, a Bidirectional-LSTM is applied to further model the long
dependencies within the characters,
$$H_{i}^\prime=(h_{i,1}^\prime, h_{i,2}^\prime, \dots, h_{i,T}^\prime) = BiLSTM(u_i),$$
which is followed by a fully connected network and a layer, projecting
the output to the dimension of [SangV99representing](None)
label space. $$p_{i,j}^{info} = CRF(Linear(h_{i,j}^\prime))$$

## Optimization [sec3.5]

The proposed network can be trained in an end-to-end manner and the
losses are generated from three parts, $$\label{losses}
\mathcal{L}=\mathcal{L}_{det} + \lambda_{recog}\mathcal{L}_{recog} + \lambda_{info}\mathcal{L}_{info}$$
where hyper-parameters $\lambda_{recog}$ and $\lambda_{info}$ control
the trade-off between losses.

$\mathcal{L}_{det}$ is the loss of text detection branch, which can be
formulated as different forms according to the selected detection heads.
Taking Faster-RCNN [RenHG017](None) as the detection head, the
detection part consists of a classification loss and a regression loss.

For sequence recognition part, the attention-based recognition loss is
$$\mathcal{L}_{recog}=-\frac{1}{T}\sum_{i=1}^{m}\sum_{t=1}^{T}log\ p(\hat{y}_{i,t}|\mathcal{H}),$$
where $\hat{y}_{i,t}$ is the ground-truth label of $t$-th character in
$i$-th text from recognition branch.

The information extraction loss is the CRFLoss, as used
in [lample2016neural](None), [wang2021towards](None).

Note that since *text reading* and *information extraction* modules are
bridged with the multi-modal context block, they can reinforce each
other. Specifically, the multi-modality features of text reading are
fully fused and essential for information extraction. At the same time,
the semantic feedback of information extraction also contributes to the
optimization of the shared convolutions and text reading module.

<div class="table*" markdown="1">

<div class="center" markdown="1">

| Category | Dataset | \#Training | \#Validation | \#Testing | \#Entity | \#Instance | Annotation Type | Source |
|:--:|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|  | Train ticket [qin2019eaten](None) | 271.44k | 30.16k | 400 | 5 | \- | \[*entity*\] | Syn |
|  | Passport [qin2019eaten](None) | 88.2k | 9.8k | 2k | 5 | \- | \[*entity*\] | Syn |
|  | Taxi Invoice | 4000 | \- | 1000 | 9 | 136,240 | \[*pos, text, entity*\] | Real |
|  | Business Email | 1146 | \- | 499 | 16 | 35,346 | \[*pos, text, entity*\] | Real |
|  | SROIE [HuangCHBKLJ19competition](None) | 626 | \- | 347 | 4 | 52,451 | \[*pos, text, entity*\] | Real |
|  | Business card [qin2019eaten](None) | 178.2k | 19.8k | 2k | 9 | \- | \[*entity*\] | Syn |
|  | FUNSD [Jaume2019FUNSDAD](None) | 149 | \- | 50 | 4 | 31,485 | \[*pos, text, entity*\] | Real |
|  | CORD$^3$ [Park2019CORDAC](None) | 800 | 100 | 100 | 30 | \- | \[*pos, text, entity*\] | Real |
|  | EPHOIE [wang2021towards](None) | 1183 | \- | 311 | 12 | 15,771 | \[*pos, text, entity*\] | Real |
|  | WildReceipt [sun2021spatial](None) | 1267 | \- | 472 | 26 | 69,000 | \[*pos, text, entity*\] | Real |
|  | Kleister-NDA[Gralinski2020KleisterAN](None) | 254 | 83 | 203 | 4 | \- | \[*pos, text, entity*\] | Real |
|  | Resume | 1045 | \- | 482 | 11 | 82,800 | \[*pos, text, entity*\] | Real |

</div>

</div>

# Benchmarks [benchmark]

As addressed in Section 1, most existing works verify their methods on
private datasets due to their privacy policies. It leads to difficulties
for fair comparisons between different approaches. Though existing
datasets like SROIE [HuangCHBKLJ19competition](None) have been
released, they mainly fall into Category III, i.e., documents with
variable layout and structured text type. The remaining three kinds of
application scenarios (Category I, II and IV) have not been studied well
because of the limited real-life datasets.

## Dataset inventory

To boost the research of VRD understanding, we here extend the
benchmarks of VRD, especially on Category I, II and IV. Table
<a href="#table:datasets" data-reference-type="ref"
data-reference="table:datasets">[table:datasets]</a> shows the detailed
statistics of these benchmarks.

-   *Category I* refers to document images with uniform layout and
    structured text type, which is very common in everyday life.
    Contrastively, its research datasets are very limited due to various
    privacy policies. Here, we find only two available benchmarks, i.e.,
    train ticket and passport dataset released by
    [qin2019eaten](None), which are generated with a synthetic
    data engine and provide only entity-level annotations. To remedy
    this issue, we release a new real-world dataset containing 5000 taxi
    invoice images. Except for providing the text position and character
    string information for OCR tasks (text detection and recognition),
    entity-level labels including 9 entities (Invoice code, Invoice
    number, Date, Get-on time, Get-off time, Price, Distance, Wait time,
    Total) are also provided. Besides, this dataset is very challenging,
    as many images are in low-quality (such as blur and occlusion).

-   *Category II* refers to those documents with fixed layout and
    semi-structured text type, like business email or national housing
    contract. NER datasets like CLUENER2020
    [xu2020cluener2020](None) are only collected for NLP tasks,
    and they provide only semantic content while ignoring the important
    layout information. As addressed in Section
    <a href="#sec:introduction" data-reference-type="ref"
    data-reference="sec:introduction">[sec:introduction]</a>, the joint
    study of OCR and IE is essential. Unfortunately, we have not found
    available datasets that contains both OCR and IE annotations. We
    also ascribe the issue to various privacy policies. We here collect
    a new business email dataset from RVL-CDIP
    [Harley2015EvaluationOD](http://arxiv.org/pdf/1502.07058v1), which has 1645 email images
    with 35346 text instances and 15 entities (To, From, CC, Subject,
    BCC, Text, Attachment, Date, To-key, From-key, CC-key, Subject-key,
    BCC-key, Attachment-key, Date-key).

-   *Category III* means documents which are with variable layout and
    structured text type like purchase receipt dataset SROIE
    [HuangCHBKLJ19competition](None). These datasets are
    usually composed of small documents (*e.g.*, purchase receipts,
    business cards, etc.), and entities are organized in a predetermined
    schema. We note that most previous literature focus on this
    category. We here list five available datasets. SROIE is a scanned
    receipt dataset widely evaluated in many methods, which is fully
    annotated and provides text position, character string and key-value
    labels. Business card is a synthesized dataset released by
    [qin2019eaten](None), and has only key-value pair
    annotations without OCR annotations. FUNSD
    [Jaume2019FUNSDAD](None) is a dataset aiming at extracting
    and structuring the textual content from noisy scanned forms. It has
    only 199 forms with four kinds of entities, i.e., question, answer,
    header and other. CORD$^2$ [Park2019CORDAC](None) is a
    consolidated receipt dataset, in which images are with text
    position, character string and multi-level semantic labels. EPHOIE
    [wang2021towards](None) is a Chinese examination paper head
    dataset, in which each image is cropped from the full examination
    paper. This dataset contains handwritten information, and is also
    fully annotated. WildReceipt [sun2021spatial](None) is a
    large receipt dataset collected from document images of unseen
    templates in the wild. It contains 25 key information categories, a
    total of about 69000 text boxes.

-   *Category IV* means documents that have variable layout and
    semi-structured text type. Different from those datasets in Category
    III, Kleister-NDA[Gralinski2020KleisterAN](None) aims to
    understand long documents (i.e., Non-disclosure Agreements
    document), but it provides only 540 documents with four general
    entity classes. To enrich benchmarks in this category, we release a
    large-scale resume dataset, which has 1527 images with ten kinds of
    entities(Name, Time, School, Degree, Specialty, Phone number,
    E-mail, Birth, Title, Security code). Since resumes are personally
    designed and customized, it is a classic document dataset with
    variable layouts and semi-structured text.

## Challenges in different kinds of documents

It will be the most straightforward task to extract entities from
documents in Category I, which attributes to its complete fixed layout
and structured text type. For this kind of documents, challenges are
mainly from the text reading part, such as the distorted interference.
The standard object detection methods like Faster-RCNN
[RenHG017](None) also can be further developed to handle this
task. In Category II, the layout is fixed, but the text is
semi-structured. Thus, in addition to modelling layout information, we
also should pay attention to mining textual information. Then some NLP
techniques like the pre-trained language model can be exploited. As to
the text reading part, long text recognition is also challenging.
Documents in Category III face the problem of complex layout. Thus the
layout modelling methods [liu2019graph](None), [PICK2020YU](None) like
graph neural networks are widely developed for coping with this issue.
The documents in Category IV are in the face of both complex layout and
NLP problems, which becomes the most challenging task.

# Experiments [experiment]

In subsection <a href="#sec-impl" data-reference-type="ref"
data-reference="sec-impl">1.1</a>, we first introduce the implementation
details of network and training skills. In subsection
<a href="#ablation" data-reference-type="ref"
data-reference="ablation">1.2</a>, we perform ablation study to verify
the effectiveness of the proposed method on four kinds of VRD datasets,
i.e., Taxi Invoice, Business Email, WildReceipt and Resume. In
subsection
<a href="#sota" data-reference-type="ref" data-reference="sota">1.3</a>,
we compare our method with existing approaches on several recent
datasets like FUNSD, SROIE, EPHOIE and WildReceipt, demonstrating the
advantages of the proposed method. Then, we provide a group of strong
baselines on four kinds of VRDs in subsection
<a href="#baseline" data-reference-type="ref"
data-reference="baseline">1.4</a>. Finally, we discuss the challenges of
the different categories of documents. Codes and models are available at
*https://davar-lab.github.io/publication/trie++.html*.

## Implementation Details [sec-impl]

### Data Selecting

To facilitate end-to-end document understanding (*text reading* and
*information extraction*), datasets should have position, text and
entity annotations. Hence, we only consider those datasets which satisfy
the above requirement. On the ablation and strong baseline experiments,
we select one classic dataset from each category, which has the largest
number of samples. They are Taxi Invoice dataset from Category I,
Business Email dataset from Category II, WildReceipt dataset from
Category III and Resume dataset from Category IV. When compared with the
state-of-the-arts, since they mainly report their results on popular
SROIE, FUNSD and EPHOIE benchmarks, we also include these benchmarks in
Section <a href="#sota" data-reference-type="ref" data-reference="sota">1.3</a>.

### Network Details

The backbone of our model is ResNet-D [he2019bag](http://arxiv.org/pdf/2001.03992v1),
followed by the FPN [LinDGHHB17feature](None) to further
enhance features. The text detection branch in *text reading module*
adopts the Faster R-CNN [RenHG017](None) network and outputs
the predicted bounding boxes of possible texts for later sequential
recognition. For each text region, its features are extracted from the
shared convolutional features by RoIAlign [HeGDG17mask](None).
The shapes are represented as $32\times256$ for Taxi Invoice and
WildReceipt, and $32\times512$ for Business Email and Resume. Then,
features are further decoded by LSTM-based attention
[cheng2017focusing](None), where the number of hidden units is
set to 256.

In the *multimodal context block*, BERT [devlin2018bert](None)
is used as the pre-trained language model. Then, convolutions of four
kernel size $[3, 5, 7, 9]$ followed by max pooling are used to extract
final textual features.

In the *information extraction module*, the number of hidden units of
BiLSTM used in entity extraction is set to 128. Hyper-parameters
$\lambda_{recog}$ and $\lambda_{info}$ in Equation
<a href="#losses" data-reference-type="ref"
data-reference="losses">[losses]</a> are all empirically set to 1 in our
experiments.

### Training Details

Our model and its counterparts are implemented under the PyTorch
framework [paszke2019pytorch](None). For our model, the AdamW
[loshchilov2017decoupled](http://arxiv.org/pdf/2311.11446v2) optimization is used. We set
the learning rate to 1e-4 at the beginning and decreased it to a tenth
at 50, 70 and 80 epochs. The batch size is set to 2 per GPU. For the
counterparts, we separately train text reading and information
extraction tasks until they are fully converged. All the experiments are
carried out on a workstation with 8 NVIDIA A100 GPUs.

### Evaluation Protocols [protocals]

We also note that different evaluation protocols are adopted in previous
works. For example in the evaluation of information extraction part,
both EATEN [qin2019eaten](None) and PICK
[PICK2020YU](None) used the defined mean entity accuracy (mEA)
and mean entity f1-score (mEF) as metrics. CUTIE
[zhao2019cutie](None) adopted the average precision (AP) as the
metric, and Chargrid [katti2018chargrid](None) developed new
evaluation metric like word error rate for evaluation. While the
majority of methods
[zhang2020trie](None), [Gralinski2020KleisterAN](None), [xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2)
used the F1-score as the evaluation metric. As a result, the non-uniform
evaluation protocols bring extra difficulties on comparisons. Therefore,
we attempt to describe a group of uniform evaluation protocols for VRD
understanding by carefully analyzing previous methods, including the
evaluation protocols of text reading and information extraction parts.

Text reading falls into the OCR community, and it has uniform evaluation
standards by referring to mainstream text detection
[liao2017textboxes](None), [liu2019Towards](None), [liu2018fots](None) and text
recognition [CRNN](None), [shi2018aster](None), [cheng2017focusing](None)
methods. *precision* (*abbr*. PRE$_d$) and *recall* (*abbr*. REC$_d$)
are used to measure performance of text localization, and *F-measure*
(*abbr*. F$_d$-m) is the harmonic average of *precision* and *recall*.
To evaluate text recognition, the *accuracy* (abbr. ACC) used in
[CRNN](None), [shi2018aster](None), [cheng2017focusing](None) is treat as its
measurement metric. When evaluating the performance of end-to-end text
detection and recognition, the end-to-end level evaluating metrics like
precision (denoted by PRE$_r$), recall (denoted by REC$_r$) and
F-measure (denoted by F$_r$-m) following [2011End](None)
without lexicon is used, in which all detection results are considered
with an IoU$>$`<!-- -->`{=html}0.5.

For information extraction, we survey the evaluation metrics from recent
research works
[zhang2020trie](None), [Gralinski2020KleisterAN](None), [xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2), [Jaume2019FUNSDAD](None), [liu2019graph](None), [wang2021towards](None), [Xu2020LayoutLMv2MP](None),
and find that the precision, recall and F1-score of entity extraction
are widely used. Hereby, we recommend the *entity precision* (abbr.
ePRE), *entity recall* (abbr. eREC) and *entity F1-score* (eF1) as the
evaluation metrics for this task.

## Ablation Study [ablation]

In this section, we perform the ablation study on Taxi Invoice, Business
Email, WildReceipt and Resume datasets to verify the effects of
different components in the proposed framework.

### Effects of multi-modality features [forward_effect]

To examine the contributions of visual, layout and textual features to
information extraction, we perform the following ablation study on four
kinds of datasets, and the results are shown in
Table <a href="#table:performance-contribution" data-reference-type="ref"
data-reference="table:performance-contribution">1</a>. *Textual feature*
means that entities are extracted using features from the text reading
module only. Since the layout information is completely lost, this
method presents the worst performance. Introducing either the *visual
features* or *layout features* brings significant performance gains.
Further fusion of the above multi-modality features gives the best
performance, which verifies the effects. We also show examples in
Figure. <a href="#fig:modality_contribution" data-reference-type="ref"
data-reference="fig:modality_contribution">[fig:modality_contribution]</a>
to verify their effects. By using the *textual feature* only, the model
misses the ‘Store-Name’ and has confusion between ‘Total’ and
‘Product-Price’ entities. Combined with the *layout feature*, the model
can recognize ‘Product-Price’ correctly. When combined with the *visual
feature*, the model can recognize Store-Name, because the *visual
feature* contains obvious visual clues such as the large font size. It
shows the best result by integrating all modality features.

<div class="center" markdown="1">

<div id="table:performance-contribution" markdown="1">

|                 |         |         |         |           |
|:----------------|:-------:|:-------:|:-------:|:---------:|
| Textual feature | $\surd$ | $\surd$ | $\surd$ |  $\surd$  |
| Layout feature  |         | $\surd$ |         |  $\surd$  |
| Visual feature  |         |         | $\surd$ |  $\surd$  |
| Taxi Invoice    |  90.34  |  98.45  |  98.71  | **98.73** |
| Business Email  |  74.51  |  82.88  |  86.02  | **87.33** |
| WildReceipt     |  72.9   |  87.75  |  83.62  | **89.62** |
| Resume          |  76.73  |  82.26  |  82.62  | **83.16** |

Accuracy results (eF1) with multi-modal features on information
extraction.

</div>

</div>

<div class="figure*" markdown="1">

<embed src="/papers/visionrich_small_dec/arXiv-2207.06744v1_md/images/modality_contr.png" />

</div>

### Effects of different components

From Table <a href="#table:components" data-reference-type="ref"
data-reference="table:components">[table:components]</a>, we see that
SaSa can boost performance, especially on the WildReceipt. This is
because, compared to the original self-attention using entities’
absolute positions only, the spatial-aware self-attention also makes use
of relative position offsets between entities, and learns their pairwise
relations. Visual examples are shown in
Figure. <a href="#fig:ras_vs_sas" data-reference-type="ref"
data-reference="fig:ras_vs_sas">1</a>. We see that ‘Product-Item’ and
‘Product-Price’ always appear in pairs. Spatial-aware self-attention can
capture such pairwise relations and then improve model performances. Its
attention map is visualized in
Figure. <a href="#fig:vis_sas" data-reference-type="ref"
data-reference="fig:vis_sas">2</a>, which demonstrates that the
spatial-aware self-attention indeed learns the pairwise relations
between entities (pair of ‘Total-Key’ and ‘Total-Value’, and pair of
‘Product-Item’ and ‘Product-Price’).

<figure id="fig:ras_vs_sas">
<embed src="/papers/visionrich_small_dec/arXiv-2207.06744v1_md/images/rsa_vs_sas.png" style="width:45.0%" />
<figcaption>Visual examples of original self-attention and spatial-aware
self-attention. Different colors denote different entities, such as , ,
, , , , . Best viewed in color.</figcaption>
</figure>

<figure id="fig:vis_sas">
<embed src="/papers/visionrich_small_dec/arXiv-2207.06744v1_md/images/vis_sas.png" style="width:49.0%" />
<figcaption>Visualization of spatial-aware self-attention. Total-Key ()
and Total-Value (), Product-Item () and Product-Price () always appear
together, and their pairwise relations can be learned. Best viewed in
color and zoom in to observe other pairwise relations.</figcaption>
</figure>

<figure id="fig:layout-distri">
<embed src="/papers/visionrich_small_dec/arXiv-2207.06744v1_md/images/layout-distribute.png" style="width:45.0%" />
<figcaption aria-hidden="true"></figcaption>
</figure>

<div class="center" markdown="1">

</div>

When introducing the prior knowledge from Bert
[devlin2018bert](None), the performance of information
extraction is significantly improved on the scenarios that require
semantics like WildReceipt, Business Email and Resume. As shown in
Figure <a href="#fig:bert_contri" data-reference-type="ref"
data-reference="fig:bert_contri">4</a>, in the Resume case, introducing
the pre-trained language model helps recognize ‘School’ and ‘Specialty’
entities, which are hard to be extracted solely using textual features.

<div class="center" markdown="1">

<div id="table:gating" markdown="1">

| Strategy       | Concatenation | Summation |  Gating   |
|:---------------|:-------------:|:---------:|:---------:|
| Taxi Invoice   |     98.41     |   98.62   | **98.73** |
| Business Email |     87.06     |   86.19   | **87.33** |
| WildReceipt    |     87.95     |   88.47   | **89.62** |
| Resume         |     81.55     |   82.26   | **83.16** |

</div>

</div>

<figure id="fig:bert_contri">
<embed src="/papers/visionrich_small_dec/arXiv-2207.06744v1_md/images/bert_contr.png" style="width:50.0%" />
<figcaption>Illustration of pre-trained language model’s effects. Best
viewed in color and zoom in.</figcaption>
</figure>

### Effects of different number of layers and heads

Table <a href="#table:performance-layers-heads-global"
data-reference-type="ref"
data-reference="table:performance-layers-heads-global">3</a> analyzes
the effects of different numbers of layers and heads in the
spatial-aware self-attention. Taxi Invoices is relatively simple and has
a fixed layout. Thus the model with 1 or 2 layers and the small number
of heads achieves promising results. For scenes with complex layout
structures like Resumes and WildReceipt, deeper layers and heads can
help improve the accuracy results. In practice, one can adjust these
settings according to the complexity of a task.

<div class="center" markdown="1">

<div id="table:performance-layers-heads-global" markdown="1">

| 1-7     | Layers | Heads |       |       |       |           |
|:--------|:------:|:-----:|:-----:|:-----:|:-----:|:---------:|
| 3-7     |        |   2   |   4   |   8   |  16   |    32     |
| 1-7     |        |       |       |       |       |           |
| Invoice |   1    | 98.27 | 98.57 | 98.45 | 98.62 |   98.00   |
|         |   2    | 98.31 | 98.39 | 98.58 | 98.52 | **98.74** |
|         |   3    | 98.51 | 98.54 | 98.48 | 98.51 |   98.56   |
|         |   4    | 98.44 | 98.58 | 98.41 | 98.70 |   98.59   |
| 1-7     |        |       |       |       |       |           |
| Email   |   1    | 86.05 | 86.41 | 85.74 | 86.94 |   86.43   |
|         |   2    | 85.95 | 87.51 | 86.78 | 87.33 |   87.59   |
|         |   3    | 86.52 | 87.86 | 87.24 | 87.15 | **88.01** |
|         |   4    | 86.48 | 87.45 | 87.82 | 87.88 |   87.64   |
| 1-7     |   1    | 78.17 | 87.8  | 88.73 | 88.18 |   88.67   |
|         |   2    | 86.26 | 88.11 | 88.21 | 89.16 |   89.11   |
|         |   3    | 77.1  | 88.62 | 88.95 | 89.48 |   89.69   |
|         |   4    | 85.48 | 89.00 | 88.63 | 89.66 | **90.15** |
| 1-7     |   1    | 82.18 | 82.52 | 81.99 | 81.83 |   82.49   |
|         |   2    | 82.7  | 82.56 | 82.97 | 82.83 |   83.57   |
|         |   3    | 82.86 | 82.09 | 83.05 | 82.78 |   82.96   |
|         |   4    | 82.75 | 83.12 | 82.43 | 82.98 | **83.46** |
| 1-7     |        |       |       |       |       |           |

Accuracy results (eF1) with different number of layers and heads in
spatial-aware self-attention.

</div>

</div>

<div class="center" markdown="1">

<div id="table:performance-e2e-vs-pipeline" markdown="1">

|             |        |           |           |           |
|:-----------:|:------:|:---------:|:---------:|:---------:|
| 1-5 Dataset | Method |           |           |           |
|  (F$_d$-m)  |        |           |           |           |
|  (F$_r$-m)  |        |           |           |           |
|    (eF1)    |        |           |           |           |
|     1-5     | base1  | **95.72** | **91.15** |   88.29   |
|             | base2  |   95.21   |   91.05   |   88.28   |
|             |  e2e   |   94.85   |   91.07   | **88.46** |
| \[1pt/1pt\] | base1  |   97.12   |   55.88   |   45.24   |
|             | base2  |   97.10   |   56.18   |   45.47   |
|             |  e2e   | **97.22** | **56.83** | **45.71** |
| \[1pt/1pt\] | base1  |   90.31   |   73.52   |   69.37   |
|             | base2  |   90.55   |   74.98   |   71.15   |
|             |  e2e   | **90.73** | **76.50** | **73.12** |
| \[1pt/1pt\] | base1  |   96.71   |   55.15   |   58.53   |
|             | base2  |   96.86   |   55.56   |   58.31   |
|             |  e2e   | **96.88** | **55.66** | **58.77** |
|     1-5     |        |           |           |           |

</div>

</div>

<div class="center" markdown="1">

<div id="table:rir" markdown="1">

|         |             |           |           |           |
|:--------|:-----------:|:---------:|:---------:|:---------:|
| Method  |             |           |           |           |
| Invoice |             |           |           |           |
| Email   | WildReceipt |  Resume   |           |           |
| base2   |    99.41    | **95.76** |   94.68   |   95.12   |
| e2e     |  **99.45**  |   95.01   | **96.11** | **97.41** |

</div>

</div>

<div class="table*" markdown="1">

<div class="center" markdown="1">

</div>

</div>

<div class="table*" markdown="1">

<div class="center" markdown="1">

</div>

</div>

### Effects of the end-to-end training

To verify the effects of the end-to-end framework on text reading and
information extraction, we perform the following experiments on four
kinds of VRD datasets. We first define two strong baselines for
comparison. (1) *Base1*. The detection, recognition and information
extraction modules are separately trained, and then pipelined as an
inference model. (2) *Base2*. The detection and recognition tasks are
jointly optimized, and then pipelined with the separately trained
information extraction task. While joint training of the three modules
is denoted as our *end-to-end* framework. Notice that all multi-modal
features (See Section
<a href="#forward_effect" data-reference-type="ref"
data-reference="forward_effect">1.2.1</a>) are integrated. The layer and
head numbers in self-attention are set as (2, 2, 4, 2) and (32, 32, 16,
32) for four different tasks (Taxi Invoice, Business Email, WildReceipt,
Resume in order), respectively.

## Comparisons with the State-of-the-Arts [sota]

Recent methods
[xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2), [Xu2020LayoutLMv2MP](None), [li2021structurallm](None), [li2021structext](None)
focused on the information extraction task by adding great number of
extra training samples like IIT-CDIP dataset
[Lewis2006BuildingAT](http://arxiv.org/pdf/2305.06148v1) and
DocBank [li2020docbank](http://arxiv.org/pdf/2006.01038v3), and then have impressive results
on the downstream datasets. Following the typical routine, we also
compare our method with them on several popular benchmarks.

**Evaluation on FUNSD** The dataset is a noisy scanned from the dataset
with 200 images. The results are shown in FUNSD column of
Table <a href="#table:sotas" data-reference-type="ref"
data-reference="table:sotas">[table:sotas]</a>. To be fair, we first
compare our method with those without introducing extra data. Our method
significantly outperforms them with a large margin (83.53 *v.s.* 81.33
of MatchVIE[tang2021matchvie](None)). When comparing with
models trained with extra data, our method is still competitive. It only
falls behind the LLMv2[Xu2020LayoutLMv2MP](None) and
SLM[li2021structurallm](None).

**Evaluation on SROIE** The dataset has 963 scanned receipt images,
which is evaluated on four entities in many works. Most of the results
are impressive, as shown in SROIE column of
Table <a href="#table:sotas" data-reference-type="ref"
data-reference="table:sotas">[table:sotas]</a>. This is because methods
tend to achieve the performance upper bound of this dataset. For
example, StrucText [li2021structext](None) (with extra data)
has achieved 96.88 of *eF1*, which only has slight advantage over 96.57
of MatchVIE[tang2021matchvie](None). Our method shows promising
results on this benchmark, with 96.80 $eF1$ in the token granularity
(same to most
works [PICK2020YU](None), [wang2021tag](None), [wang2021towards](None), [tang2021matchvie](None), [xu2019layoutlm](http://arxiv.org/pdf/2205.00476v2), [Xu2020LayoutLMv2MP](None), [zhang2020trie](None))
and 98.37 in the segment granularity (same to
StrucText [li2021structext](None)).

**Evaluation on EPHOIE** The dataset is a Chinese examination paper head
dataset. Our method obviously surpasses previous methods Similar to
SROIE, its performance upper bound is limited. That is, only 1.15% of
improvement space is left.

**Evaluation on WildReceipt** This receipt dataset
[sun2021spatial](None) is more challenging than SROIE, which is
collected from document images with unseen templates in the wild. Most
of the methods like GAT[velivckovic2018graph](None) have rapid
performance degradation compared to results in SROIE and EPHOIE. While
our method still has the best result (90.15% of *eF1*) compared to
existing methods , which verifies the advantages of the proposed method.

## Strong Baselines on Four Categories of VRD [baseline]

For the pure information extraction task, their results (as shown in
Table <a href="#table:sotas" data-reference-type="ref"
data-reference="table:sotas">[table:sotas]</a>) are calculated based on
the ground truth of detection and recognition. However, the influence of
OCR should not be neglected in reality. Considering the real
applications, i.e., , one way is to divide the task as two pipelined
steps: (1) obtaining text spotting results with a public OCR engines,
(2) and then performing the information extraction. We here provide on
four kinds of VRDs.

### Comparison of Inference Speed

We evaluate the running time of our model and its counterparts in frames
per second (*abbr*. FPS). Results are as shown in the last column of
Table <a href="#table:baseline" data-reference-type="ref"
data-reference="table:baseline">[table:baseline]</a>. Thanks to feature
sharing between *text reading* and *information extraction* modules, A
more prominent trend is that the algorithm runs faster in scenarios
where the length of texts is short in a document (Taxi Invoice and
WildReceipt), while on Resume/Business Email datasets with long texts,
the FPS drops slightly.

### Evaluations among Different Modules

In the detection part, all methods achieve the satisfactory performance
of *F$_d$-m* (larger than 90%), while the performance on WildReceipt is
the lowest. This is because the receipt images in WildReceipt are
captured in the wild, and they are of non-front views, even with folds.
When considering the end-to-end text spotting task, results on Business
and Resume are poor due to the problems of character distortion and long
text. This problem will be a new research direction for OCR. For the
end-to-end information extraction, results on Business Email are the
worst, and the second-worst is Resume. It reveals that there is plenty
of work to do concerning end-to-end information extraction.

From the perspective of systems, we surprisingly discover that the text
recognition may be the top bottleneck for end-to-end understanding VRD
on Category II, III and IV. The information extraction is another
bottleneck due to the complex layouts and long character sentence
(Referring to Table <a href="#table:baseline" data-reference-type="ref"
data-reference="table:baseline">[table:baseline]</a>,
<a href="#table:performance-contribution" data-reference-type="ref"
data-reference="table:performance-contribution">1</a> and
<a href="#table:components" data-reference-type="ref"
data-reference="table:components">[table:components]</a>). Luckily, the
end-to-end training strategy can enhance both the text reading and the
final information extraction task. In future, more attention should be
paid to the effects of text reading *w.r.t* information extraction.

## Limitations

First, our method currently requires the annotations of position,
character string and entity labels of texts in a document, and the
labeling process is cost-expensive. We will resort to
semi/weakly-supervised learning algorithms to alleviate the problem in
the future. Another limitation is that the multi-modal context block
captures context in the instance granularity, which can be much more
fine-grained if introduced token/ character granularity context. Much
more fine-grained context is beneficial to extracting entities across
text instances.

# Conclusion

In this paper, we present an end-to-end trainable network integrating
text reading and information extraction for document understanding.
These two tasks can mutually reinforce each other via a multi-modal
context block, i.e., the multi-modal features, like visual, layout and
textual features, can boost the performances of information extraction,
while the loss of information extraction can also supervise the
optimization of text reading. On various benchmarks, from structured to
unstructured text type and fixed to variable layout, the proposed method
significantly outperforms previous methods. To promote the VRD
understanding research, we provide four kinds of benchmarks along the
dimensions of layout and text type, and also contribute four groups of
strong baselines for the future study.