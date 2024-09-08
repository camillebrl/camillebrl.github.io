# Introduction

In the era of burgeoning AI, especially in
LLMs/MLLMs [gpt4v](http://arxiv.org/pdf/2311.15732v2), [gpt4v_explore](http://arxiv.org/pdf/2312.15011v1), [team2023gemini](http://arxiv.org/pdf/2405.12107v1), [anthropic2024claude](http://arxiv.org/pdf/2007.04626v3), [reid2024gemini](http://arxiv.org/pdf/2312.17661v1), [bai2023qwen](http://arxiv.org/pdf/2309.16609v1), [lu2024deepseek](http://arxiv.org/pdf/2402.17510v1), [young2024yi](http://arxiv.org/pdf/2304.11090v4), [feng2023docpedia](http://arxiv.org/pdf/2311.11810v3), [feng2023unidoc](http://arxiv.org/pdf/2308.11592v2), [hu2024mplug](None), [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1), [tang2024textsquare](http://arxiv.org/pdf/2307.04087v3), [chen2024far](http://arxiv.org/pdf/2404.16821v2), [dong2024internlm](http://arxiv.org/pdf/2404.06512v1), [li2024mini](http://arxiv.org/pdf/2305.16318v2), [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/),
**Te**xt-**C**entric **V**isual **Q**uestion
**A**nswering (**TEC-VQA**) [biten2019scene](http://arxiv.org/pdf/2304.01603v1), [singh2019towards](http://arxiv.org/pdf/1811.11903v1), [feng2023unidoc](http://arxiv.org/pdf/2308.11592v2), [feng2023docpedia](http://arxiv.org/pdf/2311.11810v3), [tang2024textsquare](http://arxiv.org/pdf/2307.04087v3), [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1), [hu2024mplug](None)
has served as a *de facto* gold proxy to evaluate AI models in the
domain of text-centric scene understanding. Compared with general
VQA [biten2019scene](http://arxiv.org/pdf/2304.01603v1), [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1), [pham2024viocrvqa](http://arxiv.org/pdf/2404.18397v1), [singh2019towards](http://arxiv.org/pdf/1811.11903v1), [mishra2019ocr](http://arxiv.org/pdf/2010.02582v1), [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2), [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177), [zhu2016visual7w](http://arxiv.org/pdf/2306.04938v1), [krishna2017visual](http://arxiv.org/pdf/1602.07332v1), [antol2015vqa](http://arxiv.org/pdf/1309.1125v1), [marino2019ok](http://arxiv.org/pdf/1906.00067v2), [sheng2021human](http://arxiv.org/pdf/1810.02358v2), [liu2024visual](http://arxiv.org/pdf/2402.11690v1), [gao2015you](http://arxiv.org/pdf/1505.05612v3), [gan2020large](http://arxiv.org/pdf/2302.02502v2), [liu-etal-2021-visually](https://doi.org/10.18653/v1/2021.emnlp-main.818),
TEC-VQA places greater emphasis on answering questions that require
understanding textual information within images. It provides a
streamlined avenue for individuals without specialized expertise to
articulate their requirements and access applications in text-centric
visual environments. However, the majority of advancements in TEC-VQA
have predominantly concentrated on high-resource languages, *e.g.*,
English [biten2019scene](http://arxiv.org/pdf/2304.01603v1), [singh2019towards](http://arxiv.org/pdf/1811.11903v1), [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1), [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2),
Chinese [qi-etal-2022-dureadervis](https://doi.org/10.18653/v1/2022.findings-acl.105), [gao2015you](http://arxiv.org/pdf/1505.05612v3),
Japanese [shimizu2018visual](http://arxiv.org/pdf/1810.02358v2), [nguyen2023vlsp2022](http://arxiv.org/pdf/1810.02358v2) and
*etc.*, thus restricting the applicability of AI models to the global
community, particularly populations speaking low-resource languages.

To tackle the problem of language diversity, several seminal
studies [raj-khan-etal-2021-towards-developing](https://doi.org/10.18653/v1/2021.findings-emnlp.151), [pfeiffer-etal-2022-xgqa](https://doi.org/10.18653/v1/2022.findings-acl.196), [changpinyo-etal-2023-maxm](https://doi.org/10.18653/v1/2023.findings-emnlp.176)
in the general VQA field, leverage off-the-shelf translation engines to
expand existing question-answer pairs from high-resource languages to
their multilingual counterparts including low-resource ones. However,
when applied to TEC-VQA, this translation-based approach may fall prey
to the “*Visual-textual misalignment*” problem as only the text in
question-answer pairs can be processed, while the visual text present in
the images is overlooked. Not to mention issues such as nuanced meaning,
contextual distortion, language bias, and question type diversity
further render the transferability of the translation protocol
infeasible for TEC-VQA. The *status quo* begs for a question: “*How can
we address the visual-textual misalignment problem for multilingual
TEC-VQA and what we stand in the MLLM era?*”

<div class="figure*" markdown="1">

|  |  |
|:--:|:--:|
| <embed src="/papers/doc_ai_databases/arXiv-2405.11985v1_md/figure/distribution/pie_chart.png" style="width:49.0%" /> | <embed src="/papers/doc_ai_databases/arXiv-2405.11985v1_md/figure/distribution/data_distribution.png"
style="width:49.0%" /> |

</div>

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2405.11985v1_md/figure/examples/example_cropped.png" />

</div>

In this work, to answer the question above, we establish MTVQA, a novel
and high-quality multilingual TEC-VQA benchmark, where all images are
collected from real-world and meticulously annotated by human experts in
nine languages: Arabic (AR), Korean (KO), Japanese (JA), Thai (TH),
Vietnamese (VI), Russian (RU), French (FR), German (DE), and
Italian (IT). More concretely, to ensure the visual-textual alignment at
most, the annotation process follows the raise-then-correct paradigm,
where a group of human annotators raises several distinct questions,
ranging from simple content extraction to text-related reasoning, and
subsequently provides answers. These QA pairs are then double-checked by
another group to ensure accuracy and consistency. Consequently, as
illustrated in
Fig. <a href="#fig:leng_statistics" data-reference-type="ref"
data-reference="fig:leng_statistics">[fig:leng_statistics]</a>, 6,678
training images and 21,829 question-answer pairs, as well as 2,116 test
images and 6,778 question-answer pairs are obtained, covering several
fine-grained scenarios, such as menus, logos, maps, bills, PPTs,
research papers, and *etc*. To our best knowledge, MTVQA is the first
TEC-VQA dataset to provide native human annotations for multilingual
text-rich scenarios, especially for low-source languages. Furthermore,
we investigate recent representative MLLMs, including GPT-4V, Gemini,
QwenVL *etc*., by juxtaposing experimental results regarding their
performance on our newly proposed MTVQA. Both for general MLLMs and
document-focused ones, the results unequivocally demonstrate that
opportunities for improvement persist within these MLLMs when applied in
multilingual text-rich scenarios.

In summary, the main contributions of this paper can be categorized into
three points:

-   We introduce the MTVQA dataset, to the best of our knowledge, which
    is the first multilingual TEC-VQA benchmark to provide human expert
    annotations for text-centric scenarios.

-   We benchmark the state-of-the-art MLLMs on our new dataset and show
    there is still room for performance improvement for these models
    under multilingual text-rich scenarios.

-   We propose a set of baselines for multilingual TEC-VQA tasks.

# Related Work

## LLMs/MLLMs for text-centric VQA

Recent advancements in
LLMs/MLLMs [gpt4v](http://arxiv.org/pdf/2311.15732v2), [gpt4v_explore](http://arxiv.org/pdf/2312.15011v1), [team2023gemini](http://arxiv.org/pdf/2405.12107v1), [anthropic2024claude](http://arxiv.org/pdf/2007.04626v3), [reid2024gemini](http://arxiv.org/pdf/2312.17661v1), [bai2023qwen](http://arxiv.org/pdf/2309.16609v1), [lu2024deepseek](http://arxiv.org/pdf/2402.17510v1), [young2024yi](http://arxiv.org/pdf/2304.11090v4), [feng2023docpedia](http://arxiv.org/pdf/2311.11810v3), [feng2023unidoc](http://arxiv.org/pdf/2308.11592v2), [hu2024mplug](None), [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1), [tang2024textsquare](http://arxiv.org/pdf/2307.04087v3), [chen2024far](http://arxiv.org/pdf/2404.16821v2), [dong2024internlm](http://arxiv.org/pdf/2404.06512v1), [li2024mini](http://arxiv.org/pdf/2305.16318v2), [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
have revolutionized VQA tasks, as demonstrated by the remarkable
zero-shot performance of these models. Notably, the high
generalizability of LLMs/MLLMs, when explicitly trained on visual text
understanding datasets and fine-tuned with instructions, has
significantly enhanced their application in text-centric VQA
scenarios [feng2023unidoc](http://arxiv.org/pdf/2308.11592v2), [feng2023docpedia](http://arxiv.org/pdf/2311.11810v3), [tang2024textsquare](http://arxiv.org/pdf/2307.04087v3), [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1), [hu2024mplug](None).
For example, LLaVAR [zhang2023llavar](http://arxiv.org/pdf/2306.17107v2),
UniDoc [feng2023unidoc](http://arxiv.org/pdf/2308.11592v2), which extend
LLaVA [liu2024visual](http://arxiv.org/pdf/2402.11690v1) into the realm of document
understanding, pioneering the text-centric VQA of MLLMs by training them
to predict texts and coordinates from document images. Furthermore,
DocPedia [feng2023docpedia](http://arxiv.org/pdf/2311.11810v3) operates visual input in the
frequency domain rather than in space, which enables higher input
resolution without increasing the input sequence. Lately,
mPLUG-DocOwl [mPLUG-DocOwl](None),
Qwen-VL [bai2023qwen](http://arxiv.org/pdf/2309.16609v1), and
TextMonkey [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1) leverage publicly
available document-related VQA datasets to further enhance the
text-centric VQA capabilities. Despite the promising results achieved by
existing LLMs/MLLMs in text-centric VQA tasks, their focus on
high-resource languages such as English or Chinese has posed challenges
in achieving reasonable performance for low-resource languages. This is
primarily due to the lack of data or benchmarks for these low-resource
languages.

## Multilingual text-centric VQA Benchmarks

VQA has garnered significant attention in recent years, with numerous
studies, datasets, and benchmarks being proposed to advance the
field [biten2019scene](http://arxiv.org/pdf/2304.01603v1), [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1), [pham2024viocrvqa](http://arxiv.org/pdf/2404.18397v1), [singh2019towards](http://arxiv.org/pdf/1811.11903v1), [mishra2019ocr](http://arxiv.org/pdf/2010.02582v1), [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2), [masry-etal-2022-chartqa](https://doi.org/10.18653/v1/2022.findings-acl.177), [zhu2016visual7w](http://arxiv.org/pdf/2306.04938v1), [krishna2017visual](http://arxiv.org/pdf/1602.07332v1), [antol2015vqa](http://arxiv.org/pdf/1309.1125v1), [marino2019ok](http://arxiv.org/pdf/1906.00067v2), [sheng2021human](http://arxiv.org/pdf/1810.02358v2), [liu2024visual](http://arxiv.org/pdf/2402.11690v1), [gao2015you](http://arxiv.org/pdf/1505.05612v3), [gan2020large](http://arxiv.org/pdf/2302.02502v2), [liu-etal-2021-visually](https://doi.org/10.18653/v1/2021.emnlp-main.818).
Many datasets have been created that encompass scene text of various
domains, including natural
images [biten2019scene](http://arxiv.org/pdf/2304.01603v1), [singh2019towards](http://arxiv.org/pdf/1811.11903v1), scanned
documents [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1), [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2),
book and movie covers [mishra2019ocr](http://arxiv.org/pdf/2010.02582v1). One notable
limitation of these datasets is their predominant focus on
English [biten2019scene](http://arxiv.org/pdf/2304.01603v1), [singh2019towards](http://arxiv.org/pdf/1811.11903v1), [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1), [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2)
or other high-resource languages such as
Chinese [qi-etal-2022-dureadervis](https://doi.org/10.18653/v1/2022.findings-acl.105), [gao2015you](http://arxiv.org/pdf/1505.05612v3) and
Japanese [shimizu2018visual](http://arxiv.org/pdf/1810.02358v2), [nguyen2023vlsp2022](http://arxiv.org/pdf/1810.02358v2), which
restricts the applicability of VQA systems for low-resource languages
such as Thai and Vietnamese.

There is a recent effort toward extending VQA tasks to a wider range of
languages [gupta2020unified](http://arxiv.org/pdf/2204.14264v2), [pfeiffer-etal-2022-xgqa](https://doi.org/10.18653/v1/2022.findings-acl.196), [vivoli2022must](http://arxiv.org/pdf/1902.05660v1), [changpinyo-etal-2023-maxm](https://doi.org/10.18653/v1/2023.findings-emnlp.176), [li2023empirical](http://arxiv.org/pdf/1810.02358v2), [raj-khan-etal-2021-towards-developing](https://doi.org/10.18653/v1/2021.findings-emnlp.151)
by providing a multilingual VQA datasets. For example,
 [gao2015you](http://arxiv.org/pdf/1505.05612v3) created a free-form bilingual VQA dataset
(FM-IQA) contains over 150,000 images and 310,000 freestyle Chinese
question-answer pairs and their English translations.
[raj-khan-etal-2021-towards-developing](https://doi.org/10.18653/v1/2021.findings-emnlp.151) developed a
large-scale multilingual and code-mixed VQA dataset (MuCo-VQA)
supporting five languages. Of more relevance are the works xGQA (8
languages) [pfeiffer-etal-2022-xgqa](https://doi.org/10.18653/v1/2022.findings-acl.196) and MaXM (7
languages) [changpinyo-etal-2023-maxm](https://doi.org/10.18653/v1/2023.findings-emnlp.176), which apply
translation-based protocols to expand VQA data beyond English. However,
the translation-based multilingual VQA datasets inherently face issues,
such as the “Visual-textual misalignment” problem, where only the text
in question-answer pairs is processed, while the visual text in images
is overlooked. Additionally, the nuanced meaning and context are often
distorted; language bias introduced by machine translation models, and
the coverage of certain question types is limited, as highlighted
by [changpinyo-etal-2023-maxm](https://doi.org/10.18653/v1/2023.findings-emnlp.176). Moreover, none of the
previous multilingual datasets focus on text-centric scenarios where
multilingual text frequently occurs.

Our benchmark distinguishes itself by focusing on multilingual
text-centric VQA scenarios using human expert annotations. To the best
of our knowledge, the MTVQA benchmark is the first dataset to provide
native human annotations for such scenarios. It covers 9 languages,
thereby facilitating the training and evaluation of multilingual models
in diverse linguistic contexts. Additionally, our dataset can gauge the
VQA system’s ability for not only high-resource languages but also those
that are typically underrepresented in current
datasets [biten2019scene](http://arxiv.org/pdf/2304.01603v1), [singh2019towards](http://arxiv.org/pdf/1811.11903v1), [mathew2021docvqa](http://arxiv.org/pdf/2111.05547v1), [mathew2022infographicvqa](http://arxiv.org/pdf/2104.12756v2), [gao2015you](http://arxiv.org/pdf/1505.05612v3).

The MTVQA benchmark addresses a significant gap in existing datasets by
catering to the crucial needs of low-resource languages through
annotations from native speakers across multiple languages. Our
pioneering efforts distinctly position the MTVQA benchmark as a unique
multilingual VQA resource, advancing the frontier of machine learning
research.

# MTVQA Benchmark

The MTVQA Benchmark covers 9 languages: Arabic (AR), Korean (KO),
Japanese (JA), Thai (TH), Vietnamese (VI), Russian (RU), French (FR),
German (DE), and Italian (IT). In this section, we describe in detail
how we establish the MTVQA benchmark, including the collection of raw
image data and two-round human expert annotations, which are independent
of each other.

## Data Collection

Our purpose is to develop a multilingual VQA benchmark capable of
evaluating the QA performance of MLLMs in multilingual text-centric
scenarios, thus the raw data collection process is mainly oriented
towards text-centric images from natural scenarios and document
scenarios. To ensure the diversity and quality of data, we collect not
only the raw image data from publicly available datasets, including the
multilingual scene text recognition images from
MLT2019 [nayef2019icdar2019](http://arxiv.org/pdf/1909.07145v1) and PowerPoint slides (PPTs)
sourced from the internet, but also the data from countries of each
language. Furthermore, the collected data includes multiple fine-grained
scenarios (Fig. <a href="#fig:data_distribution" data-reference-type="ref"
data-reference="fig:data_distribution">[fig:data_distribution]</a>),
such as menus, logos, maps, bills, PPTs, research papers, and *etc*. As
a result, we gather a total of 1,220 images from document scenarios and
876 images from natural scenarios in the test set of the MTVQA
benchmark. To ensure the visual-textual alignment, for text-rich images
lacking text and language annotations, we subject them to a standardized
data cleaning process, which includes text recognition and language
classification. Afterward, we organize all the text-rich images we have
obtained into language-specific groups, preparing them for the
subsequent stage of data annotation.

## Human Expert Annotation

In order to obtain informative and accurate text-related QA pairs on the
language-specific grouped images, we recruit a group of annotators with
expertise from local regions of each language. It is worth noting that
all these annotators are native speakers of their respective languages,
ensuring their deep understanding and proficiency in the linguistic
nuances and cultural context necessary for precise annotations.
Considering the subjective nature of the text-image understanding task,
we have implemented a further division within the annotation team. This
division involves separating the team into two independent groups, with
one group dedicated to generating and responding to questions based on
the provided images, while the other group focuses on evaluating and
correcting the QA pair results. This raise-then-correct paradigm ensures
a comprehensive and reliable assessment of the text-image understanding
process. Additionally, each language’s annotation results undergo a 10%
sampling inspection by a quality inspector. If the QA pairs fail to meet
the criteria, they are sent back for re-annotation. Prior to commencing
the formal human expert annotation task, all annotators undergo unified
training and receive annotation examples. The brief diagram of the
two-round annotation process is shown in Figure
<a href="#fig:anno_process" data-reference-type="ref"
data-reference="fig:anno_process">[fig:anno_process]</a> and we
elaborate on it in the following subsections.

<div id="tab:mean_lengths" markdown="1">

|                | **AR** | **DE** | **FR** | **IT** | **JA** | **KO** | **RU** | **TH** | **VI** |
|:---------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| *Training Set* |        |        |        |        |        |        |        |        |        |
| Question       |  8.29  |  8.72  |  9.73  | 12.05  | 12.43  | 11.74  | 11.56  | 11.35  | 11.21  |
| Answer         |  9.66  |  6.96  |  7.34  | 11.24  | 12.70  | 13.56  | 12.00  | 11.26  | 13.31  |
| *Test Set*     |        |        |        |        |        |        |        |        |        |
| Question       |  8.08  |  8.29  |  9.76  | 11.93  | 12.48  |  12.2  | 11.65  | 10.98  | 10.99  |
| Answer         |  7.95  |  6.67  |  6.61  | 11.04  | 12.55  | 13.61  | 14.42  | 12.08  | 13.00  |

Mean lengths of question-answer pairs in different languages of training
set and test set, using GPT-4o tokenizer.

</div>

<span id="tab:mean_lengths" label="tab:mean_lengths"></span>

**First Round Questioning and Answering.**For the first round of
annotation tasks, we assigned 3 annotators for each language to manually
generate original QA results. Given a text-centric image from our
collection, annotators are first required to read the texts in the image
and analyze other contents in the image in a comprehensive and detailed
manner. They must then raise 4 meaningful and distinct questions based
on the content in the image and give the answers. All annotators adhere
to the following criteria: (1) the first three questions should satisfy
that answering these questions requires direct reading of the textual
information in the image, (2) the fourth question requires reasoning
about the text in the image to answer (3) the questions and answers must
be reasonably correct and consistent with the content of the image, and
(4) the answer should be as concise as possible and free of
nonsense (*e.g.*, when the question is “When is the volunteer
recruitment period”, the answer should be “9:00-16:00” rather than “The
volunteer recruitment period is 9:00-16:00”). It’s worth mentioning that
our requirement for concise answers is to make the evaluation process
more friendly and more reliable, cause we try to keep the evaluation
metrics unaffected by extraneous content in the answer sentence.

**Second round Evaluation and Correction.**To reduce the effect of human
subjective cognitive bias on our MTVQA benchmark and get high-quality
question-answer pairs, we assigned 2 annotators for each language for
the annotation evaluation and correction process. Based on the provided
images and the first-round annotation results, the annotators must
follow these rules of judgment and steps for the annotation: (1) Whether
the question is related to the text in the image. If not, discard the
current question-answer pair, (2) Whether the answer is correct. If not,
modify the answer, and (3) Whether the answer repeats the content from
the question. If so, remove the repeated content to ensure a concise
answer.

<div class="figure*" markdown="1">

<embed src="/papers/doc_ai_databases/arXiv-2405.11985v1_md/figure/annotation/annotation_process.png" />

</div>

## Data Statistics

We instruct the annotators to complete the above human expert annotation
work towards the text-centric VQA tasks and construct the MTVQA
benchmark consisting of 8,794 images and 28,607 question-answer pairs
that cover the 9 languages. The MTVQA benchmark is divided into a
training set containing 6,678 images and 21,829 question-answer pairs,
and a test set containing 2,116 images and 6,778 question-answer pairs.
The detailed data distribution can be seen in Figure
<a href="#fig:data_distribution" data-reference-type="ref"
data-reference="fig:data_distribution">[fig:data_distribution]</a>. To
visualize the vocabulary richness of our benchmark, we calculate the
word frequencies for each language and present them in the form of word
clouds as shown in Figure
<a href="#fig:word_cloud" data-reference-type="ref"
data-reference="fig:word_cloud">[fig:word_cloud]</a>. In Figure
<a href="#fig:leng_statistics" data-reference-type="ref"
data-reference="fig:leng_statistics">[fig:leng_statistics]</a> we
demonstrate the statistics of the question and answer lengths using
GPT-4o tokenizer.

<div class="figure*" markdown="1">

</div>

<div class="figure*" markdown="1">

</div>

# Experiments

## Baseline Models

For the MTVQA benchmark, we evaluate the following instruction-tuned
general MLLMs, (1) **Open-source MLLMs:**
InternVL-V1.5 [chen2023internvl](http://arxiv.org/pdf/2312.14238v3),
InternLM-Xcomposer2-4KHD [dong2024internlm](http://arxiv.org/pdf/2404.06512v1),
Mini-Gemini-HD-34B [li2024mini](http://arxiv.org/pdf/2305.16318v2),
Llava-Next-34B [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/),
DeepSeek-VL [lu2024deepseek](http://arxiv.org/pdf/2402.17510v1),
YI-VL-34B [young2024yi](http://arxiv.org/pdf/2304.11090v4),
TextSquare [tang2024textsquare](http://arxiv.org/pdf/2307.04087v3),
TextMonkey [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1) and mPLUG-DocOwl
1.5 [hu2024mplug](None); (2) **Closed-source MLLMs:** GPT-4V,
Gemini Ultra, QwenVL Max, QwenVL Plus, Claude3 Opus, Claude3 Sonnet and
GLM4V. For the closed-source MLLMs, we use the chat version through the
official APIs, while for the open-source MLLMs, we utilize the instruct
versions. It is noted that all the model weights of the open-source
MLLMs evaluated in our experiments could be downloaded from the
HuggingFace Model Hub. For the open-source MLLMs, the model size varies
from 7b to 34b.

## Implementation Details

We conduct the evaluation experiments over the baseline MLLMs with their
default settings, ignoring the effect of generation configuration on the
results. To make the output of MLLMs more evaluation-friendly, we design
the following prompt format to limit the output length: “Answer the
question using a word or phrase in the language of the question. +
\<Question\>”, where \<Question\> represents the corresponding question
of the input image. The extra prefixes added to the raw question could
limit the answer to be as concise as possible. Besides, we utilize the
InternLM-Xcomposer2-4KHD [dong2024internlm](http://arxiv.org/pdf/2404.06512v1) as the
backbone for the instructional fine-tuning experiment on the MTVQA
training set. In the instructional fine-tuning process, we follow the
default training settings [dong2024internlm](http://arxiv.org/pdf/2404.06512v1) with “HD-16”
and train on MTVQA training set for 1 epoch.

## Evaluation Results

**Zero-shot testing** To demonstrate the quantitative comparison results
in the above MLLMs, we follow
TextMonkey [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1) with accuracy as the
evaluation metric. That is, the model output is only counted as correct
if it contains the ground truth. The complete evaluation results are
shown in Table <a href="#tab:eval_results" data-reference-type="ref"
data-reference="tab:eval_results">2</a>, where Claude3 Opus achieves the
highest average accuracy of 25.7$\%$ on the 9 languages. It indicates
that the multilingual text-centric VQA tasks remain a big challenge,
even for the state-of-the-art open-source and closed-source MLLMs. From
the metrics across languages, both open-source and closed-source models
performed significantly better on Indo-European languages using the
Latin alphabet, including DE, FR, and IT in our benchmark, compared to
other languages, which results from the distribution of realistically
available training data and the genetic relationship of different
languages. In addition, all closed-source models except GLM4V outperform
the open-source model overall across the nine languages, which may be
due to the contribution of pre-training on multilingual data. We also
found that the document-focused MLLMs, like
TextSquare [tang2024textsquare](http://arxiv.org/pdf/2307.04087v3) and
TextMonkey [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1), do not significantly
outperform other open-source models on the metrics of these 9 languages.

**Instruction tuning** As shown in Table
<a href="#tab:eval_results" data-reference-type="ref"
data-reference="tab:eval_results">2</a>, the instruction tuning
experiment on MTVQA benchmark brings a 8.5$\%$ improvement in average
accuracy. With respect to specific languages, French sees the largest
improvement of 14.2$\%$ in accuracy, while Russian has the smallest
improvement of 1.7$\%$ in accuracy. The results demonstrate that MLLMs
vary in their ability to understand and learn from text-centric data in
different languages, leaving great potential for future research of
multilingual text-centric MLLMs pre-training.

<div id="tab:eval_results" markdown="1">

|   | AR | DE | FR | IT | JA | KO | RU | TH | VI | Avg. |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *Closed-Source MLLMs* |  |  |  |  |  |  |  |  |  |  |
| GPT-4V | 11.5 | 31.5 | 40.4 | 32.3 | 11.5 | 16.7 | 10.3 | 15.0 | 28.9 | 22.0 |
| Gemini Ultra | 14.7 | 32.3 | 40.0 | 31.8 | 12.3 | 17.2 | 11.8 | 20.3 | 28.6 | 23.2 |
| QwenVL Max | 7.7 | 31.4 | 37.6 | 30.2 | 18.6 | 25.4 | 10.4 | 4.8 | 23.5 | 21.1 |
| QwenVL Plus | 4.8 | 28.8 | 33.7 | 27.1 | 12.8 | 19.9 | 9.4 | 5.6 | 18.1 | 17.8 |
| Claude3 Opus | **15.1** | **33.4** | **40.6** | **34.4** | **19.4** | **27.2** | **13.0** | **19.5** | **29.1** | **25.7** |
| Claude3 Sonnet | 10.5 | 28.9 | 35.6 | 31.8 | 13.9 | 22.2 | 11.0 | 15.2 | 20.8 | 21.1 |
| GLM4V | 0.3 | 30.0 | 34.1 | 30.1 | 3.4 | 5.7 | 3.0 | 3.5 | 12.3 | 13.6 |
| *Open-Source MLLMs* |  |  |  |  |  |  |  |  |  |  |
| InternVL-V1.5 [chen2023internvl](http://arxiv.org/pdf/2312.14238v3) | 3.4 | 27.1 | 31.4 | 27.1 | 9.9 | 9.0 | 4.9 | 8.7 | 12.4 | 14.9 |
| Mini-Gemini-HD-34B [li2024mini](http://arxiv.org/pdf/2305.16318v2) | 2.2 | 25.0 | 29.2 | 25.5 | 6.1 | 8.6 | 4.1 | 4.3 | 11.8 | 13.0 |
| Llava-Next-34B [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | 3.3 | 24.0 | 28.0 | 22.3 | 3.6 | 6.1 | 2.6 | 0.4 | 9.8 | 11.1 |
| DeepSeek-VL [lu2024deepseek](http://arxiv.org/pdf/2402.17510v1) | 0.6 | 14.2 | 15.3 | 15.2 | 2.9 | 3.8 | 1.6 | 0.9 | 5.2 | 6.6 |
| YI-VL-34B [young2024yi](http://arxiv.org/pdf/2304.11090v4) | 1.7 | 13.5 | 15.7 | 12.1 | 4.8 | 5.2 | 0.8 | 3.5 | 4.1 | 6.8 |
| TextSquare [tang2024textsquare](http://arxiv.org/pdf/2307.04087v3) | 3.7 | 27.0 | 30.8 | 26.7 | 3.2 | 7.2 | 6.7 | 5.2 | 12.4 | 13.6 |
| TextMonkey [liu2024textmonkey](http://arxiv.org/pdf/2403.14252v1) | 2.0 | 18.1 | 19.9 | 22.1 | 4.6 | 7.2 | 3.2 | 0.9 | 11.1 | 9.9 |
| mPLUG-DocOwl 1.5 [hu2024mplug](None) | 1.0 | 13.9 | 14.9 | 18.2 | 2.9 | 5.0 | 2.0 | 0.9 | 6.4 | 7.2 |
| Xcomposer2-4KHD [dong2024internlm](http://arxiv.org/pdf/2404.06512v1) | 2.0 | 20.6 | 23.2 | 21.6 | 5.6 | 7.7 | 4.1 | 6.1 | 10.1 | 11.2 |
| Xcomposer-SFT | 11.8 | 31.7 | 37.4 | 29.3 | 14.5 | 12.9 | 5.8 | 13.9 | 20.2 | 19.7 |

Performance of the leading closed-source and open-source MLLMs on the
MTVQA benchmark.

</div>

<span id="tab:eval_results" label="tab:eval_results"></span>

# Limitation

The current iteration of MTVQA exhibits certain constraints that warrant
attention. Primarily, the linguistic diversity incorporated is not
exhaustive; several lesser-spoken languages remain unrepresented. Future
enhancements will aim to broaden the multilingual scope of the dataset.
Additionally, the dataset currently offers a singular canonical response
for each question. Recognizing the multifaceted nature of the inquiry,
subsequent versions will endeavor to include a spectrum of plausible
answers to reflect the varied perspectives inherent to each question.

# Conclusion

In this paper, we introduce MTVQA, a multilingual TEC-VQA benchmark
featuring high-quality human expert annotations in 9 diverse languages.
We believe that MTVQA is the first benchmark of its kind to provide
fully manual annotations specifically tailored to text-centric
scenarios. The results obtained from both closed- and open-source MLLMs
on our MTVQA dataset indicate that there is still room for improving
their performance in multilingual text-centric scenarios. Although the
current version of MTVQA has constraints regarding linguistic diversity
and singular responses per question, we are confident that this dataset
can still inspire researchers within the TEC-VQA community with new
perspectives and ideas.