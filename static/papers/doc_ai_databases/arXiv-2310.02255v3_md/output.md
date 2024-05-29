# Introduction [sec:introduction]

This complexity is observed not only in textual scenarios but also
significantly in visual contexts. For instance, when assessing a child’s
mathematical and reasoning capabilities, problems are often designed to
encompass visual contexts in addition to arithmetic calculations
[stipek1989developmental](http://arxiv.org/pdf/2002.02184v2), [pollitt2020assessing](http://arxiv.org/pdf/1506.02794v1). At the
same time, AI agents with strong mathematical reasoning capabilities in
visual contexts have a wide range of real-world applications, such as
solving complex problems in educational
disciplines [seo2015solving](http://arxiv.org/pdf/2308.02823v1), [wang2017deep](http://arxiv.org/pdf/2302.03145v1), helping
analysts with logical queries about statistical
data [wu2023bloomberggpt](None), [yang2023fingpt](None), and assisting
in theorem proving and scientific discovery in advanced research
fields [taylor2022galactica](http://arxiv.org/pdf/2405.11265v1), [dong2023large](None), [trinh2024solving](http://arxiv.org/pdf/2404.06405v2).

<figure id="fig:tease_scores">
<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/tease_scores_gpt4v_new.png" style="width:100.0%" />
<figcaption>Accuracies of on our proposed across mathematical reasoning
and visual context types. PoT GPT-4 is a textual, program-aided LLM
augmented with the Bard caption and OCR text. </figcaption>
</figure>

Numerous datasets have been curated to assess the mathematical reasoning
abilities of AI systems, with most presented purely in text form. Some
datasets such as ChartQA
[lu2021inter](None), [dahlgren2022clevr](None), [masry2022chartqa](None) have
explored mathematical reasoning in vision-language settings. However,
these datasets tend to either focus on specific tasks, like math word
problems, or particular visual contexts, such as geometry problems or
bar charts. General-purpose visual question answering (VQA) datasets on
natural scenes contain only a small portion of questions necessitating
mathematical reasoning, leaving a comprehensive investigation of
vision-language reasoning within a mathematical framework largely
unexplored.

Large Language Models (LLMs)
[openai2022chatgpt](https://openai.com/blog/chatgpt), [openai2023gpt4](None) and Large Multimodal
Models (LMMs)
[google2023bard](https://bard.google.com/), [openai2023gpt4v](https://openai.com/research/gpt-4v-system-card), [team2023gemini](http://arxiv.org/pdf/2405.12107v1) have
exhibited impressive problem-solving skills in many tasks and domains.
Recently, some studies have aimed to augment existing LLMs with
mathematical and scientific reasoning capabilities using external
tools [lu2023chameleon](http://arxiv.org/pdf/2305.00061v1), [wang2023scibench](None). However, the
ability of these foundation models to perform mathematical reasoning in
visual contexts has not been systematically examined.

In this paper, we present , a consolidated <u>**Math**</u>ematical
reasoning benchmark in <u>**Vis**</u>ual contexts. We propose a task
taxonomy to guide the development of : (1) we identify seven
mathematical reasoning types: *algebraic reasoning*, *arithmetic
reasoning*, *geometry reasoning*, *logical reasoning*, *numeric common
sense*, *scientific reasoning*, and *statistical reasoning*; (2) we
focus on five primary tasks: *figure question answering* (FQA),
*geometry problem solving* (GPS), *math word problem* (MWP), *textbook
question answering* (TQA), and *visual question answering* (VQA); and
(3) we encompass a diverse array of visual contexts, including natural
images, geometry diagrams, abstract scenes, synthetic scenes, as well as
various figures, charts, and plots. incorporates 28 existing multimodal
datasets, including 9 math-targeted question answering (MathQA) datasets
and 19 VQA datasets. In addition, we have created three new datasets
(*i.e.*, IQTest, FunctionQA, PaperQA) which are tailored to evaluating
logical reasoning on puzzle test figures, algebraic reasoning over
functional plots, and scientific reasoning with academic paper figures,
respectively. Overall, consists of 6,141 examples, with 736 of them
being newly curated
(Table <a href="#tab:statistics" data-reference-type="ref"
data-reference="tab:statistics">[tab:statistics]</a>). To facilitate
fine-grained evaluation, examples are annotated with metadata, including
question type, answer type, task category, grade level, visual context,
and required reasoning skills. Detailed descriptions of data collection
can be found in §<a href="#sec:dataset" data-reference-type="ref"
data-reference="sec:dataset">[sec:dataset]</a>,
§<a href="#sec:collection_guideline" data-reference-type="ref"
data-reference="sec:collection_guideline">[sec:collection_guideline]</a>,
and §<a href="#app:collection_details" data-reference-type="ref"
data-reference="app:collection_details">[app:collection_details]</a>.

We conduct extensive experiments on to evaluate the reasoning abilities
of 12 foundation models known for their leading performance in
mathematical and multimodal reasoning. This ensemble includes three LLMs
(*i.e*, ChatGPT, GPT-4, Claude-2), two proprietary LMMs (*i.e.*, GPT-4V,
Bard), and seven open-source LMMs. For LLMs, we examine zero-shot and
few-shot settings using two prompting strategies: chain-of-thought (CoT)
[wei2022chain](http://arxiv.org/pdf/2201.11903v6) and program-of-thought (PoT)
[chen2022program](http://arxiv.org/pdf/2211.12588v4). These LLMs can also be augmented with
off-the-shelf visual models for image captioning and OCR. We establish a
human performance baseline by engaging qualified human annotators with a
high school diploma or higher. We show that , featuring advanced topics
such as college curricula and scientific reasoning, is a very
challenging benchmark, with human performance reaching only 60.3%
accuracy.

Our results indicate that CoT GPT-4, the best-performing LLM without
visual tool augmentations, achieves an overall accuracy of 29.2%.
Multimodal Bard, the best-performing LMM, achieves 34.8%
(§<a href="#sec:results" data-reference-type="ref"
data-reference="sec:results">[sec:results]</a>), which attains only 58%
of human performance (34.8% vs 60.3%). When augmented with Bard captions
and OCR text, PoT GPT-4 obtains 33.9%, closely matching Multimodal Bard
(§<a href="#sec:fine_grained_results" data-reference-type="ref"
data-reference="sec:fine_grained_results">[sec:fine_grained_results]</a>).
Further analysis indicates that the Multimodal Bard model failures arise
from incorrect calculations and hallucinations caused by visual
perception and textual reasoning
(§<a href="#sec:qualitative_analysis" data-reference-type="ref"
data-reference="sec:qualitative_analysis">[sec:qualitative_analysis]</a>).

<figure id="fig:our_new_3_datasets">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/our_new_3_datasets.png" style="width:100.0%" />
<figcaption>Examples of our newly annotated datasets: IQTest,
FunctionQA, and PaperQA.</figcaption>
</figure>

# The Dataset [sec:dataset]

## Collection Guidelines

As discussed previously, there is a notable gap in existing benchmarks,
which primarily evaluate mathematical reasoning in textual contexts,
overlooking the intrinsic visual nature of many mathematical problems.
Our dataset, , is therefore motivated to bridge this gap, offering a
robust evaluation benchmark for mathematical reasoning intertwined with
visual understanding, thus pushing AI assistants towards general-purpose
capabilities. Our benchmark adheres to the following collection
guidelines: (1) it covers multiple tasks and topics to mirror real-world
applications; (2) it incorporates diverse visual contexts and
mathematical skills to foster a well-rounded evaluation; (3) it offers
varying levels of challenge to effectively probe and uncover the
potential limitations of current models; and (4) it provides robust
evaluation settings for deterministic evaluations.

The taxonomy for this work is introduced as follows: We identify seven
types of mathematical reasoning: *algebraic reasoning*, *arithmetic
reasoning*, *geometry reasoning*, *logical reasoning*, *numeric common
sense*, *scientific reasoning*, and *statistical reasoning*, with
detailed definitions provided in
§<a href="#sec:math_reasoning" data-reference-type="ref"
data-reference="sec:math_reasoning">[sec:math_reasoning]</a> and
examples shown in
§<a href="#app:math_examples" data-reference-type="ref"
data-reference="app:math_examples">[app:math_examples]</a>. We focus on
five primary tasks: *figure question answering* (FQA), which centers
around statistical reasoning over multiple charts and plots; *geometry
problem solving* (GPS), which deals with geometrical topics; *math word
problem* (MWP), which involves arithmetic reasoning in everyday
scenarios; *textbook question answering* (TQA), which usually entails
knowledge-intensive reasoning on scientific topics and figures; and
*visual question answering* (VQA). Furthermore, our objective is to
account for a diverse array of visual contexts, including natural
images, geometry diagrams, abstract scenes, synthetic scenes, multiple
charts and plots, scientific figures, tables, function plots, puzzle
test figures, and more, with examples shown in
§<a href="#app:visual_context" data-reference-type="ref"
data-reference="app:visual_context">[app:visual_context]</a>.

## Data Collection [sec:data_collection]

#### Collection of MathQA datasets.

We collected nine MathQA datasets in multimodal settings, including four
for GPS, two for MWP with visual contexts of synthetic scenes, abstract
diagrams, and tables, and two for TQA on college curricula (see
§<a href="#sec:source_data" data-reference-type="ref"
data-reference="sec:source_data">[sec:source_data]</a>). Annotations
such as solutions, programs, parsing results, and grounded theorems are
also collected, providing demonstration examples for LLMs. Each source
dataset is limited to up to 400 examples to ensure a balanced
representation of each source in our final compiled benchmark. In total,
we collected 2,666 examples.

#### Review and collection of VQA datasets.

Many existing VQA datasets feature instances requiring mathematical
reasoning abilities, such as arithmetic operations or numeric common
sense. Incorporating these datasets enhances problem diversity in terms
of tasks, domains, visual contexts, and reasoning skills involved. We
reviewed more than 70 datasets, collecting 19 of them that contain
math-related instances and are publicly available, as listed in
§<a href="#sec:source_data" data-reference-type="ref"
data-reference="sec:source_data">[sec:source_data]</a>. Since these
datasets are not originally math-targeted, we initially designed
heuristic rules to automatically select examples likely to involve
mathematical reasoning from a large pool of candidates. Examples with
numeric answers or those containing quantity words (as listed in
§<a href="#sec:automatic_selection" data-reference-type="ref"
data-reference="sec:automatic_selection">[sec:automatic_selection]</a>)
in the questions were selected. This automatic filtration yielded 4,949
VQA-format examples, though some false positive examples remained.
Therefore, we engaged three expert annotators to manually label these
examples to determine if they involve mathematical reasoning (more
details in § <a href="#sec:human_is_math" data-reference-type="ref"
data-reference="sec:human_is_math">[sec:human_is_math]</a>). Utilizing
majority voting and limiting each source dataset to 400 examples, we
finalized a collection of 2,739 examples.

#### Collection of three new datasets.

While the source datasets we collected encompass multiple visual
contexts and mathematical reasoning abilities, certain scenarios remain
unaddressed: logical reasoning on puzzle test diagrams, statistical
reasoning on functional plots, and scientific reasoning on academic
figures. To address these gaps, we introduced three new datasets:
IQTest, FunctionQA, and PaperQA, with examples illustrated in
Figure <a href="#fig:our_new_3_datasets" data-reference-type="ref"
data-reference="fig:our_new_3_datasets">[fig:our_new_3_datasets]</a>.
IQTest comprises 228 examples requiring inductive reasoning, abstract
thinking, pattern prediction, and calculations, sourced from puzzle test
figures on online learning platforms. FunctionQA, with 400 examples,
emphasizes subtle visual perceptions of functional plots and algebraic
reasoning concerning variables, expressions, equations, and functions.
PaperQA is a novel dataset featuring questions derived from informative
academic illustrations, including tables, figures, and charts from
online education resources, with 107 examples sourced from papers
released in August 2023 on Huggingface[^1].

To ensure data quality, all questions were manually annotated by
graduate students in STEM fields and further refined through a rigorous
review process. The GUI of the annotation tool is shown in
Figure <a href="#fig:gui_new_data_annotation" data-reference-type="ref"
data-reference="fig:gui_new_data_annotation">[fig:gui_new_data_annotation]</a>
in §<a href="#sec:annotate_new_data" data-reference-type="ref"
data-reference="sec:annotate_new_data">[sec:annotate_new_data]</a>.

## Metadata Annotation

Fine-grained metadata facilitates a comprehensive analysis of models’
reasoning capabilities across various aspects. To this end, we annotate
the examples in with information including question type, answer type,
language, source, category, task, grade level, and visual context, which
can be accurately obtained from the details provided in the source
datasets. features seven different types of mathematical reasoning
abilities, as categorized in
Table <a href="#tab:math_definition" data-reference-type="ref"
data-reference="tab:math_definition">[tab:math_definition]</a>
(§<a href="#sec:math_reasoning" data-reference-type="ref"
data-reference="sec:math_reasoning">[sec:math_reasoning]</a>). Coarse
labels of mathematical reasoning can be automatically obtained from the
details of the source datasets. To verify the quality of automatic
annotation, expert annotators manually label the mathematical reasoning
categories from seven candidates for 1,000 examples, using the
annotation tool illustrated in
§<a href="#sec:human_math_reasoning" data-reference-type="ref"
data-reference="sec:human_math_reasoning">[sec:human_math_reasoning]</a>.
The results show that 94.1% of the examples from automatic and human
annotations have the exact same set of reasoning types, while 98.79% of
the individual labels are identical, indicating that the automatic
annotation for the labeling of mathematical reasoning is highly
accurate.

## Data Preparation and Release

consists of 6,141 examples, divided into two subsets: *testmini* and
*test*. *testmini* contains 1,000 examples, intended for model
development validation or for those with limited computing resources.
The *test* set features the remaining 5,141 examples for standard
evaluation. Notably, the answer labels for *test* will not be publicly
released to prevent data contamination, and we will maintain an online
evaluation platform. To ensure that each source dataset is well
represented in *testmini* and to maintain a distribution in *testmini*
closely resembling the whole set, we adopted this sampling strategy: (1)
first, randomly sample questions with a threshold number of 4 for each
source dataset; (2) then, randomly sample the remaining questions for
each source dataset on its proportion in the entire set. The KL
Divergence and Total Variation (TV) distance between the *testmini* set
and the entire set are 0.008 and 0.035, respectively, suggesting that
*testmini* is close to the distribution of the whole set. We also
conducted several quality checks to address any unidentified errors.

## Data Analysis

<figure id="fig:source_dataset">
<table>
<thead>
<tr class="header">
<th style="text-align: left;"><strong>Statistic</strong></th>
<th style="text-align: center;"><strong>Number</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">Total questions</td>
<td style="text-align: center;">6,141</td>
</tr>
<tr class="even">
<td style="text-align: left;"> - multiple-choice questions</td>
<td style="text-align: center;">3,392 (55.2%)</td>
</tr>
<tr class="odd">
<td style="text-align: left;"> - Free-form questions</td>
<td style="text-align: center;">2,749 (44.8%)</td>
</tr>
<tr class="even">
<td style="text-align: left;"> - Questions with annotations</td>
<td style="text-align: center;">5,261 (85.6%)</td>
</tr>
<tr class="odd">
<td style="text-align: left;"> - Questions newly annotated</td>
<td style="text-align: center;">736 (12.0%)</td>
</tr>
<tr class="even">
<td style="text-align: left;">Unique number of images</td>
<td style="text-align: center;">5,487</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Unique number of questions</td>
<td style="text-align: center;">4,746</td>
</tr>
<tr class="even">
<td style="text-align: left;">Unique number of answers</td>
<td style="text-align: center;">1,464</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Source datasets</td>
<td style="text-align: center;">31</td>
</tr>
<tr class="even">
<td style="text-align: left;"> - Existing VQA datasets</td>
<td style="text-align: center;">19</td>
</tr>
<tr class="odd">
<td style="text-align: left;"> - Existing MathQA datasets</td>
<td style="text-align: center;">9</td>
</tr>
<tr class="even">
<td style="text-align: left;"> - Our newly annotated datasets</td>
<td style="text-align: center;">3</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Visual context (image) classes</td>
<td style="text-align: center;">19</td>
</tr>
<tr class="even">
<td style="text-align: left;">Maximum question length</td>
<td style="text-align: center;">213</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Maximum answer length</td>
<td style="text-align: center;">27</td>
</tr>
<tr class="even">
<td style="text-align: left;">Maximum choice number</td>
<td style="text-align: center;">8</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Average question length</td>
<td style="text-align: center;">15.6</td>
</tr>
<tr class="even">
<td style="text-align: left;">Average answer length</td>
<td style="text-align: center;">1.2</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Average choice number</td>
<td style="text-align: center;">3.4</td>
</tr>
</tbody>
</table>
<p><span id="tab:statistics" label="tab:statistics"></span></p>
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/task_source_sunburst.png" style="width:80.0%" />
<figcaption>Source dataset distribution of . FQA: figure question
answering, GPS: geometry problem solving, MWP: math word problem, TQA:
textbook question answering, VQA: visual question
answering.</figcaption>
</figure>

The main statistics of are presented in Table
<a href="#tab:statistics" data-reference-type="ref"
data-reference="tab:statistics">[tab:statistics]</a>. There are two
types of questions: multiple-choice and free-form. Answers to free-form
questions are categorized as integers, floating numbers, or lists. The
large unique number of images, questions, and answers ensures pattern
diversity in . is derived from 31 source datasets, including three newly
annotated datasets to address the missing types of mathematical
reasoning over specific visual contexts. Dataset examples in Table
<a href="#tab:math_examples" data-reference-type="ref"
data-reference="tab:math_examples">[tab:math_examples]</a>
(§<a href="#app:math_examples" data-reference-type="ref"
data-reference="app:math_examples">[app:math_examples]</a> ) highlight
the richness of mathematical reasoning involved. Examples in
§<a href="#app:visual_context" data-reference-type="ref"
data-reference="app:visual_context">[app:visual_context]</a> demonstrate
the diverse visual contexts present in . Further details on data
analysis are available in
§<a href="#app:data_analysis" data-reference-type="ref"
data-reference="app:data_analysis">[app:data_analysis]</a>.

[^1]: <https://huggingface.co/papers>

# Experiments

## Evaluation Protocols [sec:evaluation_protocol]

Recent LLMs and LMMs have been instructed to generate long responses in
conventional settings instead of short text. Therefore, we propose a new
strategy for benchmarking , unlike using human-designed or template
matching rules [lu2022learn](http://arxiv.org/pdf/2209.09513v2). The evaluation process
consists of three stages: *response generation*, *answer extraction*,
and *score calculation*. Initially, the baselines generate responses
given the input query, which incorporates the task description, the
question, the choices, and the metadata, using the template defined in
Table <a href="#tab:promt_response_generation" data-reference-type="ref"
data-reference="tab:promt_response_generation">[tab:promt_response_generation]</a>
(§<a href="#sec:promt_response_generation" data-reference-type="ref"
data-reference="sec:promt_response_generation">[sec:promt_response_generation]</a>).
Next, the short answer text is extracted from the detailed response. We
propose an answer extractor
(§<a href="#sec:promt_answer_extraction" data-reference-type="ref"
data-reference="sec:promt_answer_extraction">[sec:promt_answer_extraction]</a>)
based on LLMs such as GPT-4, inspired by its remarkable ability for text
processing [wei2022chain](http://arxiv.org/pdf/2201.11903v6). A preliminary study of 200
examples shows that GPT-4 can extract the answer text with more than
99.5% accuracy. Finally, the extracted answer is normalized to a
required answer format (e.g., an option letter or an integer), and the
target metric scores are computed. Taking advantage of the fact that the
instances in are either multiple-choice questions for textual answers or
free-form questions for numerical answers, accuracy scores are used as
metrics for deterministic evaluation.

## Experimental Setup [sec:experimental_setup]

We evaluate the models on under three setups: (a) *Text-Only LLMs*
including ChatGPT [openai2022chatgpt](https://openai.com/blog/chatgpt), GPT-4
[openai2023gpt4](None), and Claude-2 [claude2](https://www.anthropic.com/index/claude-2)
in zero-shot and two-shot settings with Chain-of-Thought (CoT)
[wei2022chain](http://arxiv.org/pdf/2201.11903v6) and Program-of-Thought (PoT)
[chen2022program](http://arxiv.org/pdf/2211.12588v4), (b) *Augmented-LLMs* where the LLMs
are provided with additional visual information including the generated
image captions from Multimodal Bard [google2023bard](https://bard.google.com/) and
the detected OCR text from EasyOCR
[jaidedai2020easyocr](https://github.com/JaidedAI/EasyOCR), (c) *LMMs* that include
open-source models such as IDEFICS-9B
[laurencon2023obelics](https://arxiv.org/pdf/2306.16527), mPLUG-OWL-LLaMA-7B
[ye2023mplug](None), miniGPT-4-LLaMA-2-7B
[zhu2023minigpt](None), LLaMA-Adapter-V2-7B
[gao2023llamaadapterv2](None), InstructBLIP-Vicuna-7B
[instructblip](https://arxiv.org/pdf/2305.06500), LLaVA-LLaMA-2-13B
[liu2023llava](http://arxiv.org/pdf/2402.11690v1), LLaVAR [zhang2023llavar](None),
and . We provide the prompts for LLMs and the hyperparameters used for
LMMs in §<a href="#app:setup" data-reference-type="ref"
data-reference="app:setup">[app:setup]</a>.

<div class="table*" markdown="1">

<span id="tab:mathvista" label="tab:mathvista"></span>

</div>

## Experimental Results [sec:results]

We compare the performance of several models, including Text-only LLMs,
Augmented LLMs, and LMMs on in Table
<a href="#tab:mathvista" data-reference-type="ref"
data-reference="tab:mathvista">[tab:mathvista]</a>. We include random
chance (*i.e.*, one of the options in multiple-choice questions, and
empty in the free-form questions) and frequency guess
(§<a href="#sec:frequent_guess" data-reference-type="ref"
data-reference="sec:frequent_guess">[sec:frequent_guess]</a>) as naive
baselines. Additionally, we established a human performance baseline
using Amazon Mechanical Turk. Eligible human annotators must have a
satisfactory annotating history, successfully pass qualification
examples, and possess a high school degree or higher. We asked each
annotator to complete five questions within 20 minutes. Further details
can be found in
§<a href="#sec:human_performance" data-reference-type="ref"
data-reference="sec:human_performance">[sec:human_performance]</a>.

Among text-only LLMs, all models outperform the random baselines, with
the 2-shot GPT-4 using chain-of-thought (CoT) prompting achieving 29.2%.
The limited performance of text-only LLMs suggests that our dataset
requires models to reason within visual contexts for optimal results.
When equipped with image captions and detected OCR text, augmented LLMs
exhibit superior performance compared to their text-only counterparts on
. Specifically, the best-performing augmented LLM is the 2-shot GPT-4
employing program-of-thought (PoT) prompting, which scores 33.9%.

The open-source models (IDEFICS to LLaVA) achieve underwhelming
performance on . This can be attributed to their lack of math reasoning
capabilities, text recognition (useful for math word problems), shape
detection (useful for geometrical problems), and chart understanding.
Notably, these models utilize different model architectures for
processing the vision (e.g., OpenCLIP, CLIP, Vit-G) and language (e.g.,
LLaMA-1, LLaMA-2), different alignment strategies (e.g., MLP projection
in LLaVA, Q-former in InstructBLIP, visual abstractor in mPLUGOwl), and
instruction tuning data (e.g., 150K instruction-response pairs from
LLaVA data, 3,500 instruction-response pairs from miniGPT-4). While
fine-tuned with instruction-following data from text-rich images, LLaVAR
does not perform well, indicating that strong text recognition abilities
do not guarantee high performance on , which requires comprehensive
visual perception and mathematical reasoning. This underscores that
there are immense possibilities for innovations in model, data, or
training objectives to improve the zero-shot performance of LMMs on .

## Fine-grained Results [sec:fine_grained_results]

We also report fine-grained scores for a comprehensive study of the
capabilities of existing models across different tasks (Table
<a href="#tab:mathvista" data-reference-type="ref"
data-reference="tab:mathvista">[tab:mathvista]</a>), mathematical
reasoning abilities (Table
<a href="#tab:mathvista" data-reference-type="ref"
data-reference="tab:mathvista">[tab:mathvista]</a>, Figures
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>,
<a href="#fig:math_reasoning_bar_chart" data-reference-type="ref"
data-reference="fig:math_reasoning_bar_chart">[fig:math_reasoning_bar_chart]</a>),
visual context types (Figures
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>,
<a href="#fig:visual_context_bar_chart" data-reference-type="ref"
data-reference="fig:visual_context_bar_chart">[fig:visual_context_bar_chart]</a>),
and grade levels (Figure
<a href="#fig:grade_level_bar_chart" data-reference-type="ref"
data-reference="fig:grade_level_bar_chart">[fig:grade_level_bar_chart]</a>).

We perform an ablation study on the augmented LLMs and present the
results in
Table <a href="#fig:llm_ablation_study" data-reference-type="ref"
data-reference="fig:llm_ablation_study">[fig:llm_ablation_study]</a>
(see §<a href="#sec:llm_ablation_study" data-reference-type="ref"
data-reference="sec:llm_ablation_study">[sec:llm_ablation_study]</a>).
The gap in the performance of the Augmented LLMs can be attributed to
poor image captions, which may not adequately describe the math in
visual contexts, the inability of the OCR to detect shapes useful for
geometrical reasoning, and the lack of mathematical reasoning
capabilities.

## Qualitative Analysis [sec:qualitative_analysis]

<figure id="fig:hor_2figs_1cap">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/bard_answer_and_explanation.png"
style="height:38.0%" />
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/bard_wrong_explanation.png" style="height:45.0%" />
<figcaption>Error analysis of Bard results: (a) presents errors in
answers and explanations; (b) delves into the details of wrong
explanations. Notations: “Answer” is “Ans.”, “Explanation” is “Exp.”,
“Partially Correct” is “Partial”, and “Not applicable” refers to
unanswerable or indeterminate cases. </figcaption>
</figure>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/137.png" style="height:2cm" alt="image" />

**Question:** Is the function ($f:$ $R$ to $R$) injective?  
**Choices:**  
(A) Yes  
(B) No  

  
**Response from Bard:**  
The function $f(x) = x^2$ is not injective because it takes two
different inputs, $-1$ and $1$, to the same output, $1$. This means that
there is no one-to-one correspondence between the domain and the range
of the function.  
  
In the image, we can see that the graph of the function passes through
the points $(-1, 1)$ and $(1, 1)$. This means that $f(-1) = f(1) = 1$.
So, if we have two different inputs, $-1$ and $1$, they both map to the
same output, $1$. This violates the definition of an injective
function.  
  
Therefore, the answer is (B).  
  
**Correct output:**   

</div>

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/236.png" style="height:1.85cm" alt="image" />

**Question:** Find $z$.  
**Choices:**  
(A) 7      (B) 9  
(C) 12    (D) 15  

  
**Response from Bard:**  
The Law of Cosines states that for a triangle with side lengths $a$,
$b$, and $c$ and angle measure $\theta$ opposite side $c$, the following
equation holds: $c^2 = a^2 + b^2 - 2ab\cos(\theta)$  
  
In this case, we have $a = 7$, $b = 9$, and . Plugging these values into
the Law of Cosines, we get:  
  
Since , we can simplify the equation to get:  
  
  
Taking the square root of both sides, we get . Since $\sqrt{130}$ is not
among the answer choices, we round $z$ to the nearest integer, which is
$\boxed{12}$.  
  
**Correct output:**

</div>

<span id="fig:bard_examples" label="fig:bard_examples"></span>

</div>

#### Success and failure analysis of Multimodal Bard.

In §<a href="#sec:results" data-reference-type="ref"
data-reference="sec:results">1.3</a>, we observe that Multimodal Bard
achieves the highest average accuracy on . Here, we analyze its
predictions through human evaluation to understand its mode of success
and failure. To do so, we ask the human workers, from Amazon Mechanical
Turk (AMT), to study Bard’s predictions given the math question, its
associated image, and the ground truth from dataset for 250 instances.
Specifically, workers were instructed to decide whether the predictions
contained the correct answer with the correct explanation. If the
workers find that the model’s explanation is incorrect, they had to
choose whether the wrong explanation was due to various failure modes
such as incorrect reasoning with *hallucination* or wrong calculations.
In our setup, we define hallucination as an introduction of incorrect
facts, in the model explanation, that is not mentioned in the context of
the image or question (e.g., in Figure
<a href="#fig:visual_hallucination" data-reference-type="ref"
data-reference="fig:visual_hallucination">[fig:visual_hallucination]</a>
and Figure
<a href="#fig:textual_hallucination" data-reference-type="ref"
data-reference="fig:textual_hallucination">[fig:textual_hallucination]</a>).
More details can be found in
§<a href="#sec:human_study_bard" data-reference-type="ref"
data-reference="sec:human_study_bard">[sec:human_study_bard]</a>.

We present the distribution of the quality of Bard’s predictions, judged
by the human annotators, in Figure
<a href="#fig:hor_2figs_1cap" data-reference-type="ref"
data-reference="fig:hor_2figs_1cap">1</a> (a). We find that 44.6% of the
Bard’s predictions had incorrect answers with incorrect explanations.
Interestingly, we observe that Bard responds with partial (6.8%) or
completely (8.1%) incorrect explanations despite giving the correct
answer to the input image and question, highlighting its failure to
reach the correct answer for the wrong reasons. In Figure
<a href="#fig:hor_2figs_1cap" data-reference-type="ref"
data-reference="fig:hor_2figs_1cap">1</a> (b), we present the
distribution over possible reasons when Bard provides incorrect
explanations. Notably, we find that 49.6% of its responses contain
hallucinations. Our analysis highlights that hallucination is a major
source of errors in the generative foundation
models [lu2023dl4math](http://arxiv.org/pdf/2212.10535v2), [ji2023survey](http://arxiv.org/pdf/2311.05232v1). We also observe
that the model responds with correct reasoning but either hallucinates
(18.6%) or performs wrong calculations (19.5%) leaving an overall
impression of being a wrong explanation.

#### Qualitative examples of Multimodal Bard.

We also present a few qualitative examples of Bard’s predictions. In
Figure <a href="#fig:bard_examples" data-reference-type="ref"
data-reference="fig:bard_examples">[fig:bard_examples]</a> (a), we find
that Bard generates the correct answer with the correct explanation,
including detecting the correct function (*i.e.*, $f(x) = x^2$) and
analyzing its properties (*i.e.*, injective) to answer the question.
However, in Figure
<a href="#fig:bard_examples" data-reference-type="ref"
data-reference="fig:bard_examples">[fig:bard_examples]</a> (b), we
observe that the model provides the correct answer (*i.e.*, 12) but with
an incorrect explanation (*i.e.*, using the law of cosines when the
question requires an understanding of the properties of isosceles
triangles). We present more examples in
§<a href="#sec:more_examples_bard" data-reference-type="ref"
data-reference="sec:more_examples_bard">[sec:more_examples_bard]</a>.
Overall, our analysis of Bard highlights its modes of failure in detail,
which could guide future foundation model design to address these
issues.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/573.png" style="height:3cm" alt="image" />

**Question:** A race car driver kept track of how many laps he drove in
the past 5 days. What is the mode of the numbers?  
  
**Detected OCR text:**  
(\[55, 9\], Laps driven),  
(\[21, 34\], Day),  
(\[81, 37\], Number of laps),  
(\[1, 64\], Thursday),  (\[129, 67\], 53),  
(\[3, 99\], Friday),  (\[129, 99\], 53),  
(\[1, 126\], Saturday),  (\[129, 129\], 53),  
(\[3, 161\], Sunday),  (\[129, 161\], 55),  
(\[1, 188\], Monday),  (\[129, 191\], 55)  
  
**Response from GPT-4:**

``` python
def mode(laps):
    return max(set(laps), key=laps.count)

laps = [53, 53, 53, 55, 55]
print(mode(laps))
```

**Correct output:**

</div>

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/755.png" style="height:3.2cm" alt="image" />

**Question:** What is this function most likely be?  
**Choices:**  
(A) polynomial (B) trigonometric  
(C) exponential (D) logarithmic  
  
**Caption from Bard:**  
The image shows a graph of a function $f(x) = c$ on a graph paper. The
$x$ axis is the horizontal axis and the $y$ axis is the vertical axis.
The graph is a horizontal line that passes through the point $(0, c)$.
... The text “$(0, c)$” means that the point $(0, c)$ is on the graph.
The text “$f(x) = c$” means that the function $f(x)$ is equal to $c$ for
all values of $x$. ... The graph of a constant function is a horizontal
line. . ...  
  
**Response from GPT-4:**  
The function shown in the image is a constant function, which is a type
of polynomial function. Thus, the answer is A.  
  
**Correct output:**

</div>

<span id="fig:gpt4_examples" label="fig:gpt4_examples"></span>

</div>

#### Qualitative examples of Augmented GPT-4.

Augmented with external visual models, CoT GPT-4 and PoT GPT-4 are able
to achieve comparable performance with Multimodal Bard. As shown in
Figure <a href="#fig:gpt4_examples" data-reference-type="ref"
data-reference="fig:gpt4_examples">[fig:gpt4_examples]</a> (a), provided
with the accurate OCR text detected in the image, PoT GPT-4 accurately
understands the structural information of the image and generates a code
snippet to perform precise statistical reasoning. In Figure
<a href="#fig:gpt4_examples" data-reference-type="ref"
data-reference="fig:gpt4_examples">[fig:gpt4_examples]</a> (b), the
caption provides some accurate descriptions of the image (e.g.,
$f(x)=c$) along with hallucination (e.g., $y=3$, the line passes through
$(0,3)$) caused by the external Bard model. Although CoT GPT-4 predicts
the correct answer given the partially correct information, the
qualities of visual information augmented by external models have an
impact on the accurate visual perception and thus the final mathematical
reasoning performance. Examples in
§<a href="#sec:model_comparison" data-reference-type="ref"
data-reference="sec:model_comparison">[sec:model_comparison]</a> show
failure cases due to hallucination caused by external visual models.

# Related Work [sec:short_related_work]

Several benchmarks
[amini2019mathqa](http://arxiv.org/pdf/1905.13319v1), [cobbe2021training](http://arxiv.org/pdf/2110.14168v2), [mishra2022lila](None), [frieder2023mathematical](http://arxiv.org/pdf/2306.16282v1)
have emerged to assess the mathematical reasoning capabilities of LLMs,
but most focus solely on text-based tasks. Current benchmarks, such as
GSM-8K [cobbe2021training](http://arxiv.org/pdf/2110.14168v2), exhibit performance
saturation. Given the rise of LMMs [li2023multimodal](http://arxiv.org/pdf/2309.10020v1),
there is a need for robust multimodal benchmarks in scientific domains.
To address this gap, we introduce a math reasoning dataset that
incorporates visual contexts.

VQA datasets
[antol2015vqa](None), [gurari2018vizwiz](None), [mobasher101parsvqa](None)
gauge the visual reasoning abilities of LMMs. Recent studies explore
assessing LMMs beyond natural images, including abstract scenes,
geometry diagrams, figures, charts, documents, and synthetic images
[lu2021inter](None), [kahou2017figureqa](None), [masry2022chartqa](None). In
this work, we introduce new datasets (IQTest, FunctionQA, PaperQA) to
create a holistic benchmark for evaluating mathematical reasoning.

Generative foundation models like GPT-3, ChatGPT, GPT-4, Claude, and
LLaMA have enabled diverse task solutions without fine-tuning.
Specialized pretraining methods like
PixStruct [lee2023pix2struct](None),
MatCha [liu2022matcha](None), and
UniChart [masry2023unichart](None) enhance chart reasoning in
visual contexts. Models like LLaVA, miniGPT4, InstructBLIP, and Bard
leverage large-scale image-text data, while specialized versions, such
as LLaVAR [zhang2023llavar](None), [ye2023mplug](None), emphasize
document understanding and math comprehension. Recent
works [bitton2023visit](None), [yu2023mm](None) evaluate
instruction-following and reasoning capabilities, underscoring the
growing importance of generative foundation models in practical
applications. We introduce as a benchmark to evaluate their math
reasoning capabilities in varied visual contexts.

# Conclusion

# Detailed Related Work [sec:related_work]

#### Mathematical reasoning benchmarks.

Recently, numerous benchmarks
[amini2019mathqa](http://arxiv.org/pdf/1905.13319v1), [cobbe2021training](http://arxiv.org/pdf/2110.14168v2), [mishra2022lila](None), [frieder2023mathematical](http://arxiv.org/pdf/2306.16282v1)
have been proposed to evaluate the mathematical reasoning capabilities
of Large Language Models (LLMs). However, most of these are textual
only [lu2023dl4math](http://arxiv.org/pdf/2212.10535v2), despite a substantial amount of
mathematical information and reasoning being encapsulated in visual
modalities. Meanwhile, some datasets exhibit performance saturation; for
instance, GPT-4 achieves 92.0% accuracy on
GSM-8K [cobbe2021training](http://arxiv.org/pdf/2110.14168v2), a dataset of grade-school
mathematics questions. On the other hand, the recent rapid advancement
of Large Multimodal Models (LMMs) necessitates the establishment of
robust multimodal benchmarks. However, current multimodal reasoning
benchmarks provide limited coverage of rigorous and scientific domains
[antol2015vqa](None), [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1), [kahou2017figureqa](None), [mathew2022infographicvqa](None),
which are key components for creating general-purpose AI assistants. To
bridge this gap, it is crucial to develop a robust math reasoning
dataset that integrates visual contexts.

#### Vision-language reasoning benchmarks.

High-quality evaluation datasets and benchmarks are a cornerstone for
assessing the progress of machine learning models to solve real-world
tasks [liao2021we](None). Prior studies such as
VQA [antol2015vqa](None), [goyal2017making](None),
VizWiz [gurari2018vizwiz](None), and
ParsVQA-Caps [mobasher101parsvqa](None) assess the
general-purpose visual question answering abilities of the LMMs, with or
without task-specific training, on open-ended questions about images. In
addition, there are several works that focus on evaluating specific
skills of the LMMs beyond natural scenes, such as abstract scenes and
shapes) [antol2015vqa](None), [lu2021iconqa](None), [ji2022abstract](http://arxiv.org/pdf/2211.16492v1),
geometry diagrams
 [seo2015solving](http://arxiv.org/pdf/2308.02823v1), [lu2021inter](None), [chen2022unigeo](None), [cao2022augmented](http://arxiv.org/pdf/2206.02978v1),
figures and
charts [methani2020plotqa](None), [masry2022chartqa](None), [kahou2017figureqa](None), [chang2022mapqa](None), [kafle2018dvqa](None),
documents (text in images)
 [singh2019towards](None), [mathew2022infographicvqa](None), [liu2023hidden](None),
or synthetic
images [dahlgren2022clevr](None), [li2023super](None), [bitton2023breaking](None).
Besides, there has been significant progress on developing datasets to
judge LMMs on skills that require external knowledge
 [schwenk2022okvqa](None), [shah2019kvqa](None), common sense
reasoning [zellers2019recognition](http://arxiv.org/pdf/2402.17213v1), [yin2021broaden](http://arxiv.org/pdf/2402.17213v1),
scientific-knowledge [lu2022learn](http://arxiv.org/pdf/2209.09513v2), [kembhavi2017you](None), [kembhavi2016diagram](http://arxiv.org/pdf/1603.07396v1),
medical understanding [zhang2023pmc](None), [lau2018dataset](http://arxiv.org/pdf/2311.18681v1). In
this work, we create new datasets (IQTest, FunctionQA, PaperQA) and
subsequently design a benchmark for holistic evaluation of the math
reasoning capabilities of the LMMs.

#### Generative foundation models and their evaluation.

Recently, there has been a surge of generative foundation
models [bommasani2021opportunities](http://arxiv.org/pdf/2110.10024v1) that are trained on
web-scale data, such as GPT-3, ChatGPT, GPT-4, Claude, LLaMA,
LLaMA-Adapter [brown2020language](http://arxiv.org/pdf/2112.07522v2), [openai2022chatgpt](https://openai.com/blog/chatgpt), [openai2023gpt4](None), [claude2](https://www.anthropic.com/index/claude-2), [touvron2023llama](None), [llamaadapter2023](None),
with the ability to solve a wide range of downstream
tasks [wei2022emergent](http://arxiv.org/pdf/2403.15796v2) without any task-specific
finetuning. Prior work has focused on evaluating their abilities to
respond to the queries from various disciplines, grounded in text, such
as QA, math, medicine, coding and
science [bubeck2023sparks](http://arxiv.org/pdf/2303.12712v5), [nori2023capabilities](None), [chen2021evaluating](http://arxiv.org/pdf/1810.11895v3), [fu2023codeapex](None), [sun2023scieval](None), [wang2023scibench](None), [huang2023c](http://arxiv.org/pdf/2305.08322v3), [huang2022language](http://arxiv.org/pdf/2404.04619v1), [liu2023agentbench](None), [llamaadapter2023](None).
Prior work, such as PixStruct [lee2023pix2struct](None),
MatCha [liu2022matcha](None), and
UniChart [masry2023unichart](None), has focused on developing
specialized pretraining recipe for improved math and chart reasoning in
visual contexts.

On the vision-language side, there are several generative foundation
models such as LLaVA, miniGPT4, InstructBLIP, Flamingo, LLaMA-Adapter
V2, Multimodal
Bard [liu2023llava](http://arxiv.org/pdf/2402.11690v1), [zhu2023minigpt](None), [instructblip](https://arxiv.org/pdf/2305.06500), [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1), [awadalla2023openflamingo](None), [gao2023llamaadapterv2](None), [google2023bard](https://bard.google.com/)
that are trained on vast amount of
paired [schuhmann2022laion](None), [sharma2018conceptual](None), [lin2014microsoft](None)
and interleaved image-text data [zhu2023multimodal](None). In
addition, there has been recent development on specialized versions of
these LMMs for document understanding where visual contexts require text
recognition, math understanding being one of
them [zhang2023llavar](None), [ye2023mplug](None). In recent times,
there have been several works, such as Visit-Bench, LVLM-eHub,
MMBench [bitton2023visit](None), [yu2023mm](None), [liu2023mmbench](None), [xu2023lvlm](None), [shao2023tiny](None),
that assess their instruction-following and reasoning capabilities. As
the generative foundation models become more relevant to real-world
applications, unlike prior work, we propose to benchmark their
capabilities of math reasoning (logical, arithmetic, statistical) on a
diverse set of visual contexts (word problems in images, natural scenes,
geometrical shapes, and plots).

#### 

# Limitations of the Benchmark

# Data Collection Guidelines [sec:collection_guideline]

## Mathematical Reasoning Definition [sec:math_reasoning]

Seven mathematical reasoning types are defined in Table
<a href="#tab:math_definition" data-reference-type="ref"
data-reference="tab:math_definition">1</a>.

<div id="tab:math_definition" markdown="1">

<table>
<caption>Definitions and proportions of seven mathematical reasoning
categories in .</caption>
<thead>
<tr class="header">
<th style="text-align: center;"><strong>Math Reasoning</strong></th>
<th style="text-align: left;"><strong>Description</strong></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td rowspan="3" style="text-align: center;">Arithmetic Reasoning<br />
()</td>
<td style="text-align: left;">It covers the <em>fundamental
operations</em> such as addition, subtraction, multiplication, division,
and understanding of n<em>umber properties</em>. It may also include the
ability to interpret numerical data in different forms.</td>
</tr>
<tr class="even">
<td style="text-align: left;">It focuses on <em>data interpretation</em>
and <em>analysis</em>, including measures (mean, median, mode),
dispersion metrics (standard deviation, range), probability concepts,
regression, correlation, and data inferences. It also identifies trends,
outliers, and patterns.</td>
</tr>
<tr class="odd">
<td style="text-align: left;">It encompasses understanding
<em>variables</em>, <em>equations</em>, and the manipulation of
<em>expressions</em> with polynomials and exponents. It also covers
solving simple to complex equations, and grasping functions, their
properties, and graphical depictions.</td>
</tr>
<tr class="even">
<td rowspan="3" style="text-align: center;">Geometry Reasoning<br />
()</td>
<td style="text-align: left;">It emphasizes <em>spatial</em>
understanding, analysis of 2D and 3D <em>figures</em>, and reasoning
about their <em>shapes, sizes, and relationships</em>. It includes
symmetry, congruency, similarity, area, volume, and
transformations.</td>
</tr>
<tr class="odd">
<td style="text-align: left;">It involves intuitive understanding of
<em>daily numerical concepts</em>, including understanding time
differences, numerical judgment, and estimates. It covers temporal
reasoning, spatial numeric assessments, and practical uses like
budgeting and time reading.</td>
</tr>
<tr class="even">
<td style="text-align: left;">It deals with the application of
mathematical concepts in <em>scientific contexts</em>. This includes
scientific notations, formula use, understanding rates, proportions, and
percentages in practical situations, and problem-solving in scientific
inquiries.</td>
</tr>
<tr class="odd">
<td style="text-align: center;">Logical Reasoning<br />
()</td>
<td style="text-align: left;">It focuses on <em>critical thinking</em>
and <em>deduction</em> from provided information, including pattern
recognition, sequence understanding, predictions, and statement
evaluation. Key components include premises, conclusions, and the use of
abstract reasoning.</td>
</tr>
</tbody>
</table>

Definitions and proportions of seven mathematical reasoning categories
in .

</div>

## Mathematical Reasoning Examples [app:math_examples]

<div id="tab:math_examples" markdown="1">

| **Math** | **Examples** |
|:---|:---|
| ARI | <img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/802.png" style="height:2.7cm" alt="image" /> |
| **Solution:** |  |
| Find the cost of the silk scraps. Multiply: \$9.08 $\times$ 4 = \$36.32 |  |
| Find the cost of the canvas scraps. Multiply: \$8.17 $\times$ 4 = \$32.68 |  |
| Now find the total cost by adding: \$36.32 + \$32.68 = \$69 |  |
| She spent \$69. |  |
| **Answer:** 69 |  |
| STA | <img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/197.png" style="height:2.4cm" alt="image" /> |
| **Answer:** 2 |  |
| ALG | <img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/973.png" style="height:3.1cm" alt="image" /> |
| **Choices:** (A) larger than (B) equal to (C) smaller than |  |
| **Answer:** (A) larger than |  |
|  |  |
| **Question:** How many zeros does this function have? |  |
| **Answer:** 1 |  |
|  |  |
| **Question:** What is the value of $y$ at $x=1$? |  |
| **Answer:** 0 |  |
|  |  |
| GEO | <img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/28.png" style="height:2.8cm" alt="image" /> |
| **Diagram logic forms:** |  |
| `PointLiesOnLine(D, Line(B, A))` |  |
| `PointLiesOnCircle(B, Circle(D, radius))` |  |
| `PointLiesOnCircle(A, Circle(D, radius))` |  |
| `PointLiesOnCircle(C, Circle(D, radius))` |  |
| **Answer:** (C) 8.5 |  |
| NUM | <img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/864.jpg" style="height:2.35cm" alt="image" /> |
| **Named entities:** Winston Churchill, Charles de Gaulle |  |
| **Wiki caption**: Winston Churchill and General de Gaulle at Marrakesh, January 1944 |  |
| **Answer:** 16 |  |
| SCI | <img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/641.jpg" style="height:2.9cm" alt="image" /> |
| **Answer:** 5.77 |  |
| LOG | <img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/98.jpg" style="height:2.3cm" alt="image" /> |
| **Solution:** |  |
| Circle + Square = 5, Triangle + Triangle = 8, |  |
| Triangle = 4. |  |
| Circle + Triangle = 7, Circle = 3. |  |
| Therefore Square = 2 |  |
| **Answer:** 2 |  |

Examples of seven mathematical reasoning categories in .

</div>

## Visual Context Types [app:visual_context]

<figure id="fig:1_geometry_diagram">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/1_Geometry_Diagram.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>geometry
diagram</em> type.</figcaption>
</figure>

<figure id="fig:2">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/2_Synthetic_Scene.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>synthetic
scene</em> type.</figcaption>
</figure>

<figure id="fig:3">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/3_Bar_Chart.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>bar chart</em>
type.</figcaption>
</figure>

<figure id="fig:4">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/4_Natural_Image.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>natural
image</em> type.</figcaption>
</figure>

<figure id="fig:5">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/5_Scientific_Figure.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>scientific
figure</em> type.</figcaption>
</figure>

<figure id="fig:6">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/6_Table.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>table</em>
type.</figcaption>
</figure>

<figure id="fig:7">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/7_Function_Plot.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>function
plot</em> type.</figcaption>
</figure>

<figure id="fig:8">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/8_Abstract_Scene.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>abstract
scene</em> type.</figcaption>
</figure>

<figure id="fig:9">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/9_Puzzle_Test.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>puzzle test</em>
type.</figcaption>
</figure>

<figure id="fig:10">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/10_Scatter_Plot.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>scatter plot</em>
type.</figcaption>
</figure>

<figure id="fig:11">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/11_Line_Plot.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>line plot</em>
type.</figcaption>
</figure>

<figure id="fig:12">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/12_Pie_Chart.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>pie chart</em>
type.</figcaption>
</figure>

<figure id="fig:13">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/13_Document_Image.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>document
image</em> type.</figcaption>
</figure>

<figure id="fig:14">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/14_Medical_Image.png" style="width:100.0%" />
<figcaption>Examples of the visual context for the <em>medical
image</em> type.</figcaption>
</figure>

<figure id="fig:15">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/contexts/15_Others.png" style="width:100.0%" />
<figcaption>Examples of the visual context for <em>other</em> types,
including word cloud, map chart, radar chart, violin plot, and heatmap
chart.</figcaption>
</figure>

## Source Dataset Summary [sec:source_data]

The source datasets are summarized in Table
<a href="#tab:data_source" data-reference-type="ref"
data-reference="tab:data_source">3</a>.

<div id="tab:data_source" markdown="1">

| **Dataset** | **Category** | **Task** | **Context** | **Math Skill** |
|:---|:--:|:--:|:--:|:--:|
| IQTest (Ours) | Math-Targeted | FQA | Puzzle Test | Logical, Arithmetic |
| PaperQA (Ours) | Math-Targeted | FQA | Charts and Plots | Scientific |
| FunctionQA (Ours) | Math-Targeted | TQA | Function Plot | Algebraic |
| Geometry3K `\citeyearpar{lu2021inter}`{=latex} | Math-Targeted | GPS | Geometry Diagram | Geometry, Algebraic |
| GeoQA+ `\citeyearpar{cao2022augmented}`{=latex} | Math-Targeted | GPS | Geometry Diagram | Geometry, Algebraic |
| GEOS `\citeyearpar{seo2015solving}`{=latex} | Math-Targeted | GPS | Geometry Diagram | Geometry, Algebraic |
| UniGeo `\citeyearpar{chen2022unigeo}`{=latex} | Math-Targeted | GPS | Geometry Diagram | Geometry, Algebraic |
| CLEVR-Math `\citeyearpar{dahlgren2022clevr}`{=latex} | Math-Targeted | MWP | Synthetic Scene | Arithmetic |
| IconQA `\citeyearpar{lu2021iconqa}`{=latex} | Math-Targeted | MWP | Abstract Scene | Arithmetic |
| TabMWP `\citeyearpar{lu2023dynamic}`{=latex} | Math-Targeted | MWP | Table | Statistical, Arithmetic |
| SciBench `\citeyearpar{wang2023scibench}`{=latex} | Math-Targeted | TQA | Scientific Figure | Scientific |
| TheoremQA `\citeyearpar{chen2023theoremqa}`{=latex} | Math-Targeted | TQA | Scientific Figure | Scientific |
| ChartQA `\citeyearpar{masry2022chartqa}`{=latex} | General VQA | FQA | Charts and Plots | Statistical |
| FigureQA `\citeyearpar{kahou2017figureqa}`{=latex} | General VQA | FQA | Charts and Plots | Statistical |
| DVQA `\citeyearpar{kafle2018dvqa}`{=latex} | General VQA | FQA | Bar Chart | Statistical |
| MapQA `\citeyearpar{chang2022mapqa}`{=latex} | General VQA | FQA | Map Chart | Statistical |
| PlotQA `\citeyearpar{methani2020plotqa}`{=latex} | General VQA | FQA | Scatter Plot | Statistical |
| DocVQA `\citeyearpar{mathew2022infographicvqa}`{=latex} | General VQA | FQA | Document Image | Statistical |
| AI2D `\citeyearpar{kembhavi2016diagram}`{=latex} | General VQA | TQA | Scientific Figure | Scientific |
| ScienceQA `\citeyearpar{lu2022learn}`{=latex} | General VQA | TQA | Scientific Figure | Scientific |
| TQA `\citeyearpar{kembhavi2017you}`{=latex} | General VQA | TQA | Scientific Figure | Scientific |
| A-OKVQA `\citeyearpar{schwenk2022okvqa}`{=latex} | General VQA | VQA | Natural Image | Arithmetic, Numeric |
| KVQA `\citeyearpar{shah2019kvqa}`{=latex} | General VQA | VQA | Natural Image | Arithmetic, Numeric |
| ParsVQA-Caps `\citeyearpar{schwenk2022okvqa}`{=latex} | General VQA | VQA | Natural Image | Arithmetic, Numeric |
| TextVQA `\citeyearpar{singh2019towards}`{=latex} | General VQA | VQA | Natural Image | Arithmetic, Numeric |
| VizWiz `\citeyearpar{gurari2018vizwiz}`{=latex} | General VQA | VQA | Natural Image | Arithmetic, Numeric |
| VQA2.0 `\citeyearpar{goyal2017making}`{=latex} | General VQA | VQA | Natural Image | Arithmetic, Numeric |
| PMC-VQA `\citeyearpar{zhang2023pmc}`{=latex} | General VQA | VQA | Medical Image | Scientific |
| VQA-RAD `\citeyearpar{lau2018dataset}`{=latex} | General VQA | VQA | Medical Image | Scientific |
| Super-CLEVR `\citeyearpar{li2023super}`{=latex} | General VQA | VQA | Synthetic Scene | Arithmetic |
| VQA-AS `\citeyearpar{antol2015vqa}`{=latex} | General VQA | VQA | Abstract Scene | Arithmetic |

Summary of the 31 different source datasets in . Among these,
FunctionQA, IQTest, and PaperQA are our newly annotated datasets. The
table provides details on their category, task, visual context, and
primary mathematical reasoning skill types.

</div>

# Data Collection Details [app:collection_details]

## Automatic Selection of Mathematical Problems [sec:automatic_selection]

<div class="table*" markdown="1">

<div class="boxedminipage" markdown="1">

1 most, least, fewest more, less, fewer, largest, smallest, greatest,
larger, smaller, greater, highest, lowest, higher, lower, increase,
decrease, minimum, maximum, max, min, mean, average, median, total, sum,
add, subtract, difference, quotient, gap, half, double, twice, triple,
square, cube, root, approximate, approximation, triangle, rectangle,
circle, square, cube, sphere, cylinder, cone, pyramid, multiply, divide,
percentage, percent, ratio, proportion, fraction, rate

</div>

</div>

## Human Labeling of Mathematical Problems [sec:human_is_math]

<figure id="fig:gui_math_labeling">
<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gui_ismath_anonymous.png" style="width:100.0%" />
<figcaption>GUI for labeling if a problem involves mathematical
reasoning.</figcaption>
</figure>

<div class="table*" markdown="1">

<div class="boxedminipage" markdown="1">

1 We are compiling a dataset that incorporates image context and
involves mathematical reasoning (MathQA in visual contexts). We have
gathered a set of examples in which some involve mathematical reasoning,
while others do not.

In our task, a question can be classified as a mathematical problem if
it

-   Involves numbers or symbols in the question text or the image
    context, AND requires further operations or transformations to be
    performed on them to reach a solution.

-   Involves more complex forms of mathematical reasoning, including
    logical reasoning, abstract thought, and understanding of patterns.

Based on the definition above, a problem is classified as a negative
example (NOT involving mathematical reasoning) if it:

-   Does not involve any numbers or quantity words, OR

-   Involves only counting, reading, or recognizing numbers, OR

-   Relies solely on factual information, such as recalling years and
    dates.

</div>

</div>

We developed an annotation tool, as illustrated in Figure
<a href="#fig:gui_math_labeling" data-reference-type="ref"
data-reference="fig:gui_math_labeling">1</a>, to enable expert
annotators to label problems that involve mathematical reasoning.
Annotators were trained using detailed instructions, as shown in Table
<a href="#tab:instruction_is_math" data-reference-type="ref"
data-reference="tab:instruction_is_math">[tab:instruction_is_math]</a>,
along with a variety of examples—positive ones that involve mathematical
reasoning and negative ones that do not. We provided three labeling
options:

-   *Yes* - This indicates that the problem involves mathematical
    reasoning.

-   *No* - This indicates that the problem does not involve mathematical
    reasoning.

-   *Unsure* - This option should be selected if it is uncertain whether
    the problem involves mathematical reasoning. (Annotators are advised
    to use this option sparingly.)

They may leave comments if they find anything incorrect or offensive for
removal at a later stage.

In our study, we employed the Fleiss Kappa score to conduct an
inter-annotator agreement analysis among three annotators tasked with
labeling examples based on mathematical reasoning. The Fleiss Kappa
score is a statistical measure used to evaluate the reliability of
agreement between multiple raters, providing a quantifiable metric to
assess the consistency across different annotators. A score of 1
indicates perfect agreement, while a score of 0 suggests no agreement
beyond what would be expected by chance. Our analysis yielded a Fleiss
Kappa score of 0.775, indicating a substantial level of consistency
among the annotators. This high degree of agreement underscores the
reliability of our annotation process and affirms the quality of the
labeled data generated for our study.

## Annotating Three New Datasets [sec:annotate_new_data]

<figure id="fig:gui_new_data_annotation">
<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gui_new_data_annotation.png" style="width:100.0%" />
<figcaption>GUI for annotating our new source datasets.</figcaption>
</figure>

## Human Labeling of Mathematical Reasoning [sec:human_math_reasoning]

<figure id="fig:gui_math_skill_labeling">
<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gui_skill_labeling_2.png" style="width:100.0%" />
<figcaption>GUI for labeling mathematical reasoning skills.</figcaption>
</figure>

# More Dataset Analysis [app:data_analysis]

#### Question distribution.

Apart from English questions, contains 6.57% non-English questions,
including languages such as Chinese and Persian. The multilingual
feature necessitates that models be capable of understanding and
processing multiple languages to ensure accurate results across the
dataset. As illustrated in Table
<a href="#fig:source_dataset" data-reference-type="ref"
data-reference="fig:source_dataset">[fig:source_dataset]</a>, the
average number of words in English questions within is 15.58, while the
maximum number of words in a question reaches 213.

Figure <a href="#fig:question_length" data-reference-type="ref"
data-reference="fig:question_length">1</a> further elucidates the
distribution of word counts, highlighting the diverse patterns of
questions. features two types of questions: multiple-choice questions
and free-form questions. For multiple-choice questions, the average
number of choices is 3.4, while the maximum number of choices is 8. In
the case of free-form questions, answers can be integers, floating-point
numbers, or lists, which can be converted into a standard format. The
standard settings in question and answer types facilitate consistent
accuracy evaluation for existing models.

<figure id="fig:question_length">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/question_length_default.png" style="width:90.0%" />
<figcaption>The distribution of the number of words per question in .
Questions with a length greater than 60 are categorized as 61 for
visualization simplicity.</figcaption>
</figure>

#### Dataset category and task type.

Source datasets in can be categorized into two types: math-targeted VQA
datasets, which are originally proposed for assessing mathematical
reasoning, and general VQA datasets, which address visual reasoning in
everyday scenarios. The distribution proportions of these two categories
(55.4% vs. 44.6%, as illustrated in Figure
<a href="#fig:category" data-reference-type="ref"
data-reference="fig:category">2</a>) within enable a balanced
examination of mathematical reasoning in both domain-specific and
general-purpose applications. The distribution of the five tasks
contained within is visualized in Figure
<a href="#fig:task" data-reference-type="ref"
data-reference="fig:task">3</a>. The relatively balanced distribution of
these tasks enhances the benchmarking robustness that our dataset
provides.

<figure id="fig:category">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/category_paired_9.png" style="width:55.0%" />
<figcaption>Category distribution of problems within .</figcaption>
</figure>

<figure id="fig:task">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/task_paired_9.png" style="width:60.0%" />
<figcaption>Task type distribution of problems within .</figcaption>
</figure>

#### Grade level.

The datasets within are categorized into four distinct grade levels:
*elementary school*, *high school*, *college*, and *not applicable*,
each representing a different level of reasoning complexity and
contextual application. The *elementary school* category aligns with the
typical mathematical curriculum of elementary education, introducing
basic topics such as arithmetic operations and introductory geometry.
*High school* level questions delve into more complex mathematical
concepts such as algebra, geometry, and introductory calculus. The
*college* category encapsulates the highest level of complexity,
featuring questions on advanced mathematical and scientific concepts
like calculus, linear algebra, and physics. Questions without specific
grade levels are categorized as *not applicable*.

The distribution of questions across these grade levels is visualized in
Figure <a href="#fig:grade_level" data-reference-type="ref"
data-reference="fig:grade_level">4</a>. This structured categorization
enriches the diversity of , providing a meaningful framework for
evaluating and benchmarking the mathematical and visual reasoning
capabilities of various models across different educational contexts,
thereby assessing their practical utility and educational relevance.

<figure id="fig:grade_level">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/grade_paired_9.png" style="width:55.0%" />
<figcaption>Distribution of questions across different grade levels
within .</figcaption>
</figure>

#### Visual context.

The datasets within encompass over 10 different visual contexts (with
the distribution shown in Figure
<a href="#fig:context" data-reference-type="ref"
data-reference="fig:context">5</a>), crucial for evaluating models’
ability to interpret and reason across diverse visual information.
Common visual contexts include geometry diagrams, synthetic scenes, bar
charts, natural images, and scientific figures as illustrated in Figure
<a href="#fig:2" data-reference-type="ref"
data-reference="fig:2">[fig:2]</a> to Figure
<a href="#fig:13" data-reference-type="ref"
data-reference="fig:13">[fig:13]</a>. Less frequent, yet equally
important visual contexts such as medical images, word clouds, map
charts, radar charts, violin plots, and heatmap charts are depicted in
Figure <a href="#fig:14" data-reference-type="ref"
data-reference="fig:14">[fig:14]</a> and Figure
<a href="#fig:15" data-reference-type="ref"
data-reference="fig:15">[fig:15]</a>. These visual contexts, ranging
from common to specialized representations, challenge the models to
decode and reason with varying visual information, contributing to a
more robust and comprehensive evaluation. The diversity in visual
contexts enriches , enhancing the benchmarking robustness and providing
a solid foundation for understanding the practical utility and
domain-specific performance of various models across different domains
and applications.

<figure id="fig:context">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/context_paired_9.png" style="width:60.0%" />
<figcaption>Visual context distribution within .</figcaption>
</figure>

#### Mathematical reasoning ability.

The datasets within encompass a spectrum of seven distinct mathematical
reasoning types, facilitating a thorough evaluation of models’
mathematical reasoning capabilities. Figure
<a href="#fig:math_distribution" data-reference-type="ref"
data-reference="fig:math_distribution">6</a> illustrates the portion of
each reasoning type involved in the problems, with arithmetic being the
most frequent and logical reasoning being the least frequent. This
distribution reflects the varying degrees of mathematical reasoning
required across different problems. Figure
<a href="#fig:math_number" data-reference-type="ref"
data-reference="fig:math_number">7</a> further delineates the
distribution of reasoning types, showcasing a mean of 1.45. The sparse
distribution observed aids in the precise analysis of each type’s
performance by the models, providing a nuanced understanding of their
strengths and weaknesses across different mathematical reasoning
domains. This structured representation of mathematical reasoning types
within not only enriches the dataset but also significantly contributes
to a more robust and comprehensive evaluation of models, aiding in the
identification of areas for improvement and the development of more
proficient mathematical reasoning models.

<figure id="fig:math_distribution">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/skill_husl_h.png" style="width:65.0%" />
<figcaption>Portion of each mathematical reasoning type involved in the
problems of .</figcaption>
</figure>

<figure id="fig:math_number">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/math_number.png" style="width:55.0%" />
<figcaption>Distribution of the number of mathematical reasoning types
within .</figcaption>
</figure>

# More Details on the Setup [app:setup]

## Frequent Guess [sec:frequent_guess]

We employ a strategy where the most frequent answers in the *testmini*
set are utilized as predictions for various question and answer types.
For multiple-choice questions, the most frequent option is selected
based on the number of available options. For instance, option $B$ is
chosen for questions with two options, aligning with the answer
distribution in *testmini*. Similarly, for questions requiring an answer
type of integer, a floating number with one decimal place, a floating
number with two decimal places, or a list, we use $2$, $1.2$, $0.21$,
and $[0, 2, 0, 2, 1, 7, 1, 2, 0, 3, 0, 6]$ respectively, in accordance
with the answer distribution observed in *testmini*.

## Prompt for Answer Extraction [sec:promt_answer_extraction]

The prompt used to instruct GPT-4 for answer extraction is illustrated
in Table
<a href="#tab:promt_answer_extraction" data-reference-type="ref"
data-reference="tab:promt_answer_extraction">1</a>.

<div id="tab:promt_answer_extraction" markdown="1">

| **Element** | **Prompt** |
|:--:|:---|
| Task description | Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. |
| Example 1 |  |
| **Question:** Which number is missing? |  |
| **Model response:** The number missing in the sequence is 14. |  |
| **Extracted answer:** |  |
|  |  |
| Example 2 |  |
| **Question:** What is the fraction of females facing the camera? |  |
| **Model response:** The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera. |  |
| **Extracted answer:** |  |
| Example 3 |  |
| **Question:** How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: \$) |  |
| **Model response:** Luca needs \$1.45 to buy a sour apple candy and a butterscotch candy. |  |
| **Extracted answer:** |  |
|  |  |
| Example 4 |  |
| **Question:** Between which two years does the line graph saw its maximum peak? |  |
| **Model response:** The line graph saw its maximum peak between 2007 and 2008. |  |
| **Extracted answer:** |  |
| Example 5 |  |
| **Question:** What fraction of the shape is blue? |  |
| **Choices:** (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5 |  |
| **Model response:** The correct answer is (B) 8/11. |  |
| **Extracted answer:** |  |

Task description along with five examples used to prompt GPT-4 for
answer extraction.

</div>

## Prompts for Response Generation [sec:promt_response_generation]

<div id="tab:promt_response_generation" markdown="1">

| **Question type** | **Answer type** | **Task instruction** |
|:--:|:--:|:---|
| multiple-choice | Text | Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end. |
|  |  | Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. |
| Free-form | Float (1) | Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end. |
|  |  | Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end. |
| Free-form | List | Please answer the question requiring a Python list as an answer and provide the final list, e.g., \[1, 2, 3\], \[1.2, 1.3, 1.4\], at the end. |

The task instructions for different question and answer types in answer
extraction. Here, Float (1) refers to a floating-point number with one
decimal place, and Float (2) refers to a floating-point number with two
decimal places.

</div>

## Prompt for Caption Generation

We instruct Multimodal Bard to generate a detailed description for an
input image, aiming to augment current LLMs with visual understanding
capabilities. The prompt is shown in
Table <a href="#tab:prompt_bard_caption" data-reference-type="ref"
data-reference="tab:prompt_bard_caption">[tab:prompt_bard_caption]</a>.

<div class="table*" markdown="1">

<div class="boxedminipage" markdown="1">

1 Describe the fine-grained content of the image or figure, including
scenes, objects, relationships, and any text present.

</div>

</div>

## Model Hyperparameters

The hyperparameters for the experiments in
§<a href="#sec:experimental_setup" data-reference-type="ref"
data-reference="sec:experimental_setup">[sec:experimental_setup]</a> are
set to their default values unless specified otherwise. Table
<a href="#tab:llm_generating_params" data-reference-type="ref"
data-reference="tab:llm_generating_params">3</a> and Table
<a href="#tab:lmm_generating_params" data-reference-type="ref"
data-reference="tab:lmm_generating_params">4</a> detail specific
generation parameters for the various large language models (LLMs) and
large multimodal models (LMMs) we evaluated, respectively.

<div id="tab:llm_generating_params" markdown="1">

| **Model** | **Generation Setup**                                        |
|:----------|:------------------------------------------------------------|
| Claude-2  | model = `claude-2`, temperature = 0, max_tokens = 1024      |
| ChatGPT   | model = `gpt-3.5-turbo`, temperature = 0, max_tokens = 1024 |
| GPT-4     | model = `gpt-4-0613`, temperature = 0, max_tokens = 1024    |

Generating parameters for various LMMs.

</div>

<div id="tab:lmm_generating_params" markdown="1">

| **Model** | **Generation Setup** |
|:-----------------|:------------------------------------------------|
| IDEFICS-9B-Instruct | max_new_tokens = 256, temperature = 1.0 |
| mPLUG-Owl-LLaMA-7B | do_sample = True, top-k = 5, max_length = 512 |
| miniGPT4-LLaMA-2-7B | num_beams = 1, temperature = 1.0, max_new_tokens = 300, max_length = 1000 |
|  | max_gen_len = 256, temperature = 0.1, top_p= 0.75 |
| LLaVAR | do_sample = True, temperature = 0.2, max_new_tokens = 1024 |
| InstructBLIP-Vicuna-7B | do_sample = False, num_beams = 5, max_length = 256, min_length = 1, top_p = 0.9, repetition_penalty = 1.0, temperature = 1 |
|  | do_sample = True, temperature = 0.2, max_new_tokens = 1024 |
|  | Chatbot URL: <https://bard.google.com>, evaluation dates range from Sep 8, 2023 to Sep 10, 2023 |
|  | Chatbot URL: <https://chat.openai.com>, evaluation dates range from Oct 7, 2023 to Oct 15, 2023 |

Generating parameters for various LMMs.

</div>

## Human Performance [sec:human_performance]

We conducted a study to evaluate human performance on the *testmini*
subset of the , utilizing Amazon Mechanical Turk (AMT). Each question
from the *testmini* subset was assigned to five annotators, all of whom
have a history of completing more than 5,000 HIT tasks and boast an
acceptance score higher than 0.99, to ensure the quality of the results.
The study comprised five test questions and two qualification questions,
which were to be answered within a 20-minute timeframe. The
qualification questions consisted of elementary math word problems
requiring basic arithmetic operations (e.g., addition and subtraction).
Only annotators who successfully answered the qualification questions
were deemed eligible for the study, and their responses were included in
the final analysis. Additionally, annotators were requested to provide
information regarding their highest level of educational attainment. We
retained the results exclusively from annotators who had achieved a high
school diploma or higher, as 30.9% of the problems in are of high-school
level difficulty and 10.8% correspond to college-level curricula.

## Multimodal Bard Assessment Task [sec:human_study_bard]

A screenshot of our AMT worker interface, utilized for the Multimodal
Bard assessment task, is provided in Figure
<a href="#fig:worker_bard" data-reference-type="ref"
data-reference="fig:worker_bard">1</a>. The workers were compensated at
a rate of \$18 per hour.

<figure id="fig:worker_bard">
<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/worker_ui_bard_assessment.png" />
<figcaption>Screenshot of the Multimodal Bard assessment task
interface.</figcaption>
</figure>

# More Experimental Results

## Results on the Test Set

Table <a href="#tab:mathvista_test_result" data-reference-type="ref"
data-reference="tab:mathvista_test_result">[tab:mathvista_test_result]</a>
reports the accuracy scores of two heuristic baselines, two leading
augmented LLMs (CoT GPT-4, PoT GPT-4), and one leading LMM
(LLaVA-LLaMA-2-13B) on the *test* subset. The minor differences between
scores on the *test* subset and the *testmini* subset, as shown in Table
<a href="#tab:mathvista" data-reference-type="ref"
data-reference="tab:mathvista">[tab:mathvista]</a>, suggest that
*testmini* effectively mirrors the *test* subset, serving as a valuable
evaluation subset for model development, especially for those who have
limited computing resources.

<div class="table*" markdown="1">

</div>

## Scores for Math Reasoning Types [sec:scores_math_reasoning]

## Scores for Various Visual Contexts [sec:scores_visual_contexts]

<a href="#fig:visual_context_bar_chart" data-reference-type="ref"
data-reference="fig:visual_context_bar_chart">1</a> illustrates the
accuracy scores of leading baselines on across a diverse range of visual
contexts. Other foundation models trail behind humans in visual
perception and reasoning across most visual context categories.
Multimodal Bard demonstrates comparable performance to humans in
questions with a visual context of geometry diagrams, showcasing its
promising capabilities in recognizing geometric shapes and
relationships. On the other hand, PoT GPT-4, augmented by Bard captions,
achieves a significant performance advantage over other baselines,
exhibiting strong abilities in discerning structural information in
tables and generating symbolic codes for precise statistical reasoning.

<figure id="fig:visual_context_bar_chart">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/visual_context_bar_chart_gpt4v.png"
style="width:100.0%" />
<figcaption>Accuracy scores of leading baselines across various visual
contexts in .</figcaption>
</figure>

## Scores Across Different Grade Levels [sec:grade_level_bar_chart]

<a href="#fig:grade_level_bar_chart" data-reference-type="ref"
data-reference="fig:grade_level_bar_chart">2</a> displays the average
accuracy scores across different grade levels (*elementary school*,
*high school*, and *college*) for the leading foundation models. Humans
exhibit the highest performance on questions at the elementary school
level , while they fare the worst on college-level questions within .
Foundation model baselines exhibit varying performance behaviors: they
achieve better accuracy scores on high school level questions compared
to the other two categories.

<figure id="fig:grade_level_bar_chart">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/grade_level_bar_chart_gpt4v.png" style="width:80.0%" />
<figcaption>Average accuracy scores across different grade levels for
.</figcaption>
</figure>

## Ablation Study for LLMs [sec:llm_ablation_study]

Table <a href="#fig:llm_ablation_study" data-reference-type="ref"
data-reference="fig:llm_ablation_study">3</a> presents an ablation study
conducted on LLMs, examining their performance under varying visual
information inputs.

<figure id="fig:llm_ablation_study">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/llm_ablation_study.png" style="width:70.0%" />
<figcaption>Average accuracy scores of LLM baselines under various
visual inputs.</figcaption>
</figure>

## LLMs with Different Shots [sec:llm_fewshot]

<figure id="fig:gpt3_prompt">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/llm_fewshot_qonly.png" style="width:100.0%" />
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/llm_fewshot_cot.png" style="width:100.0%" />
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/llm_fewshot_pot.png" style="width:100.0%" />
<figcaption></figcaption>
</figure>

## LMMs with Different Shots [sec:lmm_fewshot]

<figure id="fig:lmm_fewshot">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/lmm_fewshot.png" style="width:55.0%" />
<figcaption aria-hidden="true"></figcaption>
</figure>

## Hallucinations in Model Explanations [sec:hallucination_in_explanation]

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/19.jpg" style="height:3.5cm" alt="image" />  
**Question:** What is the highest amount this class measures?  
  
**Response from Bard:**  
.  
  
The markings on the side of the beaker are in milliliters (ml). .  
  
The other markings on the beaker are 300 ml, 200 ml, 100 ml, and . .  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:visual_hallucination"
label="fig:visual_hallucination"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/6.jpg" style="height:0.7cm" alt="image" />  
**Question:** As shown in the figure, if $CB = 4.0$, $DB = 7.0$, and $D$
is the midpoint of $AC$, then the length of $AC$ is ().  
  
**Choices:** (A) $6$cm (B) $7$cm (C) $4$cm (D) $5$cm  
  
**Response from Bard:**  
Since $D$ is the midpoint of $AC$, then $AD = DC = (AC/2)$.  
  
We know that $CB = 4.0$ and $DB = 7.0$, and that $AD = DC$. So we can
write the equation: $$\red{AD + DC = 4.0 + 7.0}$$ $$2AD = 11.0$$
$$AD = 5.5$$ $$AC = 2AD = 11.0 \text{cm}$$

Therefore, the length of $AC$ is $11.0$ cm, which is answer choice
(D).  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:textual_hallucination"
label="fig:textual_hallucination"></span>

</div>

## More Examples for Multimodal Bard [sec:more_examples_bard]

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/443.jpg" style="height:1.5cm" alt="image" />  
**Question:** As shown in the figure: In Rt$\triangle$ABC, $\angle$C =
90.0, AC = 8.0, AB = 10.0, then the value of $\text{sin} B$ is equal to
().  
  
**Choices:** (A) $\frac{3}{5}$ (B) $\frac{4}{5}$ (C) $\frac{3}{4}$ (D)
$\frac{4}{3}$  
  
**Response from Bard:**  
In a right triangle, the sine of an angle is equal to the ratio of the
opposite side to the hypotenuse. In this case, the opposite side is
$AC = 8$ and the hypotenuse is $AB = 10$. Therefore, $\text{sin} B$ =
$8/10$ = .  
  
Here is the detailed solution:

Given:

\* $\triangle ABC$ is a right triangle with right angle at $C$  
\* $AC = 8$  
\* $AB = 10$  
  
To find: $\text{sin} B$
$$\text{sin} B = \text{opposite}/\text{hypotenuse} = AC/AB = 8/10 = \red{3/5}$$  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_1" label="fig:bard_error_1"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/788.jpg" style="height:3.2cm" alt="image" />  
**Question:** What is the age gap between these two people in image?
(unit: years)  
  
**Response from Bard:**  
  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_2" label="fig:bard_error_2"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/284.png" style="height:4.8cm" alt="image" />  
**Question:** What is the accuracy of the algorithm with lowest
accuracy?  
  
**Response from Bard:**  
  
  
The other algorithms are *desert*, *pine*, and *globe*.  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_284" label="fig:bard_error_284"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/96.png" style="height:5.2cm" alt="image" />  
**Question:** What is the age gap between these two people in image?
(unit: years)  
  
**Response from Bard:**  
The dissatisfied people are 78%, the satisfied people (20%).  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_96" label="fig:bard_error_96"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/850.png" style="height:4.2cm" alt="image" />  
**Question:** Is the number of cars in front of the tiny metal thing
less than the number of large matte things in front of the cyan rubber
road bike?  
**Choice:** (A) Yes (B) No  
  
**Response from Bard:**  
The answer is (B), No.  
  
There are no cars in front of the tiny metal thing (the motorcycle).
There are two large matte things in front of the cyan rubber road bike
(the van and the bicycle).  
  
So the number of large matte things in front of the cyan rubber road
bike is greater than the number of cars in front of the tiny metal
thing.  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_850" label="fig:bard_error_850"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/644.png" style="height:3.2cm" alt="image" />  
**Question:** What is the missing number in the picture?  
**Choice:** (A) 6 (B) 8 (C) 10 (D) 11  
  
**Response from Bard:**  
.  
  
  
  
2  
10  
(8)  
  
/12  
  
  
  
**Correct output:**

</div>

<span id="fig:bard_error_644" label="fig:bard_error_644"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/617.jpg" style="height:3.2cm" alt="image" />  
**Question:** The magnitude of the acceleration vector a is
$10 \mathrm{~cm} / \mathrm{s}^2$. Use the figure to estimate the normal
components of $\mathbf{a}$.  
  
**Response from Bard:**  
The normal component of the acceleration is given by
$$a_n = a \cos \theta = 10 \text{ cm}/s^2 \cdot \cos 30^\circ = \dfrac{10 \sqrt{3}}{2} \text{ cm}/s^2$$
The normal component of the acceleration is approximately
$7.07 \text{ cm}/s^2$.  
  
Here is a breakdown of the steps involved in finding the normal
component of the acceleration:  
  
1. Find the angle between the acceleration vector and the radius
vector.  
2. Use the cosine function to find the cosine of the angle.  
3. Multiply the magnitude of the acceleration vector by the cosine of
the angle to find the normal component of the acceleration.  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_617" label="fig:bard_error_617"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/329.png" style="height:4.2cm" alt="image" />  
**Question:** The derivative of $y$ at $x=6$ is

------------------------------------------------------------------------

that at $x=8$  
**Chocies:** (A) larger than (B) equal to (C) smaller than  
  
**Response from Bard:**  
  
  
The graph of the function $y=\log_2 x$ is a concave up curve. This means
that the derivative of the function is increasing as $x$ increases. So,
.  
  
Here is a table of the values of the derivative of $y$ at different
values of $x$:  
  
$x \mid dy/dx$  
–$\mid$–  
  
  
  
  
As you can see, . .  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_329" label="fig:bard_error_329"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/23.png" style="height:5.2cm" alt="image" />  
**Question:** The derivative of $f(x)$ at $x=2$ is

------------------------------------------------------------------------

that at $x=5$  
**Chocies:** (A) larger than (B) equal to (C) smaller than  
  
**Response from Bard:**  
  
  
The derivative of a function at a point gives the slope of the line
tangent to the function’s graph at that point. The steeper the line, the
larger the derivative.  
  
In mathematical terms, the derivative of $f(x)$ is given by the
expression . Plugging in $x=2$ and $x=5$, . Therefore, .  
  
**Wrong output:**   
  
**Ground truth:**

</div>

<span id="fig:bard_error_23" label="fig:bard_error_23"></span>

</div>

## Comparisons of Different Models [sec:model_comparison]

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/852.png" style="height:2.8cm" alt="image" />

**Question:** Which number is missing?  
  
**Ground truth:**   
  
**Solution:**  
$4 + 1 = 5, 5 + 2 = 7  ~(2 = 1 + 1)$  
$10 + 4 = 14, 14 + 5 = 19 ~(5 = 4 + 1)$  
$25 + 7 = 32, 32 + 8 = 40 ~(8 = 7 + 1)$

</div>

<div class="mybody" markdown="1">

**IDEFICS:**

</div>

<div class="mybody" markdown="1">

**mPLUG-Owl:**

</div>

<div class="mybody" markdown="1">

**miniGPT4:**

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**

</div>

<div class="mybody" markdown="1">

**LLaVAR:**

</div>

<div class="mybody" markdown="1">

**InstructBLIP:**

</div>

<div class="mybody" markdown="1">

**LLaVA:**

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
  
  
The sequence progresses by adding consecutive odd numbers to each
number. The first number is 4, and we add 1 to get 5. , and so on. The
missing number is therefore the next number after 32, .  
  
Here is the complete sequence: $$4 + 1 = 5$$ $$5 + \red{3 = 7}$$
$$7 + \red{5 = 10}$$ $$10 + \red{7 = 14}$$ $$14 + \red{9 = 19}$$
$$19 + \red{11 = 25}$$ $$25 + \red{13 = 32}$$ $$32 + \red{15 = 28}$$

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
4, 5, 7, 10, 14, 19, 25, 32, ?, 52  
  
Answer: The missing number is 40.  
  
This sequence follows a pattern where the difference between each number
increases by one. For example, the difference between 4 and 5 is 1, the
difference between 5 and 7 is 2, the difference between 7 and 10 is 3,
and so on. .

**Correct output:**

</div>

<span id="fig:example_852" label="fig:example_852"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/852.png" style="height:2.8cm" alt="image" />

**Question:** Which number is missing?  
  
**Ground truth:**   
  
**Solution:**  
$4 + 1 = 5, 5 + 2 = 7  ~(2 = 1 + 1)$  
$10 + 4 = 14, 14 + 5 = 19 ~(5 = 4 + 1)$  
$25 + 7 = 32, 32 + 8 = 40 ~(8 = 7 + 1)$

</div>

<div class="mybody" markdown="1">

**Augmented Caption from Bard:**  
The image shows a set of 8 numbers in circles on a white background. The
numbers are 4, 5, 7, 10, 14, 19, 25, and 32. There is a question mark in
the last circle. The numbers are arranged in increasing order, .
Therefore, .  
  
The image is a simple math problem, and it is likely that it is being
used as part of an IQ test or other kind of intelligence test. The image
is also consistent with the webpages that you mentioned, which are all
related to IQ tests.  
  
**Augmented OCR Text from EasyOCR:**  
(\[300, 80\], 5),  
(\[83, 281\], 10), (\[285, 281\], 14), (\[491, 279\], 19),  
(\[81, 485\], 25), (\[285, 485\], 32),

</div>

<div class="mybody" markdown="1">

**CoT GPT-4:**  
The pattern in the numbers is that . For example, the second number (5)
is 1 more than the first number (4) . The third number (7) is .  
  
Following this pattern, . $$32 + \red{9 + 1 = 42}$$ So, .

</div>

<div class="mybody" markdown="1">

**PoT GPT-4:**

``` python
numbers = [4, 5, 7, 10, 14, 19, 25, 32]
diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)] # [1, 2, 3, 4, 5, 6, 7]
missing_number = numbers[-1] + diffs[-1]
print(missing_number)
```

**Output:**

</div>

<span id="fig:example_852_llm" label="fig:example_852_llm"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/540.png" style="height:2.4cm" alt="image" />

**Question:** What fraction of the shape is blue?  
  
**Choices:** (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**IDEFICS:**

</div>

<div class="mybody" markdown="1">

**mPLUG-Owl:**

</div>

<div class="mybody" markdown="1">

**miniGPT4:**

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**

</div>

<div class="mybody" markdown="1">

**LLaVAR:**

</div>

<div class="mybody" markdown="1">

**InstructBLIP:**

</div>

<div class="mybody" markdown="1">

**LLaVA:**

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
  
  
The shape is divided into 11 equal parts. . Therefore, .  
  
The other choices are incorrect. Option (A), . Option (B), 8/11, is the
fraction of the shape that is not blue. Option (D), 3/5, is not a
fraction of the shape.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**

</div>

<span id="fig:example_540" label="fig:example_540"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/540.png" style="height:2.4cm" alt="image" />

**Question:** What fraction of the shape is blue?  
  
**Choices:** (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**Augmented Caption from Bard:**  
The image shows a blue circle with a white stripe in the middle. . The
colors are blue, . There is no text present in the image.  
  
The image could be used to represent a variety of concepts, such as
fractions, colors, or the four elements. It could also be used as a
decorative element or a symbol.

</div>

<div class="mybody" markdown="1">

**CoT GPT-4:**  

</div>

<div class="mybody" markdown="1">

**PoT GPT-4:**

``` python
def fraction_of_blue(description, choices):
    parts = {
        'blue': 1,
        'green': 1,
        'yellow': 1,
        'red': 1
    }

    total_parts = sum(parts.values())
    blue_fraction = parts['blue'] / total_parts

    for choice in choices:
        fraction = choice.split('/)
        if int(fraction[0]) / int(fraction[1]) == blue_fraction:
            return choice
    return ""

choices = ['3/11', '8/11', '6/11', '3/5']
description = "The image shows a blue circle with a white stripe in the middle. The circle is divided into four parts, each with a different color. The colors are blue, green, yellow, and red. There is no text present in the image."

print(fraction_of_blue(description, choices))
```

**Output:**

</div>

<span id="fig:example_540_llm" label="fig:example_540_llm"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/458.jpg" style="height:2.8cm" alt="image" />

**Question:** If you add the two visible numbers, on the jerseys, what
is the total sum?  
  
**Ground truth:**   
  
**Solution:** 2 + 1 = 3

</div>

<div class="mybody" markdown="1">

**IDEFICS:**

</div>

<div class="mybody" markdown="1">

**mPLUG-Owl:**

</div>

<div class="mybody" markdown="1">

**miniGPT4:**

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**

</div>

<div class="mybody" markdown="1">

**LLaVAR:**

</div>

<div class="mybody" markdown="1">

**InstructBLIP:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** The two visible numbers on the jerseys are 2 and 1. .  
**Correct output:**

</div>

<div class="mybody" markdown="1">

**Multomodal Bard:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**

</div>

<div class="mybody" markdown="1">

**Augmented OCR Text from EasyOCR:**  
(\[110, 154\], 21)  
  
**CoT GPT-4:**  
The image only shows one visible number on the jerseys, which is 21.  
  

</div>

<div class="mybody" markdown="1">

**Augmented OCR Text from EasyOCR:**  
(\[110, 154\], 21)  
  
**PoT GPT-4:**

``` python
visible_numbers = [21]

total_sum = sum(visible_numbers)
print(total_sum)
```

**Execution output:**

</div>

<span id="fig:example_458_llm" label="fig:example_458_llm"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/824.png" style="height:2.8cm" alt="image" />

**Question:** How much money does Luca need to buy a sour apple candy
and a butterscotch candy? (Unit: \$)  
  
**Ground truth:**   
  
**Solution:** Add the price of a sour apple candy and the price of a
butterscotch candy: \$0.06 + \$0.07 = \$0.13. Luca needs \$0.13.

</div>

<div class="mybody" markdown="1">

**IDEFICS:** .

</div>

<div class="mybody" markdown="1">

**mPLUG-Owl:**

</div>

<div class="mybody" markdown="1">

**miniGPT4:** The answer to the question is:  
  
  
  
Explanation:  
\* The price of the sour apple candy is .  
\* The price of the butterscotch candy is .  
\* Therefore, to buy both candies.

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVAR:** .

</div>

<div class="mybody" markdown="1">

**InstructBLIP:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The price of a sour apple candy is \$0.06 and the price of a
butterscotch candy is \$0.07. So, .  
  
In two decimal places, .

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  

**Correct output:**

</div>

<span id="fig:example_824" label="fig:example_824"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/824.png" style="height:2.8cm" alt="image" />

**Question:** How much money does Luca need to buy a sour apple candy
and a butterscotch candy? (Unit: \$)  
  
**Ground truth:**   
  
**Solution:** Add the price of a sour apple candy and the price of a
butterscotch candy: \$0.06 + \$0.07 = \$0.13. Luca needs \$0.13.

</div>

<div class="mybody" markdown="1">

**Augmented Caption from Bard:**  
The image shows a table with the prices of different types of candy. The
table is on a pink background. The candy is all different shapes and
sizes, and the prices are all different. The candy is all different
colors, and the prices are all different.  
  
The following text is present in the image:  
  
\* Sour apple candy \$0.06  
\* Piece of gum \$0.07  
\* Gummy worm \$0.09  
\* Lemon drop \$0.05  
\* Piece of licorice \$0.07  
\* Butterscotch candy \$0.07  
  
**Augmented OCR Text from EasyOCR:**  
(\[9, 7\], sour apple candy),  (\[195, 5\], \$0.06),  
(\[7, 39\], piece of gum),  (\[195, 37\], \$0.07),  
(\[9, 73\], gummy worm),  (\[195, 69\], \$0.09),  
(\[7, 105\], lemon drop),  (\[195, 101\], \$0.05),  
(\[6, 132\], piece of licorice),  (\[195, 133\], \$0.07),  
(\[9, 167\], butterscotch candy),  (\[195, 165\], \$0.07)

</div>

<div class="mybody" markdown="1">

**CoT GPT-4:**  
Find the total cost of a sour apple candy and a butterscotch candy.  
  
\$0.06 + \$0.07 = \$0.13  
  
Luca needs \$0.13.  
**Correct output:**

</div>

<div class="mybody" markdown="1">

**PoT GPT-4:**

``` python
sour_apple_candy_price = 0.06
butterscotch_candy_price = 0.07

total_price = sour_apple_candy_price + butterscotch_candy_price
print(total_price)
```

**Execution output:**  
**Correct output:**

</div>

<span id="fig:example_824_llm" label="fig:example_824_llm"></span>

</div>

# A Comparative Study of GPT-4V, Bard, and Other Models [app:gpt4v_study]

GPT-4 with vision (GPT-4V) is the multimodal version of GPT-4 that is
instructed to understand multiple modalities such as texts and images.
Due to its remarkable improvements over other AI models
(§<a href="#sec:results" data-reference-type="ref"
data-reference="sec:results">[sec:results]</a> and
§<a href="#sec:fine_grained_results" data-reference-type="ref"
data-reference="sec:fine_grained_results">[sec:fine_grained_results]</a>),
we have conducted a comprehensive evaluation to understand its
capabilities, strengths, and areas for improvement. Our findings not
only validate GPT-4V’s various problem-solving skills but also shed
light on developing general-purpose multimodal AI agents.

## GPT-4V Playground for Manual Evaluation

Given that GPT-4V does not offer API access, we have performed manual
evaluations using the playground platform[^1]. For a fair comparison, we
used the same input queries as those for all the other LMMs and recorded
the responses in a single round of chat without additional feedback
(Figure <a href="#fig:gpt4v_playground" data-reference-type="ref"
data-reference="fig:gpt4v_playground">1</a>).

<figure id="fig:gpt4v_playground">
<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_playgroud.png" style="width:90.0%" />
<figcaption>The GPT-4V playground for our manual
evaluation.</figcaption>
</figure>

## Leaderboard Scores [sec:gpt4v_leaderboard]

The leaderboard in Figure
<a href="#fig:score_leaderboard" data-reference-type="ref"
data-reference="fig:score_leaderboard">2</a> highlights GPT-4V’s
substantial advancements over the current LLM and LMM baselines.
Notably, there is a 15.1% improvement over the second-best performing
Multimodal Bard model. However, a significant gap of 10.4% still exists
between GPT-4V and human performance, indicating plenty of room for
further improvement by developing new LMMs and tool-augmented LLMs.

<figure id="fig:score_leaderboard">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/score_leaderboard_gpt4v.png" style="width:90.0%" />
<figcaption aria-hidden="true"></figcaption>
</figure>

[^1]: <https://chat.openai.com/>

## Abilities in Mathematical Reasoning [sec:gpt4v_math]

This section compares the mathematical reasoning ability of GPT-4V with
that of other LLMs on , including LLaMA-Adapter-V2-7B (LLaMA-Adapter-V2
for simplification), LLaVA-LLaMA-2-13B (LLaVA for simplification), and
Multimodal Bard.

### Algebraic Reasoning

Algebraic reasoning problems on require understanding the function plots
from figures and inferring their properties. As shown in Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>, GPT-4V
demonstrates outstanding capabilities in algebraic reasoning, surpassing
all competing models and even humans. For instance, GPT-4V accurately
identifies the function plot by its equation and subsequently infers its
correct properties (Figure
<a href="#fig:example_926" data-reference-type="ref"
data-reference="fig:example_926">[fig:example_926]</a>). However, both
GPT-4V and the other LLMs face challenges in comprehending
low-resolution figures (Figure
<a href="#fig:example_346" data-reference-type="ref"
data-reference="fig:example_346">[fig:example_346]</a>) and those that
depict multiple functions (Figure
<a href="#fig:example_963" data-reference-type="ref"
data-reference="fig:example_963">[fig:example_963]</a>).

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/926.png" style="height:2.8cm" alt="image" />

**ID:** 926  
  
**Question:** What is the size of the shaded area under the curve? Round
the answer to 2 decimal places.  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:**

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The given curve is a circle with center at $(3, 0)$ and radius 3. The
shaded area is a quarter of the circle’s area. $$\begin{aligned}
\text{Area of the circle} &= \pi r^2 \\
&= 3.14 \times 3^2 \\
&= 28.26.\\
\text{Area of the shaded region} &= \frac{28.26}{4} \\
&= 7.065.
\end{aligned}$$ Rounding to two decimal places, the area is $7.07$.  
**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
To find the size of the shaded area under the curve
$f(x) = \sqrt{9 - (x - 3)^2}$, we need to integrate the function over
the given interval.  
The given function represents the upper half of a circle with a radius
of 3 and centered at the point (3,0). The shaded region corresponds to
the area of a quarter circle with a radius of 3. The formula for the
area of a circle is $\pi r^2$. For a quarter of the circle, the area is
$\frac{1}{4} \pi r^2$.  
Given: $r = 3$ $$\begin{aligned}
 \text{Area} &= \frac{1}{4} \pi (3^2) \\
 \text{Area} &= \frac{1}{4} \pi (9) \\
 \text{Area} &= \frac{9}{4} \pi \\
 \text{Area} &= 7.07 \text{~(rounded to 2 decimal places)} 
\end{aligned}$$ So, the size of the shaded area under the curve is
approximately $7.07$ square units.  
**Correct output:**

</div>

<span id="fig:example_926" label="fig:example_926"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/346.png" style="height:4.8cm" alt="image" />

**ID:** 346  
  
**Question:** What is the $y$-intercept of this function?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The $y$-intercept of the function is the point where the graph crosses
the $y$-axis. Since the $y$-axis is also the line $x=0$, the
$x$-coordinate of the $y$-intercept will always be 0. In the image, .
Therefore, .  
Here are the steps on how to find the $y$-intercept of a function:  
1. Set the $x$-coordinate to 0.  
2. Solve for the $y$-coordinate.  
In this case, we set $x=0$ and get $y=3$. Therefore, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The $y$-intercept of a function is the point where the graph of the
function crosses the $y$-axis.  
From the provided graph, .  
Thus, .

</div>

<span id="fig:example_346" label="fig:example_346"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/963.png" style="height:4.8cm" alt="image" />

**ID:** 963  
  
**Question:** For what $x$ does $f$ reach its local maximum?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The local maximum of the function is the point where the graph of the
function changes from increasing to decreasing. In the given graph, this
point is at $x = 3$. So the answer is **3**.  
Here is a more detailed explanation:

-   A local maximum point of a function is a point on the graph of the
    function where the function value is higher than all the function
    values in a small neighborhood of the point.

-   . However, for values of $x$ greater than 3, the function value
    starts to decrease. .

-   Therefore, the local maximum point of the function is at $x = 3$.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The function $f(x)$ reaches its local maximum where its first
derivative, $f'(x)$, changes from positive to negative.  
Looking at the graph, .  
So, .

</div>

<span id="fig:example_963" label="fig:example_963"></span>

</div>

### Arithmetic Reasoning

Arithmetic reasoning problems in require accurate fundamental operations
in conjunction with understanding diverse visual contexts. As
illustrated in Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>, GPT-4V
exhibits a significant improvement in arithmetic reasoning compared to
existing models. For instance, some LLMs struggle with basic arithmetic
tasks, such as determining the difference between two values in a bar
chart (Figure <a href="#fig:example_74" data-reference-type="ref"
data-reference="fig:example_74">[fig:example_74]</a>) or computing the
probability based on simple statistical data (Figure
<a href="#fig:example_718" data-reference-type="ref"
data-reference="fig:example_718">[fig:example_718]</a>).

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/74.png" style="height:5.5cm" alt="image" />

**ID:** 74  
  
**Question:** What is the difference of largest and smallest bar?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The largest bar in the image is Iceland with 100%, and the smallest bar
is Cambodia with 52.4%. So the difference between the largest and
smallest bar is 100 - 52.4 = 47.6%.  
However, the question asks for the answer in one decimal place. .  
Therefore, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
From the graph:  
Largest bar (Iceland) = 100%  
Smallest bar (Cambodia) = 52.4%  
Difference = 100% - 52.4% = 47.6%

**Correct output:**

</div>

<span id="fig:example_74" label="fig:example_74"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/718.png" style="height:1.4cm" alt="image" />

**ID:** 718  
  
**Question:** Jeffrey is the proud owner of an eclectic bow tie
collection. He keeps track of how many bow ties he has, and organizes
them by pattern and material. What is the probability that a randomly
selected bow tie is designed with swirls and is made of velvet? Simplify
any fractions.  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
.

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
To determine the probability of a randomly selected bow tie being
designed with swirls and made of velvet, we need to divide the total
number of bow ties with the specific characteristics (swirls and velvet)
by the total number of bow ties in Jeffrey’s collection.  
In the image, .  
So, the probability of a randomly selected bow tie being designed with
swirls and made of velvet is:  

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The probability that a randomly selected bow tie is designed with swirls
and is made of velvet is:
$$(\text{Number of swirls and velvet bow ties}) / (\text{Total number of bow ties})$$

.  
We can find the total number of bow ties by adding the values in all
three columns. This gives us $5 + 3 + 1 + 5 = 14$ bow ties.  
Therefore, the probability that a randomly selected bow tie is designed
with swirls and is made of velvet is:
$$\red{(8) / (14) = 0.57 = 57.14\%}$$

To two decimal places, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
To find the probability that a randomly selected bow tie is designed
with swirls and is made of velvet, we first need to find the total
number of bow ties and then determine how many of these fit the
criteria.  
Total number of bow ties: $$\text{Zigzags (Velvet)} = 5$$
$$\text{Zigzags (Cotton)} = 1$$ $$\text{Swirls (Velvet)} = 3$$
$$\text{Swirls (Cotton)} = 5$$ $$\text{Total} = 5 + 1 + 3 + 5 = 14$$

Number of bow ties that are designed with swirls and are made of velvet
= 3  
Probability = Number of desired outcomes / Total number of possible
outcomes Probability = $3/14$  
Now, converting the fraction to a decimal (rounded to two decimal
places): $3 ÷ 14 = 0.21$

**Correct output:**

</div>

<span id="fig:example_718" label="fig:example_718"></span>

</div>

### Geometry Reasoning

In geometry reasoning, the performance of GPT-4V is comparable to that
of humans on , as demonstrated in Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>. Figure
<a href="#fig:example_606" data-reference-type="ref"
data-reference="fig:example_606">[fig:example_606]</a> and Figure
<a href="#fig:example_640" data-reference-type="ref"
data-reference="fig:example_640">[fig:example_640]</a>, respectively,
present two geometry reasoning problems: one at an elementary level and
the other at a college level. For both problems, GPT-4V produces the
correct answers accompanied by detailed explanations.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/606.png" style="height:2.8cm" alt="image" />

**ID:** 606  
  
**Question:** What is the perimeter of the rectangle?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** The perimeter of the rectangle is .

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** The perimeter of the rectangle is 10 cm.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
I see the image. The rectangle has a width of 3 cm and a length of 2 cm.
.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The perimeter of a rectangle is calculated by adding up all its sides.  
Using the provided dimensions:  
Perimeter = 2 × (length + width)  
Perimeter = 2 × (3 cm + 2 cm)  
Perimeter = 2 × 5 cm  
Perimeter = 10 cm  
The perimeter of the rectangle is 10 cm.

**Correct output:**

</div>

<span id="fig:example_606" label="fig:example_606"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/640.png" style="height:2.4cm" alt="image" />

**ID:** 640  
  
**Question:** Find $\widehat{\mathrm{WN}}$ if $\triangle \mathrm{IWN}$
is equilateral and $W N=5$  
  
**Choices:** (A) $\frac { 3 } { 5 } \pi$ (B) $\frac { 5 } { 3 } \pi$ (C)
$5 \pi$ (D) $10 \pi$  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
Given that $\triangle \text{IWN}$ is equilateral, we can use the cosine
rule to find the length of the side opposite .

where $a$ and $b$ are the lengths of the adjacent sides of the
equilateral triangle. $$a = 5$$ $$b = 5$$

Substituting the values, we get:
$$\red{\cos \angle \text{WN} = \frac{5}{5} = 1}$$

So, the length of the side opposite .

Therefore,

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The correct answer is (B).

Since $\triangle IWN$ is equilateral,
$\widehat{WIN} = \widehat{WNI} = 60^\circ$. Therefore,
$\red{\widehat{WN} = \pi - 2 \cdot 60^\circ} = \boxed{\frac{5}{3}\pi}$.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
For an equilateral triangle, each interior angle is $60^{\circ}$.  
The arc corresponding to a central angle in a circle is given by:
$\text{Arc length} = \text{central angle in radians} \times \text{radius}$  
Given that $WI = IN = NW = 5$ (since $\triangle IWN$ is equilateral),
the radius of the circle is 5.  
Using the formula above:
$\widehat{WN} = 60^{\circ} \times \frac{\pi}{180^{\circ}} \times 5 = \frac{\pi}{3} \times 5 = \frac{5\pi}{3}$  
The correct option is: (B) $\frac{5}{3} \pi$.

**Correct output:**

</div>

<span id="fig:example_640" label="fig:example_640"></span>

</div>

### Logical Reasoning

Logical reasoning problems represent a different type of question in .
Solving these problems requires abstract thinking to deduce the
underlying patterns of numbers or shapes from figures. Current
foundation models struggle to effectively tackle logical reasoning
problems: GPT-4V achieves only 21.6% accuracy in logical reasoning,
which is a modest improvement of 8.1% over random chance, as shown in
Table <a href="#tab:mathvista" data-reference-type="ref"
data-reference="tab:mathvista">[tab:mathvista]</a>. The challenges that
logical reasoning problems present to current LMMs are further
highlighted in Figures
<a href="#fig:example_98" data-reference-type="ref"
data-reference="fig:example_98">[fig:example_98]</a>,
<a href="#fig:example_778" data-reference-type="ref"
data-reference="fig:example_778">[fig:example_778]</a>, and
<a href="#fig:example_750" data-reference-type="ref"
data-reference="fig:example_750">[fig:example_750]</a>.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/98.jpg" style="height:2.5cm" alt="image" />

**ID:** 98  
  
**Question:** Find the value of the square in the figure.  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
Here’s how to solve it:  
1. $5 + 8 = 13$  
2. $13 + 7 = 20$  
3. $20 / 2 = 10$  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
From the given figure: $$\begin{aligned}
\text{Circle} + \text{Square} &= 5 \\
\implies \text{Circle} &= 5 - \text{Square} \quad \dots (i)
\end{aligned}$$ $$\begin{aligned}
\text{Triangle} + \text{Triangle} &= 8 \\
\implies \text{Triangle} &= 4 \quad \dots (ii)
\end{aligned}$$ $$\begin{aligned}
\text{Circle} + \text{Triangle} &= 7 \\
\text{Using (ii),} \quad \text{Circle} + 4 &= 7 \\
\implies \text{Circle} &= 3 \quad \dots (iii)
\end{aligned}$$

Using (i) and (iii), $$\begin{aligned}
5 - \text{Square} &= 3 \\
\implies \text{Square} &= 2.
\end{aligned}$$ **Correct output:**

</div>

<span id="fig:example_98" label="fig:example_98"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/778.png" style="height:2.7cm" alt="image" />

**ID:** 778  
  
**Question:** Can you find the missing shape in this picture puzzle?  
  
**Choices:** (A) A (B) B (C) C (D) D  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The missing shape in the picture puzzle is **(D)**.  
The shapes in the puzzle are all regular polygons, and . , the second
shape is a hexagon, , and . .  
The answer choice A is a pentagon, . . . .  
Therefore, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The pattern seems to involve alternating filled and unfilled shapes.  
Following the pattern:  
- The first shape is a filled .  
- The second shape is an unfilled hexagon.  
Thus, the missing shape should be a filled .  
.

</div>

<span id="fig:example_778" label="fig:example_778"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/750.jpg" style="height:2.8cm" alt="image" />

**ID:** 750  
  
**Question:** Which number is missing?  
  
**Ground truth:**   
  
**Solution:** $5 \times 2 = 10$, $6 \times 2 = 10$, $9 \times 2 = 18$

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The missing number in the puzzle is **7**.  
The numbers in the puzzle are arranged in a circle, and . For example,
.  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
From the given image, : $$\red{12 - 10 = 2}$$ $$\red{9 - 5 = 4}$$
$$\red{6 - ? = 2}$$

.  
.

</div>

<span id="fig:example_750" label="fig:example_750"></span>

</div>

### Numeric Commonsense Reasoning

Problems involving numeric commonsense reasoning on require commonsense
knowledge about daily objects and celebrities to answer visual
questions. However, these problems present significant challenges to
existing foundation models, including GPT-4V, as depicted in Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>. For instance,
Multimodal Bard struggles to understand the optical illusion in an image
(Figure <a href="#fig:example_693" data-reference-type="ref"
data-reference="fig:example_693">[fig:example_693]</a>) and to infer the
age gap between two celebrities from another image (Figure
<a href="#fig:example_29" data-reference-type="ref"
data-reference="fig:example_29">[fig:example_29]</a>). Figure
<a href="#fig:example_19" data-reference-type="ref"
data-reference="fig:example_19">[fig:example_19]</a> poses a question
about the maximum volume a beaker can measure. However, GPT-4V lacks
commonsense knowledge regarding the use of a beaker, resulting in an
incorrect prediction.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/693.jpg" style="height:2.8cm" alt="image" />

**ID:** 693  
  
**Question:** Is the orange larger than the car?  
  
**Choices:** (A) Yes (B) No  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** No, the orange is not larger than the car.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The answer is (B), No. The orange is not larger than the car. The car is
clearly much larger than the orange. Even if it is a large orange, it is
still much smaller than the car. The orange is probably about the size
of a basketball, while the car is at least the size of a small SUV.  
.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The orange is not larger than the car; .

**Correct output:**

</div>

<span id="fig:example_693" label="fig:example_693"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/29.jpg" style="height:3.8cm" alt="image" />

**ID:** 29  
  
**Question:** What is the age gap between these two people in image?
(unit: years)  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
Based on the image you sent, .  
The king is Richard III of England, who was born in 1452. The queen is
Anne Neville, who was born in 1456. So, .  
The stained glass window is likely depicting the couple after their
marriage, so .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The two figures depicted in the stained glass are King Richard III and
Anne Neville. Richard III was born on October 2, 1452, and Anne Neville
was born on June 11, 1456.  
Calculating the age gap: $$1456 - 1452 = 4$$

So, the age gap between Richard III and Anne Neville is 4 years.

**Correct output:**

</div>

<span id="fig:example_29" label="fig:example_29"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/19.jpg" style="height:4.8cm" alt="image" />

**ID:** 19  
  
**Question:** What is the highest amount this glass measures?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** The highest amount this glass measures is 400 ml.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
The markings on the side of the beaker are in milliliters (ml). .  
. The negative markings indicate the amount of liquid that is below the
zero mark.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
.

</div>

<span id="fig:example_19" label="fig:example_19"></span>

</div>

### Scientific Reasoning

Scientific reasoning represents a distinct mathematical reasoning
ability within our . To tackle problems in this area, a model must not
only accurately interpret domain-specific information from figures, but
also possess the necessary in-domain knowledge to reason rigorously on
scientific topics. Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a> shows that
GPT-4V substantially outperforms the other foundation models. This
superiority is further illustrated by the examples in Figures
<a href="#fig:example_807" data-reference-type="ref"
data-reference="fig:example_807">[fig:example_807]</a> and
<a href="#fig:example_278" data-reference-type="ref"
data-reference="fig:example_278">[fig:example_278]</a>. However, the
failure of GPT-4V, as shown in Figure
<a href="#fig:example_725" data-reference-type="ref"
data-reference="fig:example_725">[fig:example_725]</a>, indicates that
there is considerable room for improvement.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/807.png" style="height:3.8cm" alt="image" />

**ID:** 807  
  
**Question:** Which of the following leaf shapes would have the least
amount of wind resistance and water loss?  
  
**Choices:**  
(A) Truncate (B) Acuminate  
(C) Rounded (D) Sagittate  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
. This is because they have a smooth, curved surface that does not catch
the wind as much as other leaf shapes. Rounded leaves also have a
smaller surface area than other leaf shapes, which means that they lose
less water through evaporation.  
. Truncate leaves have a flat end, which can catch the wind. . Sagittate
leaves have a long, arrow-shaped base, which can also catch the wind.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The leaf shape that would have the least amount of wind resistance and
water loss would be the one that is most slender and pointed. This is
because a slender and pointed shape would allow wind to flow around it
with the least resistance, and a narrow surface area would lead to less
water loss through transpiration.  
Among the options, the “acuminate” leaf shape is the most slender and
pointed.

**Correct output:**

</div>

<span id="fig:example_807" label="fig:example_807"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/278.png" style="height:4.8cm" alt="image" />

**ID:** 278  
  
**Question:** Among the following objects, which one has the best PSNR
score?  
  
**Choices:** (A) Lego (B)Mats (C) Mic (D)Ship  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
.

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The answer is **(D) Ship**.  
. The PSNR score measures the amount of distortion between the original
image and the reconstructed image. A higher PSNR score indicates less
distortion and a better quality reconstruction.  
The other objects have the following PSNR scores:

\* Lego: 29.0 dB  
\* Mats: 27.0 dB  
\* Mic: 30.7 dB  
.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Based on the table for PSNR (dB) scores:  
Lego:  
DMTet: 29.0  
FLeXiCubes: 28.8  
Mats:  
DMTet: 27.0  
FLeXiCubes: 26.7  
Mic:  
DMTet: 30.7  
FLeXiCubes: 30.8  
Ship:  
DMTet: 26.0  
FLeXiCubes: 25.9  
Among the choices, “Mic” has the highest PSNR scores with both DMTet and
FLeXiCubes, so the answer is (C) Mic.

**Correct output:**

</div>

<span id="fig:example_278" label="fig:example_278"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/725.png" style="height:4cm" alt="image" />

**ID:** 725  
  
**Question:** In Figure, suppose that Barbara’s velocity relative to
Alex is a constant $v_{B A}=52 \mathrm{~km} / \mathrm{h}$ and car $P$ is
moving in the negative direction of the $x$ axis. If Alex measures a
constant $v_{P A}=-78 \mathrm{~km} / \mathrm{h}$ for car $P$, what
velocity $v_{P B}$ will Barbara measure?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
.

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The velocity of car $P$ relative to Barbara is given by:
$$v_{PB} = v_{PA} - v_{BA} = -78 \mathrm{~km} / \mathrm{h} - 52 \mathrm{~km} / \mathrm{h} = -130 \mathrm{~km} / \mathrm{h}$$

Therefore, Barbara will measure a velocity of **-130**
$\mathrm{~km} / \mathrm{h}$ for car $P$.  
Here is the explanation:  
\* The velocity of car $P$ relative to Alex, $v_{PA}$, is measured by
Alex, who is stationary in frame $A$.  
\* The velocity of Barbara relative to Alex, $v_{BA}$, is measured by
Alex, who is also stationary in frame $A$.  
\* The velocity of car $P$ relative to Barbara, $v_{PB}$, is measured by
Barbara, who is moving in frame $B$ with a velocity of $v_{BA}$.  
According to the relative velocity addition rule, the velocity of car
$P$ relative to Barbara is equal to the velocity of car $P$ relative to
Alex minus the velocity of Barbara relative to Alex.  
In this case, the velocity of car $P$ relative to Alex is -78
$\mathrm{~km} / \mathrm{h}$ and the velocity of Barbara relative to Alex
is 52 $\mathrm{~km} / \mathrm{h}$. Therefore, the velocity of car $P$
relative to Barbara is -130 $\mathrm{~km} / \mathrm{h}$.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Given:  
$v_{BA}$ = 52 $\mathrm{~km} / \mathrm{h}$ (velocity of Barbara relative
to Alex)  
$v_{PA}$ = -78 $\mathrm{~km} / \mathrm{h}$ (velocity of car $P$ as
measured by Alex)  
We need to find $v_{PB}$ (velocity of car $P$ as measured by Barbara).  
Using the Galilean transformation for velocities in one dimension:
$$\begin{aligned}
v_{PB} &= \red{v_{PA} + v_{BA}} \\
v_{PB} &= \red{-78 \text{ $\mathrm{~km} / \mathrm{h}$} + 52 \text{ $\mathrm{~km} / \mathrm{h}$}} \\
v_{PB} &= \red{-26 \text{ $\mathrm{~km} / \mathrm{h}$}}
\end{aligned}$$

</div>

<span id="fig:example_725" label="fig:example_725"></span>

</div>

### Statistical Reasoning

In , problems encompass a variety of charts, plots, and graphs designed
to assess the statistical reasoning capabilities of foundation models.
As demonstrated in Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>, GPT-4V shows
strong statistical reasoning ability. For instance, GPT-4V produces
accurate answers for the format-rich table in Figure
<a href="#fig:example_649" data-reference-type="ref"
data-reference="fig:example_649">[fig:example_649]</a> and the data
analysis table in Figure
<a href="#fig:example_389" data-reference-type="ref"
data-reference="fig:example_389">[fig:example_389]</a>.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/649.png" style="height:7.8cm" alt="image" />

**ID:** 649  
  
**Question:** How many values are below 30 in Mainly are incidents of
individual misconduct?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
. , which is the percentage of women who believe that recent allegations
of sexual harassment and assault are mainly incidents of individual
misconduct.  
.  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Based on the “Mainly are incidents of individual misconduct” column, the
following values are below 30:  
- 28 (Total)  
- 22 (Women)  
- 19 (College grad+)  
- 25 (Dem/Lean Dem)  
That’s a total of 4 values.  
So, the answer is 4.

**Correct output:**

</div>

<span id="fig:example_649" label="fig:example_649"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/math/389.png" style="height:2.8cm" alt="image" />

**ID:** 389  
  
**Question:** Look at the table. Then answer the question. At a price of
\$320, is there a shortage or a surplus?  
  
**Choices:** (A) shortage (B) surplus  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
At a price of \$320, .  
Here is a table summarizing the supply and demand at each price:

<div class="center" markdown="1">

|       |        |     |     |
|:-----:|:------:|:---:|:---:|
| \$270 |        |     |     |
| \$320 |        |     |     |
| \$370 |        |     |     |
| \$420 | 19,600 |     |     |
| \$470 |        |     |     |

</div>

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
At a price of \$320:  
Quantity demanded = 20,000  
Quantity supplied = 12,500  
Since the quantity demanded (20,000) is greater than the quantity
supplied (12,500), there is a shortage.

**Correct output:**

</div>

<span id="fig:example_389" label="fig:example_389"></span>

</div>

## Abilities Across Visual Contexts [sec:gpt4v_visual]

This section compares the reasoning abilities of GPT-4V with other large
multimodal models (LLMs) on , considering various types of visual
contexts. Models used for comparison include LLaMA-Adapter-V2-7B
(simplified as LLaMA-Adapter-V2), LLaVA-LLaMA-2-13B (simplified as
LLaVA), and Multimodal Bard.

### Abstract Scene

Based on Figure <a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>, current
foundation models lag behind human performance in mathematical reasoning
in abstract scenes by a substantial margin. Consider the problems in
Figures <a href="#fig:example_742" data-reference-type="ref"
data-reference="fig:example_742">[fig:example_742]</a> and
<a href="#fig:example_7" data-reference-type="ref"
data-reference="fig:example_7">[fig:example_7]</a> that are derived from
math word problems found in elementary school curricula. Despite their
advanced capabilities, foundation models such as Multimodal Bard and
GPT-4V fail to produce the correct responses.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/742.png" style="height:1.8cm" alt="image" />

**ID:** 742  
  
**Question:** Move the ruler to measure the length of the twig to the
nearest inch. The twig is about ( ) inches long.  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
According to the image, . . Therefore, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:** .

</div>

<span id="fig:example_742" label="fig:example_742"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/7.png" style="height:2.8cm" alt="image" />

**ID:** 7  
  
**Question:** What time is shown? Answer by typing a time word, not a
number. It is ( ) after eight.  
  
**Choices:** (A) half (B) quarter (C) o’clock (D) quarter to (E) quarter
past  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The image you sent shows a clock with the hands pointing to 8:15.  
The hour hand is between 8 and 9, which means it is past 8 o’clock. The
minute hand is pointing to the 3, which is 15 minutes past. Therefore,
the time is quarter past eight.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
, which indicates that it is 45 minutes past the hour. The hour hand is
pointing a little past the 8, which means it’s after 8 o’clock.  
  
So,

The correct answer is .

</div>

<span id="fig:example_7" label="fig:example_7"></span>

</div>

### Bar Chart

As shown in Figure <a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>, foundation
models, including GPT-4V, significantly underperform humans in
mathematical reasoning when bar charts serve as the visual context.
Neither Multimodal Bard nor GPT-4V can solve the problems depicted in
Figures <a href="#fig:example_497" data-reference-type="ref"
data-reference="fig:example_497">[fig:example_497]</a> and
<a href="#fig:example_629" data-reference-type="ref"
data-reference="fig:example_629">[fig:example_629]</a>, which do not
need complex understanding and reasoning.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/497.png" style="height:4.5cm" alt="image" />

**ID:** 497  
  
**Question:** What is the value of the largest bar?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:** .

</div>

<div class="mybody" markdown="1">

**GPT-4V:** .

</div>

<span id="fig:example_497" label="fig:example_497"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/629.png" style="height:4.8cm" alt="image" />

**ID:** 629  
  
**Question:** What percentage of people prefer the object hut?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:** Based on the image, . .

</div>

<div class="mybody" markdown="1">

**GPT-4V:** .

</div>

<span id="fig:example_629" label="fig:example_629"></span>

</div>

### Function Plot

GPT-4V outperforms other baselines on problems related to function plots
and even exceeds human performance. Figures
<a href="#fig:example_767" data-reference-type="ref"
data-reference="fig:example_767">[fig:example_767]</a> and
<a href="#fig:example_39" data-reference-type="ref"
data-reference="fig:example_39">[fig:example_39]</a> show questions with
digital and hand-drawn function plots, respectively. In both cases,
GPT-4V accurately identifies their functions and infers the correct
properties.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/767.png" style="height:3.8cm" alt="image" />

**ID:** 767  
  
**Question:** What is the maximum value of $y$?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
. This is the equation of a circle with center $(0, 0)$ and . .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The equation given is for a circle: $x^2 + y^2 = 25$.  
The radius of this circle is the square root of 25, which is 5.  
The maximum value of $y$ corresponds to the highest point of the circle,
which is equal to the radius.  
So, the maximum value of $y$ is 5.

**Correct output:**

</div>

<span id="fig:example_767" label="fig:example_767"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/39.png" style="height:3.2cm" alt="image" />

**ID:** 39  
  
**Question:** Is this function odd or even?  
  
**Choices:** (A) odd (B) even  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** The function is odd.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** The function in the image is odd.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The function in the image is **odd**.  
An even function is a function where $f(-x) = f(x)$. An odd function is
a function where $f(-x) = -f(x)$.  
, if we substitute $x = -x$, we get: $$\begin{aligned}
\red{y} &= \red{a_2(-x) - a_1} \\
&= \red{-a_2x + a_1}
\end{aligned}$$

This is not equal to $f(x)$, so the function is **odd**.  
The correct choice is (A).

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The given function is $y = a^x$ where $a > 1$.  
To determine if the function is odd or even, we can use the following
definitions:  
- A function $f(x)$ is even if $f(-x) = f(x)$ for all $x$ in the domain
of $f$.  
- A function $f(x)$ is odd if $f(-x) = -f(x)$ for all $x$ in the domain
of $f$.  
For the given function $y = a^x$, when we plug in $-x$ for $x$:

$$y = a^{-x}$$

This is not equal to $-a^x$ and it’s not equal to $a^x$ either.  
Thus, the function is neither odd nor even. However, since the given
choices only include “odd” and “even”, the best answer is neither. But
since I must choose one of the provided options (A) odd.

**Correct output:**

</div>

<span id="fig:example_39" label="fig:example_39"></span>

</div>

### Geometry Diagram

Geometry diagrams are a distinct type of visual context in . To answer
questions involving these diagrams, a model must comprehend the
fine-grained details, including symbols, variables, and relations from
the figures. Additionally, it should apply associated theorems before
executing calculations to produce final responses. GPT-4V surpasses
other models and even humans due to its superior capabilities in
geometry recognition and reasoning. In the examples shown in Figures
<a href="#fig:example_478" data-reference-type="ref"
data-reference="fig:example_478">[fig:example_478]</a> and
<a href="#fig:example_410" data-reference-type="ref"
data-reference="fig:example_410">[fig:example_410]</a>, GPT-4V delivers
the correct results through the application of relevant theorems and
subsequent calculations.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/478.jpg" style="height:2cm" alt="image" />

**ID:** 478  
  
**Question:** As shown in the figure, in the diamond $ABCD$, two
diagonal lines $AC = 12.0$, $BD = 16.0$, then the edge length of this
diamond is ().  
  
**Choices:** (A) 10 (B) 8 (C) 6 (D) 5  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The answer is (A). 10.  
In a rhombus, the diagonals bisect each other at right angles. So, we
can draw two right triangles within the rhombus, each with a leg of
length 6 and a hypotenuse of length 10.  
By the Pythagorean Theorem, .  
So the answer is (A).

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Given that $AC = 12.0$ and $BD = 16.0$ in the rhombus (diamond)
$ABCD$.  
Since all sides of a rhombus are congruent, the diagonals of a rhombus
are perpendicular bisectors of each other. This means that each half of
the diagonal $BD$ is 8.0 and each half of diagonal $AC$ is 6.0.  
Using the Pythagorean theorem for one of the right triangles (formed by
the diagonals): $$\begin{aligned}
\text{Side}^2 &= \left(\text{Half of $BD$}\right)^2 + \left(\text{Half of $AC$}\right)^2 \\
\text{Side}^2 &= 8^2 + 6^2 \\
\text{Side}^2 &= 64 + 36 \\
\text{Side}^2 &= 100 \\
\text{Side} &= 10
\end{aligned}$$ **Correct output:**

</div>

<span id="fig:example_478" label="fig:example_478"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/410.png" style="height:2cm" alt="image" />

**ID:** 410  
  
**Question:** Find the perimeter of $\triangle D E F,$ if
$\triangle D E F \sim \triangle C B F,$ perimeter of
$\triangle C B F=27, D F=6,$ and $F C=8$.  
  
**Choices:** (A) 20.25 (B) 21 (C) 27 (D) 36  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** The perimeter of $\triangle D E F$ is 20.25, as
shown in the image.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
The perimeter of $\triangle D E F$ can be calculated using the
similarity theorem, which states that if two triangles are similar,
their corresponding sides are proportional. In this case, we know that
$\triangle D E F \sim \triangle C B F$ and the perimeter of
$\triangle C B F$ is 27.  
Since the ratio of the corresponding sides of similar triangles is the
same, we can use the perimeter of $\triangle C B F$ to find the
perimeter of $\triangle D E F$. The perimeter of $\triangle D E F$ is
proportional to the perimeter of $\triangle C B F$, so we can multiply
the perimeter of $\triangle C B F$ by the ratio of the corresponding
sides.  
.  
Therefore, .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
Since $\triangle DEF \sim \triangle CBF$, we have
$\frac{DE}{CB} = \frac{DF}{FC}$. Since $DE = 6$ and $FC = 8$, we have
$\frac{DE}{CB} = \frac{6}{8} = \frac{3}{4}$.  
. Substituting $\frac{3}{4}$ for $\frac{CB}{BF}$, .  
The perimeter of $\triangle DEF$ is
$DE + EF + DF = \red{6 + EF + 6 = EF + 12}$. Substituting $\frac{3}{4}$
for $\frac{EF}{BF}$, .  
Therefore, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Given $\triangle D E F \sim \triangle C B F$ and $DF = 6$ and $FC = 8$,
we can find the scale factor of the two similar triangles.  
Scale factor = $\frac{DF}{FC} = \frac{6}{8} = \frac{3}{4}$.  
Given the perimeter of $\triangle CBF = 27$, we can find the perimeter
of $\triangle DEF$ by multiplying by the scale factor.  
Perimeter of $\triangle DEF$ = $27 \times \frac{3}{4} = 20.25$.

**Correct output:**

</div>

<span id="fig:example_410" label="fig:example_410"></span>

</div>

### Line Plot

As evidenced by Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>, current models
such as GPT-4V do not perform as well as humans in mathematical
reasoning involving line plots. We speculate that the low performance is
mainly due to the difficulty in detecting OCR text in the figures and
accurately grounding the values, as illustrated by the examples in
Figures <a href="#fig:example_981" data-reference-type="ref"
data-reference="fig:example_981">[fig:example_981]</a> and
<a href="#fig:example_209" data-reference-type="ref"
data-reference="fig:example_209">[fig:example_209]</a>.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/981.png" style="height:5.5cm" alt="image" />

**ID:** 981  
  
**Question:** What is the sum of 2002, 2003 and 2004?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
I see the image.  
. .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The sum for the years 2002, 2003, and 2004 from the graph are:  
2002: 19.4%  
2003: 23.2%  
  

</div>

<span id="fig:example_981" label="fig:example_981"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/209.png" style="height:5.5cm" alt="image" />

**ID:** 209  
  
**Question:** What is the highest value in black line chart?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:** .

</div>

<div class="mybody" markdown="1">

**GPT-4V:** .

</div>

<span id="fig:example_209" label="fig:example_209"></span>

</div>

### Natural Image

includes questions that require numeric and spatial reasoning based on
text and objects in natural images. If models have limited abilities to
recognize text (OCR), as shown in Figure
<a href="#fig:example_980" data-reference-type="ref"
data-reference="fig:example_980">[fig:example_980]</a>, or to identify
visual objects, as in Figure
<a href="#fig:example_884" data-reference-type="ref"
data-reference="fig:example_884">[fig:example_884]</a>, they are
unlikely to generate correct answers to visual questions.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/980.jpg" style="height:5.8cm" alt="image" />

**ID:** 980  
  
**Question:** What is the highest number shown?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** The highest number shown is 12.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:** I see the image. .

</div>

<div class="mybody" markdown="1">

**GPT-4V:** The highest number shown on the clock is 12.

**Correct output:**

</div>

<span id="fig:example_980" label="fig:example_980"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/884.jpg" style="height:4.8cm" alt="image" />

**ID:** 884  
  
**Question:** What is the blue kite in the lower right corner shaped
like?  
  
**Choices:**  
(A) ferret (B) cat (C) cloud (D) octopus  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:** .

</div>

<div class="mybody" markdown="1">

**GPT-4V:** The blue kite in the lower right corner is shaped like (D)
octopus.

**Correct output:**

</div>

<span id="fig:example_884" label="fig:example_884"></span>

</div>

### Puzzle Test

Math reasoning with puzzle text figures is challenging for current AI
foundation models because interpreting these figures requires discerning
underlying patterns from sets of shapes, as illustrated in Figure
<a href="#fig:example_123" data-reference-type="ref"
data-reference="fig:example_123">[fig:example_123]</a>, and numbers, as
in Figure <a href="#fig:example_941" data-reference-type="ref"
data-reference="fig:example_941">[fig:example_941]</a>. There is plenty
of room for improvement.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/123.png" style="height:5.8cm" alt="image" />

**ID:** 123  
  
**Question:** The diagram represents successive rotations, starting from
the top down. Which shape comes next?  
  
**Choices:** (A) A (B) B (C) C (D) D (E) E  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
The diagram represents successive rotations, starting from the top down.
The next shape that comes next is letter D.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
.  
To find the next shape, we need to rotate the cube 90 degrees on the
z-axis. .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The diagram shows a pattern of rotating the shape in a specific
sequence. To determine the next shape:  
Starting with the first shape, when we rotate it 90 degrees
counterclockwise, we get the second shape.  
Rotating the second shape 90 degrees counterclockwise gives the third
shape.  
Thus, rotating the third shape 90 degrees counterclockwise will give us
the next shape.  
.

</div>

<span id="fig:example_123" label="fig:example_123"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/941.png" style="height:4.8cm" alt="image" />

**ID:** 941  
  
**Question:** Which is the next number in the series?  
  
**Ground truth:**   
  
**Solution:**  
$(3968 + 1) = 63^2$  
$(63 + 1) = 8^2$  
$(8 + 1) = 3^2$  
$(3 + 1) = 2^2$  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
The series is based on the square root of the number plus 1. So, . Then,
63 + 1 = 64, and its square root is 8. Continuing the pattern, 8 + 1 =
9, and its square root is 3.  
Therefore, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The pattern in the ladder involves dividing the number by the step
number to get the next number.  
$\text{Step 1:}  \quad \red{3968 \div 1 = 3968}$  
$\text{Step 2:}  \quad \red{3968 \div 2 = 1984}$  
$\text{Step 3:}  \quad \red{63 \div 3 = 21}$  
$\text{Step 4:}  \quad \red{8 \div 4 = 2}$  
$\text{Step 5:}  \quad \red{3 \div 5 = 0.6}$  
.  
However, if we assume : $$\begin{aligned}
\red{3968 \div 63} & \red{\approx 63} \\
\red{63 \div 8} & \red{= 7.875 \text{ which isn't a round number.}} \\
\red{8 \div 3} & \red{= 2.6666 \text{ again not a round number.}}
\end{aligned}$$

Without a clear and consistent pattern, it’s challenging to determine
the next number in the sequence.

</div>

<span id="fig:example_941" label="fig:example_941"></span>

</div>

### Scatter Plot

A scatter plot is a graphical representation of data points on a
two-dimensional axis, where each point represents a value from the
dataset. includes the reasoning task that requires comprehending scatter
plots taken from daily life and academic papers, as shown in Figures
<a href="#fig:example_105" data-reference-type="ref"
data-reference="fig:example_105">[fig:example_105]</a> and
<a href="#fig:example_542" data-reference-type="ref"
data-reference="fig:example_542">[fig:example_542]</a>. Although GPT-4V
outperforms other LMMs, such as Multimodal Bard, and even humans in
overall accuracy (Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>), it often
fails in the cases where fine-grained understanding is required, as in
Figure <a href="#fig:example_542" data-reference-type="ref"
data-reference="fig:example_542">[fig:example_542]</a>.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/105.png" style="height:5cm" alt="image" />

**ID:** 105  
  
**Question:** Does Dark Violet have the minimum area under the curve?  
  
**Choices:** (A) yes (B) no  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Based on the scatter plot provided, “area under the curve” is not
represented since there is no curve shown. However, if by “area under
the curve” we mean the overall distribution and position of the points
for "Dark Violet" compared to others, we can analyze the positions of
the points.  
From the scatter plot, the color “Dark Violet” consistently appears to
be the lowest on the y-axis across the entire range of the x-axis.  
Given the choices: (A) yes (B) no  
The correct answer is: (A) yes.

**Correct output:**

</div>

<span id="fig:example_105" label="fig:example_105"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/542.png" style="height:7.8cm" alt="image" />

**ID:** 542  
  
**Question:** How many models in the figure achieve an Acc score greater
than 60?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:** There are two models in the figure that achieve an Acc score
greater than 60.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
. These models are:  
\* GPT-4  
\*  
\*  
\*  
\*  
\*  
\*  
\*  
\* LLAMA2-70B  
The Acc score is the accuracy of the model, which is the percentage of
test cases that the model correctly classifies. An Acc score of 60 means
that the model correctly classifies 60% of the test cases.

</div>

<div class="mybody" markdown="1">

**GPT-4V:** .

</div>

<span id="fig:example_542" label="fig:example_542"></span>

</div>

### Scientific Scene

Answering questions based on scientific scenes poses a challenge in
aligning the scientific concepts present in the question text and those
in the accompanying figures. GPT-4V demonstrates its superior ability to
reason about scientific scenes compared to other models, as evidenced in
Figure <a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>. In the example
of Figure <a href="#fig:example_685" data-reference-type="ref"
data-reference="fig:example_685">[fig:example_685]</a>, GPT-4V adeptly
identifies two organisms in the food web and elucidates their
relationship. In another instance, shown in Figures
<a href="#fig:example_862" data-reference-type="ref"
data-reference="fig:example_862">[fig:example_862]</a> and
<a href="#fig:example_862_2" data-reference-type="ref"
data-reference="fig:example_862_2">[fig:example_862_2]</a>, both
Multimodal Bard and GPT-4V are able to use knowledge in the physical
domain to effectively ground the symbols and variables depicted in the
figure.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/685.png" style="height:3.2cm" alt="image" />

**ID:** 685  
  
**Question:** What would happen to organism c if organism b increased?  
  
**Choices:** (A) decrease (B) increase  
(C) can’t predict (D) stay same  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
. This is because organism c is dependent on organism b for its food
supply, and .

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
. . As a result, .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
In the food chain, . This means that . .  
.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
If Organism B (which appears to be some kind of insect) increased in
population, it would mean that there is more available food for Organism
C (the fish).  
Therefore, the population of Organism C would likely: (B) increase.

**Correct output:**

</div>

<span id="fig:example_685" label="fig:example_685"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/862.png" style="height:3.8cm" alt="image" />

**ID:** 862  
  
**Question:** One of the most dramatic videos on the web (but entirely
fictitious) supposedly shows a man sliding along a long water slide and
then being launched into the air to land in a water pool. Let’s attach
some reasonable numbers to such a flight to calculate the velocity with
which the man would have hit the water. Figure indicates the launch and
landing sites and includes a superimposed coordinate system with its
origin conveniently located at the launch site. From the video we take
the horizontal flight distance as $D=20.0 \mathrm{~m}$, the flight time
as $t=2.50 \mathrm{~s}$, and the launch angle as
$\theta_0=40.0^{\circ}$. Find the magnitude of the velocity at launch
and at landing.  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
Given the following information:  
\* Horizontal flight distance, $D = 20.0 \text{ m}$  
\* Flight time, $t = 2.50 \text{ s}$  
\* Launch angle, $\theta_0 = 40.0^{\circ}$  
We can find the magnitude of the velocity at launch using the following
equation:

$$v_0 = \sqrt{v_{0x}^2 + v_{0y}^2}$$

where $v_{0x}$ is the horizontal component of the launch velocity and
$v_{0y}$ is the vertical component of the launch velocity.  
The horizontal component of the launch velocity can be found using the
following equation:

$$v_{0x} = \frac{D}{t} = \frac{20.0 \text{ m}}{2.50 \text{ s}} = 8.00 \text{ m/s}$$

The vertical component of the launch velocity can be found using the
following equation:

$$v_{0y} = v_0 \sin \theta_0 = 8.00 \text{ m/s} \sin 40.0^{\circ} = 5.83 \text{ m/s}$$

Therefore, the magnitude of the velocity at launch is:

$$v_0 = \sqrt{v_{0x}^2 + v_{0y}^2} = \sqrt{(8.00 \text{ m/s})^2 + (5.83 \text{ m/s})^2} = 10.44 \text{ m/s} \approx 10.4 \text{ m/s}$$

The magnitude of the velocity at landing will be the same as the
magnitude of the velocity at launch, since there is no air resistance in
this case.  
.

</div>

<span id="fig:example_862" label="fig:example_862"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/862.png" style="height:3.8cm" alt="image" />

**ID:** 862  
  
**Question:** One of the most dramatic videos on the web (but entirely
fictitious) supposedly shows a man sliding along a long water slide and
then being launched into the air to land in a water pool. Let’s attach
some reasonable numbers to such a flight to calculate the velocity with
which the man would have hit the water. Figure indicates the launch and
landing sites and includes a superimposed coordinate system with its
origin conveniently located at the launch site. From the video we take
the horizontal flight distance as $D=20.0 \mathrm{~m}$, the flight time
as $t=2.50 \mathrm{~s}$, and the launch angle as
$\theta_0=40.0^{\circ}$. Find the magnitude of the velocity at launch
and at landing.  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Given:  
- $D = 20.0$ m  
- $t = 2.50$ s  
- $\theta_0 = 40.0^{\circ}$  
Using kinematic equations, the horizontal motion is described as:

$$D = V_{0x} \times t$$

Where $V_{0x}$ is the horizontal component of the initial velocity.
$$V_{0x} = V_0 \times \cos(\theta_0)$$

From the first equation: $$V_0 \times \cos(\theta_0) = \frac{D}{t}$$

Rearranging for $V_0$: $$V_0 = \frac{D}{t \times \cos(\theta_0)}$$

Substitute the given values to get $V_0$:

$$V_0 = \frac{20.0 \mathrm{~m}}{2.50 \mathrm{~s} \times \cos(40.0^{\circ})}$$

Using a calculator:

$$V_0 = \frac{20.0}{2.50 \times 0.766}$$ $$V_0 = \frac{20.0}{1.915}$$
$$V_0 = 10.44 \mathrm{~m/s}$$

Thus, the magnitude of the launch velocity (which will be the same as
the landing velocity, ignoring air resistance) is $10.44 \mathrm{~m/s}$.

**Correct output:**

</div>

<span id="fig:example_862_2" label="fig:example_862_2"></span>

</div>

### Synthetic Scene

Problems involving synthetic scenes require a nuanced understanding of
visual objects, such as the numbers, attributes, and positions of these
objects, as shown in Figures
<a href="#fig:example_52" data-reference-type="ref"
data-reference="fig:example_52">[fig:example_52]</a> and
<a href="#fig:example_177" data-reference-type="ref"
data-reference="fig:example_177">[fig:example_177]</a>. Although GPT-4V
demonstrates notable advancements over other models, such as Multimodal
Bard, it still falls significantly short of human performance, as shown
in Figure <a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/52.png" style="height:4cm" alt="image" />

**ID:** 52  
  
**Question:** Subtract all big red metallic spheres. Subtract all big
brown matte things. How many objects are left?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
.

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
.

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
:  
\* 1 big red metallic sphere  
\* 1 big brown matte cube  
\* 1 small red metallic sphere  
\* 1 small brown matte sphere  
:  
\* 1 big brown matte cube  
\* 1 small red metallic sphere  
\* 1 small brown matte sphere  
:

\* 1 small red metallic sphere  
\* 1 small brown matte sphere  
Therefore, .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
There are no big red metallic spheres in the image. There is 1 big brown
matte object (the cube).  
Total objects to be subtracted = 0 + 1 = 1  
Total objects in the image = 7  
Objects left = 7 - 1 = 6  
There are 6 objects left.

**Correct output:**

</div>

<span id="fig:example_52" label="fig:example_52"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/177.png" style="height:4cm" alt="image" />

**ID:** 177  
  
**Question:** Is the number of brown tandem bikes that are to the left
of the small blue matte car greater than the number of tiny blue
biplanes?  
  
**Choices:** (A) Yes (B) No  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
Yes, the number of brown tandem bikes that are to the left of the small
blue matte car is greater than the number of tiny blue biplanes.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
.

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
. , and . .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Looking at the image, there is 1 brown tandem bike to the left of the
small blue matte car. There are no tiny blue biplanes in the image.  
Thus, the number of brown tandem bikes that are to the left of the small
blue matte car is (A) Yes greater than the number of tiny blue biplanes.

**Correct output:**

</div>

<span id="fig:example_177" label="fig:example_177"></span>

</div>

### Table

Tables serve as a powerful tool to present and summarize large amounts
of data in a comprehensible manner. In particular, GPT-4V has shown
significant advancements over other foundation models and even surpasses
human performance on table-related reasoning tasks, as shown in Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>. The example in
Figure <a href="#fig:example_923" data-reference-type="ref"
data-reference="fig:example_923">[fig:example_923]</a> shows a complex
table taken from an academic paper. GPT-4V can accurately pinpoint the
target cells among numerous rows and columns. Figure
<a href="#fig:example_265" data-reference-type="ref"
data-reference="fig:example_265">[fig:example_265]</a> shows a QA task
in which the answer needs to be derived from the table regarding the
push-up competition. GPT-4V is the only model that can produce the
correct answer.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/923.png" style="height:5.3cm" alt="image" />

**ID:** 923  
  
**Question:** Which model achieves the highest score in terms of Rec?  
  
**Choices:**  
(A) Transformers Agent (GPT-4)  
(B) LLaMA-Adapter v2-7B  
(C) LLaVA-7B  
(D) Otter-9B  
(E) MM-ReAct-GPT-3.5  
(F) LLaVA-13B (LLaMA-2)  
(G) MM-ReAct-GPT-4  
  
**Ground truth:**  
  
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:** .

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
.

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The answer is **(G) MM-ReAct-GPT-4**.  
MM-ReAct-GPT-4 achieves the highest score of **33.1** in terms of Rec,
which is the ability to recognize and understand text. The other models
have the following scores in terms of Rec:  
\* Transformers Agent (GPT-4): 18.2  
\* LLaMA-Adapter v2-7B: 16.8  
\* LLaVA-7B: 28.0  
\* Otter-9B: 28.4  
\* MM-ReAct-GPT-3.5: 24.2  
\*

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The model that achieves the highest score in terms of Rec is “LLaVA-13B
(LLaMA-2)” with a score of 39.2%. The correct option is: (F) LLaVA-13B
(LLaMA-2)

**Correct output:**

</div>

<span id="fig:example_923" label="fig:example_923"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/265.png" style="height:2.2cm" alt="image" />

**ID:** 265  
  
**Question:** Justine’s P.E. class participated in a push-up
competition, and Justine wrote down how many push-ups each person could
do. How many people did at least 60 push-ups? (Unit: people)  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
Justine’s P.E. class participated in a push-up competition, and Justine
wrote down how many push-ups each person could do. . .

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
.

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
There are 4 people who did at least 60 push-ups.  
, so .  
. .

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
For the stem-and-leaf plot provided:  
6 represents the tens value, and the digits 3, 6, 7, and 7 represent the
ones value. So, the number of push-ups represented by the 6 row are 63,
66, 67, and 67.  
7 represents the tens value, and the digits 1, 3, 3, 5, 7, 7, and 8
represent the ones value. So, the number of push-ups represented by the
7 row are 71, 73, 73, 75, 77, 77, and 78.  
Counting these numbers: 4 from the 6 row + 7 from the 7 row = 11.

**Correct output:**

</div>

<span id="fig:example_265" label="fig:example_265"></span>

</div>

### Other Visual Contexts

On the reasoning tasks using *other* visual contexts, GPT-4V achieves a
higher overall accuracy than all the other models, as depicted in Figure
<a href="#fig:tease_scores" data-reference-type="ref"
data-reference="fig:tease_scores">[fig:tease_scores]</a>. For instance,
GPT-4V is the only model that is capable of generating the correct
answer to the question regarding a violin plot, as shown in Figure
<a href="#fig:example_895" data-reference-type="ref"
data-reference="fig:example_895">[fig:example_895]</a>.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/visual/895.png" style="height:4.3cm" alt="image" />

**ID:** 895  
  
**Question:** Which model has the lowest Audio-Audio Similarity and
Text-Audio Similarity scores overall?  
  
**Choices:**  
(A) MusicLDM (mix-up)  
(B) MusicLDM (original)  
(C) MusicLDM (BLM)  
(D) MusicLDM (BAM)  
(E) MuBERT  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**LLaMA-Adapter-V2:**  
.

</div>

<div class="mybody" markdown="1">

**LLaVA:**  
Based on the image, .

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
.  
The violin plot shows that the distribution of audio-audio similarity
scores for MusicLDM (BLM) is the most spread out, indicating that there
is a lot of variation in the scores. The distribution of text-audio
similarity scores for MusicLDM (BLM) is also the most spread out. This
suggests that MusicLDM (BLM) is not very consistent in its ability to
capture the similarity between audio and text.

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The model with the lowest Audio-Audio Similarity and Text-Audio
Similarity scores overall is MuBERT.  
**Correct output:**

</div>

<span id="fig:example_895" label="fig:example_895"></span>

</div>

## Self-Verification in GPT-4V [app:gpt4v_self]

Self-verification is a social psychological theory asserting that people
desire others to perceive them as they see themselves. Consequently,
individuals will take active measures to ensure that others view them in
ways that confirm their stable self-concepts
[talaifar2020self](http://arxiv.org/pdf/2405.14152v1).

Interestingly, in our experiments, GPT-4V demonstrates an ability
similar to *self-verification*. The model can inspect its own behaviors
during the course of reasoning and can take active actions to correct
its mistakes. Note that self-verification we discuss here differs from
several recent works on improving the model’s outputs based on external
feedback [peng2023check](http://arxiv.org/pdf/2302.12813v3) or additional generations
[thedawn2023yang](None). The examples in Figures
<a href="#fig:example_6" data-reference-type="ref"
data-reference="fig:example_6">[fig:example_6]</a> and
<a href="#fig:example_28" data-reference-type="ref"
data-reference="fig:example_28">[fig:example_28]</a> show that GPT-4V,
on its own, can inspect a set of candidate answers and identify the one
that is valid and meets all the given constraints. The multi-step
reasoning example in Figure
<a href="#fig:example_615" data-reference-type="ref"
data-reference="fig:example_615">[fig:example_615]</a> shows that GPT-4V
can verify the validity of (the result of) each reasoning step, and
explore alternative approaches if any invalid (intermediate) result is
detected (*e.g.,* a negative value for length).

Although self-verification does not guarantee an accurate response even
after multiple tries, especially when applying GPT-4V to visual
perception or mathematical reasoning in intricate scenarios (see Figure
<a href="#fig:example_644" data-reference-type="ref"
data-reference="fig:example_644">[fig:example_644]</a>), it is
instrumental in improving the model performance on . We also found that
GPT-4V’s self-verification is weaker for non-English tasks, such as
Mandarin, as shown in Figure
<a href="#fig:example_32" data-reference-type="ref"
data-reference="fig:example_32">[fig:example_32]</a>. It is also worth
noting that self-verification does not emerge in other foundation models
we studied, or at least it is not as robust as that of GPT-4V. As shown
in Figure <a href="#fig:example_44" data-reference-type="ref"
data-reference="fig:example_44">[fig:example_44]</a>, Multimodal Bard
first attempts a natural language solution, followed by a
program-assisted one for verification. However, the program-aided
solution leads to a different and incorrect prediction.

The emergent ability of self-verification highlights GPT-4V’s potential
in solving rigorous reasoning and theorem-proving tasks. One of the most
exciting research topics for future work is to develop a mechanism that
allows the model to activate self-verification consistently at the right
time and to use a set of alternative approaches that maximize the
success rate of task completion.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/self/615.png" style="height:1.8cm" alt="image" />

**ID:** 615  
  
**Question:** If two sides of a triangle measure 12 and 7, which of the
following cannot be the perimeter of the triangle?  
  
**Choices:** (A) 29 (B) 34 (C) 37 (D) 38  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
The sum of the lengths of any two sides of a triangle must be greater
than the length of the third side. Let $x$ be the length of the third
side.  
From the triangle inequality: $$12 + 7 > x => x < 19$$ Similarly:
$$7 + x > 12 => x > 5$$ And:
$$12 + x > 7 => x > -5 \text{~(\blue{This will always be true since the length cannot be negative})}$$
So, $$5 < x < 19$$  
  
  
  

**Correct output:**

</div>

<span id="fig:example_6" label="fig:example_6"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/6.jpg" style="height:0.7cm" alt="image" />

**ID:** 6  
  
**Question:** As shown in the figure, if $CB = 4.0$, $DB = 7.0$, and $D$
is the midpoint of $AC$, then the length of $AC$ is ().  
  
**Choices:** (A) $6$ cm (B) $7$ cm (C) $4$ cm (D) $5$ cm  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
From the figure, we can deduce:
$$AD = DC \text{~(since $D$ is the midpoint of $AC$)}$$

Now, using the entire line segment $AB$: $$AB = AD + DB$$
$$AB = DC + DB$$

Given $DB = 7.0$ cm, we can rearrange to find:
$$AD~(\text{or}~DC) = AB - DB$$ $$AD = AB - 7.0$$

Now, let’s find $AB$:

From the given information, $$AB = AC + CB$$ $$AB = AC + 4.0$$

Using the previously deduced equation: $$AD = AC + 4.0 - 7.0$$
$$AD = AC - 3.0$$

Since $D$ is the midpoint of $A$C, $AD$ will be half the length of AC:
$$AC = 2(AD)$$

From the above: $$\red{AC - 3.0 = 2(AD)}$$

Given that $AD$ and $DC$ are equal, we can represent $DC$ as:
$$DC = AC/2$$

Using the relation $CB = 4.0$ cm and $DB = 7.0$ cm:
$$\red{AC/2 + 7.0 = 4.0}$$ $$\red{AC/2 = -3.0}$$

  
From the original figure: $$AB = AD + DB$$ $$AB = DC + DB$$

Given $DB = 7.0$ cm, we have: $$AD = AB - 7.0$$ $$AD = AC/2 - 7.0$$

Using the relation $AB = AC + CB$: $$AC/2 + 7.0 = AC + 4.0$$

Subtracting $AC/2$ from both sides: $$7.0 = AC/2 + 4.0$$

Subtracting $4$ from both sides: $$AC/2 = 3.0$$

Multiplying both sides by $2$: $$AC = 6.0 \text{~cm}$$

**Correct output:**

</div>

<span id="fig:example_615" label="fig:example_615"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/28.png" style="height:2.8cm" alt="image" />

**ID:** 28  
  
**Question:** $\overline{AB}$ is a diameter, $AC=8$ inches, and $BC=15$
inches. Find the radius of the circle.  
  
**Choices:** (A) 7.5 (B) 8 (C) 8.5 (D) 17  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Given that $AB$ is a diameter and $AC = 8$ inches and $BC = 15$ inches,
we can use the Pythagorean theorem to determine the length of $AB$.  
The right triangle is formed by $A$, the center of the circle, and $C$
makes $AC$ the shorter leg, the radius the longer leg, and $BC$ the
hypotenuse.  
Let $r$ be the radius of the circle. Using the Pythagorean theorem:

Substituting the given lengths: $$\red{8^2 + r^2 = 15^2}$$
$$\red{64 + r^2 = 225}$$ $$\red{r^2 = 161}$$

Taking the square root of both sides: $$\red{r = \sqrt{161}}$$

$$\red{AB = 2r = 2\sqrt{161}}$$

$$r = \sqrt{161}/2 = 8.5 \text{~(approximately)}$$

**Correct output:**

</div>

<span id="fig:example_28" label="fig:example_28"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/644.png" style="height:3.8cm" alt="image" />

**ID:** 644  
  
**Question:** What is the missing number in the picture?  
  
**Choices:** (A) 6 (B) 8 (C) 10 (D) 11  
  
**Ground truth:**   
  
**Solution:** $8 + 4 -2 = 10$, $3 + 2 - 3 = 2$, $12 + 1 - 5 = 8$

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  
Let’s analyze the pattern in the given triangles:  
1. For the first triangle: $$\red{(8 - 4) \times 2 = 8}$$

2\. For the second triangle: $$\red{(3 - 2) \times 3 = 3}$$

Applying the same pattern to the third triangle:

$$\red{(12 - 1) \times 5 = 55}$$

However, since the answer is in single digits, it seems we take only the
unit’s place of the result.  
So, the missing number is:  
  
1. For the first triangle: $$\red{(8 + 4) / 2 = 6}$$

2\. For the second triangle:
$$\red{(3 + 2) / 3 = 5/3 = 1.67} \text{~(This does not match the number inside)}$$

But let’s continue with this pattern for the third triangle:
$$\red{(12 + 1) / 5 = 13/5 = 2.6}$$

  
  
1. For the first triangle: $$\red{(8 \times 4) / 2 = 16}$$

2\. For the second triangle: $$\red{(3 \times 2) / 3 = 2}$$

3\. For the third triangle: $$\red{(12 \times 1) / 5 = 12/5 = 2.4}$$

  
  

</div>

<span id="fig:example_644" label="fig:example_644"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/self/32.png" style="height:2.2cm" alt="image" />

**ID:** 32  
  
**Question:**

<div class="CJK" markdown="1">

UTF8gbsn 如图，在 $ABCD$ 中，$AB = AC$，$\angle CAB = 40^\circ$，则
$\angle D$ 的度数是（）。

</div>

  
(In the figure, within $ABCD$, $AB = AC$, and $\angle CAB = 40^\circ$,
then the measure of $\angle D$ is ().)  
  
**Choices:** (A) 40° (B) 50° (C) 60° (D) 70°  
  
**Ground truth:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**  

<div class="CJK" markdown="1">

UTF8gbsn 给定 $AB = AC$，$\triangle ABC$ 是等腰三角形。由于
$\angle CAB = 40^\circ$，所以
（因为在等腰三角形中，两个底角是相等的）。  
(Given $AB = AC$, $\triangle ABC$ is an isosceles triangle. Since
$\angle CAB = 40^\circ$, (because in an isosceles triangle, the two base
angles are equal).)  
考虑四边形 $ABCD$，其内角和为 $360^\circ$。已知
$\angle CAB = 40^\circ$，，并且 。  
(Considering the quadrilateral $ABCD$, its interior angles sum up to
$360^\circ$. We know that $\angle CAB = 40^\circ$, , and .)  
我们可以找到 $\angle BCA$ 的角度为:  
(We can find the angle of $\angle BCA$ as:)

，我们有:  
(, we have:)

  
()  
注意到 。所以:  
(Notice that . So:)

所以 。。  
(Thus, . .)  
  
()  
。  
(.)  
因此, ，但是 。  
(Therefore, , but .)  
所以 (So) $$\red{\angle D = 100^\circ - 40^\circ = 60^\circ}$$

</div>

</div>

<span id="fig:example_32" label="fig:example_32"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/self/44.png" style="height:3.5cm" alt="image" />

**ID:** 44  
  
**Question:** Chase wants to buy 4 kilograms of oval beads and 5
kilograms of star-shaped beads. How much will he spend? (Unit: \$)  
  
**Ground truth:**   
  
**Solution:** $8 + 4 -2 = 10$, $3 + 2 - 3 = 2$, $12 + 1 - 5 = 8$

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**  
The cost of 4 kilograms of oval beads is $4 * 2 = \$8$.  
The cost of 5 kilograms of star-shaped beads is $5 * 2 = \$10$.  
So the total cost is $\$8 + \$10 = \$18$.  
  
.

``` python
def calculate_cost(oval_beads, star_shaped_beads):
  oval_bead_cost = 2
  star_shaped_bead_cost = 2
  total_cost = oval_beads * oval_bead_cost + star_shaped_beads * star_shaped_bead_cost
  return total_cost

print(calculate_cost(4, 5))

# 35
```

</div>

<span id="fig:example_44" label="fig:example_44"></span>

</div>

## Self-Consistency for GPT-4V [app:gpt4v_consistency]

Self-consistency [wang2022self](http://arxiv.org/pdf/2404.13149v1) is a decoding strategy
for chain-of-thought prompting [wei2022chain](http://arxiv.org/pdf/2201.11903v6). A diverse
set of reasoning paths is sampled, and the most consistent answer is
selected as the final prediction. Moving beyond vanilla greedy decoding,
this method resorts to the inherent coherence and reliability of
multiple reasoning trajectories to produce a more trustworthy
conclusion. Self-consistency has been widely employed in LLMs for
complex reasoning tasks, such as math word problems and commonsense
reasoning.

In our experiments, we validated the effectiveness of using
self-consistency for GPT-4V on . Given a question and context, we ran
GPT-4V multiple times to obtain a set of different reasoning paths and
then selected the most frequent answer as the final prediction. We found
that self-consistency is instrumental in rectifying visual perception
errors (Figure <a href="#fig:example_648" data-reference-type="ref"
data-reference="fig:example_648">[fig:example_648]</a>), correcting
calculation mistakes (Figure
<a href="#fig:example_331" data-reference-type="ref"
data-reference="fig:example_331">[fig:example_331]</a>), and mitigating
hallucinations (Figure
<a href="#fig:example_430" data-reference-type="ref"
data-reference="fig:example_430">[fig:example_430]</a>). In comparison,
self-consistency is less effective when GPT-4V has difficulties in
interpreting complex visual contexts (Figures
<a href="#fig:example_157" data-reference-type="ref"
data-reference="fig:example_157">[fig:example_157]</a>,
<a href="#fig:example_551" data-reference-type="ref"
data-reference="fig:example_551">[fig:example_551]</a>) or extracting
salient information from images (Figure
<a href="#fig:example_965" data-reference-type="ref"
data-reference="fig:example_965">[fig:example_965]</a>).

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/648.png" style="height:3.3cm"
alt="image" />

**ID:** 648  
  
**Question:** What is the minimum value of this function?  
  
**Ground truth:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 1):**  
The minimum value of the function is observed where the graph is at its
lowest point along the $y$-axis.  
From the graph, the minimum value lies on the $y$-axis between -0.5 and
0. Based on the visual representation, it appears that the minimum value
of $y$ is: -0.5.  
So, the answer is -0.5.

**Wrong output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 2):**  
The graph shown represents a function with circular data points plotted
across the $x$-axis. To determine the minimum value of the function, we
need to locate the lowest point on the graph.  
From the graph, it is evident that the lowest point (or the minimum
value of the function) is at a $y$-coordinate of approximately -1.  
Therefore, the minimum value of the function is -1.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 3):**  
The graph depicts the function using circular points. The minimum value
of the function can be observed by identifying the lowest point on the
graph.  
From the graph, the lowest point seems to be at $y$ = -1.  
Therefore, the minimum value of this function is -1.

**Correct output:**

</div>

<span id="fig:example_648" label="fig:example_648"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/331.png" style="height:4.5cm"
alt="image" />

**ID:** 331  
  
**Question:** Several people compared how many Web pages they had
visited. What is the mean of the numbers?  
  
**Ground truth:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 1):**  
To find the mean of the numbers, you add up all the numbers and then
divide by the quantity of numbers. $$\begin{aligned}
\text{Mean} &= (84 + 56 + 70 + 68 + 60 + 47 + 75 + 52) ÷ 8\\
  &= 462 ÷ 8\\
  &= 57.75
\end{aligned}$$ However, since the question requires an integer answer,
the mean number of web pages visited is: 58.

**Wrong output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 2):**  
To find the mean number of web pages visited:  
First, sum the number of web pages for all the individuals:  
Bonnie: 84, Winston: 56, Percy: 70, Derek: 68, Jake: 60, Finn: 47,
Carter: 75, Kurt: 52

$$\text{Total} = 84 + 56 + 70 + 68 + 60 + 47 + 75 + 52 = 512$$

Next, divide by the number of individuals (8):
$$\text{Mean} = 512 ÷ 8 = 64$$

The mean number of web pages visited is 64.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 3):**  
First, I’ll list the number of web pages each person visited:  
Bonnie: 84, Winston: 56, Percy: 70, Derek: 68, Jake: 60, Finn: 47,
Carter: 75, Kurt: 52  
Next, I’ll group some numbers together for easier addition:
$$\begin{aligned}
(84 + 56) &= 140\\
(70 + 68) &= 138\\
(60 + 47) &= 107\\
(75 + 52) &= 127
\end{aligned}$$

Now, I’ll sum these results:

$$140 + 138 + 107 + 127 = 512$$

Now, I’ll divide the sum by the number of individuals to get the mean:

$$512 ÷ 8 = 64$$

So, the mean number of web pages visited is 64.

**Correct output:**

</div>

<span id="fig:example_331" label="fig:example_331"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/430.png" style="height:3.5cm"
alt="image" />

**ID:** 430  
  
**Question:** The derivative of $f(x)$ at $x=0$ is

------------------------------------------------------------------------

that at $x=5$  
  
**Choices:** (A) larger than (B) equal to (C) smaller than  
  
**Ground truth:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 1):**  
The graph represents the function $f(x) = |2x - 3| + 1$.  
The slope (or the derivative) of $f(x)$ is represented by the
inclination of the line.  
At $x = 0$, the slope is negative (the line is declining), and at
$x = 5$, the slope is positive (the line is inclining). The magnitude of
the slopes appears to be the same; only their directions differ.  
However, when considering the magnitude of the derivative, the value at
$x = 0$ is negative, and the value at $x = 5$ is positive, but they have
the same absolute value.

**Wrong output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 2):**  
The given function is $f(x) = |2x - 3| + 1$.  
The absolute value function is piecewise defined, so its derivative will
have different values depending on where you take it.  
For $2x - 3 > 0$, which is $x > \frac{3}{2}$ or 1.5, $f(x) = 2x - 3$.
The derivative is $f'(x) = 2$.  
For $2x - 3 < 0$, which is $x < \frac{3}{2}$ or 1.5, $f(x) = -2x + 3$.
The derivative is $f'(x) = -2$.  
At $x = 0$, the derivative $f'(x) = -2$.  
At $x = 5$, the derivative $f'(x) = 2$.  
Therefore, the derivative of $f(x)$ at $x = 0$ is smaller than that at
$x = 5$.

**Correct output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 3):**  
The function provided is $f(x) = |2x - 3| + 1$.  
The absolute value function has a V-shaped graph, as shown in the image.
The slope (or derivative) of the line to the left of the vertex is
negative, and the slope of the line to the right of the vertex is
positive.  
The derivative of $f(x)$ at $x=0$ corresponds to the slope of the line
on the left side of the vertex, which is negative (downward slope). The
derivative of $f(x)$ at $x=5$ corresponds to the slope of the line on
the right side of the vertex, which is positive (upward slope).  
Since a positive value is always larger than a negative value, the
derivative of $f(x)$ at $x=0$ is smaller than that at $x=5$.

**Correct output:**

</div>

<span id="fig:example_430" label="fig:example_430"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/157.png" style="height:4.3cm"
alt="image" />

**ID:** 157  
  
**Question:** What number is shown?  
  
**Ground truth:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 1):**  
Let’s break down the number shown using base-ten blocks:  
- There are 6 large cubes, each representing $1,000$. So,
$6 \times 1,000 = 6,000$.  
- There are 7 flats, each representing 100. So, $7 \times 100 = 700$.  
- There are 4 rods, each representing 10. So, $4 \times 10 = 40$.  
- There are 3 units, each representing 1. So, $3 \times 1 = 3$.  
Adding all of these values together: $$6,000 + 700 + 40 + 3 = 6,743$$

The number shown is $6,743$.

**Wrong output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 2):**  
The image displays a series of cubes that appear to represent numbers in
base 10 using place value.  
From left to right:  
1. There are six large cubes, which represent a value of
$6 \times 1000 = 6000$.  
2. There are three medium-sized cubes, which represent a value of
3$\times 100 = 300$.  
3. There are two small cubes, which represent a value of
$2 \times 10 = 20$.  
4. The vertical stack of single unit cubes shows nine cubes, which
represent a value of $9 \times 1 = 9$.  
Adding up these values: $$6000 + 300 + 20 + 9 = 6329$$

The number shown is $6329$.

**Wrong output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 3):**  
The image displays a series of cubes representing numbers in base 10
using place value.  
From left to right:  
1. There are seven large cubes, which represent a value of
$7 \times 1000 = 7000$.  
2. There are two medium-sized cubes, which represent a value of
$2 \times 100 = 200$.  
3. There are nine small cubes, which represent a value of
$9 \times 10 = 90$.  
4. The vertical stack of single unit cubes shows eight cubes, which
represent a value of $8 \times 1 = 8$.  
Adding up these values: $$7000 + 200 + 90 + 8 = 7298$$

The number shown is $7298$.

**Wrong output:**

</div>

<span id="fig:example_157" label="fig:example_157"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/551.png" style="height:4.8cm"
alt="image" />

**ID:** 551  
  
**Question:** What is the difference between the highest tattoos in male
and the least in female?  
  
**Ground truth:**   
  
**Solution:**  
27% (male in 2015)  
- 13% (female in 2008)  
= 14%

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 1):**  
The highest percentage of tattoos in males is 31% (in 2015), and the
least percentage of tattoos in females is 15% (in 2008).  
Difference = 31% - 15% = 16%

**Wrong output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 2):**  
The highest percentage of males with at least one tattoo is in 2015 at
31%.  
The lowest percentage of females with at least one tattoo is in 2003,
2008, and 2012, all at 15%.  
The difference between these values is: 31% - 15% = 16%

**Wrong output:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 3):**  
To determine the difference between the highest tattoos in males and the
least in females:  
Highest tattoos in males: 31% (in 2015)  
Least tattoos in females: 15% (in 2003 and 2008)  
Difference: 31% - 15% = 16%

**Wrong output:**

</div>

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/551.png" style="height:4.8cm"
alt="image" />

**Prompt:** What is the percentage of tattoos in females in 2008?  
**GPT-4V:** .  
**Prompt:** What is the percentage of tattoos in males in 2015?  
**GPT-4V:**

</div>

<span id="fig:example_551" label="fig:example_551"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/965.jpg" style="height:5cm" alt="image" />

**ID:** 965  
  
**Question:** What’s the lowest number yard line that you can see?  
  
**Ground truth:**

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 1):**  
The lowest number yard line visible in the image is .

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 2):**  
From the image provided, the lowest number yard line that is clearly
visible is .

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 3):**  
From the image provided, the lowest number yard line that can be seen is
.

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 4):**  
From the image, the lowest number yard line that is visible is .

</div>

<div class="mybody" markdown="1">

**GPT-4V (Reasoning Path 5):**  
From the image, the lowest number yard line that is visible is .

</div>

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/consistency/965.jpg" style="height:5cm" alt="image" />

**Prompt:** List the yard line numbers that are visible in the image.  
**GPT-4V:**  
  
**Prompt:** Can you see the yard numbers 40 and 50 in the image?  
**GPT-4V:**

</div>

<span id="fig:example_965" label="fig:example_965"></span>

</div>

## Abstaining from Responding [sec:gpt4v_abstain]

We observed that both Multimodal Bard and GPT-4 often abstain from
responding. Specifically, Multimodal Bard returns “Sorry, I can’t answer
this question” (7.0%) or “I can’t process this file” (0.8%), while
GPT-4V responds with “Sorry, I cannot help with that.” (7.0%). As
illustrated in Figures
<a href="#fig:example_699" data-reference-type="ref"
data-reference="fig:example_699">[fig:example_699]</a> through
<a href="#fig:example_more_safety" data-reference-type="ref"
data-reference="fig:example_more_safety">[fig:example_more_safety]</a>,
these patterns can be attributed to patterns in large language models
that have been characterized as “exaggerated safety behaviors”
[rottger2023xstest](None). Further research is necessary to
explore these behaviors in large multimodal models.

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/safety/699.jpg" style="height:2.8cm" alt="image" />

**ID:** 699  
  
**Question:** What is the age gap between the center and the rightmost
person? (unit: years)  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**

</div>

<span id="fig:example_699" label="fig:example_699"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/safety/740.jpg" style="height:2.8cm" alt="image" />

**ID:** 740  
  
**Question:** What percent of the stands are full?  
  
**Choices:** (A) 15 (B) 100 (C) 50  
  
**Ground truth:**

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**

</div>

<span id="fig:example_740" label="fig:example_740"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/safety/779.jpg" style="height:2.8cm" alt="image" />

**ID:** 779  
  
**Question:** Is the cardiac silhouette less than half the diameter of
the diaphragm?  
  
**Choices:** (A) Yes (B) No  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**

</div>

<span id="fig:example_779" label="fig:example_779"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/safety/821.jpg" style="height:5cm" alt="image" />

**ID:** 821  
  
**Question:** Are most of the people young men?  
  
**Choices:** (A) Yes (B) No  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**Multimodal Bard:**

</div>

<div class="mybody" markdown="1">

**GPT-4V:**

</div>

<span id="fig:example_821" label="fig:example_821"></span>

</div>

<div class="table*" markdown="1">

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/safety/352.jpg" style="height:3.8cm" alt="image" />

**ID:** 352  
  
**Question:** How many people will fit in the smaller vehicle?  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/images/safety/516.jpg" style="height:5cm" alt="image" />

**ID:** 516  
  
**Question:** Are most the stepping stones square?  
  
**Choices:** (A) Yes (B) No  
  
**Ground truth:**   
  

</div>

<div class="mybody" markdown="1">

**GPT-4V:**

</div>

<span id="fig:example_more_safety"
label="fig:example_more_safety"></span>

</div>

## GPT-4V for Multi-Turn Human-AI Interaction [app:gpt4v_chatbot]

This section investigates the use of GPT-4V for multi-turn human-AI
interaction on , as exemplified in the goal-directed dialog in Figure
<a href="#fig:gpt4v_824" data-reference-type="ref"
data-reference="fig:gpt4v_824">1</a>.

We found that GPT-4V is effective in engaging multi-turn goal-directed
conversations with users. In particular, GPT-4V can make good use of
*hints* (*e.g.,* user feedback or responses) to guide the conversion to
generate desirable results. For instance, it can (1) rectify visual
perception errors based on hints (Figure
<a href="#fig:gpt4v_chatbot_example_113" data-reference-type="ref"
data-reference="fig:gpt4v_chatbot_example_113">2</a>), (2) reassess
reasoning steps and calculations (Figure
<a href="#fig:gpt4v_chatbot_example_899" data-reference-type="ref"
data-reference="fig:gpt4v_chatbot_example_899">3</a>), (3) correct
misinformation using user-provided domain-specific knowledge (Figure
<a href="#fig:gpt4v_chatbot_example_147" data-reference-type="ref"
data-reference="fig:gpt4v_chatbot_example_147">4</a>), and (4) aggregate
intricate contexts over multiple turns in a human-AI conversation
(Figures
<a href="#fig:gpt4v_chatbot_example_241" data-reference-type="ref"
data-reference="fig:gpt4v_chatbot_example_241">5</a> and
<a href="#fig:gpt4v_chatbot_example_241_2" data-reference-type="ref"
data-reference="fig:gpt4v_chatbot_example_241_2">6</a>).

We also observed failure cases in our evaluation. For instance, GPT-4V
struggles to generate correct responses when questions and user hints
are ambiguous (Figure <a href="#fig:gpt4v_458" data-reference-type="ref"
data-reference="fig:gpt4v_458">7</a>), or when the model fails to
understand abstract shapes and concepts visually (Figure
<a href="#fig:gpt4v_chatbot_example_519" data-reference-type="ref"
data-reference="fig:gpt4v_chatbot_example_519">8</a>). These failures
motivate the development of more powerful, conversational foundation
models.

<figure id="fig:gpt4v_824">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_824.png" style="width:85.0%" />
<figcaption>A multi-turn dialog between a human user and
GPT-4V.</figcaption>
</figure>

<figure id="fig:gpt4v_chatbot_example_113">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_113.png" style="width:90.0%" />
<figcaption>GPT-4V initially produces an incorrect solution due to
errors in visually perceiving the number of baseballs. After receiving
user feedback, it recalibrates and generates the accurate
answer.</figcaption>
</figure>

<figure id="fig:gpt4v_chatbot_example_899">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_899.png" style="width:90.0%" />
<figcaption>GPT-4V correctly identifies the sequence of numbers in the
images but struggles to discern the underlying pattern. With feedback
from the user, GPT-4V re-evaluates the pattern and produces the correct
response to the logical reasoning question.</figcaption>
</figure>

<figure id="fig:gpt4v_chatbot_example_147">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_147.png" style="width:100.0%" />
<figcaption>Initially, GPT-4V applies an incorrect theorem to address
the scientific reasoning query. However, after receiving the appropriate
domain knowledge hint from the user, it utilizes the correct theorem,
combined with key visual cues from the images, to produce the right
answer.</figcaption>
</figure>

<figure id="fig:gpt4v_chatbot_example_241">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_241.png" style="width:100.0%" />
<figcaption>Answering this question requires comprehending a complex
table regarding AI models (such as identifying relevant rows and
columns), and up-to-date knowledge of AI research, and the ability to
perform advanced arithmetic reasoning. Without human in the loop, GPT-4V
fails to produce the correct answer.</figcaption>
</figure>

<figure id="fig:gpt4v_chatbot_example_241_2">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_241_2.png"
style="width:100.0%" />
<figcaption>(Continued) In this multi-turn human-AI conversation, the
user acts as a tutor to help GPT-4V generate the correct answer . (1)
Initially, the user directs GPT-4V to the “Model Size” column after
observing its misdirection. Despite the guidance, GPT-4V still
struggles. (2) Recognizing the model’s difficulty in identifying the
“Model Size” column, the user provides more detailed instructions. This
time, GPT-4V is on the right track but still makes some error in its
response. (3) With further information from the user, GPT-4V corrects
the remaining errors and generate the correct answer.</figcaption>
</figure>

<figure id="fig:gpt4v_458">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_458.png" style="width:90.0%" />
<figcaption aria-hidden="true"></figcaption>
</figure>

<figure id="fig:gpt4v_chatbot_example_519">
<embed src="/papers/doc_ai_databases/arXiv-2310.02255v3_md/figs/gpt4v_chatbot_example_519.png" style="width:100.0%" />
<figcaption>In this example, even after the user offers helpful hints,
GPT-4V struggles to decipher the number of triangles in the given
geometric shape. The task is challenging for AI models, including
GPT-4V, with a limited ability to understand abstract shapes
visually.</figcaption>
</figure>

## Future Directions [sec:gpt4v_future]

Text