# Introduction

Mobile app screenshots have been analyzed using machine learning from
multiple aspects. These analyses range from pixel level understanding,
e.g., layout structural analyses, UI issue detection and
correction [liLearningDenoiseRaw2022](https://doi.org/10.1145/3491102.3502042), to UI element
semantics, e.g., icon recognition, button action
prediction [sunkaraBetterSemanticUnderstanding2022](None), to
even higher-level functional analyses such as accessibility
support [liWidgetCaptioningGenerating2020](https://doi.org/10.18653/v1/2020.emnlp-main.443), screen
description [wangScreen2WordsAutomaticMobile2021](https://doi.org/10.1145/3472749.3474765), and
screen type classification [dekaRicoMobileApp2017](https://doi.org/10.1145/3126594.3126651).
Comparatively, the content understanding aspect is relatively
understudied. By content, we mean the information displayed on the
screen to convey and satisfy the purpose of using the app. Examples
include star ratings from restaurant reviews, messages from chat apps,
cuisine ingredients from recipe apps, flight status and in-flight
amenities from travel planner apps, etc. Having this capacity of
understanding is important for two reasons: First, the sole reason for
many apps and app categories to exist is to satisfy users’ information
need, e.g., weather, map navigation, and news apps. Second, for task
completion[^1], which requires the eyes-free agent capacity, the two
types of screen understandings — content and action understanding — are
inseparable in order to carry out a task successfully. Without knowing a
screen state properly, a machine learning agent is unable to self-assess
if the action is performed as expected, or unable to provide sufficient
feedback to the user to achieve true eyes-free user experience. More
intrinsically, from a pure research perspective, we are interested in
knowing the limit of machine screen content understanding[^2] and what
constitutes the challenges, given that app screenshots are entirely
human artifacts made for convenient comprehension.

Accordingly, we annotated the RICO
dataset [dekaRicoMobileApp2017](https://doi.org/10.1145/3126594.3126651) with 86,025
question-answer pairs, referred to as Screen Question Answering, or, in
short, **ScreenQA** annotations later in this work, and released the
dataset in the public domain[^3]. The ScreenQA task requires an agent to
answer a user’s question by selecting one or multiple UI elements from
the given screenshot, as will be formulated in
Section <a href="#sec:problem_setting" data-reference-type="ref"
data-reference="sec:problem_setting">[sec:problem_setting]</a>. Question
answering is employed as a touchstone to sparsely[^4] verify the quality
of screen content understanding. To the best of our knowledge, this is
the first large-scale questions answering dataset over mobile app
screenshots, and the first one to be publicly available. Much inspired
by the SQuAD dataset [rajpurkarSQuAD1000002016a](https://doi.org/10.18653/v1/D16-1264), we
hope, by releasing this set of annotations, to encourage the community
to advance technologies toward better screen content understanding. We
anticipate that the advance of such technologies will benefit beyond
just the screen UI and the human computer interaction (HCI) domains. As
we will discuss in
Section <a href="#sec:related_work" data-reference-type="ref"
data-reference="sec:related_work">[sec:related_work]</a>, other
vision-language related multimodal domains share similar challenges with
different emphases on respective modalities and contexts. Comparatively,
ScreenQA is language and layout heavy, but it also includes visual
ingredients such as icons and symbols as concise representations in
place of texts, to declutter the UI. It may also include images or art
designs that pose challenges to language centric machine learning
agents.

The remaining paper is organized in the following way:
Section <a href="#sec:problem_setting" data-reference-type="ref"
data-reference="sec:problem_setting">[sec:problem_setting]</a>
formulates the problem, including the problem description and the
evaluation metrics. We discuss relevant prior datasets and annotations
in Section <a href="#sec:related_work" data-reference-type="ref"
data-reference="sec:related_work">[sec:related_work]</a> to put this
work into perspective.
Section <a href="#sec:annotation_method" data-reference-type="ref"
data-reference="sec:annotation_method">[sec:annotation_method]</a>
describes our annotation method. The annotations are then analyzed in
Section <a href="#sec:annotation_analysis" data-reference-type="ref"
data-reference="sec:annotation_analysis">[sec:annotation_analysis]</a>
to provide readers both the qualitative and quantitative views. The
paper is concluded in
Section <a href="#conclusion" data-reference-type="ref"
data-reference="conclusion">[conclusion]</a> with a summary and a remark
on future works.

[^1]: Also referred to as automation or app control.

[^2]: This term is analogous to machine reading comprehension from
    natural language processing.

[^3]: ScreenQA dataset is released at
    <https://github.com/google-research-datasets/screen_qa>.

[^4]: Because questions are not exhaustively asked against a given
    screenshot.

# Problem Setting [sec:problem_setting]

<div class="figure*" markdown="1">

<figure id="fig:eg-skywatching">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/rico_daftin_26_bboxes.png" />
<figcaption>Ambiguous UI element boundaries. Three possibilities are
annotated.</figcaption>
</figure>

<figure id="fig:eg-weather-forecast">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/rico_wmc_weather_111_bboxes.png" />
<figcaption>Answers to “What’s the temperature on
Saturday?”</figcaption>
</figure>

<figure id="fig:eg-exercise-program">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/rico_abad_391_bboxes.png" />
<figcaption>Semantic groups are the basic unit for ordering, not element
coordinates.</figcaption>
</figure>

</div>

We state the problem and define the evaluation metrics in this section.

## Problem statement [sec:problem_statement]

The ScreenQA task requires an agent to answer a user’s question by
selecting relevant UI elements from a given single screenshot. When it
comes with multiple relevant UI elements, a list of such UI elements
whose contents *minimally* satisfy the question should be selected and
ranked in descending order of relevance to the question, if applicable,
or following the common reading order by semantic groups, as will be
described in
Section <a href="#sec:properties_and_terminologies" data-reference-type="ref"
data-reference="sec:properties_and_terminologies">1.2</a>. This assumes
that answers are directly selectable from the screen and logical
reasoning and calculation are not needed. If the screenshot does not
contain the answers to the question, the agent should respond with “\<no
answer\>”. This is summarized in
Task <a href="#task:sreenqa" data-reference-type="ref"
data-reference="task:sreenqa">[task:sreenqa]</a>.

<div class="task" markdown="1">

**Input:** a question $Q$ and a screenshot $S$  
**Output:** an answer list $A$ of UI elements selected from $S$ such
that their contents minimally satisfy $Q$. The order of $A$ is further
required to be

-   Ranked in descending order of relevance to $Q$, if applicable.

-   Otherwise, following the common reading order by semantic groups.

If no contents in $S$ can satisfy $Q$, then returns an empty list $A$.

</div>

## Properties and terminologies [sec:properties_and_terminologies]

The mobile app UI comes with some nuances. It is worth mentioning a few
properties below.

-   View hierarchy, or the structural representation used to render the
    screen, is not required in
    Task <a href="#task:sreenqa" data-reference-type="ref"
    data-reference="task:sreenqa">[task:sreenqa]</a>, to be consistent
    with the human annotation process in
    Section <a href="#sec:annotation_method" data-reference-type="ref"
    data-reference="sec:annotation_method">[sec:annotation_method]</a>.
    View hierarchy usually provides useful UI element candidates, but it
    may not always be reliable, for example, when using WebView or
    screen overlays. In such cases, a human annotator can still answer
    screen questions entirely from pixels without an issue, so we want
    to benchmark similarly. We leave the choice of dependency on view
    hierarchies to the modelers and, hence, do not require it. However,
    this comes with an ambiguity for UI element boundaries. See an
    example in
    Figure <a href="#fig:eg-skywatching" data-reference-type="ref"
    data-reference="fig:eg-skywatching">1</a>. We devise a more flexible
    answer matching process to mitigate such an impact, as will be
    discussed in
    Section <a href="#sec:answer_matching" data-reference-type="ref"
    data-reference="sec:answer_matching">1.3.3</a>.

-   Avoid question answering over long paragraphs. Although it is
    permissive by Task <a href="#task:sreenqa" data-reference-type="ref"
    data-reference="task:sreenqa">[task:sreenqa]</a>, we discourage
    annotators from asking such questions during the annotation process.
    For ScreenQA, we want to focus on learning the relationships between
    text segments arranged two-dimensionally on the screen, and leave
    the long paragraph question answering, which investigates the
    relationships between words, to the traditional NLP domain.

-   Avoid logical reasoning. This task assumes answers can directly be
    extracted from the screenshot without reasoning, entailment,
    counting and comparing numbers. This further exclude yes/no and why
    questions if not explicitly displayed on the screen. The reason is
    that we want to separate “able to read” and “able to reason” and
    focus on the former first without generating an over challenging
    dataset. A few such excluded examples are: counting items, asking
    about the weather a few days from now, what are the items cheaper
    than X dollars, etc.

-   Ordered by relevance. The task is designed to enable the eyes-free
    user experience. That is, a user may not be fully aware of how many
    relevant answers are displayed on the screen. For example, in
    Figure <a href="#fig:eg-weather-forecast" data-reference-type="ref"
    data-reference="fig:eg-weather-forecast">2</a>, when a user asks
    “What’s the temperature on Saturday?”, there are actually two
    temperatures, high and low, for each day and two Saturdays on the
    screen. In this case, the two temperatures should just follow the
    reading order, and the two Saturdays follow the relevance order as a
    user usually refers to the upcoming Saturday. For a well-designed
    mobile app, these two usually overlap well and we do not expect a
    large ambiguity here.

-   Reading order by semantic groups. Sometimes some UI elements are
    designed as semantic groups and should be referred to together to
    keep their semantic meaning. For example, in
    Figure <a href="#fig:eg-exercise-program" data-reference-type="ref"
    data-reference="fig:eg-exercise-program">3</a>, when a user asks
    “What are the first two movements of deep squat?”, then the answer
    should be “Deep Squat, 3 sets, 15x”, followed by “Lunge, 3 sets,
    10x”. In other words, the common reading order should be based on
    semantic groups as the unit, rather than simply sorted by the
    coordinates of UI elements.

Note that we set up the problem this way strategically in order to
prioritize its solvability considering the progress of current
technologies. However, practically, long vs. short texts and retrieval
vs. reasoning are naturally mixed together in the daily usage of mobile
apps. We will leave this type of composite problems to the future work.

## Evaluation metrics [sec:metrics]

We consider two types of metrics: 1) Average normalized Discounted
Cumulative Gain (Average
nDCG) [jarvelinCumulatedGainbasedEvaluation2002](https://doi.org/10.1145/582415.582418), which
is commonly used in information retrieval and ranking systems, and 2)
Average F1 score, which has been employed in closed-domain question
answering problems, such as the SQuAD
dataset [rajpurkarSQuAD1000002016a](https://doi.org/10.18653/v1/D16-1264).

One major difference between our metrics described below and the
commonly used definitions is the unit of predictions. We use the element
in the answer list $A$, described in
Task <a href="#task:sreenqa" data-reference-type="ref"
data-reference="task:sreenqa">[task:sreenqa]</a>, as the unit to
determine a hit or a miss for both metrics. Besides, as UI elements can
be ambiguous as mentioned in
Section <a href="#sec:properties_and_terminologies" data-reference-type="ref"
data-reference="sec:properties_and_terminologies">1.2</a>, we will
describe an answer matching algorithm that mitigate such an impact in
Section <a href="#sec:answer_matching" data-reference-type="ref"
data-reference="sec:answer_matching">1.3.3</a>.

### Average nDCG [sec:avg_ndcg]

We use a variant of nDCG that allows varying positions (number of
returns) as opposed to a typical fixed position. This is because, unlike
the search problem, which is fair to evaluate, say, top-10 retrieved
documents across queries, ScreenQA can have different needs of answer
lengths across different questions. For example, a question like “what
is the address of the store” expects a single returned result. A
question like “what are the login options?” expects an enumeration of
options on the screen that easily go beyond five. Accordingly, we allow
$v$arying positions as follows: Given a 1-indexed list $A$, which is the
predicted answer for the screen-question pair $(S, Q)$, and a ground
truth answer list $A_g$ for $(S, Q)$, the Discounted Cumulative Gain at
$v$arying positions (DCG$_v$) is computed by: $$\label{eq:dcg}
    \mbox{DCG}_v = \sum^{\|A\|}_{i=1} \frac{r_i}{\log_2{(i+1)}},$$ where
$\|\cdot\|$ is the size of the list argument, $r_i$ is the relevance
score for the $i$-th item of $A$. We assign the relevance score 1 for a
hit and 0 for a miss compared with the ground truth $A^g$. The
corresponding Ideal Discounted Cumulative Gain (IDCG$_v$) is computed
by: $$\label{eq:idcg}
    \mbox{IDCG}_v = \sum^{\|A^g\|}_{i=1} \frac{1}{\log_2{(i+1)}}.$$ The
nDCG$_v$ is then $$\label{eq:ndcg}
    \mbox{nDCG}_v = \frac{\mbox{DCG}_v}{\mbox{IDCG}_v}.$$ Note that
nDCG$_v$ is still between 0 and 1, hence, convenient for comparing
scores and computing the average.

For a dataset of $N$ examples, each of which is indexed by $i$ and has a
predicted answer $A_i$ and $K$ ground truth annotations
$A^g_{i, j=1 \dots K}$, the average nDCG$_v$ can be computed by
$$\label{eq:avg_ndcg}
    \mbox{avg}(\mbox{nDCG}_v) = \frac{1}{N}\sum_{i=1}^N \mbox{max}_j [ \mbox{nDCG}_v(A_i, A^g_{i,j} ) ].$$
We choose a variant nDCG as the metric because 1) we want to measure the
quality of the ranking. For example, if one incorrectly predicts the
result from the first to the third position, the discount factor brings
down the score from 1.0 to only 0.5. 2) nDCG has an orthogonal design,
which is easier to tweak toward a specific need than the mean average
precision (mAP) metric. For example, one can choose to discount faster
or slower by changing the base of the denominator $\log_2(i+1)$ and can
choose to penalize irrelevant predictions by assigning negative scores.
Mean reciprocal rank (MRR) and mAP are much less controllable in these
two aspects.

One known drawback of nDCG is that it does not naturally penalize
excessive predictions after the last relevant item. We therefore use the
average F$_1$ score as a complementary view of the agent performance.

### Average F$_1$ [sec:avg_f1]

Similar to the definition in SQuAD, the average F$_1$ score is computed
as below, following the same notation as in
<a href="#eq:avg_ndcg" data-reference-type="eqref"
data-reference="eq:avg_ndcg">[eq:avg_ndcg]</a>:

$$\label{eq:avg_f1}
    \mbox{avg}(\mbox{F}_1) = \frac{1}{N}\sum_{i=1}^N \mbox{max}_j [ \mbox{F}_1(A_i, A^g_{i,j} ) ].$$
Note that F$_1$ does not concern ranking. For some cases, such as
enumeration questions, this is desirable, as the ranking order is merely
the reading order, even if the item order is permuted, the answer
quality is in general not compromised, hence, reasonable to be assigned
the same evaluation score. On the contrary, if relevance ranking is
important, such as in
Figure <a href="#fig:eg-weather-forecast" data-reference-type="ref"
data-reference="fig:eg-weather-forecast">2</a>, then nDCG provides a
better view. Since both types of questions exist in the ScreenQA
annotations, it is more complete to evaluate against both metrics. Also
note that the unit of precision and recall computation is based on items
in $A$, unlike SQuAD, which uses words as the unit instead. We describe
how to compare items in an answer $A$ with the ground truth $A^g$ in the
next section.

### Answer matching [sec:answer_matching]

As mentioned in
Section <a href="#sec:properties_and_terminologies" data-reference-type="ref"
data-reference="sec:properties_and_terminologies">1.2</a>, the
segmentation of UI elements provided in the predicted answer list $A$
may not coincide with the UI elements in the ground truth list $A^g$.
Yet, if the overall answers are the same, the segmentation difference
should not affect the evaluation score. Therefore, we use the following
empirical procedure to mitigate such an impact, using an illustrated
example (each capitalized character is a word token): $$\begin{aligned}
    A &= ["AB", "B", "BB", "CBA"] \\
    A^g &= ["AB", "BC", "AB"],
\end{aligned}$$

1.  Concatenate items in $A$ into a single item list
    $A^c = [``ABBBBCBA"]$.

2.  Iterate through each $g \in A^g$ and check if $g$ is contained in
    any item in $A^c$. If so, mark the $g$ as HIT ($\cmark$) and mark
    the corresponding matched word token in the original $A$ and remove
    the matched part and split the remaining parts in $A^c$. Otherwise,
    mark the $g$ as MISS ($\xmark$). In this example, when $g = "AB"$,
    it is a HIT: $$\begin{aligned}
            A &= [``A_\cmark B_\cmark", ``B", ``BB", ``CBA"] \\
            A^c &= [``BBBCBA"] \\
            A^g &= [``AB"_\cmark, ``BC", ``AB"].
        
    \end{aligned}$$ Then when $g = ``BC"$, it is a HIT. Note that the
    item in $A^c$ is split into two because of matching in the middle:
    $$\begin{aligned}
            A &= [``A_\cmark B_\cmark", ``B", ``BB_\cmark", ``C_\cmark BA"] \\
            A^c &= [``BB", ``BA"] \\
            A^g &= [``AB"_\cmark, ``BC"_\cmark, ``AB"].
        
    \end{aligned}$$ Last, when $g = ``AB"$ again, it is a MISS, $A$ and
    $A^c$ unchanged, hence, omitted: $$\begin{aligned}
            A^g &= [``AB"_\cmark, ``BC"_\cmark, ``AB"_\xmark].
        
    \end{aligned}$$

3.  Finally, iterate through each $a \in A$. If any $a$ has at least one
    word token marked as HIT, then the whole $a$ is a HIT, otherwise, a
    MISS. $$\begin{aligned}
            A &= [``AB"_\cmark, ``B"_\xmark, ``BB"_\cmark, ``CBA"_\cmark].
        
    \end{aligned}$$

This procedure converts $A$ and $A^g$ into lists of HITs and MISSes.
Then the evaluation metrics
in <a href="#eq:avg_ndcg" data-reference-type="eqref"
data-reference="eq:avg_ndcg">[eq:avg_ndcg]</a>
and <a href="#eq:avg_f1" data-reference-type="eqref"
data-reference="eq:avg_f1">[eq:avg_f1]</a> can be applied. Note that
this procedure is not order invariant. This in turn makes the F$_1$
score not entirely order independent if any UI element ambiguity
happens. This choice is to avoid the permutation complexity in
evaluation. In practice, this is rarely an issue because when the
ambiguity happens, the UI elements involved are almost always tightly
close to each other, making their order practically fixed. See Case 3 in
Figure <a href="#fig:eg-skywatching" data-reference-type="ref"
data-reference="fig:eg-skywatching">1</a> as an example.

# Related Datasets and Annotations [sec:related_work]

ScreenQA has two aspects: multimodality and question answering. We
discuss related problems and datasets from these two aspects and focus
our survey on datasets that are 1) human annotated and 2) released to
the public domain.

## Multimodality

Mobile app screenshots contain nearly all possible representation of
information through pixels. Most commonly, the information is majorly by
text, blended with icons, symbols, and images.[^1] We discuss three
related multimodal domains.

### Screen UI for mobile apps

For data released in the public domain, the RICO
dataset [dekaRicoMobileApp2017](https://doi.org/10.1145/3126594.3126651) is, to the best of our
knowledge, still the largest collection of mobile app
screenshots [dekaEarlyRicoRetrospective2021](https://doi.org/10.1007/978-3-030-82681-9_8). It
contains 66k unique screenshots and their corresponding view hierarchies
from 9.7k Android apps spanning 27 app categories. Its overall approach
extended ERICA [dekaERICAInteractionMining2016](https://doi.org/10.1145/2984511.2984581), which is
an interactive trace recording tool and also released 3k traces for 18k
unique screenshots from 1k Android apps for the search intent.
LabelDroid [chenUnblindYourApps2020](https://doi.org/10.1145/3377811.3380327)
and [chenWireframebasedUIDesign2020](https://doi.org/10.1145/3391613) by the same authors
released a dataset of 55k UI screenshots from 25 categories of 7.7k
top-downloaded Android apps.

Annotations and the corresponding problems can be roughly categorized by
the scope of the contexts. At the UI element
level, [sunkaraBetterSemanticUnderstanding2022](None)
annotated 77 icon types by shape, 15 out of which are additionally
annotated with 38 semantic types, reaching about total 500k unique
annotations. This work is further concerned with how UI elements are
associated with companion labels such that the screen understanding
between UI elements can be established.
CLAY [liLearningDenoiseRaw2022](https://doi.org/10.1145/3491102.3502042) attempted to resolve the
layout and view hierarchy denoising problem, annotating 60k RICO
screenshots, a total of 1.4M UI elements with bounding boxes and types.
[liWidgetCaptioningGenerating2020](https://doi.org/10.18653/v1/2020.emnlp-main.443) annotated 163k
free-from descriptions for 61k UI elements from 22k RICO screenshots. At
the single-screen
level, [wangScreen2WordsAutomaticMobile2021](https://doi.org/10.1145/3472749.3474765) collected
text summarizations for screens, consisting of 112k screen descriptions
across 22k RICO screenshots.

At the multi-screen level, one challenging direction is screen
navigation, which requires the understanding of screen states, feasible
action spaces of the current screen, and overall task goals. Since
multiple types of understandings are involved, this problem is not
strictly focused on screen content understanding.
PixelHelp [liMappingNaturalLanguage2020b](https://doi.org/10.18653/v1/2020.acl-main.729) contains 187
multi-step instructions over 780 screenshots for four task categories.
MoTIF [burnsDatasetInteractiveVisionLanguage2022](https://doi.org/10.48550/arXiv.2202.02312)
contains 6k fine-grained instructions mixed with infeasible ones, over
for 125 apps spanning 15 app categories. From the data perspective,
annotating this type of problem is labor intensive and usually does not
scale well.

In comparison, the ScreenQA dataset is single-screen, focused on screen
contents, and based on the RICO screenshots.

### Document image understanding

Document image understanding[^2] concerns understanding documents
represented in pixels or scanned, photographed formats. This domain is
similar to mobile app screens for its text-heavy and non-sequential
nature. The most noticeable dataset is
RVL-CDIP [harleyEvaluationDeepConvolutional2015](https://doi.org/10.1109/ICDAR.2015.7333910),
a 400k-image subset from
IIT-CDIP [lewisBuildingTestCollection2006](https://doi.org/10.1145/1148170.1148307), a collection
of low-resolution noisy documents, with balanced 16 document-level
classes. FUNSD [jaumeFUNSDDatasetForm2019](https://arxiv.org/pdf/1905.13538) extracted
a 199 scanned form images from RVL-CDIP and annotated them with bounding
boxes and 4 text-segment-level classes.
SROIE [huangICDAR2019CompetitionScanned2019](https://doi.org/10.1109/ICDAR.2019.00244) has 1k
scanned receipt images for text localization, OCR, and key information
extraction of 4 entity types.
CORD [parkCORDConsolidatedReceipt2019](None) contains 11k
scanned receipt images, annotated with 9 classes and 54 subclasses for
text segments in OCR boxes. These earlier works are more about
classification for text segments or for the whole document image.

A more recent work, DocVQA [mathewDocVQADatasetVQA2021](https://doi.org/10.1109/WACV48630.2021.00225),
uses a question answering format for span/segment extraction, with an
annotation of 50k questions over 12k rectified, higher resolution
document images. DocVQA is highly related to ScreenQA for its 2D
arrangement of texts and for its extractive question answering format.
We believe that the techniques developed for screens and document images
are cross applicable.

### Visual question answering

Visual question
answering (VQA) [antolVQAVisualQuestion2015](https://doi.org/10.1109/ICCV.2015.279) and screen
UI are oftentimes mentioned together, especially in the latter
community, because of their vision-language multimodal nature. However,
VQA is distinctively different from screen understanding for two
reasons: 1) The visual context for VQA is usually light in, or even free
from, any text, while screen UI is the opposite, and 2) The images for
VQA are typically photos of natural or daily scenes with objects, while
screen UIs are information oriented and arranged in a certain visual
structure. There are some VQA variants comparatively closer to screen
UI, to mention a few: VQA for texts on objects in photos, e.g.,
VizWiz [gurariVizWizGrandChallenge2018](https://doi.org/10.1109/CVPR.2018.00380) and
TextVQA [singhVQAModelsThat2019](https://doi.org/10.1109/CVPR.2019.00851), and VQA for figures and
charts, e.g., DVQA [kafleDVQAUnderstandingData2018](https://doi.org/10.1109/CVPR.2018.00592),
FigureQA [kahouFigureQAAnnotatedFigure2018](None), and
LEAF-QA [chaudhryLEAFQALocateEncode2020](https://doi.org/10.1109/WACV45572.2020.9093269). These VQA tasks
may appear as part of screens but in general are different problems.

## Question answering

Question answering tasks can be categorized by 1) open- or
closed-domain, 2) answer formats and 3) main capacities to evaluate.[^3]
The common answer formats include
span [rajpurkarSQuAD1000002016a](https://doi.org/10.18653/v1/D16-1264),
entity [talmorWebKnowledgeBaseAnswering2018](https://doi.org/10.18653/v1/N18-1059), multiple
choice [mihaylovCanSuitArmor2018](https://doi.org/10.18653/v1/D18-1260), and
generation [xiongTWEETQASocialMedia2019](https://doi.org/10.18653/v1/P19-1496). The capacities
to evaluate range from reading
comprehension [yangWikiQAChallengeDataset2015](https://doi.org/10.18653/v1/D15-1237), multi-hop
reasoning [yangHotpotQADatasetDiverse2018](https://doi.org/10.18653/v1/D18-1259), [chenFinQADatasetNumerical2021](https://doi.org/10.18653/v1/2021.emnlp-main.300),
logic reasoning [yuReClorReadingComprehension2020](None), and
commonsense
reasoning [talmorCommonsenseQAQuestionAnswering2019](https://doi.org/10.18653/v1/N19-1421).

From this question answering perspective, ScreenQA is a closed-domain
question answering task that expects answers by span (or UI element
phrase) selection for screen reading comprehension. As described in
Section <a href="#sec:problem_setting" data-reference-type="ref"
data-reference="sec:problem_setting">[sec:problem_setting]</a>, we
instructed the data annotators to avoid multi-hop, mathematical
counting, and logic reasoning, in order to focus on the fundamental
screen comprehension capacity.

[^1]: Also videos, if we consider consecutive screenshots. We leave out
    the video modality here in the context of annotating the underlying
    RICO screenshots.

[^2]: Also referred to as document analysis and recognition (DAR) or
    simply document understanding.

[^3]: Here we only include one or two examples per format and per
    capacity. This is by no means to be comprehensive.

# Annotation Method [sec:annotation_method]

<figure id="fig:flowchart">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/flowchart_3step.png" style="width:45.0%" />
<figcaption> ScreenQA annotation process. </figcaption>
</figure>

We perform several steps to collect the ScreenQA annotations, as
depicted in Figure <a href="#fig:flowchart" data-reference-type="ref"
data-reference="fig:flowchart">1</a>. Each step is described below.

<div class="figure*" markdown="1">

<figure id="fig:insync-occlusion">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/sync_in_sync.png" />
<figcaption>In-sync VH with occluded UI elements.</figcaption>
</figure>

<figure id="fig:insync-ghosting">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/sync_in_sync2.png" />
<figcaption>In-sync VH for main content, with ghosting VH from hamburger
menu.</figcaption>
</figure>

<figure id="fig:outofsync">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/sync_main_content_out_of_sync.png" />
<figcaption>Out-of-sync VH for main content, though top bar VH is
in-sync.</figcaption>
</figure>

</div>

## Pre-filtering [sec:pre-filtering]

The pre-filtering stage filters out 1) screenshots from non-English
apps[^1], and 2) screenshots whose view hierarchies (VHs) are out of
sync with the main contents. It is a known issue that in the RICO
dataset, some screenshots and their corresponding view hierarchies are
not perfectly in sync: there exists certain time difference between view
hierarchy extraction and screenshot capturing. We want to remove those
screenshots to ensure that all ScreenQA annotations are not subject to
such data noises.

Classifying the sync quality is tricky, even for human readers. One may
not be able to differentiate between occlusion, ghosting, and the actual
out-of-sync. See Figure <a href="#fig:vh-sync" data-reference-type="ref"
data-reference="fig:vh-sync">[fig:vh-sync]</a> for examples.
Accordingly, we instructed the annotators to focus on the main content
area of the screen and make sure the bounding boxes in that area are not
corrupted, as this is where most contents of interest and questions come
from.

We use 27 annotators to perform this step. Among RICO’s 66k unique
screenshots, about 11k screenshots are from non-English apps, and
about 13k screenshots have out-of-sync view hierarchies.[^2] With the
union of these two filtered out, there remains about 51k screenshots
from English apps with in-sync VHs.

## Question annotations [sec:question-annotation]

For question annotation, we asked the annotators to frame questions
given a screenshot as the context. The annotators were expected to
compose 1) natural, daily-life questions as if using the app. 2) The
composed questions should inquire information that can directly read off
from the screen and 3) should not require logical reasoning, counting
and calculation, mathematical comparison, etc. We further required the
annotators 4) not to ask questions about any advertisement on the
screen.

The annotation UI is depicted in
Appendix <a href="#appendix:question_annotation_ui" data-reference-type="ref"
data-reference="appendix:question_annotation_ui">[appendix:question_annotation_ui]</a>.
We asked the annotators to compose up to five questions given a
screenshot in the first pass. In the second pass, we asked for up to
three questions given a screenshot and the questions previously
composed. Each pass involved one annotator for each screenshot and
whoever annotated the screenshot before is excluded from being assigned
to the same screenshot. This ensures that every screenshot is assigned
precisely two annotators to compose questions. We chose this sequential
process 1) to avoid tricky deduplication of similar questions, and 2) to
encourage annotators to diversify their questions. Note that the same
set of annotators were involved in the both passes such that each
annotator had an opportunity to develop its own question style in the
first pass before seeing others’ in the second pass. This makes sure
that we still have certain numbers of question styles in the dataset
before they converge to each other in repeated passes.

We again involved the 27 annotators. The first pass of question
annotation generated 46k questions. The second pass added additional 36k
questions. These amount to a total of 82k questions, leaving about 15k
screenshots with no questions annotated, due to lack of interesting
contents.

## Answer annotations [sec:answer-annotation]

We used the total 82k questions of 35k screenshots from the previous
two-pass question annotation step to further annotate the corresponding
answers. The annotator who composed the question is excluded from
annotating its own answer to avoid potential biases. The answer
annotation UI is shown in
Appendix <a href="#appendix:answer_annotation_ui" data-reference-type="ref"
data-reference="appendix:answer_annotation_ui">[appendix:answer_annotation_ui]</a>.

Given an example, which contains a screenshot and a question, the
annotators are tasked to

1.  Fix any grammatical errors or typos of the given question without
    altering its intention.

2.  Answer the question, based on the context of the given screenshot,
    by 1) selecting bounding boxes from the underlying view hierarchy
    leaf nodes that contain the relevant answers, or drawing bounding
    boxes if no suitable leaf nodes can be used, and 2) ranking the
    answers in descending order of relevance if applicable, or by the
    common reading order.

3.  Additionally also provide a full-sentence answer to the question.

4.  Consider two exceptions: 1) The question may be incomprehensible or
    2) the screenshot does not contain the answer to the question, due
    to the questioner’s lack of understanding of the app. Then the
    example should be marked as “invalid question” and “not answerable
    from the screenshot”, respectively.

5.  One answer is annotated for the train split, and three for the
    validation and the test splits. This is to improve the evaluation
    quality. The data split details will be described in
    Section <a href="#sec:dataset-statistics" data-reference-type="ref"
    data-reference="sec:dataset-statistics">1.5</a>.

The “invalid question” annotations are then filtered out, and the
questions that have no other answer annotations are excluded from the
overall ScreenQA dataset, as they are considered incorrectly annotated
during the question annotation phase.

## Not-answerable question annotations [sec:not-answerable-question-annotation]

<figure id="fig:q_status">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/q_status_chart_v2.png" style="width:48.0%" />
<figcaption> Chart showing the fraction of questions with answers and
not answerable. Note that validation and test splits on average have
roughly 3 answers per question, so there are cases when some annotators
considered a question to be not answerable, while others provided an
answer to that same question. Specifically, the validation and the test
splits have 2.18% and 3.23% of such questions (the segments in red).
</figcaption>
</figure>

<div class="figure*" markdown="1">

<figure id="fig:no-answer-app-name">
<img
src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/no_answer_screenshots/rico_com.facechat.android_0_384_2.png" />
<figcaption>Question: ‘<em>What is the name of the
application?</em>’</figcaption>
</figure>

<figure id="fig:no-answer-email">
<img
src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/no_answer_screenshots/rico_com.jacobsmedia.weei_0_659_2.png" />
<figcaption>Question: ‘<em>What is the contact email for tech
support?</em>’</figcaption>
</figure>

<figure id="fig:no-answer-date">
<img
src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/no_answer_screenshots/rico_com.blogspot.fuelmeter_0_17_5.png" />
<figcaption>Question: ‘<em>What is the date of version
1.3.1?</em>’</figcaption>
</figure>

</div>

The questions marked as “not answerable from the screenshot” represent a
special category of questions that check model
overtriggering (attempting to answer those which are not supposed to be
answered). Being able to come to a conclusion that the answer is not
present on the screen is an important aspect of screen understanding.
Note that it is possible that one annotator considered a question to be
not answerable, and another provided an answer to that same question.

As described in
Section <a href="#sec:question-annotation" data-reference-type="ref"
data-reference="sec:question-annotation">1.2</a>, the first two passes
of question annotations aimed to compose questions that can be answered
from the screen, so as expected, the fraction of not answerable
questions was small. We then had a third pass of question annotation to
raise this fraction to nearly 10%, see
Figure <a href="#fig:q_status" data-reference-type="ref"
data-reference="fig:q_status">5</a>. For this, we used nearly 5k
screenshots selected randomly from those where there were no such
questions yet. In this pass, we asked annotators for exactly one
additional question per screenshot that had some relation to the
information there, but could not be answered. See examples in
Figure <a href="#fig:no-answer-examples" data-reference-type="ref"
data-reference="fig:no-answer-examples">[fig:no-answer-examples]</a>.
Answer annotation was not used for these 5k questions.

## Dataset statistics [sec:dataset-statistics]

<div id="tab:dataset_stats" markdown="1">

|            | Screenshots | Questions |
|:-----------|------------:|----------:|
| Train      |    $28,378$ |  $68,980$ |
| Validation |     $3,485$ |   $8,618$ |
| Test       |     $3,489$ |   $8,427$ |
| Total      |    $35,352$ |  $86,025$ |

ScreenQA dataset split stats.

</div>

The ScreenQA dataset contains 35,352 screenshots and 86,025 questions.

It is split into train, validation and test sets in approximately
80-10-10 ratio, see
Table <a href="#tab:dataset_stats" data-reference-type="ref"
data-reference="tab:dataset_stats">1</a>. Note that all questions for
the same screenshot belong to only one split.

[^1]: This is different from “non-English screenshots”, as translation
    and dictionary apps could pose confusion.

[^2]: This out-of-sync number is different
    from [liMappingNaturalLanguage2020a](https://doi.org/10.18653/v1/2020.acl-main.729) because we focus
    on the main content area.

# Annotation Analysis [sec:annotation_analysis]

We analyze the annotations of questions and answers in this section.

<div class="table*" markdown="1">

| Category | % | Examples |  |  |
|:---|---:|:---|:---|:---|
| 1-2 (lr)3-4 UI selection & config | 18.1 | Which option is selected? | What is the selected ringtone? |  |
| Quantity number | 11.7 | How many unread messages? | How many pictures are there in Western Europe? |  |
| App name | 10.4 | What is the name of the application? | What is the app name? |  |
| Date time | 9.4 | When was “Heal the Living” released? | When is happy hour? |  |
| Price | 3.4 | How much is the gift bonus in 3rd place? | What is the price? |  |
| Name of item | 3.3 | What is the name of the drug? | What is the name of chef? |  |
| User name | 2.8 | What is the name of the user? | What is the username on telegram? |  |
| Duration | 2.5 | What is the duration of video? | How long is the song? |  |
| Enum. of avail. options | 2.5 | Which social media options are given there? | What are the options available for logging in? |  |
| Address and direction | 2.4 | What is the current location? | What is the service zip code? |  |
| Email address | 2.4 | What is an email address? | What is customer service email? |  |
| Person’s name | 2.1 | Who sang the song? | What is the last name? |  |
| Signup/login | 1.6 | Which application can be used to sign up / login? | What are the alternative choices for signing up? |  |
| Version information | 1.6 | What is the version number? | What is the new feature in version v3.1.3? |  |
| Weather | 1.5 | What is the range of temperature shown on Sunday? | What is the weather forecast for Sunday? |  |
| Score & value | 1.4 | What is height/weight of the person? | What is the score? |  |
| Yes/No | 1.1 | Is there any travel plans? | Is there any favorite? |  |
| Phone number | 1.0 | What is the phone number? | What is the prefix for the international mobile number? |  |
| \# of Stars | 0.8 | What is the star rating? | How many stars are given to the product? |  |
| Share/sharing | 0.8 | Which application can be used to share? | Where can I share this application? |  |
| Age | 0.8 | How old is ...? | What is the age? |  |
| Percentage | 0.7 | What is the percentage of ... ? | What is the brightness percentage for foreground? |  |
| Settings | 0.6 | What is the setting of ... ? | Which settings are switched on? |  |
| Quantity amount | 0.6 | How much fat is there? | What is the amount? |  |
| Permission | 0.5 | Which application is asking for permissions? | What permissions are required for MyCarTracks? |  |
| \# of Likes | 0.5 | How many likes for ... ? | How many likes does ... get? |  |
| Country | 0.5 | What is the name of the country? | Which country has the +54 code? |  |
| Distance | 0.5 | What is the visibility distance? | How far is it from ... ? |  |
| \# of Reviews | 0.4 | What is the number of comments on ... ? | How many comments? |  |
| Website | 0.3 | What is the url? | What’s the website address? |  |
| Gender | 0.3 | What is the gender? | Which gender is displayed on the screen? |  |
| How to | 0.3 | How to start on boot? | How to pronounce his name? |  |
| Currency | 0.3 | What is the currency? | What is the currency for the price? |  |
| Unit of measurement | 0.2 | What is the unit of temperature? | What is the unit of weight and length? |  |
| Language | 0.1 | Which language is used in the setting? | Which language is being translated into which language? |  |
| Color | 0.0 | What is the UI color? | What is the amount of green color? |  |
| 1-2 (lr)3-4 Others | 12.8 | What’s the average speed? | What is the user’s middle initial |  |
|  |  | What is the spending limit? | Which team has 41 points? |  |
| 1-2 Total | 100.0 |  |  |  |

</div>

<div class="figure*" markdown="1">

<figure id="fig:q_num">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/q_num_v1.1.png" />
<figcaption> Number of composed questions per screenshot. </figcaption>
</figure>

<figure id="fig:a_num">
<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/a_num_v1.1.png" />
<figcaption> Number of bounding boxes used to answer the question.
</figcaption>
</figure>

</div>

## Question analysis [sec:question-analysis]

We collected overall 86k questions over 35k unique screenshots from
RICO. Among the 86k questions, there are 47.5k unique questions.[^1]
Some screenshots receive more questions because they usually contain
more information to be asked about. Yet, the histogram still exhibits a
reasonable exponential decay with a mild slope, as depicted in
Figure <a href="#fig:q_num" data-reference-type="ref"
data-reference="fig:q_num">1</a>.

To further understand what questions have been asked, we categorize the
questions using regular expressions based on a list of empirically
determined question categories. The categories are meant to provide a
rough overview of the question annotations and by no means to provide a
precise categorization. The distribution and examples by these
categories are tabulated in
Table <a href="#tab:q_cate" data-reference-type="ref"
data-reference="tab:q_cate">[tab:q_cate]</a>. Note that the questions
were not composed at the annotators’ full discretion: They are
conditioned on the given screenshots. That is to say, the distribution
is implicitly influenced by the RICO crawling process. For example, as
RICO crawled screen traces from freshly installed apps and did not login
an account, a noticeable number of the screen traces end at a login
page. This in turn translates to a higher percentage of questions asked
about app names, email addresses, permissions to login, etc.

## Answer analysis [sec:answer-analysis]

We analyze the answer annotations in two aspects: 1) How often do we
need more than one bounding box and its text to answer the question,
and 2) How often do human annotators find the view hierarchy useful to
provide a minimal answer to the question.

Figure <a href="#fig:a_num" data-reference-type="ref"
data-reference="fig:a_num">2</a> illustrates the histogram of number of
bounding boxes used in each answer. About 84% of answers contain a
single bounding box. Among these single-bounding-box answers, 51% uses a
VH leaf node directly, while 49% uses a manually drawn bounding box. If
we consider all answers together, excluding cases when there is no
answer, still 51% uses VH leaf nodes entirely, while 48% uses manually
drawn bounding boxes. That is, for about half of the total number of
screenshots, human annotators preferred to manually draw the bounding
boxes in order to provide answers that minimally satisfy the question.
This observation reflects the necessity not to require the view
hierarchy input for ScreenQA as described in
Task <a href="#task:sreenqa" data-reference-type="ref"
data-reference="task:sreenqa">[task:sreenqa]</a>.

Interestingly, there exist some cases, about 0.8% of the questions, that
the human annotators used a mixture of VH leaf nodes and manually drawn
bounding boxes as their full answer. By inspecting those cases, we found
that these usually happen 1) when the answer is an enumeration of
“inhomogeneous” options that are organized differently on the screen,
such as using email vs. other APIs to login, and 2) when an answer needs
multiple parts to be complete, such as a date consisting of year, month,
and day scattered on the calendar UI, and a temperature or a measurement
requiring a number followed by the corresponding unit. These parts may
not be displayed in the same way, resulting in lack of useful VH leaf
nodes for some of the parts.

[^1]: Note that it is natural and valid to ask the same common questions
    over various screenshots, for example, “Which option is selected on
    the screen?” and “What is the email address?”

# Baselines [sec:baseline]

# Results [sec:result]

# Conclusion

In this work, we proposed the ScreenQA task. We annotated a large-scale
ScreenQA dataset, which contains 86,025 question-answer pairs. Compared
to other vision-language multimodal problems, such as document image
understanding and visual question answering, ScreenQA poses its unique
challenges: rich in text, diverse in apps, and blended with icons and
symbols. We hope to use the ScreenQA task and the dataset to encourage
the community to look into this screen content understanding problem, as
it enables new technologies and new user experiences.

# Acknowledgements

The authors would like to thank Srinivas Sunkara for his valuable
discussions and comments on this manuscript.

# Data annotation interfaces for question and answer collection [appendix:annotation_ui]

## Question annotation interface [appendix:question_annotation_ui]

<div class="figure*" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/qg.png" alt="image" />

</div>

The question annotation interface is shown in
Figure <a href="#fig:question-annotation-ui" data-reference-type="ref"
data-reference="fig:question-annotation-ui">[fig:question-annotation-ui]</a>.
Question annotation was performed in a sequential manner, the later and
non-overlapping annotators can see all previous questions to diversify
question framing and avoid duplication. We also used the sequential
process to provide more feedback and training to the annotators for
quality improvement.

## Answer annotation interface [appendix:answer_annotation_ui]

<div class="figure*" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2209.08199v2_md/fig/ag.png" alt="image" />

</div>

The answer annotation interface is shown in
Figure <a href="#fig:answer-annotation-ui" data-reference-type="ref"
data-reference="fig:answer-annotation-ui">[fig:answer-annotation-ui]</a>.
Answer annotators were tasked to determine if the question is valid and
if the question is answerable from the screen context. If both are
positive, the annotators need to answer the questions by 1) selecting or
drawing the bounding boxes of UI elements, 2) filling the text for each
selected/drawn bounding box on right right, and 3) ranking them
appropriately. The annotators were also tasked to review and make
necessary corrections if the question has grammatical errors or typos.