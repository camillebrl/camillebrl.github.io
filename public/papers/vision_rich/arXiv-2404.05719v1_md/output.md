<figure id="fig:opening_example">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/opening_example.png" style="width:110.0%" />
<figcaption>is able to perform <em>referring</em> tasks (<em>e.g., <span
style="color: WildStrawberry">widget classification</span>, <span
style="color: Goldenrod">icon recognition</span>, <span
style="color: Aquamarine">OCR</span>)</em> with flexible input formats
(point, box, scribble) and <em>grounding</em> tasks (<em>e.g., <span
style="color: WildStrawberry">find widget</span>, <span
style="color: Goldenrod">find icon</span>, <span
style="color: Aquamarine">find text</span>, <span
style="color: NavyBlue">widget listing)</span></em> on mobile UI
screens. These elementary tasks equip the model with rich visual and
spatial knowledge, enabling it to distinguish UI types at both coarse
and fine levels, such as between various icons or text elements. This
foundational knowledge is crucial for performing more advanced tasks.
Specifically, is able to not only discuss visual elements in <em><span
style="color: cyan">detailed description</span></em> and <em><span
style="color: cyan">perception conversation</span></em>, but also
propose goal-oriented actions in <em><span
style="color: cyan">interaction conversation</span></em> and deduce the
overall function of the screen via <em><span
style="color: cyan">function inference</span></em>.</figcaption>
</figure>

# Introduction

Mobile applications have become an important part of daily life, serving
as tools for individuals to achieve personal goals including searching
for information, making reservations, and seeking entertainment. In this
usage, we inspect the current screen visually, and perform the desired
actions based on our goals. Automating this process of perception and
interaction has the potential to help users achieve their goals with
relative ease. Moreover, it is also a valuable building block for
accessibility [edwards1995access](http://arxiv.org/pdf/2306.06811v1), multi-step UI
navigation
[hong2023cogagent](http://arxiv.org/pdf/2402.11941v2), [zhang2023appagent](https://arxiv.org/pdf/2312.13771), [wang2024mobileagent](https://arxiv.org/pdf/2401.16158),
app testing [amalfitano2011gui](http://arxiv.org/pdf/1911.05403v2), [linares2017continuous](http://arxiv.org/pdf/1801.06267v1),
usability studies [jiang2018usability](http://arxiv.org/pdf/2305.03271v2), and many others.

To facilitate seamless automation of perception and interaction within
user interfaces, a sophisticated system endowed with a set of key
capabilities is essential. Such a system must possess the ability to not
only comprehend the entirety of a screen but also to concentrate on
specific UI elements within that screen. With visual understanding as
the foundation, it should further be able to map natural language
instructions to corresponding actions within a given UI, execute
advanced reasoning, and provide exhaustive details concerning the
screens it interacts with. These requirements necessitate the
development of a vision-language model adept at both referring and
grounding in relation to UI screens. Here, *referring* requires the
system to utilize particular regional image information in the screen
input, while *grounding* involves the model’s capacity to identify and
denote precise locations on the screen in its outputs.

Existing approaches are insufficient in fully addressing these key
capabilities. On one hand, while Multimodal Large Language Models
(MLLMs) like Ferret [you2023ferret](https://arxiv.org/pdf/2310.07704),
Shikra [chen2023shikra](http://arxiv.org/pdf/2306.15195v2), and
Kosmos2 [peng2023kosmos](http://arxiv.org/pdf/2305.16103v1) demonstrate strong referring and
grounding capabilities, their scope is mainly restricted to natural
images. Directly adapting these models to UI screens can be limiting,
since UI screens typically exhibit more elongated aspect ratios and
contain smaller objects of interests (*e.g.*, icons and texts) than
natural images. Relying solely on a directly resized, low-resolution
global image could lead to loss of important visual signals that are
essential for screen understanding and interaction. On the other hand,
other works targeting directly at UI tasks have primarily focused on
processing entire screens as singular inputs (*e.g.*,
Pix2Struct [lee2023pix2struct](http://arxiv.org/pdf/2210.03347v2),
ILuvUI [jiang2023iluvui](https://arxiv.org/pdf/2310.04869),
CogAgent [hong2023cogagent](http://arxiv.org/pdf/2402.11941v2)), only supports referring
tasks with one bounding box in the input (*e.g.*,
Spotlight [li2023spotlight](https://arxiv.org/pdf/2209.14927)), and leveraging
GPT-4V [yang2023dawn](https://arxiv.org/pdf/2309.17421) to navigate UI screens, as seen in
MM-Navigator [yan2023gpt](http://arxiv.org/pdf/2311.07562v1),
AppAgent [zhang2023appagent](https://arxiv.org/pdf/2312.13771), and
MobileAgent [wang2024mobileagent](https://arxiv.org/pdf/2401.16158). Furthermore, the tasks
studied in these work do not comprehensively cover all dimensions of UI
screen understanding.

In this paper, we present Ferret-UI, the first MLLM designed to execute
precise referring and grounding tasks specific to UI screens, while
adeptly interpreting and acting upon open-ended language instructions.
We address the aforementioned limitations by focusing on three pivotal
dimensions: ($i$) improved model architecture, ($ii$) data curation, and
($iii$) benchmark establishment. For model architecture, we base our
approach on Ferret [you2023ferret](https://arxiv.org/pdf/2310.07704), an MLLM known for its
strong performances in referring and grounding with natural images. We
posit that such capabilities provide a solid foundation in interactive
UI-centric tasks. For flexible adaptation of UI screen aspect ratios, we
integrate “any resolution” (anyres) into Ferret similar to
[liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [lin2023sphinx](https://arxiv.org/pdf/2311.07575), [gao2024sphinxx](https://arxiv.org/pdf/2402.05935), but
with pre-defined grid configurations to divide the full image into
sub-images so that both portrait and landscape screens can be
accommodated. As later shown in Fig.
<a href="#fig:ferret-ui-architecture" data-reference-type="ref"
data-reference="fig:ferret-ui-architecture">[fig:ferret-ui-architecture]</a>,
sub-image features are used in addition to global image features to help
magnify details and provide enhanced visual features.

To train Ferret-UI, we generate data at different granularities,
covering basic semantic and spatial tasks for UI primitives to advanced
reasoning tasks. We first generate training samples for elementary UI
tasks using a template-based approach. This encompasses *referring*
tasks such as *widget classification*, *icon recognition*, *OCR*, and
*grounding* tasks like *find widget*, *find icon*, *find text*, and
*widget listing*. These tasks are instrumental in teaching the model to
understand the semantics and spatial positioning of UI elements,
enabling the model to make distinctions at both a broad level (among
various UI types) and a more detailed level (within specific UI types,
such as icons or text). For advanced tasks, we use
GPT-4 [openai2024gpt4](https://arxiv.org/pdf/2303.08774) to generate data, including
*detailed description*, *conversation perception*, *conversation
interaction*, and *function inference*. These advanced tasks prepare the
model to engage in more nuanced discussions about visual components,
formulate action plans with specific goals in mind, and interpret the
general purpose of a screen. Fig.
<a href="#fig:opening_example" data-reference-type="ref"
data-reference="fig:opening_example">1</a> illustrates examples of
Ferret-UI’s proficiency in handling the 11 tasks ranging from basic to
advanced.

To assess these capabilities, we develop a comprehensive test benchmark
featuring 14 diverse mobile UI tasks in terms of referring and
grounding. This includes 3 tasks from
Spotlight [li2023spotlight](https://arxiv.org/pdf/2209.14927) (*screen2words*, *widget
captions*, and *taperception*), and dual versions of the 11 UI tasks
previously described, tailored for both iPhone and Android screens. We
conduct comprehensive evaluation of a variety of UI understanding
models, including both open-source MLLMs (*e.g.*, CogAgent
[hong2023cogagent](http://arxiv.org/pdf/2402.11941v2) and Fuyu [fuyu-8b](https://www.adept.ai/blog/fuyu-8b)) and
GPT-4V. We observe that Ferret-UI significantly surpasses the base
Ferret model, illustrating the importance of domain-specific model
training. Compared to GPT-4V, Ferret-UI demonstrates superior
performance in elementary UI tasks. Notably, in the context of advanced
tasks, Ferret-UI surpasses both Fuyu and CogAgent.

Our contributions are summarized as follows. ($i$) We propose Ferret-UI
with “any-resolution” (anyres) integrated to flexibly accommodate
various screen aspect ratios. It represents the first UI-centric MLLM
that is capable of effectively executing referring, grounding, and
reasoning tasks. ($ii$) We define a set of elementary and advanced UI
tasks, for which we have meticulously gathered training samples for
model training. ($iii$) We develop a comprehensive test benchmark
encompassing all the tasks under investigation. Through careful
experiments and analysis, we offer insights into the model’s
capabilities and limitations.

# Related Work [sec:related_work]

Earlier works
[shi2017world](http://arxiv.org/pdf/2401.03546v1), [liu2018reinforcement](http://arxiv.org/pdf/1802.08802v1), [gur2018learning](http://arxiv.org/pdf/2103.01991v1), [li2020mapping](http://arxiv.org/pdf/2005.03776v2), [burns2022dataset](http://arxiv.org/pdf/2202.02312v3)
in the area focus on studying simplified web and mobile screens. With
recent advances in both
LLMs [touvron2023llama](http://arxiv.org/pdf/2402.08075v1), [openai2024gpt4](https://arxiv.org/pdf/2303.08774), [gu2023mamba](http://arxiv.org/pdf/2403.16371v1), [jiang2023mistral](http://arxiv.org/pdf/2401.13565v3), [huang2023language](https://arxiv.org/pdf/2302.14045), [driess2023palm](http://arxiv.org/pdf/2302.14030v3), [anil2023palm](http://arxiv.org/pdf/2305.10403v3)
and
MLLMs [liu2023llava](https://arxiv.org/pdf/2304.08485), [zhu2023minigpt](http://arxiv.org/pdf/2402.17510v1), [ye2023mplug](http://arxiv.org/pdf/2405.00390v2), [li2023otter](http://arxiv.org/pdf/2311.00233v2), [dai2023instructblip](http://arxiv.org/pdf/2311.00233v2), [sun2023generative](http://arxiv.org/pdf/2203.15788v1), [mckinzie2024mm1](http://arxiv.org/pdf/2403.01757v1), [li2023multimodal](http://arxiv.org/pdf/2309.10020v1),
the approaches to many research problems have been transformed,
including UI understanding. Several works have explored the use of MLLMs
for UI tasks. Specifically, ILuvUI [jiang2023iluvui](https://arxiv.org/pdf/2310.04869) and
Spotlight [li2023spotlight](https://arxiv.org/pdf/2209.14927) concentrate on single-screen
UI tasks while exploring various UI tasks by fine-tuning on
GPT-generated data and delving into UI tasks such as screen
summarization and widget interaction.

MobileAgent [wang2024mobileagent](https://arxiv.org/pdf/2401.16158) and AppAgent
[zhang2023appagent](https://arxiv.org/pdf/2312.13771) represent a different approach,
utilizing MLLMs as agents for UI screen navigation, with MobileAgent
employing external detection modules for action generation and AppAgent
leveraging overlaid UI element IDs and screen XML files for predefined
actions. CogAgent [hong2023cogagent](http://arxiv.org/pdf/2402.11941v2), built upon CogVLM
[wang2023cogvlm](http://arxiv.org/pdf/2210.00066v1), shifts the focus towards using only
screen images for complex UI navigation, eliminating the need for
UI-specific modules. Here are some more examples among other works that
utilize LLMs
[kim2023language](https://arxiv.org/pdf/2303.17491), [zheng2024synapse](https://arxiv.org/pdf/2306.07863), [deng2024mind2web](http://arxiv.org/pdf/2306.06070v3), [gur2023real](http://arxiv.org/pdf/2307.12856v4)
and MLLMs
[shaw2024pixels](http://arxiv.org/pdf/2306.00245v2), [zhan2023you](http://arxiv.org/pdf/2401.05851v1), [yan2023gpt](http://arxiv.org/pdf/2311.07562v1), [gao2023assistgui](http://arxiv.org/pdf/2401.10935v2), [zheng2024gpt](http://arxiv.org/pdf/2401.01614v2), [cheng2024seeclick](https://arxiv.org/pdf/2401.10935), [baechler2024screenai](http://arxiv.org/pdf/2402.04615v2)
in the space.

In this work, we focus on fine-grained mobile UI understanding with
MLLMs. Naturally, our work also aligns with the recent burgeoning
literature focused on empowering MLLMs for referring and grounding
tasks [zhang2023gpt4roi](http://arxiv.org/pdf/2309.12109v1), [chen2023shikra](http://arxiv.org/pdf/2306.15195v2), [peng2023kosmos](http://arxiv.org/pdf/2305.16103v1), [lai2023lisa](http://arxiv.org/pdf/2404.08767v1), [zhao2023bubogpt](http://arxiv.org/pdf/2405.17104v2), [you2023ferret](https://arxiv.org/pdf/2310.07704), [zhang2023llava](http://arxiv.org/pdf/2312.02949v1).

<figure id="fig:ferret-ui-architecture">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/ferret-ui-arch.png" />
<figcaption>Overview of Ferret-UI-anyres architecture. While
Ferret-UI-base closely follows Ferret’s architecture, Ferret-UI-anyres
incorporates additional fine-grained image features. Particularly, a
pre-trained image encoder and projection layer produce image features
for the entire screen. For each sub-image obtained based on the original
image aspect ratio, additional image features are generated. For text
with regional references, a visual sampler generates a corresponding
regional continuous feature. The LLM uses the full-image representation,
sub-image representations, regional features, and text embeddings to
generate a response.</figcaption>
</figure>

# Method

Ferret-UI is built upon Ferret [you2023ferret](https://arxiv.org/pdf/2310.07704), which is
a MLLM that excells in spatial referring and grounding within natural
images of diverse shapes and levels of detail. It can interpret and
interact with regions or objects, whether they are specified as points,
boxes, or any free-form shapes. Ferret contains a pre-trained visual
encoder (*e.g.*, CLIP-ViT-L/14) [radford2021learning](http://arxiv.org/pdf/2404.19696v1) and
a decoder-only language model (*e.g.*,
Vicuna [zheng2023judging](https://arxiv.org/pdf/2306.05685)). Furthermore, Ferret
incorporates a unique hybrid representation technique that transforms
specified regions into a format suitable for processing by the LLM. At
its core, a spatial-aware visual sampler is designed to adeptly manage
continuous features of region shapes in different sparsity levels.

To instill UI expert knowledge into Ferret, we make two extensions to
develop Ferret-UI: ($i$) the definition and construction of UI referring
and grounding tasks
(Section <a href="#sec:dataset" data-reference-type="ref"
data-reference="sec:dataset">[sec:dataset]</a>); and ($ii$) model
architecture adjustment to better deal with screen data. Specifically,
Ferret-UI includes a broad range of UI referring tasks (*e.g.*, OCR,
icon recognition, widget classification) and grounding tasks (*e.g.*,
find text/icon/widget, widget listing) for model training, building up a
strong UI understanding foundation for advanced UI interactions. Unlike
previous MLLMs that require external detection modules or screen view
files, Ferret-UI is self-sufficient, taking raw screen pixels as model
input. This approach not only facilitates advanced single-screen
interactions, but also paves the way for new applications, such as
improving accessibility. Initial explorations of the dataset result in
two modeling insights: ($i$) UI screens are predominantly characterized
by aspect ratios that are more extended compared to those found in
natural images, as evidenced in
Tab. <a href="#tab:screen_num_distribution" data-reference-type="ref"
data-reference="tab:screen_num_distribution">[tab:screen_num_distribution]</a>;
($ii$) the tasks involve many objects of interest (*i.e.*, UI widgets
like icons and texts) that are significantly smaller than the objects
typically observed in natural images. For example, many questions focus
on icons that occupy less than 0.1% of the entire screen. Thus, relying
solely on a single directly resized, low-resolution global image could
lead to significant loss of visual details.

To address this problem, we apply the idea of “any resolution” (anyres),
as advocated in SPHINX [lin2023sphinx](https://arxiv.org/pdf/2311.07575), [gao2024sphinxx](https://arxiv.org/pdf/2402.05935),
LLaVA-NeXT [liu2024llavanext](https://llava-vl.github.io/blog/2024-01-30-llava-next/), and
Monkey [li2023monkey](http://arxiv.org/pdf/2103.15488v1), to Ferret-UI. Specifically, we opt
for two grid configurations, 1x2 and 2x1, which are chosen based on the
aspect ratios of the original screens as depicted in
Tab. <a href="#tab:screen_num_distribution" data-reference-type="ref"
data-reference="tab:screen_num_distribution">[tab:screen_num_distribution]</a>.
Given a screen, the grid configuration that most closely matches its
original aspect ratio is selected. Subsequently, the screen is resized
to fit the selected grid configuration and is then partitioned into
sub-images. Intuitively, portrait screens are divided horizontally,
whereas landscape screens are divided vertically. All sub-images are
encoded separately using the same image encoder, and the LLM uses all
visual features of varying granularity with both the full image context
as well as the enhanced details. The overall architecture of Ferret-UI,
including the any-resolution adjustments, is illustrated in
Fig. <a href="#fig:ferret-ui-architecture" data-reference-type="ref"
data-reference="fig:ferret-ui-architecture">1</a>.

# Dataset and Task Formulation [sec:dataset]

In this section, we detail the process of generating datasets for model
training and evaluation. Specifically, we describe the UI detection data
collection process in
Section <a href="#sec: ui_data" data-reference-type="ref"
data-reference="sec: ui_data">1.1</a>, and we outline how we create
task-specific data from raw detections in
Section <a href="#sec: task_formulation" data-reference-type="ref"
data-reference="sec: task_formulation">1.2</a>.

## UI Data Collection [sec: ui_data]

**UI Screens.** To build a model capable of perceiving and interacting
with mobile screens, it is crucial to gather a varied collection of such
screens. This study examines screens from both iPhone and Android
devices.

For Android screens, we use a subset of the RICO dataset
[deka2017rico](http://arxiv.org/pdf/1607.07515v3). Specifically, we consider the tasks in
Spotlight [li2023spotlight](https://arxiv.org/pdf/2209.14927), whose data is publicly
available, including *screen2words*, *widgetcaptions*, and
*taperception*. We aggregate unique images for each split (train and
test) among the tasks to form our own data splits. In total, there are
26,527 train images and 3,080 test images.

For iPhone screens, we use the AMP dataset
[zhang2021screenrecognition](https://arxiv.org/pdf/2101.04893), which spans a broad
spectrum of applications. A subset is randomly selected and divided into
training and test splits. The iPhone screens come in various sizes,
resulting in a total of 84,685 training images and 9,410 test images.
The breakdown of image sizes is summarized in Tab.
<a href="#tab:screen_num_distribution" data-reference-type="ref"
data-reference="tab:screen_num_distribution">[tab:screen_num_distribution]</a>.

**UI Screen Elements Annotation.** After collecting Android and iPhone
screens, we further collect fine-grained element annotation from screens
using a pre-trained pixel-based UI detection
model [zhang2021screenrecognition](https://arxiv.org/pdf/2101.04893). For each detected UI
element, the output includes a UI type (Button, Text, Icon, Picture,
*etc.*), the corresponding bounding box, and the text displayed on it,
if any, identified by the Apple Vision Framework[^1]. We further use
heuristics from Screen
Recognition [zhang2021screenrecognition](https://arxiv.org/pdf/2101.04893) to group
individual detections into larger units, *e.g.*, multiple lines of text
are merged into one group, an image is grouped with its caption, *etc*.

## Task Formulation [sec: task_formulation]

This section describes how we convert the UI screens along with the
associated detection data to a format that can be used to train an MLLM.
We elaborate three different approaches devised for the construction of
the dataset.

<figure id="fig:elementary_task_datagen">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/elementary_task_generation.png" />
<figcaption><strong>Elementary task data generation overview</strong>. A
UI detector outputs all detected elements, with each element’s
<em>type</em>, <em>text</em>, and <em>bounding boxes</em>. These
detections are used to create training samples for elementary tasks. For
<em>grounding tasks</em>, we use all element detections to create one
sample for widget listing whereas the remaining tasks focus on one
element at a time. We separate the elements into <em>icons</em>,
<em>text</em>, and <em>non-icon/text widgets</em>. For each type, we
create one referring and one grounding sample.</figcaption>
</figure>

**Reformatting Spotlight.** We first take *screen2words*,
*widgetcaptions*, and *taperception* from the existing Spotlight
tasks [li2023spotlight](https://arxiv.org/pdf/2209.14927), and format them into
conversational QA pairs. Specifically, GPT-3.5 Turbo is used to create a
varied set of prompts from base prompts we author for respective tasks:

-   **Screen2words***: Provide a summary of this screenshot*;

-   **Widget Captions***: For the interactive element \[bbox\], provide
    a phrase that best describes its functionality*;

-   **Taperception***: Predict whether the UI element \[bbox\] is
    tappable*.

For each training example, we sample a prompt for the corresponding task
and pair it with the original source image and ground-truth answer.

**Elementary Tasks.** In addition to the Spotlight tasks, we use paired
screens and UI elements mentioned in Section
<a href="#sec: ui_data" data-reference-type="ref"
data-reference="sec: ui_data">1.1</a> to generate data for novel UI
tasks that rely on grounding and referring capabilities. We introduce 7
tasks using this approach, one set for each of Android and iPhone
screens: *OCR*, *icon recognition*, and *widget classification* for
*referring*; and *widget listing*, *find text*, *find icon*, and *find
widget* for *grounding*. We define *referring tasks* as the ones with
bounding boxes in the inputs, while *grounding tasks* are the ones with
bounding boxes in the outputs.

For each task, we also use GPT-3.5 Turbo to expand a base prompt to
introduce variants of the task question. Details for data generation are
illustrated in Fig.
<a href="#fig:elementary_task_datagen" data-reference-type="ref"
data-reference="fig:elementary_task_datagen">1</a>. The number of
training samples for each task is summarized in Tab.
<a href="#tab:task_data_num_distribution" data-reference-type="ref"
data-reference="tab:task_data_num_distribution">[tab:task_data_num_distribution]</a>.
The number of test samples for all tasks are 5K. In experiments, we
sample from this pool of training data with different ratios to
construct our training data mixture.

<figure id="fig:advanced_task_data_gen">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/advanced_task_generation.png"
style="width:98.0%" />
<figcaption><strong>Advanced task data generation overview.</strong> We
first normalize bounding box coordinates from the detection outputs,
then we send the detections, prompts, and optional one-shot example to
GPT-4. For detailed description and function inference, we pair the
generated response with a pre-selection of prompts to train Ferret-UI.
For conversation tasks, we directly transform GPT-4 output to multi-turn
conversations.</figcaption>
</figure>

**Advanced Tasks.** To incorporate reasoning abilities into our model,
we follow LLaVA [liu2023llava](https://arxiv.org/pdf/2304.08485), and additionally collect
data of 4 more formats using GPT-4. We focus on iPhone screens for this
part of the data collection, filtering our examples to those with more
than 2 but fewer than 15 detections. These examples are sent together
with prompts to GPT-4 to create data of the desired format—the actual
images are not used. Fig.
<a href="#fig:advanced_task_data_gen" data-reference-type="ref"
data-reference="fig:advanced_task_data_gen">2</a> illustrates the
training data generation process for advanced tasks.

The four tasks are *detailed description*, *conversation perception*,
*conversation interaction*, and *function inference*. Among these, we
expand base prompts for detailed description and function inference to
pair them with the GPT-4 response as the input data in our model
training. For conversations, we provide an in-context example for GPT-4
to better follow bounding box formats in its output. From the raw GPT-4
output, we parse the bounding boxes and transform them into the correct
multi-turn conversation format for our model. In total, we have created
40K valid conversations from GPT-4 generated data. More details about
our data collection pipeline and detailed analysis of our collected data
are provided in the Appendix.

While our training data collection primarily targets iPhone screens, we
assemble test sets for both iPhone and Android platforms. For each task,
we select 25 test screens from iPhone and 5 from Android. Due to
overlaps in images across different tasks, the total number of unique
images amounts to 56 for iPhone and 13 for Android. For evaluation, we
randomly select 2 QA pairs for the conversational tasks, creating two
distinct test instances with precisely one question in each input.
Utilizing these test images, we formulate 20/40/38/20 questions for
iPhone and 5/10/10/10 questions for Android, for the four tasks,
respectively.

[^1]: https://developer.apple.com/documentation/vision

# Experiments

We first present our main results in
Section <a href="#sec:main_results" data-reference-type="ref"
data-reference="sec:main_results">1.1</a>, followed by ablation studies
in Section <a href="#sec:ablation_studies" data-reference-type="ref"
data-reference="sec:ablation_studies">1.2</a>. Then, detailed analysis
of results on elementary and advanced UI tasks is provided in
Section <a href="#sec:analysis_1" data-reference-type="ref"
data-reference="sec:analysis_1">1.3</a> and
<a href="#sec:analysis_2" data-reference-type="ref"
data-reference="sec:analysis_2">1.4</a>, respectively.

**Setup.** In this section, Ferret-UI-anyres refers to the version with
any-resolution integrated, Ferret-UI-base refers to the version directly
following the Ferret architecture, and Ferret-UI refers to both
configurations. During training, both the decoder and the projection
layer are updated while the vision encoder is kept frozen. All the
training data is formatted into the instruction-following format, and
the training objective is the same as in Ferret. In total, our training
mixture has 250K samples. Ferret-UI-base takes 1 day to train while
Ferret-UI-anyres takes about 3 days on 8 A100 GPUs.

## Results [sec:main_results]

We compare the performances of Ferret-UI-base, Ferret-UI-anyres,
Ferret[^1], and GPT-4V for all tasks. We also include
Fuyu [fuyu-8b](https://www.adept.ai/blog/fuyu-8b) and
CogAgent’s [hong2023cogagent](http://arxiv.org/pdf/2402.11941v2) performance on advanced
tasks.[^2] Results are summarized in Tab.
<a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">[tab:main_results]</a>, where the
average performance within a category is reported. Performance breakdown
for elementary and advanced tasks is shown in Fig.
<a href="#fig:elementary_task_perf" data-reference-type="ref"
data-reference="fig:elementary_task_perf">1</a> and Tab.
<a href="#Tab:advanced_task_perf" data-reference-type="ref"
data-reference="Tab:advanced_task_perf">1</a>, respectively.

<figure id="fig:elementary_task_perf">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/elementary_tasks_combined.png" />
<figcaption>Elementary task performance comparison. Numerous small
widgets present on the Android screen make it more challenging for
referring and grounding, while Ferret-UI continues to outperform Ferret
and GPT-4V on almost all the elementary tasks.</figcaption>
</figure>

<div id="Tab:advanced_task_perf" markdown="1">

|  |  |  |  |  |  |  |  |  |  |  |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 2-6 (lr)7-11 | DetDes | ConvP | ConvI | FuncIn | **Avg** | DetDes | ConvP | ConvI | FuncIn | **Avg** |
| Ferret [you2023ferret](https://arxiv.org/pdf/2310.07704) |  |  |  |  |  |  |  |  |  |  |
| Fuyu [fuyu-8b](https://www.adept.ai/blog/fuyu-8b) |  |  |  |  |  |  |  |  |  |  |
| CogAgent [hong2023cogagent](http://arxiv.org/pdf/2402.11941v2) |  |  |  |  |  |  |  |  | **90.5** |  |
| Ferret-UI-base |  |  |  |  |  |  |  |  |  |  |
| Ferret-UI-anyres | **97.4** |  |  | **95.2** |  |  |  |  |  |  |
| GPT-4V [achiam2023gpt](http://arxiv.org/pdf/2311.15732v2) |  | **105.6** | **198.5** |  | **114.3** | **126.6** | **109.4** | **188.6** |  | **128.2** |

Advanced task performance comparison. *DetDes*: detailed description,
*ConvP*: conversation perception, *ConvI*: conversation interaction,
*FuncIn*: function inference.

</div>

**Public Benchmark from Spotlight [li2023spotlight](https://arxiv.org/pdf/2209.14927).**
Compared to Spotlight, Ferret-UI demonstrates superior performance in
*S2W* and *WiC*, even though Spotlight uses 80M web page screenshots and
2.69M mobile screenshots for pre-training. Ferret-UI performance falls
short on *TaP* but is still competitive; our studies further suggest
that this could be due to the noisiness of the taperception labels.
Detailed analysis is provided in the Appendix.

**Results on Elementary UI Tasks.** The average performance of all
referring and grounding tasks is summarized in Tab.
<a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">[tab:main_results]</a>, and the
performance breakdown for each task is shown in Fig.
<a href="#fig:elementary_task_perf" data-reference-type="ref"
data-reference="fig:elementary_task_perf">1</a>. For referring tasks, we
report exact match accuracy for OCR and accuracy for icon recognition
and widget classification. For each grounding task, we also report the
accuracy, where a correct bounding box is one that has an
Intersection-over-Union (IoU) with the label greater than the threshold
(0.5). Widget listing performance is not included in the average as we
treat it as an auxiliary task.

Ferret-UI outperforms Ferret and GPT-4V in most elementary tasks except
for iPhone *find text*. While GPT-4V demonstrates decent performance on
iPhone tasks, its performances on Android tasks, especially grounding
tasks, are significantly worse. Examining the predictions shows that
Android screens have more numerous and smaller widgets, making the
grounding tasks more challenging. Furthermore, Ferret-UI’s zero-shot
performance on the Referring Expression Comprehension task from
UIBert [bai2021uibert](https://arxiv.org/pdf/2107.13731) is 76% when we frame it as the
*find widget* task. Notably, with anyres added to Ferret-UI-base, iPhone
referring and grounding tasks improve by 2 points.

**Results on Advanced Tasks.** The breakdown of task performance for
advanced tasks is shown in Tab.
<a href="#Tab:advanced_task_perf" data-reference-type="ref"
data-reference="Tab:advanced_task_perf">1</a>. As the advanced tasks
require open-ended responses, we use GPT-4 to score both the label and
the prediction. We report *score for prediction* over *score for label*
as a percentage.

Ferret-UI exhibits commendable performance on advanced tasks for both
platforms, despite the absence of Android-specific data in its training
dataset. This suggests a notable transferability of UI knowledge across
different operating systems. While Fuyu [fuyu-8b](https://www.adept.ai/blog/fuyu-8b) tends
to generate answers that are generally relevant, its responses lack the
detail and precision exhibited by Ferret-UI. Conversely, GPT-4V secures
higher scores across all tasks by consistently delivering more detailed
responses than Ferret-UI, a characteristic that aligns with the
preferences of the model evaluator (GPT-4). With Ferret-UI-anyres,
iPhone advanced tasks see a huge performance boost of 20 points while
Android advanced tasks see a performance drop. As Android advanced task
data is not included in the training mix, it could be that as the model
gains enriched knowledge about iPhone screen understanding, it loses a
bit of generalizability.

## Ablation Studies [sec:ablation_studies]

**Ablation on Advanced Tasks.** The design motivation behind elementary
tasks is to enhance the model’s visual and spatial understanding of
basic UI elements. We propose that this enhanced understanding can aid
in performing more complex tasks. This hypothesis is examined by
investigating how elementary tasks influence the model’s ability to
handle advanced tasks, with findings detailed in Tab.
<a href="#advanced_task_ablation" data-reference-type="ref"
data-reference="advanced_task_ablation">[advanced_task_ablation]</a>. We
see that with only advanced task data, the performance is 64% for both
platforms. The performance of advanced tasks on iPhone shows a
consistent improvement of 5% with the addition of either iPhone or
Android elementary tasks. Similarly, adding elementary tasks from the
iPhone enhances Android’s performance on advanced tasks by about 4%,
whereas incorporating Android elementary tasks boosts this performance
by 9%. Including both iPhone and Android elementary tasks further
improves performance by 3% and 5% for iPhone and Android advanced tasks,
respectively, beyond the improvements seen with a single set of
elementary tasks. These observations support our hypothesis that
elementary tasks provide the model with enhanced visual and spatial
understanding that facilitates advanced tasks.

<div class="subtable" markdown="1">

0.45

<div id="tab:ablation_studies" markdown="1">

|  | **iPhone** | **Android** |
|:---|:--:|:--:|
| Adv. task only | 64.6 | 64.3 |
| \+ iPhone elem. | 70.3 | 68.6 |
| \+ Android elem. | 70.2 | 75.3 |
| \+ both as in <a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">[tab:main_results]</a> | **73.4** | **80.5** |

Ablation studies on the factors that impact performance on (a) Advanced
tasks and (b) Spotlight tasks.

</div>

</div>

<div class="subtable" markdown="1">

0.45

<div id="tab:ablation_studies" markdown="1">

|  | S2W | WiC | TaP |
|:---|:--:|:--:|:--:|
| Spotlight [li2023spotlight](https://arxiv.org/pdf/2209.14927) | 106.7 | 141.8 | **88.4** |
| Balanced TaP labels | 111.7 | 133.8 | 76.5 |
| Spotlight tasks only | 111.3 | 138.7 | 77.6 |
| \+ Android elem. tasks | 111.3 | 138.0 | 76.8 |
| \+ iPhone elem. tasks | 112.4 | 138.9 | 74.8 |
| \+ both | 111.3 | 138.7 | 76.0 |
| Full mixture from <a href="#tab:main_results" data-reference-type="ref"
data-reference="tab:main_results">[tab:main_results]</a> | **113.4** | **142.0** | 78.4 |

Ablation studies on the factors that impact performance on (a) Advanced
tasks and (b) Spotlight tasks.

</div>

</div>

**Ablation on Spotlight Tasks.** Motivated by a desire to explore the
impact of different data configurations on Spotlight task performance,
we specifically investigate whether adding elementary task data could
enhance the model performance, given that these tasks are designed to
improve the visual and spatial comprehension of screens. As shown in
Tab. <a href="#tab:spotlight_tasks_ablation" data-reference-type="ref"
data-reference="tab:spotlight_tasks_ablation">[tab:spotlight_tasks_ablation]</a>,
the addition of elementary task data—whether exclusively from Android,
iPhone, or a combination of both—does not significantly alter
performance across the three Spotlight tasks. This may be attributed to
the short and highly specialized UI-centric vocabulary used in responses
in elementary tasks, contrasting with the response style demanded by
Spotlight tasks. Optimal results for Spotlight tasks were observed when
data from advanced tasks were integrated alongside all elementary tasks,
even though the advanced task data was exclusively derived from iPhone
screens. Notably, this yields a 4-point boost in CIDEr score for the
widget captions with the inclusion of advanced task data. We postulate
that the free-response format of advanced task answers, which
necessitates a more sophisticated set of skills for execution, aligns
more closely with the requirements of Spotlight tasks. These tasks
demand a comprehensive understanding beyond that of recognizing
individual UI elements, as is common in elementary tasks. Moreover,
executing advanced tasks requires more sophisticated skills than
understanding one specific UI element on the screen as in elementary
tasks. Thus, it stands to reason that the skill set honed through
advanced tasks would be advantageous for tackling Spotlight tasks, which
occupy a middle ground in complexity between elementary and advanced
tasks. In one word, the structure of the task assumes greater importance
than the source platform of the data incorporated.

<figure id="fig:analyses_ocr">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/analyses_ocr.png" style="width:90.0%" />
<figcaption><strong>OCR Analysis.</strong> <em>Left</em>: predict nearby
text instead of a targeted region in the base model, corrected in
anyres. <em>Middle</em>: a tendency to predict valid words.
<em>Right</em>: Ferret-UI correctly reads cut-off text, while the
detection model produces wrong labels.</figcaption>
</figure>

## Result Analysis: Elementary UI Tasks [sec:analysis_1]

**Referring Tasks.** In analyzing Ferret-UI’s referring capabilities, we
specifically focus on OCR and widget classification predictions. The OCR
analysis reveals three notable observations, as depicted in Fig.
<a href="#fig:analyses_ocr" data-reference-type="ref"
data-reference="fig:analyses_ocr">2</a>. First, the model predicts a
neighboring text instead of the text in the targeted region. This is
common for smaller texts and texts very close to other texts.
Remarkably, with anyres integrated, such cases are alleviated,
indicating that inputting enlarged sub-images helps the model with
smaller visual details. Second, the model exhibits a tendency to predict
actual words rather than merely deciphering characters displayed on the
screen. This observation is in line with the semantic-reliance
observation of LLMs made in some existing
work [liu2024LMMOCR](https://arxiv.org/pdf/2305.07895). On UI screens, phonetically crafted
words that are commonly used as brand titles largely fall under this
category. Third, Ferret-UI demonstrates the ability to accurately
predict text that is partially cut-off, even in instances where the OCR
model returns incorrect texts.

<figure id="fig:analyses_widget_classification">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/analyses_widget_classification.png"
style="width:90.0%" />
<figcaption><strong>Widget Classification Analysis.</strong>
<em>Left</em>: a large Button consists of a Picture, Icon, and Text
misclassified as a Picture. <em>Middle</em>: a button seated on top of a
row of Tabs misclassified as a Tab. <em>Right</em>: a small,
text-surrounded icon being classified as text in the base model, but
correctly classified with anyres.</figcaption>
</figure>

Similar to OCR analysis, we show three interesting observations in Fig.
<a href="#fig:analyses_widget_classification" data-reference-type="ref"
data-reference="fig:analyses_widget_classification">3</a>. First, the
model struggles when it needs to understand relationships among widgets.
For example, if a large button is made up of a few sub-elements,
including Picture, Icon, and text, the model cannot see it as a unified
widget but tends to predict it as the sub-element that occupies the
largest space. In line with the first observation, when a Tab or an Icon
is seated on top of a row of tabs, it is highly likely to be considered
part of the tabs. Finally, we discover a common case where small icons
surrounded by texts are likely to be predicted as Text, and this is
consistent with the observation that small texts tend to be predicted as
neighboring texts. With anyres added, such cases are more likely to be
predicted correctly, in line with the observation made in OCR.

**Grounding Tasks.** Using *find text* predictions, as depicted in Fig.
<a href="#fig:analyses_find_text" data-reference-type="ref"
data-reference="fig:analyses_find_text">4</a>, we further elucidate
observations from grounding tasks. Echoing the initial observation from
the *OCR* analysis, the model may erroneously highlight a piece of text
adjacent to the targeted area. Additionally, the occurrence of multiple
instances of identical texts suggests the potential for expanding future
methods to encompass a range of answers from a singular box to multiple
boxes, thereby enhancing the model’s utility and accuracy in complex
text-finding scenarios.

<figure id="fig:analyses_find_text">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/analyses_find_text.png" style="width:90.0%" />
<figcaption><strong>Find Text Analysis.</strong> <em>Left</em>: a
neighboring text is mis-identified as the target. <em>Middle</em>:
multiple occurrences of the same text. <em>Right</em>: predicted boxes
not precise.</figcaption>
</figure>

<figure id="fig:advanced_task_output">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/advanced_task_output_comparison.png" />
<figcaption>Visualization results of advanced tasks (top to bottom:
<em>function inference</em>, <em>conversation interaction</em>,
<em>conversation perception</em>) to illustrate the differences among
various models (Fuyu vs. CogAgent vs. GPT-4V vs.
Ferret-UI).</figcaption>
</figure>

## Result Analysis: Advanced UI Tasks [sec:analysis_2]

**Grounded Conversation.** Engaging in grounded conversation is Ferret’s
unique capability. To better understand the quality of the output
bounding boxes in terms of correctness and relevance, we manually grade
all output boxes in both Ferret-UI and GPT-4V’s *converation
interaction* outputs. The accuracies for Ferret-UI and GPT-4V are 91.7%
and 93.4%, respectively. Considering Ferret-UI generates raw coordinates
whereas GPT-4V chooses from a set of pre-defined boxes, Ferret-UI’s
grounding ability on UI screens is noteworthy. Even though Ferret-UI’s
received evaluation score falls short of GPT-4V, from inspecting the
predictions as in Fig.
<a href="#fig:advanced_task_output" data-reference-type="ref"
data-reference="fig:advanced_task_output">5</a>, we notice that GPT-4V
tends to provide extra information that may not be relevant to the
question. However, these detailed answers are more favored in scoring
than Ferret-UI’s concise answers.

**UI detection model is a bottleneck.** Given that both our elementary
and advanced tasks are predicated upon the detection of UI elements,
Ferret-UI is not able to learn aspects of screens that are not detected,
such as colors, design, usability, and UI elements that the detection
model misses (*e.g.*, topmost time, WIFI, battery). For example, in
generating detailed descriptions, GPT-4V is capable of noting “The
overall design conforms to Apple’s aesthetic with a minimalistic, clean,
dark theme”, a level of insight Ferret-UI is not trained to offer due to
its reliance on detected elements alone.

**Set-of-Mark (SoM) Prompting of GPT-4V.** In our analysis of GPT-4V,
the Set-of-Mark (SoM) prompting approach [yang2023set](http://arxiv.org/pdf/2310.11441v2) is
employed, revealing several limitations. First, its effectiveness
diminishes in scenarios involving a multitude of small UI elements, a
common occurrence in Android detection tasks. The small size of some UI
components means that the addition of labels may obscure original
content or even extend beyond the intended areas. Second, limiting the
assessment to a specified collection of candidate regions restricts the
model’s ability to reference any given region freely. In the middle
example shown in Fig.
<a href="#fig:advanced_task_output" data-reference-type="ref"
data-reference="fig:advanced_task_output">5</a>, the UI detection model
treats the entire middle section as one element, covering the texts,
image, and the Buy button. Therefore, the model is not able to refer to
the “BUY” button on its own in its responses, since it is considered
part of a collective detection group.

[^1]: For Ferret, we include the pre-defined classes for icon
    classification and widget classification in the prompts while the
    remaining prompts are the same as Ferret-UI.

[^2]: For GPT-4V, we sample a random subset of 100 instances for the
    Spotlight and elementary tasks for cost efficiency. For GPT-4V
    evaluation, we follow [yang2023set](http://arxiv.org/pdf/2310.11441v2) by overlaying
    indexed bounding boxes of UI elements as visual prompts.
    Consequently, in grounding tasks, GPT-4V is enabled to make
    selections from among these candidate boxes. We detail the effort in
    the Appendix.

# Conclusion

In this paper, we introduce Ferret-UI, a specialized MLLM designed to
enhance comprehension and interaction with mobile UI screens. Through
careful design of “anyres” to accommodate various screen aspect ratios
and curation of training samples that encompass a diverse range of basic
and advanced UI tasks, Ferret-UI demonstrates remarkable proficiency in
referring, grounding, and reasoning. The advent of these enhanced
capabilities promises substantial advancements for a multitude of
downstream UI applications, thereby amplifying the potential benefits
afforded by Ferret-UI in this domain.

# Elementary Task Data Generation Details [datagen_details]

Additional details in elementary task data generation are as follows:

-   In our data generation process, we merge the two distinct
    classes—“Checked” and “Unchecked”—found in the original detection
    labels for both *Checkboxes* and *Toggles*.

-   For widget listing, the answer starts with a common phrase: *UI
    widgets present in this screen include*. Each element is formatted
    as “{displayed text} {UI type}” (*e.g.*, “login button”), except for
    text elements, which are formatted as “Text displaying {displayed
    text}”.

-   For OCR, we consider text with fewer than 10 tokens. If the text is
    exactly one token, the length needs be to 2 or greater to be
    included.

-   For tasks such as *find text*, *find icons*, and *find widget*, it
    is common to encounter screens containing multiple instances of the
    same UI element (e.g., multiple login buttons). We employ a
    filtering mechanism that excludes samples involving UI elements with
    multiple occurrences within a single screen.

-   The size of the test set is determined by selecting the smaller
    value between 5k and the total number of generated test instances.

# Advanced Task Data Quality Analysis [appendix:conv_analyses]

We conduct a thorough analysis of the quality of our collected data for
advanced tasks and provide comprehensive statistics. The vocabulary size
for each task is as follows: 30,866 for *detailed description*, 15,666
for *conversation perception*, 12,092 for *conversation interaction*,
and 14,896 for *function inference*.

In the realm of *conversation interaction*, we observe 33,649 question
turns and 29,933 answer turns. Among these, 15 question turns include
bounding boxes, whereas all answer turns include bounding boxes. We
compile the most frequently occurring tri-grams for questions and
answers in both conversation tasks. Notably, in *conversation
perception* questions, the top tri-grams include phrases like *are there
any”*, *where is the”*, and *what is the”*, while those for interactions
comprise phrases like *How can I”*, *I want to”*, and *Can I do”*.
Similarly, in perception answers, prominent tri-grams consist of
expressions such as *“bottom of the”*, *“at the top”*, and *“there is
a”*, while interaction answers primarily feature tri-grams like *“by
tapping on”*, *“tapping on the”*, and *“can tap on”*.

We present detailed distributions of tri-grams in conversation data
questions and answers in Fig.
<a href="#fig:conv_data_stat" data-reference-type="ref"
data-reference="fig:conv_data_stat">5</a>. This observation is
consistent with our intended objectives for each conversation category,
with perception focusing on visual elements and interaction emphasizing
actions. Notably, from the interaction conversation answers, we observe
that *tap* emerges as the predominant action. In future work, we aim to
explore interactions involving other actions, such as scrolling,
long-clicking, and entering text. The inclusion of two conversation
categories aims to diversify conversation topics, although a clear-cut
distinction between the two is not always feasible, and overlap between
the categories may occur.

<figure id="fig:conv_data_stat">
<figure id="fig:perception_q">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/perception_questions_trigram.png" />
<figcaption><em>Conversation perception</em> <strong>questions</strong>
trigrams distribution.</figcaption>
</figure>
<figure id="fig:perception_a">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/perception_answers_trigram.png" />
<figcaption><em>Conversation perception</em> <strong>answers</strong>
trigrams distribution.</figcaption>
</figure>
<figure id="fig:interaction_q">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/interaction_questions_trigram.png" />
<figcaption><em>Conversation interaction</em> <strong>questions</strong>
trigrams distribution.</figcaption>
</figure>
<figure id="fig:interaction_a">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/interaction_answers_trigram.png" />
<figcaption><em>Conversation interaction</em> <strong>answers</strong>
trigrams distribution.</figcaption>
</figure>
<figcaption>Trigrams for collected conversation data questions and
answers.</figcaption>
</figure>

# Taperception Label Analysis [appendix:taperception_analysis]

We meticulously label 30 test samples for *taperception* and conduct a
study on the correlation among our labels, *taperception* ground-truth
labels, Ferret-UI outputs, and GPT-4V outputs. Among the 30 samples, 5
pose challenges in deciphering without direct interaction with the
screen.

In Tab. <a href="#fig:tap_label_analysis" data-reference-type="ref"
data-reference="fig:tap_label_analysis">8</a>, we present the percentage
of agreement among different sources of predictions and labels. The term
“filtered” denotes the set of 25 instances that are unambiguous, while
“unfiltered” encompasses the entire 30 instances. Our labels exhibit a
high correlation with GPT-4V predictions, but differing significantly
from the *taperception* dataset labels. This discrepancy underscores the
complexity of predicting *tappability* solely based on single images,
highlighting the inherent challenges in obtaining clear-cut labels for
this task.

<figure id="fig:tap_label_analysis">
<figure id="fig:y equals x">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/taperception_cor_filtered.png" />
<figcaption>Filtered.</figcaption>
</figure>
<figure id="fig:three sin x">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/taperception_cor_unfiltered.png" />
<figcaption>Unfiltered.</figcaption>
</figure>
<figcaption>Agreement between different sources of taperception
predictions and labels. In unfiltered, we make the best educational
guess for the one that are ambiguous. We observe that our human
annotation correlate with GPT-4V (%76) far more than with taperception
label (%8). Even though Ferret-UI’ performance on taperception falls
behind compared to Spotlight, it could be due to the noisiness of
labels.</figcaption>
</figure>

# Advanced Task Generation Prompts [appendix:gpt4v_prompts]

We present the prompts to collect advanced task data from GPT-4 in Fig.
<a href="#gpt4_prompts" data-reference-type="ref"
data-reference="gpt4_prompts">9</a>.

<figure id="gpt4_prompts">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/gpt4_prompts.png" />
<figcaption>Prompts for GPT-4 in advanced task data
generation.</figcaption>
</figure>

<figure id="fig:gpt4v_input">
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/gpt4v_input.png" />
<figcaption>GPT-4V input image examples. Left: used in referring task,
where the question concerns one specific UI element. Right: used in
grounding task, where GPT-4V refers to the UI elements by their assigned
numeric labels.</figcaption>
</figure>

# GPT-4V Evaluation Details [gpt4v_eval]

We detail the process of creating input for GPT-4V to tackle the UI
tasks under scope.

#### \[Input Images\]

We first annotate the screenshots tailored to each specific task,
ensuring that GPT-4V has sufficient contextual information to answer the
questions. For tasks without any bounding boxes in input or output
(*screen2words*, *widget captions*, and *Advanced Tasks*), we use the
original images as the input. For tasks that refer to **one** specific
UI element using bounding box in the input, we put a magenta-colored
bounding box on the image as the input, as shown in Fig.
<a href="#fig:gpt4v_input" data-reference-type="ref"
data-reference="fig:gpt4v_input">10</a> left. For tasks that expect one
or more bounding boxes in the output, our initial explorations confirm
that GPT-4V is not able to provide bounding boxes in the output as it
gives the answer, *"Unfortunately, I’m not able to provide the exact
bounding box coordinates, as my capabilities are currently limited to
describing images and discussing the content rather than interacting
directly with the image to extract detailed metadata such as pixel
coordinates.")* and proceed to answer the question in natural language.
Therefore, for those tasks, we create an easier version where we ask
GPT-4V to choose from a fixed set of candidates. Particularly, we follow
Set-of-Mark prompting [yang2023set](http://arxiv.org/pdf/2310.11441v2) where for each UI
detection from our UI detection model, we use a magenta-colored bounding
box to mark it in the screen and inside each box we assign a numeric
label so that GPT4-V can refer to it. An example input image is shown in
Fig. <a href="#fig:gpt4v_input" data-reference-type="ref"
data-reference="fig:gpt4v_input">10</a> right.

#### \[Prompts\]

With the input images ready, we further modify the prompts to provide
GPT-4V with all the necessary information to perform all the tasks
successfully. For taperception, we instruct it to answer *“Yes.”* or
*“No.”* only without any explanations. For widget captions, we instruct
it to *“Answer in a few words.”* For *icon recognition* and *widget
classification*, we provide the list of all possible classes, and
instruct it to output the class only without any explanations. For
*OCR*, we instruct it to output the identified text only. For *find
widget*, *find text*, *find icons*, we add to the prompt *“Candidate
choices are indicated using magenta bounding boxes in the image and each
box is associated with a numeric label. Output the numeric label as your
answer, no need to explain."*

# More Example Outputs

<figure id="fig:ferret-ui-ex2-3">
<figure>
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/appendix-ex-1-ferretui.png" />
</figure>
<p><br />
</p>
<figure>
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/appendix-ex-2-ferretui.png" />
</figure>
<p><br />
</p>
<figure>
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/appendix-ex-3-ferretui.png" />
</figure>
</figure>

<figure id="fig:ferret-ui-ex2-3">
<figure>
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/appendix-ex-4-ferretui.png" />
</figure>
<p><br />
</p>
<figure>
<img src="/vision_rich/arXiv-2311.06607v3_md/figs/final_figs/appendix-ex-5-ferretui.png" />
</figure>
</figure>