# Introduction

Humans interact with the world through many channels such as vision and
language, as each individual channel has a unique advantage in
representing and communicating certain concepts, and thus facilitates a
better understanding of the world. One of the core aspirations in
artificial intelligence is to develop a general-purpose assistant that
can effectively follow multi-modal vision-and-language instructions,
aligned with human intent to complete various real-world tasks in the
wild [askell2021general](http://arxiv.org/pdf/2112.00861v3), [li2022elevater](None), [li2023multimodal](http://arxiv.org/pdf/2309.10020v1).

To this end, the community has witnessed an emergent interest in
developing language-augmented foundation vision
models [li2022elevater](None), [gan2022vision](http://arxiv.org/pdf/2110.03902v1), with strong
capabilities in open-world visual understanding such as
classification [radford2021learning](http://arxiv.org/pdf/2404.19696v1), [openclip](https://doi.org/10.5281/zenodo.5143773), [yuan2021florence](http://arxiv.org/pdf/2301.05065v2), [yang2022unicl](http://arxiv.org/pdf/2107.11945v1), [pham2021combined](http://arxiv.org/pdf/1602.01255v2),
detection [li2022grounded](http://arxiv.org/pdf/2210.09263v1), [zhong2022regionclip](http://arxiv.org/pdf/1606.00540v1), [liu2023grounding](http://arxiv.org/pdf/2401.02361v2),
segmentation [li2022language](http://arxiv.org/pdf/2304.10326v1), [zou2022generalized](http://arxiv.org/pdf/2212.11270v1), [zhang2023simple](http://arxiv.org/pdf/1809.09299v1)
and captioning [wang2022git](http://arxiv.org/pdf/2204.07780v1), [li2023blip](http://arxiv.org/pdf/2301.12597v3), as well as
visual generation and
editing [DALLE2](http://arxiv.org/pdf/2204.06125v1), [LDM](http://arxiv.org/pdf/2307.10094v1), [PARTI](http://arxiv.org/pdf/2102.11495v1), [MAKEASCENE](http://arxiv.org/pdf/2211.01602v2), [Imagen](http://arxiv.org/pdf/2205.11487v1), [li2023gligen](http://arxiv.org/pdf/2311.09144v2).
We refer readers to the *Computer Vision in the Wild* reading list for a
more up-to-date literature compilation [cvinw](https://github.com/Computer-Vision-in-the-Wild/CVinW_Readings). In this
line of work, each task is solved independently by one single large
vision model, with the task instruction implicitly considered in the
model design. Further, language is only utilized to describe the image
content. While this allows language to play an important role in mapping
visual signals to language semantics—a common channel for human
communication, it leads to models that usually have a fixed interface
with limited interactivity and adaptability to the user’s instructions.

Large language models (LLM), on the other hand, have shown that language
can play a wider role: a universal interface for a general-purpose
assistant, where various task instructions can be explicitly represented
in language and guide the end-to-end trained neural assistant to switch
to the task of interest to solve it. For example, the recent success of
ChatGPT [chatgpt](https://openai.com/blog/chatgpt/) and GPT-4 [gpt4](https://arxiv.org/pdf/2303.08774) have
demonstrated the power of aligned LLMs in following human instructions,
and have stimulated tremendous interest in developing open-source LLMs.
Among them, LLaMA [touvron2023llama](http://arxiv.org/pdf/2402.08075v1) is an open-source
LLM that matches the performance of GPT-3.
Alpaca [alpaca](https://github.com/tatsu-lab/stanford_alpaca), Vicuna [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/),
GPT-4-LLM [peng2023instruction](None) utilize various
machine-generated high-quality instruction-following samples to improve
the LLM’s alignment ability, reporting impressive performance compared
with proprietary LLMs. Importantly, this line of work is *text-only*.

In this paper, we present *visual instruction-tuning*, the first attempt
to extend instruction-tuning to the language-image multimodal space, to
pave the way towards building a general-purpose visual assistant. In
particular, our paper makes the following contributions:

-   *Multimodal instruction-following data*. One key challenge is the
    lack of vision-language instruction-following data. We present a
    data reformation perspective and pipeline to convert image-text
    pairs into an appropriate instruction-following format, using
    ChatGPT/GPT-4.

-   *Large multimodal models*. We develop a large multimodal model
    (LMM), by connecting the open-set visual encoder of
    CLIP [radford2021learning](http://arxiv.org/pdf/2404.19696v1) with the language decoder
    Vicuna [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), and fine-tuning end-to-end on our
    generated instructional vision-language data. Our empirical study
    validates the effectiveness of using generated data for LMM
    instruction-tuning, and suggests practical tips for building a
    general-purpose instruction-following visual agent. When ensembled
    with GPT-4, our approach achieves SoTA on the Science
    QA [lu2022learn](http://arxiv.org/pdf/2209.09513v2) multimodal reasoning dataset.

-   *Multimodal instruction-following benchmark*. We present LLaVA-Bench
    with two challenging benchmarks, with a diverse selection of paired
    images, instructions and detailed annotations.

-   *Open-source*. We release the following assets to the public: the
    generated multimodal instruction data, the codebase, the model
    checkpoints, and a visual chat demo.

# Related Work

**Multimodal Instruction-following Agents.**  In computer vision,
existing works that build instruction-following agents can be broadly
categorized into two classes: $(i)$ End-to-end trained models, which are
separately explored for each specific research topic. For example, the
vision-language navigation
task [anderson2018vision](http://arxiv.org/pdf/2402.11498v1), [hao2020towards](http://arxiv.org/pdf/2304.04907v1) and
Habitat [szot2021habitat](http://arxiv.org/pdf/2106.14405v2) require the embodied AI agent
to follow natural language instructions and take a sequence of actions
to complete goals in visual environments. In the image editing domain,
given an input image and a written instruction that tells the agent what
to do, InstructPix2Pix [brooks2022instructpix2pix](http://arxiv.org/pdf/2405.14785v1) edits
images by following the human instructions. $(ii)$ A system that
coordinates various models via LangChain [langchain](https://github.com/hwchase17/langchain) /
LLMs [chatgpt](https://openai.com/blog/chatgpt/), such as Visual
ChatGPT [wu2023visual](http://arxiv.org/pdf/2303.04671v1),
X-GPT [zou2022generalized](http://arxiv.org/pdf/2212.11270v1),
MM-REACT [yang2023mm](http://arxiv.org/pdf/2303.11381v1),
VisProg [gupta2022visual](http://arxiv.org/pdf/2203.15442v1), and
ViperGPT [suris2023vipergpt](http://arxiv.org/pdf/1905.11127v1). While sharing the same goal
in building instruction-following agents, we focus on developing an
end-to-end trained language-vision multimodal model for *multiple*
tasks.

**Instruction Tuning.** In the natural language processing (NLP)
community, to enable LLMs such as
GPT-3 [brown2020language](http://arxiv.org/pdf/2112.07522v2),
T5 [raffel2020exploring](http://arxiv.org/pdf/1910.10683v4),
PaLM [chowdhery2022palm](http://arxiv.org/pdf/2209.05735v4), and
OPT [zhang2022opt](None) to follow natural language
instructions and complete real-world tasks, researchers have explored
methods for LLM
instruction-tuning [ouyang2022training](http://arxiv.org/pdf/2302.05206v1), [wang2022benchmarking](http://arxiv.org/pdf/2212.12017v3), [wang2022self](http://arxiv.org/pdf/2311.00233v2),
leading to instruction-tuned counterparts such as
InstructGPT [ouyang2022training](http://arxiv.org/pdf/2302.05206v1)/ChatGPT [chatgpt](https://openai.com/blog/chatgpt/),
FLAN-T5 [chung2022scaling](http://arxiv.org/pdf/2202.03371v1),
FLAN-PaLM [chung2022scaling](http://arxiv.org/pdf/2202.03371v1), and
OPT-IML [iyer2022opt](http://arxiv.org/pdf/2210.11617v1), respectively. It turns out that
this simple approach can effectively improve the zero- and few-shot
generalization abilities of LLMs. It is thus natural to borrow the idea
from NLP to computer vision. More broadly, the teacher-student
distillation ideas with foundation models have been studied in other
topics such as image
classification [faghri2023reinforce](http://arxiv.org/pdf/2303.08983v3).
Flamingo [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1) can be viewed as the GPT-3
moment in the multimodal domain, due to its strong performance on
zero-shot task transfer and in-context-learning. Other LMMs trained on
image-text pairs include BLIP-2 [li2023blip](http://arxiv.org/pdf/2301.12597v3),
FROMAGe [koh2023grounding](http://arxiv.org/pdf/2401.13388v2), and
KOSMOS-1 [huang2023language](http://arxiv.org/pdf/2302.14045v2).
PaLM-E [driess2023palm](None) is an LMM for embodied AI. Based
on the recent “best” open-source LLM LLaMA,
OpenFlamingo [anas_awadalla_2023_7733589](https://doi.org/10.5281/zenodo.7733589) and
LLaMA-Adapter [zhang2023llama](http://arxiv.org/pdf/2207.10858v1) are open-source efforts
that enable LLaMA to use image inputs, paving the way to build
open-source multimodal LLMs. While these models present promising task
transfer generalization performance, they are not explicitly tuned with
vision-language instruction data, and their performance in multimodal
tasks usually falls short compared to language-only tasks. In this
paper, we aim to fill this gap and study its effectiveness. Finally,
note that visual instruction tuning is different from visual prompt
tuning [jia2022visual](http://arxiv.org/pdf/2309.01155v2): the former aims to improve the
model’s instruction-following abilities, while the latter aims to
improve the parameter-efficiency in model adaptation.

# GPT-assisted Visual Instruction Data Generation [sec:visual_instruc_data]

The community has witnessed a surge in the amount of public multimodal
data such as image-text pairs, ranging from
CC [changpinyo2021conceptual](http://arxiv.org/pdf/2102.08981v2) to
LAION [schuhmann2022laion](http://arxiv.org/pdf/2312.15897v1). However, when it comes to
multimodal instruction-following data, the available amount is limited,
partially because the process for creating such data is time-consuming
and less well-defined when human crowd-scouring is considered. Inspired
by the success of recent GPT models in text-annotation
tasks [gilardi2023chatgpt](http://arxiv.org/pdf/2303.15056v2), we propose to leverage
ChatGPT/GPT-4 for multimodal instruction-following data collection,
based on the widely existing image-pair data.

For an image $\Xmat_{\texttt{v}}$ and its associated caption
$\Xmat_{\texttt{c}}$, it is natural to create a set of questions
$\Xmat_{\texttt{q}}$ with the intent to instruct the assistant to
describe the image content. We prompt GPT-4 to curate such a list of
questions (see details in Appendix). Therefore, a simple way to expand
an image-text pair to its instruction-following version is
$\texttt{Human}: \Xmat_{\texttt{q}} ~\Xmat_{\texttt{v}}     \texttt{<STOP>}~
    \texttt{Assistant}: 
    \Xmat_{\texttt{c} } 
     \texttt{<STOP>}$. Though cheap to construct, this simple expanded
version lacks diversity and in-depth reasoning in both the instructions
and responses.

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

|  |  |
|:---|:--:|
|  |  |
| A group of people standing outside of a black vehicle with various luggage. |  |
| Luggage surrounds a vehicle in an underground parking area |  |
| People try to fit all of their luggage in an SUV. |  |
| The sport utility vehicle is parked in the public garage, being packed for a trip |  |
| Some people with luggage near a van that is transporting it. |  |
|  |  |
| person: \[0.681, 0.242, 0.774, 0.694\], backpack: \[0.384, 0.696, 0.485, 0.914\], suitcase: ...\<omitted\> |  |
|  |  |
|  |  |
| Question: What type of vehicle is featured in the image? |  |
| Answer: The image features a black sport utility vehicle (SUV) ...\<omitted\> |  |
|  |  |
| The image is an underground parking area with a black sport utility vehicle (SUV) parked. There are three people in the scene, with one person standing closer to the left side of the vehicle, another person in the middle, and the third person on the right side. They are all working together to pack their luggage into the SUV for a trip. ...\<omitted\> |  |
|  |  |
| Question: What challenges do these people face? |  |
| Answer: In the image, a group of people is standing outside a black SUV in a parking area, surrounded by various pieces of luggage, including suitcases and backpacks. They are facing the challenge of fitting all their luggage into the black SUV. There are multiple suitcases and backpacks to be packed, which suggests that the group has a significant amount of belongings ...\<omitted\> |  |

</div>

</div>

To mitigate this issue, we leverage language-only GPT-4 or ChatGPT as
the strong teacher (both accept only text as input), to create
instruction-following data involving visual content. Specifically, in
order to encode an image into its visual features to prompt a text-only
GPT, we use two types of symbolic representations: $(i)$ *Captions*
typically describe the visual scene from various perspectives; $(ii)$
*Bounding boxes* usually localize the objects in the scene, and each box
encodes the object concept and its spatial location. One example is
shown in the top block of
Table <a href="#tab:full_example_car_bbox" data-reference-type="ref"
data-reference="tab:full_example_car_bbox">[tab:full_example_car_bbox]</a>.

This symbolic representation allows us to encode the image as an
LLM-recognizable sequence. We use COCO
images [lin2014microsoft](None) and generate three types of
instruction-following data. One example per type is shown in the bottom
block of
Table <a href="#tab:full_example_car_bbox" data-reference-type="ref"
data-reference="tab:full_example_car_bbox">[tab:full_example_car_bbox]</a>.
For each type, we first manually design a few examples. They are the
only human annotations we have during data collection, and are used as
seed examples in in-context-learning to query GPT-4.

-   *Conversation*. We design a conversation between the assistant and a
    person asking questions about this photo. The answers are in a tone
    as if the assistant is seeing the image and answering the question.
    A diverse set of questions are asked about the visual content of the
    image, including the object types, counting the objects, object
    actions, object locations, relative positions between objects. Only
    questions that have definite answers are considered. Please see
    Appendix for the detailed prompt.

-   *Detailed description*. To include a rich and comprehensive
    description for an image, we create a list of questions with such an
    intent. We prompt GPT-4 then curate the list (see detailed prompts
    and curation process in Appendix). For each image, we randomly
    sample one question from the list to ask GPT-4 to generate the
    detailed description.

-   *Complex reasoning*. The above two types focus on the visual content
    itself, based on which we further create in-depth reasoning
    questions. The answers typically require a step-by-step reasoning
    process by following rigorous logic.

We collect 158K unique language-image instruction-following samples in
total, including 58K in conversations, 23K in detailed description, and
77k in complex reasoning, respectively. We ablated the use of ChatGPT
and GPT-4 in our early experiments, and found that GPT-4 consistently
provides higher quality instruction-following data, such as spatial
reasoning.

# Visual Instruction Tuning

## Architecture

The primary goal is to effectively leverage the capabilities of both the
pre-trained LLM and visual model. The network archtecture is illustrated
in Figure <a href="#fig:llava_arch" data-reference-type="ref"
data-reference="fig:llava_arch">1</a>. We choose
Vicuna [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) as our LLM $f_{\phiv}(\cdot)$
parameterized by $\phiv$, as it has the best instruction following
capabilities in language tasks among publicly available
checkpoints [alpaca](https://github.com/tatsu-lab/stanford_alpaca), [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [peng2023instruction](None).

<figure id="fig:llava_arch">
<embed src="/papers/text_rich/arXiv-2304.08485v2_md/figures/llava_arch.png" style="height:3.5cm" />
<figcaption> network architecture.</figcaption>
</figure>

For an input image $\Xmat_{\texttt{v}}$, we consider the pre-trained
CLIP visual encoder ViT-L/14 [radford2021learning](http://arxiv.org/pdf/2404.19696v1), which
provides the visual feature
$\Zmat_{\texttt{v}} = g(\Xmat_{\texttt{v}})$. The grid features before
and after the last Transformer layer are considered in our experiments.
We consider a simple linear layer to connect image features into the
word embedding space. Specifically, we apply a trainable projection
matrix $\Wmat$ to convert $\Zmat_{\texttt{v}}$ into language embedding
tokens $\Hmat_{\texttt{v}}$, which have the same dimensionality as the
word embedding space in the language model:
$$\Hmat_{\texttt{v}} = \Wmat \cdot \Zmat_{\texttt{v}},~ \text{with}~~
   \Zmat_{\texttt{v}} = g(\Xmat_{\texttt{v}})
    \label{eq:image_encoding}$$ Thus, we have a sequence of visual
tokens $\Hmat_{\texttt{v}}$. Note that our simple projection scheme is
lightweight, which allows us to iterate data centric experiments
quickly. More sophisticated schemes to connect the image and language
representations can also be considered, such as gated cross-attention in
Flamingo [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1) and Q-former in
BLIP-2 [li2023blip](http://arxiv.org/pdf/2301.12597v3). We leave exploring possibly more
effective and sophisticated architecture designs for as future work.

## Training

For each image $\Xmat_{\texttt{v}}$, we generate multi-turn conversation
data
$(\Xmat_{\texttt{q}}^1, \Xmat_{\texttt{a}}^1, \cdots, \Xmat_{\texttt{q}}^T, \Xmat_{\texttt{a}}^T )$,
where $T$ is the total number of turns. We organize them as a sequence,
by treating all answers as the assistant’s response, and the instruction
$\Xmat_{\texttt{instruct}}^t$ at the $t$-th turn as: $$\begin{aligned}
 \label{eq:organize_data_turn_rule}
 \Xmat_{\texttt{instruct}}^t = 
\left\{\begin{matrix}
& \text{Randomly choose}~~
[\Xmat_{\texttt{q}}^1, \Xmat_{\texttt{v}}] ~~\text{or}~~ [ \Xmat_{\texttt{v}}, \Xmat_{\texttt{q}}^1] ,  ~~~\text{the first turn}~t=1 \\ 
& \Xmat_{\texttt{q}}^t, \hspace{45mm} \text{the remaining turns}~t>1
\end{matrix}\right.
\end{aligned}$$

This leads to the unified format for the multimodal
instruction-following sequence illustrated in
Table <a href="#tab:input_sequence" data-reference-type="ref"
data-reference="tab:input_sequence">[tab:input_sequence]</a>. We perform
instruction-tuning of the LLM on the prediction tokens, using its
original auto-regressive training objective.

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

$\Xmat_{\texttt{system-message}}$  
$\texttt{Human}: \Xmat_{\texttt{instruct}}^1$ $\texttt{Assistant}$:  
$\texttt{Human}: \Xmat_{\texttt{instruct}}^2$ $\texttt{Assistant}$:
$\cdots$  

</div>

</div>

Specifically, for a sequence of length $L$, we compute the probability
of the target answers $\Xmat_{\texttt{a}}$ by:
$$p( \Xmat_{\texttt{a}} |  \Xmat_{\texttt{v}}, \Xmat_{\texttt{instruct}}) =
    \prod_{i=1}^{L} p_{\thetav} (  {\color{mygreen} \xv_i}
| \Xmat_{\texttt{v}}, \Xmat_{\texttt{instruct}, <i}, \Xmat_{\texttt{a}, <i}),
    \label{eq:auto_regressive}$$ where $\thetav$ is the trainable
parameters, $\Xmat_{\texttt{instruct}, <i}$ and $\Xmat_{\texttt{a}, <i}$
are the instruction and answer tokens in all turns before the current
prediction token ${\color{mygreen} \xv_i}$, respectively. Please see
Table <a href="#tab:input_sequence" data-reference-type="ref"
data-reference="tab:input_sequence">[tab:input_sequence]</a> for an
illustration of the prediction tokens. For the conditionals
in <a href="#eq:auto_regressive" data-reference-type="eqref"
data-reference="eq:auto_regressive">[eq:auto_regressive]</a>, we
explicitly add $\Xmat_{\texttt{v}}$ to emphasize the fact that the image
is grounded for all answers, and we omit
$\Xmat_{\texttt{system-message}}$ and all previous `<STOP>` for better
readability. For model training, we consider a two-stage
instruction-tuning procedure.

#### Stage 1: Pre-training for Feature Alignment.

To strike a balance between concept coverage and training efficiency, we
filter CC3M to 595K image-text pairs. Please see Appendix for details of
the filtering process. These pairs are converted to the
instruction-following data using the naive expansion method describe in
Section <a href="#sec:visual_instruc_data" data-reference-type="ref"
data-reference="sec:visual_instruc_data">1</a>. Each sample can be
treated as a single-turn conversation. To construct the input
$\Xmat_{\texttt{instruct}}$ in
<a href="#eq:organize_data_turn_rule" data-reference-type="eqref"
data-reference="eq:organize_data_turn_rule">[eq:organize_data_turn_rule]</a>,
for an image $\Xmat_{\texttt{v}}$, a question $\Xmat_{\texttt{q}}$ is
randomly sampled, which is a language instruction to request the
assistant to describe the image briefly. The ground-truth prediction
answer $\Xmat_{\texttt{a}}$ is the original caption. In training, we
keep both the visual encoder and LLM weights frozen, and maximize the
likelihood of <a href="#eq:auto_regressive" data-reference-type="eqref"
data-reference="eq:auto_regressive">[eq:auto_regressive]</a> with
trainable parameters $\thetav = \Wmat$ (the projection matrix) only. In
this way, the image features $\Hmat_{\texttt{v}}$ can be aligned with
the pre-trained LLM word embedding. This stage can be understood as
training a compatible visual tokenizer for the frozen LLM.

#### Stage 2: Fine-tuning End-to-End.

We always keep the visual encoder weights frozen, and continue to update
both the pre-trained weights of the projection layer and LLM in ; i.e.,
the trainable parameters are $\thetav = \{\Wmat, \phiv \}$
in <a href="#eq:auto_regressive" data-reference-type="eqref"
data-reference="eq:auto_regressive">[eq:auto_regressive]</a>. We
consider two specific use case scenarios:

-   *Multimodal Chatbot*. We develop a Chatbot by fine-tuning on the
    158K language-image instruction-following data in
    Section <a href="#sec:visual_instruc_data" data-reference-type="ref"
    data-reference="sec:visual_instruc_data">1</a>. Among the three
    types of responses, conversation is multi-turn while the other two
    are single-turn. They are uniformly sampled in training.

-   *Science QA*. We study our method on the ScienceQA
    benchmark [lu2022learn](http://arxiv.org/pdf/2209.09513v2), the first large-scale
    multimodal science question dataset that annotates the answers with
    detailed lectures and explanations. Each question is provided a
    context in the form of natural language or an image. The assistant
    provides the reasoning process in natural language and selects the
    answer among multiple choices. For training in
    <a href="#eq:organize_data_turn_rule" data-reference-type="eqref"
    data-reference="eq:organize_data_turn_rule">[eq:organize_data_turn_rule]</a>,
    we organize the data as a single turn conversation, the question &
    context as $\Xmat_{\texttt{instruct}}$, and reasoning & answer as
    $\Xmat_{\texttt{a}}$.

# Experiments

<span id="tab:visual_example_ironing"
label="tab:visual_example_ironing"></span>

We assess the performance of in instruction-following and visual
reasoning capabilities with two primary experimental settings:
multimodal chatbot and the ScienceQA dataset, respectively. We train all
models with 8$\times$ A100s, following Vicuna’s
hyperparameters [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/). We pre-train our model on the
filtered CC-595K subset for 1 epoch with a learning rate of 2e-3 and a
batch size of 128, and fine-tune on the proposed LLaVA-Instruct-158K
dataset for 3 epochs, with a learning rate of 2e-5 and a batch size of
32. See Appendix for more training details.

## Multimodal Chatbot

We developed a chatbot demo to show the image understanding and
conversation abilities of , and to study how well is able to digest
visual inputs and exhibit instruction-following capabilities. We first
use the examples in the original GPT-4 paper [gpt4](https://arxiv.org/pdf/2303.08774),
shown in
Table <a href="#tab:visual_example_ironing" data-reference-type="ref"
data-reference="tab:visual_example_ironing">[tab:visual_example_ironing]</a>
(more examples in Appendix), that require in-depth image understanding.
For comparisons, we quote the prompt and response of the multimodal
GPT-4 from their paper, and query BLIP-2 and OpenFlamingo model
checkpoints to get their response.

Surprisingly, although is trained with a small multimodal
instruction-following dataset ($\sim$`<!-- -->`{=html}80K unique
images), it demonstrates quite similar reasoning results with multimodal
GPT-4 on these examples. Note that while these images are out-of-domain
for , is still able to understand the scenes and follow the question
instruction to provide a reasonable response. In contrast, BLIP-2 and
OpenFlamingo focus on describing the image, instead of following the
user instruction to answer in an appropriate manner.

#### Quantitative Evaluation.

To gain a systematic understanding of the performance of , we propose a
quantitative metric to measure the model’s instruction-following
capability on multimodal data. Inspired by [vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), we
leverage GPT-4 to measure the quality of generated responses.
Specifically, we create triplets consisting of image, ground-truth
textual descriptions, and question. The candidate models () predict the
answers based on the question and the image. To provide an *approximate
theoretical upper bound*, we create a reference prediction based on the
question and the *ground-truth* textual descriptions, using the
text-only GPT-4. After obtaining the responses from both models, we feed
the question, visual information (in the format of textual
descriptions), and the generated responses from both assistants, to the
judge (text-only GPT-4). It evaluates the helpfulness, relevance,
accuracy, and level of detail of the responses from the assistants, and
gives an overall score on a scale of 1 to 10, where a higher score
indicates better overall performance. It is also asked to provide a
comprehensive explanation for the evaluation, for us to better
understand the models. We report relative scores *w.r.t.* the text-only
GPT-4 model that uses the textural ground truth description as visual
input. We create two benchmarks to evaluate the model’s performance.

<span id="tab:results" label="tab:results"></span>

<span id="tab:results_wild" label="tab:results_wild"></span>

#### .

We randomly select 30 images from COCO-Val-2014, and for each image, we
generate three types of questions (conversation, detailed description,
complex reasoning) using the proposed data generation pipeline in
Sec. <a href="#sec:visual_instruc_data" data-reference-type="ref"
data-reference="sec:visual_instruc_data">[sec:visual_instruc_data]</a>,
totaling 90 questions. This benchmark studies the model’s alignment
behavior and capabilities with consistent visual inputs. We vary the
training datasets to study the effectiveness of different types of
instruction-following data, and show the results in
Table <a href="#tab:results" data-reference-type="ref"
data-reference="tab:results">[tab:results]</a>. First, with instruction
tuning, the model’s ability of following user instructions improves
significantly by over 50 points. Second, adding a small amount of
detailed description and complex reasoning questions contributes to a
considerable improvement of the model’s overall capability by 7 points.
Furthermore, it also improves the model’s performance on conversational
questions, suggesting that improvements in reasoning capabilities
complement conversational abilities. Finally, we show that having all
three types of data yields the best performance at 85.1%.

<span id="tab:example_bench" label="tab:example_bench"></span>

#### .

To evaluate the model’s capability in more challenging tasks and
generalizability to novel domains, we collect a diverse set of 24 images
with 60 questions in total, including indoor and outdoor scenes, memes,
paintings, sketches, , and associate each image with a highly-detailed
and manually-curated description and a proper selection of questions. We
compare , BLIP, and OpenFlamingo in
Table <a href="#tab:results_wild" data-reference-type="ref"
data-reference="tab:results_wild">[tab:results_wild]</a>. Thanks to
visual instruction tuning, achieves significantly better performance
compared with BLIP-2 (+29%) and OpenFlamingo (+48%). Compared to the
text-only GPT-4 that has access to ground-truth labels, achieves an
impressive 81.7% performance on complex reasoning questions, with an
overall score of 67.3%.

#### Limitations.

This is designed to be challenging and to reveal a model’s weaknesses.
We provide two examples with associated captions and questions in
Table <a href="#tab:example_bench" data-reference-type="ref"
data-reference="tab:example_bench">[tab:example_bench]</a>. For the
ramen example (left), to correctly answer the name of the restaurant, it
requires the model to have a large knowledge coverage and multilingual
understanding capability; to correctly describe the side dishes, the
model may need to retrieve relevant multimodal information from
Internet. For the fridge example (right), perceiving the correct brand
of the yogurt requires the model to process high resolution images and
possess extensive knowledge coverage. We also observed an interesting
failure of , as it responds with *yes* when asked if strawberry-flavored
yogurt is present, even though the fridge contains only yogurt *and*
strawberries. This indicates that, at times, perceives the image as a
“bag of patches”, failing to grasp the complex semantics within the
image. We hope serves as a solid baseline on the benchmarks, on which
our findings can inspire future work in developing more capable LMMs.

## ScienceQA

ScienceQA [lu2022learn](http://arxiv.org/pdf/2209.09513v2) contains 21k multimodal multiple
choice questions with rich domain diversity across 3 subjects, 26
topics, 127 categories, and 379 skills. The benchmark dataset is split
into training, validation, and test splits with 12726, 4241, and 4241
examples, respectively. We consider two representative methods,
including GPT-3.5 model () with and without chain-of-thought (CoT),
LLaMA-Adapter [zhang2023llama](http://arxiv.org/pdf/2207.10858v1), as well as multimodal
chain-of-thought (MM-CoT) [zhang2023multimodal](http://arxiv.org/pdf/2401.06805v2), which is
the current SoTA method on this dataset. For more baseline numbers,
please see [lu2022learn](http://arxiv.org/pdf/2209.09513v2).

The results are reported in
Table <a href="#tab:scienceqa_model_performance" data-reference-type="ref"
data-reference="tab:scienceqa_model_performance">[tab:scienceqa_model_performance]</a>.
For , we use the visual features before the last layer, ask the model to
first predict reasons and then the answer, and train it for 12 epochs.
It yields 90.92% accuracy, which is quite close to the SoTA 91.68%. To
explore the limit of LLMs, we also prompt GPT-4 using 2-shot
in-context-learning and achieve 82.69% accuracy, which is a 7.52%
absolute gain compared with 75.17% from GPT-3.5. For a substantial
number of questions, we note that GPT-4 fails simply because it reports
that there is insufficient context such as images or plots. We consider
two schemes to combine the outcomes from our model and GPT-4. $(i)$ *A
GPT-4 complement*. Whenever GPT-4 fails to provide answers, we use the
prediction from our method. This schemes yields 90.97% accuracy, which
is almost the same as applying our method alone. $(ii)$ *GPT-4 as the
judge*. Whenever GPT-4 and produce different answers, we prompt GPT-4
again, asking it to provide its own final answer based on the question
and two outcomes. The spirit is similar with CoT, but with the external
knowledge from the other model. Surprisingly, this scheme is able to
provide consistent improvement over all question classes, and achieves a
new SoTA accuracy of 92.53%. Interestingly, the text-only GPT-4, which
cannot process images, improves the overall performance of the model on
questions that have an image as context. This is because some of these
questions do not actually require the image context for a correct
answer. The GPT-4 judge can identify such cases and correct some of the
errors that makes. See the example in Appendix. To the best of our
knowledge, this is the first time that GPT-4 is used for model
ensembling. We hope this finding can encourage future research to
explore more effective methods to leverage LLMs for model ensembling.

<div class="wrapfigure" markdown="1">

r0.5

<span id="tab:scienceqa_ablation" label="tab:scienceqa_ablation"></span>

</div>

#### Ablations.

We ablate several design choices on ScienceQA in
Table <a href="#tab:scienceqa_ablation" data-reference-type="ref"
data-reference="tab:scienceqa_ablation">[tab:scienceqa_ablation]</a>.
$(i)$ *Visual features*. We tried using the last layer feature from CLIP
vision encoder, which yields 89.96% and is 0.96% lower than the feature
before the last layer. We hypothesize that this is because CLIP’s last
layer features may focus more on global and abstract image properties
compared to the layer before it, which can focus more on localized
properties that are useful for understanding specific image details.
$(ii)$ *Chain-of-thought*. To decide the order between the answer and
reasoning process in the model prediction, we run both variants and
observe that answer-first reports the best number 89.77% accuracy in 12
epochs, while reasoning-first can quickly reach 89.77% accuracy in 6
epochs, but no further improvement with more training. Training the
model for 24 epochs does not improve the performance. We conclude that
CoT-like reasoning-first strategy can largely improve convergence, but
contributes relatively little to the final performance. $(iii)$
*Pre-training*. We skip pre-training and directly train on Science QA
from scratch – performance drops to 85.81% accuracy. The 5.11% absolute
degradation indicates the importance of our pre-training stage, in
aligning multimodal features while preserving the vast pre-trained
knowledge. $(iv)$ *Model size*. We keep all configurations the same as
our best 13B model, and train a 7B model. This yields 89.84% accuracy,
which is 1.08% lower than 90.92%, demonstrating the importance of model
scale.

# Conclusion

This paper demonstrated the effectiveness of visual instruction tuning.
We presented an automatic pipeline to create language-image
instruction-following data, based on which we train , a multimodal model
to follow human intent to complete visual tasks. It achieves the new
SoTA accuracy when fine-tuned on ScienceQA, and excellent visual chat
capabilities when fine-tuned on multimodal chat data. Besides, we
present the first benchmark to study multimodal instruction-following
capability. This paper is an initial step in visual instruction tuning,
and mainly focuses on real-life tasks. For more quantitative results of
on academic benchmarks, please refer to the improved baselines with
visual instruction tuning [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1). We hope
our work can inspire future research on building more capable multimodal
models.

**Acknowledgements.** We thank Baolin Peng and Pan Lu for valuable
discussions on instruction-tuning language models and Science QA,
respectively. We thank the LLaMA team for giving us access to their
models, and open-source projects, including Alpaca and Vicuna. This work
was supported in part by NSF CAREER IIS2150012, and Institute of
Information & communications Technology Planning & Evaluation(IITP)
grants funded by the Korea government(MSIT) (No. 2022-0-00871,
Development of AI Autonomy and Knowledge Enhancement for AI Agent
Collaboration) and (No. RS-2022-00187238, Development of Large Korean
Language Model Technology for Efficient Pre-training).

# Broader Impact

The broader impact of , a general-purpose visual assistant, has
potential benefits and risks associated with its deployment and release.
Some considerations are unique to due to its visual nature, while others
share similarities with existing instruction-following LLMs (Alpaca,
Vicuna, ). As is built upon LLaMA, Vicuna, and CLIP, it inherits some of
the issues associated with LLMs and vision encoders. In the following,
we outline both the risks and mitigation strategies in place for the
release of this model.

#### Malicious input.

To minimize potential misuse and harmful consequences, we employ two
precautionary measures for : (1) *OpenAI Filter API* for user input text
to prevent harmful or inappropriate text instructions from being
processed by the model, and (2) *NSFW Filter* for uploaded user images
to detect and block Not Safe For Work (NSFW) content or any other
potentially harmful image inputs.

#### Hallucination.

Similar to LLMs, might generate outputs that aren’t grounded in facts or
input data. This raises concerns about inferences made, especially in
critical applications (medical).

#### Biases.

Bias can be transferred from the base models to , both from the vision
encoder (CLIP) and the language decoder (LLaMA/Vicuna). This may lead to
biased outcomes or unfair representations of diverse content.

#### Energy consumption.

Though energy consumption is not a primary concern for due to a smaller
pretraining dataset (see details in
Sec. <a href="#sec:appendix_training_details" data-reference-type="ref"
data-reference="sec:appendix_training_details">3</a>), it may become a
concern when scaling up the pretraining dataset or increasing the model
size, e.g., to a larger LLaMA version like the 65B model.

#### Evaluation complexities.

Assessing the performance of is challenging as it involves both language
and visual tasks. Our evaluation benchmark covers several aspects,
including accuracy, concept coverage, reasoning ability, and creativity.
However, additional aspects need consideration, such as the degree of
visual content hallucination and fine-grained understanding of visual
content. While text-only GPT-4 based multimodal evaluation is consistent
and accurate in our study, its robustness in different situations and
capability to evaluate other unexplored aspects are subjects for future
work.

Despite these risks, we believe that the benefits of releasing to the
research community outweigh the potential harm. It allows for ongoing
investigation and improvement of the model and engages the community in
developing better mitigation strategies to address these concerns.
Moreover, the release of can spur the development of new applications
and research directions, ultimately contributing to the progress and
responsible deployment of foundation models in vision-language tasks.

# More Results

We present more qualitative results of to analyze its emergent behaviors
and observed weaknesses. For more quantitative results of on academic
benchmarks, please refer to the improved baselines with visual
instruction tuning [liu2023improvedllava](http://arxiv.org/pdf/2310.19145v1). In
Table <a href="#tab:visual_example_chichken" data-reference-type="ref"
data-reference="tab:visual_example_chichken">[tab:visual_example_chichken]</a>,
demonstrates a similar behavior as GPT-4 in another example from its
paper. Similar to the GPT-4 live demo by OpenAI, is capable of
generating the HTML/JS/CSS code for an interactive joke website based on
a simplified user input sketch in
Fig. <a href="#fig:example_website" data-reference-type="ref"
data-reference="fig:example_website">1</a>, despite a minor error. As
shown in Fig. <a href="#fig:example_recipe" data-reference-type="ref"
data-reference="fig:example_recipe">2</a>, can follow user’s
instructions in a conversational style and provide detailed responses or
creative writings. Furthermore, is able to relate the visual content to
the textual knowledge from the pretrained LLM, as demonstrated in
Fig. <a href="#fig:example_titanic" data-reference-type="ref"
data-reference="fig:example_titanic">3</a> and
Fig. <a href="#fig:example_mona_lisa" data-reference-type="ref"
data-reference="fig:example_mona_lisa">4</a>.

One interesting emergent behavior of is that it is able to understand
visual contents that are not covered in the training. For example, in
Fig. <a href="#fig:example_elon_musk" data-reference-type="ref"
data-reference="fig:example_elon_musk">5</a>, it is able to recognize
Elon Musk both in a headshot and in a humorous meme where he is dressed
as a doge, even though Elon Musk *never* appears in the training data
for either the visual feature alignment or visual instruction tuning
stages of . also demonstrates impressive OCR (optical character
recognition) ability in
Table <a href="#tab:visual_example_chichken" data-reference-type="ref"
data-reference="tab:visual_example_chichken">[tab:visual_example_chichken]</a>
and Fig. <a href="#fig:example_website" data-reference-type="ref"
data-reference="fig:example_website">1</a>, which is rarely covered in
our training data.

We hope these additional results and observations showcase the potential
of in various application areas. In future work, it is important to
investigate these emergent behaviors more thoroughly and to understand
the underlying mechanisms that enable to demonstrate such generalization
abilities. This will pave the way towards building better LMMs,
including enhancing robustness, reducing biases, and improving the
alignment and the scope of the learned vision-language representations.

<span id="tab:visual_example_chichken"
label="tab:visual_example_chichken"></span>

<figure id="fig:example_website">
<embed src="/papers/text_rich/arXiv-2304.08485v2_md/figures/website_code.png" />
<figcaption> generates HTML/JS code for an interactive website based on
user sketch inputs. The interactive interface works after fixing a minor
error (<em>in red</em>) in the generated output. There is room for
improvement in ’s output, such as splitting the joke and punchline into
two rows, and only revealing the punchline upon button click, to better
reflect the user’s intent.</figcaption>
</figure>

<figure id="fig:example_recipe">
<p><embed src="/papers/text_rich/arXiv-2304.08485v2_md/figures/fridge_recipe.png" /> <embed
src="/papers/text_rich/arXiv-2304.08485v2_md/figures/scene_and_review.png" /></p>
<figcaption> is capable of recognizing the visual content following the
user’s intent, without directly prompting for visual recognition. It
also provides a detailed response when prompted with a follow-up
request, and the generated response is closely related to the provided
visual content.</figcaption>
</figure>

<figure id="fig:example_titanic">
<embed src="/papers/text_rich/arXiv-2304.08485v2_md/figures/titanic.png" />
<figcaption> relates the movie scenes to the textual knowledge from the
pretrained LLM.</figcaption>
</figure>

<figure id="fig:example_mona_lisa">
<embed src="/papers/text_rich/arXiv-2304.08485v2_md/figures/monalisa.png" />
<figcaption> recognizes the famous art work, Mona Lisa, by Leonardo da
Vinci. When we start a new conversation, it also explains the humourous
artwork created on the web, mimicking the Mona Lisa.</figcaption>
</figure>

<figure id="fig:example_elon_musk">
<embed src="/papers/text_rich/arXiv-2304.08485v2_md/figures/elon_musk.png" />
<figcaption>An interesting emergent behavior of is its ability to
recognize Elon Musk both in a headshot and in a humorous meme where he
is dressed as a doge. This implies that the pre-trained CLIP vision
encoder may have seen images of Elon Musk. However, it is still
surprising because Elon Musk <em>never</em> appears in the training data
for either the visual feature alignment or visual instruction tuning
stages of , which indicates that the base language model generalizes to
unseen visual concepts.</figcaption>
</figure>

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

|  |  |
|:---|:--:|
|  |  |
| Which material is this rocking chair made of? |  |
| Options: (A) wood (B) silk |  |
|  |  |
| LECTURE: A material is a type of matter. |  |
| Wood, glass, metal, and plastic are common materials. |  |
| Some objects are made of just one material. |  |
| Most nails are made of metal. |  |
| Other objects are made of more than one material. |  |
| This hammer is made of metal and wood. |  |
| SOLUTION: Look at the picture of the rocking chair. The rocking chair is made of two different materials. The legs are made of wood, and the back and seat are made of silk. |  |
| The answer is B. |  |
|  |  |
| The answer is A. BECAUSE: Rocking chairs are typically made of sturdy materials that can support the weight of a person and withstand the rocking motion. Wood is a common material used for making rocking chairs due to its strength, durability, and natural appearance. |  |
|  |  |
|  |  |
| Since there is no context or image to refer to, I cannot independently analyze the rocking chair in question. However, I can try to analyze the reasoning given by the assistants. |  |
| Assistant 1, who claims to have seen the image, mentions that the legs of the rocking chair are made of wood, and the back and seat are made of silk, implying that it’s a combination of two materials. However, this response seems less plausible, as silk is not a common material used for the construction of rocking chairs due to issues with stability and durability. Typically, one would expect the frame, legs, and seat or back to be made of wood, with cushions and padding made of fabrics, such as cotton or polyester. |  |
| Assistant 2 argues that rocking chairs are typically made of sturdy materials, and wood is a common material used for making rocking chairs due to its strength, durability, and natural appearance. This reasoning stands true and is based on common knowledge. |  |
| Given the information available, the most plausible answer is that the rocking chair is made of wood, as suggested by Assistant 1. The answer is A. |  |

</div>

</div>

# Training Details [sec:appendix_training_details]

We pre-train our model on the filtered CC-595K subset for 1 epoch with a
learning rate of 2e-3 and a batch size of 128, and fine-tune on the
proposed LLaVA-Instruct-158K dataset for 3 epochs, with a learning rate
of 2e-5 and a batch size of 32. Following Vicuna, we use the Adam
optimizer with no weight decay and a cosine learning rate with a warmup
ratio of 3%. During finetuning, FSDP (Full Shard Data Parallel) and
gradient checkpointing is used to save GPU memory, and offloading is not
used. BF16 and TF32 are enabled to achieve a balance between speed and
precision.

We train all models with 8$\times$ A100s. Pretraining on CC-595K
completes within 4 hours. Finetuning on Instruct-158K completes within
10 hours. Finetuning on ScienceQA completes within 4 hours.

# Assets

Our source code, generated instruction-tuning data, proposed benchmark
are uploaded to the anonymized GitHub repository:
[LLaVA-Annonymous/LLaVA](https://github.com/LLaVA-Annonymous/LLaVA).

1.  Source Code: [link](https://github.com/LLaVA-Annonymous/LLaVA)

2.  README: [link](https://github.com/LLaVA-Annonymous/LLaVA)

3.  Instructions to launch the demo:
    [link](https://github.com/LLaVA-Annonymous/LLaVA#web-ui)

4.  All prompts and few shot examples for querying GPT-4:
    [link](https://github.com/LLaVA-Annonymous/LLaVA/tree/master/playground/data/prompts)

5.  LLaVA-Instruct-158K:
    [link](https://github.com/LLaVA-Annonymous/LLaVA/blob/master/playground/data/llava_instruct_150k.json)

6.  LLaVA-Bench:
    [COCO](https://github.com/LLaVA-Annonymous/LLaVA/blob/master/playground/data/coco2014_val_gpt4_qa_30x3.jsonl),
    [In-The-Wild](https://github.com/LLaVA-Annonymous/LLaVA/tree/master/playground/data/llava_bench_in_the_wild)

7.  Model checkpoints. The size of the model checkpoints after
    compression is 25GB, which exceeds the 5GB limit of GitHub LFS
    (Large File Storage). We’ll release the checkpoint to the public, or
    upon request with reviewers for this submission.

# Data [sec:appendix_data]

#### Instructions for brief image description.

The list of instructions used to briefly describe the image content are
shown in
Table <a href="#tab:concise_describe_instructions" data-reference-type="ref"
data-reference="tab:concise_describe_instructions">[tab:concise_describe_instructions]</a>.
They present the same meaning with natural language variance.

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

-   "Describe the image concisely."

-   "Provide a brief description of the given image."

-   "Offer a succinct explanation of the picture presented."

-   "Summarize the visual content of the image."

-   "Give a short and clear explanation of the subsequent image."

-   "Share a concise interpretation of the image provided."

-   "Present a compact description of the photo’s key features."

-   "Relay a brief, clear account of the picture shown."

-   "Render a clear and concise summary of the photo."

-   "Write a terse but informative summary of the picture."

-   "Create a compact narrative representing the image presented."

</div>

</div>

#### Instructions for detailed image description.

The list of instructions used to describe the image content in detail
are shown in
Table <a href="#tab:detailed_describe_instructions" data-reference-type="ref"
data-reference="tab:detailed_describe_instructions">[tab:detailed_describe_instructions]</a>.
They present the same meaning with natural language variance.

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

-   "Describe the following image in detail"

-   "Provide a detailed description of the given image"

-   "Give an elaborate explanation of the image you see"

-   "Share a comprehensive rundown of the presented image"

-   "Offer a thorough analysis of the image"

-   "Explain the various aspects of the image before you"

-   "Clarify the contents of the displayed image with great detail"

-   "Characterize the image using a well-detailed description"

-   "Break down the elements of the image in a detailed manner"

-   "Walk through the important details of the image"

-   "Portray the image with a rich, descriptive narrative"

-   "Narrate the contents of the image with precision"

-   "Analyze the image in a comprehensive and detailed manner"

-   "Illustrate the image through a descriptive explanation"

-   "Examine the image closely and share its details"

-   "Write an exhaustive depiction of the given image"

</div>

</div>

#### CC3M.

We extract noun-phrases using Spacy for each caption over the whole CC3M
dataset, and count the frequency of each unique noun-phrase. We skip
noun-phrases whose frequency is smaller than $3$, as they are usually
rare combinations concept and attributes that has already been covered
by other captions. We then start from the noun-phrases with lowest
remaining frequency, add the captions that contain this noun-phrase to
the candidate pool. If the frequency of the noun-phrase is larger than
$100$, we randomly choose a subset of size $100$ out of all its
captions. This results in around 595K image-text pairs.

The comparison of noun-phrase statistics before and after filtering CC3M
is shown in
Figure <a href="#fig:cmp_noun_phrase_counter" data-reference-type="ref"
data-reference="fig:cmp_noun_phrase_counter">[fig:cmp_noun_phrase_counter]</a>.
The filtered dataset shows a good coverage of concepts whose frequency
is higher from 3, but with a smaller number of image-text pairs.

<div class="figure*" markdown="1">

|                                                                          |
|:------------------------------------------------------------------------:|
| <embed src="/papers/text_rich/arXiv-2304.08485v2_md/figures/cmp_noun_phrase_counter.png" style="height:3.6cm" /> |

<span id="fig:cmp_noun_phrase_counter"
label="fig:cmp_noun_phrase_counter"></span>

</div>

# Prompts

The prompt used to generate image-based conversation from ChatGPT/GPT-4
is shown in
Table <a href="#tab:prompt_conversation" data-reference-type="ref"
data-reference="tab:prompt_conversation">[tab:prompt_conversation]</a>.

<span id="sec:appendix_prompts" label="sec:appendix_prompts"></span>

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

|  |
|:---|
|  |
| Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. Ask diverse questions and give corresponding answers. |
| Include questions asking about the visual content of the image, including the **object types, counting the objects, object actions, object locations, relative positions between objects**, etc. Only include questions that have definite answers: |
| \(1\) one can see the content in the image that the question asks about and can answer confidently; |
| \(2\) one can determine confidently from the image that it is not in the image. Do not ask any question that cannot be answered confidently. |
| Also include complex questions that are relevant to the content in the image, for example, asking about background knowledge of the objects in the image, asking to discuss about events happening in the image, etc. Again, do not ask about uncertain details. Provide detailed answers when answering complex questions. For example, give detailed examples or reasoning steps to make the content more convincing and well-organized. You can include multiple paragraphs if necessary."""} |
| \] |
|  |

</div>

</div>

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

|  |  |
|:---|:--:|
|  |  |
| A group of people standing outside of a black vehicle with various luggage. |  |
| Luggage surrounds a vehicle in an underground parking area |  |
| People try to fit all of their luggage in an SUV. |  |
| The sport utility vehicle is parked in the public garage, being packed for a trip |  |
| Some people with luggage near a van that is transporting it. |  |
|  |  |
| person: \[0.681, 0.242, 0.774, 0.694\], person: \[0.63, 0.222, 0.686, 0.516\], person: \[0.444, 0.233, 0.487, 0.34\], backpack: \[0.384, 0.696, 0.485, 0.914\], backpack: \[0.755, 0.413, 0.846, 0.692\], suitcase: \[0.758, 0.413, 0.845, 0.69\], suitcase: \[0.1, 0.497, 0.173, 0.579\], bicycle: \[0.282, 0.363, 0.327, 0.442\], car: \[0.786, 0.25, 0.848, 0.322\], car: \[0.783, 0.27, 0.827, 0.335\], car: \[0.86, 0.254, 0.891, 0.3\], car: \[0.261, 0.101, 0.787, 0.626\] |  |
|  |  |
|  |  |
| Question: What type of vehicle is featured in the image? |  |
| Answer: The image features a black sport utility vehicle (SUV). |  |
| Question: Where is the vehicle parked? |  |
| Answer: The vehicle is parked in an underground parking area, likely in a public garage. |  |
| Question: What are the people in the image doing? |  |
| Answer: The people are trying to fit all of their luggage into the SUV, likely preparing for a trip. |  |
|  |  |
| The image is an underground parking area with a black sport utility vehicle (SUV) parked. There are three people in the scene, with one person standing closer to the left side of the vehicle, another person in the middle, and the third person on the right side. They are all working together to pack their luggage into the SUV for a trip. |  |
| In the parking area, various luggage items are scattered around the vehicle. There are two backpacks, one located near the left rear wheel and the other closer to the right side of the vehicle. Additionally, there are two suitcases, one on the right side of the car and another further away near the center of the parking area. A bicycle can also be seen on the left side of the vehicle. |  |
| Other cars are parked around the main SUV, with one car positioned behind it and slightly to the left, another behind and slightly to the right, and the third car further behind on the right side. |  |
|  |  |
| Question: What challenges do these people face? |  |
| Answer: In the image, a group of people is standing outside a black SUV in a parking area, surrounded by various pieces of luggage, including suitcases and backpacks. They are facing the challenge of fitting all their luggage into the black SUV. There are multiple suitcases and backpacks to be packed, which suggests that the group has a significant amount of belongings to accommodate. They might have to strategize and arrange the luggage efficiently to ensure that everything fits properly into the vehicle. Additionally, they need to consider the comfort of the passengers and visibility while driving, so the placement of the luggage must not obstruct the driver’s view or make the passengers uncomfortable during the trip. |  |

</div>

</div>

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

|  |
|:---|
|  |
| There is a movie theater that displays the show times above the doors. |
| A red fire hydrant is deep in the snow. |
| The fire hydrant is in the snow near a recently plowed sidewalk. |
| This city has had a very hard winter with snow. |
| A hotel for dogs in the snow in winter. |
|  |
| Question: |
| What color is the fire hydrant in the image? |
| === |
| Answer: |
| The fire hydrant in the image is red. |
| === |
| Question: |
| Is the fire hydrant fully covered by snow or partially visible? |
| === |
| Answer: |
| The fire hydrant is partially visible, with a significant portion of it covered by the deep snow. |
| === |
| Question: |
| What is the condition of the sidewalk near the fire hydrant? |
| === |
| Answer: |
| The sidewalk near the fire hydrant has been recently plowed, so it is relatively clear of snow compared to the surrounding area. |
| === |
| Question: |
| What does the image suggest about the winter season in the city? |
| === |
| Answer: |
| The image suggests that the winter season in the city has been quite harsh and challenging. The significant snow accumulation seen in the image indicates that the city has experienced heavy snowfall, which can lead to various difficulties and disruptions for its residents. |
|  |
| For instance, the deep snow covering the fire hydrant could potentially hinder its accessibility in case of emergencies, which poses safety concerns. The recently plowed sidewalk implies that the city’s maintenance crews have been actively working to keep the walkways clear and safe for pedestrians, but the sheer amount of snowfall might make it difficult to keep up with the continuous snow removal efforts. |
|  |
| Furthermore, such extreme winter conditions can impact transportation, leading to traffic issues, delays in public transit, and increased risks of accidents due to slippery roads. It can also cause problems for businesses and schools, as people might face challenges commuting to work or attending classes. Additionally, the heavy snow can put extra strain on infrastructure, such as roofs and power lines, increasing the likelihood of structural damage or power outages. |
|  |
| In conclusion, the image of the red fire hydrant deep in the snow and the recently plowed sidewalk suggest that the city has faced a particularly severe winter season, with substantial snowfall that has likely caused various challenges and disruptions for its residents and infrastructure. |

</div>

</div>

<div class="table*" markdown="1">

<div class="tcolorbox" markdown="1">

|  |
|:---|
|  |
| A man is skiing in the open snow covered hills |
| A skier is making his way into the snow. |
| A skier on the foothills of a huge mountain range. |
| A skier looks at mountains while standing near a trail sign. |
| a single person skiing by an area with a lot of bushes |
|  |
| Question: |
| What is the skier doing? |
| === |
| Answer: |
| The skier is skiing in the snow-covered hills, making his way through the snow and enjoying the beautiful mountain scenery. They are also standing near a trail sign, which indicates their location on the mountain. |
| === |
| Question: |
| Are there any other people or skiers visible in the image? |
| === |
| Answer: |
| No, there is only a single person skiing in the image, and they are surrounded by an area with a lot of bushes. |
| === |
| Question: |
| What kind of skiing is the person doing? Cross-country or downhill? |
| === |
| Answer: |
| Based on what I can see in the photo, it is not possible to definitively determine whether the skier is doing cross-country or downhill skiing. However, given the presence of a trail sign and the fact that they are skiing in the foothills, it is more likely that the skier is engaged in cross-country skiing. Downhill skiing typically takes place on more steep slopes and involves ski lifts, whereas cross-country skiing often occurs on relatively flatter terrain with marked trails. |
| === |
| Question: |
| Can you explain cross-country and downhill skiing a bit more? |
| === |
| Answer: |
| Cross-country skiing and downhill skiing are two distinct types of skiing, each with its own techniques, equipment, and terrain preferences. |
|  |
| 1\. Cross-Country Skiing: |
| Also known as Nordic skiing, cross-country skiing involves traversing relatively flat terrain or gentle slopes using skis and poles. It is a popular recreational and competitive sport that focuses on endurance, balance, and aerobic fitness. Cross-country skiers use their own locomotion to move across snow-covered terrain, rather than relying on ski lifts or other forms of assistance. |
| There are two main techniques in cross-country skiing: |
| \- Classic: This technique involves skiing in parallel tracks, using a forward-and-backward gliding motion while keeping the skis parallel to each other. The skier uses their poles to propel themselves forward. |
| \- Skate skiing: This technique is similar to ice skating, where the skier pushes off one ski at an angle and glides on the other. It requires a smooth, groomed surface and is typically faster than the classic technique. |
| Cross-country ski equipment consists of lightweight skis, boots, bindings, and poles. The boots are more flexible than those used in downhill skiing, allowing for better ankle movement and control. |
|  |
| 2\. Downhill Skiing: |
| Also known as alpine skiing, downhill skiing involves descending slopes at high speeds using skis and poles for balance and control. This sport is more focused on speed, technique, and navigating challenging terrain, including steep slopes, moguls, and even jumps. |
| Downhill skiing can be further categorized into several disciplines, such as slalom, giant slalom, super-G, and downhill racing. Each discipline has its own set of rules, courses, and ski equipment. |
| Downhill ski equipment includes heavier and stiffer skis, boots, bindings, and poles than those used in cross-country skiing. The boots are more rigid to provide better support and control during high-speed descents and sharp turns. |
|  |
| In summary, cross-country skiing is an endurance-based sport that involves traveling across flat or gently sloping terrain, while downhill skiing is focused on speed and technique as skiers navigate steeper slopes and challenging terrain. Both sports require specialized equipment and techniques, but they offer different experiences and challenges to participants. |

</div>

</div>