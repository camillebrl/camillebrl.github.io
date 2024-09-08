# Introduction

Large language models (LLMs) such as GPT-3 [gpt3](http://arxiv.org/pdf/2112.07522v2), BLOOM
[bloom](None), LLaMA [llama](http://arxiv.org/pdf/2402.08075v1) have experienced
rapid development to make general artificial intelligence possible,
which demonstrates impressive zero-shot abilities on various linguistic
applications. However, except GPT-4 [gpt4](None), current
general LLMs cannot support different modalities of input and develop
impressive multimodal abilities.

Although GPT-4 [gpt4](None) has exhibited remarkable
multimodal abilities, the methods behind its extraordinary abilities
remain a mystery. Recently, researchers have been extending LLMs to
understand visual inputs in two different paradigms: systematic
collaboration and end-to-end trained models. However, systematic
collaboration approaches, including Visual ChatGPT
[visualchatgpt](None), MM-REACT [mmreact](None), and
HuggingGPT [hugginggpt](None), are designed to facilitate the
coordination of various vision models or tools to express visual
information with text descriptions. However, these approaches may not be
able to comprehend specific multimodal instructions due to their lack of
alignment with different modalities. Additionally, these approaches may
encounter challenges related to inference efficiency and cost.
End-to-end models, such as BLIP-2 [blip2](None), LLaVA
[llava](http://arxiv.org/pdf/2402.11690v1), and MiniGPT-4 [minigpt4](http://arxiv.org/pdf/2402.17510v1) aim to
use unified models to support different modalities. However, these
models have some limitations as they take frozen visual models, which
may lead to inadequate alignment due to the limited number of
parameters. Moreover, they cannot unlock various abilities due to
missing unimodal and multimodal instruction.

In this paper, we present mPLUG-Owl with an innovative modularized
training paradigm for large multi-modal language models that can support
multiple modalities concurrently, drawing inspiration from the concept
of modularization [mplug2](None), [mplug](None), [e2evlp](None), [hitea](https://doi.org/10.48550/arXiv.2212.14546). Our
method harnesses the power of pre-trained LLM, visual knowledge module,
and connected visual abstractor module to achieve effective alignment
between images and text, and utilizes a two-stage training scheme to
stimulate impressive unimodal and multimodal abilities. Our approach
even enhances the strong generation abilities of LLM by modality
collaboration between modalities. In the first step, we align the image
and text to acquire comprehensive visual knowledge using text-image
pairs, which is accomplished by training the visual knowledge module and
abstractor module with the frozen LLM module. Subsequently, we fine-tune
mPLUG-Owl with language-only and multi-modal instructions to unlock a
range of unimodal and multimodal abilities. We freeze the visual
knowledge module and train low-rank adaption (LoRA)
[lora](http://arxiv.org/pdf/2402.11485v1) on LLM and visual abstractor module jointly. This
approach allows for the effective integration of textual and visual
information, facilitating the development of versatile and robust
cognitive abilities.

Our experiments on a carefully-built visually related instruction
evaluation set OwlEval shows that mPLUG-Owl outperforms existing models
such as MiniGPT-4 [minigpt4](http://arxiv.org/pdf/2402.17510v1) and LLaVA
[llava](http://arxiv.org/pdf/2402.11690v1). We separately verifies mPLUG-Owl’s remarkable
abilities in instruction understanding, visual understanding, knowledge
transfer, and multi-turn dialogue. Abundant ablation study is performed
to show the effectiveness of our training paradigm. Furthermore, we find
some unexpected emerging ability such as multi-image correlation,
multilingual conversation and scene text understanding.

Our main contributions can be highlighted as follows:

-   We propose mPLUG-Owl, a novel training paradigm for large language
    models through modularization.

-   We carefully construct an instruction evaluation set, dubbed
    **OwlEval**, to assess the capabilities of different models in the
    context of visual-related tasks.

-   Experimental results demonstrate that mPLUG-Owl excels in
    multi-modal instruction understanding and multi-turn dialogue,
    surpassing the performance of existing models.

# Related Work

## Large Language Models

In recent times, Large Language Models (LLMs) have garnered increasing
attention for their exceptional performance in diverse natural language
processing (NLP) tasks. Initially, transformer models such as BERT
[bert](None), GPT [gpt1](http://arxiv.org/pdf/2310.01427v1), and T5
[t5](None) were developed with different pre-training
objectives. However, the emergence of GPT-3 [gpt3](http://arxiv.org/pdf/2112.07522v2),
which scales up the number of model parameters and data size, showcases
significant zero-shot generalization abilities, enabling them to perform
commendably on previously unseen tasks. Consequently, numerous LLMs such
as OPT [opt](None), BLOOM [bloom](None), PaLM
[palm](http://arxiv.org/pdf/2209.05735v4), and LLaMA [llama](http://arxiv.org/pdf/2402.08075v1) are created,
ushering in the success of LLMs. Additionally, Ouyang et al.
[instructgpt](http://arxiv.org/pdf/2302.05206v1) propose InstructGPT by aligning human
instruction and feedback with GPT-3. Furthermore, it has been applied to
ChatGPT [chatgpt](https://openai.com/blog/chatgpt), which facilitates conversational
interaction with humans by responding to a broad range of diverse and
intricate queries and instructions.

## Multi-Modal Large Language Models

Despite the successful applications of LLMs in natural language
processing, it is still struggling for LLMs to perceive other modalities
such as vision and audio. Recently, researchers have been extending
language models to understand visual inputs in two different paradigms:
systematic collaboration and end-to-end trained models. Systematic
collaboration approaches, such as Visual ChatGPT
[visualchatgpt](None), MM-REACT [mmreact](None), and
HuggingGPT [hugginggpt](None), leverage various vision experts
or tools to express visual information with text descriptions.
Subsequently, large language models, such as ChatGPT, can act as the
agents, and be prompted to select the appropriate experts and tools for
visual understanding. Finally, LLMs would summarize the output of these
experts to answer user queries. On the other hand, some approaches
[blip2](None), [flamingo](http://arxiv.org/pdf/2205.07065v1), [llava](http://arxiv.org/pdf/2402.11690v1) leverage the pre-trained large
language model to build unified models for multi-modality. For example,
Flamingo [flamingo](http://arxiv.org/pdf/2205.07065v1) freezes the pre-trained vision
encoder and large language model and fuses vision and language
modalities with gated cross-attention showing impressive few-shot
capabilities. Additionally, BLIP-2 [blip2](None) designs
Q-Former to align the visual features from the frozen visual encoder and
large language models with Flan-T5 [flant5](http://arxiv.org/pdf/2202.03371v1) and OPT
[opt](None). Moreover, PaLM-E [palm-e](http://arxiv.org/pdf/2302.14030v3)
directly inputs features from sensor modalities with PaLM
[palm](http://arxiv.org/pdf/2209.05735v4), which has 520 billion parameters, contributing
to robust performance in real-world perceptions. Furthermore, some
powerful instruction-tuned language models that built upon open-sourced
foundation model LLaMA [llama](http://arxiv.org/pdf/2402.08075v1), such as Alpaca
[alpaca](https://github.com/tatsu-lab/stanford_alpaca) and Vicuna [vicuna](https://github.com/lm-sys/FastChat), exhibit
comparable performance to ChatGPT [chatgpt](https://openai.com/blog/chatgpt) and GPT-4
[gpt4](None). MiniGPT-4 [minigpt4](http://arxiv.org/pdf/2402.17510v1) and LLaVA
[llava](http://arxiv.org/pdf/2402.11690v1) align these finetuned models with extracted
visual features from the frozen visual backbone. In contrast, mPLUG-Owl
not only aligns the representation between the vision and language
foundation model (e.g. CLIP and LLaMA) in terms of knowledge acquisition
and grounding to the real world but also can understand language and
multi-modal instructions, showcasing strong zero-shot generalization and
multi-turn conversation capabilities.

# mPLUG-Owl

<figure id="fig:compare_method">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/compare.png" />
<figcaption>Comparison between different training paradigms. All of
these methods are trained in a two-stage fashion. Stage 1 stands for
pre-training and Stage 2 represents instruction tuning.</figcaption>
</figure>

<div class="figure*" markdown="1">

<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/model.png" />

</div>

## Architecture Overview

As illustrated in Figure
<a href="#fig:compare_method" data-reference-type="ref"
data-reference="fig:compare_method">1</a>, there exist mainly three
types of end-to-end multimodal LLMs: 1) models that utilize limited
parameters with frozen LLM and visual models during pretraining and
instruction tuning, such as MiniGPT4; 2) models that incorporate
trainable LLMs and frozen visual models, exemplified by Kosmos-1; and 3)
models that involve trainable LLMs during instruction tuning and frozen
visual models, as seen in LLaVA. Nevertheless, these models exhibit
certain constraints since they depend on frozen visual models, which can
lead to insufficient alignment due to the limited number of parameters.
Furthermore, they fail to effectively stimulate a diverse set of
abilities, as they lack both unimodal and multimodal instruction.

To this end, we propose mPLUG-Owl, a multi-modal language model that is
capable of perceiving various modalities while taking the visual context
and information into account and generating corresponding outputs.
Specifically, as illustrated in Figure
<a href="#fig:model" data-reference-type="ref"
data-reference="fig:model">[fig:model]</a>, mPLUG-Owl consists of a
vision foundation model $f_{\mathbf{V}}$ to encode the visual knowledge,
a language foundation model $f_{\mathbf{L}}$, and a visual abstractor
module $f_{\mathbf{K}}$. We first obtain dense image representations
from the pre-trained visual foundation model $f_{\mathbf{V}}$. However,
such dense features would fragment the fine-grained image information
and bring large computation due to the lengthy sequence when feeding
into $f_{\mathbf{L}}$. To mitigate this issue, we employ the visual
abstractor module $f_{\mathbf{K}}$ to summarize visual information
within several learnable tokens, thereby obtaining higher semantic
visual representations and reducing computation, as illustrated in
Figure <a href="#fig:model" data-reference-type="ref"
data-reference="fig:model">[fig:model]</a>. The visual representations
are combined with text queries and fed into the language model to
generate the response.

## Training Scheme

#### Multimodal Pretraining

Large-scale language models, such as GPT-3 [gpt3](http://arxiv.org/pdf/2112.07522v2) and
LLaMA [llama](http://arxiv.org/pdf/2402.08075v1), are trained on extensive and diverse data
collected from the internet, providing them with a comprehensive
understanding of the world. This vast knowledge base endows these models
with remarkable capabilities across a range of tasks. However, the
utilization of visual information in such models remains underexplored.
Previous approaches [minigpt4](http://arxiv.org/pdf/2402.17510v1), [llava](http://arxiv.org/pdf/2402.11690v1) have employed a
limited number of additional parameters to learn the alignment between
visual data and language models, constraining their capacity to
comprehend complex visual information. To enhance the ability of
large-scale language models to perceive visual information while
integrating their internal abilities, we propose a novel training
paradigm that incorporates a trainable visual backbone $f_{\mathbf{V}}$
and an additional visual abstractor $f_{\mathbf{K}}$, while maintaining
the pre-trained language model $f_{\mathbf{L}}$ in a frozen state. This
approach enables the model to effectively capture both low-level and
higher semantic visual information and align it with the pre-trained
language model without compromising its performance.

#### Joint Instruction Tuning

Upon completion of the prior phase, the model acquires the ability to
retain a considerable amount of knowledge and provide reasonable answers
to human queries. Nonetheless, it continues to exhibit challenges in
generating coherent linguistic responses. As posited in GPT-3
[gpt3](http://arxiv.org/pdf/2112.07522v2), refining the model through instruction tuning is
essential for accurately discerning user intentions. Previous attempts
[mplug](None), [mplug2](None) in multi-modal learning have
demonstrated that joint learning from uni-modal and multi-modal sources
can lead to significant improvements owing to the collaboration between
different modalities. Building on this insight, we present a novel
vision-language joint instruction tuning strategy to facilitate better
alignment between mPLUG-Owl and human instructions and intentions.
Specifically, given that the model can comprehend the visual concepts
and knowledge depicted in images through visual knowledge learning, we
freeze the entire model and employ low-rank adaption (i.e., LoRA
[lora](http://arxiv.org/pdf/2402.11485v1)) to adapt $f_{\mathbf{L}}$ by training multiple
low-rank matrices for efficient alignment with human instructions. For
each data record, we unified them in a snippet of conversation following
Vicuna [vicuna](https://github.com/lm-sys/FastChat), and we compute the loss on the
response. During the training, we accumulate the gradient for text-only
instruction data and multi-modal instruction data for multiple batches
and updated the parameters. Therefore, by joint training with both
language and multi-modal instructions, mPLUG-Owl can better understand a
wide range of instructions and respond with more natural and reliable
output. Moreover, our approach can easily handle various text and
multi-modal instructions without the need for realignment of the vision
and language models, as required by methods such as MiniGPT-4
[minigpt4](http://arxiv.org/pdf/2402.17510v1) and LLaVA [llava](http://arxiv.org/pdf/2402.11690v1).

#### Training Objective

The model is trained using the language modeling task, which entails
learning to generate subsequent tokens based on the preceding context.
The primary objective of the training process is to maximize the
log-likelihood of the tokens. It is important to note that only discrete
tokens, such as text tokens, are considered in the calculation of the
training loss. Most significantly, the emergence of diverse capabilities
resulting from the training task during the joint instruction tuning
stage enhances the performance of mPLUG-Owl in downstream applications.

# Experiment

## Experimental Setup

#### Model Settings.

We choose ViT-L/14 [vit](http://arxiv.org/pdf/2105.15075v2) as the visual foundation model
$f_{\mathbf{V}}$ which has 24 layers with hidden dimension set as 1024
and patch size set as 14. For faster convergence, the ViT is initialized
from CLIP ViT-L/14 model pre-trained via contrastive learning. Different
with LLaVA [llava](http://arxiv.org/pdf/2402.11690v1) and MiniGPT-4
[minigpt4](http://arxiv.org/pdf/2402.17510v1), to demonstrate the effectiveness and
generalization ability, we utilize raw LLaMA-7B [llama](http://arxiv.org/pdf/2402.08075v1)
rather than its instruction-tuned variants such as Alpaca
[alpaca](https://github.com/tatsu-lab/stanford_alpaca) and Vicuna [vicuna](https://github.com/lm-sys/FastChat). The total
number of parameters of mPLUG-Owl is about 7.2B. More details about
hyper-parameters can be found in Appendix.

#### Data and Training Details.

For the first stage, we utilize the image-caption pairs from several
datasets, including LAION-400M [laion400m](None), COYO-700M
[coyo700m](https://github.com/kakaobrain/coyo-dataset), Conceptual Captions
[conceptualcap](None) and MSCOCO [cococap](None). We
use a batch size of 2.1 million tokens and train mPLUG-Owl for 50k
steps, corresponding to about 104 billion tokens. We adopt the AdamW
optimizer with $\beta=(0.9, 0.98)$, and set the learning rate and weight
decay to 0.0001 and 0.1 respectively. We warm up the training with 2k
warm-up steps then decay the learning rate with the cosine schedule. The
input image is randomly resized to $224\times 224$. Besides, we tokenize
the text input with SentencePiece [sentencepiece](None)
tokenizer. For the second stage, we gather pure text instruction data
from three distinct sources: 102k data from the Alpaca
[alpaca](https://github.com/tatsu-lab/stanford_alpaca), 90k from the Vicuna [vicuna](https://github.com/lm-sys/FastChat),
and 50k from the Baize [baize](None). Additionally, we utilize
150k multi-modal instruction data from the LLaVA dataset
[llava](http://arxiv.org/pdf/2402.11690v1). We train mPLUG-Owl for 2k steps with the batch
size 256, and the learning rate is set to 0.00002.

#### Baselines.

We compare our mPLUG-Owl with end-to-end models and systematic
collaboration approaches as follows:

-   *OpenFlamingo* [openflamingo](None) is an open-source
    version of Flamingo [flamingo](http://arxiv.org/pdf/2205.07065v1) model. We use the
    released code of OpenFlamingo-9B[^3] to run zero-shot generation.

-   *BLIP-2* [blip2](None) is pre-trained through bootstrapped
    learning from off-the-shelf frozen pre-trained image models and
    large language models using an efficient pre-training strategy. We
    use the released code of BLIP-2 ViT-G FlanT5$_{XXL}$[^4] to perform
    zero-shot generation.

-   *MiniGPT-4* [minigpt4](http://arxiv.org/pdf/2402.17510v1) utilizes a single projection
    layer to align visual information from a pre-trained vision encoder
    with LLM. Specifically, they employ the same visual encoder as used
    in BLIP-2, a ViT coupled with their pre-trained Q-Former, and Vicuna
    as LLM. We use the released demonstration[^5] to perform
    image-instruction generation.

-   *LLaVA* [llava](http://arxiv.org/pdf/2402.11690v1) applies a single projection layer to
    convert image features from pre-trained CLIP visual encoder ViT-L/14
    into the language embedding space of Vicuna. We use their released
    demonstration[^6] to perform image-instruction generation.

-   *MM-REACT* [mmreact](None) integrates ChatGPT/GPT-4 with
    various specialized vision experts to achieve multimodal reasoning
    and action. We use their released demonstration[^7] to get
    responses.

## Quantitative analysis

<figure id="fig:compare_result">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/mPLUG_Owl_compare_result_nomm.png" />
<figcaption>The comparison between mPLUG-Owl and baselines on OwlEval
with manual evaluation metrics. The order of response quality ranking is
as follows: A &gt; B &gt; C &gt; D.</figcaption>
</figure>

In order to comprehensively evaluate various models, we construct a
visually-related evaluation set **OwlEval** by collecting 82
artificially constructed questions based on 50 images, where 21 from
MiniGPT-4, 13 from MM-REACT, 9 from BLIP-2, 3 from GPT-4 and 4 collected
by us. Partial images have multiple rounds of questions, refers to
multi-turn conversation cases. These questions examine a variety of
model capabilities including natural image understanding, diagram and
flowchart comprehension, optical character recognition (OCR),
multi-modal creation, knowledge-intensive QA, and referential
interaction QA. As questions are open-ended, we employ manual evaluation
metrics to rate the model’s responses as A, B, C, or D following the
rating method proposed in Self-Instruct [self-instruct](https://doi.org/10.48550/arXiv.2212.10560).

We manually score 82 responses given by mPLUG-Owl and baselines. The
comparison results are shown in
Figure <a href="#fig:compare_result" data-reference-type="ref"
data-reference="fig:compare_result">2</a>. First, mPLUG-Owl gets 66 $A$
and $B$, while the most competitive baseline MiniGPT-4 gets 54. Second,
mPLUG-Owl doesn’t get any $D$ scores, outperforming all the models.
These results suggest that mPLUG-Owl can better understand both
instructions and images, which results in a stronger capability in
generating satisfactory responses. For a fair comparison, we have
excluded those cases in which MM-REACT failed to make predictions. The
results are shown separately in
Figure <a href="#fig:mm-react" data-reference-type="ref"
data-reference="fig:mm-react">14</a> and mPLUG-Owl still exhibits
superior performance.

To separately examine the single-turn and multi-turn conversation
capabilities, we reorganize 82 questions into a single-turn conversation
set and a multi-turn conversation set. The former contains the first
question from 50 images. The latter contains 52 questions from
multi-turn conversation cases. As shown in
Figure <a href="#fig:compare_result_s_m" data-reference-type="ref"
data-reference="fig:compare_result_s_m">3</a>, the mPLUG-Owl achieves
outstanding performance in both single-turn and multi-turn
conversations.

<figure id="fig:compare_result_s_m">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/mPLUG_Owl_compare_result_s_mturn.png" />
<figcaption>The comparison results of 50 single-turn responses (left)
and 52 multi-turn responses (right) among mPLUG-Owl and baselines on
OwlEval with manual evaluation metrics.</figcaption>
</figure>

## Ablation Study

We ablate the two-stage training scheme and the data modality of
instruction tuning. Six dimensions of abilities are defined to complete
visually related tasks, as shown in
Table <a href="#fig:mult-modle-level" data-reference-type="ref"
data-reference="fig:mult-modle-level">[fig:mult-modle-level]</a>. For
each question, we manually label the required abilities and annotate
which abilities are reflected in the model’s response.
Table <a href="#tb:ablation" data-reference-type="ref"
data-reference="tb:ablation">[tb:ablation]</a> shows the ability
accuracy of different variants of mPLUG-Owl.

<div class="table*" markdown="1">

<span id="fig:mult-modle-level" label="fig:mult-modle-level"></span>

</div>

<div class="table*" markdown="1">

</div>

**Training Strategy Ablation.** As shown in
Table <a href="#tb:ablation" data-reference-type="ref"
data-reference="tb:ablation">[tb:ablation]</a>, without joint
instruction tuning, the model is not good at instruction understanding
and fail to generalize pre-training abilities to other tasks (r1 vs r5).
With the instruction tuning alone, although the model can better
comprehend instructions, the model is incapable of achieving promising
performance in visual knowledge-related tasks due to lacking of
visually-related knowledge pretraining (r2 vs r5). With both multimodal
pretraining and joint instruction tuning, the model achieves the best
performance and demonstrates the effectiveness of our two-stage training
scheme.

**Instruction Data Ablation.** By comparing r3 with r4, text-only
instruction tuning brings more improvement in instruction understanding,
while multi-modal instruction tuning achieves better knowledge and
reasoning capabilities. This is due to that visual question answering
mainly requires the alignment of vision and language knowledge, which is
not optimized during text-only instruction tuning. Besides, we also
verify that introducing multi-modal data during instruction tuning could
further improve the model’s performance on text-only tasks, as shown in
Table <a href="#tab:text-only result" data-reference-type="ref"
data-reference="tab:text-only result">[tab:text-only result]</a> (r5 vs
r4). Concretely, following the evaluation setting as
Vicuna[vicuna](https://github.com/lm-sys/FastChat), for each question, we pair the response
of each model with the one given by ChatGPT and prompt ChatGPT[^8] to
give two scores respectively for these two responses. Table
<a href="#tab:text-only result" data-reference-type="ref"
data-reference="tab:text-only result">[tab:text-only result]</a> shows
the total score and the score ratio with the ChatGPT score as a
reference.

<div class="table*" markdown="1">

</div>

## Qualitative Analysis

In this section, we show qualitative results from our evaluation set
OwlEval.

<figure id="fig:case_JT">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/case_JT.jpg" />
<figcaption>A comparison of Knowledge-intensive QA.</figcaption>
</figure>

#### Knowledge-intensive QA

As shown in Figure <a href="#fig:case_JT" data-reference-type="ref"
data-reference="fig:case_JT">4</a>, the instruction expects the model to
identify the movie characters in the image. MM-REACT is unable to
provide an effective response to the instruction, while MiniGPT-4
understands the instruction but failed to answer the movie characters.
In contrast, mPLUG-Owl answers four out of the five characters present
in the image. This demonstrates that mPLUG-Owl has a better
understanding of the knowledge in the image.

<figure id="fig:case_Yao_Ming">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/case_Yao_Ming.jpg" />
<figcaption>A comparison of Multi-turn Conversation.</figcaption>
</figure>

#### Multi-round Conversation

The instruction in
Figure <a href="#fig:case_Yao_Ming" data-reference-type="ref"
data-reference="fig:case_Yao_Ming">5</a> requires the model to identify
the content of the image based on the referential information. The
baseline models often made mistakes when faced with referential
expressions related to spatial orientation, human behavior, and target
attributes in the questions, whereas mPLUG-Owl provided the most
accurate response. This capability stems from mPLUG-Owl’s fine-grained
understanding of the image, allowing it to locate the corresponding part
of the image based on the referential information in the instruction.

<figure id="fig:case_Final">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/case_Final.jpg" />
<figcaption>A comparison of Reasoning QA.</figcaption>
</figure>

#### Reasoning

Figure <a href="#fig:case_Final" data-reference-type="ref"
data-reference="fig:case_Final">6</a> shows an instruction asking models
to give a prediction based on visual information and explain the reason.
mPLUG-Owl analyzes the characteristics of the two teams from the aspects
of the lineup and tactics and uses them to reason for the outcome.
Although MiniGPT-4 also performs well, its persuasiveness in reasoning
is slightly inferior to mPLUG-Owl.

<figure id="fig:case_GPT4">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/OwlvsGPT4.jpg" />
<figcaption>A comparison of Joke Understanding.</figcaption>
</figure>

<figure id="fig:Memes_and_Jokes_scoreA">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/Memes_and_Jokes_scoreA.png" style="width:100.0%" />
<figcaption>More cases of Jokes Comprehension by mPLUG-Owl.</figcaption>
</figure>

#### Joke Comprehension

The case in Figure <a href="#fig:case_GPT4" data-reference-type="ref"
data-reference="fig:case_GPT4">7</a> comes from the
GPT-4[gpt4](None), which requires the model to understand and
explain a visually related joke. GPT-4 not only follows the instructions
in performing analysis panel by panel but also almost perfectly
understands the humor of the charging method. mPLUG-Owl also understands
this unusual humor, but it incorrectly identified the “VGA” to “USB”.
This is mainly due to the limitation of visual information in our
training data. More cases about joke comprehension are shown in
Figure <a href="#fig:Memes_and_Jokes_scoreA" data-reference-type="ref"
data-reference="fig:Memes_and_Jokes_scoreA">8</a>.

# Discussion and Limitation

In this section, we show some nascent abilities of mPLUG-Owl that is not
yet fully developed and discuss the limitation. Part of cases (without
scores) in this section are not in OwlEval.

<figure id="fig:appendix_case_twoimg">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/multi-image.jpg" />
<figcaption>Multi-image correlation cases.</figcaption>
</figure>

#### Multi-image Correlation

In Figure <a href="#fig:appendix_case_twoimg" data-reference-type="ref"
data-reference="fig:appendix_case_twoimg">9</a>, mPLUG-Owl shows a
emerging but not strong vision correlation capability across multiple
images. In the left case, the model could identify an identical person
in two images and correctly tell the difference of cloth color. But in
the left case, the model fails to relate 4 images and produces some text
hallucinations.

<figure id="fig:multilingual">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/multi-language.jpg" />
<figcaption>Example prompt of multilingual understanding which showcases
the multilingual abilities across Chinese, French, and Japanese,
respectively.</figcaption>
</figure>

#### Multilingual Conversation

Besides English, we further test the model’s multilingual ability. As
shown in Figure <a href="#fig:multilingual" data-reference-type="ref"
data-reference="fig:multilingual">10</a>, although there is no
multilingual data during our two-stage training, mPLUG-Owl shows a
promising multilingual understanding for Chinese, French and Japanese.
We mainly attribute this ability to the raw text knowledge in
LLaMa[llama](http://arxiv.org/pdf/2402.08075v1). However, due to the lacking of
multilingual training, mPLUG-Owl may fail to response in corresponding
languages.

#### Scene Text Understanding

In Figure <a href="#fig:OCR_1_scoreB" data-reference-type="ref"
data-reference="fig:OCR_1_scoreB">15</a>, mPLUG-Owl demonstrates its OCR
ability in some simple scenes, but we can see that the model’s
perception of numbers in images is still limited. However, for the OCR
of complex scenes, as shown in Figure
<a href="#fig:OCR_1_scoreC_a" data-reference-type="ref"
data-reference="fig:OCR_1_scoreC_a">16</a>-<a href="#fig:OCR_1_scoreC_b" data-reference-type="ref"
data-reference="fig:OCR_1_scoreC_b">17</a>, the performance of mPLUG-Owl
is more general, mainly because the perception of numbers in images is
weak, which affects the subsequent reasoning calculation.

#### Vision-only Document Comprehension

Although we did not use any document annotation data for training, the
model exhibited some text recognition and document understanding
capabilities. Hence, we delved deeper into the combination of document
understanding and functionality of our model. as illustrated in Figure
<a href="#fig:document_app" data-reference-type="ref"
data-reference="fig:document_app">11</a>, we explored movie review
writing, code generation, code explanation, chat summary, and
application guidance. The model show decent performance in (a) and (b),
but still, had some errors. Meanwhile, it was unable to provide usable
responses in (d), (e), and (f). Therefore, there is further scope to
explore our model’s potential in document understanding and downstream
applications.

<figure id="fig:document_app">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/document_cases.png" />
<figcaption>Examples about various document understanding and
application.</figcaption>
</figure>

#### Open-ended Creation

mPLUG-Owl performs well in the creation of poetry, lyrics,
advertisements and other works based on images. Its performance in some
cases is shown in Figure
<a href="#fig:create_scoreA" data-reference-type="ref"
data-reference="fig:create_scoreA">12</a>-<a href="#fig:copywriting" data-reference-type="ref"
data-reference="fig:copywriting">13</a>. However, further exploration is
needed for more functional and practical creations.

<figure id="fig:create_scoreA">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/create_scoreA.png" />
<figcaption>Open-ended creation cases.</figcaption>
</figure>

<figure id="fig:copywriting">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/create.png" />
<figcaption>Copywriting cases.</figcaption>
</figure>

# Conclusion

We propose mPLUG-Owl, a novel training paradigm that enhances the
multi-modal abilities of large language models (LLMs). Our approach
consists of modularized learning of foundation LLM, a visual knowledge
module, and a visual abstractor module, which can support multiple
modalities and facilitate diverse unimodal and multimodal abilities
through modality collaboration. We employ a two-stage method for
aligning image and text, which learns visual knowledge with the
assistance of LLM while maintaining and even improving the generation
abilities of LLM. Experimental results demonstrate the impressive
capabilities of mPLUG-Owl, indicating its potential for various
applications in multi-modal generation.

# Training Hyperparameters

We report the detailed model training hyperparameters for visual
knowledge learning in
Table <a href="#tbl:hyperparam:pt" data-reference-type="ref"
data-reference="tbl:hyperparam:pt">1</a> and vision-language joint
instruction tuning in
Table <a href="#tbl:hyperparam:ft" data-reference-type="ref"
data-reference="tbl:hyperparam:ft">2</a>.

<div id="tbl:hyperparam:pt" markdown="1">

| **Hyperparameters**               |             |
|:----------------------------------|:-----------:|
| Training steps                    |   50,000    |
| Warmup steps                      |     375     |
| Max length                        |     512     |
| Batch size of image-caption pairs |    4,096    |
| Optimizer                         |    AdamW    |
| Learning rate                     |    2e-4     |
| Learning rate decay               |   Cosine    |
| Adam $\epsilon$                   |    1e-6     |
| Adam $\beta$                      | (0.9, 0.98) |
| Weight decay                      |    0.01     |

Training hyperparameters for multi-modal pre-training stage.

</div>

<div id="tbl:hyperparam:ft" markdown="1">

| **Hyperparameters**                        |              |
|:-------------------------------------------|:------------:|
| Training steps                             |    2,000     |
| Warmup steps                               |      50      |
| Max length                                 |    1,024     |
| Batch size of text instruction data        |     128      |
| Batch size of multi-modal instruction data |     128      |
| Optimizer                                  |    AdamW     |
| Learning rate                              |     2e-5     |
| Learning rate decay                        |    Cosine    |
| AdamW $\epsilon$                           |     1e-6     |
| AdamW $\beta$                              | (0.9, 0.999) |
| Weight decay                               |    0.0001    |

Training hyperparameters for vision-language joint instruction tuning
stage.

</div>

# Comparison with MM-REACT

<figure id="fig:mm-react">
<img src="/papers/text_rich/arXiv-2304.14178v3_md/figs/mm-react.jpg" />
<figcaption>The comparison results which exclude the cases that were
generated unsuccessfully by MM-REACT.</figcaption>
</figure>

<figure id="fig:OCR_1_scoreB">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/OCR_1_scoreB.png" />
<figcaption>OCR of simple scenes (mostly scenes with few numbers and no
calculation a).</figcaption>
</figure>

<figure id="fig:OCR_1_scoreC_a">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/OCR_1_scoreC.png" />
<figcaption>OCR of complex scenes (a).</figcaption>
</figure>

<figure id="fig:OCR_1_scoreC_b">
<embed src="/papers/text_rich/arXiv-2304.14178v3_md/figs/OCR_2_scoreC.png" />
<figcaption>OCR of complex scenes (b).</figcaption>
</figure>

[^1]: Equal contribution

[^2]: Corresponding author

[^3]: <https://github.com/mlfoundations/open_flamingo>

[^4]: <https://github.com/salesforce/LAVIS/tree/main/projects/blip2>

[^5]: <https://huggingface.co/spaces/Vision-CAIR/minigpt4>

[^6]: <https://llava.hliu.cc>

[^7]: <https://huggingface.co/spaces/microsoft-cognitive-service/mm-react>

[^8]: Without access to the GPT-4, we use the ChatGPT as the suboptimal
    scorer.