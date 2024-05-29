# Introduction

# Related Works

# Method

# Experiments

## Limitation analysis

# Discussion

# Appendix

[^1]: equal contribution

The recent GPT-4 has demonstrated extraordinary multi-modal abilities,
such as directly generating websites from handwritten text and
identifying humorous elements within images. These features are rarely
observed in previous vision-language models. However, the technical
details behind GPT-4 continue to remain undisclosed. We believe that the
enhanced multi-modal generation capabilities of GPT-4 stem from the
utilization of sophisticated large language models (LLM). To examine
this phenomenon, we present MiniGPT-4, which aligns a frozen visual
encoder with a frozen advanced LLM, Vicuna, using one projection layer.
Our work, for the first time, uncovers that properly aligning the visual
features with an advanced large language model can possess numerous
advanced multi-modal abilities demonstrated by GPT-4, such as detailed
image description generation and website creation from hand-drawn
drafts. Furthermore, we also observe other emerging capabilities in
MiniGPT-4, including writing stories and poems inspired by given images,
teaching users how to cook based on food photos, and so on. In our
experiment, we found that the model trained on short image caption pairs
could produce unnatural language outputs (e.g., repetition and
fragmentation). To address this problem, we curate a detailed image
description dataset in the second stage to finetune the model, which
consequently improves the model’s generation reliability and overall
usability. Our code, pre-trained model, and collected dataset are
available at <https://minigpt-4.github.io/>.

In recent years, large language models (LLMs) have experienced rapid
advancements [instructGPT](http://arxiv.org/pdf/2302.05206v1), [chatgpt](http://arxiv.org/pdf/2307.11380v2), [gpt3](http://arxiv.org/pdf/2112.07522v2), [bloom](http://arxiv.org/pdf/2106.06683v2), [llama](http://arxiv.org/pdf/2402.08075v1), [chowdhery2022palm](http://arxiv.org/pdf/2209.05735v4), [hoffmann2022training](http://arxiv.org/pdf/2202.03371v1).
With exceptional language understanding capabilities, these models can
perform a variety of intricate linguistic tasks in a zero-shot manner.
Notably, GPT-4, a large-scale multimodal model, has been recently
introduced and demonstrated several impressive capabilities of
vision-language understanding and generation [gpt4](http://arxiv.org/pdf/2311.15732v2). For
example, GPT-4 can produce detailed and accurate image descriptions,
explain unusual visual phenomena, and even construct websites based on
handwritten text instructions.

Although GPT-4 has exhibited remarkable vision language capabilities,
the methods behind its exceptional abilities are still a mystery
[gpt4](http://arxiv.org/pdf/2311.15732v2). We believe that these impressive skills may stem
from the utilization of a more advanced large language model (LLM). LLMs
have demonstrated various emergent abilities, as evidenced in GPT-3’s
few-shot prompting setup [gpt3](http://arxiv.org/pdf/2112.07522v2) and the findings of Wei
*et al*. (2022) [wei2022emergent](https://openreview.net/forum?id=yzkSU5zdwD). Such emergent
properties are hard to find in smaller-scale models. It is conjectured
that these emergent abilities are also applicable to multi-modal models,
which could be the foundation of GPT-4’s impressive visual description
capabilities.

To substantiate our hypothesis, we present a novel vision-language model
named MiniGPT-4. It utilizes an advanced large language model (LLM),
Vicuna [vicuna2023](https://vicuna.lmsys.org), which is built upon
LLaMA [llama](http://arxiv.org/pdf/2402.08075v1) and reported to achieve 90% of ChatGPT’s
quality as per GPT-4’s evaluation, as the language decoder. In terms of
visual perception, we employ the same pretrained vision components of
BLIP-2 [blip2](http://arxiv.org/pdf/2301.12597v3) that consists of a ViT-G/14 from
EVA-CLIP [fang2022eva](http://arxiv.org/pdf/2402.18128v1) and a Q-Former network. MiniGPT-4
adds a single projection layer to align the encoded visual features with
the Vicuna language model and freezes all the other vision and language
components. MiniGPT-4 is initially trained for 20k steps using a batch
size of 256 on 4 A100 GPUs, leveraging a combined image captioning
dataset that includes images from LAION [laion](http://arxiv.org/pdf/2111.02114v1),
Conceptual
Captions [changpinyo2021conceptual](http://arxiv.org/pdf/2102.08981v2), [sharma2018conceptual](http://arxiv.org/pdf/2304.13130v1),
and SBU [ordonez2011im2text](http://arxiv.org/pdf/2204.00679v1) to align visual features
with the Vicuna language model. Nevertheless, merely aligning visual
features with the language model (LLM) is inadequate to ensure robust
visual conversation capabilities, resembling that of a chatbot. The
presence of underlying noise in raw image-text pairs can lead to subpar
language outputs. Therefore, we collect another  3,500 detailed image
description pairs to further fine-tune the model with a designed
conversational template in order to improve the naturalness of the
generated language and its usability.

<figure id="fig:overview">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/overview.png" style="width:75.0%" />
<figcaption> <strong>The architecture of MiniGPT-4.</strong> It consists
of a vision encoder with a pretrained ViT and Q-Former, a single linear
projection layer, and an advanced Vicuna large language model. MiniGPT-4
only requires training the linear projection layer to align the visual
features with the Vicuna. </figcaption>
</figure>

In our experiments, we discovered that MiniGPT-4 possesses numerous
capabilities similar to those demonstrated by GPT-4. For instance,
MiniGPT-4 can generate intricate image descriptions, create websites
based on handwritten text instructions, and explain unusual visual
phenomena. Furthermore, our findings revealed that MiniGPT-4 also has a
variety of other intriguing abilities not showcased in the GPT-4
demonstrations. For example, MiniGPT-4 can directly generate detailed
cooking recipes from food photos, write stories or poems inspired by
images, write advertisements for products in images, identify problems
shown in photos and provide corresponding solutions, and retrieve rich
facts about people, movies, or art directly from images, among other
capabilities. These abilities are absent in previous vision-language
models like Kosmos-1 [kosmos](http://arxiv.org/pdf/2302.14045v2) and BLIP-2
[blip2](http://arxiv.org/pdf/2301.12597v3) that use less powerful language models. This
further validates that integrating visual features with an advanced
language model is one of the keys to enhancing vision-language models.

We present a summary of our key findings:

-   Our research reveals with compelling evidence that by aligning
    visual features with advanced large language models like Vicuna,
    MiniGPT-4 can achieve advanced vision-language capabilities
    comparable to those exhibited in the GPT-4 demonstrations.

-   Our findings suggest that training merely one projection layer can
    effectively align a pretrained vision encoder with the large
    language model. Our MiniGPT-4 only requires training approximately
    10 hours on 4 A100 GPUs.

-   We discovered that simply aligning visual features with large
    language models using short image caption pairs is not sufficient
    for developing a well-performing model and leads to unnatural
    language generation. Further finetuning with a small but detailed
    image description pairs can address this limitation and
    significantly improves its usability.

**Large language models** have experienced tremendous success in recent
years due to the scaling up of training data and an increase in the
number of parameters. Early models, such as BERT [bert](http://arxiv.org/pdf/1810.04805v2),
GPT-2 [gpt2](http://arxiv.org/pdf/2203.12926v1), and T5 [t5](http://arxiv.org/pdf/1910.10683v4), laid the
foundation for this progress. Subsequently,
GPT-3 [gpt3](http://arxiv.org/pdf/2112.07522v2), with a massive scale of 175 billion
parameters, was introduced, demonstrating significant breakthroughs
across numerous language benchmarks. This development inspired the
creation of various other large language models, including
Megatron-Turing NLG [smith2022using](http://arxiv.org/pdf/2201.11990v3),
Chinchilla [hoffmann2022training](http://arxiv.org/pdf/2202.03371v1),
PaLM [chowdhery2022palm](http://arxiv.org/pdf/2209.05735v4),
OPT [zhang2022opt](http://arxiv.org/pdf/2405.04515v2),
BLOOM [scao2022bloom](http://arxiv.org/pdf/2106.06683v2), and
LLaMA [llama](http://arxiv.org/pdf/2402.08075v1), among others. Wei *et
al.* [wei2022emergent](https://openreview.net/forum?id=yzkSU5zdwD) further discovered several
*emergent abilities*, which appear exclusively in large models. The
emergence of these abilities underscores the importance of scaling up in
the development of large language models. Moreover, by aligning the
pre-trained large language model GPT-3 with human intent, instructions
and human feedback, InstructGPT [instructGPT](http://arxiv.org/pdf/2302.05206v1) and
ChatGPT [chatgpt](http://arxiv.org/pdf/2307.11380v2) enable conversational interactions
with humans and can answer a wide range of diverse and complex
questions. More recently, several open-sourced models, such as
Alpaca [alpaca](https://github.com/tatsu-lab/stanford_alpaca) and Vicuna [vicuna2023](https://vicuna.lmsys.org),
have been developed based on LLaMA [llama](http://arxiv.org/pdf/2402.08075v1) and also
exhibit similar performance.

**Leveraging Pre-trained LLMs in Vision-Language Tasks.** In recent
years, the trend of using autoregressive language models as decoders in
vision-language tasks has gained significant
traction [visualgpt](http://arxiv.org/pdf/2102.10407v5), [kosmos](http://arxiv.org/pdf/2302.14045v2), [yang2022zero](http://arxiv.org/pdf/2206.08155v2), [tiong2022plug](http://arxiv.org/pdf/2210.08773v3), [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1), [blip2](http://arxiv.org/pdf/2301.12597v3), [blip1](http://arxiv.org/pdf/2311.01038v2), [palm_e](http://arxiv.org/pdf/2302.14030v3).
This approach takes advantage of cross-modal transfer, allowing
knowledge to be shared between language and multimodal domains.
Pioneering studies like VisualGPT [visualgpt](http://arxiv.org/pdf/2102.10407v5) and
Frozen [tsimpoukelli2021multimodal](http://arxiv.org/pdf/2106.13884v2) have demonstrated
the benefits of employing a pre-trained language model as a
vision-language model decoder.
Flamingo [alayrac2022flamingo](http://arxiv.org/pdf/2205.07065v1) was then developed to
align a pre-trained vision encoder and language model using gated
cross-attention, and was trained on billions of image-text pairs,
showcasing impressive in-context few-shot learning capabilities.
Following that, BLIP-2 [blip2](http://arxiv.org/pdf/2301.12597v3) was introduced, employing
a Flan-T5 [flanT5](http://arxiv.org/pdf/2202.03371v1) with a Q-Former to efficiently align
visual features with the language model. Most recently,
PaLM-E [palm_e](http://arxiv.org/pdf/2302.14030v3), featuring 562 billion parameters, has
been developed to integrate real-world continuous sensor modalities into
an LLM, thereby establishing a connection between real-world perceptions
and human languages. GPT-4 [gpt4](http://arxiv.org/pdf/2311.15732v2) has also been recently
released, showcasing more powerful visual understanding and reasoning
abilities after pre-training on a vast collection of aligned image-text
data.

LLMs, such as ChatGPT, have proven to be powerful tools in enhancing the
performance of vision-language tasks by collaborating with other
specialized models. For instance, Visual
ChatGPT [visualChatGPT](http://arxiv.org/pdf/2303.04671v1) and
MM-REACT [yang2023mmreact](http://arxiv.org/pdf/2303.11381v1) showcase how ChatGPT can act
as a coordinator, integrating with diverse visual foundation models and
facilitating their collaboration to tackle more complex challenges.
ChatCaptioner [chatcaptioner](http://arxiv.org/pdf/2303.06594v1) treats ChatGPT as a
questioner, prompting diverse questions for BLIP-2 to answer. Through
multi-round conversations, ChatGPT extracts visual information from
BLIP-2 and effectively summarizes the image content. Video
ChatCaptioner [chen2023video](http://arxiv.org/pdf/2304.04227v3) extends this approach,
applying it to video spatiotemporal understanding.
ViperGPT [vipergpt](http://arxiv.org/pdf/1905.11127v1) demonstrates the potential of
combining an LLM with different vision models to address complex visual
queries programmatically. In contrast, MiniGPT-4 directly aligns visual
information with the language model to accomplish diverse
vision-language tasks without the usage of external vision models.

MiniGPT-4 aims to align visual information from a pretrained vision
encoder with an advanced large language model (LLM). Specifically, we
utilize the Vicuna [vicuna2023](https://vicuna.lmsys.org) as our language decoder,
which is constructed upon LLaMA [llama](http://arxiv.org/pdf/2402.08075v1) and can perform
a wide range of complex linguistic tasks. For visual perception, we
employ the same visual encoder as used in
BLIP-2 [blip2](http://arxiv.org/pdf/2301.12597v3), a ViT
backbone [fang2022eva](http://arxiv.org/pdf/2402.18128v1) coupled with their pre-trained
Q-Former. Both language and vision models are open-sourced. We target to
bridge the gap between the visual encoder and LLM using a linear
projection layer, with an overview of our model displayed in
Fig.<a href="#fig:overview" data-reference-type="ref"
data-reference="fig:overview">[fig:overview]</a>.

To achieve an effective MiniGPT-4, we propose a two-stage training
approach. The initial stage involves pretraining the model on a large
collection of aligned image-text pairs to acquire vision-language
knowledge. In the second stage, we finetune the pretrained model with a
smaller but high-quality image-text dataset with a designed
conversational template to enhance generation reliability and usability.

## First pretraining stage

During the initial pretraining stage, the model is designed to acquire
vision-language knowledge from a large collection of aligned image-text
pairs. We regard the output from the injected projection layer as a soft
prompt for the LLM, prompting it to generate the corresponding
ground-truth texts.

Throughout the entire pretraining process, both the pretrained vision
encoder and the LLM remain frozen, with only the linear projection layer
being pretrained. We use a combined dataset of Conceptual Caption
[changpinyo2021conceptual](http://arxiv.org/pdf/2102.08981v2), [sharma2018conceptual](http://arxiv.org/pdf/2304.13130v1), SBU
[ordonez2011im2text](http://arxiv.org/pdf/2204.00679v1) and LAION [laion](http://arxiv.org/pdf/2111.02114v1)
to train our model. Our model undergoes 20,000 training steps with a
batch size of 256, covering approximately 5 million image-text pairs.
The entire process takes about 10 hours to complete, utilizing 4 A100
(80GB) GPUs.

**Issues of the first pretraining stage** Following the first
pretraining stage, our MiniGPT-4 demonstrates the capacity to possess a
wealth of knowledge and offer reasonable responses to human inquiries.
However, we have observed instances where it produces incoherent
linguistic outputs, such as repetitive words or sentences, fragmented
sentences, or irrelevant content. These issues hinder MiniGPT-4’s
ability to engage in a fluent visual conversation with humans.

We also observed similar challenges encountered in GPT-3. Despite its
pretraining on a extensive language dataset, GPT-3 struggles to generate
language outputs that are accurately aligned with users’ intentions.
Through a process of instruction fine-tuning and reinforcement learning
from human feedback, GPT-3 evolves into GPT-3.5
[instructGPT](http://arxiv.org/pdf/2302.05206v1), [chatgpt](http://arxiv.org/pdf/2307.11380v2) and becomes capable of producing
more human-friendly outputs. This phenomenon bears a resemblance to the
current state of MiniGPT-4 following its initial pretraining stage. As
such, it is not surprising that our model may struggle to generate
fluent and natural human language outputs at this stage.

## Curating a high-quality alignment dataset for vision-language domain.

To achieve greater naturalness in the generated language and enhance the
model’s usability, a second-stage alignment process is essential. While
in the realm of NLP, instruction fine-tuning datasets
[alpaca](https://github.com/tatsu-lab/stanford_alpaca) and conversations [sharegpt](https://github.com/domeccleston/sharegpt)
are easily accessible, no equivalent datasets exist for the
vision-language domain. To address this deficiency, we carefully curated
a detailed image description dataset, specifically tailored for
vision-language alignment purposes. This dataset is subsequently
utilized to fine-tune our MiniGPT-4 during the second-stage alignment
process.

#### Initial aligned image-text generation

In the initial phase, we employ the model derived from the first
pretraining stage to generate comprehensive descriptions of input
images. To enable our model to produce more detailed image descriptions,
we designed a prompt that adheres to the conversational format of the
Vicuna [vicuna2023](https://vicuna.lmsys.org) language model, as shown below. In
this prompt, *\<ImageFeature\>* represents the visual features produced
by the linear projection layer.

*\###Human: \<Img\>\<ImageFeature\>\</Img\>Describe this image in
detail. Give as many details as possible. Say everything you see.
\###Assistant:*

To identify incomplete sentences, we examine whether the generated
sentence exceeds 80 tokens. If it does not, we incorporate an additional
prompt, *\###Human: Continue \###Assistant:* , prompting our MiniGPT-4
to extend the generation process. By concatenating the outputs from both
steps, we can create a more comprehensive image description. This
approach enables us to generate image-text pairs with detailed and
informative image descriptions. We randomly select 5,000 images from the
Conceptual Caption dataset
[changpinyo2021conceptual](http://arxiv.org/pdf/2102.08981v2), [sharma2018conceptual](http://arxiv.org/pdf/2304.13130v1) and use
the pretrained model to generate corresponding language descriptions for
each image.

#### Data post-processing

The above automatically generated image descriptions contain noisy or
incoherent descriptions, such as repetition of words or sentences,
fragmented sentences, or irrelevant content. In order to fix these
issues, we employ ChatGPT to mend the descriptions by utilizing the
following prompt:

*Fix the error in the given paragraph. Remove any repeating sentences,
meaningless characters, not English sentences, and so on. Remove
unnecessary repetition. Rewrite any incomplete sentences. Return
directly the results without explanation. Return directly the input
paragraph if it is already correct without explanation.*

Upon completing the post-processing stage, we manually verify the
correctness of each image description to guarantee its high quality.
Specifically, we first identified several frequently shown errors (*“I’m
sorry I made a mistake...”, or “I apologize for that ...”*) and then
hard-coded rules to automatically filter them out. We also manually
refine the generated captions by eliminating redundant words or
sentences that ChatGPT fails to detect. Finally, only approximately
3,500 out of 5,000 image-text pairs satisfy our requirement, and these
pairs are subsequently utilized for the second-stage alignment process.

## Second-stage finetuning

During the second stage, we finetune our pretrained model with the
curated high-quality image-text pairs. During the finetuning, we use the
predefined prompts in the following template:

*\###Human: \<Img\>\<ImageFeature\>\</Img\>\<Instruction\>###Assistant:*

In this prompt, *\<Instruction\>* represents a randomly sampled
instruction from our predefined instruction set containing variant forms
of instructions such as *“Describe this image in detail”* or *“Could you
describe the contents of this image for me”*. It is important to note
that we do not calculate the regression loss for this specific
text-image prompt.

As a result, MiniGPT-4 is now capable of producing more natural and
reliable language outputs. Furthermore, we observed that this
fine-tuning process is remarkably efficient, only requiring a mere 400
training steps with a batch size of 12, which takes around 7 minutes
with a single A100 GPU.

In the experiment, we aim to showcase the diverse and emergent
capabilities of our MiniGPT-4 model through various qualitative
examples. These abilities include generating detailed image
descriptions, identifying amusing aspects within memes, providing food
recipes from photos, writing poems for images, etc. Additionally, we
present quantitative results on the task of image captioning.

<figure id="fig:ad">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/fig5.png" style="width:90.0%" />
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/fig3.png" style="width:90.0%" />
<figcaption>Advertisement promotion</figcaption>
</figure>

## Uncovering emergent abilities with MiniGPT-4 through qualitative examples

MiniGPT-4 demonstrates many advanced abilities compared to traditional
vision-language models. For example, it can describe images in detail
and interpret the humorous aspects of a given meme. Here, we
qualitatively compared our model to one of the leading vision-language
models, BLIP-2 [blip2](http://arxiv.org/pdf/2301.12597v3), with eight distinct examples,
each highlighting a different ability.

An example in Fig.<a href="#fig:detailed" data-reference-type="ref"
data-reference="fig:detailed">[fig:detailed]</a> demonstrates that
MiniGPT-4 effectively identifies various elements within the image, such
as busy city streets, clock towers, shops, restaurants, motorcycles,
people, streetlights, and clouds. In contrast, BLIP-2 can only cover
city streets, people, and motorcycles in its image caption generation.
Another example presented in
Fig.<a href="#fig:ab_meme" data-reference-type="ref"
data-reference="fig:ab_meme">2</a> shows that MiniGPT-4 successfully
explains why the meme is humorous. It interprets that the lying dog is
feeling the same way as many people do on Monday, which is often
considered to be the most dreaded day of the week. In contrast, BLIP-2
only briefly describes the image content and fails to comprehend the
amusing aspects of the image.

We also showcase MiniGPT-4’s other abilities by demonstrating other
distinctive abilities. These include creating advertising promotions
based on a given image (Fig.<a href="#fig:ad" data-reference-type="ref"
data-reference="fig:ad">1</a>), retrieving factual information from a
movie photograph (Fig.<a href="#fig:movie" data-reference-type="ref"
data-reference="fig:movie">[fig:movie]</a>), generating a food recipe
from a food image (Fig.<a href="#fig:cook" data-reference-type="ref"
data-reference="fig:cook">[fig:cook]</a>), diagnosing plant diseases and
suggesting treatment plans
(Fig.<a href="#fig:plant" data-reference-type="ref"
data-reference="fig:plant">[fig:plant]</a>), creating a website from a
hand-written draft
(Fig.<a href="#fig:ab_website" data-reference-type="ref"
data-reference="fig:ab_website">3</a>), and writing poems inspired by an
image (Fig.<a href="#fig:poem" data-reference-type="ref"
data-reference="fig:poem">[fig:poem]</a>). These abilities are absent in
traditional vision-language models like BLIP-2 (utilizing Flan-T5
XXL [flanT5](http://arxiv.org/pdf/2202.03371v1) as a language model), which use less
powerful language models (LLMs). This contrast indicates that those
advanced vision-language abilities only emerge when the visual features
are properly aligned with an advanced LLM such as Vicuna
[vicuna2023](https://vicuna.lmsys.org).

<figure id="fig:overall_label">
<figure id="fig:ab_meme">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/rebuttal_meme.png" style="width:90.0%" />
<figcaption>Meme explaining</figcaption>
</figure>
<figure id="fig:ab_website">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/rebuttal_web.png" style="width:90.0%" />
<figcaption>Website Creating</figcaption>
</figure>
<figcaption>Model generations from BLIP-2, BLIP-2 finetuned our second
stage data (BLIP-2 FT), MiniGPT-4 finetuned with Local Narrative data in
the second stage (MiniGPT-4 LocNa), MiniGPT-4 model without Q-Former
(MiniGPT-4 No Q-Former), and MiniGPT-4.</figcaption>
</figure>

## Quantitative analysis

<figure id="fig:Limitation">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/ablation.png" style="width:90.0%" />
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/failure.png" style="width:90.0%" />
<figcaption>An example of MiniGPT-4’s limitations. MiniGPT-4
hallucinates unexisting tablecloths and can’t locate the windows
correctly.</figcaption>
</figure>

#### Advanced Abilities

To quantify performance on advanced vision-language tasks, we compiled a
small evaluation dataset comprising 4 tasks: meme interpretation with
the question “Explain why this meme is funny.”, recipe generation with
the question “How should I make something like this?”, advertisement
creation with the prompt “Help me draft a professional advertisement for
this.”, and poem composition with “Can you craft a beautiful poem about
this image?”. In total, we collect 100 diverse images, with 25 images
allocated to each task. We asked human evaluators to determine whether
the model generation satisfies the request. We compared our results with
BLIP-2 [blip2](http://arxiv.org/pdf/2301.12597v3) and present the findings in
Tab.<a href="#tab: quanti_adv" data-reference-type="ref"
data-reference="tab: quanti_adv">[tab: quanti_adv]</a>. In meme
interpretation, poem writing, and advertisement creation, BLIP-2 largely
struggles to fulfill any requests. For recipe generation, BLIP-2
succeeds in 4 out of 25 cases. In contrast, MiniGPT-4 manages to address
the requests in recipes, advertisements, and poem generation in nearly
80% of the instances. Furthermore, MiniGPT-4 correctly comprehends the
challenging humor understanding in memes in 8 out of 25 cases.

#### Image Captioning

We evaluate the performance of MiniGPT-4 on the COCO caption benchmark
and compare it with BLIP-2 [blip2](http://arxiv.org/pdf/2301.12597v3). Our model’s
generated captions typically contain rich visual details. As such,
conventional similarity-based image-caption evaluation metrics struggle
to provide an accurate evaluation of our models. In this regard, we
evaluate the performance by checking if the generated captions cover all
the ground truth captions’ information with the help of ChatGPT and
details can be found in
Appx.<a href="#appx: caption_eval" data-reference-type="ref"
data-reference="appx: caption_eval">[appx: caption_eval]</a>. Results in
Tab.<a href="#human_evaluation" data-reference-type="ref"
data-reference="human_evaluation">[human_evaluation]</a> shows that
MiniGPT-4 outperforms BLIP-2 in generating captions that are more
closely aligned with the ground-truth visual objects and relationships.
With a success rate of 66.2%, MiniGPT-4 is considerably more accurate
than BLIP-2, which achieves only 27.5%. Further evaluation on
traditional VQA tasks can be found in
Appx.<a href="#appx: vqa" data-reference-type="ref"
data-reference="appx: vqa">[appx: vqa]</a>.

## Analysis on the second-stage finetuning

#### Effectiveness of the second-stage finetuning

The utilization of only the model pretrained after the first pretraining
stage may result in failures, such as the occurrence of repetitive words
or sentences, fragmented sentences, or irrelevant content. However,
these issues have been largely mitigated through the second-stage
finetuning process. This can be observed in
Fig.<a href="#fig:secondstage" data-reference-type="ref"
data-reference="fig:secondstage">[fig:secondstage]</a>, where MiniGPT-4
generates incomplete captions before the second-stage finetuning.
However, after the second-stage finetuning, MiniGPT-4 is capable of
generating complete and fluent captions. In this section, we investigate
the importance and effectiveness of the second-stage finetuning
approach.

To quantify the impact of second-stage finetuning, we randomly sampled
100 images from the COCO test set and investigated the model performance
on two tasks: detailed description generation and poem writing. The
prompts used were “*Describe the image in detail.*” and “*Can you write
a beautiful poem about this image?*”. These tasks were performed by both
the models before and after second-stage finetuning. We manually counted
the number of failure generations for the model in each stage. The
results are presented in
Tab.<a href="#exp:stage2ablation" data-reference-type="ref"
data-reference="exp:stage2ablation">[exp:stage2ablation]</a>. Prior to
the second-stage finetuning, approximately 1/3 of the generated outputs
failed to match ground truth captions or poems. In contrast, the model
after second-stage fineuning has less than two failure cases out of the
100 test images for both tasks. These experimental results demonstrate
that second-stage finetuning yields a significant improvement in the
quality of generated outputs. A qualitative example of the model
generation before and after the second-stage finetuning is shown in
Fig.<a href="#fig:secondstage" data-reference-type="ref"
data-reference="fig:secondstage">[fig:secondstage]</a>.

#### Can the original BLIP-2 benefit from the second-stage data?

In this study, we finetune BLIP-2 [blip2](http://arxiv.org/pdf/2301.12597v3) with our
second-stage data in the same way as MiniGPT-4, and check if it can
obtain similar advanced abilities as MiniGPT-4. The finetuned BLIP-2 is
denoted as BLIP-2 FT. Note that MiniGPT-4 uses the same visual module as
BLIP-2; while BLIP-2 uses FlanT5 XXL [flanT5](http://arxiv.org/pdf/2202.03371v1) as the
language model, which is not as strong as the Vicuna
[vicuna2023](https://vicuna.lmsys.org) model used in our MiniGPT-4 model. We rely
on the same prompts to assess the advanced capabilities of our model.
Qualitative results are shown in
Fig.<a href="#fig:overall_label" data-reference-type="ref"
data-reference="fig:overall_label">4</a>,
<a href="#fig:ab_cook" data-reference-type="ref"
data-reference="fig:ab_cook">[fig:ab_cook]</a>, and
<a href="#fig:ab_des" data-reference-type="ref"
data-reference="fig:ab_des">[fig:ab_des]</a>. We discover that BLIP-2 FT
still generates short responses and fails to generalize to advanced
tasks like meme explaining and website coding
(Fig.<a href="#fig:overall_label" data-reference-type="ref"
data-reference="fig:overall_label">4</a>). Our finding suggests that
BLIP-2’s relatively weaker language model FlanT5 XXL benefits less from
such a small dataset, and highlights the effectiveness of a more
advanced LLM in a VLM system.

#### Second stage with Localized Narratives 

The dataset Localized Narratives [pont2020connecting](http://arxiv.org/pdf/2302.11217v2) is
a detailed image description dataset where annotators describe images
while simultaneously localizing the corresponding regions. Here, we test
the performance of our model by replacing our self-collected dataset in
the second-stage with the Localized Narratives dataset. The model is
denoted as MiniGPT-4 LocNa. Qualitative results in
Fig.<a href="#fig:overall_label" data-reference-type="ref"
data-reference="fig:overall_label">4</a>,
<a href="#fig:ab_cook" data-reference-type="ref"
data-reference="fig:ab_cook">[fig:ab_cook]</a>, and
<a href="#fig:ab_des" data-reference-type="ref"
data-reference="fig:ab_des">[fig:ab_des]</a> show that MiniGPT-4 LocNa
can generate long image descriptions
(Fig.<a href="#fig:ab_des" data-reference-type="ref"
data-reference="fig:ab_des">[fig:ab_des]</a>). However, the generated
outputs have lower quality with monotonous expressions. Besides,
MiniGPT-4 LocNa does not generalize as well as the original MiniGPT-4 in
other complex tasks like explaining why the meme is funny
(Fig.<a href="#fig:ab_meme" data-reference-type="ref"
data-reference="fig:ab_meme">2</a>). The performance gap may be due to
the monotonous and repeated image descriptions in Localized Narratives.

## Ablation on the architecture designs

To further demonstrate the effectiveness of using one single linear
layer to align visual features with LLM, we conduct experiments with
different architecture designs, including (a) removing the Q-Former and
directly mapping the VIT’s output to Vicuna’s embedding space (i.e.,
without Q-former), (b) using three linear layers instead of one layer,
and (c) additionally finetuning the Q-Former in the vision module. All
the variants are trained in the same way as the original design. Results
on AOK-VQA [schwenk2022okvqa](http://arxiv.org/pdf/2206.01718v1) and GQA
[hudson2019gqa](http://arxiv.org/pdf/2112.05136v1) datasets in
Tab.<a href="#tab: ablation" data-reference-type="ref"
data-reference="tab: ablation">[tab: ablation]</a> show that the variant
(a) **MiniGPT-4 w/o Q-Former** has a similar performance to the original
design. Qualitative results of this variant in
Fig.<a href="#fig:overall_label" data-reference-type="ref"
data-reference="fig:overall_label">4</a>,
<a href="#fig:ab_cook" data-reference-type="ref"
data-reference="fig:ab_cook">[fig:ab_cook]</a>, and
<a href="#fig:ab_des" data-reference-type="ref"
data-reference="fig:ab_des">[fig:ab_des]</a> also show similar advanced
skills. This reveals that the Q-Former from BLIP-2 doesn’t plays a
critical roles for advanced skills. Besides, both variants (b)
**MiniGPT-4+ 3 Layers** and (c) **MiniGPT-4 + finetuning Q-Former**,
perform slightly worse than the original MiniGPT-4. This indicates a
single projection layer is sufficient to align the vision encoder and
the large language model in our limited training data setting.

#### Hallucination

As MiniGPT-4 is built upon LLMs, it inherits LLM’s limitations like
hallucinating nonexistent knowledge. An example in Fig.
<a href="#fig:Limitation" data-reference-type="ref"
data-reference="fig:Limitation">[fig:Limitation]</a> shows that
MiniGPT-4 incorrectly identifies the presence of white tablecloths in
the image, despite their absence. Here, we use the metric
$\text{CHAIR}_i$ [rohrbach2018object](http://arxiv.org/pdf/1809.02156v2) to gauge the
hallucination rate of the generation, with the two distinct prompts to
control the model generation length: *MiniGPT-4 (long)*: Please describe
this image as detailed as possible. *MiniGPT-4 (short)*: Please describe
the image shortly and precisely, in less than 20 words.

Results in Tab.<a href="#tab:hallu" data-reference-type="ref"
data-reference="tab:hallu">[tab:hallu]</a> show that longer captions
tend to have higher hallucination rates. For example, MiniGPT-4 (long)
generates captions averaging 175 words with a higher hallucination rate,
while MiniGPT-4 (short) averages 28.8 words with a lower rate. BLIP-2,
averaging 6.5 words, hallucinates less but covers fewer objects as seen
in Tab.<a href="#human_evaluation" data-reference-type="ref"
data-reference="human_evaluation">[human_evaluation]</a>. Hallucination
in detailed image descriptions is still an unresolved issue. Using
Reinforcement Learning with AI feadback with hallucination detection
modules may be a potential solution.

#### Spatial Information Understanding

MiniGPT-4’s visual perception remains limited. It may struggle to
differentiate spatial localization. For example, MiniGPT-4 in Fig.
<a href="#fig:Limitation" data-reference-type="ref"
data-reference="fig:Limitation">[fig:Limitation]</a> fails to identify
the location of the windows. This limitation may stem from a lack of
aligned image-text data designed for spatial information understanding.
Training on such datasets like RefCOCO
[kazemzadeh2014referitgame](http://arxiv.org/pdf/1808.08754v1) or Visual Genome
[krishna2017visual](http://arxiv.org/pdf/1602.07332v1) could potentially alleviate this
issue.

How does MiniGPT-4 obtain these advanced abilities? Many of the advanced
vision-language capabilities demonstrated by GPT-4 can be understood as
compositional skills rooted in two foundational skills: image
understanding and language generation. Take the task of image-based poem
writing as an example. Advanced LLMs like ChatGPT and Vicuna can already
craft poems based on users’ instructions. If they acquire the ability to
understand images, compositionally generalizing to the task of
image-based poem writing even without having image-poem pairs in their
training data is possible.

In the first pretraining stage, MiniGPT-4 learns to understand images by
modeling the correlation between images and short image descriptions
from image caption datasets. However, the language style in these image
caption datasets differs from that of modern LLMs’ generation, which
leads to distorted language generation and hinders successful
compositional generalization. Therefore, we introduce a second-stage
finetuning to restore the language generation ability. MiniGPT-4 after
the two-stage training successfully generalizes to many advanced
compositional vision-language abilities like website coding from drafts
or meme interpretation, verifies our assumption. Future research might
delve deeper into the mechanism of compositional generalization and seek
ways to enhance them. We hope our work, as an early exploration of these
vision-based LLM capabilities, will spur further investigations in this
domain.

## More Qualitative Results

<figure id="fig:movie">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/fig4.png" style="width:100.0%" />
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/fig6.png" style="width:100.0%" />
<figcaption>Factual retrieval</figcaption>
</figure>

<figure id="fig:poem">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/website.png" style="width:100.0%" />
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/fig8.png" style="width:100.0%" />
<figcaption>Poem writing</figcaption>
</figure>

<figure id="fig:plant">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/fig1.png" style="width:100.0%" />
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/fig2.png" style="width:100.0%" />
<figcaption>Plant cultivating</figcaption>
</figure>

## Evaluation in traditional VQA benchmarks [appx: vqa]

The aim of this study is to replicate the remarkable multi-modal
capabilities demonstrated in GPT-4, such as generating detailed image
descriptions and creating websites from hand-drawn drafts. To emphasize
the most crucial component of advanced vision-language skills, the
methodology of MiniGPT-4 is intentionally kept minimal. For instance,
the learnable model capacity is limited (only one linear layer), and
MiniGPT-4 is trained with just 5 million pairs, in contrast to BLIP-2
with 129 million image-text pairs. Such a pared-down approach is
anticipated to yield suboptimal results on traditional benchmarks. While
this isn’t our primary goal, we offer a quantitative analysis of the VQA
datasets A-OKVQA (multi-choice) [schwenk2022okvqa](http://arxiv.org/pdf/2206.01718v1) and
GQA [hudson2019gqa](http://arxiv.org/pdf/2112.05136v1). Additionally, to showcase the
potential of MiniGPT-4 with traditional benchmarks, we conduct a
straightforward ablation study. Here, we simply unfreeze the LLM using
LoRA [hu2021lora](http://arxiv.org/pdf/2402.11485v1) and incorporate more training data
from the VQAv2, OKVQA, and A-OKVQA datasets during the second finetuning
stage. Results in Tab. <a href="#tab_supp" data-reference-type="ref"
data-reference="tab_supp">[tab_supp]</a> indicate that the original
MiniGPT-4 lags behind BLIP-2 by a reasonable margin, and merely
augmenting the learning capacity and the training data results in a
substantial performance improvement, which confirms our expectations. We
believe our model’s performance on conventional vision benchmarks can be
enhanced with a carefully designed training strategy (e.g., dataset
sample ratios, learning rate schedule, etc.), more training
data/datasets, and additional learnable parameters. Since enhancing
performance on traditional vision benchmarks isn’t this project’s
objective, we reserve this aspect for future research.

## Details of Caption Evaluation [appx: caption_eval]

We employ ChatGPT to determine whether the baseline models cover all the
objects and visual relations presented in the ground-truth captions. For
the COCO evaluation dataset, we randomly choose one ground-truth caption
and treat it as the reference caption. We apply the following prompt to
perform the evaluation.

*There is one image caption1 ‘{ground-truth caption}’, and there is
another image caption2 ‘{comparison caption}’. Does image caption2 cover
all the objects and visual relations shown in image caption1? Only
answer yes or no without any explanation.*

## More qualitative ablation results

<figure id="fig:ab_des">
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/rebuttal_cook.png" style="width:100.0%" />
<embed src="/papers/text_rich/arXiv-2304.10592v2_md/figs/rebuttal_des.png" style="width:90.0%" />
<figcaption>Ablation Study on Detailed Description</figcaption>
</figure>