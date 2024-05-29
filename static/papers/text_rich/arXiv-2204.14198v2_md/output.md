#### Acknowledgments and Disclosure of Funding.

This research was funded by DeepMind. We would like to thank many
colleagues for useful discussions, suggestions, feedback, and advice,
including: Samuel Albanie, Relja Arandjelović, Kareem Ayoub,
Lorrayne Bennett, Adria Recasens Continente, Tom Eccles,
Nando de Freitas, Sander Dieleman, Conor Durkan, Aleksa Gordić,
Raia Hadsell, Will Hawkins, Lisa Anne Hendricks, Felix Hill,
Jordan Hoffmann, Geoffrey Irving, Drew Jaegle, Koray Kavukcuoglu,
Agustin Dal Lago, Mateusz Malinowski, Soňa Mokrá, Gaby Pearl,
Toby Pohlen, Jack Rae, Laurent Sifre, Francis Song, Maria Tsimpoukelli,
Gregory Wayne, and Boxi Wu.

# Appendix [appendix]

<figure id="fig:results">
<img src="scaling_vlm_neurips" />
<figcaption> <strong>results overview.</strong> <em>Left</em>: Our
largest model, dubbed , outperforms state-of-the-art fine-tuned models
on 6 of the 16 tasks we consider with no fine-tuning. For the 9 tasks
with published few-shot results, sets the new few-shot state of the art.
<em>Note:</em> We omit RareAct, our 16th benchmark, as it is a zero-shot
benchmark with no available fine-tuned results to compare to.
<em>Right</em>: performance improves with model size and number of
shots. </figcaption>
</figure>

# Introduction

One key aspect of intelligence is the ability to quickly learn to
perform a new task given a short
instruction [griffiths2019doing](None), [markman1989categorization](None).
While initial progress has been made towards a similar capability in
computer vision, the most widely used paradigm still consists of first
pretraining on a large amount of supervised data, before fine-tuning the
model on the task of
interest [lu2019vilbert](None), [wang2021ufo](None), [zellers2022merlot](None).
However, successful fine-tuning often requires many thousands of
annotated data points. In addition, it often requires careful per-task
hyperparameter tuning and is also resource intensive. Recently,
multimodal vision-language models trained with a contrastive
objective [align](None), [clip](None) have enabled zero-shot adaptation
to novel tasks, without the need for fine-tuning. However, because these
models simply provide a similarity score between a text and an image,
they can only address limited use cases such as classification, where a
finite set of outcomes is provided beforehand. They crucially lack the
ability to generate language, which makes them less suitable to more
open-ended tasks such as captioning or visual question-answering. Others
have explored visually-conditioned language
generation [wang2021simvlm](None), [tsimpoukelli2021multimodal](None), [cho2021unifying](None), [wang2022unifying](None), [xu2021vlm](None)
but have not yet shown good performance in low-data regimes.

We introduce , a Visual Language Model (VLM) that sets a new state of
the art in few-shot learning on a wide range of open-ended vision and
language tasks, simply by being prompted with a few input/output
examples, as illustrated in
Figure <a href="#fig:teaser" data-reference-type="ref"
data-reference="fig:teaser">[fig:teaser]</a>. Of the 16 tasks we
consider, also surpasses the fine-tuned state of the art on 6 tasks,
despite using orders of magnitude less task-specific training data (see
Figure <a href="#fig:results" data-reference-type="ref"
data-reference="fig:results">1</a>). To achieve this, Flamingo takes
inspiration from recent work on large language models (LMs) which are
good few-shot
learners [gpt3](None), [gopher](None), [chinchilla](None), [chowdhery2022palm](None). A
single large LM can achieve strong performance on many tasks using only
its text interface: a few examples of a task are provided to the model
as a prompt, along with a query input, and the model generates a
continuation to produce a predicted output for that query. We show that
the same can be done for image and video understanding tasks such as
classification, captioning, or question-answering: these can be cast as
text prediction problems with visual input conditioning. The difference
from a LM is that the model must be able to ingest a multimodal prompt
containing images and/or videos interleaved with text. have this
capability—they are visually-conditioned autoregressive text generation
models able to ingest a sequence of text tokens interleaved with images
and/or videos, and produce text as output. leverage two complementary
pre-trained and frozen models: a vision model which can “perceive”
visual scenes and a large LM which performs a basic form of reasoning.
Novel architecture components are added in between these models to
connect them in a way that preserves the knowledge they have accumulated
during computationally intensive pre-training. are also able to ingest
high-resolution images or videos thanks to a
Perceiver-based [jaegle2021perceiver](None) architecture that
can produce a small fixed number of visual tokens per image/video, given
a large and variable number of visual input features.

A crucial aspect for the performance of large LMs is that they are
trained on a large amount of text data. This training provides
general-purpose generation capabilities that allows these LMs to perform
well when prompted with task examples. Similarly, we demonstrate that
the way we train the models is crucial for their final performance. They
are trained on a carefully chosen

mixture of complementary large-scale multimodal data coming only from
the web, *without using any data annotated for machine learning
purposes*. After this training, a model can be directly adapted to
vision tasks via simple few-shot learning without any task-specific
tuning.

**Contributions.** In summary, our contributions are the following:
**(i)** We introduce the family of VLMs which can perform various
multimodal tasks (such as captioning, visual dialogue, or visual
question-answering) from only a few input/output examples. Thanks to
architectural innovations, the models can efficiently accept arbitrarily
interleaved visual data and text as input and generate text in an
open-ended manner. **(ii)** We quantitatively evaluate how  models can
be adapted to various tasks via few-shot learning. We notably reserve a
large set of held-out benchmarks which have not been used for validation
of any design decisions or hyperparameters of the approach. We use these
to estimate unbiased few-shot performance. **(iii)** sets a new state of
the art in few-shot learning on a wide array of 16 multimodal language
and image/video understanding tasks. On 6 of these 16 tasks, also
outperforms the fine-tuned state of the art despite using only 32
task-specific examples, around 1000 times less task-specific training
data than the current state of the art. With a larger annotation
budget,  can also be effectively fine-tuned to set a new state of the
art on five additional challenging benchmarks: VQAv2, VATEX, VizWiz,
MSRVTTQA, and HatefulMemes.

# Approach [sec:approach]

<figure id="fig:overview">
<embed src="/papers/text_rich/arXiv-2204.14198v2_md/figures/fig2_overview_interleaved_v2.png" />
<figcaption> <strong>architecture overview.</strong> Flamingo is a
family of visual language models (VLMs) that take as input visual data
interleaved with text and produce free-form text as output.
</figcaption>
</figure>

This section describes Flamingo: a visual language model that accepts
text interleaved with images/videos as input and outputs free-form text.
The key architectural components shown in
Figure <a href="#fig:overview" data-reference-type="ref"
data-reference="fig:overview">2</a> are chosen to leverage pretrained
vision and language models and bridge them effectively. First, the
Perceiver Resampler
(Section <a href="#sec:transformer_resampler" data-reference-type="ref"
data-reference="sec:transformer_resampler">2.1</a>) receives
spatio-temporal features from the Vision Encoder (obtained from either
an image or a video) and outputs a fixed number of visual tokens.
Second, these visual tokens are used to condition the frozen LM using
freshly initialised cross-attention layers
(Section <a href="#sec:xattn_dense" data-reference-type="ref"
data-reference="sec:xattn_dense">2.2</a>) that are interleaved between
the pretrained LM layers. These new layers offer an expressive way for
the LM to incorporate visual information for the next-token prediction
task. Flamingo models the likelihood of text $y$ conditioned on
interleaved images and videos $x$ as follows: $$\begin{aligned}
    p(y | x) = \prod_{\ell=1}^L p(y_\ell | y_{< \ell}, x_{\leq \ell}),
    \label{eq:modeling}
\end{aligned}$$ where $y_{\ell}$ is the $\ell$-th language token of the
input text, $y_{<\ell}$ is the set of preceding tokens, $x_{\leq \ell}$
is the set of images/videos preceding token $y_{\ell}$ in the
interleaved sequence and $p$ is parametrized by a model. The ability to
handle interleaved text and visual sequences
(Section <a href="#sec:multi_im_att" data-reference-type="ref"
data-reference="sec:multi_im_att">2.3</a>) makes it natural to use
models for in-context few-shot learning, analogously to GPT-3 with
few-shot text prompting. The model is trained on a diverse mixture of
datasets as described in
Section <a href="#sec:datasets" data-reference-type="ref"
data-reference="sec:datasets">2.4</a>.

## Visual processing and the Perceiver Resampler [sec:transformer_resampler]

**Vision Encoder: from pixels to features.** Our vision encoder is a
pretrained and frozen Normalizer-Free ResNet (NFNet)
[nfnets](None) – we use the F6 model. We pretrain the vision
encoder using a contrastive objective on our datasets of image and text
pairs, using the two-term contrastive loss from [clip](None).
We use the output of the final stage, a 2D spatial grid of features that
is flattened to a 1D sequence. For video inputs, frames are sampled at 1
FPS and encoded independently to obtain a 3D spatio-temporal grid of
features to which learned temporal embeddings are added. Features are
then flattened to 1D before being fed to the Perceiver Resampler. More
details on the contrastive model training and performance are given in
Appendix <a href="#app:contrastive_details" data-reference-type="ref"
data-reference="app:contrastive_details">[app:contrastive_details]</a><a href="#app:contrastive_details" data-reference-type="ref"
data-reference="app:contrastive_details">[app:contrastive_details]</a>
and
Appendix <a href="#app:contrastive_ablation" data-reference-type="ref"
data-reference="app:contrastive_ablation">[app:contrastive_ablation]</a><a href="#app:contrastive_ablation" data-reference-type="ref"
data-reference="app:contrastive_ablation">[app:contrastive_ablation]</a>,
respectively.

**Perceiver Resampler: from varying-size large feature maps to few
visual tokens.** This module connects the vision encoder to the frozen
language model as shown in
Figure <a href="#fig:overview" data-reference-type="ref"
data-reference="fig:overview">2</a>. It takes as input a variable number
of image or video features from the vision encoder and produces a fixed
number of visual outputs (64), reducing the computational complexity of
the vision-text cross-attention. Similar to
Perceiver [jaegle2021perceiver](None) and
DETR [carion2020end](None), we learn a predefined number of
latent input queries which are fed to a Transformer and cross-attend to
the visual features. We show in our ablation studies
(Section <a href="#sec:ablations" data-reference-type="ref"
data-reference="sec:ablations">3.3</a>) that using such a
vision-language resampler module outperforms a plain Transformer and an
MLP. We provide an illustration, more architectural details, and
pseudo-code in
Appendix <a href="#app:transformer_resampler" data-reference-type="ref"
data-reference="app:transformer_resampler">[app:transformer_resampler]</a><a href="#app:transformer_resampler" data-reference-type="ref"
data-reference="app:transformer_resampler">[app:transformer_resampler]</a>.

## Conditioning frozen language models on visual representations [sec:xattn_dense]

Text generation is performed by a Transformer decoder, conditioned on
the visual representations produced by the Perceiver Resampler. We
interleave pretrained and frozen text-only LM blocks with blocks trained
from scratch that cross-attend to the visual output from the Perceiver
Resampler.

<figure id="fig:xattn_dense">
<embed src="/papers/text_rich/arXiv-2204.14198v2_md/figures/fig4_xattn_dense.png" />
<figcaption> <strong><span class="smallcaps">gated xattn-dense</span>
layers.</strong> To condition the LM on visual inputs, we insert new
cross-attention layers between existing pretrained and frozen LM layers.
The keys and values in these layers are obtained from the vision
features while the queries are derived from the language inputs. They
are followed by dense feed-forward layers. These layers are
<em>gated</em> so that the LM is kept intact at initialization for
improved stability and performance.</figcaption>
</figure>

**Interleaving new <span class="smallcaps">gated xattn-dense</span>
layers within a frozen pretrained LM.** We freeze the pretrained LM
blocks, and insert *gated cross-attention dense* blocks
(Figure <a href="#fig:xattn_dense" data-reference-type="ref"
data-reference="fig:xattn_dense">3</a>) between the original layers,
trained from scratch. To ensure that at initialization, the conditioned
model yields the same results as the original language model, we use a
$\tanh$-gating mechanism [hochreiter1997long](http://arxiv.org/pdf/2103.15232v1). This
multiplies the output of a newly added layer by $\tanh(\alpha)$ before
adding it to the input representation from the residual connection,
where $\alpha$ is a layer-specific learnable scalar initialized to
$0$ [bachlechner2021rezero](None). Thus, at initialization, the
model output matches that of the pretrained LM, improving training
stability and final performance. In our ablation studies
(Section <a href="#sec:ablations" data-reference-type="ref"
data-reference="sec:ablations">3.3</a>), we compare the proposed
<span class="smallcaps">gated xattn-dense</span> layers against recent
alternatives [desai2021virtex](None), [luo2022vc](None) and explore the
effect of how frequently these additional layers are inserted to trade
off between efficiency and expressivity. See
Appendix <a href="#app:xattn_dense" data-reference-type="ref"
data-reference="app:xattn_dense">[app:xattn_dense]</a><a href="#app:xattn_dense" data-reference-type="ref"
data-reference="app:xattn_dense">[app:xattn_dense]</a> for more details.

**Varying model sizes.** We perform experiments across three models
sizes, building on the 1.4B, 7B, and 70B parameter Chinchilla
models [chinchilla](None); calling them respectively , and .
For brevity, we refer to the last as throughout the paper. While
increasing the parameter count of the frozen LM and the trainable
vision-text <span class="smallcaps">gated xattn-dense</span> modules, we
maintain a fixed-size frozen vision encoder and trainable Perceiver
Resampler across the different models (small relative to the full model
size). See
Appendix <a href="#sec:models_details" data-reference-type="ref"
data-reference="sec:models_details">[sec:models_details]</a><a href="#sec:models_details" data-reference-type="ref"
data-reference="sec:models_details">[sec:models_details]</a> for further
details.

## Multi-visual input support: per-image/video attention masking [sec:multi_im_att]

The image-causal modelling introduced in
Equation <a href="#eq:modeling" data-reference-type="eqref"
data-reference="eq:modeling">[eq:modeling]</a> is obtained by masking
the full text-to-image cross-attention matrix, limiting which visual
tokens the model sees at each text token. At a given text token, the
model attends to the visual tokens of the image that appeared just
before it in the interleaved sequence, rather than to all previous
images (formalized and illustrated in
Appendix <a href="#app:multi-visual-details" data-reference-type="ref"
data-reference="app:multi-visual-details">[app:multi-visual-details]</a><a href="#app:multi-visual-details" data-reference-type="ref"
data-reference="app:multi-visual-details">[app:multi-visual-details]</a>).
Though the model only *directly* attends to a single image at a time,
the dependency on all previous images remains via self-attention in the
LM. This single-image cross-attention scheme importantly allows the
model to seamlessly generalise to any number of visual inputs,
regardless of how many are used during training. In particular, we use
only up to 5 images per sequence when training on our interleaved
datasets, yet our model is able to benefit from sequences of up to 32
pairs (or “shots”) of images/videos and corresponding texts during
evaluation. We show in
Section <a href="#sec:ablations" data-reference-type="ref"
data-reference="sec:ablations">3.3</a> that this scheme is more
effective than allowing the model to cross-attend to all previous images
directly.

## Training on a mixture of vision and language datasets [sec:datasets]

<span id="sec:training" label="sec:training"></span>

We train the models on a mixture of three kinds of datasets, all scraped
from the web: an interleaved image and text dataset derived from
webpages, image-text pairs, and video-text pairs.

**M3W: Interleaved image and text dataset.**
<span id="sec:interleaved_datasets"
label="sec:interleaved_datasets"></span> The few-shot capabilities of
Flamingo models rely on training on interleaved text and image data. For
this purpose, we collect the *MultiModal MassiveWeb* () dataset. We
extract both text and images from the HTML of approximately 43 million
webpages, determining the positions of images relative to the text based
on the relative positions of the text and image elements in the Document
Object Model (DOM). An example is then constructed by inserting
`<image>` tags in plain text at the locations of the images on the page,
and inserting a special `<EOC>` (*end of chunk*) token (added to the
vocabulary and learnt) prior to any image and at the end of the
document. From each document, we sample a random subsequence of $L=256$
tokens and take up to the first $N=5$ images included in the sampled
sequence. Further images are discarded in order to save compute. More
details are provided in
Appendix <a href="#app:datasets" data-reference-type="ref"
data-reference="app:datasets">[app:datasets]</a><a href="#app:datasets" data-reference-type="ref"
data-reference="app:datasets">[app:datasets]</a>.

**Pairs of image/video and text.** For our image and text pairs we first
leverage the ALIGN [align](None) dataset, composed of 1.8
billion images paired with alt-text. To complement this dataset, we
collect our own dataset of image and text pairs targeting better quality
and longer descriptions: LTIP (Long Text & Image Pairs) which consists
of 312 million image and text pairs. We also collect a similar dataset
but with videos instead of still images: VTP (Video & Text Pairs)
consists of 27 million short videos (approximately 22 seconds on
average) paired with sentence descriptions. We align the syntax of
paired datasets with the syntax of M3W by prepending `<image>` and
appending `<EOC>` to each training caption (see
Appendix <a href="#app:vtp_and_itp" data-reference-type="ref"
data-reference="app:vtp_and_itp">[app:vtp_and_itp]</a><a href="#app:vtp_and_itp" data-reference-type="ref"
data-reference="app:vtp_and_itp">[app:vtp_and_itp]</a> for details).

**Multi-objective training and optimisation strategy.** We train our
models by minimizing a weighted sum of per-dataset expected negative
log-likelihoods of text, given the visual inputs:
$$\sum_{m=1}^{M} \lambda_m \cdot \mathbb{E}_{(x, y)\sim \mathcal{D}_m} \left[ -\sum_{\ell=1}^L \log p(y_\ell | y_{< \ell}, x_{\leq \ell})\right],$$
where $\mathcal{D}_m$ and $\lambda_m$ are the $m$-th dataset and its
weighting, respectively. Tuning the per-dataset weights $\lambda_m$ is
key to performance. We accumulate gradients over all datasets, which we
found outperforms a “round-robin”
approach [cho2021unifying](None). We provide further training
details and ablations in
Appendix <a href="#app:large_scale_training" data-reference-type="ref"
data-reference="app:large_scale_training">[app:large_scale_training]</a><a href="#app:large_scale_training" data-reference-type="ref"
data-reference="app:large_scale_training">[app:large_scale_training]</a>.

## Task adaptation with few-shot in-context learning [sec:adapt-vlm]

Once Flamingo is trained, we use it to tackle a visual task by
conditioning it on a multimodal interleaved prompt. We evaluate the
ability of our models to rapidly adapt to new tasks using **in-context
learning**, analogously to GPT-3 [gpt3](None), by interleaving
support example pairs in the form of $(image, text)$ or $(video, text)$,
followed by the query visual input, to build a prompt (details in
Appendix <a href="#app:in_context_eval_details" data-reference-type="ref"
data-reference="app:in_context_eval_details">[app:in_context_eval_details]</a><a href="#app:in_context_eval_details" data-reference-type="ref"
data-reference="app:in_context_eval_details">[app:in_context_eval_details]</a>).
We perform **open-ended** evaluations using beam search for decoding,
and **close-ended** evaluations using our model’s log-likelihood to
score each possible answer. We explore **zero-shot generalization** by
prompting the model with two text-only examples from the task, with no
corresponding images. Evaluation hyperparameters and additional details
are given in
Appendix <a href="#app:fewshot-eval-hyper" data-reference-type="ref"
data-reference="app:fewshot-eval-hyper">[app:fewshot-eval-hyper]</a><a href="#app:fewshot-eval-hyper" data-reference-type="ref"
data-reference="app:fewshot-eval-hyper">[app:fewshot-eval-hyper]</a>.

# Experiments [sec:experiments]

Our goal is to develop models that can rapidly adapt to diverse and
challenging tasks. For this, we consider a wide array of 16 popular
multimodal image/video and language benchmarks. In order to validate
model design decisions during the course of the project, 5 of these
benchmarks were used as part of our development
(<span class="smallcaps">dev</span>) set: COCO, OKVQA, VQAv2, MSVDQA and
VATEX. Performance estimates on the <span class="smallcaps">dev</span>
benchmarks may be biased, as a result of model selection. We note that
this is also the case for prior work which makes use of similar
benchmarks to validate and ablate design decisions. To account for this,
we report performance on an additional set of 11 benchmarks, spanning
captioning, video question-answering, as well as some less commonly
explored capabilities such as visual dialogue and multi-choice
question-answering tasks. The evaluation benchmarks are described in
Appendix <a href="#sec:eval_benchmarks" data-reference-type="ref"
data-reference="sec:eval_benchmarks">[sec:eval_benchmarks]</a><a href="#sec:eval_benchmarks" data-reference-type="ref"
data-reference="sec:eval_benchmarks">[sec:eval_benchmarks]</a>. We keep
all evaluation hyperparameters fixed across all benchmarks. Depending on
the task, we use four few-shot prompt templates we describe in more
detail in
Appendix <a href="#app:fewshot-eval-hyper" data-reference-type="ref"
data-reference="app:fewshot-eval-hyper">[app:fewshot-eval-hyper]</a><a href="#app:fewshot-eval-hyper" data-reference-type="ref"
data-reference="app:fewshot-eval-hyper">[app:fewshot-eval-hyper]</a>. We
emphasize that *we do not validate any design decisions on these 11
benchmarks* and use them solely to estimate unbiased few-shot learning
performance of our models.

Concretely, estimating few-shot learning performance of a model involves
prompting it with a set of *support* samples and evaluating it on a set
of *query* samples. For the <span class="smallcaps">dev</span>
benchmarks that are used both to validate design decisions and
hyperparameters, as well as to report final performance, we therefore
use four subsets: *validation support*, *validation query*, *test
support* and *test query*. For other benchmarks, we need only the latter
two. We report in
Appendix <a href="#sec:eval_benchmarks" data-reference-type="ref"
data-reference="sec:eval_benchmarks">[sec:eval_benchmarks]</a><a href="#sec:eval_benchmarks" data-reference-type="ref"
data-reference="sec:eval_benchmarks">[sec:eval_benchmarks]</a> how we
form these subsets.

We report the results of the  models on few-shot learning in
Section <a href="#sec:fewshot_openended" data-reference-type="ref"
data-reference="sec:fewshot_openended">3.1</a>.
Section <a href="#sec:ft_results" data-reference-type="ref"
data-reference="sec:ft_results">3.2</a> gives  fine-tuned results. An
ablation study is given in
Section <a href="#sec:ablations" data-reference-type="ref"
data-reference="sec:ablations">3.3</a>.
Appendix <a href="#app:more_performance" data-reference-type="ref"
data-reference="app:more_performance">[app:more_performance]</a><a href="#app:more_performance" data-reference-type="ref"
data-reference="app:more_performance">[app:more_performance]</a>
provides more results including ’s performance on the ImageNet and
Kinetics700 classification tasks, and on our contrastive model’s
performance. Appendix <a href="#app:qual_res" data-reference-type="ref"
data-reference="app:qual_res">[app:qual_res]</a><a href="#app:qual_res" data-reference-type="ref"
data-reference="app:qual_res">[app:qual_res]</a> includes additional
qualitative results.

## Few-shot learning on vision-language tasks [sec:fewshot_openended]

**Few-shot results.** Results are given in
Table <a href="#tab:fewshot_all_tasks" data-reference-type="ref"
data-reference="tab:fewshot_all_tasks">[tab:fewshot_all_tasks]</a>.
outperforms by a large margin *all* previous zero-shot or few-shot
methods on the 16 benchmarks considered. This is achieved with as few as
four examples per task, demonstrating practical and efficient adaptation
of vision models to new tasks. More importantly, is often competitive
with state-of-the-art methods additionally fine-tuned on up to hundreds
of thousands of annotated examples. On six tasks, even outperforms the
fine-tuned SotA despite using a *single* set of model weights and only
32 task-specific examples. Finally, despite having only used the
<span class="smallcaps">dev</span> benchmarks for design decisions, our
results generalize well to the other benchmarks, confirming the
generality of our approach.

**Scaling with respect to parameters and shots.** As shown in
Figure <a href="#fig:results" data-reference-type="ref"
data-reference="fig:results">1</a>, the larger the model, the better the
few-shot performance, similar to GPT-3 [gpt3](None). The
performance also improves with the number of shots. We further find that
the largest model better exploits larger numbers of shots.
Interestingly, even though our models were trained with sequences
limited to only 5 images on , they are still able to benefit from up to
32 images or videos during inference. This demonstrates the flexibility
of the architecture for processing a variable number of videos or
images.

## Fine-tuning as a pretrained vision-language model [sec:ft_results]

While not the main focus of our work, we verify that when given more
data, models can be adapted to a task by fine-tuning their weights. In
Table <a href="#tab:ft-sota-table-compressed" data-reference-type="ref"
data-reference="tab:ft-sota-table-compressed">[tab:ft-sota-table-compressed]</a>,
we explore fine-tuning our largest model, , for a given task with no
limit on the annotation budget. In short, we do so by fine-tuning the
model on a short schedule with a small learning rate by additionally
unfreezing the vision backbone to accommodate a higher input resolution
(details in Appendix <a href="#app:finetuning" data-reference-type="ref"
data-reference="app:finetuning">[app:finetuning]</a><a href="#app:finetuning" data-reference-type="ref"
data-reference="app:finetuning">[app:finetuning]</a>). We find that we
can improve results over our previously presented in-context few-shot
learning results, setting a new state of the art on five additional
tasks: VQAv2, VATEX, VizWiz, MSRVTTQA, and HatefulMemes.

## Ablation studies [sec:ablations]

In
Table <a href="#tab:ablation-table-no-classif" data-reference-type="ref"
data-reference="tab:ablation-table-no-classif">[tab:ablation-table-no-classif]</a>,
we report our ablation results using  on the *validation* subsets of the
five <span class="smallcaps">dev</span> benchmarks with 4 shots. Note
that we use smaller batch sizes and a shorter training schedule compared
to the final models. The **Overall score** is obtained by dividing each
benchmark score by its state-of-the-art (SotA) performance from
Table <a href="#tab:fewshot_all_tasks" data-reference-type="ref"
data-reference="tab:fewshot_all_tasks">[tab:fewshot_all_tasks]</a> and
averaging the results. More details and results are given in
Appendix <a href="#app:all_ablation_studies" data-reference-type="ref"
data-reference="app:all_ablation_studies">[app:all_ablation_studies]</a><a href="#app:all_ablation_studies" data-reference-type="ref"
data-reference="app:all_ablation_studies">[app:all_ablation_studies]</a>
and
Table <a href="#tab:ablation-table-appendix" data-reference-type="ref"
data-reference="tab:ablation-table-appendix">[tab:ablation-table-appendix]</a>.

**Importance of the training data mixture.** As shown in row **(i)**,
getting the right training data plays a crucial role. In fact, removing
the interleaved image-text dataset leads to a *decrease of more than
$17\%$* in performance while removing the conventional paired image-text
pairs also decreases performance (by $9.8\%$), demonstrating the need
for different types of datasets. Moreover, removing our paired
video-text dataset negatively affects performance on all video tasks. We
ablate replacing our image-text pairs (ITP) by the publicly available
LAION-400M dataset [schuhmann2021laion](None), which leads to a
slight degradation in performance. We show in row **(ii)** the
importance of our gradient accumulation strategy compared to using
round-robin updates [cho2021unifying](None).

**Visual conditioning of the frozen LM.** We ablate the use of the
0-initialized tanh gating when merging the cross-attention output to the
frozen LM output in row **(iii)**. Without it, we see a drop of $4.2\%$
in our overall score. Moreover, we have noticed that disabling the
0-initialized tanh gating leads to training instabilities. Next, we
ablate different conditioning architectures in row **(iv)**.
<span class="smallcaps">vanilla xattn</span>, refers to the vanilla
cross-attention from the original Transformer
decoder [vaswani2017attention](None). In the
<span class="smallcaps">grafting</span> approach
from [luo2022vc](None), the frozen LM is used as is with no
additional layers inserted, and a stack of interleaved self-attention
and cross-attention layers that take the frozen LM output are learnt
from scratch. Overall, we show that our <span class="smallcaps">gated
xattn-dense</span> conditioning approach works best.

**Compute/Memory vs. performance trade-offs.** In row **(v)**, we ablate
the frequency at which we add new <span class="smallcaps">gated
xattn-dense</span> blocks. Although adding them at every layer is
better, it significantly increases the number of trainable parameters
and time complexity of the model. Notably, inserting them every fourth
block accelerates training by $66\%$ while only decreasing the overall
score by $1.9\%$. In light of this trade-off, we maximize the number of
added layers under hardware constraints and add a
<span class="smallcaps">gated xattn-dense</span> every fourth layer for
and every seventh for . We further compare in row **(vi)** the Perceiver
Resampler to a MLP and a vanilla Transformer given a parameter budget.
Both underperform the Perceiver Resampler while also being slower.

**Vision encoder.** In row **(vii)**, we compare our NFNet-F6 vision
encoder pretrained with contrastive learning (details in
Appendix <a href="#app:contrastive_details" data-reference-type="ref"
data-reference="app:contrastive_details">[app:contrastive_details]</a><a href="#app:contrastive_details" data-reference-type="ref"
data-reference="app:contrastive_details">[app:contrastive_details]</a>)
to the publicly available CLIP ViT-L/14 [clip](None) model
trained at 224 resolution. Our NFNet-F6 has a $+5.8\%$ advantage over
the CLIP ViT-L/14 and $+8.0\%$ over a smaller NFNet-F0 encoder, which
highlights the importance of using a strong vision backbone.

**Freezing LM components prevents catastrophic forgetting.** We verify
the importance of freezing the LM layers at training in row **(viii)**.
If trained from scratch, we observe a large performance decrease of
$-12.9\%$. Interestingly, fine-tuning our pretrained LM also leads to a
drop in performance of $-8.0\%$. This indicates an instance of
“catastrophic forgetting” [mccloskey1989catastrophic](None),
in which the model progressively forgets its pretraining while training
on a new objective. In our setting, freezing the language model is a
better alternative to training with the pre-training dataset
(MassiveText) in the mixture.

# Related work

**Language modelling and few-shot adaptation.** Language modelling has
recently made substantial progress following the introduction of
Transformers [vaswani2017attention](None). The paradigm of
first pretraining on a vast amount of data followed by an adaptation on
a downstream task has become
standard [mikolov2010recurrent](None), [graves2013generating](None), [jozefowicz2016exploring](None), [howard2018universal](None), [bert](None), [t5](None), [sutskever2011generating](None), [gpt3](None).
In this work, we build on the 70B Chinchilla language
model [chinchilla](None) as the base LM for . Numerous works
have explored techniques to adapt language models to novel tasks using a
few examples. These include adding small adapter
modules [houlsby2019parameter](None), fine-tuning a small part
of the LM [zaken_bitfit_2022](None), showing in-context
examples in the prompt [gpt3](None), or optimizing the
prompt [li2021prefix](None), [lester2021power](None) through gradient
descent. In this paper, we take inspiration from the
in-context [gpt3](None) few-shot learning technique instead of
more involved few-shot learning approaches based on metric
learning [doersch2020crosstransformers](None), [vinyals2016matching](None), [snell2017prototypical](None), [tian2020rethinking](None)
or
meta-learning [finn2017model](None), [bertinetto2018meta](None), [zintgraf2019fast](None), [requeima2019fast](None), [gordon2018meta](None), [bertinetto2016learning](None).

**When language meets vision.** These LM breakthroughs have been
influential for vision-language modelling. In particular,
BERT [bert](None) inspired a large body of vision-language
work [lu2019vilbert](None), [su2019vl](None), [chen2020uniter](None), [hendricks2021decoupling](None), [wang2021vlmo](None), [li2020oscar](None), [tan2019lxmert](None), [zhu2020actbert](None), [wang2021ufo](None), [li2020hero](None), [gan2020large](None), [fu2021violet](None), [zellers2021merlot](None), [zellers2022merlot](None), [singh2021flava](None), [sun2019videobert](None).
We differ from these approaches as do not require fine-tuning on new
tasks. Another family of vision-language models is based on contrastive
learning [alayrac2020self](None), [clip](None), [align](None), [zhai2021lit](None), [pham2021combined](None), [miech2020end](None), [bain2021frozen](None), [yuan2021florence](None), [li2021align](None), [yao2021filip](None), [jain2021mural](None).
differs from contrastive models as it can generate text, although we
build and rely upon them for our vision encoder. Similar to our work are
VLMs able to generate text in an autoregressive
manner [vinyals2015show](None), [donahue2015long](None), [luo2020univl](None), [hu2021scaling](None), [dai2022](None).
Concurrent
works [wang2021simvlm](None), [cho2021unifying](None), [wang2022unifying](None), [zhu2021uni](None), [li2022blip](None)
also propose to formulate numerous vision tasks as text generation
problems. Building on top of powerful pretrained language models has
been explored in several recent works. One recent line of
work [tsimpoukelli2021multimodal](None), [eichenberg2021magma](None), [mokady2021clipcap](None), [luo2022vc](None), [yang2021empirical](None), [zeng2022socraticmodels](None)
proposes to freeze the pretrained LM weights to prevent catastrophic
forgetting [mccloskey1989catastrophic](None). We follow this
idea by freezing the Chinchilla LM layers [chinchilla](None)
and adding learnable layers within the frozen LM. We differ from prior
work by introducing the first LM that can ingest arbitrarily interleaved
images, videos, and text.

**Web-scale vision and language training datasets.** Manually annotated
vision and language datasets are costly to obtain and thus relatively
small (10k-100k) in
scale [young2014image](None), [chen2015microsoft](None), [antol2015vqa](None), [marino2019ok](None), [wang2019vatex](None), [xiao2021next](None).
To alleviate this lack of data, numerous
works [align](None), [sharma2018conceptual](None), [changpinyo2021conceptual](None), [thomee2016yfcc100m](None)
automatically scrape readily available paired vision-text data. In
addition to such paired data, we show the importance of also training on
entire multimodal webpages containing interleaved images and text as a
single sequence. Concurrent work CM3 [aghajanyan2022cm3](None)
proposes to generate HTML markup from pages, while we simplify the text
prediction task by only generating plain text. We emphasize few-shot
learning and vision tasks while CM3 [aghajanyan2022cm3](None)
primarily evaluates on language-only benchmarks in a zero-shot or
fine-tuned setup.

# Discussion [sec:discussion]

**Limitations.** First, our models build on pretrained LMs, and as a
side effect, directly inherit their weaknesses. For example, LM priors
are generally helpful, but may play a role in occasional hallucinations
and ungrounded guesses. Furthermore, LMs generalise poorly to sequences
longer than the training ones. They also suffer from poor sample
efficiency during training. Addressing these issues can accelerate
progress in the field and enhance the abilities of VLMs like Flamingo.

Second, the classification performance of lags behind that of
state-of-the-art contrastive
models [clip](None), [pham2021combined](None). These models directly
optimize for text-image retrieval, of which classification is a special
case. In contrast, our models handle a wider range of tasks, such as
open-ended ones. A unified approach to achieve the best of both worlds
is an important research direction.

Third, in-context learning has significant advantages over
gradient-based few-shot learning methods, but also suffers from
drawbacks depending on the characteristics of the application at hand.
We demonstrate the effectiveness of in-context learning when access is
limited to only a few dozen examples. In-context learning also enables
simple deployment, requiring only inference, generally with no
hyperparameter tuning needed. However, in-context learning is known to
be highly sensitive to various aspects of the
demonstrations [zhao2021calibrate](None), [truefewshot](None), and its
inference compute cost and absolute performance scale poorly with the
number of shots beyond this low-data regime. There may be opportunities
to combine few-shot learning methods to leverage their complementary
benefits. We discuss the limitations of our work in more depth in
Appendix <a href="#sec:limitations" data-reference-type="ref"
data-reference="sec:limitations">[sec:limitations]</a><a href="#sec:limitations" data-reference-type="ref"
data-reference="sec:limitations">[sec:limitations]</a>.

**Societal impacts.** In terms of societal impacts, offers a number of
benefits while carrying some risks. Its ability to rapidly adapt to a
broad range of tasks have the potential to enable non-expert users to
obtain good performance in data-starved regimes, lowering the barriers
to both beneficial and malicious applications. is exposed to the same
risks as large language models, such as outputting offensive language,
propagating social biases and stereotypes, as well as leaking private
information [weidinger2021harms](None), [chinchilla](None). Its ability
to additionally handle visual inputs poses specific risks such as gender
and racial biases relating to the contents of the input images, similar
to a number of visual recognition
systems [hendricks2018women](None), [zhao2021understanding](None), [buolamwini2018gender](None), [de2019does](None), [schwemmer2020diagnosing](None).
We refer the reader to
Appendix <a href="#sec:broader_impact" data-reference-type="ref"
data-reference="sec:broader_impact">[sec:broader_impact]</a><a href="#sec:broader_impact" data-reference-type="ref"
data-reference="sec:broader_impact">[sec:broader_impact]</a> for a more
extensive discussion of the societal impacts of our work, both positive
and negative; as well as mitigation strategies and early investigations
of risks relating to racial or gender bias and toxic outputs. Finally we
note that, following prior work focusing on language
models [thoppilan2022lamda](None), [perez2022red](None), [menick2022teaching](None),
the few-shot capabilities of could be useful for mitigating such risks.

**Conclusion.** We proposed Flamingo, a general-purpose family of models
that can be applied to image and video tasks with minimal task-specific
training data. We also qualitatively explored interactive abilities of 
such as “chatting” with the model, demonstrating flexibility beyond
traditional vision benchmarks. Our results suggest that connecting
pre-trained large language models with powerful visual models is an
important step towards general-purpose visual understanding.

# Checklist [checklist]

1.  For all authors...

    1.  Do the main claims made in the abstract and introduction
        accurately reflect the paper’s contributions and scope?

    2.  Did you describe the limitations of your work?

    3.  Did you discuss any potential negative societal impacts of your
        work?

    4.  Have you read the ethics review guidelines and ensured that your
        paper conforms to them?

2.  If you are including theoretical results...

    1.  Did you state the full set of assumptions of all theoretical
        results?

    2.  Did you include complete proofs of all theoretical results?

3.  If you ran experiments...

    1.  Did you include the code, data, and instructions needed to
        reproduce the main experimental results (either in the
        supplemental material or as a URL)?

    2.  Did you specify all the training details (e.g., data splits,
        hyperparameters, how they were chosen)?

    3.  Did you report error bars (e.g., with respect to the random seed
        after running experiments multiple times)?

    4.  Did you include the total amount of compute and the type of
        resources used (e.g., type of GPUs, internal cluster, or cloud
        provider)?

4.  If you are using existing assets (e.g., code, data, models) or
    curating/releasing new assets...

    1.  If your work uses existing assets, did you cite the creators?

    2.  Did you mention the license of the assets?

    3.  Did you include any new assets either in the supplemental
        material or as a URL?

    4.  Did you discuss whether and how consent was obtained from people
        whose data you’re using/curating?

    5.  Did you discuss whether the data you are using/curating contains
        personally identifiable information or offensive content? .

5.  If you used crowdsourcing or conducted research with human
    subjects...

    1.  Did you include the full text of instructions given to
        participants and screenshots, if applicable?

    2.  Did you describe any potential participant risks, with links to
        Institutional Review Board (IRB) approvals, if applicable?

    3.  Did you include the estimated hourly wage paid to participants
        and the total amount spent on participant compensation?