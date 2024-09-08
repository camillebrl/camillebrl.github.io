# Introduction

Large language models (LLMs) like ChatGPT [chatgpt](https://openai.com/blog/chatgpt),
BLOOM [bloom](None), and LLaMA [llama](http://arxiv.org/pdf/2402.08075v1) have
undergone rapid development to enable the realization of general
artificial intelligence, boasting impressive zero-shot capabilities
across diverse linguistic applications. With the LLM as the language
decoder, Multimodal large language models (MLLMs) such as
MiniGPT-4 [minigpt4](http://arxiv.org/pdf/2402.17510v1), LLaVA [llava](http://arxiv.org/pdf/2402.11690v1), and
mPLUG-Owl [mplugowl](http://arxiv.org/pdf/2405.00390v2) have demonstrated remarkable
zero-shot performance in various open-ended vision-and-language tasks.
These models are trained to align text and images during the
pre-training phase, and then to promote diverse abilities during the
instruction tuning phase. Interestingly, these MLLMs exhibit superficial
OCR-free text recognition abilities without explicit training on visual
text understanding datasets [mplugowl](http://arxiv.org/pdf/2405.00390v2), [llmocr](http://arxiv.org/pdf/2305.07895v5).
Nevertheless, due to lacking specific training, these models still face
the challenge of comprehending intricate relationships between visual
text and objects in diverse types of images, such as charts, documents
and webpages.

By performing unified instruction tuning for Document Understanding upon
the mPLUG-Owl [mplugowl](http://arxiv.org/pdf/2405.00390v2), we further propose a
modularized MLLM [mplug](None), [mplug2](None), namely mPLUG-DocOwl.
Our approach utilizes a modularized framework similar to mPLUG-Owl
[mplugowl](http://arxiv.org/pdf/2405.00390v2), which incorporates a visual abstractor
module to link a pre-trained LLM with a visual knowledge module,
achieving the alignment of text and images. To enhance diverse document
understanding capabilities, we reorganize various downstream document
understanding tasks in the same form of instructions. To maintain
general uni/multi-modal abilities, we also include language-only and
general vision-and-language instruction datasets used by mPLUG-Owl to
train the mPLUG-DocOwl. During training, both the visual knowledge
module and LLM decoder are frozen, only the visual abstractor and the
Low-Rank Adaption (LoRA) [lora](https://openreview.net/forum?id=nZeVKeeFYf9) in LLM are fine-tuned.

mPLUG-DocOwl achieves ocr-free state-of-the-art performance on multiple
commonly used document understanding datasets. Furthermore, our
experiments on a carefully-built document instruction understanding
evaluation set LLMDoc shows that mPLUG-DocOwl achieves significantly
better visual text understanding performance on various domains than
existing MLMMs.

Our main contributions can be highlighted as follows:

-   We propose a modularized MLLM, **mPLUG-DocOwl**, which is the first
    one to balance language-only, general vision-and-language, and
    document understanding based on unified instruction tuning.

-   We carefully construct an instruction understanding test set with
    human evaluation, dubbed **LLMDoc**, to assess diverse document
    understanding capabilities.

-   Empirical results demonstrate that our mPLUG-DocOwl surpasses
    existing methods on ocr-free document understanding, including
    multiple standard benchmarks and LLMDoc.

# Related Work

## Visual Text Understanding

There are two types of models for understanding images that contain rich
textual information. The first kind of
approaches [layoutlm](https://doi.org/10.1145/3394486.3403172), [layoutlmv3](None), [qctextcap](http://arxiv.org/pdf/2302.02124v2), [udop](http://arxiv.org/pdf/2212.02623v3), [tap](None)
utilize off-the-shelf OCR models or APIs to recognize text from images,
and then design pretraining tasks to facilitate cross-modality alignment
between visual and textual inputs. On the other hand, end-to-end
approaches [dessurt](http://arxiv.org/pdf/2203.16618v3), [donut](http://arxiv.org/pdf/2305.09520v1), [pix2struct](None) utilize a
high-resolution image encoder to learn text recognition during the
pretraining stage. Both two types of models rely on specific finetuning
on different downstream datasets and can’t achieve open-domain
instruction understanding performance like Multimodal Large Language
Models.

## Multimodal Large Language Model

Large Language Models (LLMs) have demonstrated impressive zero-shot
abilities across various open-ended tasks. Recent research has also
explored the application of LLMs for multi-modal generation, utilizing
two different paradigms: systematic collaboration and end-to-end trained
models. Systematic collaboration approaches, such as Visual ChatGPT
[visualchatgpt](None) and MM-REACT [mmreact](None),
leverage various vision experts or tools to express visual information
with text descriptions. Subsequently, LLMs, such as ChatGPT
[chatgpt](https://openai.com/blog/chatgpt), can act as agents and select appropriate
experts and tools for visual understanding. Finally, LLMs would
summarize the output of these experts to answer user queries. On the
other hand, some approaches, such as MiniGPT-4
[minigpt4](http://arxiv.org/pdf/2402.17510v1), LLaVA [llava](http://arxiv.org/pdf/2402.11690v1), and mPLUG-Owl
[mplugowl](http://arxiv.org/pdf/2405.00390v2), leverage LLMs to build unified models for
multi-modality with limited connected parameters. These methods show
superficial OCR-free text recognition abilities under the zero-shot
setting. However, for complicated document understanding, due to lacking
in-domain training, they encounter challenges in handling diverse image
types, recognizing rich texts and comprehending relationships between
visual semantic and text information. In this work, through unified
instruction tuning, mPLUG-DocOwl achieves much better document
understanding performance and maintains general uni/multi-modal
abilities.

# Conclusion

In this work, we infuse diverse ocr-free document understanding
capabilities into mPLUG-Owl by incorporating document understanding data
into instruction finetuning. Experiment results demonstrate that our
mPLUG-DocOwl achieves comparable or even better performance than
existing OCR-free methods. Besides, benefiting from language-only and
general vision-and-language instruction tuning, mPLUG-DocOwl can better
comprehend user instructions and intentions, enabling more complex
interactions. Moreover, human evaluation on LLMDoc reveals that
mPLUG-DocOwl still struggles with document-related commonsense
reasoning, mathematical calculations, and creative generation. This
provides valuable insights about developing stronger document
understanding abilities with the LLM in the future.

[^1]: Equal contribution

[^2]: Corresponding author

# Experiment

## LLMDoc

Existing benchmarks are hard to evaluate the open-ended instruction
understanding results given by MLMMs. For better compare the instruction
understanding performance in the document domain, we further construct a
test set with human evaluation, namely .

#### Data Collection

To comprehensively evaluate the model’s abilities, we consider five
scenarios to construct our evaluation dataset, including table (TabFact
[TabFact](http://arxiv.org/pdf/2311.06592v1)), chart (ChartQA [chartqa](None)),
document (DocVQA [docvqa](None)), natural image (TextVQA
[textvqa](None)) and webpage (VisualMRC
[visualmrc](http://arxiv.org/pdf/2101.11272v2)). Specifically, for each dataset, we sample
20 images from the test split. For 10 of these images, we adopt a raw
question as the instruction. While for the other 10, we ask annotators
to write instructions requiring stronger capabilities like
summarization, inference, and calculation. In total, we obtain 100 test
samples.

#### Human Evaluation

Following the rating criteria proposed in
Self-Instruct [self-instruct](https://doi.org/10.48550/arXiv.2212.10560), we perform the human
evaluation to score the model’s responses, where A \> B \> C \> D and A
represents ‘correct and satisfying response’, B means ‘acceptable
response with minor imperfections’, C refers to ‘response to the
instruction but has significant errors’ and D means ‘irrelevant or
invalid response’.

<div class="wrapfigure" markdown="1">

r0.5 <img src="/papers/text_rich_doc/arXiv-2307.02499v1_md/figs/llm_comp.png" alt="image" />

</div>

We compare with other popular mult-modal large language models,
including mPLUG-Owl [mplugowl](http://arxiv.org/pdf/2405.00390v2) and
Mini-GPT4 [minigpt4](http://arxiv.org/pdf/2402.17510v1), on . As shown in
<a href="#fig:llm_comp" data-reference-type="ref+Label"
data-reference="fig:llm_comp">[fig:llm_comp]</a>, achieves significantly
better performance, with 37 responses being scored as “A”, demonstrating
the stronger understanding ability of in diverse document scenarios.
Besides, it’s worth noting that all models have some responses scored as
“C” or “D”, showing that instruction understanding performance in the
document domain is still far from promising and needs more endeavor.

## Benchmark Evaluation

Besides human evaluation, we also compare our with ocr-free
state-of-the-art document understanding models on public datasets.
<a href="#tab:due_eval" data-reference-type="ref+Label"
data-reference="tab:due_eval">[tab:due_eval]</a> shows the comparison
with Dessurt [dessurt](http://arxiv.org/pdf/2203.16618v3), Donut [donut](http://arxiv.org/pdf/2305.09520v1)
and Pix2Struct [pix2struct](None) on
DUE-Benchmark [due](None), which mainly requires the text
recognition and layout understanding abilities on documents and tables.
Besides, <a href="#tab:other_eval" data-reference-type="ref+Label"
data-reference="tab:other_eval">[tab:other_eval]</a> presents the
evaluation on the chart, natural image and webpage datasets, which ask
stronger ability to relate visual semantics and text information.
Without finetuning on each dataset, our achieves comparable or even
better performance.

## Qualitative Analysis

<figure id="fig:cases">
<embed src="/papers/text_rich_doc/arXiv-2307.02499v1_md/figs/cases.png" style="width:100.0%" />
<figcaption>Qualitative results of . The crucial regions and
corresponding words are annotated with the same colors for clearer
visualization. Wrong answers are colored <span
style="color: red">red</span>.</figcaption>
</figure>

#### Benchmark Results.

Qualitative results on different types of images are shown in
<a href="#fig:cases" data-reference-type="ref+Label"
data-reference="fig:cases">1</a>. Crucial regions and corresponding
responses are annotated with the same colors. Case (a) shows that can
accurately find the answer from a webpage screenshot with complex
contents. Case (b) shows that is even able to understand hand-drawn
tables and correctly recognize handwritten fonts. In case (c), can
summarize key points from a chart. It successfully understands that the
table is about internet usage and infers that “Never” means “Never used
internet”. However, it also generates illusory outputs, such as "in the
United States". The question in case (d) requires the model to
understand the “Result” column, compare the points and return the date
with the best results. Case (e) demonstrates that our model is capable
of processing scanned documents and distinguishing company and person
names. Case (f) shows that can not only recognize small and blurry text
but also perform simple calculations following the user intent.

<figure id="fig:cases_human_1">
<embed src="/papers/text_rich_doc/arXiv-2307.02499v1_md/figs/cases_human.1.png" style="width:100.0%" />
<figcaption>Qualitative comparison between and Mini-GPT4 on . Part
one.</figcaption>
</figure>

<figure id="fig:cases_human_2">
<embed src="/papers/text_rich_doc/arXiv-2307.02499v1_md/figs/cases_human.2.png" style="width:100.0%" />
<figcaption>Qualitative comparison between and Mini-GPT4 on . Part
two.</figcaption>
</figure>

#### Results

<a href="#fig:cases_human_1" data-reference-type="ref+Label"
data-reference="fig:cases_human_1">2</a> and
<a href="#fig:cases_human_2" data-reference-type="ref+Label"
data-reference="fig:cases_human_2">3</a> present the comparison between
and Mini-GPT4 on .
<a href="#fig:cases_human_1" data-reference-type="ref+Label"
data-reference="fig:cases_human_1">2</a> (a) requires models to convert
a table into JSON format. Our correctly understands the instruction and
return a string in JSON format, but misses the last row. Mini-GPT4 fails
to comprehend the instruction and doesn’t understand the content within
the table. In
<a href="#fig:cases_human_1" data-reference-type="ref+Label"
data-reference="fig:cases_human_1">2</a> (b), both and Mini-GPT4
correctly recognize the name of the shop. However, Mini-GPT4 overlooks a
smaller sign indicating clothes in this shop are medical uniforms. As
for chart understanding in
<a href="#fig:cases_human_2" data-reference-type="ref+Label"
data-reference="fig:cases_human_2">3</a> (c), Mini-GPT4 gives a wrong
answer and redundant response, while our gives a concise and correct
response. In
<a href="#fig:cases_human_2" data-reference-type="ref+Label"
data-reference="fig:cases_human_2">3</a> (d), Bernadette’s actual
purpose is to confirm with Suzy if she would like to have the copy sent
overnight. This not only requires the model to accurately recognize the
text, but also to understand the relationships between involved persons.
recognizes the phrase "request a copy of chapter," but misunderstands
the subject and object. Mini-GPT4 only comprehends that this image is a
mail scenario and provides a vague and hallucinatory response. In
<a href="#fig:cases_human_2" data-reference-type="ref+Label"
data-reference="fig:cases_human_2">3</a> (e), gives a correct summary of
the two latest news but Mini-GPT4 generates news irrelevant to the
webpage screenshot.

<figure id="fig:bad_case_1">
<embed src="/papers/text_rich_doc/arXiv-2307.02499v1_md/figs/bad_cases.png" style="width:100.0%" />
<figcaption>Failure cases on . Part one.</figcaption>
</figure>

<figure id="fig:bad_case_2">
<embed src="/papers/text_rich_doc/arXiv-2307.02499v1_md/figs/bad_cases.2.png" style="width:100.0%" />
<figcaption>Failure cases on . Part two.</figcaption>
</figure>

The contains many challenging instruction understanding cases in the
document domain.
<a href="#fig:bad_case_1" data-reference-type="ref+Label"
data-reference="fig:bad_case_1">4</a> and
<a href="#fig:bad_case_2" data-reference-type="ref+Label"
data-reference="fig:bad_case_2">5</a> show some wrong responses given by
. In <a href="#fig:bad_case_1" data-reference-type="ref+Label"
data-reference="fig:bad_case_1">4</a> (a), only takes note of the three
names in the picture, but ignores the fact that the user itself is also
a speaker. In <a href="#fig:bad_case_1" data-reference-type="ref+Label"
data-reference="fig:bad_case_1">4</a> (b), fails to perform multi-step
calculations on multiple elements in the image. In
<a href="#fig:bad_case_2" data-reference-type="ref+Label"
data-reference="fig:bad_case_2">5</a> (c), the model can understand the
scene and the text in it, but fantasizes about non-existent characters.
In <a href="#fig:bad_case_2" data-reference-type="ref+Label"
data-reference="fig:bad_case_2">5</a> (d), fails to understand the
instruction for writing news and only read the texts in the tablet.