[^1]:  Corresponding author.





# Introduction

Large Multimodal Models (LMMs; [gpt4v](https://cdn.openai.com/papers/GPTV_System_Card.pdf), [gemini](http://arxiv.org/pdf/2405.12107v1), [claude3](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf))
have achieved notable advancements in recent years, demonstrating
superior performance in diverse domains, including image and video
understanding [ureader](http://arxiv.org/pdf/2311.13165v1), [xc2-4k](http://arxiv.org/pdf/2404.06512v1), digital agent
development [appagent](http://arxiv.org/pdf/2312.13771v2), and
robotics [roboflamingo](http://arxiv.org/pdf/2311.01378v3). The imperative to comprehend a
wide range of tasks and intricate scenes underscores the critical role
of the visual encoder, which is mostly a Vision
Transformer (ViT; [vit](http://arxiv.org/pdf/2105.15075v2)). However, ViT’s quadratic
spatial complexity and output of excessive visual tokens limit its
application in diverse and high-resolution
tasks [ureader](http://arxiv.org/pdf/2311.13165v1), [li2023otterhd](http://arxiv.org/pdf/1102.1442v1), [xc2-4k](http://arxiv.org/pdf/2404.06512v1), [cheng2023can](http://arxiv.org/pdf/1505.06659v1). The
excessive visual tokens lead to a significant computational burden in
the Large Language Model (LLM; [llama](None), [llama2](https://doi.org/10.48550/arXiv.2307.09288)), far
exceeding the computational cost imposed by the quadratic spatial
complexity in the visual encoder. Such redundancy in the visual tokens
not only sacrifices efficiency but also impedes the effective extraction
of visual information [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [xc2-4k](http://arxiv.org/pdf/2404.06512v1). While a range
of methods (Tab. <a href="#tab:table-1" data-reference-type="ref"
data-reference="tab:table-1">[tab:table-1]</a>; [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [vary](http://arxiv.org/pdf/2312.06109v1))
have been proposed to remedy the quadratic spatial complexity of ViT,
they fail to mitigate the key problem, the redundancy in the visual
tokens [fastv](http://arxiv.org/pdf/2403.06764v2), [lin2023vila](http://arxiv.org/pdf/2306.16774v1).

Hierarchical visual backbones [resnet](http://arxiv.org/pdf/1608.05895v1), [senet](http://arxiv.org/pdf/2209.08294v1), [davit](http://arxiv.org/pdf/2108.01778v1), which
can be considered as counterparts to ViT, can well address the problem
of excessive visual tokens due to their inherent ***Information
Compression*** process. Specifically, features are sequentially
compressed across stages in hierarchical backbones. They compress visual
features by *32$\times$* [resnet](http://arxiv.org/pdf/1608.05895v1), [liu2022convnet](http://arxiv.org/pdf/2007.00649v1) compared
to ViT with only *14$\times$* [vit](http://arxiv.org/pdf/2105.15075v2). Therefore, at the
same resolution they generate fewer than *1/4* visual tokens compared to
ViT, significantly alleviating computational burdens on the LLM.
Moreover, hierarchical visual encoders, typically designed with linear
spatial complexity [liu2022convnet](http://arxiv.org/pdf/2007.00649v1), [davit](http://arxiv.org/pdf/2108.01778v1), [resnet](http://arxiv.org/pdf/1608.05895v1),
effectively tackle both the issue of excessive visual tokens and the
quadratic visual complexity.

We choose to employ ConvNeXt among the hierarchical visual encoders due
to its excellent performance [convnext-vs-vit](https://arxiv.org/pdf/2311.09215), [fc-clip](http://arxiv.org/pdf/2308.02487v2)
and the availability of off-the-shelf contrastive language-image
pretrained weights (CLIP; [clip](http://arxiv.org/pdf/2404.19696v1)), which mainstream
visual encoders of LMMs
adopt [blip2](http://arxiv.org/pdf/2301.12597v3), [llava-v1](http://arxiv.org/pdf/2402.11690v1), [qwen-vl](http://arxiv.org/pdf/2308.12966v3), [mm1](http://arxiv.org/pdf/2403.01757v1). However, directly
replacing ViT with ConvNeXt leads to inferior performance on general
capabilities
benchmarks (Section <a href="#sec:updating" data-reference-type="ref"
data-reference="sec:updating">[sec:updating]</a>). This can be
attributed to the fact that ConvNeXt is pretrained on low resolution,
whereas we directly apply it to
high-resolution [openclip](https://doi.org/10.5281/zenodo.5143773), [laion5b](http://arxiv.org/pdf/2312.15897v1). Moreover, the
pretraining data for ConvNeXt is considered to be of low
quality [metaclip](http://arxiv.org/pdf/2309.16671v4), [openclip](https://doi.org/10.5281/zenodo.5143773), [laion5b](http://arxiv.org/pdf/2312.15897v1) compared to ViT’s
pretraining data [clip](http://arxiv.org/pdf/2404.19696v1). To address these issues, we
propose to update the visual encoder rather than freezing it.
Surprisingly, updating the visual encoder enables ConvNeXt to perform
comparably to ViT on general benchmarks. On fine-grained benchmarks, we
observe that ConvNeXt outperforms ViT. These findings indicate that even
when compressing visual tokens to an equal quantity, the higher
resolution model’s features still contain more fine-grained information.
This observation inspires us to further scale up the resolution.
However, further scaling the resolution beyond 1024 leads to the
generation of excessive visual tokens. To mitigate this issue, we
further compress the visual information with an additional ConvNeXt
stage to enhance the inherent *information compression* of hierarchical
backbones. The visual inputs would be compressed by *64$\times$* rather
than *32$\times$* to further reduce the redundancy. Hence, ConvLLaVA
generates only 576 visual tokens when processing 1536 resolution inputs,
which is equivalent to the number of visual tokens generated by ViT when
processing 336 resolution
inputs (Section <a href="#sec:add-stage" data-reference-type="ref"
data-reference="sec:add-stage">[sec:add-stage]</a>).

In summary, we introduce ConvLLaVA whose visual encoder is a five-stage
ConvNeXt. ConvLLaVA compresses high-resolution images into
information-rich visual features, effectively avoiding the generation of
excessive visual tokens (in
Tab. <a href="#tab:table-1" data-reference-type="ref"
data-reference="tab:table-1">[tab:table-1]</a>; [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [minigemini](http://arxiv.org/pdf/2305.16318v2), [llava-hr](http://arxiv.org/pdf/2403.03003v1)).
Furthermore, thanks to the translation equivalence of convolution,
ConvLLaVA can be trained on low-resolution and evaluated on higher
resolutions, and it can also handle images of arbitrary aspect ratio.
Extensive experiments have demonstrated the effectiveness of our method.
ConvLLaVA 7B outperforms LLaVA-1.5-13B across various benchmarks,
including MME [mme](http://arxiv.org/pdf/2306.05179v2),
MMBench [liu2023mmbench](http://arxiv.org/pdf/2005.12661v2),
SEEDBench [li2023seed](http://arxiv.org/pdf/2311.15759v1),
RealWorldQA [grok1_5](https://x.ai/blog/grok-1.5v), TextVQA [textvqa](http://arxiv.org/pdf/2003.12462v2),
DocVQA [docvqa](http://arxiv.org/pdf/2111.05547v1), POPE [pope](http://arxiv.org/pdf/2402.15721v1), and
MMVet [mmvet](http://arxiv.org/pdf/2402.15896v1).

# Related Work

**Large Multimodal Models.** To harness the potential of Large Language
Models and incorporate visual information, BLIP series
models [blip2](http://arxiv.org/pdf/2301.12597v3), [dai2023instructblip](https://arxiv.org/pdf/2305.06500) propose the Q-former,
which generates visual tokens for LLMs to interpret visual data.
Meanwhile, LLaVA [llava-v1](http://arxiv.org/pdf/2402.11690v1) employs a single linear layer
to map visual features to the word embedding space, allowing LLMs to
perceive vision features. These approaches utilize the ViT as the visual
encoder [clip](http://arxiv.org/pdf/2404.19696v1), [vit](http://arxiv.org/pdf/2105.15075v2), [honeybee](http://arxiv.org/pdf/2312.06742v2), [lin2023vila](http://arxiv.org/pdf/2306.16774v1), [minigpt](http://arxiv.org/pdf/2402.17510v1),
primarily tailored for low-resolution visual data (e.g., 224 or 336
resolution). Moreover, Qwen-VL [qwen-vl](http://arxiv.org/pdf/2308.12966v3) and
mPLUG-owl2 [mplug-owl2](http://arxiv.org/pdf/2311.04257v2) scale the resolution of ViT to
448 by updating the weights of ViT. However, these methods fail to
further scale up resolution due to the quadratic spatial complexity of
ViT, while ConvNeXt can scale up the resolution with the linear cost
increase. Qwen-VL [qwen-vl](http://arxiv.org/pdf/2308.12966v3) and
mPLUG-owl2 [mplug-owl2](http://arxiv.org/pdf/2311.04257v2) also explore to reduce the visual
tokens via resampler. However, recent
studies [honeybee](http://arxiv.org/pdf/2312.06742v2), [xc2-4k](http://arxiv.org/pdf/2404.06512v1) show that convolution or simply
concatenation performs better than resampler.

**High-resolution LMMs with Cropping.** The representative cropping
method for high-resolution LMMs is introduced in
LLaVA-NExT [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), which partitions an image into
four patches, each encoded separately by ViT and subsequently
concatenated for LLM processing. A collection of methods have adopted
cropping to scale up
resolution [ureader](http://arxiv.org/pdf/2311.13165v1), [lin2023sphinx](http://arxiv.org/pdf/2311.07575v1), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [xc2-4k](http://arxiv.org/pdf/2404.06512v1).
While effective in reducing ViT complexity, cropping compromises the
structural integrity of the image, thus potentially impacting overall
performance. Moreover, the proliferation of visual tokens introduced by
cropping poses significant complexity on LLMs and challenges the
retrieval capabilities of LLMs [xc2-4k](http://arxiv.org/pdf/2404.06512v1).

**High-resolution LMMs with Extra Visual Encoders.** Incorporating an
auxiliary visual encoder for high-resolution image understanding would
not significantly increase the number of visual tokens.
Vary [vary](http://arxiv.org/pdf/2312.06109v1) and Deepseek-VL [deepseek-vl](http://arxiv.org/pdf/2402.17510v1)
utilize SAM [sam](http://arxiv.org/pdf/2305.01275v1) as a high-resolution visual encoder to
augment the feature of ViT. MiniGemini-HD [minigemini](http://arxiv.org/pdf/2305.16318v2)
and LLaVA-HR [llava-hr](http://arxiv.org/pdf/2403.03003v1) employ
ConvNeXt [openclip](https://doi.org/10.5281/zenodo.5143773) to process high-resolution images and
use cross-attention or adapters to extract features from the
high-resolution input. However, these methods introduce additional
complexity through supplementary visual encoders and associated
hyperparameters. Furthermore, extracting features from low-quality
representations (e.g., LAION-CLIP-ConvNeXt) may potentially compromise
LMMs’ performance [gadre2024datacomp](http://arxiv.org/pdf/2004.12070v2), [metaclip](http://arxiv.org/pdf/2309.16671v4).

# ConvLLaVA [sec:method]

We present ConvLLaVA, as illustrated in
Fig. <a href="#fig:structure" data-reference-type="ref"
data-reference="fig:structure">1</a> (b), whose visual encoder is a
five-stage ConvNeXt. We first introduce the overall architecture and the
advantages of our ConvLLaVA in
Section <a href="#sec:convllava" data-reference-type="ref"
data-reference="sec:convllava">1.1</a>. The two major optimizations:
updating the visual encoder and training an additional stage are
introduced in Section <a href="#sec:updating" data-reference-type="ref"
data-reference="sec:updating">1.2</a> and
Section <a href="#sec:add-stage" data-reference-type="ref"
data-reference="sec:add-stage">1.3</a>.

## ConvNeXt as Standalone Visual Encoder [sec:convllava]

<figure id="fig:structure">
<embed src="/papers/vision_rich/arXiv-2405.15738v1_md/images/method.png" />
<figcaption>We show the structure for LLaVA and ConvLLaVA in (a) and
(b). ConvNeXt has a hierarchical structure which compresses visual
tokens between stages. The training procedure is composed of three
training stages and the trainable parameters for each stage are shown in
(c). </figcaption>
</figure>

The architecture of ConvLLaVA is identical to most popular general LMMs,
*e.g.*, LLaVA [llava-v1](http://arxiv.org/pdf/2402.11690v1), [llava-v1-5](http://arxiv.org/pdf/2310.19145v1),
Qwen-VL [qwen-vl](http://arxiv.org/pdf/2308.12966v3), and VILA [lin2023vila](http://arxiv.org/pdf/2306.16774v1).
These models comprise three components as shown in
Fig. <a href="#fig:structure" data-reference-type="ref"
data-reference="fig:structure">1</a> (a): a vision encoder $g()$, a
large language model $f()$, and a vision-language projector $h()$.
Specifically, the vision model encodes the visual inputs $\vx$ into
latent visual embeddings $g(\vx)$. The vision-language projector then
maps the latent visual embeddings into the embedding space of the
language model $\vz = h(g(\vx))$. Given the visual embeddings $\vz$ and
text embeddings $\vt$ encoded by the language tokenizer, these
embeddings are concatenated along the sequence dimension and then passed
to the language model. Finally, the vision language model is trained
with language modeling loss [gpt](http://arxiv.org/pdf/2310.01427v1). Considering that our
study mainly focuses on the visual encoder, we employ a two-layer MLP
and Vicuna-7B [vicuna](http://arxiv.org/pdf/2306.05685v4) as the projector and language
model following LLaVA-1.5 [llava-v1-5](http://arxiv.org/pdf/2310.19145v1). Rather than using
CLIP-VIT [clip](http://arxiv.org/pdf/2404.19696v1), we introduce
CLIP-ConvNeXt [liu2022convnet](http://arxiv.org/pdf/2007.00649v1), [openclip](https://doi.org/10.5281/zenodo.5143773) as the standalone
visual encoder.

<div class="wrapfigure" markdown="1">

r0.4 <embed src="/papers/vision_rich/arXiv-2405.15738v1_md/images/flops.png" />

</div>

**ConvNeXt.** The basic block of ConvNeXt comprises a depth-wise
convolution and a feed-forward network [liu2022convnet](http://arxiv.org/pdf/2007.00649v1).
The depth-wise convolution has a *7$\times$`<!-- -->`{=html}7* kernel
size, and the computation complexity is $\mathcal{O}(k^2CN)$, where $k$,
$C$, and $N$ are the kernel size, number of channels, and number of
visual tokens, respectively. In contrast, the complexity of
self-attention in ViT is $\mathcal{O}(4C^2N+2CN^2)$. Consequently, the
spatial complexity of ConvNeXt is significantly lower than ViT. The
input is initially processed by a *4$\times$`<!-- -->`{=html}4*
non-overlapping convolution downsampling layer. Subsequently, the
features are successively fed into the four stages of ConvNeXt, while
each stage comprises several ConvNeXt blocks. Feature maps are
downsampled by *2$\times$*, and dimensions are expanded by *2$\times$*
between stages. The output of the ConvNeXt is downsampled by
*32$\times$*, rather than *14$\times$* of ViT-L. Hence, ConvNeXt
produces less than *1/4* visual tokens compared to ViT, which alleviates
the computation load of the language model. Benefiting from the linear
spatial complexity and fewer visual tokens, the computation reduction of
LMMs from ViT-L (<span style="color: red">red</span> line) to ConvNeXt
(<span style="color: blue">blue</span> line) is almost *8$\times$* as
illustrated in Fig. <a href="#fig:quality" data-reference-type="ref"
data-reference="fig:quality">[fig:quality]</a>.

**Five-stage ConvNeXt$\dag$.** Leveraging ConvNeXt as the visual encoder
is efficient for encoding 768 resolution images, while scaling
resolutions to higher than 768 produces excessive visual tokens.
Previous studies [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [minigemini](http://arxiv.org/pdf/2305.16318v2) neglect to
explore compressing visual tokens, while compressing visual tokens has
been proven to be reasonable since there is redundancy in the visual
representation [lin2023vila](http://arxiv.org/pdf/2306.16774v1), [fastv](http://arxiv.org/pdf/2403.06764v2). These studies suggest
that we can further downsample visual features using ConvNeXt. We
propose to compress visual features by incorporating ConvNeXt blocks for
stage 5 into the original four-stage model. We prefer using ConvNeXt
blocks over other structures due to the following three reasons (1) The
five-stage ConvNeXt, as a whole, could be transferred as a visual
encoder for other LMMs, whereas downsampling in the projector does not
offer such flexibility (2) ConvNeXt blocks maintain translation
equivariance, allowing them to effectively process images of any aspect
ratio, unlike attention blocks. (3) The impact on performance from the
downsampling stage is minimal, except that the resampler consistently
underperforms compared to other methods, as evidenced
by [honeybee](http://arxiv.org/pdf/2312.06742v2), [xc2-4k](http://arxiv.org/pdf/2404.06512v1), [mm1](http://arxiv.org/pdf/2403.01757v1). Finally, we denote the overall
five-stage ConvNeXt as ConvNeXt$\dag$. At 1536 resolution,
ConvNeXt$\dag$ reduces the number of visual tokens to 576, equivalent to
that of ViT at 336 resolution. This would reduce the total computation
by *6$\times$* *w.r.t.* the original ConvNeXt
(<span style="color: blue">blue</span> line) to ConvNeXt$\dag$
(<span style="color: teal">green</span> line) as shown in
Fig. <a href="#fig:quality" data-reference-type="ref"
data-reference="fig:quality">[fig:quality]</a>. Our approach is more
computationally efficient than cropping methods, which often produce an
excessive number of visual
tokens [mm1](http://arxiv.org/pdf/2403.01757v1), [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [li2023monkey](http://arxiv.org/pdf/2103.15488v1). Furthermore, by
eliminating the need for cropping and merging, ConvLLaVA avoids the
global view, thereby further reducing the number of visual tokens.

## Updating ConvNeXt is Essential [sec:updating]

The mainstream optimization
approach [llava-v1](http://arxiv.org/pdf/2402.11690v1), [lin2023vila](http://arxiv.org/pdf/2306.16774v1) freezes the vision
encoder during training, as it has better performance and is more
efficient than updating the visual encoder [prismatic](http://arxiv.org/pdf/2402.07865v1).
However, freezing ConvNeXt during training is sub-optimal. Hence, we
conduct depth analysis to prove that freezing the visual encoder (i.e.,
ConvNeXt) would inherit the defects from pretraining, and updating
ConvNeXt may both improve the quality of representations and adapt them
to high-resolution inputs.

**Setups of Freezing ConvNeXt.** The optimization procedure is the same
as LLaVA-1.5 [llava-v1-5](http://arxiv.org/pdf/2310.19145v1). For training the projector and
instruction tuning, we use the same 558k caption dataset and 665k
instruction data, respectively. Our visual encoder CLIP-ConvNeXt-L is
pretrained on 256 resolution and fine-tuned with 320 resolution based on
LAION-2B [liu2022convnet](http://arxiv.org/pdf/2007.00649v1), [openclip](https://doi.org/10.5281/zenodo.5143773). We directly increase
the resolution to 512 and 768 when applying ConvNeXt as the vision
encoder. As for the baseline, we use ViT which is pretrained on 336
resolution with OpenAI WIT dataset [clip](http://arxiv.org/pdf/2404.19696v1). The training
and inference speed for ConvNeXt on 768 resolution is on par with ViT on
336 resolution. Hence, we consider the comparison between 768-resolution
ConvNeXt and 336-resolution ViT to be fair. Detailed training procedure
is shown in Tab. <a href="#tab:hy-llava" data-reference-type="ref"
data-reference="tab:hy-llava">[tab:hy-llava]</a>.

**Benchmarks.** We use four standard benchmarks to evaluate the results:
two general capability benchmarks,
MMbench [liu2023mmbench](http://arxiv.org/pdf/2005.12661v2),
SEEDBench [li2023seed](http://arxiv.org/pdf/2311.15759v1), and two fine-grained OCR
benchmarks, TextVQA [textvqa](http://arxiv.org/pdf/2003.12462v2) and
DocVQA [docvqa](http://arxiv.org/pdf/2111.05547v1). It is worth noting that our evaluation
procedure for TextVQA differs slightly from
LLaVA-1.5 [llava-v1-5](http://arxiv.org/pdf/2310.19145v1), as we use VLMEVALKIT which does
not include OCR tokens in the question.

**Results for Freezing the Visual Encoder.** As shown in
Tab. <a href="#tab:freezing-encoder" data-reference-type="ref"
data-reference="tab:freezing-encoder">[tab:freezing-encoder]</a>, we
observe the following results:

\(1\) ConvNeXt has significant advantages over ViT on OCR benchmarks. On
TextVQA and DocVQA, both 512 and 768 resolution ConvNeXt outperforms ViT
due to their higher resolution [prismatic](http://arxiv.org/pdf/2402.07865v1), [mplug-owl2](http://arxiv.org/pdf/2311.04257v2).
Even with fewer visual tokens, the 512-resolution ConvNeXt still
outperforms the 336-resolution ViT.

\(2\) The overall general capability of ConvNeXt is inferior to ViT. For
general benchmarks, on SEEDBench, 768-resolution ConvNeXt performs
comparably with ViT. While on MMBench, ConvNeXt underperforms ViT. We
hypothesize that there are two reasons for the performance gap on
MMbench: First, ConvNeXt is pretrained on low resolution but directly
applied on high resolution. Such employment affects the quality of
visual features. Second, the pretrained representation for ConvNeXt may
be inferior to OpenAI’s ViT [clip](http://arxiv.org/pdf/2404.19696v1).

The results imply that increasing resolution without training could
affect the quality of representation and hamper the performance of LMMs.
However, studies have shown that simply updating the visual encoder
during instruction tuning can hinder
performance [prismatic](http://arxiv.org/pdf/2402.07865v1). To mitigate this issue,
ShareGPT4V [sharegpt4v](http://arxiv.org/pdf/1809.10312v1) provides an effective training
protocol and a high-quality dataset for updating the visual encoder.
Therefore, we adopt this effective method to update the visual encoder.

**Setups of Updating ConvNeXt.** To update the visual encoder, we first
leverage the 558k caption dataset for projector
initialization [llava-v1-5](http://arxiv.org/pdf/2310.19145v1). Then, we apply a
high-quality caption dataset, ShareGPT4V-PT [sharegpt4v](http://arxiv.org/pdf/1809.10312v1),
to train the entire vision-language model including the visual encoder.
Finally, the LLaVA 665k instruction tuning dataset is used for visual
instruction tuning. The detailed training procedure is shown
in Tab. <a href="#tab:hy-sharegpt4v" data-reference-type="ref"
data-reference="tab:hy-sharegpt4v">[tab:hy-sharegpt4v]</a>. The last 12
layers of ViT-L are trainable (according to
ShareGPT4V [sharegpt4v](http://arxiv.org/pdf/1809.10312v1)). For ConvNeXt, we update the
last 18 blocks (ConvNeXt-L has a total of 36 blocks).

**Results for Updating the Visual Encoder.** As shown in
Tab. <a href="#tab:ShareGPT4V" data-reference-type="ref"
data-reference="tab:ShareGPT4V">[tab:ShareGPT4V]</a>, we observe the
following results:

\(1\) ConvNeXt has significant advantages over ViT on the OCR benchmark.
The improvement for 768 resolution ConvNeXt is larger than 336
resolution ViT (6.3/10.4 *v.s.* 4.6/5.2). These results demonstrate the
idea of compressing high-resolution visual inputs to a small
number (*e.g.*, 576) of information-rich visual tokens is feasible.
Compressing does not lead to great information loss. Even with the same
number of tokens, ConvNeXt preserves more fine-grained visual
information and significantly outperforms ViT.

\(2\) For general benchmarks, ConvNeXt performs on par with ViT.
Specifically, ConvNeXt outperforms ViT on SEEDBench and performs on par
with ViT on MMBench. Notably, the performance gap between the 768
resolution ConvNeXt and the 336 resolution ViT on MMBench is narrowed
from 3.3 to 0.3 compared with freezing the visual encoder. This implies
that updating the visual encoder is essential. To further support this,
we show the results of updating the visual encoder with more data in
Appendix <a href="#app:more-data" data-reference-type="ref"
data-reference="app:more-data">[app:more-data]</a>.

Generally, the updated ConvNeXt performs better than ViT on these 4
benchmarks. This evidences that updating the ConvNeXt significantly
enhances the performances, underscoring its critical importance.
Previous methods employ ConvNeXt as an auxiliary visual encoder and
directly increase the resolution to 1024 [llava-hr](http://arxiv.org/pdf/2403.03003v1) or
1536 [minigemini](http://arxiv.org/pdf/2305.16318v2). They fail to identify the problem that
scaling up the resolution without updating ConvNeXt would compromise the
performance. Our method, delving deeper into the root of the issue,
provides a simple yet effective solution to scaling up the resolution.

## Training with Stage 5 Scales up Resolution to 1536 [sec:add-stage]

As we mentioned in
Section <a href="#sec:convllava" data-reference-type="ref"
data-reference="sec:convllava">1.1</a>, scaling resolution to higher
than 768 would generate excessive visual tokens. To reduce the
redundancy and mitigate the excessive computational demands on the large
language model (LLM), we propose training stage 5 for the ConvNeXt model
to compress the visual information (training protocol shown in
Fig. <a href="#fig:structure" data-reference-type="ref"
data-reference="fig:structure">1</a> (c)).

**Implementation Details.** We employ a three-stage training protocol.
In the projector initialization stage, we train the fifth stage layers
and the projector with the ShareGPT4V-PT
data [sharegpt4v](http://arxiv.org/pdf/1809.10312v1). In the second stage, we train the
entire model with the ShareGPT4V-PT data. For instruction tuning, we
utilize the 665k LLaVA instruction data to train the LLM and the
projector. The training protocol is similar to the protocol for updating
the visual encoder. The only difference is that we train the fifth stage
and projector with ShareGPT4V-PT data, while experiments in
Section <a href="#sec:updating" data-reference-type="ref"
data-reference="sec:updating">1.2</a> train the projector with the 558k
caption data in the first training stage. We add 6 layers in stage 5 and
tune the last three stages in the second training phase. Ablation
studies on these hyper-parameters are included in
Appendix <a href="#app:stage-add-layers" data-reference-type="ref"
data-reference="app:stage-add-layers">[app:stage-add-layers]</a>.

**Results for ConvNeXt$\dag$.** We present the results of adding stage 5
to ConvNeXt in Tab. <a href="#tab:add-stage" data-reference-type="ref"
data-reference="tab:add-stage">[tab:add-stage]</a>. Scaling up the
resolution consistently improves performance on SEEDBench, TextVQA, and
DocVQA, which require fine-grained understanding and benefit from the
higher resolution. These results highlight the effectiveness of our
method of training stage 5. However, on MMBench, the performance of
ConvNeXt$\dag$ exhibits a slight drop when scaling the resolution from
1024 to 1536. The resolution of 1536 is approximately six times higher
than the pretraining resolution (256). Adapting the pretrained visual
encoder to effectively extract global information from such a
significant increase in resolution requires a substantial amount of
training data. In Section <a href="#sec:exp" data-reference-type="ref"
data-reference="sec:exp">[sec:exp]</a>, we verify this hypothesis by
providing sufficient data to the visual encoder in the second training
stage.

<figure id="fig:resolution-tokens">
<figure>
<embed src="/papers/vision_rich/arXiv-2405.15738v1_md/images/seed.png" />
</figure>
<figure>
<embed src="/papers/vision_rich/arXiv-2405.15738v1_md/images/docvqa.png" style="width:99.0%" />
</figure>
<figcaption>Comparisons of ConvNeXt and ConvNeXt<span
class="math inline">†</span> on SEEDBench and DocVQA. The marked number
above the line shows the resolution of the model. </figcaption>
</figure>

**On Scaling Resolution.** When we increase the resolution, the number
of visual tokens also increases. These two factors are entangled, and
there has been a lack of in-depth investigation into the relationship
between them. Previous work claims that raw resolution matters more than
the number of visual tokens [lin2023vila](http://arxiv.org/pdf/2306.16774v1). We experiment
on the general benchmark SEEDBench and OCR benchmark DocVQA to
investigate these assumptions. Our method provides control experiments
to reveal the relationship between resolution and the number of visual
tokens. We compare the results of ConvNeXt (trained in
Section <a href="#sec:updating" data-reference-type="ref"
data-reference="sec:updating">1.2</a>) and ConvNeXt$\dag$ (trained in
Section <a href="#sec:add-stage" data-reference-type="ref"
data-reference="sec:add-stage">1.3</a>) as the visual encoder for LMMs
under the same number of visual tokens. The two series of models are
pretrained with ShareGPT4V-PT and instruction-tuned with 665k LLaVA
instruction data. ConvNeXt$\dag$ has an additional stage to compress the
number of visual tokens to 1/4. Hence, the differences between these two
series models have been largely reduced. Our control experiments reveal
novel findings:

\(1\) When the number of visual tokens is the same, the higher
resolution model exhibits better performance on SEEDBench and DocVQA. In
the Fig.<a href="#fig:resolution-tokens" data-reference-type="ref"
data-reference="fig:resolution-tokens">2</a>, the green line
consistently outperforms the blue line. This is because that
high-resolution model provides finer-grained and higher-quality visual
features even if the output number of visual tokens is the same.
Previous work [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), [li2023monkey](http://arxiv.org/pdf/2103.15488v1), [xc2-4k](http://arxiv.org/pdf/2404.06512v1) which
scales up the resolution by splitting the image into patches would
generate excessive visual tokens. Such cropping methods significantly
sacrifice efficiency and challenge the retrieval capability of LLM. Our
core discovery presents a promising approach to enrich the information
contained in visual features without compromising efficiency.
Compressing high-resolution images into information-rich visual tokens
is more efficient than the cropping method. Training a stage to further
compress visual features provides a manner to increase resolution and
maintain a moderate computational cost.

\(2\) The importance of the number of visual tokens varies across
different benchmarks at equivalent resolution. For general benchmarks
like SEEDBench, the performance drop brought by compressing visual
tokens for the 768-resolution models is marginal (0.9 on SEEDBench).
However, for OCR benchmarks like DocVQA, the performance drop for the
model with fewer visual tokens is substantial (9.1 on DocVQA). Overall,
these results demonstrate that while compressing visual tokens causes
only slight information loss on general benchmarks, but leads to
significant information loss on fine-grained OCR benchmarks.

# Experiments [sec:exp]

Our results demonstrate that scaling up the resolution of ConvNeXt and
updating the visual encoder are two effective approaches to training an
advanced, high-resolution Language-Multimodal Model. However, we found
that the available training data was insufficient to fully unleash the
potential of these approaches. Consequently, we scaled up the
high-quality training data to address this limitation.

## Training Setups

**Training Stages.** We adopt a three-stage training protocol to train
ConvLLaVA as shown in
Fig. <a href="#fig:structure" data-reference-type="ref"
data-reference="fig:structure">[fig:structure]</a> (c). The training
process is categorized into three stages: (1) *Projector
Initialization.* We train the fifth stage of the ConvNeXt model and the
vision-language projector. We utilize caption data including
ShareGPT4V-PT [sharegpt4v](http://arxiv.org/pdf/1809.10312v1),
ShareGPT4V [sharegpt4v](http://arxiv.org/pdf/1809.10312v1), and ALLaVA
captions [allava](http://arxiv.org/pdf/2112.07133v2), totaling approximately 2M examples.
(2) *Vision-Language Pretraining.* We employ caption data including
ShareGPT4V-PT [sharegpt4v](http://arxiv.org/pdf/1809.10312v1),
ShareGPT4V [sharegpt4v](http://arxiv.org/pdf/1809.10312v1), ALLaVA [allava](http://arxiv.org/pdf/2112.07133v2),
and a 190k open-sourced subset of VFLAN [vflan](http://arxiv.org/pdf/2403.04343v1),
amounting to 2.9M data. (3) *Visual Instruction Tuning.* We fine-tune
the model with the 665k LLaVA instruction
dataset [llava-v1-5](http://arxiv.org/pdf/2310.19145v1). In each stage, we train the model
for 1 epoch with the AdamW optimizer. The cosine learning rate schedule
is also applied.

**Implementation Details.** We utilize the LAION-2B pretrained
ConvNeXt-L model as our visual encoder [openclip](https://doi.org/10.5281/zenodo.5143773). In the
three training stages, the resolution is scaled up to a fixed value. We
train ConvLLaVA at 768, 1024, and 1536 resolutions. The learning rates
in the three training stages are 3e-4, 2e-5, and 2e-5, respectively.
Meanwhile, the batch sizes are 256, 256, and 128. Training the ConvLLaVA
768 resolution model takes approximately 18 hours on 2 A800 machines.
The instruction tuning costs 20 hours for LLaVA-NExT 7B on an A100
machine [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/), while it tasks only 9 hours for our
1536 resolution ConvLLaVA on a single machine.

**Evaluation Benchmarks.** To systematically investigate the performance
of our model, we include more benchmarks for evaluation, including
MME [mme](http://arxiv.org/pdf/2306.05179v2), MMBench [liu2023mmbench](http://arxiv.org/pdf/2005.12661v2),
SEEDBench [li2023seed](http://arxiv.org/pdf/2311.15759v1),
MMMU [yue2023mmmu](http://arxiv.org/pdf/2311.16502v3), MMVet [mmvet](http://arxiv.org/pdf/2402.15896v1),
RealWorldQA [grok1_5](https://x.ai/blog/grok-1.5v), TextVQA [textvqa](http://arxiv.org/pdf/2003.12462v2),
DocVQA [docvqa](http://arxiv.org/pdf/2111.05547v1), and POPE [pope](http://arxiv.org/pdf/2402.15721v1). Our
results are measured by VLMEVALKIT. We also assess the performance on
grounding benchmarks, including RefCOCO [refcoco](http://arxiv.org/pdf/1808.08754v1),
RefCOCO+, and RefCOCOg [refcocog](http://arxiv.org/pdf/1511.02283v3).

<span id="tab:main" label="tab:main"></span>

<div id="tab:grounding" markdown="1">

|  |  |  |  |  |  |  |  |  |  |  |  |  |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Method | Res. | \#V Tokens | LLM | RefCOCO |  |  | RefCOCO+ |  |  | RefCOCOg |  | Avg |
|  |  |  |  | val | test-A | test-B | val | test-A | test-B | val | test |  |
| LLaVA-1.5 | 336 | 576 | 7B | 76.3 | 83.2 | 67.9 | 66.8 | 77.0 | 56.8 | 70.4 | 70.0 | 71.1 |
| LLaVA-1.5 | 336 | 576 | 13B | 84.0 | 89.5 | 77.1 | 76.3 | 84.3 | 66.1 | 78.8 | 78.3 | 79.3 |
| ConvLLaVA | 768 | 144 | 7B | 84.5 | 89.0 | 79.2 | 77.7 | 84.9 | 69.7 | 79.8 | 79.7 | 80.6 |
| ConvLLaVA | 1024 | 256 | 7B | 85.5 | 89.6 | 78.8 | 79.3 | 86.1 | 70.3 | 80.6 | 81.2 | 81.4 |
| ConvLLaVA | 1536 | 576 | 7B | **86.5** | **90.6** | **80.5** | **80.0** | **86.8** | **71.5** | **82.0** | **82.4** | **82.3** |

Results on referring expression comprehension tasks. The models in this
table are trained with the same grounding data. We mark the best
performance of the model **bold**.

</div>

## Quantitative Results

We perform a comprehensive comparison with state-of-the-art models on 7
different benchmarks (Tab. <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>). Our model achieves consistent
improvements compared to LLaVA-1.5. Our 7B model even exhibits
comparable performance with LLaVA-1.5 13B and LLaVA-NExT
7B [llava-v1-6](https://llava-vl.github.io/blog/2024-01-30-llava-next/). On OCR benchmarks like TextVQA and
DocVQA, our model outperforms the LLaVA-1.5 7B and 13B models. Since OCR
benchmarks are sensitive to resolution, our ConvLLaVA series models
demonstrate consistent improvement on TextVQA and DocVQA with higher
resolution, showcasing the effectiveness of scaling up resolution.
Notably, our model surpasses Qwen-VL-Chat on DocVQA which has millions
of document training data. While there is only a limited number of
document data in our training dataset. This shows the benefits of the
high-resolution design of our model. ConvLLaVA outperforms LLaVA-NExT on
MMBench, TextVQA, POPE, and MMVet.

For grounding benchmarks, our model and LLaVA are trained with the same
set of grounding data. The comparison between them is fair. On RefCOCO,
RefCOCO+, and RefCOCOg, ConvLLaVA exhibits consistent improvement when
increasing
resolution (Tab. <a href="#tab:grounding" data-reference-type="ref"
data-reference="tab:grounding">1</a>). ConvLLaVA outperforms LLaVA-7B
and 13B model on all 8 test splits. This demonstrates the benefits of
higher resolution for grounding tasks. Our 7B model also surpasses 13B
LLaVA model on all 8 benchmarks.

## Understanding Any Aspect Ratio Images and Highre Resolutions

Thanks to the translation equivalence of convolution neural network, our
model could be trained on a fixed resolution but inference on higher
resolution and with an arbitrary aspect ratio. We test such ability on
our 1536 resolution model ConvLLaVA.

<div class="wraptable" markdown="1">

r0.4

|   Input Shape   |   SEED   |   Text   |   Doc    |
|:---------------:|:--------:|:--------:|:--------:|
|  (1536, 1536)   | **70.2** | **65.8** |   59.0   |
| short side=1536 |   68.9   |   64.6   |   65.0   |
| short side=1664 |   67.3   |   64.2   | **65.7** |

<span id="tab:shape" label="tab:shape"></span>

</div>

The original image preprocessing process is padding the image to a
square, resizing the image to 1536, and center
cropping [llava-v1-5](http://arxiv.org/pdf/2310.19145v1). We canceling padding and center
cropping. Hence, the short side of the image should just be resized to
1536 and keep the original aspect ratio. This is the setting of how we
test images of any aspect ratio. The results are shown in
Tab. <a href="#tab:shape" data-reference-type="ref"
data-reference="tab:shape">[tab:shape]</a>. We observe that on the
general benchmark, SEEDBench, the performance slightly decreases. On OCR
benchmarks, especially on DocVQA, the performance is improved. The
reason for this we think is that the image aspect ratio in DocVQA is not
1:1, forcely transforming the image into a square would lower the
resolution of the image.

We also test ConvLLaVA when resizing the short side of images to 1664
resolution which is higher than its pretrained 1536 resolution. We
observe that on DocVQA the performance could be further improved to
65.7.

## Discussions [sec:discussions]

**Architectures and data.** While we have demonstrated the effectiveness
of our method, there remains room for further improvement. The ConvNeXt
architecture we use is tailored for low-resolution image understanding
(e.g., 256), with a kernel size of 7 optimized for such resolutions.
However, as the resolution increases to 1536, the relatively small
kernel size may limit the model capacity when the resolution is
extremely high. Besides, the number of layers in the ConvNeXt four
stages (3, 3, 27, 3) is designed for a 4-stage model and may not be
optimal for our 5-stage model. Therefore, a potential future direction
could involve designing a five-stage, linear spatial complexity,
hierarchical high-resolution vision encoder. We emphasize the critical
role of the five-stage visual encoder since it is fit for
high-resolution LMM. It compresses visual features by *64$\times$*,
greatly reducing the redundancy in its visual tokens. In contrast,
four-stage visual encoders, designed for traditional computer vision
tasks, output excessive tokens when resolution is high.

**Linear spatial complexity and information compression.** We identify
*linear spatial complexity* and *information compression* procedure as
two critical properties for future visual encoders of LMMs. These
properties ensure the efficiency of both the visual encoder and the LLM,
respectively. Furthermore, they are crucial for multi-image, interleaved
image and text, and video understanding tasks, as these tasks commonly
result in numerous visual tokens. We anticipate that future research
will focus more on these two directions to further advance the research
of LMMs.

**Trade-off between compression and retrieval for high-resolution
understanding.** Our method, ConvLLaVA, compresses a 1536-resolution
image to 576 visual tokens with a 64$\times$ compression ratio. While
concurrent work [xc2-4k](http://arxiv.org/pdf/2404.06512v1), [internvl1.5](http://arxiv.org/pdf/2404.16821v2) explores retrieving
fine-grained image information from long visual token sequences. In the
context of high-resolution image understanding, compressing visual
information maintains computational efficiency, but excessive
compression may lead to information loss. Conversely, retaining a large
number of visual tokens avoids information loss but sacrifices
efficiency and challenges the retrieval capabilities of LLMs.
Consequently, a trade-off emerges between visual information compression
and retrieval capabilities for high-resolution understanding. Future
research should explore an optimal balance between these two factors.

# Conclusion

In this paper, we have critically examined the limitations of the visual
encoder for current LMMs: quadratic spatial complexity and numerous
visual tokens. The excessive visual tokens are the more fundamental
problem. These drawbacks hinder LMMs from efficiently understanding
high-resolution images. Consequently, we propose ConvLLaVA, whose visual
encoder is a hierarchical backbone, ConvNeXt, to mitigate this issue.
ConvLLaVA compresses high-resolution visual information into
information-rich visual representation rather than preserving all the
redundancy in the visual representation. Extensive experimental results
have demonstrated the efficacy of our proposed method. Our 7B parameter
model exhibits superior performance compared to the LLaVA-1.5 13B model.
Furthermore, our method is flexible in encoding images with arbitrary
shapes and resolutions. Our work highlights the advantages of
hierarchical visual backbones for LMMs, addressing critical challenges
while maintaining simplicity and efficiency.

# Acknowledgments [acknowledgments]

This work is supported in part by the National Natural Science
Foundation of China under Grants 62321005 and 62276150.

# Training Visual Encoder with More Data [app:more-data]

In Section <a href="#sec:updating" data-reference-type="ref"
data-reference="sec:updating">[sec:updating]</a>, we observe that
updating the visual encoder is essential for ConvNeXt as the standalone
encoder. We compare the two visual encoders with more training data in
Tab. <a href="#tab:allava-sharegpt4v" data-reference-type="ref"
data-reference="tab:allava-sharegpt4v">[tab:allava-sharegpt4v]</a>. For
the visual language training stage, we use ALLaVA and ShareGPT4V-PT. We
train the last two stages for ConvNeXt and the last 12 layers for ViT.
With more training data, ConvNeXt outperforms ViT on all the 4
benchmarks. These results validate the advantages of ConvNeXt over ViT.
This ConvNeXt model even outperforms the 768-resolution ConvLLaVA model
on some benchmarks due to its higher number of visual tokens. However,
the training and inference speed is much slower than the 768-resolution
ConvLLaVA model due to the increased number of visual tokens. The 1536
resolution ConvLLaVA, featuring outputting the same number of visual
tokens, outperforms this model. This shows higher resolution model may
have a higher model capacity to learn from data.

# Hyperparameters for 5-stage ConvNeXt [app:stage-add-layers]

We discuss the choice of hyperparameters in this section.

**Number of Trained Stages.** We conduct an ablation study to determine
the optimal number of stages for vision-language pretraining at 768
resolution. We find that fine-tuning from stage 3 yields better results
than fine-tuning from stage 4
(Tab. <a href="#tab:stages-high" data-reference-type="ref"
data-reference="tab:stages-high">[tab:stages-high]</a>). While the
performances of fine-tuning from stage 2 and stage 3 are comparable, we
opt for fine-tuning from stage 3 due to its fewer trainable parameters.

**Number of Layers in Stage 5.** We ablate on the number of ConvNeXt
layers in stage 5. Given that the number of layers in each stage is a
multiple of 3 in ConvNeXt-L, we experiment with 3, 6, and 9 layers in
stage 5. For simplicity, we perform the experiments on ConvNeXt 768. We
observe a slight decrease in performance when adding 9 layers in stage 5
(Tab. <a href="#tab:ablation-layers" data-reference-type="ref"
data-reference="tab:ablation-layers">[tab:ablation-layers]</a>).
However, it’s hard to determine whether adding 3 or 6 layers is more
beneficial for these four benchmarks. Hence, we conduct experiment on
the 1536 resolution to further investigate this
hyperparameter (Tab. <a href="#tab:add-layers-1536" data-reference-type="ref"
data-reference="tab:add-layers-1536">[tab:add-layers-1536]</a>). The
results show that adding 6 layers could be better. We opt for 6 layers
in our experiments.

# Training protocol for each experiment [app:implementations]

The detailed training hyper-parameters are shown in the following
tables.

<div id="tab:hy-llava" markdown="1">

| Training Stage  |       1        |       2        |
|:---------------:|:--------------:|:--------------:|
| Visual Encoder  |                |                |
|    Projector    |                |                |
|       LLM       |                |                |
|      data       | LLaVA LCS-558K | LLaVA SFT 665k |
|       lr        |      1e-3      |      2e-5      |
|   batch size    |      256       |      128       |
|   lr schedule   |  cosine decay  |  cosine decay  |
| lr warmup ratio |      0.03      |      0.03      |
|      epoch      |       1        |       1        |
|    optimizer    |     AdamW      |     AdamW      |

The training protocol for
Tab. <a href="#tab:freezing-encoder" data-reference-type="ref"
data-reference="tab:freezing-encoder">[tab:freezing-encoder]</a>.

</div>

<span id="tab:hy-llava" label="tab:hy-llava"></span>

<div id="tab:hy-sharegpt4v" markdown="1">

| Training Stage  |       1        |       2       |       3        |
|:---------------:|:--------------:|:-------------:|:--------------:|
| Visual Encoder  |                |               |                |
|    Projector    |                |               |                |
|       LLM       |                |               |                |
|      data       | LLaVA LCS-558K | ShareGPT4V-PT | LLaVA SFT 665k |
|       lr        |      1e-3      |     2e-5      |      2e-5      |
|   batch size    |      256       |      256      |      128       |
|   lr schedule   |  cosine decay  | cosine decay  |  cosine decay  |
| lr warmup ratio |      0.03      |     0.03      |      0.03      |
|      epoch      |       1        |       1       |       1        |
|    optimizer    |     AdamW      |     AdamW     |     AdamW      |

The training protocol for
Tab. <a href="#tab:ShareGPT4V" data-reference-type="ref"
data-reference="tab:ShareGPT4V">[tab:ShareGPT4V]</a>.

</div>

<span id="tab:hy-sharegpt4v" label="tab:hy-sharegpt4v"></span>

<div id="tab:hy-5stages" markdown="1">

| Training Stage  |       1       |       2       |       3        |
|:---------------:|:-------------:|:-------------:|:--------------:|
|    ConvNeXt     |               |               |                |
|     Stage 5     |               |               |                |
|    Projector    |               |               |                |
|       LLM       |               |               |                |
|      data       | ShareGPT4V-PT | ShareGPT4V-PT | LLaVA SFT 665k |
|       lr        |     3e-4      |     2e-5      |      2e-5      |
|   batch size    |      256      |      256      |      128       |
|   lr schedule   | cosine decay  | cosine decay  |  cosine decay  |
| lr warmup ratio |     0.03      |     0.03      |      0.03      |
|      epoch      |       1       |       1       |       1        |
|    optimizer    |     AdamW     |     AdamW     |     AdamW      |

The training protocol for
Tab. <a href="#tab:add-stage" data-reference-type="ref"
data-reference="tab:add-stage">[tab:add-stage]</a>,
Tab. <a href="#tab:stages-high" data-reference-type="ref"
data-reference="tab:stages-high">[tab:stages-high]</a>, and
Tab. <a href="#tab:ablation-layers" data-reference-type="ref"
data-reference="tab:ablation-layers">[tab:ablation-layers]</a>

</div>

<span id="tab:hy-5stages" label="tab:hy-5stages"></span>

<div id="tab:hy-main-results" markdown="1">

| Training Stage  |       1        |       2       |       3        |
|:---------------:|:--------------:|:-------------:|:--------------:|
|    ConvNeXt     |                |               |                |
|     Stage 5     |                |               |                |
|    Projector    |                |               |                |
|       LLM       |                |               |                |
|      data       | ShareGPT4V-PT  | ShareGPT4V-PT | LLaVA SFT 665k |
|                 |   ShareGPT4V   |  ShareGPT4V   |                |
|                 | ALLaVA Caption | ALLaVA, VFLAN |                |
|       lr        |      3e-4      |     2e-5      |      2e-5      |
|   batch size    |      256       |      256      |      128       |
|   lr schedule   |  cosine decay  | cosine decay  |  cosine decay  |
| lr warmup ratio |      0.03      |     0.03      |      0.03      |
|      epoch      |       1        |       1       |       1        |
|    optimizer    |     AdamW      |     AdamW     |     AdamW      |

The training protocol for
Tab. <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a>, and
Tab. <a href="#tab:grounding" data-reference-type="ref"
data-reference="tab:grounding">[tab:grounding]</a>

</div>

<span id="tab:hy-main-results" label="tab:hy-main-results"></span>