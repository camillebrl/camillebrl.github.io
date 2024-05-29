# Introduction

<div class="figure*" markdown="1">

<img src="/papers/doc_ai_databases/arXiv-2404.07503v1_md/assets/manufacture_hd.png" style="width:65.0%" alt="image" />
<span id="fig:conceptual_image" label="fig:conceptual_image"></span>

</div>

The rapid advancement of artificial intelligence (AI) technologies has
led to their widespread adoption across numerous domains, from assistant
agents (e.g., ACT-1, from Adept AI[^1]) and software development (e.g.,
Devin, from Cognition Lab[^2]) to healthcare
[singhal2022large](https://arxiv.org/abs/2212.13138) and finance
[zheng2022ai](http://arxiv.org/pdf/2106.01901v1). However, the success of AI models heavily
relies on the availability of large, diverse, and high-quality datasets
for training and evaluation. Acquiring such datasets can be a
significant challenge due to data scarcity
[babbar2019data](http://arxiv.org/pdf/2208.00147v1), privacy concerns
[abay2019privacy](http://arxiv.org/pdf/1801.01594v2), and the sheer cost of data collection
and annotation [gilardi2023chatgpt](http://arxiv.org/pdf/2303.15056v2). Pessimists predict
that we will run out of fresh text data in 2050 and image data in 2060
[villalobos2022will](https://arxiv.org/abs/2211.04325).

Synthetic data has emerged as a promising solution to address these
challenges [nikolenko2021synthetic](http://arxiv.org/pdf/1909.11512v1). Synthetic data
refers to artificially generated data that mimics the characteristics
and patterns of real-world data, but is created through algorithms
[saxton2019analysing](https://openreview.net/forum?id=H1gR5iR5FX), generative models
[borisov2022language](https://arxiv.org/abs/2210.06280), [meng2022generating](http://arxiv.org/pdf/2004.13952v2), or even
simulations [vezhnevets2023generative](https://arxiv.org/abs/2312.03664), [liu2023training](https://arxiv.org/abs/2305.16960),
rather than being directly created by humans. By leveraging synthetic
data, we can not only overcome the limitations of real-world data but
also unlock the potential to develop more robust, reliable, and fair AI
models [lucini2021real](http://arxiv.org/pdf/2208.07943v1), [lu2023machine](https://arxiv.org/abs/2302.04062).

One of the many benefits of synthetic data is that it can be generated
at scale, providing an abundant supply of training and testing data for
AI models. This is particularly valuable in domains where real-world
data is scarce or difficult to obtain (e.g., weather data covering all
conditions [li2023seeds](https://arxiv.org/abs/2306.14066), [lam2023learning](http://arxiv.org/pdf/2402.00059v1)). Second,
synthetic data can be tailored to specific requirements, such as
ensuring a balanced representation of different classes by introducing
controlled variations (e.g., up-weighting low-resource languages in
multilingual language learning [przystupa2019neural](https://doi.org/10.18653/v1/W19-5431)).
This level of control over data characteristics can improve model
performance and generalization. Third, synthetic data can help mitigate
privacy concerns by creating anonymized or de-identified datasets that
do not contain sensitive personal
information [howe2017synthetic](https://arxiv.org/abs/1710.08874), [el2020practical](http://arxiv.org/pdf/2401.06883v1). This is
crucial in domains such as healthcare, where patient privacy is of
utmost importance [dahmen2019synsys](http://arxiv.org/pdf/2304.03243v1), [wei2019generative](http://arxiv.org/pdf/1910.05827v1).

Despite its promise, synthetic data also presents challenges that need
to be addressed. One of them is ensuring the factuality and fidelity of
synthetic data [wood2021fake](https://doi.org/10.1109/ICCV48922.2021.00366), [heusel2017gans](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html), as models
trained on false, hallucinated or biased synthetic data may fail to
generalize to real-world
scenarios [van2023synthetic](http://arxiv.org/pdf/2305.09235v2), [guarnera2020deepfake](http://arxiv.org/pdf/2004.10448v1).
Researchers must develop sophisticated generative models and evaluation
metrics to create synthetic data that accurately reflects the complex
patterns and relationships found in real-world data. Another challenge
is the potential for synthetic data to amplify biases or introduce new
biases if not carefully designed and
validated [barbierato2022methodology](http://arxiv.org/pdf/2203.04462v1), [gupta2021transitioning](https://arxiv.org/abs/2105.04144).
We believe rigorous testing and fairness assessments are necessary to
mitigate these risks.

In this paper, we track the current state of synthetic data research and
discuss current best practices and lessons learned. The rest of the
paper is organized as follows.
Section <a href="#sec:training" data-reference-type="ref"
data-reference="sec:training">[sec:training]</a> provides an overview of
synthetic data generation techniques and their applications in model
training, presenting case studies and empirical evidence.
Section <a href="#sec:evaluation" data-reference-type="ref"
data-reference="sec:evaluation">[sec:evaluation]</a> discusses the
usefulness of synthetic data in evaluation.
Section <a href="#sec:limitation_risks" data-reference-type="ref"
data-reference="sec:limitation_risks">[sec:limitation_risks]</a>
discusses the challenges and limitations of synthetic data, and in
Section <a href="#sec:future" data-reference-type="ref"
data-reference="sec:future">[sec:future]</a> we outline potential
solutions and future research directions.

[^1]: ACT-1: <https://www.adept.ai/blog/act-1>

[^2]: Devin: <https://www.cognition-labs.com/introducing-devin>

# Synthetic Data in Training [sec:training]

Synthetic data, which is generated by mimicking authentic data collected
from the real world, has proven to be an effective and relatively
low-cost alternative of real data. This section explores several notable
domains that leverage synthetic training data.

# Synthetic Data in Evaluation [sec:evaluation]

Synthetic data is widely used in evaluations of different perspectives:

#### Factuality.

AI systems may generate information or responses that are not grounded
in factual knowledge or data, leading to the creation of misleading or
false content, formally known as
*hallucination* [ji2023survey](http://arxiv.org/pdf/2311.05232v1). Factuality evaluation
aims to ensure the consistency of the knowledge in the AI system’s
output with the knowledge provided by its training data and knowledge
base [ji2023survey](http://arxiv.org/pdf/2311.05232v1), [zhang2023siren](https://arxiv.org/abs/2309.01219). Early
statistical-based hallucination evaluation methods relied on n-grams to
directly calculate the overlap of vocabulary between the input and
output content  [dhingra2019handling](https://doi.org/10.18653/v1/P19-1483), [wang2020towards](https://doi.org/10.18653/v1/2020.acl-main.101).
However, these methods have limitations, as they only consider lexical
overlap and do not account for semantics or sentence meaning
[ji2023survey](http://arxiv.org/pdf/2311.05232v1), making them unsuitable for evaluating
more complex forms of hallucination. Subsequent assurance methods
shifted from statistical approaches to model-based methods, which are
more robust compared to token-difference-based methods
[honovich2021q2](https://doi.org/10.18653/v1/2021.emnlp-main.619). While these model-based evaluation
methods are more advanced than their predecessors, they still have
limitations. For example, the models can only output the degree of
hallucination and may struggle to pinpoint specific errors
[falke2019ranking](https://doi.org/10.18653/v1/P19-1213).
[feng-etal-2023-factkb](https://doi.org/10.18653/v1/2023.emnlp-main.59) propose to combine LLMs
generation with random walks on knowledge graphs to generate synthetic
evaluation data for factuality, which is aware of entities and relations
on the graphs. [Wei2024LongformFI](https://api.semanticscholar.org/CorpusID:268724304) created a synthetic
dataset called LongFact for long-form factuality evaluation and used
Google Search as the grounding source and LLM for the automated
judgement, to achieve human-level accuracy but with significally lower
cost [min2023factscore](http://arxiv.org/pdf/2402.05629v3).

#### Safety.

Red teaming is a powerful technique for evaluating the safety and
robustness of AI
models [ganguli2022red](https://arxiv.org/abs/2209.07858), [casper2023explore](https://arxiv.org/abs/2306.09442). By generating
diverse and realistic scenarios designed to elicit unaligned or harmful
outputs [casper2023red](http://arxiv.org/pdf/2302.10894v3), red teaming can expose
vulnerabilities and weaknesses in AI
systems [perez2022red](https://aclanthology.org/2022.emnlp-main.225). For example,
[perez2022discovering](http://arxiv.org/pdf/2211.04476v2) use LMs to generate datasets for
evaluating the behavior of other LMs. They end up producing 154
high-quality datasets which are verified by humans, and discover new
cases of inverse scaling where LMs get worse with size.
[hubinger2024sleeper](https://arxiv.org/abs/2401.05566) leverage synthetic data to trigger
backdoor attacks to LMs at scale; they find LMs can exhibit deceptive
behavior and create a false impression of safety under such attacks, and
standard “safety training” could not remove such deception easily. These
methods demonstrate the feasibility of using AI assistance to scale up
human oversight [bowman2022measuring](https://arxiv.org/abs/2211.03540) over complex
problems and unseen domains.

#### Assisting human evaluation.

Recent studies have shown that in many cases, synthetic judgements from
large-scale LMs (LLMs) can serve as qualified, fast, and low-cost
alternatives to actual human
evaluation [doi:10.1073/pnas.2305016120](https://doi.org/10.1073/pnas.2305016120). Using GPT-4 as
the judge, Alpaca Eval [alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) and MT
Bench [zheng2023judging](https://arxiv.org/pdf/2306.05685) are two popular benchmarks that
measure the comprehensive abilities of LM-based ChatBot. In coding
tasks, synthetic environment is a common choice to aid human evaluation,
as humans can make the assessment more efficiently via actual executions
and analysis on running logs. [gu2024cruxeval](https://arxiv.org/abs/2401.03065) propose
CRUXEval, a code execution reasoning benchmark consisting of 800 Python
functions generated by CodeLLaMA-34B. Similarly,
[liu2024codemind](https://arxiv.org/abs/2402.09664) introduce CodeMind, a framework to
gauge the code reasoning abilities of LLMs on Independent Execution
Reasoning (IER), Dependent Execution Reasoning (DER), and Specification
Reasoning (SR). All these evaluations based on synthetic data show
strong correlation with real human judgements.

# Challenges and Limitations of Synthetic Data [sec:limitation_risks]

While synthetic data offers numerous benefits and applications, it is
crucial to acknowledge and address the potential challenges and
limitations associated with its use. This section delves into three
significant concerns surrounding synthetic data:

#### Misuse of synthetic data might proliferate misinformation.

The potential misuse of synthetic data is a significant concern that
must be addressed to ensure the responsible development of AI systems.
Current AI models become increasingly capable of generating human-like
data ranging from text [reid2024gemini](https://arxiv.org/abs/2403.05530), [team2023gemini](https://arxiv.org/abs/2312.11805),
images [saharia2022photorealistic](http://arxiv.org/pdf/2205.11487v1), [ramesh2022hierarchical](https://arxiv.org/abs/2204.06125),
songs [^1], to even videos (e.g., OpenAI SORA [^2]). This can be
particularly dangerous when synthetic data is used to impersonate real
people, manipulate public opinion, or influence political processes.
Moreover, the dissemination of synthetic data-driven misinformation can
erode trust in legitimate information sources, making it increasingly
difficult for people to distinguish between truth and
falsehood [byman2023deepfakes](http://arxiv.org/pdf/2209.09111v1), [rid2020active](http://arxiv.org/pdf/2005.13466v2). To
mitigate these risks, it is crucial for researchers, developers, and
policymakers to establish clear guidelines and best practices for the
ethical generation and use of synthetic data, including robust
mechanisms for detecting and countering synthetic
misinformation [groh2022deepfake](http://arxiv.org/pdf/2105.06496v2). By proactively
addressing these challenges, we can harness the benefits of synthetic
data while minimizing its potential for harm.

#### Synthetic data might cause ambiguity in AI alignment.

The increasing use of synthetic data in aligning AI models (e.g.,
Constitutional AI [bai2022constitutional](https://arxiv.org/abs/2212.08073)) can introduce
significant ambiguity and uncertainty. The goal of AI alignment is to
ensure that AI systems behave in ways that are aligned with human values
and intentions. However, synthetic data, which is artificially generated
rather than collected from real-world sources, may not accurately
represent the nuances and complexities of human values and
preferences [zhou2024real](https://arxiv.org/abs/2403.05020). This discrepancy can lead to
AI models learning from data that is
biased [feng2023pretraining](https://arxiv.org/abs/2305.08283), [liu2021mitigating](https://ojs.aaai.org/index.php/AAAI/article/view/17744),
ungrounded [liu2022mind](https://arxiv.org/abs/2210.05359), [patel2021mapping](https://openreview.net/forum?id=gJcEM8sxHK), or
misrepresentative of real-world
scenarios [weidinger2021ethical](https://arxiv.org/abs/2112.04359), [ji2023survey](http://arxiv.org/pdf/2311.05232v1). As a
result, AI systems trained on synthetic data may exhibit behaviors that
are misaligned with human expectations, potentially leading to
unintended consequences or even harmful
actions [zou2023universal](https://arxiv.org/abs/2307.15043), [anderljung2023frontier](https://arxiv.org/abs/2307.03718).
Moreover, the ambiguity introduced by synthetic data can make it
challenging to interpret and understand the decision-making processes of
AI models [lightman2023let](https://arxiv.org/abs/2305.20050), further complicating the
task of ensuring alignment. To mitigate these risks, it is crucial for
researchers to carefully consider the limitations and potential
drawbacks of using synthetic data in alignment research and to develop
robust methods for validating and testing AI models trained on such
data.

#### Training with synthetic data makes evaluation decontamination harder.

The use of synthetic data in model training poses significant challenges
to fair evaluation. Evaluation benchmarks are often created by referring
to public text sources, such as coursework websites or forums.
Consequently, it is arguable that all publicly available benchmark test
cases might occasionally be included in the pre-training data of
LLMs [hoffmann2022empirical](http://arxiv.org/pdf/2309.08777v2), [gao2020pile](https://arxiv.org/abs/2101.00027). The use of
synthetic data exacerbates this issue rather than mitigating it.
Although the community has proposed several techniques to detect such
evaluation contamination, such as *min-$k$%
prob* [shi2023detecting](https://arxiv.org/pdf/2310.16789), which checks the probabilities
of $k$ long-tail tokens, these token-level decontamination methods are
inadequate when the model is trained with synthetic data. Synthetic data
might include rephrased versions of the benchmark
data [oren2023proving](https://arxiv.org/abs/2310.17623), [mattern2023membership](https://arxiv.org/abs/2305.18462), rendering
token-level decontamination ineffective. In addition to developing more
advanced evaluation contamination detection techniques, we recommend
that model developers invest in creating and maintaining in-house and
protected evaluation benchmarks. These proprietary benchmarks should be
carefully safeguarded to prevent leakage and ensure the integrity of the
evaluation process.

[^1]: Make songs with Suno AI: <https://app.suno.ai/>

[^2]: OpenAI Sora:
    <https://openai.com/research/video-generation-models-as-world-simulators>

# Directions for Future Work [sec:future]

As the field of synthetic data continues to evolve, there are several
promising directions for future research and development. This section
outlines three key areas that warrant further exploration:

#### Synthetic data scaling.

The impressive performance of many over-trained small language models
(e.g., Mistral series models [jiang2023mistral](https://arxiv.org/abs/2310.06825), and
Gemma series models [team2024gemma](https://arxiv.org/abs/2403.08295), *inter alia*)
demonstrates the necessity of training with large amount of tokens (even
passing the compute-optimal chinchilla
law [rae2021scaling](https://arxiv.org/abs/2112.11446)). However, whether we have similar
conclusions on the training with synthetic data is still an open
question, as the quality of synthetic data may not be as consistent as
real-world data [yu2024large](http://arxiv.org/pdf/2306.15895v2). Future research should
investigate the scaling laws for synthetic data and determine the
optimal balance between the quantity and quality of synthetic samples.
This exploration could help us understand the most effective strategies
for leveraging synthetic data in training large-scale language models,
potentially leading to more efficient and cost-effective
approaches [muennighoff2024scaling](http://arxiv.org/pdf/2202.03371v1).

#### Further improving quality and diversity of synthetic data.

While existing methods for generating synthetic data have shown promise,
there is still room for improvement in terms of creating high-quality,
attributed synthetic samples that closely mimic real-world data. Future
research should focus on developing new advanced techniques (or based on
existing ones such as Generative Adversarial Networks
(GANs) [goodfellow2020generative](http://arxiv.org/pdf/1810.12576v1) or Diffusion
Models [ho2020denoising](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html), *inter alia*) that can control
and manipulate specific attributes of the generated data, enabling the
creation of diverse and customizable synthetic datasets. Additionally,
researchers should explore methods that can incorporate domain-specific
knowledge to ensure the generated data adheres to the underlying
constraints and patterns present in the target domain (e.g., via
Retrieval Augmented Generation
(RAG) [lewis2020retrieval](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html), [borgeaud2022improving](https://proceedings.mlr.press/v162/borgeaud22a.html)) while
maintaining the data quality. By advancing the state-of-the-art in
attributed synthetic data generation, we can unlock new opportunities
for privacy-preserving analysis [assefa2020generating](http://arxiv.org/pdf/2111.12984v1),
and model training across various fields, from healthcare (e.g.,
synthetic medical
images [frid2018synthetic](http://arxiv.org/pdf/1803.01229v1), [wei2019generative](http://arxiv.org/pdf/1910.05827v1)) and
finance (e.g., simulated trading
trajectories [zheng2022ai](http://arxiv.org/pdf/2106.01901v1)) to social
sciences [argyle2023out](http://arxiv.org/pdf/2209.06899v1), [park2023generative](http://arxiv.org/pdf/2208.04024v1) and beyond.

#### Towards high-fidelity and more efficient scalable oversight.

As AI models become increasingly complex and autonomous, it becomes
challenging to monitor and assess their behavior using traditional
oversight methods that rely on human supervision or real-world
data [amodei2016concrete](https://arxiv.org/abs/1606.06565). Future research should
explore the use of synthetic data for high-fidelity scalable oversight
of these advanced systems. Existing methods typically simulate a certain
scenario in social iterations, such as
debate [leike2018scalable](https://arxiv.org/abs/1811.07871),
reflection [zhang2023exploring](https://arxiv.org/abs/2310.02124), or
revisions [liu2023training](https://arxiv.org/abs/2305.16960) to obtain synthetic data,
while new approaches could cover more comprehensive scenarios and more
modalities [sun2023aligning](https://arxiv.org/abs/2309.14525), as recent studies have
found many issues of simulation that only covers a narrowed
down [cheng-etal-2023-compost](https://doi.org/10.18653/v1/2023.emnlp-main.669) or
over-simplified [zhou2024real](https://arxiv.org/abs/2403.05020) scenes. Looking forward,
another growing direction could be how to achieve scalable oversight
more efficiently—given that we have the full control over the synthetic
data generation, we can probably provide more targeted oversights with
less synthetic data. As the need for effective AI governance and
regulation grows, synthetic data will play an increasingly vital role in
enabling more trustworthy scalable oversight mechanisms that promote
robust, accountable, and safe deployment of AI technologies for the
benefit of
society [askell2021general](https://arxiv.org/abs/2112.00861), [bowman2022measuring](https://arxiv.org/abs/2211.03540).

#### The emergent self-improvement capability.

We typically choose the most capable model to generate synthetic data,
as its generation is of higher quality. However, an intriguing question
arises: can a model generate synthetic data that is better than the data
it was trained on, thus enabling it to improve itself? This concept of
self-improvement through synthetic data generation is an exciting avenue
for future research. If a model can generate higher-quality data than
its original training set, it could potentially bootstrap its own
performance by iteratively learning from the enhanced synthetic
data [chen2024selfplay](https://arxiv.org/pdf/2401.01335). This self-improvement
capability could lead to the emergence of more advanced AI systems that
can autonomously refine their skills and knowledge over
time [burns2023weak](https://arxiv.org/abs/2312.09390), [huang-etal-2023-large](https://doi.org/10.18653/v1/2023.emnlp-main.67). Although
recent work shows encouraging progress in this
direction [chen2024selfplay](https://arxiv.org/pdf/2401.01335), [yuan2024self](https://arxiv.org/abs/2401.10020), the upper
bound of self-improvement and the underlying reasons for its
effectiveness remain open questions. Future research should investigate
the theoretical underpinnings and practical feasibility of
self-improvement through synthetic data generation in more diverse
scenarios, examining the necessary conditions, potential limitations,
and associated risks. By unlocking the potential of emergent
self-improvement capabilities, we could enable more adaptable,
efficient, and autonomous learning
processes [lecun2022path](http://arxiv.org/pdf/1409.8027v2).

# Conclusion

Synthetic data has emerged as a promising solution to address the
challenges of data scarcity, privacy concerns, and high costs in AI
development. By generating realistic and diverse datasets, synthetic
data enables the training and evaluation of AI models at scale across
various domains. As we approach human-level or even superhuman-level
intelligence, obtaining synthetic data becomes even more crucial, given
that models need better-than-average-human quality data to progress.
However, ensuring the factuality, fidelity, and lack of bias in
synthetic data remains a critical challenge.

Future research directions on synthetic data could focus on improving
the fidelity and controllability of generative models and developing
standardized evaluation and contamination protocols and tools. We could
also explore the integration of synthetic data with other techniques and
its application in other domains. Despite the challenges, the potential
benefits of synthetic data in advancing AI research are significant. By
leveraging synthetic data responsibly and effectively, we can build more
powerful, inclusive, and trustworthy AI systems that benefit society as
a whole.