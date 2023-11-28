---
title: Multimodal Foundation Language Models
date: 2023-11-21
tags: ["Multimodal", "LLMs", "MLLMs"]
math: true
author: ["Camille Barboule"]
categories: ["my_researches"]
---
I will explore here the inner workings of Multimodal Large Language Models (MLLMs), integrating both linguistic and visual data processing. This post will delve into their different architectures, how they process and merge diverse data types, and the principles behind their functionalities.

---

Not all multimodal systems are LLMs: text-to-image models (Midjourney, Stable Diffusion, Dall-E are multimodal but do not have a language model component). Multimodal models can mean several things:
- Input and output are of different modalities (e.g. text-to-image, image-to-text)
- Inputs are multimodal (e.g. a system that can process both text and images)
- Outputs are multimodal (e.g. a system that can generate both text and images) 

Here, we focus on multimodal inputs only. It means that we are focusing on models that can process both text and images. 

As for this type of models, we will call "MLLM", ie "Multimodal Large Language Models", as described in paper [mPLUG-Owl](https://arxiv.org/pdf/2304.14178.pdf), **there exists 2 types of foundation MLLMs**: The first type is about **Systematic-Collaboration-Approaches** (ie models that facilitate the coordination between various tools / models handling the different modalities) and the second type is about **End-To-End-Approaches** (ie unified models that support different modalities).

# <b>1) End-To-End-Approaches</b>
In such approaches, there is a unified architecture handling different modalities, **composed of a Visual Encoder and a LLM as a core**. Most of these approaches use a pretrained vision encoder and a pretrained language models. 

The LLM can be of any type. However, the visual encoder is not random: it must be trained with an alignment of the visual and language modalities, because:
- Alignment ensures that the visual and textual information are semantically coherent. For example, if an image shows a cat, the vision encoder must be able to convey this information to the LLM in a way that the LLM recognizes it as a cat and generates relevant text.
- For training a multimodal model, it is important that data from different modalities are aligned in a way that the model can learn correspondences between them.
## 1.1) Visual encoder
Visual encoders in multimodal models are typically **either based on Convolutional Neural Networks (CNNs)**, like the ResNet architecture, **or on Transformer-based architectures** known as Visual Transformers. These models are designed to process and understand visual information, transforming raw image data into structured, feature-rich representations. 
- CNNs are known for their deep layers of convolution operations. They are adept at extracting hierarchical visual features from images
- Visual Transformers, inspired by the success of Transformers in NLP, apply the self-attention mechanism to process images. They treat an image as a sequence of patches and learn to focus on different parts of an image, capturing complex relationships and contextual information within the image.

Then, **these visual encoders need to be trainined with a text - image alignment**, as explained below. Here are the **3 ways of aligning the 2 modalities**, and thus training the Visual Transformer:
### 1.1.1) Visual encoder having a Bi-Encoder Architecture
A bi-encoder architecture for visual encoders involves **two separate encoders**: one for processing text and another for processing images. These encoders **operate independently to encode their respective inputs** into a shared embedding space, where the **similarity between text and image representations can be measured**. This is how these visual encoders are trained: by **Constrastive Learning**: In the training set, there are 
batches of N pairs of similar image, text. The encoders are jointly trained to maximize the cosine similarity of these pairs, while for the batches of N² - N pairs of in correct pairs (image, text), the encoders are jointly trained to minimize the cosine similarity of these pairs.
- [CLIP](https://arxiv.org/pdf/2103.00020.pdf), for "Contrastive Language-Image Pretraining" was released by OpenAI and is a prime example of a bi-encoder architecture. It uses a Transformer for text and a ResNet (or Vision Transformer) for images. CLIP is trained using a contrastive learning approach, where the model learns to match corresponding text and image pairs among a batch of non-corresponding pairs.
- [FLORENCE](https://arxiv.org/pdf/2111.11432.pdf), introduced by Microsoft, is another model that employs a bi-encoder architecture. It leverages large-scale pretraining across various data sources to learn a universal visual representation.
- [ALIGN](https://arxiv.org/pdf/2102.05918.pdf), standing for "A Large-scale Image and Noisy-Text Embedding", is also a model that utilizes a bi-encoder structure with an EfficientNet-based image encoder and a BERT-based text encoder. It's trained on a large dataset of noisy text-image pairs, making it adept at handling real-world, uncurated data.
### 1.1.2) Visual encoder having an Encoder-Decoder Architecture
An encoder-decoder architecture in visual encoders typically involves an **encoder module to process the image** and a **decoder module to generate corresponding textual output**. This architecture is common in tasks that require the generation of text from images, such as **image captioning**.
- [SimVLM](https://arxiv.org/pdf/2108.10904.pdf), standing for "Simple Vision Language Model" uses an encoder-decoder architecture where the encoder processes visual inputs and the decoder generates textual descriptions. It employs a 'prefix language modeling' objective for pretraining, allowing it to understand and generate natural language descriptions of images effectively.
- [VirTex](https://arxiv.org/pdf/2006.06666.pdf), standing for "Vision and Text", is a model that uses a CNN encoder to process images and a Transformer-based decoder to generate textual descriptions. It's trained on a dataset of images and captions, learning to generate accurate and contextually relevant text descriptions for a given image.
### 1.1.3) Visual encoder having both Architectures: Bi-Encoder + Encoder-Decoder
Some visual encoders take the advantage of both techniques: they are trained using both contrastive learning and cross-modality prediction. This approach combines the **strengths of contrastive methods** like [CLIP](https://arxiv.org/pdf/2103.00020.pdf) and **generative methods** like [SimVLM](https://arxiv.org/pdf/2108.10904.pdf):
- [CoCa](https://arxiv.org/pdf/2205.01917.pdf), standing for "Contrastive Loss and Captioning Loss", is a model from Google that uses this type of dual approaches for the visual encoder. The contrastive loss is applied between unimodal image and text embeddings, while the captioning loss is used on the multimodal decoder outputs, which predict text tokens autoregressively.

## 1.2) Communication between Visual Encoder's output and the LLM
How, there is a **need to make the output of the visual encoder and the LLM communicate**: and this is not direct! For the LLM decoder to effectively understand and interact with data from the vision encoder, it is necessary for the representations generated by the encoder to be in a **format or context that is comprehensible to the LLM**. This means that the visual data must be transformed into a representation that makes sense in the linguistic domain. There are 2 ways of doing this link between the visual encoder output and the LLM:
### 1.2.1) Some models here take a frozen visual encoder, as well as a frozen LLM, and use a projection matrix to make the connection between the output of the vision encoder and the LLM

![](projection_matrix.png)

- [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) fuses vision and language modalities with **gated cross-attention** showing impressive few-shot capabilities (= projection matrix). 
- [BLIP-2](https://arxiv.org/pdf/2301.12597.pdf) uses a **Q-Former** to align the visual features from the frozen visual encoder and large language models.
- [MiniGPT-4](https://arxiv.org/pdf/2304.10592.pdf) uses a **pretrained Q-Former (from BLIP-2)**. Then, they **train a linear layer** on top of this. This linear layer aims at transforming the features outputed from Q-Former into a format that the Vicuna language model can understand and process as if they were linguistic inputs, to make this output interract with the LLM Vicuna. Then, the **linear layer is instruct-tune to adapt the model to visual instructions**.
### 1.2.2) Other models also have a projection matrix to modify the output of the visual encoder, but they then fine-tune the LLM in order to make it understand the visual encoder's output
Thanks to fine-tuning / instruct-tuning, the models are exposed to tasks or datasets that are both text and visual inputs. The LLM is thus fine-tuned to understand and relate the outputs of the visual encoder to their language understanding. This **fine-tuning helps the model learn how to integrate and interpret visual data alongside textual information**:
- [LLaVA](https://arxiv.org/pdf/2304.08485.pdf) utilizes a **straightforward linear layer as projection matrix** to connect the visual encoder's output to the language model. The simplicity of a linear projection layer means that it lacks the sophistication to handle complex relations or nuanced translations between visual features and language concepts. To compensate for this, LLaVA undergoes a fine-tuning process. During **fine-tuning, the language model is specifically trained to better understand and interpret these visual tokens**. This training involves exposing the model to tasks or datasets that include both visual and textual data, enabling it to learn how to effectively combine and make sense of these two types of information.
### 1.2.3) Other models also have a projection matrix, but fine-tune in the pretraining stage the visual encoder in order to make its output adapted to the LLM's input expected
- [mPLUG-Owl](https://arxiv.org/pdf/2304.14178.pdf) has a **projection matrix called "visual abstractor", that it trains**. But it also **fine-tunes its Vision Transformer (ViT-L/14) to adapt its output to the LLM**. This ViT is initialized from the CLIP ViT-L/14 model, fine-tuned using using image-caption pairs from several datasets, such as LAION-400M, COYO-700M, Conceptual Captions, and MSCOCO.

# <b>2) Systematic-Collaboration-Approaches</b>
In such approaches, **LLMs act as the agents in such systems**, and are **prompted to select the appropriate experts and tools** for visual understanding / modality understanding. So these systems employ collaboration with various specialized models to handle tasks beyond the scope of traditional text-based models, particularly in the visual domain. Here are 3 examples of Systematic-Collaboration-Approaches systems:
## 2.1) Systems using a prompt manager between LLM & Visual Foundation Models (VFMs)
- [Visual ChatGPT](https://ar5iv.labs.arxiv.org/html/2303.04671) is a system that **integrates Visual Foundation Models (VFMs) with ChatGPT**, enabling it to interact not only through text but also with images : ChatGPT is not modified, nor are VFMs, and so, to allow ChatGPT to interract with VFMs, there is a **Prompt Manager in between**, which bridges the gap between ChatGPT and VFMs by informing ChatGPT about the capabilities of each VFM, and converting visual information into a language format for ChatGPT, and finally managing histories, priorities, and conflicts among different VFMs. So this process involves multiple steps which are guided by the Prompt Manager: 

![](visual_chatgpt.PNG)

- [HuggingGPT](https://ar5iv.labs.arxiv.org/html/2303.17580) **connects large language models like ChatGPT with various AI models** from the Hugging Face community to solve multimodal AI tasks through 4 stages: **Task Planning, Model Selection, Task Execution, and Response Generation**. In this concept, an LLM acts as a controller, managing and organizing the cooperation of expert models. The LLM first plans a list of tasks based on the user request and then assigns expert models to each task. After the experts execute the tasks, the LLM collects the results and responds to the user: 

![](hugginggpt.PNG)

    - In Task Planning, ChatGPT analyzes user requests and disassembles them into solvable tasks. 
    - During Model Selection, it chooses appropriate models from Hugging Face based on their descriptions. 
    - In Task Execution, these models are invoked and executed, and their results are returned to ChatGPT. 
    - In Response Generation, ChatGPT integrates all model predictions and generates user responses.

## 2.2) Systems modifying LLM to generate specific action-requests tokens calling VFMs
- [MM-REACT](https://ar5iv.labs.arxiv.org/html/2303.11381) is a system that empowers ChatGPT to process and understand multimodal information through action requests. Indeed, in MM-REACT, **ChatGPT is instructed to say specific watchwords in action request if a vision expert is required** to interpret the visual inputs. The action request is composed of textual prompts that represent the expert name called and file names for visual signals. Regular expression matching is applied to parse the expert’s name and the file path, which are then used to call the vision expert (action execution): 

![](mm_react.PNG)
