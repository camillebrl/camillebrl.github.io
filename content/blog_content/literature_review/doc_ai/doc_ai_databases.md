---
title: "Doc AI Databases"
themes: ["literature_review"]
subthemes: ["doc_ai"]
tags: ["VLMs", "MLLMs", "Doc AI"]
toc: false
math: true
---

Document AI gathers several tasks, such as Document Classification, Document Information Extraction, Document reconstruction, Document Captioning, Document summarization, and Document Question Answering.

What is notable is that (1) {{< colored >}}Multipage VrDU datasets have recently emerged and are steadily increasing{{< /colored >}}, indicating a shift in the field towards this type of task. These datasets encompass a {{< colored >}}wide variety of document types{{< /colored >}}, including traditional PDF documents, charts, tables, web pages, arXiv papers, diagrams, and application pages, showcasing the diversity of the field. Additionally (2), there is a {{< colored >}}growing emphasis on datasets that cover tasks requiring abstract or numerical reasoning{{< /colored >}}, which demands higher levels of cognitive processing. Finally (3), there is an {{< colored >}}increasing focus on incorporating multiple types of data through various instructions{{< /colored >}} in these datasets.

{{< bigger >}}Question Answering{{< /bigger >}}

Question answering is a prevalent task in natural language processing (NLP) where a {{< colored >}}model must provide a natural language response to a question based on a given passage{{< /colored >}}. This concept {{< colored >}}extends to images, evolving into Visual Question Answering (VQA){{< /colored >}}. Visual Question Answering (VQA) entails answering questions posed in natural language about the content of images, combining computer vision and NLP to interpret and respond to specific queries about visual elements. 

Those VQA datasets are a little tricky: we can always {{< colored >}}wonder who is answering the question for visual question answering: the LLM-only, or the LLM thanks to the visual representation it got as input?{{< /colored >}} So it is really important to take a dataset that can be only answered with the image(s) it contains. One way to evaluate that is to compare the performance on such datasets with the visual representation given to the LLM and without. 

[Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/pdf/2406.16860) presented an {{< colored >}}overview of some dataset and the performance on them with and without visual encoder{{< /colored >}}. This shows the dataset really requiring vision and the ones not:

![](/literature_review/doc_ai/vlm/bench_cat.png)

For instance, AI2D (datasets with questions on figures) displays less than a 5% gap between vision enabled and disabled, suggesting that these benchmarks may not significantly depend on visual input and rather heavily rely on the base LLM.

While {{< colored >}}VQA typically involves a single question and answer per image{{< /colored >}}, {{< colored >}}Conversational VQA (Conv VQA) features a series of questions and answers within a single conversation{{< /colored >}}, enabling multiple interactions about the same image. Also, in VQA, images can vary widely in format: 

- {{< colored >}}Conventional VQA focuses on realistic or synthetic photos{{< /colored >}}, with questions about object recognition, attributes, and spatial relationships within the image.
- {{< colored >}}Scene Text VQA deals with realistic photos that include textual elements within the scene{{< /colored >}}, such as a restaurant sign, focusing on recognizing and understanding text associated with objects in the image.
- {{< colored >}}Chart VQA addresses images of charts, such as bar graphs, line charts, or pie charts{{< /colored >}}, with questions often involving trend recognition, value comparisons, and identifying specific chart properties.
- {{< colored >}}Diagram VQA, also known as Infographic VQA, involves interpreting diagrammatic images that explain processes or relationships{{< /colored >}}. Diagrams are complex graphics comprising a graphic space, various elements, and their interrelationships. Questions in this category typically ask about components and connections depicted in the diagrams.
- {{< colored >}}Document VQA relates to images of various document types, including business PDFs, web pages, forms, receipts, and invoices{{< /colored >}}. VQA tasks with documents resemble Machine Reading Comprehension (MRC) tasks, involving questions about textual content that are answered using text extracted from the document. Document-based VQA incorporates image processing to analyze visual input alongside textual queries.
- {{< colored >}}Multipage VQA involves answering questions that require understanding and integrating information across multiple pages of a document, such as a presentation or report, rather than focusing on a single page per question{{< /colored >}}, which is typical for traditional VQA tasks.
- {{< colored >}}Open-domain VQA involves answering questions based on a broad database of documents, images, tables, texts, and charts{{< /colored >}}. The model includes a {{< colored >}}retriever component that identifies the specific document likely to contain the answer{{< /colored >}} before generating a response.

{{< bigger >}}Image classification{{< /bigger >}}

Image classification is a core task in computer vision that involves {{< colored >}}sorting images into predefined categories based on their content{{< /colored >}}. When applied to documents, this process narrows down to identifying various types of document images, such as charts, diagrams, and other visual elements. For instance, chart image classification focuses on recognizing and categorizing charts extracted from documents into specific predefined categories. This task is complex due to the wide range of chart types, each with distinct features and structures. {{< colored >}}Likewise, diagrams and other document visuals are classified based on their characteristics and the information they convey{{< /colored >}}, thereby improving the automation and comprehension of document content across numerous applications.

{{< bigger >}}Information Extraction{{< /bigger >}}

Information extraction (IE) is a prevalent task in natural language processing (NLP). It involves the {{< colored >}}automatic extraction of structured information from unstructured or semi-structured sources{{< /colored >}} like text documents, web pages, or databases. This process includes {{< colored >}}identifying and extracting specific types of entities{{< /colored >}} (such as names of people, organizations, and locations) and their relationships from the text. When applied to documents, this task begins by identifying relevant elements within the document, such as text blocks, images, charts, or specific visual patterns. These identified elements are then annotated with meaningful labels or metadata that describe their content or function within the document.

{{< bigger >}}Document Reconstruction{{< /bigger >}}

Document reconstruction is a specialized task that shares similarities with optical character recognition (OCR). Its aim is to {{< colored >}}reconstruct an image page of a document in a manner that preserves both its physical appearance and informational content{{< /colored >}}. For textual content, this process involves assembling text blocks of the image and converting the image document in a textual format. For VrD, which contain specific layout, this reconstruction task aims {{< colored >}}converting the image document into structured formats like Markdown or HTML{{< /colored >}}, facilitating clear organization and presentation of the content. This means that the input data are a set of images (screenshots of a scientific paper, of a webapp, of a webpage, …) and the output data are the correspond Markdown / LaTex / HTML code of these screenshots. {{< colored >}}In the case of graphical documents such as charts, reconstruction extends to extracting and reformatting underlying data into non-image formats, such as tables or Markdown{{< /colored >}}.

{{< bigger >}}Captioning{{< /bigger >}}

Captioning is a common task in computer vision that involves {{< colored >}}creating descriptive text for images, charts, diagrams, and tables{{< /colored >}}. This process provides context and summarizes the visual content, making it accessible and understandable.

{{< doc_ai_databases >}}