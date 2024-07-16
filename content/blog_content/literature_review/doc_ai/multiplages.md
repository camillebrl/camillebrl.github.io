---
title: "Document Understanding Models handling Multipages Documents"
themes: ["literature_review"]
subthemes: ["doc_ai"]
tags: ["Multipages", "Doc AI"]
toc: false
math: true
---

Most current document processing models often struggle with maintaining context and coherence across multiple pages, leading to fragmented and inaccurate outputs. Some recent models have developed {{< colored >}}techniques to handle a document as a whole, and not page by page{{< /colored >}}. However, these advancements are still in their early stages and face several challenges. For instance, managing long-range dependencies within lengthy documents requires substantial computational resources, and ensuring the coherence and accuracy of information throughout the entire document remains a complex task. We review here some methods allowing multiple page document understanding. 

We classify them in 2 types: those requiring an OCR module that first extracts text from documents, and those not depending on OCR tools:

## {{< posttitle >}}1. OCR-free Models (VLMs) for multipage document handling{{< /posttitle >}}

{{< multipage_ocr_free >}}

## {{< posttitle >}}2. OCR-dependent Models for multipage document handling{{< /posttitle >}}

{{< multipage_ocr >}}