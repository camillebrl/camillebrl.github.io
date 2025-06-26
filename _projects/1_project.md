---
layout: page
title: DocCustomKIE
description: An end-to-end pipeline for custom Key Information Extraction from documents
img: assets/img/doccustomkie/doccustomkie.png
importance: 1
category: work
related_publications: false
---

## A Complete Pipeline for Document Information Extraction

DocCustomKIE is an end-to-end solution for extracting key information from any type of document. The project provides a complete workflow from data annotation to model deployment, requiring minimal data and no GPU for training or inference.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/doccustomkie/doc_kie_annotation.png" title="annotation interface" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/doccustomkie/doc_kie_training.png" title="training process" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/doccustomkie/doc_kie_inference.png" title="inference results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    From left to right: intuitive annotation interface, automated training pipeline, and inference with extracted entities visualization.
</div>

## Key Features

The pipeline offers several innovative features that make document processing accessible and efficient:

- **📄 Minimal Data Requirements**: Only about 10 documents needed for good performance
- **🎨 Intuitive Labeling Interface**: Create custom labels for any information you need to extract
- **🔍 Advanced OCR**: Intelligent region detection using Sobel gradients for optimal text recognition
{% include figure.liquid loading="eager" path="assets/img/doccustomkie/sobel_gradient_detection.png" title="sobel gradient detection" class="img-fluid rounded z-depth-1" %}
- **🔄 Automatic Data Augmentation**: Generate labeled variants to strengthen model robustness
- **🤖 Lightweight Transformer Fine-tuning**: Adapt Microsoft's LayoutLMv3-base without GPU requirements
- **🖥️ Modular Inference Interface**: Visualize extracted information or customize the pipeline for your needs

## Technical Approach

The system employs a sophisticated multi-stage process:

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        <div class="img-fluid rounded z-depth-1" style="background: #f8f9fa; padding: 20px;">
            <svg viewBox="0 0 1200 800" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: auto;">
                <!-- Background -->
                <rect width="1200" height="800" fill="#f8f9fa"/>
                
                <!-- Title -->
                <text x="600" y="40" font-family="Arial, sans-serif" font-size="28" font-weight="bold" text-anchor="middle" fill="#2c3e50">DocCustomKIE Pipeline Architecture</text>
                
                <!-- Stage 1: Document Input & OCR -->
                <g transform="translate(50, 100)">
                    <!-- Container -->
                    <rect x="0" y="0" width="300" height="600" rx="10" fill="#e8f4f8" stroke="#3498db" stroke-width="2"/>
                    <text x="150" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle" fill="#2c3e50">1. Document Processing</text>
                    
                    <!-- Document Input -->
                    <rect x="50" y="60" width="200" height="80" rx="5" fill="#fff" stroke="#3498db" stroke-width="2"/>
                    <text x="150" y="90" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Document Upload</text>
                    <text x="150" y="110" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(PDF/PNG/JPG)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 140 L 150 170" stroke="#3498db" stroke-width="2" fill="none" marker-end="url(#arrowblue)"/>
                    
                    <!-- Sobel Gradient -->
                    <rect x="50" y="170" width="200" height="80" rx="5" fill="#fff" stroke="#3498db" stroke-width="2"/>
                    <text x="150" y="200" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Sobel Gradient</text>
                    <text x="150" y="220" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">Block Detection</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 250 L 150 280" stroke="#3498db" stroke-width="2" fill="none" marker-end="url(#arrowblue)"/>
                    
                    <!-- Block Processing -->
                    <rect x="50" y="280" width="200" height="80" rx="5" fill="#fff" stroke="#3498db" stroke-width="2"/>
                    <text x="150" y="310" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Block Processing</text>
                    <text x="150" y="330" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Upscale 3x + Sharpen)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 360 L 150 390" stroke="#3498db" stroke-width="2" fill="none" marker-end="url(#arrowblue)"/>
                    
                    <!-- Tesseract OCR -->
                    <rect x="50" y="390" width="200" height="80" rx="5" fill="#fff" stroke="#3498db" stroke-width="2"/>
                    <text x="150" y="420" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Tesseract OCR</text>
                    <text x="150" y="440" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Per Block)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 470 L 150 500" stroke="#3498db" stroke-width="2" fill="none" marker-end="url(#arrowblue)"/>
                    
                    <!-- Bbox Reconstruction -->
                    <rect x="50" y="500" width="200" height="80" rx="5" fill="#fff" stroke="#3498db" stroke-width="2"/>
                    <text x="150" y="530" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Bbox Reconstruction</text>
                    <text x="150" y="550" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Original Scale)</text>
                </g>
                
                <!-- Arrow to Stage 2 -->
                <path d="M 350 400 L 420 400" stroke="#2c3e50" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
                
                <!-- Stage 2: Annotation & Training -->
                <g transform="translate(420, 100)">
                    <!-- Container -->
                    <rect x="0" y="0" width="300" height="600" rx="10" fill="#fef9e7" stroke="#f39c12" stroke-width="2"/>
                    <text x="150" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle" fill="#2c3e50">2. Annotation & Training</text>
                    
                    <!-- Manual Annotation -->
                    <rect x="50" y="60" width="200" height="80" rx="5" fill="#fff" stroke="#f39c12" stroke-width="2"/>
                    <text x="150" y="90" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Manual Annotation</text>
                    <text x="150" y="110" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Custom Labels)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 140 L 150 170" stroke="#f39c12" stroke-width="2" fill="none" marker-end="url(#arroworange)"/>
                    
                    <!-- BIO Tagging -->
                    <rect x="50" y="170" width="200" height="80" rx="5" fill="#fff" stroke="#f39c12" stroke-width="2"/>
                    <text x="150" y="200" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">BIO Tagging</text>
                    <text x="150" y="220" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(B-/I-/O Format)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 250 L 150 280" stroke="#f39c12" stroke-width="2" fill="none" marker-end="url(#arroworange)"/>
                    
                    <!-- Data Augmentation -->
                    <rect x="50" y="280" width="200" height="80" rx="5" fill="#fff" stroke="#f39c12" stroke-width="2"/>
                    <text x="150" y="310" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Data Augmentation</text>
                    <text x="150" y="330" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Noise, Rotation, etc.)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 360 L 150 390" stroke="#f39c12" stroke-width="2" fill="none" marker-end="url(#arroworange)"/>
                    
                    <!-- JSONL Export -->
                    <rect x="50" y="390" width="200" height="80" rx="5" fill="#fff" stroke="#f39c12" stroke-width="2"/>
                    <text x="150" y="420" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">JSONL Export</text>
                    <text x="150" y="440" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Normalized Bboxes)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 470 L 150 500" stroke="#f39c12" stroke-width="2" fill="none" marker-end="url(#arroworange)"/>
                    
                    <!-- LayoutLMv3 Training -->
                    <rect x="50" y="500" width="200" height="80" rx="5" fill="#fff" stroke="#f39c12" stroke-width="2"/>
                    <text x="150" y="530" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">LayoutLMv3 Fine-tuning</text>
                    <text x="150" y="550" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Token Classification)</text>
                </g>
                
                <!-- Arrow to Stage 3 -->
                <path d="M 720 400 L 790 400" stroke="#2c3e50" stroke-width="3" fill="none" marker-end="url(#arrow)"/>
                
                <!-- Stage 3: Inference -->
                <g transform="translate(790, 100)">
                    <!-- Container -->
                    <rect x="0" y="0" width="300" height="600" rx="10" fill="#ebfaf0" stroke="#27ae60" stroke-width="2"/>
                    <text x="150" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle" fill="#2c3e50">3. Inference</text>
                    
                    <!-- New Document -->
                    <rect x="50" y="60" width="200" height="80" rx="5" fill="#fff" stroke="#27ae60" stroke-width="2"/>
                    <text x="150" y="90" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">New Document</text>
                    <text x="150" y="110" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Same OCR Pipeline)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 140 L 150 170" stroke="#27ae60" stroke-width="2" fill="none" marker-end="url(#arrowgreen)"/>
                    
                    <!-- Model Prediction -->
                    <rect x="50" y="170" width="200" height="80" rx="5" fill="#fff" stroke="#27ae60" stroke-width="2"/>
                    <text x="150" y="200" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Model Prediction</text>
                    <text x="150" y="220" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Label per Token)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 250 L 150 280" stroke="#27ae60" stroke-width="2" fill="none" marker-end="url(#arrowgreen)"/>
                    
                    <!-- BIO Merging -->
                    <rect x="50" y="280" width="200" height="80" rx="5" fill="#fff" stroke="#27ae60" stroke-width="2"/>
                    <text x="150" y="310" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">BIO Merging</text>
                    <text x="150" y="330" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Entity Reconstruction)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 360 L 150 390" stroke="#27ae60" stroke-width="2" fill="none" marker-end="url(#arrowgreen)"/>
                    
                    <!-- Post-processing -->
                    <rect x="50" y="390" width="200" height="80" rx="5" fill="#fff" stroke="#27ae60" stroke-width="2"/>
                    <text x="150" y="420" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Post-processing</text>
                    <text x="150" y="440" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Clean Duplicates)</text>
                    
                    <!-- Arrow -->
                    <path d="M 150 470 L 150 500" stroke="#27ae60" stroke-width="2" fill="none" marker-end="url(#arrowgreen)"/>
                    
                    <!-- Output -->
                    <rect x="50" y="500" width="200" height="80" rx="5" fill="#fff" stroke="#27ae60" stroke-width="2"/>
                    <text x="150" y="530" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#2c3e50">Structured Output</text>
                    <text x="150" y="550" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(JSON + Visualization)</text>
                </g>
                
                <!-- Future Feedback Loop -->
                <g transform="translate(570, 650)">
                    <rect x="-120" y="0" width="240" height="60" rx="30" fill="#e8d5f2" stroke="#9b59b6" stroke-width="2" stroke-dasharray="5,5"/>
                    <text x="0" y="25" font-family="Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle" fill="#2c3e50">Future: Feedback Loop</text>
                    <text x="0" y="45" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#7f8c8d">(Human Corrections → Retraining)</text>
                </g>
                
                <!-- Feedback Arrow -->
                <path d="M 940 580 C 940 650, 570 650, 570 580" stroke="#9b59b6" stroke-width="2" fill="none" stroke-dasharray="5,5" marker-end="url(#arrowpurple)"/>
                
                <!-- Arrow definitions -->
                <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                        <path d="M0,0 L0,6 L9,3 z" fill="#2c3e50"/>
                    </marker>
                    <marker id="arrowblue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                        <path d="M0,0 L0,6 L9,3 z" fill="#3498db"/>
                    </marker>
                    <marker id="arroworange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                        <path d="M0,0 L0,6 L9,3 z" fill="#f39c12"/>
                    </marker>
                    <marker id="arrowgreen" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                        <path d="M0,0 L0,6 L9,3 z" fill="#27ae60"/>
                    </marker>
                    <marker id="arrowpurple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                        <path d="M0,0 L0,6 L9,3 z" fill="#9b59b6"/>
                    </marker>
                </defs>
            </svg>
        </div>
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        <svg viewBox="0 0 600 700" xmlns="http://www.w3.org/2000/svg">
            <!-- Background -->
            <rect width="600" height="700" fill="#f8f9fa"/>
            
            <!-- Title -->
            <text x="300" y="40" font-family="Arial, sans-serif" font-size="24" font-weight="bold" text-anchor="middle" fill="#2c3e50">BIO Tagging Example</text>
            
            <!-- Example Document Section -->
            <g transform="translate(50, 80)">
                <!-- Document Container -->
                <rect x="0" y="0" width="500" height="180" rx="8" fill="#fff" stroke="#ddd" stroke-width="2"/>
                <text x="250" y="25" font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#2c3e50">Original Document Text</text>
                
                <!-- Document Text Lines -->
                <text x="20" y="60" font-family="Courier New, monospace" font-size="14" fill="#333">Patient: John Smith</text>
                <text x="20" y="85" font-family="Courier New, monospace" font-size="14" fill="#333">Born: 15/03/1985 in Paris, France</text>
                <text x="20" y="110" font-family="Courier New, monospace" font-size="14" fill="#333">Address: 123 Main Street, Lyon</text>
                <text x="20" y="135" font-family="Courier New, monospace" font-size="14" fill="#333">Doctor: Dr. Marie Dupont</text>
            </g>
            
            <!-- Arrow Down -->
            <path d="M 300 260 L 300 290" stroke="#2c3e50" stroke-width="3" fill="none" marker-end="url(#arrowdown)"/>
            
            <!-- OCR Tokenization -->
            <g transform="translate(50, 300)">
                <rect x="0" y="0" width="500" height="100" rx="8" fill="#e8f4f8" stroke="#3498db" stroke-width="2"/>
                <text x="250" y="25" font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#2c3e50">After OCR Tokenization</text>
                
                <!-- Tokens -->
                <g transform="translate(20, 45)">
                <!-- Row 1 -->
                <rect x="0" y="0" width="60" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="30" y="20" font-family="monospace" font-size="12" text-anchor="middle">Patient:</text>
                
                <rect x="70" y="0" width="40" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="90" y="20" font-family="monospace" font-size="12" text-anchor="middle">John</text>
                
                <rect x="120" y="0" width="50" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="145" y="20" font-family="monospace" font-size="12" text-anchor="middle">Smith</text>
                
                <rect x="180" y="0" width="40" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="200" y="20" font-family="monospace" font-size="12" text-anchor="middle">Born:</text>
                
                <rect x="230" y="0" width="80" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="270" y="20" font-family="monospace" font-size="12" text-anchor="middle">15/03/1985</text>
                
                <rect x="320" y="0" width="25" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="332" y="20" font-family="monospace" font-size="12" text-anchor="middle">in</text>
                
                <rect x="355" y="0" width="50" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="380" y="20" font-family="monospace" font-size="12" text-anchor="middle">Paris,</text>
                
                <rect x="415" y="0" width="50" height="30" rx="4" fill="#fff" stroke="#3498db"/>
                <text x="440" y="20" font-family="monospace" font-size="12" text-anchor="middle">France</text>
                </g>
            </g>
            
            <!-- Arrow Down -->
            <path d="M 300 400 L 300 430" stroke="#2c3e50" stroke-width="3" fill="none" marker-end="url(#arrowdown)"/>
            
            <!-- BIO Tagged Result -->
            <g transform="translate(50, 440)">
                <rect x="0" y="0" width="500" height="200" rx="8" fill="#fef9e7" stroke="#f39c12" stroke-width="2"/>
                <text x="250" y="25" font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#2c3e50">BIO Tagged Tokens</text>
                
                <!-- Tagged Tokens -->
                <g transform="translate(20, 45)">
                <!-- Token 1: Patient: -->
                <g>
                    <rect x="0" y="0" width="60" height="30" rx="4" fill="#e0e0e0" stroke="#999"/>
                    <text x="30" y="20" font-family="monospace" font-size="11" text-anchor="middle">Patient:</text>
                    <rect x="0" y="35" width="60" height="20" rx="3" fill="#666" stroke="#666"/>
                    <text x="30" y="48" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="white">O</text>
                </g>
                
                <!-- Token 2: John -->
                <g transform="translate(70, 0)">
                    <rect x="0" y="0" width="40" height="30" rx="4" fill="#ffcccc" stroke="#ff6666"/>
                    <text x="20" y="20" font-family="monospace" font-size="11" text-anchor="middle">John</text>
                    <rect x="0" y="35" width="40" height="20" rx="3" fill="#ff6666" stroke="#ff6666"/>
                    <text x="20" y="48" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">B-NAME</text>
                </g>
                
                <!-- Token 3: Smith -->
                <g transform="translate(120, 0)">
                    <rect x="0" y="0" width="50" height="30" rx="4" fill="#ffcccc" stroke="#ff6666"/>
                    <text x="25" y="20" font-family="monospace" font-size="11" text-anchor="middle">Smith</text>
                    <rect x="0" y="35" width="50" height="20" rx="3" fill="#ff9999" stroke="#ff9999"/>
                    <text x="25" y="48" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">I-NAME</text>
                </g>
                
                <!-- Token 4: Born: -->
                <g transform="translate(180, 0)">
                    <rect x="0" y="0" width="40" height="30" rx="4" fill="#e0e0e0" stroke="#999"/>
                    <text x="20" y="20" font-family="monospace" font-size="11" text-anchor="middle">Born:</text>
                    <rect x="0" y="35" width="40" height="20" rx="3" fill="#666" stroke="#666"/>
                    <text x="20" y="48" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="white">O</text>
                </g>
                
                <!-- Token 5: 15/03/1985 -->
                <g transform="translate(230, 0)">
                    <rect x="0" y="0" width="80" height="30" rx="4" fill="#ccffcc" stroke="#66cc66"/>
                    <text x="40" y="20" font-family="monospace" font-size="11" text-anchor="middle">15/03/1985</text>
                    <rect x="0" y="35" width="80" height="20" rx="3" fill="#66cc66" stroke="#66cc66"/>
                    <text x="40" y="48" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">B-DATE</text>
                </g>
                
                <!-- Token 6: in -->
                <g transform="translate(320, 0)">
                    <rect x="0" y="0" width="25" height="30" rx="4" fill="#e0e0e0" stroke="#999"/>
                    <text x="12" y="20" font-family="monospace" font-size="11" text-anchor="middle">in</text>
                    <rect x="0" y="35" width="25" height="20" rx="3" fill="#666" stroke="#666"/>
                    <text x="12" y="48" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="white">O</text>
                </g>
                
                <!-- Token 7: Paris, -->
                <g transform="translate(355, 0)">
                    <rect x="0" y="0" width="50" height="30" rx="4" fill="#ccccff" stroke="#6666ff"/>
                    <text x="25" y="20" font-family="monospace" font-size="11" text-anchor="middle">Paris,</text>
                    <rect x="0" y="35" width="50" height="20" rx="3" fill="#6666ff" stroke="#6666ff"/>
                    <text x="25" y="48" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">B-CITY</text>
                </g>
                
                <!-- Token 8: France -->
                <g transform="translate(415, 0)">
                    <rect x="0" y="0" width="50" height="30" rx="4" fill="#ccccff" stroke="#6666ff"/>
                    <text x="25" y="20" font-family="monospace" font-size="11" text-anchor="middle">France</text>
                    <rect x="0" y="35" width="50" height="20" rx="3" fill="#9999ff" stroke="#9999ff"/>
                    <text x="25" y="48" font-family="Arial" font-size="10" font-weight="bold" text-anchor="middle" fill="white">I-CITY</text>
                </g>
                </g>
                
                <!-- Second row example -->
                <g transform="translate(20, 115)">
                <!-- Doctor: -->
                <g>
                    <rect x="0" y="0" width="50" height="30" rx="4" fill="#e0e0e0" stroke="#999"/>
                    <text x="25" y="20" font-family="monospace" font-size="11" text-anchor="middle">Doctor:</text>
                    <rect x="0" y="35" width="50" height="20" rx="3" fill="#666" stroke="#666"/>
                    <text x="25" y="48" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="white">O</text>
                </g>
                
                <!-- Dr. -->
                <g transform="translate(60, 0)">
                    <rect x="0" y="0" width="30" height="30" rx="4" fill="#ffccff" stroke="#ff66ff"/>
                    <text x="15" y="20" font-family="monospace" font-size="11" text-anchor="middle">Dr.</text>
                    <rect x="0" y="35" width="30" height="20" rx="3" fill="#ff66ff" stroke="#ff66ff"/>
                    <text x="15" y="48" font-family="Arial" font-size="9" font-weight="bold" text-anchor="middle" fill="white">B-DOC</text>
                </g>
                
                <!-- Marie -->
                <g transform="translate(100, 0)">
                    <rect x="0" y="0" width="45" height="30" rx="4" fill="#ffccff" stroke="#ff66ff"/>
                    <text x="22" y="20" font-family="monospace" font-size="11" text-anchor="middle">Marie</text>
                    <rect x="0" y="35" width="45" height="20" rx="3" fill="#ff99ff" stroke="#ff99ff"/>
                    <text x="22" y="48" font-family="Arial" font-size="9" font-weight="bold" text-anchor="middle" fill="white">I-DOC</text>
                </g>
                
                <!-- Dupont -->
                <g transform="translate(155, 0)">
                    <rect x="0" y="0" width="50" height="30" rx="4" fill="#ffccff" stroke="#ff66ff"/>
                    <text x="25" y="20" font-family="monospace" font-size="11" text-anchor="middle">Dupont</text>
                    <rect x="0" y="35" width="50" height="20" rx="3" fill="#ff99ff" stroke="#ff99ff"/>
                    <text x="25" y="48" font-family="Arial" font-size="9" font-weight="bold" text-anchor="middle" fill="white">I-DOC</text>
                </g>
                </g>
            </g>
            
            <!-- Legend -->
            <g transform="translate(50, 660)">
                <text x="0" y="0" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#2c3e50">Legend:</text>
                <text x="60" y="0" font-family="Arial, sans-serif" font-size="12" fill="#333">
                <tspan font-weight="bold">B-</tspan> = Beginning of entity
                </text>
                <text x="200" y="0" font-family="Arial, sans-serif" font-size="12" fill="#333">
                <tspan font-weight="bold">I-</tspan> = Inside entity
                </text>
                <text x="320" y="0" font-family="Arial, sans-serif" font-size="12" fill="#333">
                <tspan font-weight="bold">O</tspan> = Outside (not an entity)
                </text>
            </g>
            
            <!-- Arrow definition -->
            <defs>
                <marker id="arrowdown" markerWidth="10" markerHeight="10" refX="5" refY="5" orient="auto">
                <path d="M0,0 L5,10 L10,0 L5,5 z" fill="#2c3e50"/>
                </marker>
            </defs>
            </svg>
    </div>
</div>
<div class="caption">
    System architecture showing the complete pipeline, and example of BIO tagging for multi-token entities.
</div>

### 1. Intelligent OCR with Block Detection

The system uses Sobel gradient analysis to intelligently detect text regions in documents. Each region is then upscaled (3x) and sharpened before applying Tesseract OCR, resulting in significantly improved text recognition accuracy compared to standard whole-image OCR.

### 2. BIO Tagging for Entity Recognition

The annotation interface automatically handles multi-token entities using the BIO (Beginning-Inside-Outside) tagging scheme. When annotating "John Smith Ltd.", the system generates:
- `B-COMPANY` for "John"
- `I-COMPANY` for "Smith"  
- `I-COMPANY` for "Ltd."

This allows the model to learn complex entity boundaries and handle fragmented text naturally.

### 3. Data Augmentation Pipeline

After annotation, the system automatically generates augmented versions of your documents with various transformations:
- Contrast and brightness adjustments
- Gaussian noise addition
- Slight rotations and perspective changes
- All while preserving and transferring the original annotations using Levenshtein distance matching

## Workflow
annotate.py → layoutlmv3_ft.py → inference.py

1. **Annotation Phase**: Upload documents, run OCR, and annotate key information with custom labels
2. **Training Phase**: Fine-tune LayoutLMv3 on your annotated data (typically 2-4 hours on CPU)
3. **Inference Phase**: Process new documents and extract structured information

## Future Development: Feedback Loop

The next major enhancement will introduce a feedback loop in the inference pipeline, allowing continuous improvement of the model through human corrections:

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/doccustomkie/doc_kie_feedback_loop.jpg" title="feedback loop diagram" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Planned feedback loop architecture: predictions → human corrections → model retraining with incremental learning.
</div>

### Planned Features:

1. **Interactive Correction Interface**: Review and correct model predictions directly in the inference UI
2. **Incremental Learning**: Retrain the model on corrected data without forgetting previous knowledge
3. **Active Learning**: Prioritize uncertain predictions for human review
4. **Version Control**: Track model improvements over time with performance metrics

This feedback mechanism will enable the system to continuously improve its accuracy on your specific document types, creating a truly adaptive solution.

## Use Cases

DocCustomKIE has been successfully applied to various document types:
- Medical records and prescriptions
- Administrative forms and certificates
- Financial documents and invoices
- Legal contracts and agreements
- Any structured or semi-structured documents requiring information extraction

## Technical Stack

- **OCR**: Tesseract with custom pre-processing pipeline
- **Model**: Microsoft LayoutLMv3-base (multimodal transformer)
- **Framework**: PyTorch with Hugging Face Transformers
- **Interface**: Flask web application
- **Data Format**: JSONL for annotations, supporting standard KIE datasets

The system is designed to be lightweight and accessible, running efficiently on standard hardware without specialized GPU requirements.