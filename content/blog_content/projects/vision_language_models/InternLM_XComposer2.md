---
title: "Focus on the model InternLM_XComposer2 & 4KHD"
themes: ["projects"]
tags: ["VLM", "InternLM", "PLoRA"]
---

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test of the model</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/0.21.1/axios.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/2.0.3/marked.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        input, textarea, button { width: 100%; margin-bottom: 10px; padding: 10px; }
        #result { margin-top: 20px; white-space: pre-wrap; }
        .highlight-button {
            border: 1px solid black;
            background-color: red;
            color: black;
            width: auto;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s, color 0.3s;
            margin-top: 30px;
        }
        .highlight-button:hover {
            background-color: #ff6666; /* Rouge plus clair pour la surbrillance */
            color: black;
        }
        hr.black {
            background-color: black;
        }
        hr.red {
            background-color: red;
        }
    </style>
</head>
<body>
    <hr class="black">
    <hr class="red">
    <hr class="black">
    <h1 style="text-decoration: underline;">Test of the model</h1>
    <input type="file" id="imageInput" accept="image/*">
    <textarea id="questionInput" rows="3" placeholder="Enter your question here" style="border: 1px solid black;"></textarea>
    <label id="cropLabel" for="maxCropsSelect">In how many crops (maximum) do you want to cut the image to do high-resolution analysis? (the more details on the image, the bigger):</label>
        <select id="maxCropsSelect">
            <option value="">Select</option>
    </select>
    <button onclick="submitQuestion()" class="highlight-button">Run</button>
    <div id="result"></div>

    <script>
        const select = document.getElementById('maxCropsSelect');
        for (let i = 1; i <= 40; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = i;
            select.appendChild(option);
        }

        async function submitQuestion() {
            const imageFile = document.getElementById('imageInput').files[0];
            const question = document.getElementById('questionInput').value;
            const maxCrops = document.getElementById('maxCropsSelect').value;
            const resultDiv = document.getElementById('result');

            if (!imageFile || !question) {
                resultDiv.textContent = "Please upload an image, enter a question, and select the max number of crops.";
                return;
            }

            resultDiv.textContent = "Processing...";

            try {
                // Convert image to base64
                const base64Image = await convertToBase64(imageFile);

                // Send request to your API
                const response = await axios.post('YOUR_API_ENDPOINT', {
                    image: base64Image,
                    question: question,
                    maxCrops: parseInt(maxCrops)
                });

                resultDiv.textContent = response.data.answer;
            } catch (error) {
                resultDiv.textContent = "An error occurred: " + error.message;
            }
        }

        function convertToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.readAsDataURL(file);
                reader.onload = () => resolve(reader.result.split(',')[1]);
                reader.onerror = error => reject(error);
            });
        }
    </script>
</body>
</html>

<html>
<head>
    <style>
        hr.black {
            background-color: black;
        }
        hr.red {
            background-color: red;
        }
    </style>
</head>
<body>
    <hr class="black">
    <hr class="red">
    <hr class="black">
    <h1 style="text-decoration: underline;">Explainations</h1>
</body>
</html>

**[InternLM-XComposer2](https://arxiv.org/pdf/2401.16420)** model tackles the visual input through its image encoding process. The **`encode_img`** function utilizes a {{< yellow_colored >}}CLIP{{< /yellow_colored >}} (Contrastive Language-Image Pre-training) Vision Tower to transform the input image into a high-dimensional representation. This process begins with {{< yellow_colored >}}resizing the image to 490x490 pixels{{< /yellow_colored >}} and normalizing its values. The CLIP Vision Tower then divides the image into 14x14 patches, processes these through multiple layers of self-attention and feedforward networks, and ultimately produces a 1024-dimensional vector that encapsulates the image's salient features.

However, this 1024-dimensional representation isn't directly compatible with the language model's expected input. To bridge this gap, the model employs a {{< yellow_colored >}}projection mechanism, which includes the innovative PLoRA (Programmable Low-Rank Adaptation) technique{{< /yellow_colored >}}. This projection {{< yellow_colored >}}expands the image representation from 1024 to 4096 dimensions{{< /yellow_colored >}}, aligning it with the language model's input space. The PLoRA approach allows for efficient fine-tuning and adaptation of this projection, enhancing the model's flexibility in handling visual information. The Partial-LoRA (PLoRA) approach {{< yellow_colored >}}introduces additional weight matrices (WA and WB) specifically for visual tokens{{< /yellow_colored >}}. These matrices serve a crucial purpose in {{< yellow_colored >}}adapting the output of the visual encoder to be compatible with the input requirements of the language model (LLM){{< /yellow_colored >}}.

Next, the model processes the textual input using its **`encode_text`** function. This function leverages the core **`InternLM2Model`** , which {{< yellow_colored >}}embeds each token of the input text into a 4096-dimensional vector{{< /yellow_colored >}}, incorporating positional information through rotary embeddings. The embedded text then passes through 32 layers of transformer blocks, each refining and contextualizing the representation. This process results in a sequence of 4096-dimensional vectors that capture the deep semantic and contextual nuances of the input text.

With both visual and textual inputs now represented in the same 4096-dimensional space, the model {{< yellow_colored >}}concatenates these embeddings{{< /yellow_colored >}}. This concatenation creates a unified representation that incorporates both the visual and textual information, allowing the model to process both modalities simultaneously.

Finally, this combined representation is {{< yellow_colored >}}fed into the InternLM language model for the generation phase{{< /yellow_colored >}}.



```
!pip install transformers==4.33.2 accelerate==0.20.3 typing_extensions==4.12.2 numpy bitsandbytes einops # Important to get those transformers & accelerate versions: otherwise, there is a problem in model.generate method. Indeed, https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py file has been modified in more recent versions (in the _expand_inputs_for_generation method mainly) which generates dimensions incompatibilities between tensors.
!pip install --upgrade --force-reinstall openai # Otherwise I had dependency problems with typing_extensions (one method did not exist)

ckpt_path = 'internlm/internlm-xcomposer2-vl-7b'
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir="./").eval().cuda()
model.tokenizer = tokenizer

im_path = "{your_image}.PNG"
text = '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{your_question}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

with torch.cuda.amp.autocast():
  model_answer = model_gen(model, text, im_path)

def model_gen(model, text, images, need_bos=True, padding=False):
    """
    Generates text based on a combination of text and image inputs.

    Parameters:
    - model: The multimodal model used for processing and generation.
    - text: Input text string.
    - images: List of input images or a single image.
    - need_bos: Boolean indicating whether to add a beginning-of-sequence token.
    - padding: Function for image padding (if required).

    Returns:
    - output_text: Generated text based on the multimodal input.
    """

    # Initialize variables
    pt1 = 0  # Pointer to the current position in the text
    embeds = []  # List to store all embeddings (text and image)
    im_mask = []  # List to store the mask differentiating text from image
    images = [images]  # Ensure images is always a list, even if only one image is provided
    images_loc = [0]  # Location where image(s) should be inserted in the text

    for i, pts in enumerate(images_loc + [len(text)]):
        # Subtext Extraction
        # Purpose: Process the input text in segments, alternating with image processing
        # Importance: Maintains original order and context of text and images in the multimodal input
        subtext = text[pt1:pts]

        # Text Processing
        if need_bos or len(subtext) > 0:
            # Convert text to embeddings
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            # Create mask for text (zeros)
            # Purpose: Differentiates between text and image inputs in combined embedding sequence
            # Explanation: 0s in mask correspond to text tokens
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            need_bos = False

        # Image Processing
        if i < len(images):
            # Image loading and preprocessing
            try:
                image = Image.open(images[i]).convert('RGB')
            except:
                image = images[i].convert('RGB')

            if padding:
                image = padding(image)

            # Visual Processor
            # Purpose: Prepares raw images for analysis by the model
            # Explanation: Performs operations like resizing, normalizing pixel values,
            #              converting to PyTorch tensor, and applying model-specific transformations
            # Note: This is not a tokenizer like for text, but rather a preprocessing step for images
            processed_image = model.vis_processor(image).unsqueeze(0).cuda() # resize the image to 490x490

            # Image Embedding
            # Purpose: Converts processed image into dense vector representation
            # Explanation: Transforms image into series of vectors (usually one per image patch)
            #              that can be processed alongside text embeddings
            image_embeds = model.encode_img(processed_image) # image embedding of 1225x4096 dimension


            embeds.append(image_embeds)
            # Create mask for image (ones)
            # Explanation: 1s in mask correspond to image embeddings
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())

        # Update the pointer to the next position in the text
        pt1 = pts

    # Combining embeddings and masks
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    # Convert mask to boolean tensor for efficient processing
    im_mask = im_mask.bool()

    # Text generation
    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                        temperature=1.0, max_new_tokens=500, num_beams=3,
                        do_sample=False, repetition_penalty=1.0)

    # Post-processing
    # Extract the first (and typically only) generated sequence
    output_token = outputs[0]
    # Remove special tokens (0 or 1) at the beginning if present
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    # Remove everything after '[UNUSED_TOKEN_145]' to match input format
    # Input format: '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()

    return output_text

```




**[InternLM-XComposer2_4KHD](https://arxiv.org/pdf/2404.06512)** model begins by creating an {{< yellow_colored >}}overall view, resizing the entire image to a fixed size of 336 x 336 pixels{{< /yellow_colored >}}, providing a global understanding. Simultaneously, it employs a {{< yellow_colored >}}dynamic image partitioning strategy{{< /yellow_colored >}}; A {{< yellow_colored >}}scale factor is calculated to determine how many 336x336 pixel sub-images can fit within the image while maintaining its aspect ratio{{< /yellow_colored >}}. The image is then resized according to this factor: the {{< yellow_colored >}}new width becomes scale * 336, and the height is adjusted to maintain the original aspect ratio{{< /yellow_colored >}}. Features extracted from these crops are then reassembled into an extended feature map and flattened to obtain the final local features. The model {{< yellow_colored >}}merges the global and local views, inserting a special 'separator' token between them for distinction{{< /yellow_colored >}}. To preserve the 2D structure of the image, a learnable newline token is added at the end of each row of features before flattening.
This approach allows the model to maintain both a comprehensive global understanding and high-resolution local details, while preserving spatial structure information. As a result, the model can effectively process very high-resolution images while retaining an understanding of their overall structure, enabling it to handle resolutions up to 4K HD.


```

!pip install transformers==4.33.2 accelerate==0.20.3 typing_extensions==4.12.2 numpy flash_attn bitsandbytes einops # Important to get those transformers & accelerate versions: otherwise, there is a problem in model.generate method. Indeed, https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py file has been modified in more recent versions (in the _expand_inputs_for_generation method mainly) which generates dimensions incompatibilities between tensors.
!pip install --upgrade --force-reinstall openai # Otherwise I had dependency problems with typing_extensions (one method did not exist)

import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt_path = 'internlm/internlm-xcomposer2-4khd-7b'
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir="./").eval().cuda()
model.tokenizer = tokenizer
samples = json.load(open('data/chartqa/ChartQA Dataset/test/test_human.json'))

def padding_336(b):
    width, height = b.size
    tar = int(np.ceil(height / 336) * 336)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding],
        fill=[255, 255, 255])
    return b

def HD_transform(img, hd_num=25):
    """
    The HD_transform function is designed to prepare images for high-definition processing by the model. Its main purpose is to resize and pad the image in a way that allows it to be effectively divided into a grid of 336x336 pixel sub-images. Here's how it works:

    Image Orientation: It first checks if the image is portrait (height > width). If so, it transposes the image to landscape orientation for consistent processing.
    Scale Calculation: The function calculates a scale factor that determines how many 336x336 sub-images can fit within the image while maintaining its aspect ratio. It does this by incrementing a 'scale' variable until the product of 'scale' and the ceiling of (scale / aspect ratio) exceeds the specified 'hd_num' (default 25).
    Resizing: The image is then resized based on this scale factor. The new width is scale * 336, and the new height is calculated to maintain the original aspect ratio.
    Padding: After resizing, the image is padded to ensure its height is a multiple of 336. This is done by the padding_336 function, which adds white space to the top and bottom of the image as needed.
    Final Adjustment: If the image was transposed initially, it's transposed back to its original orientation.

    The result is an image that can be evenly divided into a grid of 336x336 sub-images, with the total number of sub-images not exceeding 'hd_num'.
    """

    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * 336)
    new_h = int(new_w / ratio)
    img = transforms.functional.resize(
        img,
        [new_h, new_w],
    )
    img = padding_336(img)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)
    return img

class ImageProcessorHD:
    """
    The ImageProcessorHD class is a custom image processor that applies the HD_transform and then normalizes the image for model input. Here's what it does:

    Initialization: It sets up normalization parameters (mean and standard deviation) and creates a transformation pipeline that includes converting the image to a tensor and normalizing it.
    Processing: When called, it performs the following steps:
    a. If given a file path, it opens the image and converts it to RGB.
    b. It applies the HD_transform to the image, preparing it for high-definition processing.
    c. It then applies the transformation pipeline (ToTensor and normalization) to the HD-transformed image.

    The purpose of this processing is to prepare the image in a way that maximizes the model's ability to analyze detailed visual information. By dividing the image into a grid of sub-images, it allows the model to focus on different parts of the image at a high resolution. This is particularly useful for tasks that require understanding complex visual elements, such as charts or graphs in the ChartQA dataset.
    The 'hd_num' parameter (set to 25 in this case) controls the maximum number of sub-images that can be created. This balances between providing high-resolution detail and managing computational resources, as processing more sub-images requires more memory and computation time.
        """
    def __init__(self, resolution=560, hd_num=25):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.normalize = transforms.Normalize(mean, std)
        self.resolution = resolution
        self.hd_num = hd_num
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])
    def __call__(self, item):
        if isinstance(item, str):
            item = Image.open(item).convert('RGB')
            print("image size : ", item.size)
        hd_transformed = HD_transform(item, hd_num=self.hd_num)
        print("hd_transformed : ", hd_transformed.size)
        final_transformed = self.transform(hd_transformed)
        return final_transformed

def preprocess_data(dataset, image_processor):
    processed_dataset = []
    for item in dataset:
        image_path = f'data/chartqa/ChartQA Dataset/test/png/{item["imgname"]}'
        sample = {'text_input': '[UNUSED_TOKEN_146]user\nAnswer the question using a single word or phrase.{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'.format(item['query'])}
        sample['image'] = image_processor(image_path)
        sample['label'] = item["label"]
        processed_dataset.append(sample)
    return processed_dataset

# Initialisation du processeur d'image
image_processor = ImageProcessorHD()

# Prétraitement du dataset
processed_dataset = preprocess_data(samples, image_processor)

# Génération des réponses
def adapted_model_gen(model, text, image):
    """
    Génère une réponse basée sur le texte et l'image d'entrée.
    """
    # Obtenir les embeddings du texte
    text_embeds = model.encode_text(text, add_special_tokens=True)
    print(f"Text embeds shape: {text_embeds.shape}, dtype: {text_embeds.dtype}")
    # Conversion de l'image
    image = image.to(model.device).to(model.dtype)
    print(f"Converted image dtype: {image.dtype}")
    # Encodage de l'image
    image_embeds = model.encode_img(image.unsqueeze(0), hd_num=25)
    print(f"Image embeds shape: {image_embeds.shape}, dtype: {image_embeds.dtype}")
    # Conversion des embeddings de texte si nécessaire
    text_embeds = text_embeds.to(image_embeds.dtype)
    # Concaténation des embeddings
    combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
    print(f"Combined embeds shape: {combined_embeds.shape}, dtype: {combined_embeds.dtype}")
    # Création du masque d'image
    im_mask = torch.cat([
        torch.ones(image_embeds.shape[:2], dtype=torch.bool),
        torch.zeros(text_embeds.shape[:2], dtype=torch.bool)
    ], dim=1).to(model.device)

    # Génération de la réponse
    outputs = model.generate(
        inputs_embeds=combined_embeds,
        im_mask=im_mask,
        temperature=1.0,
        max_new_tokens=500,
        num_beams=3,
        do_sample=False,
        repetition_penalty=1.0
    )

    # Post-traitement
    output_token = outputs[0]
    if output_token[0] in [0, 1]:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
    return output_text


# Utilisation avec le dataset prétraité
def generate_responses(model, processed_dataset):
    responses = []
    for sample in processed_dataset:
        text = sample['text_input']
        image = sample['image'].cuda()  # Déplacer l'image sur GPU si ce n'est pas déjà fait

        response = adapted_model_gen(model, text, image)
        responses.append(response)
    return responses

# Supposons que 'model' est votre modèle chargé et prêt à être utilisé
responses = generate_responses(model, processed_dataset)

# Affichage des réponses (ou tout autre traitement que vous souhaitez faire avec les réponses)
for i, response in enumerate(responses):
    print(f"Question {i+1}: {processed_dataset[i]['text_input']}")
    print(f"Response: {response}")
    print("---")

```