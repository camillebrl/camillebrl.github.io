---
title: "Image encoding"
date: 2023-11-17 18:32:17+00:00
themes: ["literature_review"]
subthemes: ["vision_models"]
tags: ["CNN", "ViT", "Computer Vision", "VLMs"]
---
# How to represent a visual input, ie how to encode an image?
An image can be represented either in the spatial domain or in the frequency domain
## Spatial-domain image representation
TODO
## Frequency-domain image representation
TODO




The representation of an image can be either pyramids or filter ##### to review TODO

# Goal is the come up with the function with the best representation possible of the image
Depending on the representation chosen, it make some operations easy and others hard.

## A good representation of an image makes a good compression of it
A first main goal of image representation (encoder) is to compress the image efficiently. The compression of an image allows to capture its essential characteristics. It is a main goal for image representation (encoder) mainly because "the simplest is most likely to be true" ([Occam's razor](link_to_occam_razor)), but also because compression allows less memory to handle the image, which might be useful.

### Autoencoders to learn to compress the image
An autoencoder is a function that maps data back to itself via low-dimensional representional bottleneck. 

### Contrastive Learning to learn to compress the image

## A good representation of an image makes a good prediction from the representation
Another main goal of image representation (encoder) is to get a representation that allows to make good predictions. A prediction can be about imputation (predicting / filing missing pixels), about the future, the past, cause and effects, etc. Most representation-learning methods in vision are about learning compressed encodings of the image that are also predictive.

