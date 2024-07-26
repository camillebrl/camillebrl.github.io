---
title: "Image generation"
date: 2024-07-01
themes: ["literature_review"]
subthemes: ["vision_models"]
tags: ["diffusion", "GAN", "ViT", "generation"]
---




{{< bigger >}}Diffusion models{{< /bigger >}}

Diffusion models are a class of models used in supervised learning to generate images from noise. The fundamental idea is to gradually corrupt a clean image by adding noise, and then train a model to reverse this process by progressively denoising the image.

![](/literature_review/computer_vision/diffusion1.png)
*Forward diffusion process, from [Antonio Torralba et al.'s book](https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/), 485*

{{< formula >}}

In the forward diffusion process, a clean image \( x_0 \) is corrupted by iteratively adding noise to obtain a sequence of images \((x_0, x_1, \dots, x_T)\). At each step \( t \), the image becomes noisier according to the formula:

\[ x_t = \sqrt{(1 - \beta_t)} x_{t-1} + \sqrt{\beta_t} \epsilon_t \]

where \(\epsilon_t \sim \mathcal{N}(0, I)\) is Gaussian noise and \(\beta_t\) is a coefficient controlling the amount of noise added at each step. After \( T \) steps, the image \( x_T \) resembles pure noise.

{{< /formula >}}

![](/literature_review/computer_vision/diffusion2.png)
*Reverse diffusion process, from [Antonio Torralba et al.'s book](https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/), 485*

{{< formula >}}

The goal of diffusion models is to learn to reverse this process. For this, a neural network \( f_\theta \) is used to predict the clean image at each step from the noisy image:

\[ x_{t-1} = f_\theta(x_t) \]

The model is trained using supervised examples where the noisy image \( x_t \) is mapped to the slightly less noisy image \( x_{t-1} \). This process is repeated for all steps until the clean image \( x_0 \) is obtained.

The model \( f_\theta \) is called a "denoiser." It learns to remove a bit of noise at each step. By applying this denoiser iteratively, starting from pure noise, the process should converge to a less noisy image resembling one of our training examples.

To formalize, if we follow the noise process:

\[ x_t = \sqrt{(1 - \beta_t)} x_{t-1} + \sqrt{\beta_t} \epsilon_t \]

where \(\epsilon_t \sim \mathcal{N}(0, I)\) and \(\beta_t\) is a noise coefficient, then the diffusion model learns to reverse this process by:

\[ x_{t-1} = f_\theta(x_t) \]

{{< /formula >}}

This diffusion model is trained using supervised examples from the forward diffusion process. Once trained, the model can be used to generate images starting from pure noise and applying the denoising model iteratively.