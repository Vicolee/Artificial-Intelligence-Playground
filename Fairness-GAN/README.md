# Fairness Generative Adversarial Networks (FGANs)

This repository is a project to emulate - as much as possible - within the limits of my understanding the [GAN: Generating Datasets with Fairness Properties Using a Generative Adversarial Network by Sattigeri et. al.](https://krvarshney.github.io/pubs/SattigeriHCV_safeml2019.pdf). The dataset which I chose to train my FGAN on was [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

Note that the FGAN research paper did not explicitly state the hyperparameters used and only briefly glossed over the architecture used. As such, the architecture which I use might not be exactly the same as those used in the Fairness GAN research paper.

The below diagram shows the high-level model of FGAN which I have attempted to emulate.

<img src="https://github.com/Vicolee/Artificial-Intelligence-Playground/blob/master/Fairness-GAN/Images/fgan_model.png" width="600" height="600">

Below shows the architecture I used for the Generator.

<img src="https://github.com/Vicolee/Artificial-Intelligence-Playground/blob/master/Fairness-GAN/Images/generator_model.png" width="600" height="600">

Below shows the architecture I used for the Discriminator.

<img src="https://github.com/Vicolee/Artificial-Intelligence-Playground/blob/master/Fairness-GAN/Images/discriminator_model.png" width="600" height="600">

Below shows the generator and discriminator resblocks used.

<img src="https://github.com/Vicolee/Artificial-Intelligence-Playground/blob/master/Fairness-GAN/Images/resblock.png" width="700" height="600">

The table below shows the parameters used for each layer of my Generator model.

<img src="https://github.com/Vicolee/Artificial-Intelligence-Playground/blob/master/Fairness-GAN/Images/Generator.png" width="600" height="600">

The table below shows the parameters used for each layer of my Discriminator model.

<img src="https://github.com/Vicolee/Artificial-Intelligence-Playground/blob/master/Fairness-GAN/Images/Discriminator.png" width="600" height="600">



## References:
Fairness Generative Adversarial Network - https://arxiv.org/pdf/1805.09910.pdf

spectral_normalization.py - https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py

CelebA dataset - http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
