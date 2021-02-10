Project 5: Generating Images with Normalizing Flows

## Main Idea
This project implements and experiments with different flow based models and compares the result with the GAN generated images. The flow based 
models are combined with the GAN to get FlowGAN which has been trained on the following loss objectives
- Adversarial training
- Likelihood based training
- Hybrid training

Image generation experiements are also conducted with the, along with the training
- vanilla DCGAN 
- RealNVP
- Glow

To check the modelling of different data, 2d data is also used as an alternative to the complex image data like MNIST/ CIFAR10. 

Different experiments we have done
- OOD detection
- Discriminator score evaluation
- Running FID score plots
- Comparison of different model training using Likelihoods
- Comparison of images generated with FID and IS score.

## Experiemented models

- [x] FlowGAN (3 different loss objectives)
- [x] FlowGAN without WGAN
- [x] RealNVP
- [x] Vanilla DCGAN
- [x] Glow
- [x] 2d data

## Dataset

We have used the [MNIST](http://yann.lecun.com/exdb/mnist/) and the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets for most of our experiments. The 2d data experiments are done with the synthetic data.

## Metrics
We have used the Inception Score and Frechet Inception score to report the quality of the generated images from the different models. Detailed comparisons and results can be found in the table reported in [wiki update page](https://wiki.tum.de/pages/viewpage.action?pageId=718667942).

## Developer Notes
There are following folder in this repository denoting different groups of experiments conducted. 

|Folder|Used for|
|----------|:-------------:|
|2d data Code | The experiments with synthetic 2d data have been done here. |
|dcgan	| The vanilla DCGAN is used to start our initial experiments. |
|flowgan | Final models with the WGAN GP. Same code is used for without WGAN experiments as well with relevant changes.|
|glow | A different flow based model used separately. Comparable with RealNVP.|
|ms	| Sample inception score calculation source code. |
|realnvp | The central flow based models used in our flow gan experiments. |
|plot_loss.ipynb | Sample loss plotting source code used to report different loss curves. |
