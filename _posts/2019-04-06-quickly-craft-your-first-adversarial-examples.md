---
title: "Craft your first adversarial examples"
layout: post
date: 2019-04-06 14:15
mathjax: true
headerImage: true
tag:
- adversarial examples
- neural networks
star: false
category: blog
author: davidglavas
description: A quick tutorial on crafting adversarial examples, inputs to a machine learning model that have been intentionally designed to cause the model to malfunction.
---

<p align="center">
  <img src="https://github.com/davidglavas/davidglavas.github.io/blob/master/_posts/Figures/2019-04-06-quickly-craft-your-first-adversarial-examples/coverComparison.png?raw=true">
</p>


## TL;DR
The goal of this post is to help you craft your first adversarial examples quickly. We use Keras running on top of TensorFlow to train the target neural network, then we craft the adversarial examples and demonstrate their effect on the target network. Train the target network yourself by running [this](https://github.com/davidglavas/Craft-your-first-adversarial-examples/blob/master/trainTargetModel.py) or download it [here](https://github.com/davidglavas/Craft-your-first-adversarial-examples/blob/master/MNIST_model.h5). Craft the adversarial examples by running [this](https://github.com/davidglavas/Craft-your-first-adversarial-examples/blob/master/craftAdversarialExamples.py), make sure to have the target network in the same directory.


### What are adversarial examples?
An adversarial example is an input to a machine learning (ML) model that has been intentionally designed to cause the model to malfunction. We will see how to generate such inputs for deep neural networks using one of the earliest methods, the fast gradient sign method (FGSM). But first, let's motivate the study of adversarial examples by taking a look at some of the threats they pose to deployed models in the real world. Feel free to skip this part if you want to craft asap.

Most early adversarial example research was performed under unrealistic conditions, but recently there has been an increasing number of more practical attacks [1]. Many works assume the adversary to have full access to the target model (white-box), but Papernot et al. have shown that attacks are possible even if the adversary has no access to the underlying model (black-box) [2]. Even state of the art machine learning models that are being offered as a service have been shown to be vulnerable to such black-box attacks [3]. Most works assume a threat model in which the adversary can feed data directly into the classifier on a digital level, but researchers have shown that adversarial examples that are printed onto paper and are perceived through a camera by the target network, are also classified incorrectly [4]. Researchers even printed 3D adversarial objects that are robust towards viewpoint shifts, camera noise, and other natural transformations [5].

Sharif et al. create physical adversarial examples to deceive state of the art neural network based face detection and commercial face recognition systems [6]. These systems are widely used for various sensitive purposes such as surveillance and access control. They print a pair of eyeglass frames, which allows the adversary that wears them to evade being recognized or to impersonate another individual.  Their attack is physically realizable and inconspicuous, meaning that they create not a digital, but a physical adversarial accessory which doesn't attract the attention of humans (eg. security guard), but which effectively turns the carrier of the accessory into an adversarial example. 

Another interesting practical attack involves the use of adversarial examples to deceive road sign recognition network [7]. Eykholt et al. apply stickers to road signs which cause the target network to interpret a physical stop sign as a speed limit 45 sign. They show that attackers can physically modify objects such as road signs to reliably to cause classification errors in deep learning based systems under widely varying distances, angles, and resolutions.

Despite these threats, and despite the many approaches to protect neural networks that have been proposed, there is no known reliable defense against adversarial examples so far.


### Train the target network
Adversarial examples need a target, some model to deceive. You can use any neural network really, but you might need to adapt the crafting process if the model's interface changes. Here we train a Keras sequential model that achieves >99% accuracy on MNIST. You can either run the [code](https://github.com/davidglavas/Craft-your-first-adversarial-examples/blob/master/trainTargetModel.py) and train the model yourself, or [download](https://github.com/davidglavas/Craft-your-first-adversarial-examples/blob/master/MNIST_model.h5) the trained model and proceed to the next section. Note that the training might take well over an hour if you run TensorFlow on a CPU.


### Craft the adversarial examples
At this point you should have a trained Keras sequential model stored on disk. You can craft the adversarial examples by running [this](https://github.com/davidglavas/Craft-your-first-adversarial-examples/blob/master/craftAdversarialExamples.py) (put the target model in the same directory as the script). The script loads the MNIST dataset, loads the trained model, crafts adversarial examples for the trained model from the MNIST test set, and stores the adversarial examples on disk. Congratulations, you have crafted 10000 adversarial examples, 61.31% of which cause the target model to return the wrong result. Note that this is the same model that achieves >99% accuracy on the original test set.

For a given example, FGSM computes the derivative of the model’s loss function with respect to each pixel, then it modifies each pixel in the direction of the gradient by a chosen perturbation size $\epsilon$. Given an example $x$, this method computes an adversarial example $x^*$ as

$$x^{*} = x + \epsilon sign(\nabla_x J(\Theta, x, y)),$$

where $J(\theta, x, y)$ is the target model's loss function, with $\theta$ as the model's parameters, and $y$ as the label of the given example $x$.

Let's take a closer look at the effect of an adversarial example on the target network. On the left we see a natural, on the right an adversarial example:

<p align="center">
  <img src="https://github.com/davidglavas/davidglavas.github.io/blob/master/_posts/Figures/2019-04-06-quickly-craft-your-first-adversarial-examples/fourSideBySide.png?raw=true">
</p>

Here is the output of the model's softmax layer for the left (natural) example:

``` python
[[0.000000 0.000000 0.000000 0.000000 0.999992 0.000000 0.000000 0.000000 0.000000 0.000008]]
```

Here is the output for the right (adversarial) example:

``` python
[[0.051990 0.006389 0.050749 0.010629 0.253468 0.015621 0.035309 0.007679 0.120034 0.448132]]
```

We can interpret each of the numbers as the model's certainty. The number at the i-th index is how certain the model is that the given example belongs to the i-th class. The index of the greatest number in this array corresponds to the predicted class. We can see that the network is fairly certain about the left image being a four, whereas for the right image it's convinced that it's a nine.


Note that generated adversarial examples don't neccesarily succeed at deceiving the target network. In this case, 38.69% of the crafted adversarial examples fail at doing so. We can see one such case in the following example: 

<p align="center">
  <img src="https://github.com/davidglavas/davidglavas.github.io/blob/master/_posts/Figures/2019-04-06-quickly-craft-your-first-adversarial-examples/twoSideBySide.png?raw=true">
</p>

Here is the output for the left (natural) example:

``` python
[[0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000]]
```

Here is the output for the right (adversarial) example:

``` python
[[0.163652 0.104641 0.446153 0.014896 0.021194 0.021860 0.091441 0.009633 0.096387 0.030143]]
```

We see that for the natural example, the model is certain that it's a two. For the adversarial example, the model's certainty that it's a two drops significantly, but it still thinks it's a two.

To conclude, we trained a neural network on MNIST that achieves >99% accuracy on the test set. Then we used FGSM to craft an adversarial test set for which the same network achieves 38.69% accuracy. In the end, we examined the effect of adversarial examples on the model's softmax output. Note that there are more powerful attack algorithms that result in less perceptible perturbations of the image, and that cause the target model to make mistakes with much higher certainty. For example, performing the same experiment using Carlini and Wagner's attack (C&W) results in the target model having less than 1% accuracy on the adversarial test set.
<br/><br/>
<br/><br/>


### References:

[1]: Sun, Lu, Mingtian Tan, and Zhe Zhou. "A survey of practical adversarial example attacks." Cybersecurity 1.1 (2018): 9.

[2]: Papernot, Nicolas, Patrick McDaniel, and Ian Goodfellow. "Transferability in machine learning: from phenomena to black-box attacks using adversarial samples." arXiv preprint arXiv:1605.07277 (2016).

[3]: Papernot, Nicolas, et al. "Practical black-box attacks against machine learning." Proceedings of the 2017 ACM on Asia conference on computer and communications security. ACM, 2017.

[4]: Kurakin, Alexey, Ian Goodfellow, and Samy Bengio. "Adversarial examples in the physical world." arXiv preprint arXiv:1607.02533 (2016).

[5]: Athalye, Anish, et al. "Synthesizing robust adversarial examples." arXiv preprint arXiv:1707.07397 (2017).

[6]: Sharif, Mahmood, et al. "Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. ACM, 2016.

[7]: Eykholt, Kevin, et al. "Robust physical-world attacks on deep learning models." arXiv preprint arXiv:1707.08945 (2017).
