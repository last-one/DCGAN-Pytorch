# DCGAN in Pytorch
PyTorch implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)(DCGAN), which is a stabilize Generative Adversarial Networks. The origin code can be found [here](https://github.com/soumith/dcgan.torch).

## Network architecture
![alt tag](DCGAN.png)

* Generator
	* input: a vector with z_size.
	* hidden layers: Four 4x4 transposed convolutional layers (1024, 512, 256, and 128 kernels, respectively) with ReLU
	* output layer: 4x4 transposed convolutional layer (out_size channels, 4096nodes = 64x64 size image) with Tanh.
	* BatchNormalization is used except for output layer.

* Discriminator
	* input: a vector with in_size * image_size * image_size.
	* hidden layers: Four 4x4 convolutional layers (128, 256, 512, and 1024 kernels, respectively) with LeakyReLU.
	* output layer: 4x4 convolutional layer (1 node) with Sigmoid.
	* BatchNormalization is used except for 1st hidden layer and output layer.

## MNIST dataset
* For MNIST image, the in_size is 1 and image_size is 64.
### Results
* The learning rate is 0.0002, batch size is 128 and the optimizer is Adam.

<div align='center'>
<img src='MNIST_results/result.gif'>
</div>

## References
1. https://github.com/soumith/dcgan.torch
2. https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN
3. https://github.com/togheppi/DCGAN
4. https://github.com/carpedm20/DCGAN-tensorflow
