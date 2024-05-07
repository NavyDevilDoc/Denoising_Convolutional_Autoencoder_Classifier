I love computer vision. While I have gone deep down the RAG hole lately, CV will always be my passion. Specifically, denoising photos is
a ton of fun, so I figure, why not build a program to denoise AND classify?

This program is designed to work with either the Fashion MNIST or CIFAR-10 datasets. First, it applied a Gaussian noise filter to blur the
pictures. Next, the pictures are run through a denoising convolutional autoencoder followed by running them through a CNN for classification.
In the past, there were issues with getting the pipeline to work seamlessly with both datasets, but I think I've ironed them out at this
point. If anyone notices any issues, don't hesitate to let me know. What I'd like to do is use this as a gateway into image enhancement
work, so the better I get at this, the happier I'll be.
