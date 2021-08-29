# Facial Expression Recognition & Comparative Study on Densenet161 and Resnet152 using Deep Learning, PyTorch, and Transfer Learning
![UI](download.png)

Facial Expression Recognition can be featured as one of the classification jobs people might like to include in the set of computer vision. The job of our project will be to look through a camera that will be used as eyes for the machine and classify the face of the person (if any) based on his current expression/mood.

Face recognition is a method of identifying or verifying the identity of an individual using their face. It is one of the most important computer vision applications with great commercial interest. Recently, face recognition technologies greatly advanced with deep learning-based methods.

Face recognition in static images and video sequences, captured in unconstrained recording conditions, is one of the most widely studied topics in computer vision due to its extensive range of applications in surveillance, law enforcement, bio-metrics, marketing, and many more.

DATASET
------
https://www.kaggle.com/apollo2506/facial-recognition-dataset

Pretrained Model
-----

**DenseNet161**

Recent work has shown that convolutional networks can be substantially deeper, more accurate, and efficient to train if they contain shorter connections between layers close to the input and those close to the output. In this paper, we embrace this observation and introduce the Dense Convolutional Network (DenseNet), which connects each layer to every other layer in a feed-forward fashion. Whereas traditional convolutional networks with L layers have L connections - one between each layer and its subsequent layer - our network has L(L+1)/2 direct connections. For each layer, the feature maps of all preceding layers are used as inputs, and their own feature maps are used as inputs into all subsequent layers. DenseNets have several compelling advantages: they alleviate the vanishing gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters. We evaluate our proposed architecture on four highly competitive object recognition benchmark tasks (CIFAR-10, CIFAR-100, SVHN, and ImageNet). DenseNets obtain significant improvements over the state-of-the-art on most of them, whilst requiring less memory and computation to achieve high performance. 
Authors: Gao Huang, Zhuang Liu, Kilian Q. Weinberger, Laurens van der Maaten

<p align="left">
    <img src="https://cdn-images-1.medium.com/max/800/0*x37oN_kC5z0sD-rI.jpg" width="770" height="470">
  </p>
<p align="left">
    <img src="https://cdn-images-1.medium.com/max/800/1*8u_aFzHgNyW3a1ENM0BoTg.jpeg" width="770" height="470">
  </p>
