# Whale Sound Detection
Transfer Learning Xception was used for recognizing and classifying whale voices using spectrograms and time series of signal complexity.

<p align="center">
	<img src="https://github.com/NasonovIvan/NNSoundRecognition/blob/main/images/marinexplore_kaggle.png" width="350">
</p>

## Project Structure

```
project_root/
│
├── src/
│   ├── InceptionTime/
│   │   ├── main.py      # Main script for InceptionTime model
│   │   ├── train.py     # Training script for InceptionTime
│   │   └── model.py     # InceptionTime model definition
│   │
│   ├── AttentionLSTM/
│   │   ├── main.py      # Main script for AttentionLSTM model
│   │   ├── train.py     # Training script for AttentionLSTM
│   │   └── model.py     # AttentionLSTM model definition
│   │
│   ├── Xception/
│   │   ├── main.py      # Main script for Xception model
│   │   ├── train.py     # Training script for Xception
│   │   └── model.py     # Xception model definition
│   │
│   ├── VisTransformer/
│   │   ├── main.py      # Main script for VisTransformer model
│   │   ├── train.py     # Training script for VisTransformer
│   │   └── model.py     # VisTransformer model definition
│   │
│   ├── notebooks/       # Jupyter notebooks for quick testing
│   │
│   └── utils/
│       ├── data_preparation.py  # Dataset preparation utilities
│       ├── evaluate.py          # Model evaluation utilities
│       ├── functions.py         # Miscellaneous utility functions
│       ├── imports.py           # Common imports
│       ├── path_names.py        # File and folder path definitions
│       └── utils.py             # General utility functions
│
└── README.md
```

## Run the code

Install the required packages:
```
pip install -r requirements.txt
```

Run the train (with noise or not data) or test (the same) of the following models:
- InceptionTime
- AttentionLSTM

```
python [InceptionTime|AttentionLSTM]/main.py [--train|--train_noise|--test|--test_noise]
```
In these models `test_article` means test the network on the data, which are different from the training:
- Xception
- VisTransformer

```
python [Xception|VisTransformer]/main.py [--train|--test|--test_article]
```

### Description:
In this project I analyze [The Marinexplore and Cornell University Whale Detection Challenge](https://www.kaggle.com/c/whale-detection-challenge), where participants were tasked with developing an algorithm to correctly classify audio clips containing sounds from the North Atlantic right whale.

In my work, I concentrated on the analysis of spectrogram images by the [Xception](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf) neural network and applied [InceptionTime](https://link.springer.com/article/10.1007/s10618-020-00710-y) network with network, based on Multi-Head Attention, to information characteristics classification.

### Data:
The Kaggle training set includes approximately 30,000 labeled audio files. Each file encodes a two second monophonic audio clip in AIFF format with a 2000 Hz sampling rate. Based on these files, I obtained spectrogram images, as in the example below:
<p align="center">
	<img src="https://github.com/NasonovIvan/NNSoundRecognition/blob/main/images/train4.png" width="350">
</p>

### Formation of information characteristics
It is proposed to use such information characteristics as spectral entropy and complexity as features forming time series for recognizing acoustic signals of biological origin. In this case, different variants of statistical complexity $C_{SQ}, C_{JSD}, C_{TV}$ reach their maximum values on different discrete distributions, i.e. they are best suited for detection of signals with different spectral structure, which allows us to use them together with entropy $H$ as features in the classification task.

### Inception

The InceptionTime neural network is based on the idea of using Inception modules to analyse time series data. Each such module contains three convolution branches with different sizes of kernels in parallel branches, after which their outputs are combined to obtain a common feature representation.

Each Inception module applies univariate convolutions with kernels 1, 2 and 4 to analyse information at different levels of detail of the time series. After each convolution, a ReLU activation function is applied and the result is passed through a Dropout layer that resets the network weights to zero with some probability, which prevents overfitting. After processing the data in each branch, the results of all branches are combined by concatenation into a single matrix, which in turn is fed into the MaxPooling layer to reduce dimensionality and extract the most meaningful features from the input data. It is the network's ability to generalise different layers of information from time series that allows it to learn better from such data. The model is completed with two fully connected layers using the Dropout layer.

A network based on InceptionTime but modified by Residual technology has also been developed. This mechanism is introduced in the Inception block, in which the input vector is transferred to the final layer by a separate channel, added to the data processed through one-dimensional convolution layers. The Residual technique effectively combats the problem of gradient fading and speeds up the learning process. The main goal of this modification is to evaluate the impact of Residual on the network's ability to classify time series.

<p align="center">
	<img src="https://github.com/NasonovIvan/NNSoundRecognition/blob/main/images/inceptiontime-res-1.png" width="800">
</p>

### Attention

A third neural network was also developed, which is a model consisting of a recurrent LSTM (Long Short-Term Memory) layer, a Multi-Head Attention layer and several full-link layers. The LSTM recurrence layer is used to analyse a sequence of data over time. This layer is used to identify long term dependencies in a series of information features.

The main layer for feature selection in this network is Multi-Head Attention, which allows the model to consider different parts of the input data to highlight important features and establish relationships between them. Before applying the Self-Attention mechanism, the input data goes through three linear transformations of the weight matrices $W_Q, W_K, W_V$ (which are configured as model parameters during the training of the network) necessary to create $Q, K, V$ matrices for each element of the sequence to further represent them in a lower dimensional space. The importance of each element relative to the others is then evaluated by the scalar product of the weights.

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

where $d_k$ - specifies the dimensionality of the space $K$ and $Q$, and may vary depending on the specific architecture and task. The division in the formula above by $\sqrt{d_k}=4$ is performed to normalise the magnitude of the attention weights and control the scale of the values. 
The main property of this layer is the use of multiple transformations to emphasise different aspects of the data. To obtain a common feature representation of the data, the outputs of each transformation are combined into a single vector $Z_{0-7} \otimes W_0$, which takes into account the relationships between sequence elements and highlights the most important features.

In the network, the data is fed to the LSTM and Multi-Head Attention blocks in parallel, and subsequently their outputs are combined into a single layer, from which they are passed through a normalisation layer to improve stability, accelerate training convergence and address gradient fading. The data is then fed into the fully-connected TimeDistributed layers, which process the data in each time step independently of each other. A schematic of the Self-Attention network model is shown in the figure below

<p align="center">
	<img src="https://github.com/NasonovIvan/NNSoundRecognition/blob/main/images/attention-1.png" width="800">
</p>

### Xception:
[Xception](https://arxiv.org/abs/1610.02357) is a convolutional neural network that is 71 layers deep. I loaded a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories. As a result, the network has learned rich feature representations for a wide range of images, and it is what I used for recognizing and classifying whale voices' spectrograms.

<p align="center">
	<img src="https://github.com/NasonovIvan/NNSoundRecognition/blob/main/images/depthwise.png" width="350">
</p>

Training and testing of neural networks has been carried out on two different datasets of similar nature. For training, a dataset from the Kaggle competition with audio recordings of whale songs with each recording lasting 2 seconds and sampling rate of 2 kHz was chosen.

The neural networks were tested on the dataset from the [article](https://arxiv.org/abs/2001.09127), whose parameters differ from the training dataset: they are audio signals of 3 seconds duration each and a sampling rate of 1 kHz. In addition, these two datasets contain data separated by both the time and geography of their recording, i.e., generally speaking, they are potentially quite different entities, united only by the common nature of the audio signature.

### Results:

#### InceptionTime and Self-Attention

As a result of training InceptionTime, InceptionTime Residual neural networks and a network based on Self-Attention architecture with LSTM blocks to solve the problem of binary classification of information feature time series, the following values of metrics on the test dataset were obtained as shown in the table below:

| Network                | AUC ROC | Accuracy | F1-score | Recall | Precision |
|------------------------|---------|----------|----------|--------|-----------|
| InceptionTime          | 0.906   | 0.811    | 0.776    | 0.720  | 0.841     |
| InceptionTime Residual | **0.909** | **0.814** | **0.788** | **0.763** | **0.813** |
| Self-Attention         | 0.894   | 0.801    | 0.776    | 0.762  | 0.790     |

The main quality indicator of the signal detection problem is the ROC curve (receiver operating characteristic) and the area under it AUC ROC. The figure shows the error matrix and ROC-curve of the best trained classifier InceptionTime Residual.

<p align="center">
	<img src="https://github.com/NasonovIvan/NNSoundRecognition/blob/main/images/ROC_incept_ris-1.png" width="350"> <img src="https://github.com/NasonovIvan/NNSoundRecognition/blob/main/images/ConfMatr_incept_ris-1.png" width="270">
</p>

Analysing the obtained results, it can be observed that all networks showed similar high performance with close AUC ROC values. Interestingly, the Self-Attention based network achieves performance of the main quality metrics comparable to the InceptionTime network, but has better Recall and Precision metrics, which indicates the good ability of this architecture to classify time series. InceptionTime network with Residual technology ranks first in all metrics in the table, which shows the high efficiency of this method.

#### Xception

Xception and its application based on transfer learning technology are effective tools for solving binary audio signal classification problems and allow, having learnt on one dataset, to apply the acquired knowledge on third-party data. These results can be useful for marine mammal monitoring and research, as well as for other applications in audio signal processing and computer vision.

| Dataset                | Accuracy | F1-score | Recall | Precision |
|------------------------| ---------|----------|--------|-----------|
| Kaggle          		 |   0.99   |   0.99   |  0.99  |    0.99   |
| Article 				 |   0.67   |   0.88   |  0.67  |    0.76   |


### References:
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf)
- [InceptionTime: Finding AlexNet for time series classification](https://link.springer.com/article/10.1007/s10618-020-00710-y)
- [Characterizing Time Series via Complexity-Entropy Curves](https://arxiv.org/abs/1705.04779)
