# [Modulation Identification]Deep Learning-Based Classification of Multipath Fading Channel Modulation: 16QAM, 32QAM, and 64QAM [with Python code]

Introduction: Autonomous signal modulation recognition has a wide range of applications in military, such as electronic warfare, jamming, and detection of enemy signals. Due to the current dense radio communication environment, frequency spectrum perception is becoming increasingly challenging. As a result, there is an increasing need for automatic signal recognition and classification.
Typically, there are direct and primary reflections between fixed radio transmitters and receivers. The time difference in received signal power due to changes in transmission medium or path is known as fading. These obstacles can be buildings, trees, rain, lightning, etc. These obstacles produce complex transmission effects on the transmitted signals. Fading includes multipath delay spread and Doppler spread.
In this paper, we are interested in frequency-selective fading, where multiple reflection paths and different path delays exist between the transmitter and receiver. Additionally, our fading channel is static, meaning that the receiver does not move, thus setting the Doppler shift to zero. This can be achieved using the MatLab model SISO based on Rayleigh fading [3] and [4].
The paper describes the classification of multipath fading signals from 3 to 7 paths (reflections). We use a simplified CNN model; then we compare it with state-of-the-art CNNs. The rest of the paper is organized as follows: Section 2 describes the signal generation and methods used in deep learning. Section 3 introduces the research results and discussions. Finally, Section 4 summarizes the paper.
In this section, we introduce the design and experiments for deep learning using multipath fading channels. Our study is based on two parts. First, various multipath and SNR values are considered to generate signals. In the second part, several convolutional neural networks are used to distinguish (classify) various multipath signals.
A. Signal Generation
We created six datasets, one for noisy signals alone; the other five for noisy signals and Rayleigh channels. The number of reflections in each dataset varied from 3 to 7.
In all datasets, we generated signals modulated with 16QAM, 32QAM, and 64QAM. We chose these three modulation schemes because they are commonly used for signal modulation. As shown in Figure 2, these multipath signals naturally mix Gaussian noise.
We used 350 samples for training and 150 samples for testing each dataset at different SNR levels. We used a sampling rate of 20kHz and assumed an outdoor setting with a delay of approximately 1e-5. In the constellation diagrams, the images we see depend on the spread of points. Signals with greater noise occupy larger spaces, as seen in SNR-30 dB. The constellation (points) size is 80 x 80 pixels. In contrast, at SNR -20 dB, the points diffuse within a smaller space of 30x30. We use 20 x 20 to represent SNR-15 dB and 12 x 12 to represent SNR-5 and 0.
B. CNN Model
In the next two sections, we first explain our model and then in the following section, we explain other CNN models. In our work, we use Conda to manage and separate the system packages environment. We used Pytorch and Jupyter to run Python code. We trained data using three epochs and used the ADAM method (Optim.Adam) as the optimizer.
1) Our model: Our efficient CNN model computes quickly, with a six-layer architecture as shown in Figure 3. Our algorithm uses two convolutional layers and two pooling layers, then feeds the data through fully connected hidden layers to our output. Our model follows conv/relu/pool/conv/relu/pool/fc/relu/fc. Our first convolutional layer will have three input channels, six output channels, kernel size of 3 (producing 3x3 filters), and stride of 1 pixel. They are passed in the form of nn.Conv2d(3,6,3,1).
Step by step completion: Firstly, set up the convolutional layer using torch.nn.Conv2d(). The first layer has three input channels. We will specify six output channels for feature extraction. We set the kernel size to 3 to create 3x3 filters, and the stride to 1. The second layer will take our six input channels and provide 16 output channels. Then, we use torch.nn.Line() to set up fully connected layers. The input size of (54x54x16) is determined by the impact of our kernel on the size of the input image. A 3x3 filter applied to an 28x28 image leaves one pixel edge on all four sides. In a single layer, the size changes from 28x28 to 26x26. We apply twice the kernel and twice the pooling layer, so the output will be ((28-2)/2)-2)/2=5.5, which is rounded down to 5 pixels per side. Then, we define the forward method. Activation can be applied to a row of convolutions using F.relu(), and pooling is completed using F.max_pool2d(). Then, we use .view() to flatten the data of the fully connected layers.
2) Advanced Image Classification Models: Since our problem is an image classification one, we optimized the most advanced image classification models to handle our dataset. We tested ResNet50, EfficientNetb4, and MobileNetV3. We used PyTorch Image Models (TIMM) to load pre-trained weights. PyTorch Image Models are a collection of image models, layers, utilities, optimizers, schedulers, data loaders/augmentations, and reference training/validation scripts, aiming to consolidate various SOTA models together and capable of replicating ImageNet training results [16].

Results and discussion: Below are our results for Rayleigh fading channel classification, considering the impact of SNR, with varying reflections from 3 to 7, as well as the impact of using different CNN models for classification:
A. Impact of Signal-to-Noise Ratio (SNR):
In our experiments, we first tested the impact of SNR on signal constellation for QAM signals. Figure 1 shows the effect of SNR ranging from -30 dB to 20 dB on QAM signal modulation. We wanted to compare the signal with noise to obtain signals with noise and multipath fading.
Using our CNN model, the results overall are good; we achieved 100% accuracy, with a signal-to-noise ratio ranging from -30 dB to -10 dB. At a signal-to-noise ratio of -15 dB, precision began to decrease, reaching 99.11%. We clearly observed the impact of a signal-to-noise ratio of -20 dB on 77.778%. Finally, at worst case SNR-30, accuracy dropped to 35.33%.
B. Multipath Rayleigh effect
In this section, we tested the impact of multipath fading channels on the signal constellation in outdoor settings with path delay represented by 1e-5. Our experiments included reflections of the fading channel three, four, five, six, and seven times. The maximum Doppler shift of the signal was zero, and the configurations for path delay and average path gain were as follows:
· Path delay for three paths = [0 500 800]*1e-5
· Path length for three paths = [0 -10 -20]
· Path delay for four paths = [0 500 800 1400]*1e-5
· Path length for four paths = [0 -10 -15 -20]
· Path delay for five paths = [0 500 800 1400 2000]*1e-5
· Path length for five paths = [0 -5 -10 -15 -20]
· Path delay for six paths = [0 500 800 1400 2000]*1e-5
· Path length for six paths = [0 -5 -10 -15 -20]
· Path delay for seven paths = [0 500 800 1400 2000]*1e5
· Path length for seven paths = [0 -5 -10 -15 -20]
In Figure 4, we present the constellation diagram of signals with three multipath fading channels, which shows a promising outcome since we started by considering SNR as the best case and achieved 100% accuracy. We obtained the same precision at an SNR-10 dB. As the signal-to-noise ratio dropped to -15 dB, we began to see an error classification rate of 92.444%. This trend continued as we reduced it to -20 dB and within the range of 64.444%. Our worst-case scenario at -30 dB showed an accuracy of 34.222%.
Figure 5 illustrates a comparison between our CNN model using only noisy signals and signals other than Rayleigh channel paths. Both have around 100% accuracy, reaching up to -10 dB; thereafter, the accuracy fluctuates unpredictably, with no clear correlation between the number of reflections and accuracy.
C. Impact of Using State-of-the-Art CNN Models
We conducted all experiments using other CNN models: ResNet50, EfficientNet-b4, and MobileNetV3-Large-100. We noticed that they had very similar accuracies, but our model took much less time to run. The average computation time for our model was 100 seconds, while MobileNetV3 took an average of 400 seconds, and EfficientNet_b4 an average of 2000 seconds. Our model is three times faster than MobileNetV3 and about 20 times faster than ResNet50 and EfficientNet-b4. Body.6 presents the accuracy of different CNN models in four Rayleigh model scenarios. We conducted the experiments on a MacBook Pro 2019 equipped with a 2.3 GHz 8-core Intel Core i9 and 16 GB 2400 MHz DDR4 memory.
Conclusions The study shows that in the case of Rayleigh fading, we can obtain good results from the constellation diagram. Using our CNN model, we are able to classify the QAM family with 16, 32 and 64 modulation orders, all with a signal-to-noise ratio (SNR) of 100% in -10. Our model performs as well or better than the state-of-the-art CNN models tested, and it takes much less time.

## Images

![(Modulation Identification)Deep Learning-Based Classification of Multipath Fading Channel Modulation: 16QAM, 32QAM, and 64QAM [with Python code]](01.jpg)
![(Modulation Identification)Deep Learning-Based Classification of Multipath Fading Channel Modulation: 16QAM, 32QAM, and 64QAM [with Python code]](02.jpg)
![(Modulation Identification)Deep Learning-Based Classification of Multipath Fading Channel Modulation: 16QAM, 32QAM, and 64QAM [with Python code]](03.jpg)
![(Modulation Identification)Deep Learning-Based Classification of Multipath Fading Channel Modulation: 16QAM, 32QAM, and 64QAM [with Python code]](04.jpg)
![(Modulation Identification)Deep Learning-Based Classification of Multipath Fading Channel Modulation: 16QAM, 32QAM, and 64QAM [with Python code]](05.jpg)
![(Modulation Identification)Deep Learning-Based Classification of Multipath Fading Channel Modulation: 16QAM, 32QAM, and 64QAM [with Python code]](06.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
