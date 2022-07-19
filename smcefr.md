---
author:
- Cade Brown <cade@utk.edu>
- Piotr Luszczek
title: 'SMCEFR: Sentinel-3 Satellite Dataset'
---

Check out the source on GitHub: <https://github.com/cadebrown/smcefr>

Link to downloads: <https://github.com/cadebrown/smcefr/releases>

Introduction
============

![Figure 1: A sample of 18 images from the `smcefr-mini`
dataset](https://cade.site/files/smcefr/miniset.jpg){#smcefr-18}

Satellite data is important for many environmental sciences, as they
give scientists a bird's eye view of large areas of the Earth. Modern
orbital sensor allow them to collect additional research input remotely
with high precision in order to emply data-intensive analysis workflows.
For example, the Sentinel-3 satellite collects images that are used by
scientists for a wide variety of research tasks, such as monitoring
ocean and land surface temperatures, bootstrapping models for weather
forecasts or atmospheric conditions' prediction, and modeling and
predicting climate change.

However, in its original form, the data size and the format are unwieldy
for rapid data exploration. Our goal was to remedy this by simplifying
the dataset. To this end, it has been prepossessed to use a more
approachable format, which is described in
Section [2](#DS){reference-type="ref" reference="DS"}, but here is a brief overview:

SMCEFR: Sentinel-3 Satellite is a dataset consisting of
1024x1024 red-gree-blue (RGB) images generated from the
Sentinel-3 satellite via the Ocean and Land Color Instrument (OLCI).
In order to facilitate the competition challenge we applied
simplifications to the original and reduced the data volume. This
preprocessing created a subset of the data and produced RGB images
that are easier to analyze with model prototypes using tools like
Python and NumPy/Pillow modules or Tensorflow or OpenCV in C++ and
many other modern image processing software. Our goal was to give the
participants the ability to rapidly test and visualize their
algorithms. Furthermore, we posed challenge questions to guide the
potential research directions for the participants could explore to
present their own insights. In particular, we encourage approaches
originating from computer vision, numerical programming, and machine
learning.


Data Source {#DS}
===========

The original data for this dataset came from the Copernicus
Hub [[2]](#R2), which hosts large-scale historical records allowing
access to multi-sensor data. However, the hub's access policy requires a
complicated workflow to query, filter, download, and preprocess the
available datasets. We recongize the need of the full applications to
have accesss to the extra geospatial and spectral information. While
this could be extremely useful, for the purposes of a data challenge
competition with a broad appeal, a smaller and more manageable dataset
is desirable. In the following, we summarize the specifications that
guided the creation of this dataset, which we will call `SMCEFR` (SMC's
Earth Full Resolution):

-   Data from the Ocean Land and Color Instrument (OLCI) on the
    Sentinel-3 was queried.

-   Data was considered from the OLCI's Earth Full Resolution (EFR) data
    product [[3]](#R3).

-   Only the scans visualizating a part of the North or Sourth American
    continents were used, to provide a more consistent dataset.

-   Data from the selected time periods and only in the past 3 years
    were used.

-   All images are center-cropped to the exact 1024x1024 resolution, or
    were discarded if only a smaller image was available. This naturally
    avoided the need for normalization and resizing.

-   Multispectral data (20+ bands) are reduced to standard visible RGB
    representation to reduce the dimensionality.

-   The images with the most extreme cases of sparsity or low-entropy
    were discarded. For example, images that are completely white or
    blue which represented dense cloud cover or ocean water,
    respectively.

In the end, we produced smaller scale datasets in specific sizes, and
encoded as traditional RGB images (PNG image format) with a 1024x1024
size. Figure [1](#smcefr-18){reference-type="ref" reference="smcefr-18"}
presents a sample of 18 images from our reduced data set. These are
available from the GitHub release page [[1]](#R1), and are meant to
be easily accessible using any of the following software packages or
workflows:

-   Python with NumPy/SciPy, Pillow, Pandas modules.

-   Python with OpenCV module.

-   Python with Tensorflow or PyTorch.

Filename Schema
---------------

For clear identification, the PNG images in the dataset are named by
followiing a strict scheme and the details of the format are described
in Figure [2](#sen3-fname){reference-type="ref" reference="sen3-fname"}.
Although this information is auxiliary for most intended processing
scenarios, it could serve as additional input and guide supervision
during training.

![Figure 2: Sentinel-3 Generic Filename Schema Image](https://cade.site/files/smcefr/filename.png){#sen3-fname}

Accessing Dataset
-----------------

Our dataset is available for download on GitHub [[1]](#R1). The dowload is compact: it is a single Tar file compressed with Gzip
([smcefr-full.tar.gz](smcefr-full.tar.gz)). The location of the file is
[github.com/cadebrown/smcefr/releases/download/v1.0.0/smcefr-full.tar.gz](https://github.com/cadebrown/smcefr/releases/download/v1.0.0/smcefr-full.tar.gz)
and it can be expanded with the command: 


```shell
$ tar -xvf smcefr-full.tar.gz
```

which will create a directory that contains all the PNG files we
selected. These can then be easily read with the OpenCV library,
Python's Pillow, Tensorflow, and many other media frameworks.

Additionally, the source code and implementations for the dataset
reduction mentioned above can be found on GitHub [[1]](#R1).

Reporting Problems
------------------

If any errors arise during the usage of the dataset, an issue can be
filed on the GitHub page [[1]](#R1), or by directly contacting the
authors of this paper: corresponding author is Cade Brown, .

Challenge Questions
===================

To motivate the potential analysis methods for the dataset, we present
below sample challenge questions and directions that explore the
prospective ideas in data science, computer vision, and machine
learning.

Cloud Identification (#1)
-------------------------

What methods can be used to segment and identify the regions of the
images that are partially or completely obstructed by the cloud cover?

Potential approaches:

-   Computer Vision: thresholding, K-means clustering, and edge
    detection should allow for a straightforward identification of
    clouds as a mask or segmentation map.

-   Machine Learning: many methods could be used, for example, the U-Net
    architecture [[4]](#R4). Most methods would require some ground
    truth dataset, i.e., using supervised machine learning. Also,
    unsupervised clustering followed by a programmer-assisted
    classification on the learned clustering patterns could be used as
    well. i.e., using unsupervised machine learning. A survey of
    available methods is available online by Sanatan
    Mishra [[5]](#R5).

Noise Removal (#2)
------------------

Consider a case where the sensor data is incomplete, of varied quality,
or totally degraded. Is there any way to take partially damaged/noisy
data, and reconstruct something "closer" to the original sensor reading?

Be sure to consider the measure "closeness" carefully. For example, can
useful metrics (PSNR, MSE) be used to compare performance of various
methods?

For this task, we suggest creating copies of the "ground truth" dataset,
and then adding a random amount of noise, followed by processing the
copy as input.

-   Computer Vision: standard denoising techniques (Non-local Means
    Denoising, available in OpenCV [[6]](#R6)) would produce usable
    results quite quickly.

-   Machine Learning: using unsupervised learning techniques, with the
    noise-augmented copies, one could use any number of optimization
    approaches. One example using Tensorflow that uses the concepts of
    autoencoders and latent space is available online [[7]](#R7).
    For using this approach, a good intro to latent space can be found
    online, written by Ekin Tui [[8]](#R8).

Image Compression (#3)
----------------------

To preserve data integrity, the dataset is provided as PNG, in the
lossless data compression format mode. However, for a variety of
purposes, it would be useful to allow a small amount of error in
exchange for an appreciatiable size reduction. What can research
methods be used to save space while keeping the best quality possible?

-   Numerical Programming: software libraries exist for HPC and
    scientific workloads, for example zfp [[9]](#R9), FFT, cosine
    transform, and wavelet-based methods.

-   Machine Learning: Using Generative Adversarial Networks (GANs), some
    models have been trained to perform image compression
    ~4x times better than JPEG, such as
    HIFIC [[10]](#R10), which comes with included pre-trained
    models and is freely available to download. For code samples and
    their documentation, see the Tensorflow Compression code [[11]](#R11).
    Additionally, a custom autoencoder could be trained specifically for
    the task, so that the compression is optimized for this dataset, and
    potentially allowing for even more fine-tuned performance in terms
    of the compression ratio.



References
==========

<a name="R1">[1]</a> Cade Brown: *SMCEFR: Sentinel-3 Satellite Dataset*
<https://github.com/cadebrown/smcefr>

GitHub: Dataset Download: `smcefr-full`(1GB)
<https://github.com/cadebrown/smcefr/releases/download/v1.0.0/smcefr-full.tar.gz>.

<a name="R2">[2]</a> Copernicus Open Access Hub, <https://scihub.copernicus.eu/>.

<a name="R3">[3]</a> Sentinel-3 OLCI EFR Data Product Description,
<https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-3-olci/level-1/fr-or-rr-toa-radiances>.

<a name="R4">[4]</a> Olaf Ronneberger, Philipp Fischer, Thomas Brox: *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI Springer, LNCS,
Vol.9351: 234--241, 2015. <https://arxiv.org/pdf/1505.04597.pdf>

<a name="R5">[5]</a> Sanatan Mishra: *Unsupervised Learning and Data Clustering.*
<https://towardsdatascience.com/unsupervised-learning-and-data-clustering-eeecb78b422a>

<a name="R6">[6]</a> OpenCV: *Image Denoising.*
<https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html>.

<a name="R7">[7]</a> Easy Tensorflow: *Noise Removal Autoencoder*
<https://www.easy-tensorflow.com/tf-tutorials/autoencoders/noise-removal>

<a name="R8">[8]</a> Ekin Tiu: *Understanding Latent Space in Machine Learning*. <https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d>.


<a name="R9">[9]</a> zfp: *Compressed Floating-Point and Integer Arrays*. <https://computing.llnl.gov/projects/zfp>.

Peter Lindstrom, *Fixed-Rate Compressed Floating-Point Arrays* IEEE
Transactions on Visualization and Computer Graphics, 20(12): 2674--2683,
December 2014, doi:10.1109/TVCG.2014.2346458.
<https://ieeexplore.ieee.org/document/6876024>


<a name="R10">[10]</a> Mentzer, Fabian and Toderici, George D and Tschannen, Michael and
Agustsson, Eirikur: *High-Fidelity Generative Image Compression.* Advances
in Neural Information Processing Systems, Vol.33, 2020
<https://hific.github.io/>

<a name="R11">[11]</a>Tensorflow Source Code: *Compression*. <https://github.com/tensorflow/compression/tree/master/models/hific>


