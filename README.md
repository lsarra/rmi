# Renormalized Mutual Information for Artificial Scientific Discovery

![Sketch](figures/spiral.jpg)

This repository shows how to implement the Renormalized Mutual Information described in the paper "Renormalized Mutual Information for Artificial Scientific Discovery" by Leopoldo Sarra, Andrea Aiello and Florian Marquardt.

## Introduction
Renormalized Mutual Information is a quantity that allows to quantify the dependence between a given random variable and a deterministic function of it. This can be used to extract a low-dimensional feature of a high-dimensional system by maximizing the regularized mutual information that the feature has with the system.

![Sketch](figures/sketch.jpg)

Usual Mutual Information can't be used in this context, because it would always diverge for any choice of the feature. This is due to the deterministic dependence of the feature with the high dimensional variable. The mere addition of noise to mutual information to regularize the divergence is not enough to solve the problem: there is no guarantee that the optimal feature is not affected by the noise.

That's the reason why Renormalized Mutual Information should be used in the context of deterministic and continuous functions of random variables. Please refer to the [paper](https://arxiv.org/abs/2005.01912) for more information.

Here, we show how to implement the estimation and optimization of Renormalized Mutual Information in the case of low dimensional features. This case is easier to handle because the entropy of the feature can be estimated efficiently with a histogram.

## Estimating Mutual Information

The class in [renormalizedmutualinformation.py](renormalizedmutualinformation.py) can be directly used to estimate renormalized mutual information, for features that you provide (i.e. expressions f(x) of the high-dimensional variables x). 
Please refer to the notebook [FeatureComparisonExample.ipynb](FeatureComparisonExample.ipynb) for some comments and usage examples.

## Feature Extraction

Feature extraction is about automatically finding the optimal feature for a given distribution of high-dimensional data x. Please refer to the notebook [FeatureExtractorExample.ipynb](FeatureExtractorExample.ipynb) for an example. 


# Citation
If you find this code useful in your work, please cite our article
"Renormalized Mutual Information for Artificial Scientific Discovery", Leopoldo Sarra, Andrea Aiello, Florian Marquardt, arXiv:2005.01912

available on

https://arxiv.org/abs/2005.01912

---
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
