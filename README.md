## Identify dolphin species acoustically through machine learning.

This repository contains code for training and testing machine learning models for identifying delphinid species by acoustic characteristics of their vocalizations. Subfolders linked below contain two different classification methods, one using convolutional neural networks (delphinID) and another using Random Forest analysis (ROCCA), for acoustically classifying species. Our documentation here focuses on training and applying models using automated detections of vocalizations made in the open-source software [PAMGuard](https://www.pamguard.org/), though the same workflow and code can be applied to detections extracted elsewhere.

Base classifiers (delphinID or ROCCA) are trained to classify whistles or clicks to species and these predictions are then integrated to predict acoustic events based on information from these two types of vocalizations.

![Alt text](images/methods_simple_1.PNG)
##
### which.dolphin/

#### └── [delphinID](https://github.com/tristankleyn/which.dolphin/tree/main/delphinID)/

delphinID models are convolutional neural networks (CNNs) developed in TensorFlow and trained to classify species by latent characteristics in concatenated normalised frequency power spectra of whistles or clicks. 

#### └── [rocca](https://github.com/tristankleyn/which.dolphin/tree/main/rocca)/

[ROCCA](https://www.pamguard.org/rocca/rocca.html) (Real-time Odontocete Call Classification Algorithm) ([Oswald et al., 2007](https://pubs.aip.org/asa/jasa/article/122/1/587/813007)) models are Random Forest models developed in WEKA and trained to classify species based on select acoustic-temporal measurements extracted from whistles or clicks.

#### └── [eventClassifier](https://github.com/tristankleyn/which.dolphin/tree/main/eventClassifier)/

Folder containing simple graphical user interface developed in R _Shiny_, which allows users to integrate predictions from whistle and click classifiers (either ROCCA or delphinID) with an event classifier model to form final species predictions for isolated acoustic events.

