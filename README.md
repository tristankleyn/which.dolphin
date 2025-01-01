## Identify dolphin species acoustically through machine learning.

This is a repository containing multi-stage machine learning classification workflows for automatically identifying dolphin species by the characteristics of their acoustic vocalizations, specifically their echolocation clicks and  tonal whistles. 


![Alt text](images/methods_simple_1.PNG)
##
### which.dolphin/

#### └── [delphinID](https://github.com/tristankleyn/which.dolphin/tree/main/delphinID)/

delphinID models are convolutional neural networks (CNNs) developed in TensorFlow and trained to classify species by latent characteristics in concatenated average frequency spectra of whistles or clicks. 

#### └── [rocca](https://github.com/tristankleyn/which.dolphin/tree/main/rocca)/

[ROCCA](https://www.pamguard.org/downloads.php?cat_id=5) (Real-time Odontocete Call Classification Algorithm) ([Oswald et al., 2007](https://pubs.aip.org/asa/jasa/article/122/1/587/813007)) models are Random Forest models developed in WEKA and trained to classify species based on select acoustic-temporal measurements extracted from whistles or clicks.




