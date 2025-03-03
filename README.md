## Identify dolphin species acoustically with machine learning.
This repository contains code for developing and applying machine learning models for identifying delphinid species by acoustic characteristics of their vocalizations. Models are designed to be used in conjunction with the open-source acoustic analysis software [PAMGuard](https://www.pamguard.org/), while interested developers are welcome to suggest alterations or additions to any code.

![Alt text](images/methods_simple_1.PNG)

## How it works
Classifier models provided here are designed to work in a two-stage process: **base classifiers** classify individual delphinid vocalizations, or short segments containing multiple successive vocalizations, to species based on acoustic characteristics of either whistles or echolocation clicks. The output of the base whistle and click classifiers, which are lists of probabilities of classification for each species based on each vocalization, or each segment of vocalizations, and is then fed into an **event classifier**, which integrates predictions across the two base classifiers, synthesizing information from whistles and clicks across the entire event. This approach was motivated by the *BANTER* classification method outlined in [1]. We developed two separate methods for base classification of whistles and clicks, for which sub-repositories containing code and further information are linked below.

### which.dolphin/

> #### └── [delphinID](https://github.com/tristankleyn/which.dolphin/tree/main/delphinID)/
> Convolutional neural networks (CNNs) developed in Python TensorFlow and trained to classify species by latent characteristics in concatenated normalised frequency power spectra of whistles or clicks. 

> #### └── [ROCCA](https://github.com/tristankleyn/which.dolphin/tree/main/rocca)/
> Random Forest models developed in WEKA [2] to classify species based on select acoustic-temporal measurements extracted from whistles or clicks.

> #### └── [eventClassifier](https://github.com/tristankleyn/which.dolphin/tree/main/eventClassifier)/
> Graphical user interface for integrating predictions from base classifiers (ROCCA or delphinID) to form event-level species predictions for recordings containing delphinid vocalizations.

### Do-it-yourself 🔧
I've done my best to provide all information information and code required for anyone interested to develop their own delphinID or ROCCA classifiers from scratch - this option requires some familiarity with Python. I've also implemented a functionality in the eventClassifier interface whereby users can harness transfer learning to more intuitively train delphinID or ROCCA event classifiers on their own data - this option requires no coding at all. Find code and guidance for these options through the links below:

🌱 Develop [delphinID](https://github.com/tristankleyn/which.dolphin/tree/main/delphinID) or [ROCCA](https://github.com/tristankleyn/which.dolphin/tree/main/rocca) classifiers from scratch.
🧠 [Use transfer learning](https://github.com/tristankleyn/which.dolphin/tree/main/eventClassifier) to produce a new event classifier. 



## References
[1] Rankin, S., Archer, F., Keating, J.L., Oswald, J.N., Oswald, M., Curtis, A. and Barlow, J., 2017. Acoustic classification of dolphins in the California Current using whistles, echolocation clicks, and burst pulses. Marine Mammal Science, 33(2), pp.520-540.

[2] Oswald, J.N., and M. Oswald. 2013. ROCCA (Real-time Odontocete Call Classification Algorithm) User’s Manual. Prepared for Naval Facilities Engineering Command Atlantic, Norfolk, Virginia under HDR Environmental, Operations and Construction, Inc Contract No. CON005-4394-009, Subproject 164744, Task Order 03, Agreement # 105067. Prepared by Bio-Waves, Inc., Encinitas, California.
