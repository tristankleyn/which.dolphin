
## Integrate information from whistles and clicks to classify delphinid events.
<p align="justify">
This folder contains a graphical user <em>Shiny</em> interface for integrating predictions from ROCCA or delphinID classifiers to classify events to species based on information from  whistles and clicks. This workflow for integrating information from multiple sources of acoustic data was motivated by successful classification in prior studies [1] [2] [3] using similar methods, such as <em>BANTER</em> classification [2]. Continue reading to learn how to use the prediction outputs from classifiers used in PAMGuard to classify acoustic events using our eventClassifier interface.
</p>

### eventClassifier/

#### └── www/

> #### └── EventClassifier_ROCCA.rds
> Random Forest model for classifying events to species based on the output of ROCCA classifiers.

> #### └── EventClassifier_delphinID.rds
> Random Forest model for classifying events to species based on the output of delphinID classifiers.

> #### └── trackDB.sqlite3
> Database for storing base classification output from PAMGuard.

> #### └── [app.R](https://github.com/tristankleyn/which.dolphin/blob/main/eventClassifier/app.R)
> _Shiny_ application for classifying events. 

> #### └── [runApp.R](https://github.com/tristankleyn/which.dolphin/blob/main/eventClassifier/runApp.R)
> Script for installing required packages and running _Shiny_ app.

> #### └── [requirements.txt](https://github.com/tristankleyn/which.dolphin/blob/main/eventClassifier/requirements.txt)
> Packages required for running eventClassifier application.

##
### The eventClassifier interface
##
#### Run runApp.R script to launch eventClassifier interface
<p align="justify">
You can run the runApp.R script either by dragging its file into an R console window or sourcing the file within the console directly. This script will install any packages on your device required for the eventClassifier interface to function (see requirements.txt) before the launching the interface in a browser window.
</p>

```R
source('---INPUT PATH---/which.dolphin-main/eventClassifier/runApp.R')
```
##
#### Select PAMGuard database to monitor delphinid event classifications using ROCCA or delphinID classifier output
<p align="justify">
Below is a screenshot the eventClassifier interface displaying classification results for an example database containing classification output from delphinID whistle and click classifiers.
</p>

![image](https://github.com/user-attachments/assets/8b687701-01e4-4435-89a0-f44bfa621478)



##

### Easy transfer learning with eventClassifier
<p align="justify">
The "Add Labels" function in eventClassifier can be used to assign new labels to events. These labels can then in turn be used to train and evaluate a new Random Forest event classifier, which can be done automatically via the "Create new classifier" option. New event classifiers are trained using the event barcodes, feature vectors representing probabilities of classification for the original seven northeast Atlantic species the base delphinID or ROCCA classifiers were trained on based on whistles or clicks. While these barcodes were originally intended as direct predictors for the Atlantic species, they can be repurposed in eventClassifier for training a new model on a novel composition of labels. Below are screenshots of the labelling functionality in eventClassifier and the displayed output after creating a new classifier on user-created labels. New models and diagnostic reports are automatically exported to the eventClassifier folder.
</p>

<div align="center">
  <img src="https://github.com/user-attachments/assets/4d3f159a-26e2-4239-a3e6-ce1c0f12667c" alt="Labeling Functionality">
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/0e44e883-58c5-494a-8546-7f73cb9b34f9" alt="New Classifier Output">
</div>

  
##
### FAQ's & Comments
##
#### ❓ What's the difference between ROCCA and delphinID classifiers? 
<p align="justify">
ROCCA uses Random Forest analysis to predict species based on measured characteristics of whistle contours and click spectra, while delphinID uses deep learning to predict species based on average spectra of whistle contours and clicks. Both classifier types can run using automated detections made in PAMGuard.
</p>

#### ❓ What are the best settings to use for decision score and minimum whistles and clicks?
<p align="justify">
Generally, we find that both ROCCA and delphinID classifiers classify with higher accuracy when using a higher decision score threshold. This improvement, however, comes at the cost of discarding a portion of classifications, which is higher for higher decision score thresholds. 
</p>

#### 💡 Our classifiers are only as good as the detections you feed it!
<p align="justify">
The tools desribed here are not detectors - they are designed to classify detections of delphinid vocalizations to species level. ROCCA and delphinID classifiers were developed and tested using high signal-to-noise ratio detections, very few of which were false detections. Thus, when using our classifiers to classify novel data, it is highly beneficial to validate a portion of your detections to ensure false detection rates are minimised prior to classification.
</p>




## References

[1] Lu, Y., Mellinger, D. and Klinck, H., 2013, June. Joint classification of whistles and echolocation clicks from odontocetes. In Proceedings of Meetings on Acoustics (Vol. 19, No. 1). AIP Publishing.

[2] Rankin, S., Archer, F., Keating, J.L., Oswald, J.N., Oswald, M., Curtis, A. and Barlow, J., 2017. Acoustic classification of dolphins in the California Current using whistles, echolocation clicks, and burst pulses. Marine Mammal Science, 33(2), pp.520-540.

[3] Rankin, S., Sakai, T., Archer, F.I., Barlow, J., Cholewiak, D., DeAngelis, A.I., McCullough, J.L., Oleson, E.M., Simonis, A.E., Soldevilla, M.S. and Trickey, J.S., 2024. Open-source machine learning BANTER acoustic classification of beaked whale echolocation pulses. Ecological Informatics, 80, p.102511.


