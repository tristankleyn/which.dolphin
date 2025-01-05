## Integrate whistles and clicks to classify delphinid events.
This folder contains a graphical user _Shiny_ interface for integrating predictions from ROCCA or delphinID classifiers to classify events to species based on information from  whistles and clicks. Learn below how to use predictions exported from PAMGuard's ROCCA or Deep Learning modules in PAMGuard databases as input for event classification. 

### eventClassifier/

#### └── www/

#### └── EventClassifier_7sp.rds
Random Forest model for classifying events to species based on the output of ROCCA or delphinID classifiers.

#### └── [app.R](https://github.com/tristankleyn/which.dolphin/blob/main/eventClassifier/app.R)
_Shiny_ application for classifying events. 

#### └── [runApp.R](https://github.com/tristankleyn/which.dolphin/blob/main/eventClassifier/runApp.R)
R script for installing required packages and running _Shiny_ app.

#### └── [requirements.txt](https://github.com/tristankleyn/which.dolphin/blob/main/eventClassifier/requirements.txt)
R packages required for running eventClassifier application.

### How to use the eventClassifier application

