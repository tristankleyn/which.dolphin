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

##
### Using the event classifier app
##
#### 1. Run the below R code to download necessary packages and launch the eventClassifier app interface
```R
packages <- readLines("requirements.txt")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}
shiny::runApp('eventClassifier')
```
##
#### 2. Select database containing output of PAMGuard classifiers (delphinID or ROCCA)
![image](https://github.com/user-attachments/assets/963a3dac-71e8-4d71-a69e-927559f05c53)
##
#### 3. Adjust optional thresholds to filter predictions
![image](https://github.com/user-attachments/assets/f1e49228-af05-4c24-8120-02584fb3767d)
##
#### 4. Export classified events to .csv format
