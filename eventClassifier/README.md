## Integrate whistles and clicks to classify delphinid events.
<p align="justify">
This folder contains a graphical user <em>Shiny</em> interface for integrating predictions from ROCCA or delphinID classifiers to classify events to species based on information from  whistles and clicks. Learn below how to use predictions exported from PAMGuard's ROCCA or Deep Learning modules in PAMGuard databases as input for event classification. 
</p>

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
<p align="justify">
The event classifier app requires upload of PAMGuard databases (.sqlite3) containing information pertaining to the sound files used and the classification output from either the ROCCA or Deep Learning module. These data are stored in the _Sound_Aquisition_ and _ROCCA_Whistle_Stats_ or _Deep_Learning_ tables, respectively. Note that upon selecting the database file, the app may take up to several minutes to process files containing many acoustic events, which are defined as individual files analysed in PAMGuard. To process smaller batches of data, users can adjust the period of days from which data are analysed. Users may then specify the minimum number of whistles or clicks required per event for classification, as well as minimum decision score threshold. The decision score is defined as the product of the maximum species likelihood for a given prediction and the difference between the first and second species likelihoods for the same prediction. For example, if the most likely species predicted for an event was predicted with a likelihood of 0.40 and the second most likely species with a likelihood of 0.30, the decision score would be equal to 0.40 x (0.40-0.30) = 0.04. 
</p>

![image](https://github.com/user-attachments/assets/963a3dac-71e8-4d71-a69e-927559f05c53)

##
#### 3. View classification results and adjust thresholds to filter predictions
<p align="justify">
Upon finishing uploading and classifying the data, a visual display will appear to the right of the control panel showing classification results. The scatterplot view colours event classifications by species and decision score (points further to up and to the right represent higher scores). The orange line represents the decision score threshold specified by the user to filter out predictions of low confidence. The table below provides further information on classified events: The <strong>eventID</strong> column specifies the filename of an acoustic event, <strong>clicks</strong> shows the number of click predictions used by the event classifier, <strong>whistles</strong> shows the number of whistle predictions used by the event classifier, <strong>predictedSpecies</strong> gives the species predicted by the event classifer, <strong>score</strong> shows the decision score for the event, and columns further to the right show classification likelihoods for each individual species included by the event classifier.
</p>

![image](https://github.com/user-attachments/assets/f1e49228-af05-4c24-8120-02584fb3767d)

##
#### 4. Export classified events
<p align="justify">
Users can either export all classified events contained within the range of dates specified or only classified events above the decision thresholds applied. Results are saved to the eventClassifier folder in .csv format.
</p>
