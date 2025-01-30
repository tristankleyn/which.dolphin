## Integrate information from whistles and clicks to classify delphinid events.
<p align="justify">
This folder contains a graphical user <em>Shiny</em> interface for integrating predictions from ROCCA or delphinID classifiers to classify events to species based on information from  whistles and clicks. This workflow for integrating information from multiple sources of acoustic data was motivated by successful classification in prior studies [1] [2] [3] using similar methods, such as the <em>BANTER</em> method proposed in Rankin <em>et al.,</em> (2017). Continue reading to learn how to use the prediction outputs from classifiers used in PAMGuard to classify acoustic events using our eventClassifier interface.
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
#### 1. Run runApp.R script to launch eventClassifier interface
<p align="justify">
You can run the runApp.R script either by dragging its file into an R console window or sourcing the file within the console directly. This script will install any packages on your device required for the eventClassifier interface to function (see requirements.txt) before the launching the interface in a browser window.
</p>

```R
source('---INPUT PATH---/which.dolphin-main/eventClassifier/runApp.R')
```
##
#### 2. Select database containing output of PAMGuard classifiers (delphinID or ROCCA)
<p align="justify">
The event classifier app requires upload of PAMGuard databases (.sqlite3) containing information pertaining to the sound files used and the classification output from either the ROCCA or Deep Learning module. These data are stored in the <em>Sound_Aquisition</em> and <em>ROCCA_Whistle_Stats</em> or <em>Deep_Learning</em> tables, respectively. Note that upon selecting the database file, the app may take up to several minutes to process files containing many acoustic events, which are defined as individual files analysed in PAMGuard. To process smaller batches of data, users can adjust the period of days from which data are analysed. Users may then specify the minimum number of whistles or clicks required per event for classification, as well as minimum decision score threshold. The decision score is defined as the product of the maximum species likelihood for a given prediction and the difference between the first and second species likelihoods for the same prediction. For example, if the most likely species predicted for an event was predicted with a likelihood of 0.40 and the second most likely species with a likelihood of 0.30, the decision score would be equal to 0.40 x (0.40-0.30) = 0.04. 
</p>

![image](https://github.com/user-attachments/assets/d4584e59-3dde-4022-8159-5486d322a0d4)


##
#### 3. View classification results and adjust thresholds to filter predictions
<p align="justify">
Upon finishing uploading and classifying the data, a visual display will appear to the right of the control panel showing classification results. A barplot shows the number of classifications for each species, while a pageable table below provides further information on classified events: The <strong>eventID</strong> column specifies the filename of an acoustic event, <strong>clicks</strong> shows the number of click predictions used by the event classifier, <strong>whistles</strong> shows the number of whistle predictions used by the event classifier, <strong>predictedSpecies</strong> gives the species predicted by the event classifer, <strong>score</strong> shows the decision score for the event, and columns further to the right show classification likelihoods for each individual species included by the event classifier.
</p>

![image](https://github.com/user-attachments/assets/aa368f76-fb9d-4f2b-9640-8b1f42eb5dea)



##
#### 4. Export classified events
<p align="justify">
Users can either export all classified events contained within the range of dates specified or only classified events above the decision thresholds applied. Results are saved to the eventClassifier folder in .csv format.
</p>

![image](https://github.com/user-attachments/assets/53a70ce9-475a-42a9-9975-cd09352dda7e)


## References

[1] Lu, Y., Mellinger, D. and Klinck, H., 2013, June. Joint classification of whistles and echolocation clicks from odontocetes. In Proceedings of Meetings on Acoustics (Vol. 19, No. 1). AIP Publishing.

[2] Rankin, S., Archer, F., Keating, J.L., Oswald, J.N., Oswald, M., Curtis, A. and Barlow, J., 2017. Acoustic classification of dolphins in the California Current using whistles, echolocation clicks, and burst pulses. Marine Mammal Science, 33(2), pp.520-540.

[3] Rankin, S., Sakai, T., Archer, F.I., Barlow, J., Cholewiak, D., DeAngelis, A.I., McCullough, J.L., Oleson, E.M., Simonis, A.E., Soldevilla, M.S. and Trickey, J.S., 2024. Open-source machine learning BANTER acoustic classification of beaked whale echolocation pulses. Ecological Informatics, 80, p.102511.


