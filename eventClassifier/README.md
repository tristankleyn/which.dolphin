<img width="545" alt="image" src="https://github.com/user-attachments/assets/e1b7b051-fd61-4eb1-b3de-738e41bb0cfb" />

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
#### Run runApp.R script to launch eventClassifier interface
<p align="justify">
You can run the runApp.R script either by dragging its file into an R console window or sourcing the file within the console directly. This script will install any packages on your device required for the eventClassifier interface to function (see requirements.txt) before the launching the interface in a browser window.
</p>

```R
source('---INPUT PATH---/which.dolphin-main/eventClassifier/runApp.R')
```
##
#### Select example of tracking database to monitor delphinid event classifications using ROCCA or delphinID classifiers
<p align="justify">
Below is a screenshot the eventClassifier interface displaying classification results for an example database containing classification output from delphinID whistle and click classifiers.
</p>

![image](https://github.com/user-attachments/assets/5815e106-58be-4670-9dfa-a024e55a3484)

<p align="justify">
1. Select the type of classifiers (ROCCA or delphinID) used for classifying whistles and clicks in PAMGuard.
2. Choose a database to identify and classify acoustic events.
3-4. Select the database tables containing click and whistle classifier predictions (not needed for ROCCA)
5. Select a decision score threshold below which event classifications are discarded from the displayed results.
6-8. Select thresholds for the minimum number of click and/or whistle predictions to use for event classification.
9. Filter events by range of dates.
10. Choose between plot displays (classification counts or classification map)
11. Click the 'Classify' button to initialise or refresh results
12. Download classification reuslts for all events or only events above thresholds.
13. Plot display (classification counts or classification map)
14. Table display, showing classification results for each identified event.
</p>
  
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


