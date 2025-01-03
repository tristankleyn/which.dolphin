<img src="./images/logo_2.PNG" alt="Example image" width="600">

##
This repository contains code for training and testing delphinID models, convolutional neural networks designed to accurately identify delphinid species by latent features in the frequency spectra of their echolocation click and whistle vocalizations, detected from passive acoustic recordings using [PAMGuard software](https://www.pamguard.org/). Code is available in the following scripts:

### delphinID/

#### └── [classify_main.py](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/classify_main.py)
Python script for training, evaluating, and exporting delphinID models. 

#### └── [classify_functions.py](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/classify_functions.py)

Python script containing functions required for classify_main script

#### └── [compiledata_main.R](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/compiledata_main.R)

R script for extracting features from PAMGuard detections and preparing data in format required for training and evaluating models.

#### └── [compiledata_functions.R](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/compiledata_functions.R)

R script containing functions required for compiledata_main script



## Using pre-trained classifiers for Northeast Atlantic delphinid species 
Trained models for classifying recordings of seven northeast Atlantic delphinid species (Short-beaked common dolpins _(Delphinus delphis)_, Common bottlenose dolphins _(Tursiops truncatus)_, Risso's dolphins _(Grampus griseus)_, Atlantic white-sided dolphins _(Lagenorhynchus acutus)_, white-beaked dolphins _(Lagenorhynchus albirostris)_, killer whales _(Orcinus orca)_, and long-finned pilot whales _(Globicephala melas)_) are available and citable for use [here](https://zenodo.org/records/14578299?preview=1).

The northeast Atlantic delphinid classifier predicts events with an average accuracy of 86.3% (90% CI 82.5-90.1%) across the seven species, ranging from 80% accuracy for short-beaked common dolphins to 92% for white-beaked dolphins. F1 score (accuracy x precision) is shown for each species below:

**Northeast Atlantic classifier performance (F1 score = accuracy x precision)**
| Species name | Event, whistles only | Event, clicks only | Event, whistles and clicks |
|-----------------|-----------------|-----------------|-----------------|
| Delphinus delphis | 0.20 | 0.60 | 0.50 |
| Grampus griseus | 0.11 | 0.80 | 0.70 |
| Globicephala melas | 0.14 | 0.46 | 0.65 |
| Lagenorhynchus acutus | 0.37 | --- | 0.58 |
| Lagenorhynchus albirostris | 0.54 | 0.86 | 0.90 |
| Orcinus orca | 0.57 | --- | 0.80 |
| Tursiops truncatus | 0.20 | 0.45 | 0.81 |
| **All species** | **0.30** | **0.57** | **0.76** |

## Train your own delphinID classifiers
The R and Python scripts described above can be used to train and test your own delphinID classifier models through the following steps:

1. Make sure the latest versions of [R](https://cran.r-project.org/) and [Python](https://www.python.org/downloads/) are installed on your device.

2. Clone or download this repository to your device
   `git clone https://github.com/tristankleyn/which.dolphin.git`
   

4. Detect whistle and click vocalizations in passive acoustic recordings. Support for running automatic detectors in PAMGuard can be found [here](https://www.pamguard.org/tutorials/getstarted.html), though classifiers can be trained with data from any software.

<p align="center">
   <img width="545" alt="image" src="https://github.com/user-attachments/assets/719973a2-1e02-4b2b-8803-1c10dccea0a9" />
   <br>
   <em>PAMGuard display showing bearings, waveforms, and spectra of click detections in passive acoustic data.</em>
</p>

6. Generate detection frame examples from whistle and click detections. [compiledata_main.R](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/compiledata_main.R) and its functions [compiledata_functions.R](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/compiledata_functions.R) can be used to generate detection frames for detections made in PAMGuard. Other software or custom methods can alternatively be used to generate detection frames. All detection frames to should be saved into .csv files within ./delphinID/data - this is done automatically when using the scripts provided.

<p align="center">
   <img width="580" alt="image" src="https://github.com/user-attachments/assets/101ba3b2-43da-424f-aef2-42df0e045deb" />
   <br>
   <em>Detection frames represent average frequencies present in detections within 4-second time windows.</em>
</p>

8. Train and evaluate classifier models using [classify_main.py](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/classify_main.py) and its functions [classify_functions.py](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/compiledata_functions.R). All examples in each unique encounter will form a separate testing set for evaluating a new classifier trained on all other encounters in the dataset, while models and results are exported to ./delphinID/data. Classification parameters used in the "classify_main.py" script, which are described in the table below, can be adjusted to achieve optimal results.

<p align="center">
   <img width="581" alt="image" src="https://github.com/user-attachments/assets/a726f747-546d-46c0-81b2-d0fbc5003645" />
   <br>
   <em>Classifiers are iteratively trained and tested using cross validation across all unique encounters in the dataset.</em>
</p>




#### Adjustable classification hyperparameters
| Parameter | Default | Description |
|-----------------|-----------------|-----------------|
| nmin | 3 | Minimum threshold for the number of clicks per detection frame to be used for classification |
| dd | (0.1, 100) | Minimum and maximum detection density of whistle detection frames to be used for classification |
| nmax | 30 | Maximum number of examples per encounter used for training |
| batch_size | 1 | Number of examples used for training before retraining internal model parameters |
| epochs | 20 | Number of training epochs for each bootstrap of training and validation data |
| partitions | 5 | Number of different partitions/bootstraps of training and validation data to train model on | 
| seed | 42 | Initial random seed for training |
| use_selectencs | False | Use custom list of select encounters for training |
| omit | [] | Custom list of select encounters to omit from training and testing |
| split | 0.33 | Proportion of training data used for validation in each training epoch |
| model_format | 'saved_model' | 'saved_model' or '.keras' format for saving CNN models |

#### Additional hyperparameters (recommended to be kept at default settings)
| Parameter | Default | Description |
|-----------------|-----------------|-----------------|
| resize | 1 | Factor to compress input arrays by (i.e. factor of 2 halves array length) | Compressing input arrays reduces training time/computational expense |
| nfiltersconv | 16 | Number of filters used in 1D convolutional layers in CNN model |
| kernelconv | 3 | Size of filters, or kernels, used in 1D convolutional layers in CNN model |
| padding | 'same' | Zero-pad input features to match output size of 1D convolutional layer in CNN model |
| maxpool | 2 | Size of sliding window for max pooling layer in CNN model |
| densesize | 10 | Size of dense layer in CNN model | 
| dropout | 0.2 | Proportion of neurons randomly discarded by dropout in each training step |
| patience | 20 | Number of epochs before early stopping callback during model training |





## FAQ's
##### What is a detection frame?
Detection frames are representations of the average frequency content in detections of clicks or whistles contained within a 4-second time window and are the input features used by delphinID classifier models. To train the northeast Atlantic delphinID classifiers, detection frames were produced for clicks and whistles in slightly different ways. Average frequency power spectra for individual clicks were calculated using the R package _PAMpal_ [[1]](https://taikisan21.github.io/PAMpal/) and then averaged together for all click detections within 4-second time windows and normalised to form detection frames. Whistle detection frames were instead calculated as arrays of the relative density of frequency values within detected whistle peak frequency contours. While limitations of our dataset restricted us to training our classifiers on low frequency spectra of clicks (10-40 kHz) and whistles (2-20 kHz), detection frames can be characterised across any desired frequency range. One main benefit of using a detection-based approach over one using spectrogram images as input is that a majority of the information contained in the input is relevant to our signals of interest, so long as detections of signals are accurate and of high signal-to-noise ratio. Training classifiers on inputs that are largely robust to background noise is likely to benefit their performance and generalisability. 

#### What settings should I use for detecting signals in PAMGuard?
Detailed support and tutorials for using automatic detectors in PAMGuard can be found at [pamguard.org](https://www.pamguard.org/tutorials/getstarted.html). For training accurate classifiers, users should target whistles and clicks of high signal-to-noise ratio that are unmasked by other loud sound sources. Certain settings within automatic detectors, such as trigger thresholds, trigger filters, and spectral criteria can be used to filter detections.

## References
[1] Sakai, T., 2020. PAMpal: Load and process passive acoustic data. R package version 0.9, 14.





