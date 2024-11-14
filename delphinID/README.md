![Alt text](images/logo_2.PNG)
##
This repostiory contains code for training and testing delphinID models, convolutional neural networks designed to accurately identify delphinid species by latent features in the frequency spectra of their echolocation click and whistle vocalizations, detected from passive acoustic recordings using [PAMGuard software](https://www.pamguard.org/). Code is available in the following scripts:

### delphinID/

#### └── [classify_main.py](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/classify_main.py)
Python script for training, evaluating, and exporting delphinID models.

#### └── [classify_functions.py](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/classify_functions.py)

Python script containing functions required for classify_main script

#### └── [compiledata_main.R](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/compiledata_main.R)

R script for extracting features from PAMGuard detections and preparing data in format required for training and evaluating models.

#### └── [compiledata_functions.R](https://github.com/tristankleyn/which.dolphin/blob/main/delphinID/compiledata_functions.R)

R script containing functions required for compiledata_main script

## User manual
### Terminology
| Term | Description |
|-----------------|-----------------|
| Accuracy | The proportion of correct (out of total) predictions made by a classifier.  |
| Bandwidth | A range of frequencies (Hz) |
| Broadband | Containing a wide range of frequencies (Hz) |
| Click | Broadband pulse signal used by dolphins for echolocation |
| Detection frame | A 2D representation of PAMGuard detections containing one or more vocalizations. Peak-frequency contours are plotted across frequency (Hz, vertical axis) and time (sec, horizontal axis) to represent whistles in whistle detection frames. Average normalised power spectra are plotted across proportion of energy (vertical axis) and frequency (Hz, horizontal axis) in click detection frames. |
| Encounter | A period of time containing one or more dolphin vocalizations. Independent encounters are spatiotemporally separate from one another (at least 1 hour or 5 km apart). | 
| Narrowband | Containing a narrow range of frequencies (Hz) | 
| Nyquist frequency | The maximum frequency (Hz) that can be detected (half the sampling rate) | 
| Sampling rate | The frequency (Hz) at which audio samples are recorded | 
| Spectrogram | A 2D representation of sound showing the frequency (Hz, vertical axis) and amplitude (colour intensity) of sound over time (sec, horizontal axis) | 
| Whistle | A narrowband signal used by dolphins for social communication. |

### Overview
delphinID is a deep learning-based framework designed for identifying delphinid species in underwater recordings (Kleyn et al., in prep.). Written and trained in Python for identifying seven commonly occurring North Atlantic species, it uses detections from the Click Detector and Whistle & Moan Detector module in the open-source PAMGuard software (Gillespie, 2008). to inform species prediction. A defining element of delphinID’s ability to accurately classify delphinid events to species is its utilisation of readily available tools, alongside conservative post-processing of detections, to reduce the amount of ‘hard work’ required from its neural networks. PAMGuard’s whistle and click detection algorithms extract acoustic profiles of vocalizations and are flexible in design, providing a “human in the loop” step for assuring quality of data before input to classification. delphinID learns latent species-specific patterns in average spectra of whistle and click detections to inform species prediction based on both types of vocalization. Below is a brief description of the delphinID classification workflow.

#### Principles of Operation 
There are four main stages to classifying acoustic events with delphinids using delphinID.
##### 1)	Detect echolocation clicks
##### 2)	Detect narrowband whistle fragments
##### 3)	Classification of detection frames (using whistles and clicks separately)
##### 4)	Classification of events (using whistles and clicks together)

![Alt text](images/workflow_1.PNG)

delphinID is a set of deep learning models coded in Python 3.0 for classifying dolphin vocalizations to species. The current models available are tested for classifying seven species found in the northeast Atlantic Ocean. The models classify acoustic representations of clicks and whistle fragments detected in PAMGuard to inform species identity. Detections must be made using the pre-existing Click Detector and Whistle and Moan Detector modules in PAMGuard. Acoustic representations of clicks and whistle fragments, which are made separately and herein referred to as detection frames, are normalised arrays containing the average frequency power spectra of all whistle fragments or clicks detected within a rolling 4-second window. The delphinID click and whistle models predict species identity based on these frames and predictions are accumulated over time to inform an overall prediction for each acoustic encounter Finally, encounter predictions from the separate whistle and click models are combined into a merged feature which is used by a Random Forest model to form a final prediction based on both vocalization types. For full details on the methods used and model evaluation, please refer to our publication (Kleyn et al., in prep.). 

##### What is a detection frame?
Detection frames are the units of classification used by delphinID, representing the average frequency power spectra of either whistles or clicks detected across a 4-second time window. For whistles, spectra are calculated between 2-20 kHz while for clicks spectra range from 10-40 kHz. Higher frequencies may contain information that could improve classification but were not used due to the high proportion of available training data that were restricted in sampling rate to 96 kHz or below and also due to interferring noise sources across several data sources between 40-48 kHz. The frequency ranges selected thus represented a range where useful information pertaining to the frequency content of whistles and clicks could be reliably extracted from detections and used to train delphinID models. A benefit of using detection frames over spectrogram images, which have been traditionally used for machine learning classification problems in acoustics, is that 100% of the information contained in a detection frame is relevant to the signals of interest. While the influence of background noise is difficult to prevent entirely, this technique significantly limits it compared with spectrogram-based approaches. As mentioned previously, automated whistle contour extraction algorithms such as the Whistle & Moan Detector in PAMGuard are known to produce fragemented detections, often missing a significant portion of a whistle contour. A benefit of using detection frames as classification features over parameters measured from individual whistles, as has been done extensively in previous classification studies, is that the average frequency power spectra of whistle across 4-second windows is relatively robust across different degrees of fragmentation compared with individual whistle measurements. 

### Configuring supporting PAMGuard modules
#### a) Click Detector
Click detections made using the PAMGuard Click Detector (REF) are used as input for delphinID’s click classification. The Click Detector module identifies likely echolocation clicks by monitoring energy level changes in frequency bands over time (REF). As the delphinID click classifier was trained on thousands of verified, high signal-to-noise ratio click detections, it is the responsibility of the user to ensure accurate click detection. This is only possible through use of appropriate Click Detector settings, of which there are many. There are several ways to improve click detection (see PAMGuard Help menu), three of which are described below:

##### 1.	Digital pre-filter and digital trigger filter
Pre-filters and trigger filters help to focus detection on the frequency band of interest. For training the delphinID click model, an IIR Butterworth high-pass filter was used for pre-filtering and trigger filtering. A fourth-order high-pass was set at 500 Hz for pre-filtering, while a second-order high pass was set at 2000 Hz for trigger filtering.

##### Detection Parameters  Trigger
A high trigger threshold can be used to eliminate false positive detections. While this can often result in missed detections, it increases the likelihood of detections being true. A conservative minimum trigger threshold of +20 dB was used to filter detections used to train and evaluate the delphinID model.

##### Click classification
Classifying clicks based on spectral properties helps to distinguish noise from biological signals. Clicks of the seven species targeted by delphinID for classification all contain a high proportion of energy between 15 and 120 kHz. Limited by the sampling rates of the recordings in the dataset available, delphinID only analyses spectral properties up to 40 kHz. We thus used a simple click classification scheme to filter out detections, where those that did not contain a significant proportion of energy above 15 kHz were discarded. 

#### b) Whistle and Moan Detector
Whistle detections made using the PAMGuard Whistle & Moan Detector (Gillespie et al., 2013) are used as input for delphinID’s whistle classification. The Whistle & Moan detection scans the spectrogram array produced by PAMGuard’s FFT Engine module for local time-frequency regions of peak frequency, that is, time-frequency pixels showing high intensity relative to their neighbours. Whistles are characteristically long duration (generally 50-3000 ms) and narrowband. The Whistle & Moan Detector joins time-frequency pixels of peak frequency together according to parameters set by the user. It’s ability to connect these regions to accurately trace whistles depends both on the FFT representation (time and frequency resolution of the spectrogram) and the background noise of the recording analysed. As with clicks, accurate detection is assumed prior to classification and is the responsibility of the user. Two examples of paramters that can be adjusted to improve the accuracy of whistle detection are **Minimum/maximum frequency** and **Threshold**:

##### Minimum/maximum frequency
delphinID’s whistle classifier uses the entire 2-20 kHz frequency band to classify whistle vocalizations, so generally speaking, this is the band that should be used for detection. However, noisy recordings may contain periodically repeating sound sources within this range that can be falsely detected by the Whistle & Moan Detector. In this case, and if the user notices no evident biological signals in the same frequency range as the noise, the frequency range of detection can be adjusted to minimise false detections.

##### Threshold
The detection threshold of the Whistle & Moan Detector works similarly to that of the Click Detector, limiting detections to those only showing a signal-to-noise ratio above a dB threshold. This threshold, which was set consistently at +6 dB for training and testing the delphinID whistle classifier, should be adjusted to prioritise minimisation of the proportion of false positive detections (even at the cost of false negative detections).

### Performance metrics & the importance of quality detections
While performance varies widely between species for classifying events based on solely whistles (**species accuracy/precision 0.41-0.72/0.44-0.73; mean 0.54/0.56; F1=0.30**) or clicks (**species accuracy/precision 0.52-0.96/0.50-1.00; mean 0.76/0.75; F1=0.57**) alone, classification accuracy is more consistent for events containing both types of vocalization (**species accuracy/precision 0.80-1.00/0.85-1.00; mean 0.88/0.086; F1=0.76**).

delphinID classification performance (F1 score = accuracy x precision)
| Species | Event, whistles only | Event, clicks only | Event, whistles and clicks |
|-----------------|-----------------|-----------------|-----------------|
| Delphinus delphis | 0.20 | 0.60 |  |
| Grampus griseus | 0.29/0.35 | 0.89/0. |  |
| Globicephala melas | Containing a wide range of frequencies (Hz) |
| Lagenorhynchus acutus | Broadband pulse signal used by dolphins for echolocation |
| Lagenorhynchus albirostris | A 2D representation of PAMGuard detections containing one or more vocalizations. Peak-frequency contours are plotted across frequency (Hz, vertical axis) and time (sec, horizontal axis) to represent whistles in whistle detection frames. Average normalised power spectra are plotted across proportion of energy (vertical axis) and frequency (Hz, horizontal axis) in click detection frames. |
| Orcinus orca | A period of time containing one or more dolphin vocalizations. Independent encounters are spatiotemporally separate from one another (at least 1 hour or 5 km apart). | 
| Tursiops truncatus | Containing a narrow range of frequencies (Hz) | 


Performance is, however, highly sensitive to the quality of the detections fed into it. We therefore encourage some form of quality assurance when running automated detection of clicks and whistles for use with delphinID suggest prioritising a low false detection rate over avoiding missed detections (which can be unavoidable in noisy recordings). The performance metrics cited are based on classifiers trained and evaluate using certain thresholds for discarding  low signal-to-noise or false detections. 





