![Alt text](images/logo_2.PNG)
## Overview
This repostiory contains code for training and testing delphinID models, convolutional neural networks designed to accurately identify delphinid species by latent features in the frequency spectra of their echolocation click and whistle vocalizations, detected from passive acoustic recordings using [PAMGuard software](https://www.pamguard.org/). Code is available in the following scripts:

### delphinID/

#### └── classify_main.py

Python script for training, evaluating, and exporting delphinID models.

#### └── classify_functions.py

Python script containing functions required for classify_main script

#### └── compiledata_main.R

R script for extracting features from PAMGuard detections and preparing data in format required for training and evaluating models.

#### └── compiledata_functions.R

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
delphinID is a deep learning-based framework designed for identifying delphinid species in underwater recordings (Kleyn, Janik and Oswald, in prep.). Written and trained in Python for identifying seven commonly occurring North Atlantic species, it uses detections from the Click Detector and Whistle & Moan Detector module in the open-source PAMGuard software (Gillespie, 2008). to inform species prediction. 
A defining element of delphinID’s ability to accurately classify delphinid vocalizations to species is its utilisation of readily available tools, alongside carefully selected filters, to reduce the amount of ‘hard work’ required from its neural networks. PAMGuard’s whistle and click detection algorithms extract acoustic profiles of vocalizations and are flexible in design, providing a “human in the loop” step for assuring quality of data before input to classification.
Users of delphinID use these tools to feed detections into the classification model. Below is a brief description of the delphinID classification workflow.

#### Principles of Operation 
There are five main stages to classifying acoustic encounters with delphinids using delphinID (Figure A). 
1.	Detection of echolocation click vocalizations (clicks)
2.	Detection of narrowband whistle vocalizations (whistles)
3.	Selection of classification parameters in delphinID
4.	Classification of detection frames
5.	Classification of acoustic encounters

delphinID is a set of deep learning models coded in Python 3.0 for classifying dolphin vocalizations to species. The current models available are tested for classifying seven species found in the northeast Atlantic Ocean. The models classify acoustic representations of clicks and whistle fragments detected in PAMGuard to inform species identity. Detections must be made using the pre-existing Click Detector and Whistle and Moan Detector modules in PAMGuard. Acoustic representations of clicks and whistle fragments, which are made separately and herein referred to as detection frames, are normalised arrays containing the average frequency power spectra of all whistle fragments or clicks detected within a rolling 4-second window. The delphinID click and whistle models predict species identity based on these frames and predictions are accumulated over time to inform an overall prediction for each acoustic encounter Finally, encounter predictions from the separate whistle and click models are combined into a merged feature which is used by a Random Forest model to form a final prediction based on both vocalization types. For full details on the methods used and model evaluation, please refer to our publication (Kleyn et al., in prep.). 

### Configuring supporting PAMGuard modules
#### a) Click Detector
Click detections made using the PAMGuard Click Detector (REF) are used as input for delphinID’s click classification. The Click Detector module identifies likely echolocation clicks by monitoring energy level changes in frequency bands over time (REF). As the delphinID click classifier was trained on thousands of verified, high signal-to-noise ratio click detections, it is the responsibility of the user to ensure accurate click detection. This is only possible through use of appropriate Click Detector settings, of which there are many. Here, we recommend adjustments to three key Click Detector parameters to improve detection accuracy, though these recommendations should be tailored to the characteristics of the recording in question.
1.	Digital pre-filter and digital trigger filter
Pre-filters and trigger filters help to focus detection on the frequency band of interest. For training the delphinID click model, an IIR Butterworth high-pass filter was used for pre-filtering and trigger filtering. A fourth-order high-pass was set at 500 Hz for pre-filtering, while a second-order high pass was set at 2000 Hz for trigger filtering.

2.	Detection Parameters  Trigger
A high trigger threshold can be used to eliminate false positive detections. While this can often result in missed detections, it increases the likelihood of detections being true. A conservative minimum trigger threshold of +20 dB was used to filter detections used to train and evaluate the delphinID model.

3.	Click classification
Classifying clicks based on spectral properties helps to distinguish noise from biological signals. Clicks of the seven species targeted by delphinID for classification all contain a high proportion of energy between 15 and 120 kHz. Limited by the sampling rates of the recordings in the dataset available, delphinID only analyses spectral properties up to 48 kHz. We thus used a simple click classification scheme to filter out detections, where those that did not contain a significant proportion of energy above 15 kHz were discarded. These settings are shown in Figure B below.

#### b) Whistle and Moan Detector
Whistle detections made using the PAMGuard Whistle & Moan Detector (Gillespie et al., 2013) are used as input for delphinID’s whistle classification. The Whistle & Moan detection scans the spectrogram array produced by PAMGuard’s FFT Engine module for local time-frequency regions of peak frequency, that is, time-frequency pixels showing high intensity relative to their neighbours. Whistles are characteristically long duration (generally 50-3000 ms) and narrowband. The Whistle & Moan Detector joins time-frequency pixels of peak frequency together according to parameters set by the user. It’s ability to connect these regions to accurately trace whistles depends both on the FFT representation (time and frequency resolution of the spectrogram) and the background noise of the recording analysed. As with clicks, accurate detection is assumed prior to classification and is the responsibility of the user. Here are two key Whistle & Moan Detector parameters that can be adjusted to improve the accuracy of whistle detection:
1.	Minimum/maximum frequency
delphinID’s whistle classifier uses the entire 2-20 kHz frequency band to classify whistle vocalizations, so generally speaking, this is the band that should be used for detection. However, noisy recordings may contain periodically repeating sound sources within this range that can be falsely detected by the Whistle & Moan Detector. In this case, and if the user notices no evident biological signals in the same frequency range as the noise, the frequency range of detection can be adjusted to minimise false detections.

2.	Threshold
The detection threshold of the Whistle & Moan Detector works similarly to that of the Click Detector, limiting detections to those only showing a signal-to-noise ratio above a dB threshold. This threshold, which was set consistently at +6 dB for training and testing the delphinID whistle classifier, should be adjusted to prioritise minimisation of the proportion of false positive detections (even at the cost of false negative detections).

For both click and whistle detection, all settings used for training and evaluation the delphinID classifiers can be found in Appendix XXX. We strongly encourage users to manually monitor a portion of their detections until a satisfactorily low false positive rate is achieved. We again stress that delphinID does not require a large number of vocalizations to accurate classify to species. We encourage users to thus prioritise minimising false positive detections over minimising false negative (missed) detections. If a recording is very noisy with vocalizations exhibiting considerably low signal-to-noise ratios, it should not be used for classification. 



