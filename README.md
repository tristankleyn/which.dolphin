# delphinID - deep acoustic classification for delphinid species
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
| Column 1 Header | Column 2 Header |
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

