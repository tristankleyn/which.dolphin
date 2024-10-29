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
| Row 1, Column 1 | Row 1, Column 2 |
| Row 2, Column 1 | Row 2, Column 2 |
| Row 3, Column 1 | Row 3, Column 2 |

