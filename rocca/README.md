## Random Forest acoustic classifiers for delphinid species
[classify_main.py](https://github.com/tristankleyn/which.dolphin/blob/main/rocca/classify_main.py) contains code to train and test Random Forest classifiers based on acoustic measurements extracted using the ROCCA (Real-time Odontocete Call Classification Algorithm) module in [PAMGuard](https://www.pamguard.org/). Existing classifiers developed for different geographic regions covering different compositions of delphinid species can be downloaded [here](https://www.pamguard.org/rocca/rocca.html) and are intended for use within PAMGuard's ROCCA module. See [ROCCA User's Manual](https://www.navymarinespeciesmonitoring.us/files/5413/9422/0614/Rocca_User_Manual_Revised_FINAL.pdf) (Oswald & Oswald, 2013) for guidance and documentation.

### rocca/

#### └── [classify_main.py](https://github.com/tristankleyn/which.dolphin/blob/main/rocca/classify_main.py)
Python script for training, evaluating, and exporting Random Forest classifiers. 

#### └── [classify_functions.py](https://github.com/tristankleyn/which.dolphin/blob/main/rocca/classify_functions.py)

Python script containing functions required for classify_main script


