## Random Forest acoustic classifiers for delphinid species

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14936662.svg)](https://doi.org/10.5281/zenodo.14936662)

[classify_main.py](https://github.com/tristankleyn/which.dolphin/blob/main/rocca/classify_main.py) contains code to train and test Random Forest classifiers based on acoustic measurements extracted using the ROCCA (Real-time Odontocete Call Classification Algorithm) module in [PAMGuard](https://www.pamguard.org/). Existing classifiers developed by Julie & Michael Oswald [1] for species compositions in different geographic regions can be accessed [here](https://www.pamguard.org/rocca/rocca.html) and used within PAMGuard's ROCCA module. Please refer to the [ROCCA User's Manual](https://www.navymarinespeciesmonitoring.us/files/5413/9422/0614/Rocca_User_Manual_Revised_FINAL.pdf) [1] for further documentation.

### rocca/

#### └── [classify_main.py](https://github.com/tristankleyn/which.dolphin/blob/main/rocca/classify_main.py)
Python script for training, evaluating, and exporting Random Forest classifiers. 

#### └── [classify_functions.py](https://github.com/tristankleyn/which.dolphin/blob/main/rocca/classify_functions.py)

Python script containing functions required for classify_main script

## References

[1] Oswald, J.N., and M. Oswald. 2013. ROCCA (Real-time Odontocete Call Classification Algorithm) User’s
Manual. Prepared for Naval Facilities Engineering Command Atlantic, Norfolk, Virginia under HDR
Environmental, Operations and Construction, Inc Contract No. CON005-4394-009, Subproject 164744,
Task Order 03, Agreement # 105067. Prepared by Bio-Waves, Inc., Encinitas, California.

