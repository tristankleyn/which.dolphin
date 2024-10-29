# delphinID - deep acoustic classification for delphinid species
## Overview
This repostiory contains code for training and testing delphinID models, convolutional neural networks designed to accurately identify delphinid species by latent features in the frequency spectra of their echolocation click and whistle vocalizations, detected from passive acoustic recordings using [PAMGuard software](https://www.pamguard.org/). Code is available here in following structure:

### classify/

#### └── pycache/

Folder containing bytecode for Python application

#### └── models/

Folder containing whistle and click classifier models, formatted as .joblib files and built using [Scikit-Learn](https://scikit-learn.org/stable/) for different species combinations to use in app.

#### └── html/

Folder containing .txt files storing HTML content for the application.

#### └── www/

Folder containing images sourced by application.

#### └── app.py

Python script of application

#### └── requirements.txt

Text file containing list of Python packages required to run app.py

## Running the classifier app
