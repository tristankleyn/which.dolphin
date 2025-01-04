# (1) Set the appropriate working directory
setwd('C:/Users/tk81/Downloads/delphinID/setup/eventClassifier')

# (2) Load list of required packages
packages <- readLines("requirements.txt")

# (3) Install required packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# (4) Run app
shiny::runApp('C:/Users/tk81/Downloads/delphinID/setup/eventClassifier')
