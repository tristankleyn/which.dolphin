# (1) Set the appropriate working directory
setwd('___SET PATH TO FOLDER CONTAINING eventClassifier FOLDER___')

# (2) Load list of required packages
packages <- readLines("eventClassifier/requirements.txt")

# (3) Install required packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# (4) Run app
shiny::runApp('eventClassifier')
