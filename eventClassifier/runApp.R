# (1) Set the appropriate working directory
dir_path <- tcltk::tk_choose.dir(caption="Select folder with eventClassifier")
setwd(dir_path)

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

