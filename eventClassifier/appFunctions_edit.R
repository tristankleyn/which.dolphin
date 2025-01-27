library(tidyverse)
library(rvest)
library(janitor)
library(dbplyr)
library(ggplot2)
library(RSQLite)
library(DBI)
library(shiny)
library(shinyWidgets)
library(randomForest)
library(jsonlite)

processdataRocca <- function(db_con, dateRange) {
  data <- dbGetQuery(db_con, paste0("SELECT * FROM Rocca_Whistle_Stats"))
  SAQ <- dbGetQuery(db_con, paste0("SELECT * FROM Sound_Acquisition"))
  data$UTC <- as.POSIXct(data$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
  SAQ$UTC <- as.POSIXct(SAQ$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
  specieslist <- c('Dde', 'Ggr', 'Gme', 'Lal', 'Oor', 'Ttr')
  
  # Filter data based on date range
  if (!is.null(dateRange)) {
    data <- data[data$UTC >= dateRange[1] & data$UTC <= dateRange[2] + 1, ]
    SAQ <- SAQ[SAQ$UTC >= dateRange[1] & SAQ$UTC <= dateRange[2] + 1, ]
  }
  
  event_count <- 1
  test_events <- data.frame()
  for (evID in unique(SAQ$SystemName)) {
    subSAQ <- subset(SAQ, SystemName == evID)
    if (dim(subSAQ)[1] > 0) {
      t0 <- NaN
      t1 <- NaN
      for (i in 1:dim(subSAQ)[1]) {
        if (grepl('Start', subSAQ$Status[i])) {
          t0 <- as.POSIXct(subSAQ$UTC[i], '%Y-%m-%d %H:%M:%OS', tz='UTC')
        } else if (grepl('Stop', subSAQ$Status[i])) {
          t1 <- as.POSIXct(subSAQ$UTC[i], '%Y-%m-%d %H:%M:%OS', tz='UTC')
        }
      }
      
      if (!is.na(t0) & !is.na(t1)) {
        sub <- subset(data, UTC > t0 & UTC <= t1)
        if (length(dim(sub)) > 0) {
          if (dim(sub)[1] > 0) {
            count <- 1
            namesc <- c()
            namesw <- c()
            for (sp in specieslist) {
              namesc <- append(namesc, sprintf('%s_c', sp))
              namesw <- append(namesw, sprintf('%s_w', sp))
              votes <- c()
              for (i in 1:dim(sub)[1]){
                vl <- sub$voteList[i]
                endind <- gregexpr(')', vl)[[1]][1]
                vl <- substr(vl, 2, endind-2)
                dash_inds <- as.numeric(gregexpr('-', vl)[[1]])
                if (count == 1) {
                  ind0 <- 1
                  ind1 <- dash_inds[count]-1
                  p <- as.numeric(substr(vl, ind0, ind1))
                } else if (count == 6) {
                  ind0 <- dash_inds[count-1] + 1
                  ind1 <- nchar(vl)
                  p <- as.numeric(substr(vl, ind0, ind1))
                } else {
                  ind0 <- dash_inds[count-1] + 1
                  ind1 <- dash_inds[count] - 1
                  p <- as.numeric(substr(vl, ind0, ind1))
                }
                votes <- append(votes, p)
              }
              sub[[sp]] <- votes
              count <- count + 1
            }
            
            dfc <- subset(sub, freqPeak >= 10000 & freqPeak <= 40000)
            dfw <- subset(sub, freqMin >= 2000 & freqMax <= 20000 & duration >= 0.2)
            
            dfc_e <- colMeans(dfc[,specieslist])/sum(colMeans(dfc[,specieslist]))
            if (any(is.na(dfc_e))) {
              for (name in names(dfc_e)) {
                dfc_e[[name]] <- runif(1,10,15)
              }
              dfc_e <- dfc_e/sum(dfc_e)
            }
            
            dfw_e <- colMeans(dfw[,specieslist])/sum(colMeans(dfw[,specieslist]))
            if (any(is.na(dfw_e))) {
              for (name in names(dfw_e)) {
                dfw_e[[name]] <- runif(1,10,15)
              }
              dfw_e <- dfw_e/sum(dfw_e)
            }
            
            names(dfc_e) <- namesc
            names(dfw_e) <- namesw
            xtest <- data.frame(cbind(t(dfc_e), t(dfw_e)))
            
            evDur <- as.numeric(seconds(difftime(sub$UTC[dim(sub)[1]], sub$UTC[1])))*60
            xtest$startUTC <- sub$UTC[1]
            xtest$endUTC <- sub$UTC[dim(sub)[1]]
            xtest$duration <- evDur
            xtest$eventID <- str_trim(as.character(evID))
            xtest$clicks <- dim(dfc)[1]
            xtest$whistles <- dim(dfw)[1]
            print(sprintf('Event %s start %s', event_count, xtest$startUTC))
            print(sprintf('Event %s end %s', event_count, xtest$endUTC))
            test_events <- rbind(test_events, xtest)
            sub <- data.frame()
            event_count <- event_count + 1
          }
        }
      }
    }
  }
  dbDisconnect(db_con)
  list(df=test_events)
}


processdataDelphinid <- function(db_con, dateRange) {
  data <- dbGetQuery(db_con, paste0("SELECT * FROM Deep_Learning_Classifier_Predictions"))
  SAQ <- dbGetQuery(db_con, paste0("SELECT * FROM Sound_Acquisition"))
  data$UTC <- as.POSIXct(data$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
  SAQ$UTC <- as.POSIXct(SAQ$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
  specieslist_w <- c('Dde', 'Ggr', 'Gme', 'Lac', 'Lal', 'Oor', 'Ttr')
  specieslist_c <- c('Dde', 'Ggr', 'Gme', 'Lal', 'Ttr')
}


table_c <- "Deep_Learning_Classifier_Predictions"
table_w <- "xxx"

db_con <- dbConnect(RSQLite::SQLite(), 'C:/Users/tk81/Downloads/PAMGuard_clicks_spectrum_test3.sqlite3')
tables <- RSQLite::dbListTables(db_con)
on.exit(dbDisconnect(db_con))
data_c <- dbGetQuery(db_con, paste0(sprintf("SELECT * FROM %s", table_c)))
SAQ <- dbGetQuery(db_con, paste0("SELECT * FROM Sound_Acquisition"))
dbDisconnect(db_con)

dateRange <- NULL
# Filter data based on date range
if (!is.null(dateRange)) {
  data_c <- data_c[data_c$UTC >= dateRange[1] & data_c$UTC <= dateRange[2] + 1, ]
  SAQ <- SAQ[SAQ$UTC >= dateRange[1] & SAQ$UTC <= dateRange[2] + 1, ]
}


event_count <- 1
test_events <- data.frame()
for (z in 1:1) {
  subSAQ <- subset(SAQ, SystemName == 'xxx')
  if (1 > 0) {
    t0 <- data_c$UTC[1]
    t1 <- data_c$UTC[nrow(data_c)]
#    for (i in 1:dim(subSAQ)[1]) {
#      if (grepl('Start', subSAQ$Status[i])) {
#        t0 <- as.POSIXct(subSAQ$UTC[i], '%Y-%m-%d %H:%M:%OS', tz='UTC')
#      } else if (grepl('Stop', subSAQ$Status[i])) {
#        t1 <- as.POSIXct(subSAQ$UTC[i], '%Y-%m-%d %H:%M:%OS', tz='UTC')
#      }
#    }
    
    if (!is.na(t0) & !is.na(t1)) {
      sub_c <- subset(data_c, UTC > t0 & UTC <= t1)
      if (length(dim(sub_c)) > 0) {
        if (dim(sub_c)[1] > 0) {
        
          preds <- data_c$Predicition
          dfc <- data.frame()
          for (i in 1:nrow(sub_c)) {
            p <- as.numeric(fromJSON(preds[i])$predictions)
            dfc <- rbind(dfc, t(data.frame(p)))
          }
          names(dfc) <- specieslist_c
          vals <- t(data.frame(colMeans(dfc)/sum(colMeans(dfc))))
          rownames(vals) <- c(1)
          
          xtest <- data.frame(startUTC=sub_c$UTC[1])
          evDur <- as.numeric(seconds(difftime(sub_c$UTC[dim(sub_c)[1]], sub_c$UTC[1])))*60
          xtest$endUTC <- sub_c$UTC[dim(sub_c)[1]]
          xtest$duration <- evDur
          xtest$eventID <- 'A'
#          xtest$eventID <- str_trim(as.character(evID))
          xtest$nc <- dim(dfc)[1]
          xtest <- cbind(xtest, vals)
          print(sprintf('Event %s start %s', event_count, xtest$startUTC))
          print(sprintf('Event %s end %s', event_count, xtest$endUTC))
          test_events <- rbind(test_events, xtest)
          sub <- data.frame()
          event_count <- event_count + 1
        }
      }
    }
  }
}

