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
  list(df=test_events)
}

processdataDelphinID <- function(db_con, dateRange, ctable=NULL, wtable=NULL) {
  ctable <- gsub(' ', '', paste(ctable, '_Predictions'))
  wtable <- gsub(' ', '', paste(wtable, '_Predictions'))
  print(sprintf('%s - %s', ctable, wtable))
  cdata <- dbGetQuery(db_con, paste0(sprintf("SELECT * FROM %s", ctable)))
  wdata <- dbGetQuery(db_con, paste0(sprintf("SELECT * FROM %s", wtable)))
  SAQ <- dbGetQuery(db_con, paste0("SELECT * FROM Sound_Acquisition"))
  wdata$UTC <- as.POSIXct(wdata$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
  cdata$UTC <- as.POSIXct(cdata$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
  SAQ$UTC <- as.POSIXct(SAQ$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
  cspecies <- c('Dde', 'Ggr', 'Gme', 'Lal', 'Ttr')
  wspecies <- c('Dde', 'Ggr', 'Gme', 'Lac', 'Lal', 'Oor', 'Ttr')
  
  # Filter data based on date range
  if (!is.null(dateRange)) {
    cdata <- cdata[cdata$UTC >= dateRange[1] & cdata$UTC <= dateRange[2] + 1, ]
    wdata <- wdata[wdata$UTC >= dateRange[1] & wdata$UTC <= dateRange[2] + 1, ]
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
        csub <- subset(cdata, UTC > t0 & UTC <= t1)
        wsub <- subset(wdata, UTC > t0 & UTC <= t1)
        
        if (length(dim(csub)) > 0) {
          if (dim(csub)[1] > 0) {
            count <- 1
            namesc <- c()
            for (sp in cspecies) {
              namesc <- append(namesc, sprintf('%s_c', sp))
              votes <- c()
              for (i in 1:dim(csub)[1]){
                preds <- csub$Predicition[i]
                p <- as.numeric(fromJSON(preds)$predictions[[count]])
                votes <- append(votes, p)
              }
              csub[[sp]] <- votes
              count <- count + 1
            }
          } else {
            csub <- data.frame(t(data.frame(rep(NaN, length(cspecies)))))
            rownames(csub) <- c(1)
            names(csub) <- cspecies
          }
        } else {
          csub <- data.frame(t(data.frame(rep(NaN, length(cspecies)))))
          rownames(csub) <- c(1)
          names(csub) <- cspecies
        }
        
        if (length(dim(wsub)) > 0) {
          if (dim(wsub)[1] > 0) {
            count <- 1
            namesw <- c()
            for (sp in wspecies) {
              namesw <- append(namesw, sprintf('%s_w', sp))
              votes <- c()
              for (i in 1:dim(wsub)[1]){
                preds <- wsub$Predicition[i]
                p <- as.numeric(fromJSON(preds)$predictions[[count]])
                votes <- append(votes, p)
              }
              wsub[[sp]] <- votes
              count <- count + 1
            }
          } else {
            wsub <- data.frame(t(data.frame(rep(NaN, length(wspecies)))))
            rownames(wsub) <- c(1)
            names(wsub) <- wspecies
          }
        } else {
          wsub <- data.frame(t(data.frame(rep(NaN, length(wspecies)))))
          rownames(wsub) <- c(1)
          names(wsub) <- wspecies
        }
            dfc <- csub
            dfw <- wsub
            print(wsub)
          
            dfc_e <- colMeans(dfc[,cspecies])/sum(colMeans(dfc[,cspecies]))
            if (any(is.na(dfc_e))) {
              for (name in names(dfc_e)) {
                dfc_e[[name]] <- runif(1,10,15)
              }
              dfc_e <- dfc_e/sum(dfc_e)
            }

            namesc <- c('Dde_c', 'Ggr_c', 'Gme_c', 'Lal_c', 'Ttr_c')
            
            dfw_e <- colMeans(dfw[,wspecies])/sum(colMeans(dfw[,wspecies]))
            if (any(is.na(dfw_e))) {
              for (name in names(dfw_e)) {
                dfw_e[[name]] <- runif(1,10,15)
              }
              dfw_e <- dfw_e/sum(dfw_e)
            }
            
            names(dfc_e) <- namesc
            names(dfw_e) <- namesw
            xtest <- data.frame(cbind(t(dfc_e), t(dfw_e)))
            print(xtest)
            
            startT <- min(c(csub$UTC[1], wsub$UTC[1]))
            endT <- max(c(csub$UTC[dim(csub)[1]], wsub$UTC[dim(wsub)[1]]))
            evDur <- as.numeric(seconds(difftime(endT, startT)))*60

            xtest$startUTC <- startT
            xtest$endUTC <- endT
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
  list(df=test_events)
  }


