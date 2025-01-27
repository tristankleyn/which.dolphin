##IMPORT NECESSARY MODULES
library(PAMpal)
library(zoo)
library(lubridate)
library(RSQLite)
library(stringr)

savefolder <- 'I:/newSpectra1/220125'               #INPUT FOLDER PATH FOR SAVING RESULTS TO
binaryfolder <- 'I:/newSpectra1/binary'     #INPUT FOLDER PATH WHERE PAMGUARD BINARY DATA IS STORED
dbfolder <- 'I:/detections122024'                #INPUT FOLDER PATH WHERE PAMGUARD DATABASES ARE STORED

fileN1 <- 109
fileN2 <- length(dir(dbfolder))
site <- 'test'
selectdb <- NULL
singleFile <- NULL

GENERATE_CLICKS <- TRUE
GENERATE_WHISTLES <- FALSE
if (!is.null(selectdb)) { 
  binaryfolder <- gsub(' ', '', paste(binaryfolder, '/', selectdb))
  selectdb <- gsub(' ', '', paste(selectdb, '.sqlite3'))
  }

for (fileN in fileN1:fileN2) {
  filename <- dir(dbfolder)[fileN]
  if (!is.null(selectdb)) { 
    filename <- selectdb
  }
  print(sprintf('Loading file %s/%s: %s', fileN, fileN2, filename))
  dbname = sprintf('%s/%s', dbfolder, filename)
  con <- dbConnect(drv=RSQLite::SQLite(), dbname=dbname)
  tables <- dbListTables(con)
  if (length(tables) > 0) {
    saq <- dbGetQuery(conn=con, statement="SELECT * FROM Sound_Acquisition")
    saq$SystemName <- str_remove_all(saq$SystemName, " ")
    if (!is.null(singleFile)) {
      saq <- subset(saq, SystemName==singleFile)
    }
    for (item in unique(saq$Status)) {
      if (grepl("Start", item)) {
        saq <- subset(saq, Status == item)
      }
    }
    dbDisconnect(con)
    
    sr <- as.integer(saq$sampleRate[1])
    temp_res <- 0.0213333                       #INPUT TEMPORAL RESOLUTION OF FFT USED FOR WHISTLE DETECTION
    
    myStudy <- retrieveDetections(dbname=dbname, binaryfolder=binaryfolder, sr=sr)
    
    
    ##GENERATE CLICK DETECTION FRAMES
    if (GENERATE_CLICKS == TRUE) {
      clickInfo <- clickData(myStudy, fft=as.integer(.00533333*sr)+1, selectEvent=singleFile)
      cdata <- clickInfo[[1]]
      
      if (class(cdata) == 'data.frame') {
        if (dim(cdata)[1] > 1) {
          noise <- clickInfo[[4]]
          noise[is.na(noise)] <- 0
          
          newnoise <- c()
          for (i in 1:length(noise)) {
            if (i < length(noise)) {
              n1 <- noise[i]
              n2 <- noise[i+1]
              new <- seq(n1, n2, (n2-n1)/17)
              new <- new[1:length(new)-1]
              for (val in new) {
                newnoise <- append(newnoise, val)
              }
            } else {
              newnoise <- append(newnoise, noise[i])
            }
          }
          
          noise <- newnoise - min(newnoise)
          
          
          a <- saq$UTC[1]
          hr1 <- hour(a)
          min1 <- minute(a)
          sec1 <- second(a)
          time1 <- 3600*hr1 + 60*min1 + sec1
          
          a <- cdata$UTC[1]
          hr2 <- hour(a)
          min2 <- minute(a)
          sec2 <- second(a)
          time2 <- 3600*hr2 + 60*min2 + sec2
          
          dt <- time2 - time1
          cfilt <- filterClicks(cdata, dt=0)
          
          if (dim(cfilt)[1] > 1) {
            startrec <- cfilt$eventId[1]
            endrec <- cfilt$eventId[dim(cfilt)[1]]
            
            avSpec <- calculateAverageSpectra(myStudy, ev=startrec, window=FALSE,
                                              wl=as.integer(.00533333*sr)+1, noise=FALSE, 
                                              plot=c(FALSE, FALSE))
            
            spec <- avSpec$allSpec[,avSpec$UID %in% cfilt$UID]
            nspec <- noise
            
            if (sum(nspec) != 0) {
              if (is.null(dim(nspec))) {
                x <- nspec
                L <- 1
              } else {
                x <- rowMeans(nspec)
                L <- dim(nspec)[2]
              }
              y0 <- as.integer(10*length(x)/(sr/2000))
              y1 <- as.integer(40*length(x)/(sr/2000))
              n <- rollmeanr(x, k=3)
              n <- n[y0:y1]
              n <- n - min(n)
              n <- n/sum(n)
              ndata <- data.frame()
              ndata <- rbind(ndata, n)
              names(ndata) <- seq(1, dim(ndata)[2])
              ndata <- cbind(n=L, ndata)
              ndata <- cbind(rec_id=startrec, ndata)
              ndata <- cbind(site='none', ndata)
              rec0 <- startrec
            }    
            
            ind1 <- gregexpr('_', filename)[[1]][1]
            ind2 <- gregexpr('.sqlite', filename)[[1]][1]
            name <- substr(filename, ind1+1, ind2-1)
            if (!is.null(singleFile)) {
              f <- substr(singleFile, 1, nchar(singleFile)-4)
              name <- gsub(' ', '', paste(name, '_', f))
            }
            
            createClickSpectra(subsub=cfilt, spec=spec, nspec=nspec, site=site, nmin=3, sr=sr,
                               startrect0=0, startrec=startrec, endrec=endrec, include_empty = TRUE, remove_dups=FALSE,
                               dest=sprintf('%s/clickspectra_%s.csv', savefolder, name), file=NULL)
          }
          
        }
        
      }
    }
    
    #GENERATE WHISTLE DETECTION FRAMES
    if (GENERATE_WHISTLES == TRUE) {
      whistleInfo <- whistleData(myStudy, selectEvent=singleFile)
      wdata <- whistleInfo[[1]]
      bwdata <- whistleInfo[[2]]
      
      if (class(wdata) == 'data.frame') {
        if (dim(wdata)[1] > 1) {
          a <- saq$UTC[1]
          hr1 <- hour(a)
          min1 <- minute(a)
          sec1 <- second(a)
          time1 <- 3600*hr1 + 60*min1 + sec1
          
          a <- wdata$UTC[1]
          hr2 <- hour(a)
          min2 <- minute(a)
          sec2 <- second(a)
          time2 <- 3600*hr2 + 60*min2 + sec2
          
          dt <- time2-time1
          wfilt <- filterWhistles(wdata, dt=0)
          
          ind1 <- gregexpr('_', filename)[[1]][1]
          ind2 <- gregexpr('.sqlite', filename)[[1]][1]
          name <- substr(filename, ind1+1, ind2-1)
          if (!is.null(singleFile)) {
            f <- substr(singleFile, 1, nchar(singleFile)-4)
            name <- gsub(' ', '', paste(name, '_', f))
          }
          
          createWhistleSpectra(wdata=wfilt, bwdata=bwdata, temp_res=temp_res, startrect0=0, startrec=startrec, endrec=endrec, site=site,
                               min_combined_bandwidth = 0, min_bandwidth = 0, dd_thr=0.05, include_empty=FALSE, remove_dups=TRUE,
                               dest=sprintf('%s/%s/whistlespectra_%s.csv', savefolder, site, name))
        }
      }
    }
      
  } else {
    dbDisconnect(con)
  }
}

path <- 'I:/temp/test2201'
dir(path)
x1 <- read.csv(sprintf('%s/%s', path, dir(path)[4]))
x2 <- read.csv(sprintf('%s/%s', path, dir(path)[5]))

st <- 37
sub1 <- subset(x1, starttime == st)
sub2 <- subset(x2, starttime == st)
xx1 <- as.numeric(sub1[,ind:dim(sub1)[2]])
xx2 <- as.numeric(sub2[,ind:dim(sub2)[2]])
ind <- which(names(sub1) == 'X1')
d <- data.frame(f = seq(1,80,1),
                x1=xx1/sum(xx1),
                x2=xx2/sum(xx2))

library(ggplot2)
ggplot(data=d, aes(x=f, y=x1)) +
  geom_line(data=d, aes(x=f, y=x1), col='blue') + 
  geom_line(data=d, aes(x=f, y=x2), col='red') + 
  ylim(c(0,0.03))
