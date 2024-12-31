##IMPORT NECESSARY MODULES
library(PAMpal)
library(zoo)
library(lubridate)
library(RSQLite)

savefolder <- '...'        #INPUT FOLDER PATH FOR SAVING RESULTS TO
binaryfolder <- '...'      #INPUT FOLDER PATH WHERE PAMGUARD BINARY DATA IS STORED
dbfolder <- '...'          #INPUT FOLDER PATH WHERE PAMGUARD DATABASES ARE STORED

fileN1 <- 1
fileN2 <- len(dir(dbfolder))
for (fileN in fileN1:fileN2) {
  filename <- dir(dbfolder)[fileN]
  print(sprintf('Loading file %s/%s: %s', fileN, fileN2, filename))
  
  dbname = sprintf('%s/%s', dbfolder, filename)
  con <- dbConnect(drv=RSQLite::SQLite(), dbname=dbname)
  tables <- dbListTables(con)
  if (length(tables) > 0) {
    saq <- dbGetQuery(conn=con, statement="SELECT * FROM Sound_Acquisition")
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
    clickInfo <- clickData(myStudy, fft=as.integer(.00533333*sr)+1)
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
        cfilt <- filterClicks(cdata, dt=dt)
        
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
          
          createClickSpectra(subsub=cfilt, spec=spec, nspec=nspec, site='none', nmin=3, sr=sr,
                             startrect0=0, startrec=startrec, endrec=endrec, include_empty = TRUE,
                             dest=sprintf('%s/clickspectra_%s.csv', savefolder, name), file=NULL)
    }
    
  }
  
    }
    
    #GENERATE WHISTLE DETECTION FRAMES
    whistleInfo <- whistleData(myStudy)
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
        wfilt <- filterWhistles(wdata, dt=dt)
        
        createWhistleSpectra(wdata=wfilt, bwdata=bwdata, temp_res=temp_res, startrect0 = 0, startrec = 'test', endrec = 'test', site='test',
                             min_combined_bandwidth = 0, min_bandwidth = 0, dd_thr=0.0, include_empty=TRUE,
                             dest=sprintf('%s/whistlespectra_%s.csv', savefolder, name))
      }
    }
      
  } else {
    dbDisconnect(con)
  }
}









