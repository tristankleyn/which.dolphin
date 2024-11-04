library(PAMpal)
library(zoo)
library(lubridate)

fr <- list()
fr['96000'] <- 96000/2048
sr <- 96000

#w_spec <- read.table('I:/MarineScotland/whistlespectra.csv', sep=',', header=TRUE)
#c_spec <- read.table('I:/MarineScotland/clickspectraNS.csv', sep=',', header=TRUE)
#unique(w_spec$site)
#unique(c_spec$site)
dir('I:/MarineScotland/detections')


site <- 'crudenbay'
monthstart <- '07'
monthend <- '07'
daystart <- '01'
dayend <- '26'

#compile detections
results <- analyseDetections(site=site,
                             monthstart=monthstart,
                             monthend=monthend,
                             daystart=daystart,
                             dayend=dayend,
                             thr_c=1,
                             thr_w=1,
                             plot_events=NULL)


#returns from analyseDetections function
myStudy <- results[[1]]             ####PAMpal acoustic study object
events <- results[[2]]              ####detection positive events
events_all <- results[[3]]          ####all minute detection rates
cfilt <- results[[4]]               ####individual click detections
noise <- results[[5]]               ####background noise spectra
wfilt <- results[[6]]               ####individual whistle detections 
bwdata <- results[[7]]              ####whistle binary data

thr_c <- 1
thr_w <- 1

#show events found and save results
print(events)
dim(events)
startrow <- 1
endrow <- dim(events)[1]
rec0 <- 0

#save detection frames
for (i in startrow:endrow) {
  t0 <- events$startTime[i]
  t1 <- events$endTime[i]
  if (events$clicks[i] >= thr_c) {
    subc <- subset(cfilt, UTC >= t0 & UTC < t1)
    subc$t <- as.numeric(subc$UTC - subc$UTC[1])
    startrec <- subc$eventId[1]
    endrec <- subc$eventId[dim(subc)[1]]
#    subn <- subset(noise, eventId == startrec)

    avSpec <- calculateAverageSpectra(myStudy, ev=startrec, wl=as.integer(.00533333*sr)+1, noise=FALSE, plot=c(FALSE, FALSE))
    spec <- avSpec$allSpec[,avSpec$UID %in% subc$UID]
    nspec <- noise
    
    if (startrec != rec0 & sum(nspec) != 0) {
      if (is.null(dim(nspec))) {
        x <- nspec
        L <- 1
      } else {
        x <- rowMeans(nspec)
        L <- dim(nspec)[2]
      }
      y0 <- as.integer(10*length(x)/(96000/2000))
      y1 <- as.integer(40*length(x)/(96000/2000))
      n <- rollmeanr(x, k=3)
      n <- n[y0:y1]
      n <- n - min(n)
      n <- n/sum(n)
      ndata <- data.frame()
      ndata <- rbind(ndata, n)
      names(ndata) <- seq(1, dim(ndata)[2])
      ndata <- cbind(n=L, ndata)
      ndata <- cbind(rec_id=startrec, ndata)
      ndata <- cbind(site=site, ndata)
      if ('background.csv' %in% dir('I:/MarineScotland')) {
        write.table(ndata, 'I:/MarineScotland/background.csv', sep=',', quote=TRUE, append=TRUE, row.names=FALSE, col.names=FALSE)
      } else {
        write.table(ndata, 'I:/MarineScotland/background.csv', sep=',', quote=TRUE, append=FALSE, row.names=FALSE, col.names=TRUE)
      }
      rec0 <- startrec
      print('Saved noise background to background.csv')
    }    
    
    createClickSpectra(subsub=subc, spec=spec, nspec=nspec, site=site, 
                       startrect0=t0, startrec=startrec, endrec=endrec, 
                       dest='tempc1.csv', file=sprintf('clickspectraNS_%s.csv', site))
    
  }
  if (events$whistles[i] >= thr_w) {
    sub <- subset(wfilt, UTC >= t0 & UTC < t1)
    sub$t <- as.numeric(sub$UTC - sub$UTC[1])
    startrec <- sub$eventId[1]
    endrec <- sub$eventId[dim(sub)[1]]
    
    createWhistleSpectra(wdata=sub, bwdata=bwdata, freq_res=as.numeric(fr[as.character(sr)][1]), 
                         min_combined_bandwidth = 500, min_bandwidth = 500,
                         site=site, startrec=startrec, startrect0=t0, endrec=endrec,
                         dest='tempw1.csv', file=sprintf('whistlespectraMF_%s.csv', site))
  }
}

#save events logs
if (length(events) > 0) {
  if (sprintf('events_%s.csv', site) %in% dir('I:/MarineScotland')) {
    write.table(events, file=sprintf('I:/MarineScotland/events_%s.csv', site), 
                sep =',', append=TRUE, row.names=FALSE, col.names=FALSE)
    write.table(events_all, file=sprintf('I:/MarineScotland/events_all_%s.csv', site), 
                sep =',', append=TRUE, row.names=FALSE, col.names=FALSE)
  } else {
    write.table(events, file=sprintf('I:/MarineScotland/events_%s.csv', site), 
                sep =',', append=FALSE, row.names=FALSE, col.names=TRUE)
    write.table(events_all, file=sprintf('I:/MarineScotland/events_all_%s.csv', site), 
                sep =',', append=FALSE, row.names=FALSE, col.names=TRUE)
  }
} else {
  if (sprintf('events_%s.csv', site) %in% dir('I:/MarineScotland')) {
    write.table(events_all, file=sprintf('I:/MarineScotland/events_all_%s.csv', site), 
                sep =',', append=TRUE, row.names=FALSE, col.names=FALSE)
  } else {
    write.table(events_all, file=sprintf('I:/MarineScotland/events_all_%s.csv', site), 
                sep =',', append=FALSE, row.names=FALSE, col.names=TRUE)
  }
}


