library(PAMpal)
library(zoo)
library(lubridate)

###functions###
retrieveDetections <- function(dbname, binaryfolder, sr) {
  myPps <- PAMpalSettings(db = dbname,
                          binaries = binaryfolder,
                          sr_hz = sr,
                          filterfrom_khz = 2,
                          filterto_khz = NULL,
                          winLen_sec = 0.002,
                          verbose=0)
  myStudy <- processPgDetections(myPps)
  return(myStudy)
} 

vocRate <- function(sub, d1, d2, w=30, step=60, verbose=0) {
  t <- c()
  rate <- c()
  start <- d1
  end <- d2
  dd <- 0
  
  count <- 0
  while (difftime(end, start, units='sec') >= w) {
    a <- as.POSIXct(start-w)
    b <- as.POSIXct(start+w)
    dd1 <- format(as.Date(start,format="%Y-%m-%d %h:%m:%s"), format = "%d")
    if (dd1 != dd) {
      print(start)
      dd <- dd1
    }
    subsub <- subset(sub, UTC >= a & UTC < b)
    t <- append(t, start)
    rate <- append(rate, dim(subsub)[1])
    start <- start + step
    count <- count + 1
  }
  
  df <- data.frame(t=t, rate=rate)
  return(df)
}

clickData <- function(myStudy, fft, selectEvent=NULL) {
  cdata <- getClickData(myStudy)
  if (!is.null(selectEvent)) {
    cdata <- subset(cdata, eventId == selectEvent)
  }
  if (length(cdata) != 0) {
    subcdata <- cdata[order(cdata$UTC),]
    subcdata$t <- as.numeric(subcdata$UTC - subcdata$UTC[1])
    subsub <- subset(subcdata, detectorName %in% c('Click_Detector_1'))
    subBg <- subset(subcdata, detectorName %in% c('Click_Detector_0'))
    
    avSpec <- calculateAverageSpectra(myStudy, evNum = 1:1000, wl=fft, noise=FALSE, plot=c(FALSE, FALSE))
    spec <- avSpec$allSpec[,avSpec$UID %in% subsub$UID]
    
    avSpec <- calculateAverageSpectra(myStudy, evNum = 1:1000, wl=32, noise=TRUE, plot=c(FALSE, TRUE))
    noise <- avSpec$allNoise
    noise <- rowMeans(noise)
    uid <- avSpec$UID[avSpec$UID %in% subsub$UID]
    print(sprintf('Found %s clicks in recording file.', dim(subsub)[1]))
    return(list(subsub, spec, subBg, noise))
  } else {
    return(list(0,0))
  }
}


filterClicks <- function(cdata, dt=0, w=1, nmin=3) {
  omitrows <- c()
  ici <- c()
  cdata$t <- cdata$t + dt
  for (i in 1:dim(cdata)[1]) {
    t0 <- cdata$UTC[i]
    if (i > 1) {
      ici <- append(ici, difftime(cdata$UTC[i+1], cdata$UTC[i], units='sec'))
    } else {
      ici <- append(ici, 100)
    }
    
    a <- t0-(w/2)
    b <- t0+(w/2)
    sub <- subset(cdata, UTC >= a & UTC < b)
    if (dim(sub)[1] < nmin) {
      omitrows <- append(omitrows, i)
    }
  }
  
  cdata$ici <- ici
#  cfilt <- cdata[-omitrows,]
  cfilt <- cdata
  return(cfilt)
}

createClickSpectra <- function(subsub, spec, nspec=NULL, dest='temp.csv', file='clickspectra.csv', startrect0=NULL, sr=96000,
                               dur=4, step=1, nmin=3, k=3, maxout=1800, site=NULL, startrec=NULL, endrec=NULL, include_empty=FALSE, remove_dups=FALSE) {
  t0 <- 0
  t1 <- t0 + dur
  data <- data.frame()
  starts <- c()
  nclicks <- c()
  snr <- c()
  noise <- c()
  nlevel <- 0
  
  if (max(subsub$t) > maxout) {
    maxt <- maxout
  } else {
    maxt <- max(subsub$t)
  }
  
  while (t0 < maxt) {
    t1 <- t0 + dur
    inds <- which(subsub$t >= t0 & subsub$t < t1)
    if (length(inds) >= nmin) {
      N <- length(inds)
      spectrum <- rowMeans(spec[,inds], na.rm=TRUE)
      
      if (is.null(nspec)) {
        snr_val <- mean(spectrum)
      } else {
        spectrum <- spectrum - min(spectrum)
        snr_val <- mean(spectrum)
#        spectrum <- spectrum - nspec
      }
      y <- rollmeanr(spectrum, k)
      y0 <- as.integer(10*length(y)/(sr/2000))
      y1 <- as.integer(40*length(y)/(sr/2000))
      y <- y[y0:y1]
      
      speclevel <- mean(y)
      nlevel <- mean(nspec)

      x <- seq(10, 40, 30/length(y))
      x <- x[1:length(x)-1]
      
      ynew <- c()
      ccount <- 1
      for (i in 1:length(y)) {
        if (i%%2 != 0 & i != length(y)) {
          ynew <- append(ynew, mean(c(y[i], y[i+1])))
          ccount <- ccount + 1
        } 
      }
      
      ynew <- as.numeric(ynew)
      ynew <- ynew/sum(ynew)
      print(length(ynew))
      
      if (any(is.nan(y)) == FALSE) {
        data <- rbind(data, ynew)
        starts <- append(starts, startrect0 + t0)
        nclicks <- append(nclicks, N)
        noise <- append(noise, nspec)
        snr <- append(snr, snr_val)
        t0 <- t0 + step
        
      } else {
        t0 <- t0 + step
      }
      
    } else {
      if (include_empty==TRUE) {
        f <- seq(10, 40, 30/79)
        y <- rep(0, length(f))
        data <- rbind(data, y)
        nclicks <- append(nclicks, length(inds))
        noise <- append(noise, 0)
        snr <- append(snr, 0)
        starts <- append(starts, startrect0 + t0)
      }
      t0 <- t0 + step
    }
  }
  
  if (dim(data)[1] > 0) {
    names(data) <- seq(1, dim(data)[2])
    data <- cbind(noiselevel=nlevel, data)
    data <- cbind(SNR=snr, data)
    data <- cbind(n=nclicks, data)
    data <- cbind(dur=dur, data)
    data <- cbind(starttime = starts, data)
    data <- cbind(endrec=endrec, data)
    data <- cbind(startrec=startrec, data)
    data <- cbind(site=site, data)
    
    if (remove_dups == TRUE) {
      newdata <- data.frame()
      for (k in 1:nrow(data)) {
        if (k == 1) {
          newdata <- rbind(newdata, data[k,])
        } else {
          if (data$SNR[k] != data$SNR[k-1] & data$n[k] != data$n[k-1]) {
            newdata <- rbind(newdata, data[k,])
          }
        }
      }
      data <- newdata
    }
    
    write.table(data, dest, sep=',', quote=TRUE, row.names=FALSE, col.names = TRUE)
    print(sprintf('Done. Saved %s click %ss-detection frames to file.', dim(data)[1], dur))
  }
    
#    temp <- read.csv(dest)
#    if (dim(temp)[1] > 1) {
#      omitrows <- c()
#      for (i in 2:dim(temp)[1]) {
#        a <- temp$SNR[i-1]
#        b <- temp$SNR[i]
#        if (!(any(c(is.na(a), is.na(b))))) {
#          if (a==b) {
#            omitrows <- append(omitrows, i)
#          }
#        }
#      }
#      if (length(omitrows) > 0) {
#        temp <- temp[-omitrows,]
#      }
#    }
#  
#    temp$starttime <- as.POSIXct(temp$starttime)
#    
#    if (file %in% dir('I:/newSpectra')) {
#      path <- sprintf('I:/newSpectra/%s', file)
#      write.table(temp, path, append=TRUE, quote=TRUE, sep=',', row.names=FALSE, col.names=FALSE)
#    } else {
#      path <- sprintf('I:/newSpectra/%s', file)
#      write.table(temp, path, append=FALSE, quote=TRUE, sep=',', row.names=FALSE, col.names=TRUE)
#    }
#    print(sprintf('Done. Saved %s click %ss-detection frames to file.', dim(data)[1], dur))
#  } else {
#    print('Done. No detection frames created.')
#  }
  
}

whistleData <- function(myStudy, dt=0, selectEvent=NULL) {
  wdata <- getWhistleData(myStudy)
  if (!is.null(selectEvent)) {
    wdata <- subset(wdata, eventId==selectEvent)
  }
  if (length(wdata) != 0) {
    wdata <- wdata[order(wdata$UTC),]
    wdata$t <- as.numeric(wdata$UTC - wdata$UTC[1])
    bwdata <- getBinaryData(myStudy, UID=wdata$UID, type='whistle')
    print(sprintf('Found %s whistles in recording file.', dim(wdata)[1]))
    
    return(list(wdata, bwdata))
  } else {
    return(list(0,0))
  }
}

filterWhistles <- function(wdata, mindur=0.2, dt=0) {
  omitUID <- c()
  wdata <- subset(wdata, duration >= 0.2)
  wdata$t <- wdata$t + dt
  for (file in unique(wdata$BinaryFile)) {
    sub <- subset(wdata, BinaryFile == file)
    ind <- sapply(gregexpr("_", file), tail, 1)
    yr <- substr(file, ind-8, ind-5)
    mo <- substr(file, ind-4, ind-3)
    day <- substr(file, ind-2, ind-1)
    hr <- substr(file, ind+1, ind+2)
    min <- substr(file, ind+3, ind+4)
    sec <- substr(file, ind+5, ind+6)
    dt <- sprintf('%s-%s-%s %s:%s:%s', yr, mo, day, hr, min, sec)
    t0 <- as.POSIXct(dt, tz='UTC')
    t1 <- t0+10
    
    subsub <- subset(sub, UTC < t1)
    if (dim(subsub)[1] > 0) {
      omitUID <- append(omitUID, subsub$UID)
    }
  }
  
  wfilt <- subset(wdata, !(UID %in% omitUID))
  return(wfilt)
}


createWhistleSpectra <- function(wdata, bwdata, temp_res, dest='temp.csv', file='whistlespectra.csv', startrect0=NULL, 
                                 f0=2000, f1=20000, 
                                 w=4, step=1, dd_thr=0.05, min_combined_bandwidth=0, min_bandwidth=0, include_empty=FALSE,
                                 maxout=1800, site=NULL, startrec=NULL, endrec=NULL, remove_dups=FALSE) {
  t0 <- 0
  data <- data.frame()
  starts <- c()
  dd <- c()
  fstep = as.integer((f1-f0)/180)
  
  if (max(wdata$t) > maxout) {
    maxt <- maxout
  } else {
    maxt <- max(wdata$t)
  }
  
  while (t0 < maxt) {
    t1 <- t0 + w
    inds <- as.numeric(rownames(subset(wdata, t >= t0 & t < t1)))
    binary_inds <- c()
    for (ind in inds) {
      binary_inds <- append(binary_inds, which(as.numeric(ind) == as.numeric(rownames(wdata))))
    }
    if (length(binary_inds) > 0) {
      ctr <- c()
      bandwidths <- c()
      dt <- c()
      for (i in binary_inds) {
        ctr <- append(ctr, bwdata[[i]]$contour*temp_res)
        bandwidths <- append(bandwidths, max(ctr)-min(ctr))
        ns <- bwdata[[i]]$nSlices
        L <- bwdata[[i]]$sampleDuration/sr
        dt <- append(dt, L/ns)
      }
      
      dt <- mean(dt)
      y <- c()
      
      f <- seq(f0, f1-fstep, fstep)
      den <- length(ctr)/(w/dt)
      
      combined_bandwidth <- max(ctr) - min(ctr)
      if (all(ctr) < 50) {
        ctr <- ctr*1000
      }
      
      if (den >= dd_thr & combined_bandwidth >= min_combined_bandwidth & all(bandwidths >= min_bandwidth)) {
        for (i in f) {
          y <- append(y, sum(ctr >= i & ctr < i+fstep)/length(ctr))
        }
        data <- rbind(data, y)
        dd <- append(dd, den)
        starts <- append(starts, startrect0 + t0)
        t0 <- t0 + step
        
      } else {
        t0 <- t0 + step
      }
    } else {
      if (include_empty==TRUE) {
        f <- seq(f0, f1-fstep, fstep)
        y <- rep(0, length(f))
        data <- rbind(data, y)
        dd <- append(dd, 0)
        starts <- append(starts, startrect0 + t0)
      }
      t0 <- t0 + step
    } 
  }
  
  if (dim(data)[1] > 0) {
    print(dim(data))
    names(data) <- seq(1, dim(data)[2])
    data <- cbind(dd=dd, data)
    data <- cbind(dur=w, data)
    data <- cbind(starttime = starts, data)
    data <- cbind(endrec=endrec, data)
    data <- cbind(startrec=startrec, data)
    data <- cbind(site=site, data)
    
    if (remove_dups == TRUE) {
      newdata <- data.frame()
      for (k in 1:nrow(data)) {
        if (k == 1) {
          newdata <- rbind(newdata, data[k,])
        } else {
          if (data$dd[k] != data$dd[k-1]) {
            newdata <- rbind(newdata, data[k,])
          }
        }
      }
      data <- newdata
    }
    
    write.table(data, dest, sep=',', quote=TRUE, row.names=FALSE, col.names = TRUE)
    print(sprintf('Done. Saved %s whistle %ss-detection frames to file.', dim(data)[1], w))
  }
}
    
#    temp <- read.csv(dest)
#    if (dim(temp)[1] > 1) {
#      omitrows <- c()
#      for (i in 2:dim(temp)[1]) {
#        a <- temp$dd[i-1]
#        b <- temp$dd[i]
#        if (!(any(c(is.na(a), is.na(b))))) {
#          if (a==b) {
#            omitrows <- append(omitrows, i)
#          }
#        }
#      }
#      if (length(omitrows) > 0) {
#        temp <- temp[-omitrows,]
#      }
#    }
#    
#    temp$starttime <- as.POSIXct(temp$starttime)

#   if (file %in% dir('I:/newSpectra')) {
#      path <- sprintf('I:/newSpectra/%s', file)
#      write.table(temp, path, append=TRUE, quote=TRUE, sep=',', row.names=FALSE, col.names=FALSE)
#    } else {
#      path <- sprintf('I:/newSpectra/%s', file)
#      write.table(temp, path, append=FALSE, quote=TRUE, sep=',', row.names=FALSE, col.names=TRUE)
#    }
#    print(sprintf('Done. Saved %s whistle %ss-detection frames to file.', dim(data)[1], w))
#  } else {
#    print('Done. No detection frames created.')
#  }


downsample <- function(x, k=3) {
  if (k > 1) {
    new <- c()
    for (i in 1:length(x)) {
      if (i%%k == 0) {
        new <- append(new, mean(x[(i-k+1):i]))
      }
    } 
    new <- new/sum(new)
  } else {
    new <- x
  }
  return(new)
}

###analysis function###

analyseDetections <- function(site, monthstart, daystart, monthend, dayend, thr_c=15, thr_w=5, plot_events=NULL) {
  fr <- list()
  fr['96000'] <- 96000/2048
  sr <- 96000
  
  item <- sprintf('%s_%s%s%s%s', site, daystart, monthstart, dayend, monthend)
  dbname <- sprintf('I:/MarineScotland/detections/%s.sqlite3', item)
  binaryfolder <- sprintf('I:/MarineScotland/detections/binary/%s', item)
  split_path <- function(x) if (dirname(x)==x) x else c(basename(x),split_path(dirname(x)))
  temp_res <- as.numeric(fr[as.character(sr)][1])
  d1 <- as.POSIXct(sprintf("2019-%s-%s 00:00:00", monthstart, daystart), tz="UTC")
  d2 <- as.POSIXct(sprintf("2019-%s-%s 23:59:59", monthend, dayend), tz="UTC")
  
  print(sprintf('Loading PAMGuard detections from %s (%s.%s-%s.%s)...', site, daystart, monthstart, dayend, monthend))
  myStudy <- retrieveDetections(dbname, binaryfolder, sr)
  clickInfo <- clickData(myStudy, fft=as.integer(.00533333*sr)+1)
  whistleInfo <- whistleData(myStudy)
  
  print('')
  print('Compiling click detections...')
  cdata <- clickInfo[[1]]
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

  cfilt <- filterClicks(cdata)
  sub <- subset(cfilt, UTC >= d1 & UTC < d2)
  dfc <- vocRate(sub, d1, d2, verbose = 1)
  print('')
  
  print('Compiling whistle detections')
  wdata <- whistleInfo[[1]]
  bwdata <- whistleInfo[[2]]
  wfilt <- filterWhistles(wdata)
  sub <- subset(wfilt, UTC >= d1 & UTC < d2)
  dfw <- vocRate(sub, d1, d2, verbose=1)
  print('')
  
  print('Calculating detection rates per minute...')
  all(dfc$t == dfw$t)
  df1 <- data.frame(t=dfc$t, rate_w=dfw$rate, rate_c=dfc$rate)
  
  d1 <- as.POSIXct(sprintf("2019-%s-%s 00:00:00", monthstart, daystart), tz="UTC")
  d2 <- as.POSIXct(sprintf("2019-%s-%s 23:59:59", monthend, dayend), tz="UTC")
  start <- d1
  startTime <- c()
  endTime <- c()
  clicks <- c()
  whistles <- c()
  w = 60
  dd = 0
  while (start-w < d2) {
    t0 <- start
    t1 <- start + w
    dd1 <- format(as.Date(start,format="%Y-%m-%d %h:%m:%s"), format = "%d")
    if (dd1 != dd) {
      print(start)
      dd <- dd1
    }
    sub <- subset(df1, t >= t0 & t < t1)
    startTime <- append(startTime, t0-30)
    endTime <- append(endTime, t1-30)
    ccount <- sum(sub$rate_c)
    wcount <- sum(sub$rate_w)
    clicks <- append(clicks, ccount)
    whistles <- append(whistles, wcount)
    start <- start + w
  }
  
  
  df2 <- data.frame(startTime=startTime, endTime=endTime, clicks=clicks, whistles=whistles)
  
  cpos <- as.integer(df2$clicks >= thr_c)
  wpos <- as.integer(df2$whistles >= thr_w)
  both <- cpos*wpos
  
  k = 1
  a <- subset(df2 , clicks >= thr_c | whistles >= thr_w)
  plot(rollmean(df1$t, k=k), rollmean(df1$rate_c, k=k), type='l', col='red', ylim=c(0,100))
  lines(rollmean(df1$t, k=k), rollmean(df1$rate_w, k=k), col='blue')
  
  lines(x=df2$startTime, y=replicate(dim(df2)[1], thr_c), col='darkred', lty='dashed')
  lines(x=df2$startTime, y=replicate(dim(df2)[1], thr_w), col='navy', lty='dashed')
  
  if (!(is.null(plot_events))) {
    segments(x0=df2$startTime, y0=cpos*100-10, x1=df2$endTime, y1 = cpos*100-10, lwd=7, col='red')
    segments(x0=df2$startTime, y0=wpos*100-15, x1=df2$endTime, y1 = wpos*100-15, lwd=7, col='blue')
    segments(x0=df2$startTime, y0=both*100-20, x1=df2$endTime, y1 = both*100-20, lwd=7, col='lightgreen')
  }

  
  if (dim(a)[1] > 0) {
    events <- cbind(site=site, a)
  } else {
    events <- NULL
  }
  
  events_all <- cbind(site=site, df2)
  
  return(list(myStudy, events, events_all, cfilt, noise, wfilt, bwdata))
}

##group spectra
groupSpectra <- function(events, dfc, dfw, selectsite, n, voctype) {
  
  subc <- subset(dfc, site == selectsite)
#  subc$starttime <- as.POSIXct(subc$starttime)
  subw <- subset(dfw, site == selectsite & dd >= 0.1)
#  subw$starttime <- as.POSIXct(subw$starttime)
  
  a <- events$startTime[n]
  b <- events$endTime[n]
  if (voctype == 'w') {
    subsub <- subset(subw, starttime >= a & starttime < b)
    xmin <- 2
    xmax <- 20
    ymax <- 1
  } else{
    subsub <- subset(subc, starttime >= a & starttime < b)
    xmin <- 10
    xmax <- 40
    ymax <- 0.025
  }
  
  ind <- which(names(subsub) == 'X1')
  spec <- subsub[,ind:dim(subsub)[2]]
  k = 5
  medspec <- downsample(as.numeric(sapply(spec, median)), k=k)
  ss_thr <- 1000
  desc <- data.frame()
  if (k > 1) {
    spec_ds <- data.frame()
  } else {
    spec_ds <- spec
  }
  
  for (i in 1:dim(spec)[1]) {
    if (i == 1) {
      s <- as.numeric(spec[i,])
      s <- downsample(s, k=k)
      f <- seq(xmin, xmax, (xmax-xmin)/length(s))
      f <- f[1:length(f)-1]
      ss <- sum(100*((s-medspec)**2))
      medf <- f[which(cumsum(s) >= 0.5)[1]]
      z <- data.frame(id=i, ss=ss, medf=medf)
      desc <- rbind(z, desc)
      if (k > 1) {
        spec_ds <- rbind(s, spec_ds)
      }
    } else {
      s <- as.numeric(spec[i,])
      s <- downsample(s, k=k)
      ss <- sum(100*((s-medspec)**2))
      medf <- f[which(cumsum(s) >= 0.5)[1]]
      z <- data.frame(id=i, ss=ss, medf=medf)
      desc <- rbind(z, desc)
      if (k > 1) {
        spec_ds <- rbind(s, spec_ds)
      }
    }
  }
  
  if (dim(spec_ds)[1] > 1) {
    dist_mat <- dist(spec_ds, method = 'euclidean')
    hclust_avg <- hclust(dist_mat, method = 'average')
    plot(hclust_avg, main='Detection frames')
    labels <- cutree(hclust_avg, k = 3)
    subsub$label <- labels
  } else {
    subsub$label <- c(1)
  }

  
  
  return(list(subsub, spec_ds, labels))
}


