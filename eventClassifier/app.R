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

model <- readRDS('EventClassifier_7sp.rds')
options(shiny.maxRequestSize=1000*1024^2)

####SHINY####
ui <- fluidPage(
  
  img(src='header_01.png', style = "height: auto; width: 30%;"),
  
  titlePanel(title=''),
  
  sidebarPanel(
    selectInput("classifierType", "Select PAMGuard classifier", c("ROCCA", "Deep Learning Classifier"), "ROCCA"),
    fileInput("file1", "Select database", accept='.sqlite3'),
    sliderInput("evScore", "Minimum event score", 0, 0.5, 0),
    sliderInput("minClicks", "Minimum clicks", 0, 100, 0),
    sliderInput("minWhistles", "Minimum whistles", 0, 100, 0),
    dateRangeInput("dateRange", "Filter dates", start = Sys.Date() - 3652, end = Sys.Date()),
    downloadButton("downloadFiltered", "Download filtered events"),
    downloadButton("downloadAllData", "Download all events")
  ),
  mainPanel(
    fluidRow(
      column(12,   # Full width column for both plot and table
             plotOutput(outputId = "plt", width = "100%"),  # Set width to 100% of column
             tableOutput(outputId = 'table1'),  # Table also spans 100% of column
      )
    )
  ),
  
  tags$style(type="text/css",
             ".shiny-output-error { visibility: hidden; }",
             ".shiny-output-error:before { visibility: hidden; }"
  )
)

server <- function(input, output, session) {

  sql_data <- reactive({
    req(input$file1)
    
    print(input$file1)
    db_con <- dbConnect(RSQLite::SQLite(), input$file1$datapath)
    on.exit(dbDisconnect(db_con))
    
    data <- dbGetQuery(db_con, paste0("SELECT * FROM Rocca_Whistle_Stats"))
    SAQ <- dbGetQuery(db_con, paste0("SELECT * FROM Sound_Acquisition"))
    data$UTC <- as.POSIXct(data$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
    SAQ$UTC <- as.POSIXct(SAQ$UTC, '%Y-%m-%d %H:%M:%OS', tz='UTC')
    specieslist <- c('Dde', 'Ggr', 'Gme', 'Lal', 'Oor', 'Ttr')
    
    # Filter data based on date range
    if (!is.null(input$dateRange)) {
      data <- data[data$UTC >= input$dateRange[1] & data$UTC <= input$dateRange[2] + 1, ]
      SAQ <- SAQ[SAQ$UTC >= input$dateRange[1] & SAQ$UTC <= input$dateRange[2] + 1, ]
    }
    
    event_count <- 1
    test_events <- data.frame()
    for (evID in unique(SAQ$SystemName)) {
      subSAQ <- subset(SAQ, SystemName == evID)
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

          evDur <- as.numeric(seconds(difftime(sub$UTC[dim(sub)[1]], sub$UTC[1])))
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
    
    
    probs <- data.frame(predict(model, test_events, type='prob'))
    predspecieslist <- c('Common', 'Rissos', 'LF pilot whale', 'Atlantic white-sided', 
                         'White-beaked', 'Orca', 'Bottlenose')
    
    
    pred_df <- data.frame()
    for (i in 1:dim(probs)[1]) {
      row <- as.numeric(probs[i, ])
      pred <- predspecieslist[which.max(row)]
      conf <- max(row)
      prom <- row[rev(order(row))][1] - row[rev(order(row))][2]
      score <- prom*conf
      predrow <- data.frame(eventID=test_events$eventID[i], clicks=as.integer(test_events$clicks[i]), whistles=as.integer(test_events$whistles[i]),
                            duration=test_events$duration[i], predictedSpecies=pred, score=score, prom=prom, conf=conf)
      
      spcount <- 1
      for (sp in predspecieslist) {
        predrow[[sp]] <- row[spcount]
        spcount <- spcount + 1
      }
      pred_df <- rbind(pred_df, predrow)
    }
    
    allpreds <- pred_df
    pred_df <- subset(pred_df, clicks >= input$minClicks)
    pred_df <- subset(pred_df, whistles >= input$minWhistles)
    pred_df <- subset(pred_df, score >= input$evScore)
    evScore <- input$evScore
    list(df=test_events, preds=pred_df, allpreds=allpreds, evScore=evScore)
    
  })
  
  output$plt <- renderPlot({
    req(input$file1)
    
    plot_data <- sql_data()
    df <- plot_data$preds
    x <- seq(0,1,0.01)
    y <- as.numeric(plot_data$evScore)/x
    thr <- data.frame(x=x, y=y)
    
    colors <- c('Common'='royalblue',
                'Rissos'='darkred',
                'LF pilot whale'='forestgreen',
                'Atlantic white-sided'='purple',
                'White-beaked'='grey',
                'Orca'='orange',
                'Bottlenose'='turquoise')
    
    ggplot() + 
      geom_point(data=df, aes(x=conf, y=prom, fill=predictedSpecies, col=predictedSpecies), cex=3, alpha=0.75) +
      scale_fill_manual(values=colors) + 
      scale_color_manual(values=colors) + 
      geom_line(data=thr, aes(x=x, y=y), col='orange', lwd=1, alpha=0.75) + 
      xlim(c(0,1)) + 
      ylim(c(0,1)) + 
      theme_minimal()
  })
  
  output$table1 <- renderTable({
    show_table <- sql_data()$preds
    show_table <- show_table[, !(names(show_table) %in% c('prom', 'conf'))]
    show_table
  })
  
  output$downloadFiltered <- downloadHandler(
    filename = function() {
      paste("ClassifiedEvents", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(sql_data()$preds, file, row.names = FALSE)
    }
  )
  
  output$downloadAllData <- downloadHandler(
    filename = function() {
      paste("AllEvents", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(sql_data()$allpreds, file, row.names = FALSE)
    }
  )
  
}

shinyApp(ui, server)

