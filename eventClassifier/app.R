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
library(DT)
library(shinyjs)

model <- readRDS('EventClassifier_7sp.rds')
options(shiny.maxRequestSize=1000*1024^2)
source('appFunctions.R')


####SHINY####
ui <- fluidPage(
  useShinyjs(), 
  
  tags$head(
    tags$style(HTML("
      .shiny-title {
        text-align: left; 
        color: #a6b8a6;   /* Change font color to a red shade */
        font-size: 24px;  /* Change font size to 36px */
        font-weight: lighter;
        margin-left: 20px; 
      }
      
      .loading-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background-color: #a6b8a6;
          animation: pulse 1.5s infinite ease-in-out;
          position: absolute;
          left: 50%;
          top: 50%;
          transform: translateX(-50%);
      }
      
      @keyframes pulse {
        0% { transform: translateX(-50%) scale(1); }
        50% { transform: translateX(-50%) scale(1.5); }
        100% { transform: translateX(-50%) scale(1); }
      }
    "))
  ),
  
  titlePanel(title = div(class = "shiny-title", "Classify acoustic events")),
  
  sidebarPanel(
    selectInput("classifierType", "Select supported PAMGuard classifier", c("ROCCA Classifier", "delphinID Classifier"), "ROCCA"),
    fileInput("file1", "Select database", accept='.sqlite3'),
    sliderInput("evScore", "Minimum event score", 0, 0.5, 0),
    sliderInput("minClicks", "Minimum clicks", 0, 100, 0),
    sliderInput("minWhistles", "Minimum whistles", 0, 100, 0),
    dateRangeInput("dateRange", "Filter dates", start = Sys.Date() - 3652, end = Sys.Date()),
    downloadButton("downloadFiltered", "Download filtered events"),
    downloadButton("downloadAllData", "Download all events")
  ),
  

  mainPanel(
    div(id = "loading", class = "loading-dot", style = 'display:none;'),
    
    fluidRow(
      column(12,   # Full width column for both plot and table
             plotOutput(outputId = "plt", width = "100%", style = "margin-bottom: 5px;"),  # Set width to 100% of column
      )
    ),
    fluidRow(
      column(12,
             DTOutput("table1", style = "margin-top: 5px;"))
    )
  ),
  
  tags$style(type="text/css",
             ".shiny-output-error { visibility: hidden; }",
             ".shiny-output-error:before { visibility: hidden; }"
  )
)

server <- function(input, output, session) {

  process_data <- reactive({
    req(input$file1)
    print(input$file1)
    shinyjs::show("loading")
    
    db_con <- dbConnect(RSQLite::SQLite(), input$file1$datapath)
    on.exit(dbDisconnect(db_con))
    dateRange <- input$dateRange
    
    testevents <- processdataRocca(db_con, dateRange)[[1]]
    list(df=testevents)
  })
  
  
  predictions <- reactive ({
    test_events <- process_data()$df
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
    list(preds=pred_df, allpreds=allpreds, evScore=evScore)
    
  })
  
  output$plt <- renderPlot({
    req(input$file1)
    
    plot_data <- predictions()
    shinyjs::hide("loading")
    df <- plot_data$preds
    x <- seq(0,1,0.01)
    y <- as.numeric(plot_data$evScore)/x
    thr <- data.frame(x=x, y=y)
    
    custom_colors <- c('Common'='royalblue',
                'Rissos'='darkred',
                'LF pilot whale'='forestgreen',
                'Atlantic white-sided'='purple',
                'White-beaked'='grey',
                'Orca'='orange',
                'Bottlenose'='turquoise')
    
    all_levels <- c("Common", "Rissos", "LF pilot whale", "Atlantic white-sided", "White-beaked", "Orca", "Bottlenose")
    df$predictedSpecies <- factor(df$predictedSpecies, levels = all_levels)
    df_full <- data.frame(predictedSpecies = all_levels) %>%
      left_join(df %>% count(predictedSpecies), by = "predictedSpecies") %>%
      mutate(n = ifelse(is.na(n), 0, n))
    
    ggplot(data = df_full, aes(x = predictedSpecies, y=n, fill = predictedSpecies)) +
      geom_bar(stat='identity', fill='#a6b8a6', color = "#a6b8a6") +
      theme_minimal() +
      scale_y_continuous(
        breaks = seq(1, max(df_full$n), by = as.integer(max(df_full$n)/10))) + 
      theme(
        axis.text.x = element_text(size = 12),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),  
        axis.line.x = element_blank(), 
        legend.position = "none")
  }, width = 900, height = 300, res = 96)
  
  output$table1 <- renderDT({
    show_table <- predictions()$preds
    shinyjs::hide("loading")
    ind <- which(names(show_table) == 'score')
    show_table <- show_table[,1:ind]
    show_table <- show_table[, !(names(show_table) %in% c('prom', 'conf'))]
    show_table$duration <- as.integer(show_table$duration)
    
    datatable(show_table, options=list(pageLength=10, dom='tip', paging=TRUE)) %>% formatRound(columns = c("score"), digits = 3)
  })
  
  output$downloadFiltered <- downloadHandler(
    filename = function() {
      paste("ClassifiedEvents", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(predictions()$preds, file, row.names = FALSE)
    }
  )
  
  output$downloadAllData <- downloadHandler(
    filename = function() {
      paste("AllEvents", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(predictions()$allpreds, file, row.names = FALSE)
    }
  )
  
}

shinyApp(ui, server)

