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
library(shinyBS)

rseed <- 42
model <- readRDS('EventClassifier_ROCCA.rds')
model_D <- readRDS('EventClassifier_delphinID.rds')
options(shiny.maxRequestSize=1000*1024^2)
source('appFunctions.R')


####SHINY####
ui <- fluidPage(
  useShinyjs(), 
  
  tags$head(
    tags$style(HTML("
      .shiny-title {
        text-align: left; 
        color: #97a695;   /* Change font color to a red shade */
        font-size: 24px;  /* Change font size to 36px */
        font-weight: lighter;
        margin-left: 20px; 
      }
      
      .loading-container {
      position: absolute; /* Position it absolutely within the parent container */
      top: 50%; /* Center vertically */
      left: 50%; /* Center horizontally */
      transform: translate(-50%, -50%); /* Offset by 50% of its size to fully center */
      text-align: center;
      width: 100%; /* Ensure it takes up full width of the parent container */
      display: flex;
      flex-direction: column;  /* Stack text and dot vertically */
      justify-content: center;  /* Center content vertically */
      align-items: center;  /* Center content horizontally */
      }
      
      .loading-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background-color: #97a695;
      animation: pulse 1.5s infinite ease-in-out;
      margin-top: 5px;  /* Adjust spacing between text and dot */
      }
    
    .loading-text {
      position: relative;
      top: -45px;  /* Adjust to move the text above the dot */
      font-size: 16px;
      color: #97a695;
      font-weight: lighter;
      white-space: nowrap;  /* Prevents text from wrapping */
      overflow: hidden;     /* Hides anything overflowing */
      text-overflow: ellipsis; /* Add ellipsis for overflow text */
      max-width: 200px;  /* Set a maximum width if needed */
      }

    @keyframes pulse {
      0% { transform: translateX(-50%) scale(1); }
      50% { transform: translateX(-50%) scale(1.5); }
      100% { transform: translateX(-50%) scale(1); }
    }
    "))
  ),
  
  titlePanel(title = div(class = "shiny-title", "eventClassifier/")),
  
  sidebarPanel(
    selectInput("classifierType", "Select supported PAMGuard classifier", c("ROCCA Classifier", "delphinID Classifier"), "delphinID Classifier"),
    fileInput("file1", "Select database", accept='.sqlite3'),
    uiOutput("ctableSelectUI"),
    uiOutput("wtableSelectUI"), 
    sliderInput("evScore", "Minimum decision score", 0, 0.2, 0),
    sliderInput("minClicks", "Minimum click predictions", 0, 100, 0),
    sliderInput("minWhistles", "Minimum whistle predictions", 0, 100, 0),
    dateRangeInput("dateRange", "Filter dates", start = Sys.Date() - 3652, end = Sys.Date()),
    actionButton("classifyButton", "Classify", class = "btn-primary"),
    br(),
    br(),
    downloadButton("downloadFiltered", "Download filtered events"),
    downloadButton("downloadAllData", "Download all events")
  ),
  

  mainPanel(
    div(id = "loading", class = "loading-container", style = 'display:none;',
        div(class = "loading-dot"),
        div(class = "loading-text", "Classifying events...")),
    
    div(
      style = "margin:0%",
      fluidRow(
        column(12,   # Full width column for both plot and table
               plotOutput(outputId = "plt", width = "100%", height = "300px"))),
    ),
    
    div(
      style = "margin-top:5%;",
      fluidRow(
      column(12,
             DTOutput("table1", height = "200px")))
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
    
    set.seed(rseed)
    db_con <- dbConnect(RSQLite::SQLite(), input$file1$datapath)
    on.exit(dbDisconnect(db_con))
    dateRange <- input$dateRange
    if (input$classifierType == 'ROCCA Classifier') {
      testevents <- processdataRocca(db_con, dateRange)[[1]]
    } else if (input$classifierType == 'delphinID Classifier') {
      req(input$cTable)
      testevents <- processdataDelphinID(db_con, dateRange, ctable=input$cTable, wtable=input$wTable)[[1]]
    }
    
    list(df=testevents)
  })
  
  
  predictions <- eventReactive (input$classifyButton, {
    set.seed(rseed)
    test_events <- process_data()$df
    if (input$classifierType == 'ROCCA Classifier') {
      probs <- data.frame(predict(model, test_events, type='prob'))
    } else if (input$classifierType == 'delphinID Classifier') {
      probs <- data.frame(predict(model_D, test_events, type='prob'))
    }
    
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
    if (input$classifierType == 'delphinID Classifier') {
      colnames(pred_df)[colnames(pred_df) == "whistles"] <- "whistleFrames"
      colnames(pred_df)[colnames(pred_df) == "clicks"] <- "clickFrames"
    }
    
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
    
    ymax <- max(df_full$n)
    block <- as.integer(max(df_full$n)/10)
    if (block < 1) {
      block <- 1
    }
    
    ggplot(data = df_full, aes(x = predictedSpecies, y=n, fill = predictedSpecies)) +
      geom_bar(stat='identity', fill='#a6b8a6', color = "#a6b8a6") +
      theme_minimal() +
      scale_y_continuous(
        breaks = seq(1, ymax, by = block)) + 
      theme(
        axis.text.x = element_text(size = 12),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),  
        axis.line.x = element_blank(), 
        legend.position = "none",
        plot.margin = unit(c(0, 0, 0, 0), "cm"))
  }, width = 900, height = 300, res = 96)
  
  output$table1 <- renderDT({
    show_table <- predictions()$preds
    shinyjs::hide("loading")
    ind <- which(names(show_table) == 'score')
    show_table <- show_table[,1:ind]
    show_table <- show_table[, !(names(show_table) %in% c('prom', 'conf'))]
    show_table$duration <- as.integer(show_table$duration)
    
    datatable(show_table, options=list(pageLength=5, dom='tip', paging=TRUE)) %>% formatRound(columns = c("score"), digits = 3)
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
  
  output$ctableSelectUI <- renderUI({
  req(input$file1)  # Ensure file is uploaded
  req(input$classifierType == "delphinID Classifier")  # Show only if delphinID is selected
  
  # Try to extract table names
  db_con <- dbConnect(RSQLite::SQLite(), input$file1$datapath)
  on.exit(dbDisconnect(db_con))  # Ensure disconnection after use
  
  tables <- dbListTables(db_con)  # Get table names
  if (length(tables) == 0) return(NULL)  # Return NULL if no tables found
  
  tables_new <- c()
  s <- NULL
  for (t in tables) {
    if (grepl('_Predictions', t)) {
      inds <- as.numeric(gregexpr('_', t)[[1]])
      tables_new <- append(tables_new, substr(t, 1, inds[length(inds)]-1))
      if (grepl('click', t)|grepl('Click', t)) {
        s <- t
      }
    }
  }
  if (!is.null(s)) {
    inds <- as.numeric(gregexpr('_', s)[[1]])
    s <- substr(s, 1, inds[length(inds)]-1)
  }
  
  selectInput("cTable", "delphinID click classifier", choices = tables_new, selected=s)
})
  
  output$wtableSelectUI <- renderUI({
    req(input$file1)  # Ensure file is uploaded
    req(input$classifierType == "delphinID Classifier")  # Show only if delphinID is selected
    
    # Try to extract table names
    db_con <- dbConnect(RSQLite::SQLite(), input$file1$datapath)
    on.exit(dbDisconnect(db_con))  # Ensure disconnection after use
    
    tables <- dbListTables(db_con)  # Get table names
    if (length(tables) == 0) return(NULL)  # Return NULL if no tables found
    
    tables_new <- c()
    s <- NULL
    for (t in tables) {
      if (grepl('_Predictions', t)) {
        inds <- as.numeric(gregexpr('_', t)[[1]])
        tables_new <- append(tables_new, substr(t, 1, inds[length(inds)]-1))
        if (grepl('w', t)|grepl('W', t)|grepl('whistle', t)|grepl('Whistle', t)) {
          s <- t
        }
      }
    }
    if (!is.null(s)) {
      inds <- as.numeric(gregexpr('_', s)[[1]])
      s <- substr(s, 1, inds[length(inds)]-1)
    }
    
    selectInput("wTable", "delphinID whistle classifier", choices = tables_new, selected=s)
  })
  
}

shinyApp(ui, server, options = list(launch.browser = TRUE))

