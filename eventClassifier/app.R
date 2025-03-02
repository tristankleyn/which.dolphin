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
  tags$head(
    tags$title("eventClassifier"),  
    tags$link(rel = "icon", type = "image/png", href = "favicon.png")  # Sets the favicon
  ),
  
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
    selectInput("selectDB", "Select database", c("exampleDB_Atlantic", "exampleDB_ES2019", "exampleDB_BB2022", "exampleDB_WS2024", "exampleDB_SD2025", "exampleDB_HWDTmixed", "exampleDB_StTt", "trackDB"), selected="exampleDB"),
    uiOutput("ctableSelectUI"),
    uiOutput("wtableSelectUI"), 
    sliderInput("evScore", "Minimum decision score", 0, 0.2, 0, step=0.025),
    sliderInput("minClicks", "Minimum click predictions", 0, 100, 0, step=5),
    radioButtons(inputId = "AndOr", label = NULL, choices = list("AND" = 1, "OR" = 2)),
    sliderInput("minWhistles", "Minimum whistle predictions", 0, 100, 0, step=5),
    dateRangeInput("dateRange", "Filter dates", start = Sys.Date() - 3652, end = Sys.Date()),
    selectInput("plotType", "Show plot", c("Counts", "Map"), selected="Counts"),
    actionButton("classifyButton", "Classify", class = "btn-primary"),
    actionButton("addLabelsButton", "Add Labels"), 
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
        column(12,
               plotOutput(outputId = "plt", width = "100%", height = "300px"))),
    ),
    
    div(
      style = "margin-top:5%;",
      fluidRow(
        column(12,
               DTOutput("table1", height = "200px"))
      ),
      fluidRow(
        column(12, align = "right",
               shinyjs::hidden(div(id = "testButtonDiv", actionButton("testButton", "Create new classifier", style = "background-color: #849d7c; color: white; margin-bottom: 20px;")))
        )
      )
    )
  ),
  
  tags$style(type="text/css",
             ".shiny-output-error { visibility: hidden; }",
             ".shiny-output-error:before { visibility: hidden; }"
  )
)

server <- function(input, output, session) {
  
  rseed <- 123 # Set a seed for reproducibility
  
  # Reactive value to store labels
  labels_rv <- reactiveVal(NULL)
  predictions_rv <- reactiveVal(NULL)
  testevents_rv <- reactiveVal(NULL)
  current_page <- reactiveVal(1)
  
  process_data <- eventReactive(input$classifyButton, {
    req(input$classifyButton)
    shinyjs::show("loading")
    
    set.seed(rseed)
    db_con <- dbConnect(RSQLite::SQLite(), sprintf('%s.sqlite3', input$selectDB))
    print(getwd())
    print(RSQLite::dbListTables(db_con))
    on.exit(dbDisconnect(db_con))
    dateRange <- input$dateRange
    if (input$classifierType == 'ROCCA Classifier') {
      testevents <- processdataRocca(db_con, dateRange)[[1]]
    } else if (input$classifierType == 'delphinID Classifier') {
      req(input$cTable)
      testevents <- processdataDelphinID(db_con, dateRange, ctable = input$cTable, wtable = input$wTable, randseed = rseed)[[1]]
    }
    
    testevents_rv(list(df = testevents))
    list(df = testevents)
  })
  
  predictions <- eventReactive(input$classifyButton, {
    req(input$classifyButton)
    set.seed(rseed)
    test_events <- process_data()$df
    
    bc <- test_events$barcode
    if (input$classifierType == 'ROCCA Classifier') {
      probs <- data.frame(predict(model, test_events, type = 'prob'))
    } else if (input$classifierType == 'delphinID Classifier') {
      probs <- data.frame(predict(model_D, test_events, type = 'prob'))
    }
    
    predspecieslist <- c('De. delphis', 'Gr. griseus', 'Gl. melas', 'La. acutus',
                         'La. albirostris', 'Or. orca', 'Tu. truncatus')
    
    ind1 <- which(names(test_events) == 'Dde_c')
    ind2 <- which(names(test_events) == 'Ttr_w')
    pcadf <- test_events[, ind1:ind2]
    PCAvars <- names(pcadf)
    
    pred_df <- data.frame()
    for (i in 1:dim(probs)[1]) {
      row <- as.numeric(probs[i, ])
      pred <- predspecieslist[which.max(row)]
      conf <- max(row)
      prom <- row[rev(order(row))][1] - row[rev(order(row))][2]
      score <- prom * conf
      predrow <- data.frame(eventID = test_events$eventID[i], clicks = as.integer(test_events$clicks[i]), whistles = as.integer(test_events$whistles[i]),
                            duration = test_events$duration[i], predictedSpecies = pred, score = score, prom = prom, conf = conf, barcode = bc[i])
      
      spcount <- 1
      for (sp in predspecieslist) {
        predrow[[sp]] <- row[spcount]
        spcount <- spcount + 1
      }
      pred_df <- rbind(pred_df, predrow)
    }
    
    if (nrow(pcadf) > 1) {
      pca_result <- prcomp(pcadf, scale. = TRUE)
    } else {
      pca_result <- prcomp(pcadf, scale. = FALSE)
    }
    
    pca_result <- data.frame(pca_result$x)
    pcadf$predictedSpecies <- pred_df$predictedSpecies
    pcadf$PC1 <- pca_result$PC1
    pcadf$PC2 <- pca_result$PC2
    pcadf$clicks <- pred_df$clicks
    pcadf$whistles <- pred_df$whistles
    pcadf$score <- pred_df$score
    
    if (input$AndOr == 1) {
      pcadf <- subset(pcadf, score >= input$evScore & clicks >= input$minClicks & whistles >= input$minWhistles)
    } else {
      pcadf <- subset(pcadf, score >= input$evScore & (clicks >= input$minClicks | whistles >= input$minWhistles))
    }
    
    allpreds <- pred_df
    if (input$AndOr == 1) {
      pred_df <- subset(pred_df, score >= input$evScore & clicks >= input$minClicks & whistles >= input$minWhistles)
    } else {
      pred_df <- subset(pred_df, score >= input$evScore & (clicks >= input$minClicks | whistles >= input$minWhistles))
    }
    
    pred_df <- subset(pred_df, clicks > 0 | whistles > 0)
    
    evScore <- input$evScore
    if (input$classifierType == 'delphinID Classifier') {
      colnames(pred_df)[colnames(pred_df) == "whistles"] <- "whistleFrames"
      colnames(pred_df)[colnames(pred_df) == "clicks"] <- "clickFrames"
    }
    
    labels_rv(NULL) #reset labels.
    shinyjs::show("testButtonDiv") # Show the button
    
    predictions_rv(list(preds = pred_df, allpreds = allpreds, evScore = evScore, PCAdf = pcadf)) #update reactive value
    
    list(preds = pred_df, allpreds = allpreds, evScore = evScore, PCAdf = pcadf)
  })
  
  
  observeEvent(input$addLabelsButton, {
    preds <- predictions_rv()$preds
    labels_rv(data.frame(eventID = preds$eventID, label = character(nrow(preds)), stringsAsFactors = FALSE))
  })
  
  observeEvent(input$testButton, {
    showModal(modalDialog(
      title = "Training new classifier...",
      easyClose = TRUE,
      footer = NULL
    ))
    
    if (!is.null(testevents_rv())) {
      isolate({
        current_events <- testevents_rv()
        if (!is.null(current_events$df)) {
          current_events$df <- left_join(current_events$df, labels_rv(), by = "eventID")
          testevents_rv(current_events) # Update the reactive value
          
          # Filtering and display after update
          filtered_events <- testevents_rv()$df %>%
            filter(label != "")
          
          splist <- unique(filtered_events$label)
          splist <- splist[order(splist)]
          print(splist)
          sampsizes <- rep(min(table(filtered_events$label)), length(unique(filtered_events$label)))
          m_all <- randomForest(as.factor(label) ~ Dde_c + Ggr_c + Gme_c + Lal_c + Ttr_c + Dde_w + Ggr_w + Gme_w + Lac_w + Lal_w + Oor_w + Ttr_w,
                            data=filtered_events, ntree=500, strata=as.factor(filtered_events$label), sampsize=sampsizes, na.action = na.roughfix)
          
          dgn <- data.frame()
          for (k in 1:nrow(filtered_events)) {
            ev <- filtered_events$eventID[k]
            test <- subset(filtered_events, eventID == ev)
            train <- subset(filtered_events, eventID != ev)
            testsp <- test$label[1]
            
            sampsizes <- rep(min(table(train$label)), length(unique(train$label)))
            m <- randomForest(as.factor(label) ~ Dde_c + Ggr_c + Gme_c + Lal_c + Ttr_c + Dde_w + Ggr_w + Gme_w + Lac_w + Lal_w + Oor_w + Ttr_w,
                              data=train, ntree=500, strata=as.factor(train$label), sampsize=sampsizes, na.action = na.roughfix)
            
            probs <- predict(m, test, type='prob')
            pred <- splist[which.max(probs)]
            conf <- max(probs)
            prom <- probs[rev(order(probs))][1] - probs[rev(order(probs))][2]
            probs <- data.frame(probs)
            score <- prom*conf
            
            predrow <- data.frame(label=testsp, eventID=ev, pred=pred, conf=conf, prom=prom, score=score)
            for (lab in splist) {
              predrow[[lab]] <- probs[[lab]]
            }
            
            dgn <- rbind(dgn, predrow)
          }
          
          
          
          saveRDS(m_all, paste("eventClassifier_", Sys.Date(), ".rds", sep = ""))
          
          dgn$correct <- as.integer(dgn$pred == dgn$label)
          acc <- mean(dgn$correct)
          acc_formatted <- sprintf("%.3f", acc) # Round to 3 decimal places
          write.table(dgn, "model_diagnostics.csv", sep=',', row.names=FALSE)
          
          
          calc_lab_acc <- function(dgn) {
            accuracies <- list()
            accuracies[['All labels']] <- mean(dgn$correct)
            ulabs <- unique(dgn$label)
            ulabs <- ulabs[order(ulabs)]
            for (lab in ulabs) {
              sub <- subset(dgn, label == lab)
              sub_acc <- mean(sub$correct)
              accuracies[[lab]] <- sub_acc
            }
            
            return(accuracies)
          }
          
          # Convert head_table to a formatted string for display
          
          accuracies <- calc_lab_acc(dgn)
          output_lines <- c("ESTIMATED CLASSIFICATION ACCURACY:")
          for (label in names(accuracies)) {
            acc_formatted <- sprintf("%.3f", accuracies[[label]])
            output_lines <- c(output_lines, paste0("", label, ": ", acc_formatted))
          }
          
          output_str <- paste(output_lines, collapse = "<br>")
          
          showModal(modalDialog(
            title = "Exported new event classifier to eventClassifier/",
            HTML(paste0("<div style='font-size: 17px;'>", output_str, "</div>")), # Increase font size
            easyClose = TRUE,
            footer = NULL
          ))
        }
      })
    }
    
  })
  
  output$plt <- renderPlot({
    req(input$classifyButton)
    req(predictions())
    req(if(input$plotType == "Counts"){predictions_rv()$preds} else{predictions()$PCAdf})
    
    if (input$plotType == "Counts") {
      plot_data <- predictions_rv()
      shinyjs::hide("loading")
      df <- plot_data$preds
      x <- seq(0,1,0.01)
      y <- as.numeric(plot_data$evScore)/x
      thr <- data.frame(x=x, y=y)
      
      custom_colors <- c('De. delphis'='royalblue',
                         'Gr. griseus'='darkred',
                         'Gl. melas'='forestgreen',
                         'La. acutus'='purple',
                         'La. albirostris'='grey',
                         'Or. orca'='orange',
                         'Tu. truncatus'='turquoise')
      
      all_levels <- c('De. delphis', 'Gr. griseus', 'Gl. melas', 'La. acutus',
                      'La. albirostris', 'Or. orca', 'Tu. truncatus')
      
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
        geom_bar(stat='identity', fill='#849d7c', color = "#849d7c") +
        theme_minimal() +
        scale_y_continuous(
          breaks = seq(1, ymax, by = block)) +
        theme(
          axis.text.x = element_text(size = 8),
          axis.ticks.x = element_blank(),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.line.x = element_blank(),
          legend.position = "none",
          plot.margin = unit(c(0, 0, 0, 0), "cm"))
    } else {
      plot_data <- predictions_rv()
      shinyjs::hide("loading")
      df <- plot_data$PCAdf
      df <- subset(df, clicks > 0 | whistles > 0)
      custom_colors <- c('De. delphis'='royalblue',
                         'Gr. griseus'='darkred',
                         'Gl. melas'='forestgreen',
                         'La. acutus'='purple',
                         'La. albirostris'='grey',
                         'Or. orca'='orange',
                         'Tu. truncatus'='turquoise')
      
      all_levels <- c('De. delphis', 'Gr. griseus', 'Gl. melas', 'La. acutus',
                      'La. albirostris', 'Or. orca', 'Tu. truncatus')
      
      label_counts <- df %>%
        group_by(predictedSpecies) %>%
        summarise(n = n()) %>%
        mutate(label_n = paste0(predictedSpecies, " (n = ", n, ")"))
      
      df <- df %>% left_join(label_counts, by='predictedSpecies')
      
      ggplot(df, aes(x = PC1, y = PC2, color = label_n, label=1:nrow(df))) +
        geom_text(size = 3, alpha = 0.6, show.legend = FALSE) +
        stat_ellipse(level = 0.90) +
        theme_minimal(base_size = 8) +
        theme(legend.text = element_text(size = 8),
              axis.title.x = element_text(size=8),
              axis.title.y = element_text(size=8),
        ) +
        xlab('Component 1') +
        ylab('Component 2') +
        guides(color = guide_legend(title = "Classified species"))
    }
    
  }, width = 1000, height = 300, res = 192)
  
  output$table1 <- renderDT({
    req(input$classifyButton)
    req(predictions_rv())
    
    show_table <- predictions_rv()$preds
    bc <- show_table$barcode
    shinyjs::hide("loading")
    ind <- which(names(show_table) == 'score')
    show_table <- show_table[, 1:ind]
    show_table <- show_table[, !(names(show_table) %in% c('prom', 'conf'))]
    show_table[['delphinID']] <- bc
    show_table$duration <- as.integer(show_table$duration)
    row.names(show_table) <- NULL
    
    # Merge labels if available
    if (!is.null(labels_rv())) {
      show_table <- left_join(show_table, labels_rv(), by = "eventID")
    }
    
    datatable(show_table, options = list(pageLength = 5, dom = 'tip', paging = FALSE), editable = TRUE) %>%
      formatRound(columns = c("score"), digits = 3)
  })
  
  observeEvent(input$table1_cell_edit, {
    info <- input$table1_cell_edit
    if (!is.null(labels_rv())) {
      labels <- labels_rv()
      edited_row <- predictions_rv()$preds[info$row, ]
      event_id <- edited_row$eventID
      labels$label[labels$eventID == event_id] <- as.character(info$value)
      labels_rv(labels)
      
      # Update the datatable using dataTableProxy and replaceData
      show_table <- predictions_rv()$preds
      if (!is.null(labels_rv())) {
        show_table <- left_join(show_table, labels_rv(), by = "eventID")
      }
      
      page_info <- input$table1_state
      if (!is.null(page_info)) {
        current_page(page_info$start / 5 + 1)
      }

      replaceData(dataTableProxy("table1"), show_table)
    }
  })
  
  output$downloadFiltered <- downloadHandler(
    filename = function() {
      paste("ClassifiedEvents", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      preds <- predictions()$preds
      if (!is.null(labels_rv())) {
        preds <- left_join(preds, labels_rv(), by = "eventID")
      }
      write.csv(preds, file, row.names = FALSE)
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
    req(input$classifierType == "delphinID Classifier")  # Show only if delphinID is selected
    
    # Try to extract table names
    
    db_con <- dbConnect(RSQLite::SQLite(), sprintf('%s.sqlite3', input$selectDB))
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
    req(input$classifierType == "delphinID Classifier")  # Show only if delphinID is selected
    
    # Try to extract table names
    db_con <- dbConnect(RSQLite::SQLite(), sprintf('%s.sqlite3', input$selectDB))
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