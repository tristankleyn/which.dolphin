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
rds_files <- list.files(pattern = "\\.rds$") # Get all files ending with .rds
source('appFunctions.R')


####SHINY####
ui <- fluidPage(
  tags$head(
    tags$title("eventClassifier"),
    tags$link(rel = "icon", type = "image/png", href = "favicon.png") # Sets the favicon
  ),
  
  useShinyjs(),
  
  tags$head(
    tags$style(HTML("
      .shiny-title {
        text-align: left;
        color: #97a695;
        font-size: 24px;
        font-weight: lighter;
        margin-left: 20px;
      }

      .loading-container {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        width: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
      }

      .loading-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #97a695;
        animation: pulse 1.5s infinite ease-in-out;
        margin-top: 5px;
      }

      .loading-text {
        position: relative;
        top: -45px;
        font-size: 16px;
        color: #97a695;
        font-weight: lighter;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 200px;
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
    selectInput("selectDB", "Select database", c("exampleDB_ES2019", "exampleDB_BB2022", "exampleDB_WS2024", "trackDB"), selected = "trackDB"),
    selectInput("classifierType", "Select supported PAMGuard classifier", c("ROCCA Classifier", "delphinID Classifier"), "delphinID Classifier"),
    selectInput("classifierSelect", "Select event classifier", choices = NULL),
    uiOutput("ctableSelectUI"),
    uiOutput("wtableSelectUI"),
    sliderInput("evScore", "Minimum decision score", 0, 0.2, 0, step = 0.025),
    sliderInput("minClicks", "Minimum click predictions", 0, 100, 0, step = 5),
    radioButtons(inputId = "AndOr", label = NULL, choices = list("AND" = 1, "OR" = 2)),
    sliderInput("minWhistles", "Minimum whistle predictions", 0, 100, 0, step = 5),
    dateRangeInput("dateRange", "Filter dates", start = Sys.Date() - 3652, end = Sys.Date()),
    selectInput("plotType", "Show plot", c("Counts", "Map"), selected = "Counts"),
    actionButton("classifyButton", "Classify", class = "btn-primary"),
    actionButton("addLabelsButton", "Add Labels"),
    actionButton("groupEventsButton", "Group events"),
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
      div(style = "height: 20px;"),
      fluidRow(
        column(12,
               column(6, align = "left", shinyjs::hidden(div(id = "bulkLabelDiv", style = "display: inline-block;",
                                                             div(style = "display: inline-block;", numericInput("startRow", "Start row", value = 1, min = 1, width = "100px")),
                                                             div(style = "display: inline-block;", numericInput("endRow", "End row", value = 1, min = 1, width = "100px")),
                                                             div(style = "display: inline-block;", textInput("bulkLabelText", "Label", width = "150px")),
                                                             actionButton("bulkLabelButton", "Bulk label", style = "background-color: #849d7c; color: white; margin-bottom: 20px; display: inline-block;")
               ))),
               column(6, align = "right", shinyjs::hidden(div(id = "testButtonDiv", style = "display: inline-block;", actionButton("testButton", "Create new classifier", style = "background-color: #849d7c; color: white; margin-bottom: 20px;"))))
        ),
        fluidRow(
          column(12,
                 column(6, align = "left", shinyjs::hidden(div(id = "bulkEventIDDiv", style = "display: inline-block;",
                                                               div(style = "display: inline-block;", numericInput("startRowEventID", "Start row", value = 1, min = 1, width = "100px")),
                                                               div(style = "display: inline-block;", numericInput("endRowEventID", "End row", value = 1, min = 1, width = "100px")),
                                                               div(style = "display: inline-block;", textInput("bulkEventIDText", "Event group", width = "150px")),
                                                               actionButton("bulkEventIDButton", "Bulk group label", style = "background-color: #849d7c; color: white; margin-bottom: 20px; display: inline-block;")
                 )))
          )
        )
      )
    )),
  
  tags$style(type = "text/css",
             ".shiny-output-error { visibility: hidden; }",
             ".shiny-output-error:before { visibility: hidden; }"),
)

server <- function(input, output, session) {
  
  rseed <- 123 # Set a seed for reproducibility
  
  # Reactive value to store labels
  labels_rv <- reactiveVal(NULL)
  predictions_rv <- reactiveVal(NULL)
  testevents_rv <- reactiveVal(NULL)
  current_page <- reactiveVal(1)
  group_events_clicked <- reactiveVal(FALSE)
  
  shinyjs::hide("bulkLabelDiv")
  shinyjs::hide("groupEventsButton")
  
  observe({
    rds_files <- list.files(pattern = "\\.rds$")
    classifier_type <- input$classifierType
    
    # Filter files based on classifier type
    if (classifier_type == "ROCCA Classifier") {
      filtered_files <- grep("rocca|ROCCA", rds_files, value = TRUE, ignore.case = TRUE)
    } else if (classifier_type == "delphinID Classifier") {
      filtered_files <- grep("delphinid|delphinID", rds_files, value = TRUE, ignore.case = TRUE)
    } else {
      filtered_files <- rds_files # if classifier type is not selected, show all.
    }
    
    # Determine default selection
    selected_file <- if (length(filtered_files) > 0) filtered_files[1] else ""
    
    # Update selectInput
    updateSelectInput(session, "classifierSelect", choices = rds_files, selected = selected_file)
  })
  
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
    
    testevents$uid <- 1:nrow(testevents)
    testevents_rv(list(df = testevents))
    list(df = testevents)
  })
  
  predictions <- eventReactive(input$classifyButton, {
    req(input$classifyButton)
    set.seed(rseed)
    
    model <- readRDS(input$classifierSelect)
    test_events <- process_data()$df
    
    bc <- test_events$barcode
    probs <- data.frame(predict(model, test_events, type = 'prob'))
    
    if (input$classifierSelect == 'EventClassifier_delphinID.rds' | input$classifierSelect == 'EventClassifier_ROCCA.rds') {
      predspecieslist <- c('De. delphis', 'Gr. griseus', 'Gl. melas', 'La. acutus',
                           'La. albirostris', 'Or. orca', 'Tu. truncatus')
    } else { 
      predspecieslist <- model$classes
      }
    
    
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
      predrow <- data.frame(uid=test_events$uid[i], eventID = test_events$eventID[i], clicks = as.integer(test_events$clicks[i]), whistles = as.integer(test_events$whistles[i]),
                            minutes = test_events$minutes[i], predictedSpecies = pred, score = score, prom = prom, conf = conf, barcode = bc[i])
      
      spcount <- 1
      for (sp in predspecieslist) {
        predrow[[sp]] <- row[spcount]
        spcount <- spcount + 1
      }
      pred_df <- rbind(pred_df, predrow)
    }
    pred_df$eventGroup <- NA

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
    
    allpreds <- subset(allpreds, clicks > 0 | whistles > 0)
    pred_df <- subset(pred_df, clicks > 0 | whistles > 0)
    
    evScore <- input$evScore
    if (input$classifierType == 'delphinID Classifier') {
      colnames(pred_df)[colnames(pred_df) == "whistles"] <- "whistleFrames"
      colnames(pred_df)[colnames(pred_df) == "clicks"] <- "clickFrames"
    }
    
    labels_rv(NULL) #reset labels.

    
    predictions_rv(list(preds = pred_df, allpreds = allpreds, evScore = evScore, PCAdf = pcadf)) #update reactive value
    
    list(preds = pred_df, allpreds = allpreds, evScore = evScore, PCAdf = pcadf)
  })
  
  
  observeEvent(input$addLabelsButton, {
    shinyjs::show("testButtonDiv") # Show make new classifier button
    shinyjs::show("bulkLabelDiv") # Show bulk labelling options
    shinyjs::show("bulkEventIDDiv")
    shinyjs::show("groupEventsButton")
    
    preds <- predictions_rv()$preds
    if (group_events_clicked()) {
      labels_rv(data.frame(uid = preds$uid, label = character(nrow(preds)), eventGroup = character(nrow(preds)), stringsAsFactors = FALSE))
    } else {
      labels_rv(data.frame(uid = preds$uid, label = character(nrow(preds)), stringsAsFactors = FALSE))
    }
  })
  
  observeEvent(input$groupEventsButton, {
    group_events_clicked(TRUE)
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
          current_events$df <- left_join(current_events$df, labels_rv(), by = "uid")
          
          if (group_events_clicked() && !is.null(predictions_rv())) {
            if ("eventGroup" %in% names(current_events$df)) {
              current_events$df <- current_events$df %>%
                left_join(predictions_rv()$preds[, c("uid", "eventGroup")], by = "uid", suffix = c(".old", "")) %>%
                mutate(eventGroup = ifelse(is.na(eventGroup), eventGroup.old, eventGroup)) %>%
                select(-eventGroup.old)
            } else {
              current_events$df <- left_join(current_events$df, predictions_rv()$preds[, c("uid", "eventGroup")], by = "uid")
            }
          }
          
          testevents_rv(current_events) # Update the reactive value
          
          # Filtering and display after update
          filtered_events <- testevents_rv()$df %>%
            filter(label != "")
          
          if ('eventGroup' %in% names(filtered_events)) {
            filtered_events$eventGroup <- ifelse(is.na(filtered_events$eventGroup), filtered_events$eventID, filtered_events$eventGroup)
          }
          
          splist <- unique(filtered_events$label)
          splist <- splist[order(splist)]
          print('Classes found:')
          print(splist)
          
          sampsizes <- rep(min(table(filtered_events$label)), length(unique(filtered_events$label)))
          m_all <- randomForest(as.factor(label) ~ Dde_c + Ggr_c + Gme_c + Lal_c + Ttr_c + Dde_w + Ggr_w + Gme_w + Lac_w + Lal_w + Oor_w + Ttr_w,
                                data = filtered_events, ntree = 500, strata = as.factor(filtered_events$label), sampsize = sampsizes, na.action = na.roughfix
          )
          
          dgn <- data.frame()
          if ('eventGroup' %in% names(filtered_events)) {
            uevents <- unique(filtered_events$eventGroup)[order(unique(filtered_events$eventGroup))]
          } else {
            uevents <- unique(filtered_events$eventID)[order(unique(filtered_events$eventID))]
          }
          
          for (k in 1:length(uevents)) {
            uid_t <- filtered_events$uid[k]
            test <- subset(filtered_events, uid == uid_t)
            if ('eventGroup' %in% names(filtered_events)) {
              ev <- test$eventGroup[1]
              train <- subset(filtered_events, eventGroup != ev)
            } else {
              ev <- test$eventID[1]
              train <- subset(filtered_events, eventID != ev)
            }
            
            testsp <- test$label[1]
            sampsizes <- rep(min(table(train$label)), length(unique(train$label)))
            m <- randomForest(as.factor(label) ~ Dde_c + Ggr_c + Gme_c + Lal_c + Ttr_c + Dde_w + Ggr_w + Gme_w + Lac_w + Lal_w + Oor_w + Ttr_w,
                              data = train, ntree = 500, strata = as.factor(train$label), sampsize = sampsizes, na.action = na.roughfix
            )
            
            probs <- predict(m, test, type = 'prob')
            pred <- splist[which.max(probs)]
            conf <- max(probs)
            prom <- probs[rev(order(probs))][1] - probs[rev(order(probs))][2]
            probs <- data.frame(probs)
            score <- prom * conf
            
            predrow <- data.frame(label = testsp, eventID = ev, pred = pred, conf = conf, prom = prom, score = score)
            for (lab in splist) {
              predrow[[lab]] <- probs[[lab]]
            }
            
            dgn <- rbind(dgn, predrow)
          }
          
          saveRDS(m_all, paste("eventClassifier_", Sys.Date(), ".rds", sep = ""))
          
          dgn$correct <- as.integer(dgn$pred == dgn$label)
          acc <- mean(dgn$correct)
          if (acc >= 0.75) {
            emj <- '&#x1F642;'
          } else if (acc >= 0.5) {
            emj <- '&#x1F610'
          } else {
            emj <- '&#x1F928'
          }
          
          acc_formatted <- sprintf("%.3f", acc) # Round to 3 decimal places
          write.table(dgn, "model_diagnostics.csv", sep = ',', row.names = FALSE)
          
          # Convert head_table to a formatted string for display
          
          output_lines <- sprintf("%s Estimated classification accuracy: %s", emj, acc_formatted)
          output_str <- paste(output_lines, collapse = "<br>")
          if ('eventGroup' %in% names(filtered_events)) {
            output_str1 <- sprintf("Classifier confusion matrix estimated from cross-validated testing across %s event groups shows true classes along the vertical axis and predicted classes along the horizontal. Model and diagnostics saved in eventClassifier/", length(uevents))
          } else {
            output_str1 <- sprintf("Classifier confusion matrix estimated from cross-validated testing across %s events shows true classes along the vertical axis and predicted classes along the horizontal. Model and diagnostics saved in eventClassifier/", nrow(dgn))
          }
          
          showModal(modalDialog(
            title = paste("eventClassifier_", Sys.Date(), ".rds", sep = ""),
            tags$style(HTML(".modal-content {background-color: #f0f0f0; color: #333;}")),
            HTML(paste0("<div style='font-size: 17px; font-weight: bold;'>", output_str, "</div>")),
            HTML("<br>"),
            DTOutput("table_in_modal"),
            HTML(paste0("<br><div style='font-size: 15px;'>", output_str1, "</div>")),
            easyClose = TRUE,
            footer = NULL
          ))
        }
      })
    }
  })
  
  output$table_in_modal <- renderDT({
    req(input$testButton)
    df <- read.csv('model_diagnostics.csv')
    cm <- list()
    splist <- unique(df$label)[order(unique(df$label))]
    for (lab1 in splist) {
      sub <- subset(df, label == lab1)
      vals <- c()
      for (lab2 in splist) {
        p <- sum(sub$pred == lab2)/nrow(sub)
        vals <- append(vals, p)
      }
      cm[[lab1]] <- vals
    }
    cm <- t(data.frame(cm))
    rownames(cm) <- splist
    colnames(cm) <- splist
    datatable(cm, options = list(searching=FALSE, paging=FALSE, info=FALSE)) %>% formatRound(columns=names(cm), digits=3) %>% formatStyle(names(cm), fontWeight = "bold") 
  })
  
  output$plt <- renderPlot({
    req(input$classifyButton)
    req(predictions())
    req(if(input$plotType == "Counts"){predictions_rv()$preds} else{predictions()$PCAdf})
    
    model <- readRDS(input$classifierSelect) 
    
    if (input$plotType == "Counts") {
      plot_data <- predictions_rv()
      shinyjs::hide("loading")
      df <- plot_data$preds
      x <- seq(0,1,0.01)
      y <- as.numeric(plot_data$evScore)/x
      thr <- data.frame(x=x, y=y)
      
      if (input$classifierSelect == 'EventClassifier_delphinID.rds' | input$classifierSelect == 'EventClassifier_ROCCA.rds') {
        all_levels <- c('De. delphis', 'Gr. griseus', 'Gl. melas', 'La. acutus',
                             'La. albirostris', 'Or. orca', 'Tu. truncatus')
      } else { 
        all_levels <- model$classes
      }
      
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
      
      if (input$classifierSelect == 'EventClassifier_delphinID.rds' | input$classifierSelect == 'EventClassifier_ROCCA.rds') {
        all_levels <- c('De. delphis', 'Gr. griseus', 'Gl. melas', 'La. acutus',
                        'La. albirostris', 'Or. orca', 'Tu. truncatus')
      } else { 
        all_levels <- model$classes
      }
      
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
    show_table[['barcode']] <- bc
    show_table$minutes <- round(as.numeric(show_table$minutes), digits = 1)
    row.names(show_table) <- NULL
    
    # Merge labels if available
    if (!is.null(labels_rv())) {
      show_table <- left_join(show_table, labels_rv(), by = "uid") #use uid
    }
    
    # Add eventGroup column if button clicked
    if (group_events_clicked()) {
      show_table$eventGroup <- predictions_rv()$preds$eventGroup
    }
    
    # Remove uid column
    show_table <- show_table[, !(names(show_table) %in% c('uid'))]
    
    datatable(show_table, options = list(pageLength = 5, dom = 'tip', paging = FALSE), editable = TRUE) %>%
      formatRound(columns = c("score"), digits = 3)
  })
  
  observeEvent(input$table1_cell_edit, {
    info <- input$table1_cell_edit
    if (!is.null(labels_rv())) {
      labels <- labels_rv()
      edited_row <- predictions_rv()$preds[info$row, ]
      uid <- edited_row$uid #Use uid
      
      if (info$col == which(names(predictions_rv()$preds) == "eventGroup")) {
        # EventGroup column was edited
        labels$eventGroup[labels$uid == uid] <- as.character(info$value)
      } else if (info$col == which(names(predictions_rv()$preds) == "label")) {
        # Label column was edited
        labels$label[labels$uid == uid] <- as.character(info$value)
      }
      
      labels_rv(labels)
      
      # Update the datatable using dataTableProxy and replaceData
      show_table <- predictions_rv()$preds
      if (!is.null(labels_rv())) {
        show_table <- left_join(show_table, labels_rv(), by = "uid") # Use uid
      }
      
      if (!group_events_clicked()) {
        show_table <- show_table %>% select(-eventGroup)
      }
      
      page_info <- input$table1_state
      if (!is.null(page_info)) {
        current_page(page_info$start / 5 + 1)
      }
      
      replaceData(dataTableProxy("table1"), show_table)
    }
  })
  
  observeEvent(input$bulkEventIDButton, {
    req(input$startRowEventID, input$endRowEventID, input$bulkEventIDText)
    
    start_row <- input$startRowEventID
    end_row <- input$endRowEventID
    new_eventGroup <- input$bulkEventIDText #Change variable name
    
    preds <- predictions_rv()$preds
    testevents <- testevents_rv()$df
    
    if (start_row <= end_row && start_row >= 1 && end_row <= nrow(preds)) {
      for (i in start_row:end_row) {
        uid <- preds$uid[i]
        preds$eventGroup[preds$uid == uid] <- new_eventGroup #Change column being changed
        testevents$eventID[testevents$uid == uid] <- testevents$eventID[testevents$uid == uid] #remove change to eventID
      }
      predictions_rv(list(preds = preds, allpreds = predictions_rv()$allpreds, evScore = predictions_rv()$evScore, PCAdf = predictions_rv()$PCAdf))
      testevents_rv(list(df = testevents))
      
      # Update the datatable
      show_table <- predictions_rv()$preds
      if (!is.null(labels_rv())) {
        show_table <- left_join(show_table, labels_rv(), by = "uid")
      }
      if (group_events_clicked()) {
        show_table$eventGroup <- predictions_rv()$preds$eventGroup
        show_table <- show_table[, c(1:3, ncol(show_table), 4:(ncol(show_table) - 1))]
      }
      replaceData(dataTableProxy("table1"), show_table)
    } else {
      showNotification("Invalid row indices.", type = "error")
    }
  })
  
  observeEvent(input$bulkLabelButton, {
    req(input$startRow, input$endRow, input$bulkLabelText)
    
    start_row <- input$startRow
    end_row <- input$endRow
    label_text <- input$bulkLabelText
    
    preds <- predictions_rv()$preds
    
    if (!is.null(labels_rv())) {
      labels <- labels_rv()
    } else {
      labels <- data.frame(uid = preds$uid, label = character(nrow(preds)), stringsAsFactors = FALSE) #use uid
    }
    
    if (start_row <= end_row && start_row >= 1 && end_row <= nrow(preds)) {
      for (i in start_row:end_row) {
        uid <- preds$uid[i] #use uid
        labels$label[labels$uid == uid] <- label_text #use uid
      }
      labels_rv(labels)
      
      # Update the datatable
      show_table <- predictions_rv()$preds
      show_table <- left_join(show_table, labels_rv(), by = "uid") #use uid
      replaceData(dataTableProxy("table1"), show_table)
    } else {
      showNotification("Invalid row indices.", type = "error")
    }
  })
  
  output$downloadFiltered <- downloadHandler(
    filename = function() {
      paste("ClassifiedEvents", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      preds <- predictions()$preds
      if (!is.null(labels_rv())) {
        preds <- left_join(preds, labels_rv(), by = "uid") #use uid
      }
      if (group_events_clicked()) {
        preds$eventGroup <- predictions()$preds$eventGroup
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