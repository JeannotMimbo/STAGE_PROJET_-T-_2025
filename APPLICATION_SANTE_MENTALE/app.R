# --------------------------
# Chargement des librairies
# --------------------------
library(shiny)          # Application web interactive
library(randomForest)   # Modèle Random Forest
library(rpart)          # Arbre de décision
library(e1071)          # SVM
library(pROC)           # Courbe ROC
library(shinythemes)    # Thèmes visuels pour Shiny

# --------------------------
# Chargement des modèles pré-entraînés
# --------------------------
modele_logit <- readRDS("~/Documents/Jeannot_projet/Student/modeles/modele_logit.rds")
rf_model     <- readRDS("~/Documents/Jeannot_projet/Student/modeles/rf_model.rds")
tree_model   <- readRDS("~/Documents/Jeannot_projet/Student/modeles/tree_model.rds")
svm_model    <- readRDS("~/Documents/Jeannot_projet/Student/modeles/svm_model.rds")
data_model   <- readRDS("~/Documents/Jeannot_projet/Student/modeles/data_model.rds")

# Extraire les niveaux de facteurs utilisés lors de l'entraînement
levels_year   <- levels(model.frame(modele_logit)$year_of_study)
levels_gender <- levels(model.frame(modele_logit)$gender)

# --------------------------
# Interface utilisateur (UI)
# --------------------------
ui <- fluidPage(
  theme = shinytheme("flatly"),  # Choix du thème
  
  # Titre de l'application avec icône
  titlePanel(title = div(
    icon("heartbeat"), 
    "Application de Prédiction de la Dépression Chez les Étudiants"
  )),
  
  sidebarLayout(
    sidebarPanel(
      h4("Informations de l'étudiant"),
      
      # Entrée utilisateur : données personnelles
      numericInput("age", "Âge :", value = 20, min = 15, max = 30),
      numericInput("cgpa", "Moyenne Générale (CGPA) :", value = 3.0, min = 0, max = 4, step = 0.01),
      selectInput("gender", "Genre :", choices = levels_gender),
      selectInput("year_of_study", "Année d'étude :", choices = levels_year),
      
      tags$hr(),
      h4("Symptômes psychologiques"),
      
      # Entrée utilisateur : symptômes
      selectInput("anxiety", "Souffrez-vous d'anxiété ?", choices = c("Oui" = 1, "Non" = 0)),
      selectInput("panic_attack", "Avez-vous des crises de panique ?", choices = c("Oui" = 1, "Non" = 0)),
      selectInput("sought_help", "Avez-vous déjà cherché de l’aide ?", choices = c("Oui" = 1, "Non" = 0)),
      
      tags$hr(),
      h4("Paramètres de prédiction"),
      
      # Choix du modèle de machine learning
      selectInput("modele", "Choisissez le modèle à utiliser :", 
                  choices = c("Régression Logistique" = "logit",
                              "Random Forest" = "rf",
                              "Arbre de Décision" = "tree",
                              "SVM" = "svm")),
      
      # Bouton d'action
      actionButton("predict", "Prédire", icon = icon("magic"), class = "btn btn-success btn-block")
    ),
    
    mainPanel(
      # Onglets d'affichage des résultats
      tabsetPanel(
        tabPanel("Résultat de la prédiction",
                 br(),
                 h4("Résultat :"),
                 verbatimTextOutput("result"),
                 tags$hr(),
                 h5("Interprétation du résultat"),
                 htmlOutput("interpretation")  # Affichage du message dynamique
        ),
        
        tabPanel("Courbes ROC comparées",
                 br(),
                 plotOutput("rocMulti")),
        
        tabPanel(" Comparaison des précisions",
                 br(),
                 plotOutput("bar_accuracy")),
        
        tabPanel("Statistiques sur la base de données",
                 br(),
                 h5("Nombre d'étudiants en dépression par genre :"),
                 verbatimTextOutput("depression_by_gender"))
      )
    )
  ) 
)

# --------------------------
# Partie serveur
# --------------------------
server <- function(input, output) {
  # Variable réactive pour suivre si une prédiction a été faite
  pred_done <- reactiveVal(FALSE)
  
  # Lorsque l'utilisateur clique sur "Prédire"
  observeEvent(input$predict, {
    pred_done(TRUE)  # Active l'état de prédiction
    
    # Préparer les données utilisateur pour la prédiction
    new_data <- data.frame(
      age = input$age,
      cgpa = input$cgpa,
      anxiety = as.integer(input$anxiety),
      panic_attack = as.integer(input$panic_attack),
      sought_help = as.integer(input$sought_help),
      year_of_study = factor(input$year_of_study, levels = levels_year),
      gender = factor(input$gender, levels = levels_gender)
    )
    
    # Statistiques sur la base de données
    output$depression_by_gender <- renderPrint({
      table(data_model$gender[data_model$depression == 1])
    })
    
    # Affichage des courbes ROC pour comparaison des modèles
    output$rocMulti <- renderPlot({
      proba_logit <- predict(modele_logit, newdata = data_model, type = "response")
      roc_logit <- roc(data_model$depression, proba_logit)
      
      svm_probs <- attr(predict(svm_model, newdata = data_model, probability = TRUE), "probabilities")[, "1"]
      roc_svm <- roc(data_model$depression, svm_probs)
      
      tree_probs <- predict(tree_model, newdata = data_model, type = "prob")[, "1"]
      roc_tree <- roc(data_model$depression, tree_probs)
      
      plot(roc_logit, col = "blue", main = "Courbes ROC")
      lines(roc_svm, col = "red")
      lines(roc_tree, col = "green")
      legend("bottomright", legend = c("Logistique", "SVM", "Arbre"),
             col = c("blue", "red", "green"), lty = 1, lwd = 2)
    })
    
    # Affichage des précisions des modèles sous forme de barplot
    output$bar_accuracy <- renderPlot({
      accuracies <- c(Logit = 0.80, Arbre = 0.75, RF = 0.85, SVM = 0.78)
      barplot(accuracies, main = "Accuracies", col = c("skyblue", "lightgreen", "orange", "pink"), ylim = c(0, 1))
    })
    
    # Prédiction selon le modèle choisi
    prediction <- tryCatch({
      if (input$modele == "logit") {
        proba <- predict(modele_logit, newdata = new_data, type = "response")
        ifelse(proba > 0.5, "Dépression probable", "Pas de dépression")
      } else if (input$modele == "rf") {
        pred <- predict(rf_model, newdata = new_data)
        ifelse(pred == 1, "Dépression probable", "Pas de dépression")
      } else if (input$modele == "tree") {
        pred <- predict(tree_model, newdata = new_data, type = "class")
        ifelse(pred == 1, "Dépression probable", "Pas de dépression")
      } else if (input$modele == "svm") {
        pred_probs <- attr(predict(svm_model, newdata = new_data, probability = TRUE), "probabilities")[, "1"]
        ifelse(pred_probs > 0.5, "Dépression probable", "Pas de dépression")
      }
    }, error = function(e) {
      paste("Erreur :", e$message)
    })
    
    # Affichage du résultat de la prédiction
    output$result <- renderText({ prediction })
    
    # Affichage dynamique du message selon le résultat
    output$interpretation <- renderUI({
      if (grepl("Dépression probable", prediction)) {
        HTML("<p style='color:red;'><strong>⚠️ Risque élevé de dépression détecté.</strong><br>Il est conseillé de consulter un professionnel de santé mentale.</p>")
      } else if (grepl("Pas de dépression", prediction)) {
        HTML("<p style='color:green;'><strong>✅ Aucun signe de dépression détecté.</strong><br>Continuez à prendre soin de votre santé mentale !</p>")
      } else {
        HTML("")
      }
    })
  })
  
  # Permet au client (UI) de savoir si une prédiction a été faite
  output$prediction_done <- reactive({
    pred_done()
  })
  outputOptions(output, "prediction_done", suspendWhenHidden = FALSE)
}

# --------------------------
# Lancement de l'application
# --------------------------
shinyApp(ui = ui, server = server)
