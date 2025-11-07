try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)
library(readr)
library(dplyr)
library(caret)
library(rpart)
library(e1071)
library(randomForest)
library(gbm)
library(xgboost)
library(ipred)
library(class)
train_and_evaluate_cat_model <- function(model_name, model_info, train_data, test_data) {
  trained_model <- train(subscription ~ ., data = train_data, method = model_info$method)
  predictions <- predict(trained_model, newdata = test_data)
  confusion_matrix <- confusionMatrix(predictions, test_data$subscription)
  cat("Confusion Matrix for", model_name, ":\n")
  print(confusion_matrix)
}
train_and_evaluate_num_model <- function(model_name, model, train_data, test_data) {
  trained_model <- train(subscription ~ ., data = train_data, method = model)
  predictions <- predict(trained_model, newdata = test_data)
  confusion_matrix <- confusionMatrix(predictions, test_data$subscription)
  cat("Confusion Matrix for", model_name, ":\n")
  print(confusion_matrix)
}
data <- read.csv("C:/Users/aquil/OneDrive/Documents/ALY 6040 DATA MINING/FINAL PROJECT/bank.csv")
categorical_cols <- c("age", "job", "marital", "education", "default", "housing", "loan", 
                      "contact", "month", "day_of_week", "poutcome", "subscription")
data[categorical_cols] <- lapply(data[categorical_cols], as.factor)
set.seed(123)
train_index <- createDataPartition(data$subscription, p = 0.7, list = FALSE)
train_data_cat <- data[train_index, ]
test_data_cat <- data[-train_index, ]
constant_vars <- sapply(train_data_cat, function(x) length(unique(x)) == 1)
train_data_cat <- train_data_cat[, !constant_vars, drop = FALSE]
test_data_cat <- test_data_cat[, !constant_vars, drop = FALSE]
single_level_factors <- sapply(train_data_cat, function(x) length(levels(x)) == 1)
train_data_cat <- train_data_cat[, !single_level_factors, drop = FALSE]
test_data_cat <- test_data_cat[, !single_level_factors, drop = FALSE]
models_cat <- list(
  "Decision Tree" = list(method = "rpart"),
  "Naive Bayes" = list(method = "naive_bayes"),
  "Gradient Boosting" = list(method = "gbm")
)
train_and_evaluate_cat_model <- function(model_name, model_info, train_data, test_data) {
  if (model_name == "Decision Tree") {
    model <- train(subscription ~ ., data = train_data, method = "rpart")
  } else if (model_name == "Naive Bayes") {
    model <- train(subscription ~ ., data = train_data, method = "naive_bayes")
  } else if (model_name == "Gradient Boosting") {
    model <- train(subscription ~ ., data = train_data, method = "gbm")
  }
  predictions <- predict(model, newdata = test_data)
  confusion_matrix <- confusionMatrix(predictions, test_data$subscription)
  cat("Confusion Matrix for", model_name, ":\n")
  print(confusion_matrix)
}
for (model_name in names(models_cat)) {
  train_and_evaluate_cat_model(model_name, models_cat[[model_name]], train_data_cat, test_data_cat)
}


#dont run 

#DECISION TREE HEATMAP
conf_matrix <- matrix(c(4239, 90, 79, 90), nrow = 2, byrow = TRUE,
heatmap(conf_matrix, 
        col = heat.colors(10), 
        xlab = "Predicted Label", 
        ylab = "True Label",
        main = "Confusion Matrix")
text(x = rep(1:2, each = 2), y = rep(1:2, 2), labels = conf_matrix, pos = c(1, 3), col = "white")
# NAIVE BAYES HEATMAP
conf_matrix <- matrix(c(4318, 0, 180, 0), nrow = 2, byrow = TRUE,
                      dimnames = list(Actual = c("no", "yes"), Predicted = c("no", "yes")))
heatmap(conf_matrix, 
        col = heat.colors(10), 
        xlab = "Predicted Label", 
        ylab = "True Label",
        main = "Confusion Matrix")
text(x = rep(1:2, each = 2), y = rep(1:2, 2), labels = conf_matrix, pos = c(1, 3), col = "white")

# GRADIENT BOOSTING HEATMAP
conf_matrix <- matrix(c(4272, 46, 122, 58), nrow = 2, byrow = TRUE,
                      dimnames = list(Actual = c("no", "yes"), Predicted = c("no", "yes")))
heatmap(conf_matrix, 
        col = heat.colors(10), 
        xlab = "Predicted Label", 
        ylab = "True Label",
        main = "Confusion Matrix")
text(x = rep(1:2, each = 2), y = rep(1:2, 2), labels = conf_matrix, pos = c(1, 3), col = "white")
