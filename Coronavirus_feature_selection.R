## Load necessary libraries
library(dplyr)
library(quanteda)
library(caret)
# library(e1071)
library(wordcloud)
library(ggplot2)
library(glmnet)
library(doParallel)

# Create and register the cluster
cl <- makeCluster(8)
registerDoParallel(cl)

# Step #1 Data Ingesting into R 
# download datasets, if necessary

# download_kaggle_dataset("datatattle/covid-19-nlp-text-classification", "./data/covid")

set.seed(42)
# Importing Train Data
corona_train <- read.csv("./data/covid/Corona_NLP_train.csv")[, c("OriginalTweet", "Sentiment")]
# corona_train <- sample_frac(corona_train, .1)

# Importing Test Data
corona_test <- read.csv("./data/covid/Corona_NLP_test.csv")[, c("OriginalTweet", "Sentiment")]
# corona_test <- sample_frac(corona_test, .1)

# Recode Sentiment to factors
corona_train <- corona_train %>%
  filter(Sentiment != "Neutral") %>%
  mutate(Sentiment = recode(Sentiment,
                            "Extremely Negative" = "Negative",
                            "Extremely Positive" = "Positive"),
         Sentiment = factor(Sentiment, levels = c("Negative", "Positive")))

corona_test <- corona_test %>%
  filter(Sentiment != "Neutral") %>%
  mutate(Sentiment = recode(Sentiment,
                            "Extremely Negative" = "Negative",
                            "Extremely Positive" = "Positive"),
         Sentiment = factor(Sentiment, levels = c("Negative", "Positive")))

# Preprocessing and tokenization using quanteda
preprocess_text <- function(text_column) {
  tokens <- tokens(text_column, 
                   what = "word", 
                   remove_punct = TRUE, 
                   remove_numbers = TRUE,
                   remove_symbols = TRUE) %>%
    tokens_tolower() %>%
    tokens_remove(stopwords("english")) %>%
    tokens_wordstem()
  
  # Create a Document-Feature Matrix (DFM)
  dfm <- dfm(tokens)
  
  return(dfm)
}

train_dfm <- dfm_trim(preprocess_text(corona_train$OriginalTweet), min_termfreq = 10)
test_dfm <- dfm_match(preprocess_text(corona_test$OriginalTweet), features = featnames(train_dfm))

# Convert DFM to sparse matrix
train_sparse <- as(train_dfm, "dgCMatrix")
test_sparse <- as(test_dfm, "dgCMatrix")

# Convert Sentiment to numeric for glmnet
y_train_numeric <- as.numeric(corona_train$Sentiment) - 1
y_test_numeric <- as.numeric(corona_test$Sentiment) - 1

# Fit the logistic regression model with glmnet
model <- glmnet(
  train_sparse, 
  y_train_numeric, 
  family = "binomial", 
  parallel = TRUE
)

# Predict on training and test datasets
train_predictions_prob <- predict(model, train_sparse, s = min(model$lambda), type = "response")
test_predictions_prob <- predict(model, test_sparse, s = min(model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))

# Calculate F1 score function
calculate_f1 <- function(actual, predicted) {
  confusion <- confusionMatrix(predicted, actual)
  f1 <- confusion$byClass["F1"]
  return(f1)
}

# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(corona_train$Sentiment, train_predictions)
test_f1_score <- calculate_f1(corona_test$Sentiment, test_predictions)

# Calculate the number of features
num_features <- ncol(train_sparse)

coefficients <- coef(model, s = min(model$lambda))
non_zero_features <- sum(coefficients != 0) - 1  # Subtract 1 for the intercept

# Calculate the difference in F1 score between train and test data
f1_difference <- train_f1_score - test_f1_score

# Print the results
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("F1 score on training data:", train_f1_score, "\n")
cat("F1 score on test data:", test_f1_score, "\n")
cat("Difference in F1 score between train and test data:", f1_difference, "\n")



#CORRELATION FEATURE SELECTION

correlations <- apply(train_sparse, 2, function(x) cor(x, y_train_numeric))
max_corr <- max(abs(correlations))

# Step 2: Select features based on a correlation threshold
select_features <- function(threshold) {
  selected_features <- which(abs(correlations) > threshold)
  return(selected_features)
}

# Step 3: Sweep through various thresholds and perform 5-fold cross-validation
thresholds <- seq(0, max_corr, length.out = 10)
cv_results <- data.frame(threshold = numeric(), mean_f1 = numeric())

for (threshold in thresholds) {
  selected_features <- select_features(threshold)
  
  if (length(selected_features) == 0) {
    next
  }
  
  train_sparse_selected <- train_sparse[, selected_features, drop = FALSE]
  
  if (ncol(train_sparse_selected) < 2) {
    next
  }
  
  # Perform 5-fold cross-validation
  train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
  
  f1_scores <- c()
  
  # Define custom F1 score summary function
  f1_summary <- function(data, lev = NULL, model = NULL) {
    confusion <- confusionMatrix(data$pred, data$obs)
    f1 <- confusion$byClass["F1"]
    c(F1 = f1)
  }
  
  for (i in 1:5) {
    folds <- createFolds(y_train_numeric, k = 5, list = TRUE, returnTrain = TRUE)
    f1_fold <- c()
    
    for (j in 1:5) {
      train_index <- folds[[j]]
      test_index <- setdiff(seq_len(nrow(train_sparse_selected)), train_index)
      
      x_train_cv <- train_sparse_selected[train_index, ]
      y_train_cv <- y_train_numeric[train_index]
      x_test_cv <- train_sparse_selected[test_index, ]
      y_test_cv <- y_train_numeric[test_index]
      
      model_cv <- glmnet(x_train_cv, y_train_cv, family = "binomial", parallel = TRUE)
      
      pred_cv <- predict(model_cv, x_test_cv, s = min(model_cv$lambda), type = "response")
      pred_cv <- factor(ifelse(pred_cv > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))
      actual_cv <- factor(ifelse(y_test_cv == 1, "Positive", "Negative"), levels = c("Negative", "Positive"))
      
      f1_fold <- c(f1_fold, calculate_f1(actual_cv, pred_cv))
    }
    
    f1_scores <- c(f1_scores, mean(f1_fold))
  }
  
  cv_results <- rbind(cv_results, data.frame(threshold = threshold, mean_f1 = mean(f1_scores)))
}

# Determine the optimal threshold
optimal_threshold <- cv_results$threshold[which.max(cv_results$mean_f1)]

# Step 4: Use the optimal threshold to train the final model
selected_features <- select_features(optimal_threshold)
train_sparse_selected <- train_sparse[, selected_features]
test_sparse_selected <- test_sparse[, selected_features]

final_model <- glmnet(train_sparse_selected, y_train_numeric, family = "binomial", parallel = TRUE)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, train_sparse_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, test_sparse_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))

# Calculate F1 score function
calculate_f1 <- function(actual, predicted) {
  confusion <- confusionMatrix(predicted, actual)
  f1 <- confusion$byClass["F1"]
  return(f1)
}

# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(corona_train$Sentiment, train_predictions)
test_f1_score <- calculate_f1(corona_test$Sentiment, test_predictions)

# Calculate the number of features
num_features <- ncol(train_sparse_selected) 

coefficients <- coef(final_model, s = min(final_model$lambda))
non_zero_features <- sum(coefficients != 0) - 1  # Subtract 1 for the intercept

# Calculate the difference in F1 score between train and test data
f1_difference <- train_f1_score - test_f1_score

# Print the results
cat("Maximum Absolute Correlation:", max_corr, "\n")
cat("Optimal Threshold:", optimal_threshold, "\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("F1 score on training data:", train_f1_score, "\n")
cat("F1 score on test data:", test_f1_score, "\n")
cat("Difference in F1 score between train and test data:", f1_difference, "\n")


# RECURSIVE FEATURE ELIMINATION


rfe_glmnet <- function(x, y, sizes, fold = 5, parallel = TRUE) {
  results <- data.frame(num_features = integer(), mean_f1 = double())
  control <- rfeControl(functions = caretFuncs, method = "cv", number = fold, verbose = TRUE, allowParallel = parallel)
  
  for (size in sizes) {
    subset <- rfe(x, y, sizes = size, rfeControl = control, method = "glmnet")
    subset_res <- subset$resample
    mean_f1 <- mean(subset_res$F1)
    results <- rbind(results, data.frame(num_features = size, mean_f1 = mean_f1))
  }
  
  best_size <- results$num_features[which.max(results$mean_f1)]
  return(list(best_size = best_size, results = results))
}

# Perform recursive feature elimination
sizes <- seq(10, ncol(train_sparse), length.out = 10)
rfe_results <- rfe_glmnet(train_sparse, y_train_numeric, sizes = sizes)

cat("Optimal Number of Features:", rfe_results$best_size, "\n")


# Select optimal features
optimal_features <- rfe(train_sparse, y_train_numeric, sizes = rfe_results$best_size, rfeControl = rfeControl(functions = caretFuncs, method = "cv", number = 5, allowParallel = TRUE), method = "glmnet")

# Fit the final model using selected features
final_model <- glmnet(optimal_features$fit$finalModel$x, y_train_numeric, family = "binomial", parallel = TRUE)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = optimal_features$fit$finalModel$x, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, newx = test_sparse[, optimal_features$optVariables], s = min(final_model$lambda), type = "response")




# Stop the cluster to clean up resources
stopCluster(cl)

# Unregister the parallel backend (optional, to ensure no residual registration)
registerDoSEQ()
