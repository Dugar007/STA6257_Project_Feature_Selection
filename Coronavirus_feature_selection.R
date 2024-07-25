## Load necessary libraries
library(dplyr)
library(quanteda)
library(caret)
library(glmnet)
library(doParallel)
library(jsonlite)

# Global parameter to force overwriting of existing cache files
FORCE_OVERWRITE <- TRUE

numCores <- max(c(1, detectCores() - 2))
cl <- makeCluster(numCores)
registerDoParallel(cl)

# Function to cache parameters
cache_parameter <- function(name, value = NULL, path = "cache/", prefix = "param_") {
  # Ensure the cache directory exists
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  
  # Construct the full file name
  file_name <- paste0(path, prefix, name, ".json")
  
  # Check for file existence
  if (file.exists(file_name) && !FORCE_OVERWRITE) {
    # Load and return the value from the file
    cached_value <- fromJSON(file_name)
    # Cast to the appropriate type
    if (cached_value$type == "numeric") {
      return(as.numeric(cached_value$value))
    } else if (cached_value$type == "integer") {
      return(as.integer(cached_value$value))
    } else if (cached_value$type == "list") {
      return(as.list(cached_value$value))
    } else if (cached_value$type == "vector_numeric") {
      return(as.numeric(cached_value$value))
    } else if (cached_value$type == "vector_integer") {
      return(as.integer(cached_value$value))
    } else {
      stop("Unsupported cached value type.")
    }
  } else {
    cached_value <- NULL
  }
  
  if (is.null(value)) {
    return(cached_value)
  } else
  {
    # Determine the type of the value and write it to the file
    if (is.numeric(value) && length(value) == 1) {
      value_type <- "numeric"
    } else if (is.integer(value) && length(value) == 1) {
      value_type <- "integer"
    } else if (is.list(value)) {
      value_type <- "list"
    } else if (is.numeric(value) && length(value) > 1) {
      value_type <- "vector_numeric"
    } else if (is.integer(value) && length(value) > 1) {
      value_type <- "vector_integer"
    } else {
      stop("Unsupported value type. Only numeric, integer, vectors, and list are supported.")
    }
    # Write the value to the file as JSON
    write_json(list(type = value_type, value = value), file_name)
    
    # Return the value
    return(value)
  }
}

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
y_train <- corona_train$Sentiment
y_test <- corona_test$Sentiment

# Fit the logistic regression model with glmnet
baseline_lambda = 0
baseline_maxit = 500000
baseline_alpha = 0
validation_folds = 10

baseline_model <- glmnet(
  train_sparse, 
  y_train, 
  family = "binomial", 
  alpha = baseline_alpha,
  lambda.min.ratio = baseline_lambda,
  maxit = baseline_maxit,
  parallel = TRUE
)

# Predict on training and test datasets
train_predictions_prob <- predict(baseline_model, train_sparse, s = min(baseline_model$lambda), type = "response")
test_predictions_prob <- predict(baseline_model, test_sparse, s = min(baseline_model$lambda), type = "response")

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

coefficients <- coef(baseline_model, s = min(baseline_model$lambda))
non_zero_features <- sum(coefficients != 0) - 1  # Subtract 1 for the intercept

# Calculate the difference in F1 score between train and test data
f1_difference <- train_f1_score - test_f1_score

# Print the results
cat("BASELINE RESULTS\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("F1 score on training data:", train_f1_score, "\n")
cat("F1 score on test data:", test_f1_score, "\n")
cat("Difference in F1 score between train and test data:", f1_difference, "\n")



#CORRELATION FEATURE SELECTION

correlations <- apply(train_sparse, 2, function(x) cor(x, as.numeric(y_train)))
abs_correlations <- abs(correlations)

# Step 2: Select features based on a correlation threshold
select_features <- function(threshold) {
  selected_features <- which(abs_correlations > threshold)
  return(selected_features)
}

# Step 3: Sweep through various thresholds and perform 10-fold cross-validation
percentiles <- seq(0, 1, length.out = 20)
thresholds <- quantile(abs_correlations, percentiles)
cv_results <- data.frame(threshold = numeric(), mean_f1 = numeric())

optimal_threshold <- cache_parameter('covid_cfs_optimal_threshold')
if (is.null(optimal_threshold)) {
  for (threshold in thresholds) {
    selected_features <- select_features(threshold)
    
    if (length(selected_features) == 0) {
      next
    }
    
    train_sparse_selected <- train_sparse[, selected_features, drop = FALSE]
    
    if (ncol(train_sparse_selected) < 2) {
      next
    }
    
    # Perform k-fold cross-validation
    train_control <- trainControl(method = "cv", number = validation_folds, verboseIter = TRUE)
    
    f1_scores <- c()
    
    # Define custom F1 score summary function
    f1_summary <- function(data, lev = NULL, model = NULL) {
      confusion <- confusionMatrix(data$pred, data$obs)
      f1 <- confusion$byClass["F1"]
      c(F1 = f1)
    }
    
    for (i in 1:validation_folds) {
      folds <- createFolds(y_train, k = validation_folds, list = TRUE, returnTrain = TRUE)
      f1_fold <- c()
      
      for (j in 1:validation_folds) {
        train_index <- folds[[j]]
        test_index <- setdiff(seq_len(nrow(train_sparse_selected)), train_index)
        
        x_train_cv <- train_sparse_selected[train_index, ]
        y_train_cv <- y_train[train_index]
        x_test_cv <- train_sparse_selected[test_index, ]
        y_test_cv <- y_train[test_index]
        
        model_cv <- glmnet(x_train_cv, y_train_cv, alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda, family = "binomial", parallel = TRUE)
        
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
  optimal_threshold <- cache_parameter('covid_cfs_optimal_threshold', cv_results$threshold[which.max(cv_results$mean_f1)])
}


# Step 4: Use the optimal threshold to train the final model
selected_features <- select_features(optimal_threshold)
train_sparse_selected <- train_sparse[, selected_features]
test_sparse_selected <- test_sparse[, selected_features]

final_model <- glmnet(train_sparse_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda, parallel = TRUE)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, train_sparse_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, test_sparse_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))

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
cat("CFS RESULTS\n")
cat("Optimal Threshold:", optimal_threshold, "\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("F1 score on training data:", train_f1_score, "\n")
cat("F1 score on test data:", test_f1_score, "\n")
cat("Difference in F1 score between train and test data:", f1_difference, "\n")


# RECURSIVE FEATURE ELIMINATION
# Define custom RFE function
custom_rfe <- function(x, y, sizes, fold = validation_folds, parallel = TRUE) {
  results <- data.frame(num_features = integer(), mean_f1 = double())
  
  current_features <- seq_len(ncol(x))  # Start with all features
  best_features <- current_features
  
  for (size in sizes) {
    cat("Evaluating size:", size, "\n")
    
    # Fit glmnet model to get coefficients
    model <- glmnet(x[, current_features, drop = FALSE], y, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)
    coefs <- as.matrix(coef(model, s = model$lambda.min))
    
    # Get indices of the top 'size' features by their absolute coefficient values
    if (length(current_features) > size) {
      selected_features <- order(abs(coefs[-1, 1]), decreasing = TRUE)[1:size]
      current_features <- current_features[selected_features]
    }
    
    x_selected <- x[, current_features, drop = FALSE]
    
    folds <- createFolds(y, k = fold, list = TRUE, returnTrain = TRUE)
    f1_scores <- c()
    
    for (fold_idx in seq_along(folds)) {
      train_idx <- folds[[fold_idx]]
      test_idx <- setdiff(seq_len(nrow(x_selected)), train_idx)
      
      x_train_cv <- x_selected[train_idx, ]
      y_train_cv <- y[train_idx]
      x_test_cv <- x_selected[test_idx, ]
      y_test_cv <- y[test_idx]
      
      model_cv <- glmnet(x_train_cv, y_train_cv, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)
      
      if (length(model_cv$lambda) == 0) {
        next
      }
      
      pred_cv_prob <- predict(model_cv, x_test_cv, s = min(model_cv$lambda), type = "response")
      pred_cv <- factor(ifelse(pred_cv_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))
      
      f1 <- calculate_f1(y_test_cv, pred_cv)
      f1_scores <- c(f1_scores, f1)
    }
    
    mean_f1 <- mean(f1_scores, na.rm = TRUE)
    results <- rbind(results, data.frame(num_features = size, mean_f1 = mean_f1))
    
    if (mean_f1 == max(results$mean_f1, na.rm = TRUE)) {
      best_features <- current_features
    }
  }
  
  best_size <- results$num_features[which.max(results$mean_f1)]
  return(list(best_size = best_size, best_features = best_features, results = results))
}


# Generate sizes in geometric progression
generate_sizes <- function(start_size, end_size, num_steps) {
  ratio <- (end_size / start_size)^(1 / (num_steps - 1))
  sizes <- start_size * (ratio ^ (0:(num_steps - 1)))
  return(round(sizes))
}

# Perform recursive feature elimination
sizes <- generate_sizes(ncol(X_train), 100, 20)

best_features <- cache_parameter('covid_rfe_best_features')
if (is.null(best_features)) {
  rfe_results <- custom_rfe(train_sparse, y_train, sizes = sizes)
  best_features <- cache_parameter('covid_rfe_best_features', rfe_results$best_features)
}

# Fit the final model using selected features
train_sparse_selected = train_sparse[, best_features, drop=FALSE]
test_sparse_selected = test_sparse[, best_features, drop=FALSE]
final_model <- glmnet(train_sparse_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = train_sparse_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, newx = test_sparse_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))


# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(corona_train$Sentiment, train_predictions)
test_f1_score <- calculate_f1(corona_test$Sentiment, test_predictions)

# Calculate the number of features
num_features <- ncol(train_sparse_selected) 

coefficients <- coef(final_model, s = min(final_model$lambda))
non_zero_features <- sum(coefficients != 0) - 1  # Subtract 1 for the intercept

# Calculate the difference in F1 score between train and test data
f1_difference <- train_f1_score - test_f1_score

cat("RFE RESULTS\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("F1 score on training data:", train_f1_score, "\n")
cat("F1 score on test data:", test_f1_score, "\n")
cat("Difference in F1 score between train and test data:", f1_difference, "\n")

#LASSO REGRESSION

best_lambda <- cache_parameter('covid_lasso_best_lambda')
if (is.null(best_lambda)) {
  cv_lasso <- cv.glmnet(train_sparse, y_train, family = "binomial", alpha = 1, maxit = baseline_maxit, nfolds = validation_folds)
  best_lambda <- cache_parameter('covid_lasso_best_lambda', cv_lasso$lambda.min)
}

# Fit the final Lasso model using the best lambda
final_model <- glmnet(train_sparse, y_train, family = "binomial", maxit = baseline_maxit, alpha = 1, lambda = best_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = train_sparse, s = best_lambda, type = "response")
test_predictions_prob <- predict(final_model, newx = test_sparse, s = best_lambda, type = "response")

# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(corona_train$Sentiment, train_predictions)
test_f1_score <- calculate_f1(corona_test$Sentiment, test_predictions)

# Calculate the number of features
num_features <- ncol(train_sparse) 

coefficients <- coef(final_model, s = best_lambda)
non_zero_features <- sum(coefficients != 0) - 1  # Subtract 1 for the intercept

# Calculate the difference in F1 score between train and test data
f1_difference <- train_f1_score - test_f1_score

cat("LASSO RESULTS\n")
cat("Best lambda from cross-validation: ", best_lambda, "\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("F1 score on training data:", train_f1_score, "\n")
cat("F1 score on test data:", test_f1_score, "\n")
cat("Difference in F1 score between train and test data:", f1_difference, "\n")


#CFS + RFE REGRESSION
correlation_threshold = thresholds[5]
best_features <- cache_parameter('covid_cfs_rfe_best_features')
if (is.null(best_features)) {
  cfs_selected_features <- select_features(correlation_threshold)
  
  train_sparse_cfs_selected <- train_sparse[, cfs_selected_features, drop = FALSE]
  
  # Perform recursive feature elimination
  sizes <- generate_sizes(ncol(train_sparse_cfs_selected), 100, 20)
  rfe_results <- custom_rfe(train_sparse_cfs_selected, y_train, sizes = sizes)
  
  best_features <- cache_parameter('covid_cfs_rfe_best_features', rfe_results$best_features)
}

# Fit the final model using selected features
train_sparse_selected = train_sparse[, best_features, drop=FALSE]
test_sparse_selected = test_sparse[, best_features, drop=FALSE]
final_model <- glmnet(train_sparse_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = train_sparse_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, newx = test_sparse_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, "Positive", "Negative"), levels = c("Negative", "Positive"))


# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(corona_train$Sentiment, train_predictions)
test_f1_score <- calculate_f1(corona_test$Sentiment, test_predictions)

# Calculate the number of features
num_features <- ncol(train_sparse_selected) 

coefficients <- coef(final_model, s = min(final_model$lambda))
non_zero_features <- sum(coefficients != 0) - 1  # Subtract 1 for the intercept

# Calculate the difference in F1 score between train and test data
f1_difference <- train_f1_score - test_f1_score

cat("CFS + RFE RESULTS\n")
cat("Correlation Threshold:", correlation_threshold, "\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("F1 score on training data:", train_f1_score, "\n")
cat("F1 score on test data:", test_f1_score, "\n")
cat("Difference in F1 score between train and test data:", f1_difference, "\n")



# Stop the cluster to clean up resources
stopCluster(cl)

# Unregister the parallel backend (optional, to ensure no residual registration)
registerDoSEQ()
