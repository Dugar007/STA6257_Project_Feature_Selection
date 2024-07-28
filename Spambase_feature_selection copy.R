## Load necessary libraries
library(dplyr)
library(caret)
library(glmnet)
library(doParallel)
library(jsonlite)

# Global parameter to force overwriting of existing cache files
FORCE_OVERWRITE <- TRUE

# numCores <- max(c(1, detectCores() - 2))
# cl <- makeCluster(numCores)
# registerDoParallel(cl)

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

set.seed(42)

# Step #1 Data Ingesting into R 
# download the datasets
download_and_extract_zip <- function(url, dest_dir) {
  # Ensure the destination directory exists
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE)
  }
  
  # Create a temporary file to hold the downloaded zip file
  temp_zip <- tempfile(fileext = ".zip")
  
  # Download the zip file
  download.file(url, temp_zip, mode = "wb")
  
  # Extract the contents of the zip file
  unzip(temp_zip, exdir = dest_dir)
  
  # Remove the temporary zip file
  unlink(temp_zip)
}

# download_and_extract_zip("https://archive.ics.uci.edu/static/public/94/spambase.zip", "./data/spambase")

#Importing Spam Data
last_57_lines <- tail(readLines("./data/spambase/spambase.names"), 57)
# Function to extract and sanitize column names
extract_column_names <- function(line) {
  # Extract the name before the colon
  name <- strsplit(line, ":")[[1]][1]
  # Sanitize the name by replacing special characters with underscores
  sanitized_name <- gsub("[^a-zA-Z0-9_]", "_", name)
  return(sanitized_name)
}

# Apply the function to each line to get the column names
raw_column_names <- c(sapply(last_57_lines, extract_column_names), 'flag_spam')


# Ensure unique column names
unique_column_names <- make.unique(raw_column_names)
spam_data <- read.csv("./data/spambase/spambase.data", header = FALSE, col.names = unique_column_names) %>%
            mutate(flag_spam = factor(flag_spam, levels = c(0, 1)))
spam_data <- sample_frac(spam_data, .1)

# Create a train-test split
trainIndex <- sample(1:nrow(spam_data), size = 0.8 * nrow(spam_data))

# Split the data
spam_train <- spam_data[trainIndex, ]
spam_test <- spam_data[-trainIndex, ]

X_train_raw <- spam_train %>% select(-flag_spam)
X_test_raw  <- spam_test %>% select(-flag_spam)

y_train  <- spam_train$flag_spam
y_test <- spam_test$flag_spam

# scale the data, calculate all two-way interactions, and drop the intercept
preprocess_params <- preProcess(X_train_raw, method = c("center", "scale"))
X_train <- model.matrix( ~ .^2, predict(preprocess_params, X_train_raw))[, -1]
X_test <- model.matrix( ~ .^2, predict(preprocess_params, X_test_raw))[, -1]

# Fit the logistic regression model with glmnet
baseline_lambda = 0
baseline_maxit = 500000
baseline_alpha = 0
validation_folds = 10

baseline_model <- glmnet(
  X_train, 
  y_train, 
  family = "binomial", 
  alpha = baseline_alpha,
  lambda.min.ratio = baseline_lambda,
  maxit = baseline_maxit,
  parallel = TRUE
)

plot(baseline_model)

# Predict on training and test datasets
train_predictions_prob <- predict(baseline_model, X_train, s = min(baseline_model$lambda), type = "response")
test_predictions_prob <- predict(baseline_model, X_test, s = min(baseline_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, 1, 0), levels = c(0, 1))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, 1, 0), levels = c(0, 1))

# Calculate F1 score function
calculate_f1 <- function(actual, predicted) {
  confusion <- confusionMatrix(predicted, actual)
  f1 <- confusion$byClass["F1"]
  return(f1)
}


# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(spam_train$flag_spam, train_predictions)
test_f1_score <- calculate_f1(spam_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train)

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

correlations <- apply(X_train, 2, function(x) cor(x, as.numeric(y_train)))
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
optimal_threshold <- cache_parameter('Spambase_cfs_optimal_threshold')
if (is.null(optimal_threshold)) {
  for (threshold in thresholds) {
    selected_features <- select_features(threshold)
    
    if (length(selected_features) == 0) {
      next
    }
    
    X_train_selected <- X_train[, selected_features, drop = FALSE]
    
    if (ncol(X_train_selected) < 2) {
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
        test_index <- setdiff(seq_len(nrow(X_train_selected)), train_index)
        
        x_train_cv <- X_train_selected[train_index, ]
        y_train_cv <- y_train[train_index]
        x_test_cv <- X_train_selected[test_index, ]
        y_test_cv <- y_train[test_index]
        
        model_cv <- glmnet(x_train_cv, y_train_cv, alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda, family = "binomial", parallel = TRUE)
        
        pred_cv <- predict(model_cv, x_test_cv, s = min(model_cv$lambda), type = "response")
        pred_cv <- factor(ifelse(pred_cv > 0.5, 0, 1), levels = c(0, 1))
        actual_cv <- factor(ifelse(y_test_cv == 1, 1, 0), levels = c(0, 1))
        
        f1_fold <- c(f1_fold, calculate_f1(actual_cv, pred_cv))
      }
      
      f1_scores <- c(f1_scores, mean(f1_fold))
    }
    
    cv_results <- rbind(cv_results, data.frame(threshold = threshold, mean_f1 = mean(f1_scores)))
  }
  
  # Determine the optimal threshold
  optimal_threshold <-  cache_parameter('Spambase_cfs_optimal_threshold', cv_results$threshold[which.max(cv_results$mean_f1)]) 
}



# Step 4: Use the optimal threshold to train the final model
selected_features <- select_features(optimal_threshold)
X_train_selected <- X_train[, selected_features]
X_test_selected <- X_test[, selected_features]

final_model <- glmnet(X_train_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda, parallel = TRUE)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, X_train_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, X_test_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, 1, 0), levels = c(0, 1))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, 1, 0), levels = c(0, 1))

# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(spam_train$flag_spam, train_predictions)
test_f1_score <- calculate_f1(spam_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train_selected) 

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
      pred_cv <- factor(ifelse(pred_cv_prob > 0.5, 1, 0), levels = c(0, 1))
      
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

best_features <- cache_parameter('Spambase_rfe_best_features')
if (is.null(best_features)) {
  rfe_results <- custom_rfe(X_train, y_train, sizes = sizes)
  best_features <- cache_parameter('Spambase_rfe_best_features', rfe_results$best_features)
}

# Fit the final model using selected features
X_train_selected = X_train[, best_features, drop=FALSE]
X_test_selected = X_test[, best_features, drop=FALSE]
final_model <- glmnet(X_train_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = X_train_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, newx = X_test_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, 1, 0), levels = c(0, 1))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, 1, 0), levels = c(0, 1))


# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(spam_train$flag_spam, train_predictions)
test_f1_score <- calculate_f1(spam_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train_selected) 

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
best_lambda <- cache_parameter('Spambase_lasso_best_lambda')
if (is.null(best_lambda)) {
  cv_lasso <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, maxit = baseline_maxit, nfolds = validation_folds)
  best_lambda <- cache_parameter('Spambase_lasso_best_lambda', cv_lasso$lambda.min)
}

# Fit the final Lasso model using the best lambda
final_model <- glmnet(X_train, y_train, family = "binomial", maxit = baseline_maxit, alpha = 1, lambda = best_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = X_train, s = best_lambda, type = "response")
test_predictions_prob <- predict(final_model, newx = X_test, s = best_lambda, type = "response")

# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(spam_train$flag_spam, train_predictions)
test_f1_score <- calculate_f1(spam_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train) 

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
best_features <- cache_parameter('Spambase_cfs_rfe_best_features')
if (is.null(best_features)) {
  cfs_selected_features <- select_features(correlation_threshold)
  
  X_train_cfs_selected <- X_train[, cfs_selected_features, drop = FALSE]
  
  # Perform recursive feature elimination
  sizes <- generate_sizes(ncol(X_train_cfs_selected), 100, 20)
  rfe_results <- custom_rfe(X_train_cfs_selected, y_train, sizes = sizes)
  
  best_features <- cache_parameter('Spambase_cfs_rfe_best_features', rfe_results$best_features)
}


# Fit the final model using selected features
X_train_selected = X_train[, best_features, drop=FALSE]
X_test_selected = X_test[, best_features, drop=FALSE]
final_model <- glmnet(X_train_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = X_train_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, newx = X_test_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, 1, 0), levels = c(0, 1))
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, 1, 0), levels = c(0, 1))


# Calculate F1 scores for training and test datasets
train_f1_score <- calculate_f1(spam_train$flag_spam, train_predictions)
test_f1_score <- calculate_f1(spam_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train_selected) 

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

# # Stop the cluster to clean up resources
# stopCluster(cl)
# 
# # Unregister the parallel backend (optional, to ensure no residual registration)
# registerDoSEQ()
