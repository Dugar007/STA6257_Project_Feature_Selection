## Load necessary libraries
library(dplyr)
library(caret)
library(glmnet)
library(jsonlite)

# Global parameter to force overwriting of existing cache files
FORCE_OVERWRITE <- FALSE

baseline_lambda = 0
baseline_maxit = 500000
baseline_alpha = 0
validation_folds = 10

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
    } else if (cached_value$type == "named_list_numeric") {
      return(as.list(setNames(as.numeric(cached_value$value), cached_value$names)))
    } else {
      stop("Unsupported cached value type.")
    }
  } else {
    cached_value <- NULL
  }
  
  if (is.null(value)) {
    return(cached_value)
  } else {
    # Determine the type of the value and write it to the file
    if (is.numeric(value) && length(value) == 1) {
      value_type <- "numeric"
    } else if (is.integer(value) && length(value) == 1) {
      value_type <- "integer"
    } else if (is.list(value)) {
      if (!is.null(names(value)) && all(sapply(value, is.numeric))) {
        value_type <- "named_list_numeric"
      } else {
        value_type <- "list"
      }
    } else if (is.numeric(value) && length(value) > 1) {
      value_type <- "vector_numeric"
    } else if (is.integer(value) && length(value) > 1) {
      value_type <- "vector_integer"
    } else {
      stop("Unsupported value type. Only numeric, integer, vectors, and list are supported.")
    }
    
    # Prepare the data to be written as JSON
    if (value_type == "named_list_numeric") {
      json_data <- list(type = value_type, value = unname(value), names = names(value))
    } else {
      json_data <- list(type = value_type, value = value)
    }
    
    # Write the value to the file as JSON
    write_json(json_data, file_name)
    
    # Return the value
    return(value)
  }
}

geometric_sizes <- function(start_size, end_size, num_steps) {
  ratio <- (end_size / start_size)^(1 / (num_steps - 1))
  sizes <- start_size * (ratio ^ (0:(num_steps - 1)))
  return(round(sizes))
}

# Calculate performance score function
calculate_binary_performance <- function(actual, predicted) {
  confusion <- confusionMatrix(predicted, actual)
  performance <- confusion$overall['Accuracy']
  return(performance)
}


# Function to return the list of features with non-zero coefficients
nonzero_feature_indicies <- function(model, lambda) {
  coefs <- coef(model, s = lambda)
  nonzero_indices <- which(coefs != 0)
  # Exclude the intercept (first coefficient)
  nonzero_indices <- nonzero_indices[-1]
  return(nonzero_indices)
}


# Function to calculate performance at different thresholds
CFS_binary_logistic <- function(X, y, num_folds, num_bins, min_vars, geometric_spacing) {
  
  y_levels <- levels(y)  # Get the levels from y
  
  # Step 1: Calculate correlations
  correlations <- apply(X, 2, function(x) cor(x, as.numeric(y)))
  abs_correlations <- abs(correlations)
  
  # Step 2: Select features based on a correlation threshold
  select_features <- function(threshold) {
    selected_features <- which(abs_correlations >= threshold)
    return(selected_features)
  }
  
  # Step 3: Sweep through various thresholds and perform 10-fold cross-validation
  if (geometric_spacing){
    sizes <- geometric_sizes(ncol(X), min_vars, num_bins)
  }
  else {
    sizes <- round(seq(from = ncol(X), to = min_vars, length.out = num_bins))
  }
  
  percentiles <- 1 - sizes / ncol(X)
  thresholds <- quantile(abs_correlations, percentiles)
  results <- list()
  
  for (threshold in thresholds) {
    selected_features <- select_features(threshold)
    
    if (length(selected_features) == 0) {
      next
    }
    
    X_selected <- X[, selected_features, drop = FALSE]
    
    if (ncol(X_selected) < 2) {
      next
    }
    
    # Perform k-fold cross-validation
    # train_control <- trainControl(method = "cv", number = num_folds, verboseIter = TRUE)
    
    scores <- c()
    
    for (i in 1:num_folds) {
      folds <- createFolds(y, k = num_folds, list = TRUE, returnTrain = TRUE)
      fold_scores <- c()
      
      for (j in 1:num_folds) {
        train_index <- folds[[j]]
        test_index <- setdiff(seq_len(nrow(X_selected)), train_index)
        
        x_train_cv <- X_selected[train_index, ]
        y_train_cv <- y[train_index]
        x_test_cv <- X_selected[test_index, ]
        y_test_cv <- y[test_index]
        model_cv <- glmnet(x_train_cv, y_train_cv, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda, alpha = baseline_alpha, family = "binomial", parallel = TRUE)
        
        pred_cv <- predict(model_cv, x_test_cv, s = min(model_cv$lambda), type = "response")
        pred_cv <- factor(ifelse(pred_cv > 0.5, y_levels[2], y_levels[1]), levels = y_levels)
        
        fold_scores <- c(fold_scores, calculate_binary_performance(y_test_cv, pred_cv))
      }
      
      scores <- c(scores, mean(fold_scores))
    }
    
    feature_indices <- toJSON(selected_features)
    results[[feature_indices]] <- mean(scores)
  }
  
  return(results)
}


RFE_binary_logistic <- function(X, y, num_folds, num_bins, min_vars, geometric_spacing) {
  results <- list()
  
  y_levels <- levels(y)  # Get the levels from y
  
  if (geometric_spacing){
    sizes <- geometric_sizes(ncol(X), min_vars, num_bins)
  }
  else {
    sizes <- round(seq(from = ncol(X), to = min_vars, length.out = num_bins))
  }
  
  current_features <- seq_len(ncol(X))  # Start with all features
  
  for (size in sizes) {
    cat("Evaluating size:", size, "\n")
    
    # Fit glmnet model to get coefficients
    model <- glmnet(X[, current_features, drop = FALSE], y, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)
    coefs <- as.matrix(coef(model, s = model$lambda.min))
    
    # Get indices of the top 'size' features by their absolute coefficient values
    if (length(current_features) > size) {
      selected_features <- order(abs(coefs[-1, 1]), decreasing = TRUE)[1:size]
      current_features <- current_features[selected_features]
    }
    
    x_selected <- X[, current_features, drop = FALSE]
    
    folds <- createFolds(y, k = num_folds, list = TRUE, returnTrain = TRUE)
    scores <- c()
    
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
      pred_cv <- factor(ifelse(pred_cv_prob > 0.5, y_levels[2], y_levels[1]), levels = y_levels)
      
      scores <- c(scores, calculate_binary_performance(y_test_cv, pred_cv))
    }
    
    mean_score <- mean(scores, na.rm = TRUE)
    feature_indices <- toJSON(current_features)
    results[[feature_indices]] <- mean_score
  }
  
  return(results)
}

subset_performance <- function(named_list) {
  # Extract lengths of the lists and corresponding performance scores
  lengths <- sapply(names(named_list), function(x) {
    list_obj <- fromJSON(x)
    length(list_obj)
  })
  
  performance <- as.numeric(named_list)
  
  # Create a data frame for plotting
  data <- data.frame(lengths = lengths, performance = performance)
  
  return(data)
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

#Importing spam_alt Data
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
spam_alt_data <- read.csv("./data/spambase/spambase.data", header = FALSE, col.names = unique_column_names) %>%
            mutate(flag_spam = factor(flag_spam, levels = c(0, 1)))
spam_alt_data <- sample_frac(spam_alt_data, .1)

# Create a train-test split
trainIndex <- sample(1:nrow(spam_alt_data), size = 0.8 * nrow(spam_alt_data))

# Split the data
spam_alt_train <- spam_alt_data[trainIndex, ]
spam_alt_test <- spam_alt_data[-trainIndex, ]

X_train_raw <- spam_alt_train %>% select(-flag_spam)
X_test_raw  <- spam_alt_test %>% select(-flag_spam)

y_train  <- spam_alt_train$flag_spam
y_test <- spam_alt_test$flag_spam

levels = levels(y_train)

# scale the data, calculate all two-way interactions, and drop the intercept
preprocess_params <- preProcess(X_train_raw, method = c("center", "scale"))
X_train <- model.matrix( ~ .^2, predict(preprocess_params, X_train_raw))[, -1]
X_test <- model.matrix( ~ .^2, predict(preprocess_params, X_test_raw))[, -1]


baseline_model <- glmnet(
  X_train, 
  y_train, 
  family = "binomial", 
  alpha = baseline_alpha,
  lambda.min.ratio = baseline_lambda,
  maxit = baseline_maxit,
  parallel = TRUE
)

# Predict on training and test datasets
train_predictions_prob <- predict(baseline_model, X_train, s = min(baseline_model$lambda), type = "response")
test_predictions_prob <- predict(baseline_model, X_test, s = min(baseline_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)

# Calculate accuracy scores for training and test datasets
train_accuracy_score <- calculate_binary_performance(spam_alt_train$flag_spam, train_predictions)
test_accuracy_score <- calculate_binary_performance(spam_alt_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train)
non_zero_features <- length(nonzero_feature_indicies(baseline_model, min(baseline_model$lambda)))

# Calculate the difference in accuracy score between train and test data
accuracy_difference <- train_accuracy_score - test_accuracy_score

# Print the results
cat("BASELINE RESULTS\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("accuracy score on training data:", train_accuracy_score, "\n")
cat("accuracy score on test data:", test_accuracy_score, "\n")
cat("Difference in accuracy score between train and test data:", accuracy_difference, "\n")



#CORRELATION FEATURE SELECTION
cfs_feature_subsets <- cache_parameter('spam_alt_cfs_feature_subsets')
if (is.null(cfs_feature_subsets)) {
  cfs_feature_subsets <- cache_parameter('spam_alt_cfs_feature_subsets', CFS_binary_logistic(X_train, y_train, num_folds=validation_folds, num_bins = 20, min_vars = 50, geometric_spacing = TRUE))
}
optimal_subset <- fromJSON(names(cfs_feature_subsets)[which.max(cfs_feature_subsets)])

# Step 4: Use the optimal threshold to train the final model
X_train_selected <- X_train[, optimal_subset]
X_test_selected <- X_test[, optimal_subset]

final_model <- glmnet(X_train_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda, parallel = TRUE)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, X_train_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, X_test_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)

# Calculate accuracy scores for training and test datasets
train_accuracy_score <- calculate_binary_performance(spam_alt_train$flag_spam, train_predictions)
test_accuracy_score <- calculate_binary_performance(spam_alt_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train_selected) 
non_zero_features <- length(nonzero_feature_indicies(final_model, min(final_model$lambda)))

# Calculate the difference in accuracy score between train and test data
accuracy_difference <- train_accuracy_score - test_accuracy_score

# Print the results
cat("CFS RESULTS\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("accuracy score on training data:", train_accuracy_score, "\n")
cat("accuracy score on test data:", test_accuracy_score, "\n")
cat("Difference in accuracy score between train and test data:", accuracy_difference, "\n")

plot_data <- subset_performance(cfs_feature_subsets)
ggplot(plot_data, aes(x = lengths, y = performance)) +
  geom_point() +
  geom_line() +
  labs(title = "Performance vs Number of Features",
       x = "Number of Features",
       y = "Performance Score")

# RECURSIVE FEATURE ELIMINATION

feature_subsets <- cache_parameter('spam_alt_rfe_feature_subsets')
if (is.null(feature_subsets)) {
  feature_subsets <- cache_parameter('spam_alt_rfe_feature_subsets', RFE_binary_logistic(X_train, y_train, num_folds=validation_folds, num_bins = 20, min_vars = 50, geometric_spacing = TRUE))
}
optimal_subset <- fromJSON(names(feature_subsets)[which.max(feature_subsets)])

# Fit the final model using selected features
X_train_selected = X_train[, optimal_subset, drop=FALSE]
X_test_selected = X_test[, optimal_subset, drop=FALSE]
final_model <- glmnet(X_train_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = X_train_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, newx = X_test_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)

# Calculate accuracy scores for training and test datasets
train_accuracy_score <- calculate_binary_performance(spam_alt_train$flag_spam, train_predictions)
test_accuracy_score <- calculate_binary_performance(spam_alt_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train_selected) 
non_zero_features <- length(nonzero_feature_indicies(final_model, min(final_model$lambda)))

# Calculate the difference in accuracy score between train and test data
accuracy_difference <- train_accuracy_score - test_accuracy_score

cat("RFE RESULTS\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("accuracy score on training data:", train_accuracy_score, "\n")
cat("accuracy score on test data:", test_accuracy_score, "\n")
cat("Difference in accuracy score between train and test data:", accuracy_difference, "\n")

plot_data <- subset_performance(feature_subsets)
ggplot(plot_data, aes(x = lengths, y = performance)) +
  geom_point() +
  geom_line() +
  labs(title = "Performance vs Number of Features",
       x = "Number of Features",
       y = "Performance Score")


#LASSO REGRESSION
lambdas <- cache_parameter('spam_alt_lasso_lambdas')
feature_subsets_sizes <- cache_parameter('spam_alt_lasso_feature_subsets_sizes')
if (is.null(lambdas)) {
  cv_lasso <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, maxit = baseline_maxit, nfolds = validation_folds)
  lambda_values <- cv_lasso$lambda
  non_zero_features <- sapply(lambda_values, nonzero_feature_indicies, model = cv_lasso)
  non_zero_feature_sizes <- sapply(non_zero_features, length)
  performance <- cv_lasso$cvm
  lambdas <- cache_parameter('spam_alt_lasso_lambdas', setNames(as.list(performance), as.character(lambda_values)))
  feature_subsets_sizes <- cache_parameter('spam_alt_lasso_feature_subsets_sizes', setNames(as.list(performance), as.character(non_zero_feature_sizes)))
}
optimal_lambda <- as.double(names(lambdas)[which.min(lambdas)])


# Fit the final Lasso model using the best lambda
final_model <- glmnet(X_train, y_train, family = "binomial", maxit = baseline_maxit, alpha = 1, lambda.min.ratio = optimal_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = X_train, s = optimal_lambda, type = "response")
test_predictions_prob <- predict(final_model, newx = X_test, s = optimal_lambda, type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)

# Calculate accuracy scores for training and test datasets
train_accuracy_score <- calculate_binary_performance(spam_alt_train$flag_spam, train_predictions)
test_accuracy_score <- calculate_binary_performance(spam_alt_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train) 
non_zero_features <- length(nonzero_feature_indicies(final_model, min(final_model$lambda)))

# Calculate the difference in accuracy score between train and test data
accuracy_difference <- train_accuracy_score - test_accuracy_score

cat("LASSO RESULTS\n")
cat("Best lambda from cross-validation: ", optimal_lambda, "\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("accuracy score on training data:", train_accuracy_score, "\n")
cat("accuracy score on test data:", test_accuracy_score, "\n")
cat("Difference in accuracy score between train and test data:", accuracy_difference, "\n")


plot_data <- data.frame(lengths = as.integer(names(feature_subsets_sizes)), performance = as.numeric(feature_subsets_sizes))

ggplot(plot_data, aes(x = lengths, y = performance)) +
  geom_point() +
  geom_line() +
  labs(title = "Performance vs Number of Features",
       x = "Number of Features",
       y = "Performance Score")


#CFS + RFE REGRESSION

starting_features = fromJSON(names(cfs_feature_subsets)[3])
feature_subsets <- cache_parameter('spam_alt_cfs_rfe_feature_subsets')
if (is.null(feature_subsets)) {
  X_train_cfs_selected <- X_train[, starting_features, drop = FALSE]
  feature_subsets <- cache_parameter('spam_alt_cfs_rfe_feature_subsets', RFE_binary_logistic(X_train_cfs_selected, y_train, num_folds=validation_folds, num_bins = 20, min_vars = 50, geometric_spacing = FALSE))
}
optimal_subset <- fromJSON(names(feature_subsets)[which.max(feature_subsets)])

# Fit the final model using selected features
X_train_selected = X_train[, optimal_subset, drop=FALSE]
X_test_selected = X_test[, optimal_subset, drop=FALSE]
final_model <- glmnet(X_train_selected, y_train, family = "binomial", alpha = baseline_alpha, maxit = baseline_maxit, lambda.min.ratio = baseline_lambda)

# Predict on training and test datasets
train_predictions_prob <- predict(final_model, newx = X_train_selected, s = min(final_model$lambda), type = "response")
test_predictions_prob <- predict(final_model, newx = X_test_selected, s = min(final_model$lambda), type = "response")

# Convert probabilities to class labels (0 or 1)
train_predictions <- factor(ifelse(train_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)
test_predictions <- factor(ifelse(test_predictions_prob > 0.5, levels[2], levels[1]), levels = levels)

# Calculate accuracy scores for training and test datasets
train_accuracy_score <- calculate_binary_performance(spam_alt_train$flag_spam, train_predictions)
test_accuracy_score <- calculate_binary_performance(spam_alt_test$flag_spam, test_predictions)

# Calculate the number of features
num_features <- ncol(X_train_selected) 
non_zero_features <- length(nonzero_feature_indicies(final_model, min(final_model$lambda)))

# Calculate the difference in accuracy score between train and test data
accuracy_difference <- train_accuracy_score - test_accuracy_score

cat("CFS + RFE RESULTS\n")
cat("Number of features:", num_features, "\n")
cat("Number of non-zero features:", non_zero_features, "\n")
cat("accuracy score on training data:", train_accuracy_score, "\n")
cat("accuracy score on test data:", test_accuracy_score, "\n")
cat("Difference in accuracy score between train and test data:", accuracy_difference, "\n")

plot_data <- subset_performance(feature_subsets)
ggplot(plot_data, aes(x = lengths, y = performance)) +
  geom_point() +
  geom_line() +
  labs(title = "Performance vs Number of Features",
       x = "Number of Features",
       y = "Performance Score")
