
# Advanced Statistical Modeling - Feature Selection Group 

# Birds' Songs Numeric Dataset
# The data has been downloaded from https://www.kaggle.com/datasets/fleanend/birds-songs-numeric-dataset
# Unzipped the files and renamed train to bird_train and test to bird_test files respectively.

# Step #1 Data Ingesting into R  

#download datasets, if necessary

# download_kaggle_dataset <- function(dataset, path) {
#   # Check if the kaggle command is available
#   if (system("which kaggle", intern = TRUE) == "") {
#     stop("Kaggle API is not installed or not in PATH. Please install it first.")
#   }
#   
#   # Ensure the destination directory exists
#   if (!dir.exists(path)) {
#     dir.create(path, recursive = TRUE)
#   }
#   
#   # Construct the download command
#   command <- sprintf("kaggle datasets download -d %s -p %s", dataset, path)
#   
#   # Execute the command
#   system(command)
#   
#   # Unzip the downloaded file
#   zipfile <- list.files(path, pattern = "*.zip", full.names = TRUE)
#   if (length(zipfile) > 0) {
#     unzip(zipfile, exdir = path)
#     file.remove(zipfile)
#   }
# }
# 
# download_kaggle_dataset("fleanend/birds-songs-numeric-dataset", "./data/birds")

#Importing Train Data

#bird_train <- read.csv("./data/birds/train.csv")
bird_train <-read.csv("C:/IMP_LEARNING/Masters/Advanced Statistical Modeling/R_Coding/bird_train.csv")
dim(bird_train)
head(bird_train,3)

#Importing Test Data 

#bird_test <- read.csv("./data/birds/test.csv")
bird_test <- read.csv("C:/IMP_LEARNING/Masters/Advanced Statistical Modeling/R_Coding/bird_test.csv")

dim(bird_test)
head(bird_test,3)

#Data Cleaning and Analysis
column_names <- names(bird_train)
column_names
unique(bird_train$species)
unique(bird_train$genus)
unique(bird_train$id)


# Load the ggplot2 package
library(ggplot2)
# Plotting in graph to see the distribution and find outlier 
ggplot(bird_train, aes(x = species)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Species", y = "Count", title = "Distribution")

### Generic Data Curation for all the methods ###

library(dplyr)

dim(bird_train)
dim(bird_test)

# Convert the 'species' column to a factor
bird_train$species <- as.factor(bird_train$species)
bird_test$species <- as.factor(bird_test$species)

# Convert the factor to numeric
bird_train$species_numeric <- as.numeric(bird_train$species)
bird_test$species_numeric <- as.numeric(bird_test$species)
dim(bird_train)
dim(bird_test)


# Print the first few rows to check the conversion
head(bird_train)

# Select only numeric columns
numeric_bird_train <- bird_train %>% select(where(is.numeric))
numeric_bird_test <- bird_test %>% select(where(is.numeric))
dim(numeric_bird_train)
dim(numeric_bird_test)

####################################################################
## Filter Methods Filter methods select features based on statistical measures ##

# Load necessary packages
#install.packages("corrr")
#install.packages("caret")
library(corrr)
library(caret)

# Remove highly correlated features from the training set
correlationMatrix <- cor(numeric_bird_train[, -ncol(numeric_bird_train)])

# Find highly correlated features (correlation > 0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.75)

numeric_bird_train_filtered <- numeric_bird_train[, -highlyCorrelated]

# Remove highly correlated features from the testing set
numeric_bird_test_filtered <- numeric_bird_test[, -highlyCorrelated]
print(highlyCorrelated)

# Train a model using the filtered training set
model <- train(species_numeric ~ ., data = numeric_bird_train_filtered, method = "glm")

# Make predictions on the filtered testing set
predictions <- predict(model, newdata = numeric_bird_test_filtered)

# Convert predictions and actual values to factors
predictions <- factor(predictions)
actual <- factor(numeric_bird_test_filtered$species_numeric)##--$species_numeric $species_numeric

# Ensure they have the same levels
levels(predictions) <- levels(actual)

# Convert predictions and actual values to factors with the same levels
predictions <- factor(predictions, levels = unique(c(predictions, numeric_bird_test_filtered$species_numeric)))
actual <- factor(numeric_bird_test_filtered$species_numeric, levels = unique(c(predictions, numeric_bird_test_filtered$species_numeric)))
# Ensure they have the same levels
levels(predictions) <- levels(actual)
# Create the confusion matrix
confusionMatrix(predictions, actual)


####################################################################
## Wrapper methods evaluate feature subsets based on the performance of a specific machine learning model ##

# 

# Load the dataset
#install.packages("randomForest")
#data(numeric_bird_train)
library(caret)
library(randomForest)
# Ensure factor levels match

dim(numeric_bird_test)
dim(numeric_bird_train)
column_names <- names(numeric_bird_train)
column_names
column_names <- names(numeric_bird_test)
column_names

levels(numeric_bird_test$species_numeric) <- levels(numeric_bird_train$species_numeric)
# Define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, method = "cv", number = 3)

# Run the RFE algorithm
results <- rfe(numeric_bird_train[, -ncol(numeric_bird_train)], 
                $species_numeric, 
               sizes = c(1:5, 10, 15), 
               rfeControl = control)

# Print the results
print(results)

# List the chosen features
predictors(results)


predictions <- predict(results, newdata = numeric_bird_test)
print(predictions)
print(numeric_bird_test$species)
confusion_matrix <- table(predictions, numeric_bird_test$species)
print(confusion_matrix)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy)) # Accuracy: 0.00170454545454545"

####################################################################
## Embedded methods perform feature selection during the model training process
## LASSO

# 

# Install and load the glmnet package
#install.packages("glmnet")
library(glmnet)


# Prepare the data
x_train <- as.matrix(numeric_bird_train[, -which(names(numeric_bird_train) == "species_numeric")])
y_train <- numeric_bird_train$species_numeric

x_test <- as.matrix(numeric_bird_test[, -which(names(numeric_bird_test) == "species_numeric")])
y_test <- numeric_bird_test$species_numeric

# Perform k-fold cross-validation to find the optimal lambda
cv_model <- cv.glmnet(x_train, y_train, alpha = 1)

# Find the optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min

# Train the LASSO model using the optimal lambda
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)

# Make predictions on the test set
predictions <- predict(lasso_model, s = best_lambda, newx = x_test)

# Evaluate the model performance
mse <- mean((predictions - y_test)^2)
print(paste("Mean Squared Error:", mse)) ## 276.295

####################################################################
## Hybrid methods combine aspects of filter, wrapper, and embedded methods 
## Filter with Random Forest

# 

library(caret)
library(randomForest)
# Load the datasets


# Prepare the data
x_train <- numeric_bird_train[, -which(names(numeric_bird_train) == "species_numeric")]
y_train <- numeric_bird_train$species_numeric

x_test <- numeric_bird_test[, -which(names(numeric_bird_test) == "species_numeric")]
y_test <- numeric_bird_test$species_numeric

# Step 1: Filter Method - Use correlation to filter features
correlation_matrix <- cor(x_train)
highly_correlated <- findCorrelation(correlation_matrix, cutoff = 0.75)
x_train_filtered <- x_train[, -highly_correlated]
x_test_filtered <- x_test[, -highly_correlated]

# Step 2: Wrapper Method - Use Recursive Feature Elimination (RFE)--Took 10 hrs
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
rfe_results <- rfe(x_train_filtered, y_train, sizes = c(1:5, 10, 15), rfeControl = control)

# Print the results
print(rfe_results)

# List the chosen features
selected_features <- predictors(rfe_results)
print(selected_features)

# Train the final model using the selected features--10 mins
final_model <- randomForest(x_train_filtered[, selected_features], y_train, importance = TRUE)

# Make predictions on the test set
predictions <- predict(final_model, newdata = x_test_filtered[, selected_features])

# Evaluate the model performance
confusion_matrix <- table(predictions, y_test)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))-- 0.00340909090909091



