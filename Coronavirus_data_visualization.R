# Load necessary libraries
library(dplyr)
library(quanteda)
library(caret)
library(e1071)
# library(parallel)
# library(foreach)
# library(doParallel)

# Setup parallel processing
# num_cores <- detectCores() - 1 # Use one less than the total number of cores
# cl <- makeCluster(num_cores)
# registerDoParallel(cl)

# Step #1 Data Ingesting into R 
# download datasets, if necessary

# download_kaggle_dataset("datatattle/covid-19-nlp-text-classification", "./data/covid")

set.seed(42)
# Importing Train Data
corona_train <- read.csv("./data/covid/Corona_NLP_train.csv")[, c("OriginalTweet", "Sentiment")]
# corona_train <- sample_frac(corona_train, .5)

# Importing Test Data
corona_test <- read.csv("./data/covid/Corona_NLP_test.csv")[, c("OriginalTweet", "Sentiment")]
# corona_test <- sample_frac(corona_test, .5)

# Load ggplot2 package
# Plotting in graph to see the distribution and find outlier
library(ggplot2)
ggplot(corona_train, aes(x = Sentiment)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Sentiment", y = "Count", title = "Distribution")

# Recode Sentiment to factors
corona_train <- corona_train %>%
  mutate(Sentiment = recode(Sentiment,
                            "Extremely Negative" = "Negative",
                            "Extremely Positive" = "Positive"),
         Sentiment = factor(Sentiment, levels = c("Negative", "Neutral", "Positive")))

corona_test <- corona_test %>%
  mutate(Sentiment = recode(Sentiment,
                            "Extremely Negative" = "Negative",
                            "Extremely Positive" = "Positive"),
         Sentiment = factor(Sentiment, levels = c("Negative", "Neutral", "Positive")))

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

# Parallelized preprocessing
train_dfm <- preprocess_text(corona_train$OriginalTweet)
test_dfm <- preprocess_text(corona_test$OriginalTweet)

# Convert DFM to data frame
train_data <- convert(dfm_trim(train_dfm, min_termfreq = 10), to = "data.frame")
test_data <- convert(dfm_match(test_dfm, features = featnames(train_dfm)), to = "data.frame") 

# Add the Sentiment column
train_data$Sentiment <- corona_train$Sentiment
test_data$Sentiment <- corona_test$Sentiment

# Train a Naive Bayes model without cross-validation
model <- naiveBayes(Sentiment ~ ., data = train_data)

# Predict on the training data
train_predictions <- predict(model, newdata = train_data)

# Predict on the test data
test_predictions <- predict(model, newdata = test_data)

# Confusion matrix to evaluate the performance
confusionMatrix(train_predictions, train_data$Sentiment)
confusionMatrix(test_predictions, test_data$Sentiment)

# Stop the cluster
# stopCluster(cl)
# registerDoSEQ()

