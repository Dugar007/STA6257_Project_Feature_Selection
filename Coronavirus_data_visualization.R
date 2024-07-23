# Load necessary libraries
library(dplyr)
library(quanteda)
library(caret)
library(e1071)
library(wordcloud)
# Step #1 Data Ingesting into R 
# download datasets, if necessary

# download_kaggle_dataset("datatattle/covid-19-nlp-text-classification", "./data/covid")

set.seed(42)
# Importing Train Data
corona_train <- read.csv("./data/covid/Corona_NLP_train.csv")[, c("OriginalTweet", "Sentiment")]

# Importing Test Data
corona_test <- read.csv("./data/covid/Corona_NLP_test.csv")[, c("OriginalTweet", "Sentiment")]

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


train_dfm <- dfm_trim(preprocess_text(corona_train$OriginalTweet), min_termfreq = 10)
test_dfm <- preprocess_text(corona_test$OriginalTweet)

# Convert DFM to data frame
train_data <- convert(train_dfm, to = "data.frame")
test_data <- convert(dfm_match(test_dfm, features = featnames(train_dfm)), to = "data.frame") 


# Calculate and visualize mean, standard deviation, and range of tokens per document
token_counts <- rowSums(train_dfm)
mean_tokens <- mean(token_counts)
median_tokens <- median(token_counts)
sd_tokens <- sd(token_counts)
range_tokens <- range(token_counts)

train_stats <- data.frame(
  Mean = mean_tokens,
  SD = sd_tokens,
  Min = range_tokens[1],
  median = median_tokens,
  Max = range_tokens[2]
)

print("Token statistics for training data:")
print(train_stats)

# Calculate and report sparsity
num_columns <- ncol(train_dfm)
num_nonzero <- sum(train_dfm@x != 0)
total_elements <- prod(dim(train_dfm))
sparsity <- ((total_elements - num_nonzero) / total_elements) * 100

cat("Number of columns in the training dataset:", num_columns, "\n")
cat("Average percent of values that are 0 in the training dataset:", sparsity, "%\n")

# Create a word frequency cloud
word_freq <- colSums(train_dfm)
wordcloud(names(word_freq), freq = word_freq, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2"))