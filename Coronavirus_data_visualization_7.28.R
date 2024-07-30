# Load necessary libraries
library(dplyr)
library(quanteda)
library(caret)
library(e1071)
library(wordcloud)
library(ggplot2)
# Step #1 Data Ingesting into R 
# download datasets, if necessary

# download_kaggle_dataset("datatattle/covid-19-nlp-text-classification", "./data/covid")

set.seed(42)
# Importing Train Data
corona_train <- read.csv("./data/covid/Corona_NLP_train.csv")[, c("OriginalTweet", "Sentiment")]

# Importing Test Data
corona_test <- read.csv("./data/covid/Corona_NLP_test.csv")[, c("OriginalTweet", "Sentiment")]



# Recode Sentiment to factors
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
test_dfm <- preprocess_text(corona_test$OriginalTweet)

# Convert DFM to data frame
train_data <- convert(train_dfm, to = "data.frame")
test_data <- convert(dfm_match(test_dfm, features = featnames(train_dfm)), to = "data.frame") 

# Plotting in graph to see the distribution and find outlier

ggplot(corona_train, aes(x = Sentiment, fill = Sentiment)) +
  geom_bar() +
  scale_fill_manual(values = c("Negative" = "pink", "Positive" = "lightblue")) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(x = "Sentiment", y = "Count", title = "Distribution of Sentiment Labels")


# Calculate and visualize mean, standard deviation, and range of tokens per document
token_counts <- rowSums(train_dfm)


num_bins <- length(seq(min(token_counts), max(token_counts), by = 5))
unique_colors <- rainbow(num_bins)

ggplot(data = data.frame(token_counts), aes(x = token_counts, fill = factor((..x.. %/% 5) * 5))) +
  geom_histogram(binwidth = 5, color = "black") +
  scale_fill_manual(values = unique_colors, name = "Token Count Range") +
  scale_x_continuous(breaks = seq(0, 40, by = 5)) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Histogram of Token Counts",
    x = "Token Counts",
    y = "Frequency"
  )


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