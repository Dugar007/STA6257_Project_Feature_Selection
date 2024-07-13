
# Advanced Statistical Modeling - Feature Selection Group 

# Coronavirus
# The data has been downloaded from https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification
# Unzipped the files and renamed train to corona_train and test to corona_test files respectively.

# Step #1 Data Ingesting into R 


#download datasets, if necessary

download_kaggle_dataset <- function(dataset, path) {
  # Check if the kaggle command is available
  if (system("which kaggle", intern = TRUE) == "") {
    stop("Kaggle API is not installed or not in PATH. Please install it first.")
  }
  
  # Ensure the destination directory exists
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  
  # Construct the download command
  command <- sprintf("kaggle datasets download -d %s -p %s", dataset, path)
  
  # Execute the command
  system(command)
  
  # Unzip the downloaded file
  zipfile <- list.files(path, pattern = "*.zip", full.names = TRUE)
  if (length(zipfile) > 0) {
    unzip(zipfile, exdir = path)
    file.remove(zipfile)
  }
}

download_kaggle_dataset("datatattle/covid-19-nlp-text-classification", "./data/covid")


#Importing Train Data
corona_train <- read.csv("./data/covid/Corona_NLP_train.csv")
dim(corona_train)
head(corona_train,3)


#Importing Test Data 

corona_test <- read.csv("./data/covid/Corona_NLP_test.csv")
dim(corona_test)
head(corona_test,3)



#Data Cleaning and Analysis
column_names <- names(corona_train)
column_names
unique(corona_train$Sentiment)
unique(corona_train$Location)
na_counts <- colSums(is.na(corona_train))
na_counts


# Load the ggplot2 package
library(ggplot2)
# Plotting in graph to see the distribution and find outlier 
ggplot(corona_train, aes(x = Sentiment)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Sentiment", y = "Count", title = "Distribution")

corona_train_limit <- corona_train[, c("OriginalTweet", "Sentiment")]

print(corona_train_limit)


#install.packages("tidytext")
library(tidytext)

### Cleaning Steps ##

# Remove URLs

corona_train_limit$OriginalTweet <- gsub("http\\S+|www\\S+", "", corona_train_limit$OriginalTweet)

# Convert to lower
corona_train_limit$OriginalTweet <- tolower(corona_train_limit$OriginalTweet)

# Remove Spaces
#corona_train_limit$OriginalTweet <- gsub("\\s+", " ", corona_train_limit$OriginalTweet)

#Remove Special Characters, # and @
#corona_train_limit$OriginalTweet <- gsub("[^a-zA-Z\\s]", "", corona_train_limit$OriginalTweet)
corona_train_limit$OriginalTweet <- gsub("#\\w+", "", corona_train_limit$OriginalTweet)
corona_train_limit$OriginalTweet <- gsub("@\\w+", "", corona_train_limit$OriginalTweet)


#Tokenize the tweets
#install.packages("tokenizers")
library(tokenizers)



# Word tokenization
corona_train_limit$OriginalTweet_Token <- tokenize_words(corona_train_limit$OriginalTweet)
print(corona_train_limit)















