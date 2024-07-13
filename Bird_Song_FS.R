
# Advanced Statistical Modeling - Feature Selection Group 

# Birds' Songs Numeric Dataset
# The data has been downloaded from https://www.kaggle.com/datasets/fleanend/birds-songs-numeric-dataset
# Unzipped the files and renamed train to bird_train and test to bird_test files respectively.

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

download_kaggle_dataset("fleanend/birds-songs-numeric-dataset", "./data/birds")

#Importing Train Data

bird_train <- read.csv("./data/birds/train.csv")
dim(bird_train)
head(bird_train,3)

#Importing Test Data 

bird_test <- read.csv("./data/birds/test.csv")
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

#Adding  column Genus and species to make it unique --Train Data
bird_train$gen_spec <- paste(bird_train$genus, bird_train$species, sep = "_")
unique(bird_train$gen_spec)

#Adding  column Genus and species to make it unique --Test Data
bird_test$gen_spec <- paste(bird_test$genus, bird_test$species, sep = "_")


ggplot(bird_train, aes(x = gen_spec)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "gen_spec", y = "Count", title = "Distribution")


