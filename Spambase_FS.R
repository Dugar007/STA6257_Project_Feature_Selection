
# Advanced Statistical Modeling - Feature Selection Group 

# Spambase
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

download_and_extract_zip("https://archive.ics.uci.edu/static/public/94/spambase.zip", "./data/spambase")

#Importing Spam Data
spam_data <- read.csv("./data/spambase/spambase.data")
dim(spam_data)
head(spam_data,3)
colMax <- function(data) sapply(data, max, na.rm = TRUE)
max_across_columns <- colMax(spam_data)
max_across_columns # we need to normalize the values 
normalized_data <- scale(spam_data, center = FALSE, scale = apply(spam_data, 2, max) - apply(spam_data, 2, min))

#Importing Names data

spam_names <- readLines("./data/spambase/spambase.names", n = -1, skip = 35)
spam_names_57 <-tail(spam_names,57) # Taking only required columns
cat(spam_names_57, sep = "\n")
spam_names_57 
transposed_df <- t(spam_names_57)# Transposing rows to columns
transposed_df



#Data Cleaning and Analysis
column_names <- names(spam_data)
column_names
unique(spam_data$X1)
unique(spam_data$X0)



# Load the ggplot2 package
library(ggplot2)
# Plotting in graph to see the distribution and find outlier 
ggplot(spam_data, aes(x = X1)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Spam Or Not", y = "Count", title = "Distribution")





