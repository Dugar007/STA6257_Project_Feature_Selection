library(ggplot2)
library(dplyr)

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


# Plotting in graph to see the distribution and find outlier 
ggplot(spam_data, aes(x = flag_spam)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Spam Or Not", y = "Count", title = "Distribution")
