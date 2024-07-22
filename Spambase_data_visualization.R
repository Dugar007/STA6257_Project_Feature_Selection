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
spam_data <- read.csv("./data/spambase/spambase.data")


# Plotting in graph to see the distribution and find outlier 
ggplot(spam_data, aes(x = X1)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Spam Or Not", y = "Count", title = "Distribution")
