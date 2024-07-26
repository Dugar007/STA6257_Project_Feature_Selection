library(ggplot2)
library(dplyr)

# Step #1 Data Ingesting into R 
# download datasets, if necessary
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

# download_kaggle_dataset("fleanend/birds-songs-numeric-dataset", "./data/birds")

#Importing Data
bird_data <- rbind(read.csv("./data/birds/train.csv"), read.csv("./data/birds/test.csv")) %>%
  filter(species %in% c('europaeus', 'schoenobaenus')) %>%
  mutate(species = factor(species, levels = c('europaeus', 'schoenobaenus'))) %>%
  select(-c(id, genus))

#Importing Data
bird_train <- read.csv("./data/birds/train.csv") %>%
  filter(species %in% species_levels) %>%
  mutate(species = factor(species, levels = species_levels)) %>%
  select(-c(id, genus))

bird_test<- read.csv("./data/birds/test.csv") %>%
  filter(species %in% species_levels) %>%
  mutate(species = factor(species, levels = species_levels)) %>%
  select(-c(id, genus))


# Plotting in graph to see the distribution and find outlier 
ggplot(bird_test, aes(x = species)) +
  geom_bar() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Species", y = "Count", title = "Distribution")