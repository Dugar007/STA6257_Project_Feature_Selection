library(ggplot2)
library(dplyr)

#Importing Spam Data
spam_data <- read.csv("spambase.data")
head(spam_data)


# Plotting in graph to see the distribution and find outlier 
spam_data$X1 <- factor(spam_data$X1, levels = c(0, 1), labels = c("Not Spam", "Spam"))

ggplot(spam_data, aes(x = X1, fill = X1)) +
  geom_bar(color = "black") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10, color = "darkblue"),
    axis.text.y = element_text(size = 10, color = "darkblue"),
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    panel.grid.major = element_line(color = "lightgrey"),
    panel.grid.minor = element_blank()
  ) +
  labs(x = "Spam Or Not", y = "Count", title = "Distribution of Spam vs Not Spam") +
  scale_y_continuous(labels = scales::comma) +
  scale_x_discrete(labels = c("Not Spam" = "Not Spam", "Spam" = "Spam")) +
  scale_fill_manual(values = c("Not Spam" = "red", "Spam" = "blue")) +
  guides(fill = guide_legend(title = NULL))
