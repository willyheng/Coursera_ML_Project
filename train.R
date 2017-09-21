if (!file.exists("pml-training.csv")) {
  fileUrl<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileUrl, "pml-training.csv", method="curl")

  fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileUrl, "pml-testing.csv", method="curl")
}

library(dplyr)
library(caret)

# Read data
nachar = c("NA", "", "#DIV/0!")
training_raw <- read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings = nachar)
testing_raw <- read.csv("pml-testing.csv", stringsAsFactors=FALSE, na.strings = nachar)

# Data Cleaning

## Remove columns with all na
training_raw <- training_raw[,!apply(is.na(training_raw), 2, all)]

## Remove other unnecessary columns
exclude <- c("X", "num_window", "user_name", "raw_timestamp_part_1", 
             "raw_timestamp_part_2", "cvtd_timestamp", "new_window")
training <- training_raw %>%
  dplyr::select(-one_of(exclude))

## Remove columns with lack of data
dataCount <- sapply(training, function(x) sum(!is.na(x)))
training <- training[, which(dataCount > 1000)]

## Remove unnecessary columns in testing dataframe
testing <- testing_raw[, names(testing_raw) %in% names(training)]

## Convert data type
training <- training %>% 
  mutate_if(is.integer, as.numeric) %>%
  mutate(classe = as.factor(classe)) %>%
  na.omit

testing <- testing %>%
  mutate_if(is.integer, as.numeric) 

# Training

## Calculate principal components
components <- prcomp(training[,-ncol(training)], scale=TRUE)
propExplained <- cumsum(components$sdev**2/sum(components$sdev**2))

## Plot principal components
plot(propExplained, main="Prop of Variance explained by PC",
     xlab = "PC", ylab = "%", type="l")

## Use up to PC5 to train 
set.seed(62433)
preProc <- preProcess(training, method="pca", pcaComp=5)
trainPC <- predict(preProc, training)
trainControl <- trainControl(method="cv", number=10)

modelFit <- train(x = trainPC[,-1], y=training$classe, model="lda")

# Predict for Train
predictTrain <- predict(modelFit, trainPC)
confusionMatrix(training$classe, predictTrain)

# Predict for Test
testPC <- predict(preProc, testing)
predictTest <- predict(modelFit, testPC)
confusionMatrix(testing$classe, predict(modelFit2, testPC))
