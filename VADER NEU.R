rm(list=ls())

#Data retrieved from https://www.kaggle.com/dsv/3877817
data <- read.csv("C:/Users/ekaba/Desktop/Arbeit und Uni/7. Semester/Data Science/EcoPreprocessed.csv", header = T)

#data info
summary(data)
str(data)
table(data$division)

#Only positive and negative as sentiment
data <- subset(data, division != "neutral")
table(data$division)

#For replicability
set.seed(11)

#Split to training and test data
dt = sort(sample(nrow(data), nrow(data)*.7))
train<-data[dt,]
test<-data[-dt,]

table(train$division)
table(test$division)

## VADER
library(vader)
vader <- vader_df(test$review)
vader <- cbind(vader, test$division)
names(vader)[names(vader) == "test$division"] <- "division"

#Confusion matrix of VADER for negative sentiment

tnNEG <- nrow(vader[vader$compound >= 0.1 & vader$division == "positive", ])
tpNEG <- nrow(vader[vader$compound < 0.1 & vader$division == "negative", ])
fnNEG <- nrow(vader[vader$compound >= 0.1 & vader$division == "negative", ])
fpNEG <- nrow(vader[vader$compound < 0.1 & vader$division == "positive", ])



#Evaluation parameters for VADER
accVADER <- (tpNEG+tnNEG)/(tpNEG+tnNEG+fpNEG+fnNEG)
#for negative sentiment
precVADERneg <- (tpNEG)/(tpNEG+fpNEG)
recaVADERneg <- (tpNEG)/(tpNEG+fnNEG)
F1VADERneg <- (2*precVADERneg*recaVADERneg)/(precVADERneg+recaVADERneg)

#for postitive sentiment
tpPOS <- nrow(vader[vader$compound >= 0.1 & vader$division == "positive", ])
tnPOS <- nrow(vader[vader$compound < 0.1 & vader$division == "negative", ])
fpPOS <- nrow(vader[vader$compound >= 0.1 & vader$division == "negative", ])
fnPOS <- nrow(vader[vader$compound < 0.1 & vader$division == "positive", ])

precVADERpos <- (tpPOS)/(tpPOS+fpPOS)
recaVADERpos <- (tpPOS)/(tpPOS+fnPOS)
F1VADERpos <- (2*precVADERpos*recaVADERpos)/(precVADERpos+recaVADERpos)

## NAIVE BAYES 
# source for code: https://www.kaggle.com/code/victornugraha/sentiment-analysis-w-lstm-naive-bayes-in-r/notebook
library(e1071)
##ML evaluation
library(caret)
library(tm)

#into corpus form 
trainNB1 <- VCorpus(VectorSource(train$review))
testNB1 <- VCorpus(VectorSource(test$review))
trainNB1
testNB1

#tokenization
trainNB1 <- DocumentTermMatrix(trainNB1)
testNB1 <- DocumentTermMatrix(testNB1)
trainNB1
testNB1

#Feature-selection
freq_words <- findFreqTerms(trainNB1, lowfreq = 100)
trainNB1 <- trainNB1[ , freq_words]
testNB1 <- testNB1[ , freq_words]
trainNB1
testNB1

#bernoulli converter
bnc <- function(x){
  x <- as.factor(ifelse(x > 0, 1, 0))
  return(x)
}

trainbc <- apply(trainNB1, MARGIN = 2, bnc)
testbc <- apply(testNB1, MARGIN = 2, bnc)

trainNB2 <- as.factor(train$division)
testNB2 <- as.factor(test$division)

head(trainNB2)
head(testNB2)

#Final NB classifier with add 1 Laplace smoothing
NBclassifier <- naiveBayes(trainbc, trainNB2, laplace = 1)

#Testing
testPRED <- stats::predict(NBclassifier, testbc)

#Evaluation with confusion matrix with negative sentiment
evalNB <- confusionMatrix(data = as.factor(testPRED), reference = as.factor(testNB2))
evalNB










