library(maxent)
library(SnowballC)
library(tm)
library(RTextTools)
library(tidyverse)
library(ggplot2)
library(wordcloud)

#Functions
preprocess_data <- function(data) {
  #lower corpus
  data <- tm_map(data, content_transformer(tolower))
  
  #remove punctuation
  data <- tm_map(data, content_transformer(removePunctuation))
  
  #remove stop words
  data <- tm_map(data, content_transformer(removeWords), stopwords("english"))
  
  #stemDoc
  data <- tm_map(data, stemDocument)
  
  #remove numbers
  data <- tm_map(data, content_transformer(removeNumbers))
  
  #strip white spaces
  data <- tm_map(data, stripWhitespace)
  
  return(data)
}

set.seed(999)

#Prepare data
data("USCongress", package = "RTextTools")

#congress <- USCongress[!(USCongress$major == 99),]
congress <- USCongress[,]

# topic labels
major_topics <- tibble(
  major = c(1:10, 12:21, 99),
  label = c(
    "Macroeconomics", "Civil rights, minority issues, civil liberties",
    "Health", "Agriculture", "Labor and employment", "Education", "Environment",
    "Energy", "Immigration", "Transportation", "Law, crime, family issues",
    "Social welfare", "Community development and housing issues",
    "Banking, finance, and domestic commerce", "Defense",
    "Space, technology, and communications", "Foreign trade",
    "International affairs and foreign aid", "Government operations",
    "Public lands and water management", "Other, miscellaneous"
  )
)

(congress <- as_tibble(USCongress) %>%
    mutate(text = as.character(text)) %>%
    left_join(major_topics))


#Change major 99 to 22 to plot
congress$major[congress$major == 99] <- 22

#Plot labels
ggplot(congress, aes(x=major)) + geom_bar(aes(fill=label))


#Separate in train/test
sample <- sample.int(n = nrow(congress), size = floor(.70 * nrow(congress)), replace = F)
train <- congress[sample, ]
test <- congress[-sample, ]

train_corpus <- Corpus(VectorSource(train$text))
test_corpus <- Corpus(VectorSource(test$text))

#Preprocess both corpus
train_corpus <- preprocess_data(train_corpus)
test_corpus <- preprocess_data(test_corpus)

sparse_train <- as.compressed.matrix(DocumentTermMatrix(train_corpus))
sparse_test <- as.compressed.matrix(DocumentTermMatrix(test_corpus))

#TODO: decidir que nfold usamos
f <- tune.maxent(sparse_train, train$major, nfold = 3, showall = TRUE, verbose = TRUE)
f2 <- f <- tune.maxent(sparse_train, train$major, nfold = 5, showall = TRUE, verbose = TRUE)

print(f)
print(f2)

#Creamos el modelo en base al mejor pct_best_fit anterior
model <- maxent(sparse_train, congress$major, l1_regularizer = 0.0, l2_regularizer = 0.2, 
                use_sgd = 0, set_heldout = 0)

model_nfold5 <- maxent(sparse_train, congress$major, l1_regularizer = 0.0, l2_regularizer = 0.4, 
                       use_sgd = 0, set_heldout = 0)

#utilizando el dataset test
results <- predict(model, sparse_test)



print(which.max(results[1, 2:length(results[1,])]))

#Get confusion matrix

#Relevant topics
relevant_topics <- c()

confusion_mat <- data.frame(label=integer(), predicted=integer(), retrieved=integer(), relevant=integer())

for(row in 1:nrow(results)) {
  row_label <- as.numeric(results[row, 1])
  predicted <- as.numeric(names(which.max(results[row, 2:ncol(results)])))
  
  is_relevant <- 0
  if(predicted == 20 | predicted == 4 | predicted == 7 | predicted == 8) {
    is_relevant <- 1
  }
  
  confusion_mat <- rbind(confusion_mat, data.frame(label=row_label, predicted=predicted, retrieved=0, relevant=0))
}





#Word cloud with all data
all_corpus <- Corpus(VectorSource(congress$text))
all_corpus <- preprocess_data(all_corpus)
matrix_all_data <- as.matrix(TermDocumentMatrix(all_corpus))
words <- sort(rowSums(matrix_all_data), decreasing = TRUE)
df_all_data <- data.frame(word = names(words), freq = words)

wordcloud(words=df_all_data$word,freq=df_all_data$freq, min.freq=1, max.words=300, 
          random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))


#Muestra del preprocesamiento
for(i in 1:5) print(congress$text[i])
for(i in 1:5) print(all_corpus[[i]]$content)

