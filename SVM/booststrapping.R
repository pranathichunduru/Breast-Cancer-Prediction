# Load required libraries 
rm(list = ls())

library(dplyr)
library(plyr)
library(caret)
library(randomForest)
library(corrplot)
library(R.utils)
library(doSNOW)
library(ggplot2)


#Breast Cancer dataset 
raw_data <- read.csv("/path/to/breast_cancer_data.csv")

#remove columns with no values
raw_data <- raw_data[, -which(colMeans(is.na(raw_data)) > 0.6)]

#remove data with dependent and identifier variable 
bc_data <- raw_data[ , -which(names(raw_data) %in% c("id"))]
X <- raw_data[ , -which(names(raw_data) %in% c("diagnosis","id"))]
y <- as.factor(raw_data$diagnosis)


pred_corr <- cor(X) # get correlations
corrplot(pred_corr, method = "circle") #plot matrix

#remove highly correlated variables
pred_corr[upper.tri(pred_corr)] <- 0
diag(pred_corr) <- 0
X.new <- X[,!apply(pred_corr,2,function(x) any(x > 0.99))]

bc_data_new <- cbind(X.new,y)
colnames(bc_data_new)[29] <- c("Outcome")

## NOTE: You can either save the bootstrapped dataset to  use it for implementing different classification algorithms or train the model within loop to avoid allocating additonal storage space.  
#I have created a directory to save the bootstrapped dataset to compre performance of different models on the breaset cancer dataset. 

# ## Implementing Bootstrapping with Replacement 
# Within class bootstrapping 
benign_class <- bc_data_new[bc_data_new$Outcome == "B",] 
malignant_class <- bc_data_new[bc_data_new$Outcome != "B",]



mainDir <- "path/to/directory/"
subDir <-  "name-the-directory"

dir.create(file.path(mainDir, subDir), showWarnings = TRUE)

mainDir_1 <- paste0(mainDir , subDir , "/")
subDir_1 <- "Training"
subDir_2 <- "Validation"
subDir_3 <- "Test"

dir.create(file.path(mainDir_1, subDir_1), showWarnings = T)
dir.create(file.path(mainDir_1, subDir_2), showWarnings = T)
dir.create(file.path(mainDir_1, subDir_3), showWarnings = T)


setwd(file.path(mainDir, subDir))
n.iterations <- cmdArg(n.iterations = 1000L)

for (idx in 1: n.iterations){
  set.seed(idx)
    #for BC = benign class
  index.train.benign <- sample (nrow(benign_class), replace = TRUE)
  training.set.benign <- benign_class[index.train.benign,]
  
  index.test.val.benign <- data.frame(index = setdiff (seq_len (nrow (benign_class)), index.train.benign))
  index.val.benign <- sample(index.test.val.benign$index, size = 0.5*nrow(index.test.val.benign))
  index.test.benign <- index.test.val.benign[!(index.test.val.benign$index %in% index.val.benign),]

  val.set.benign <- benign_class[index.val.benign,]
  test.set.benign <- benign_class[index.test.benign,]
  
  
  
  # for  BC = Malignant class
  index.train.malignant <- sample (nrow(malignant_class), replace = TRUE)
  training.set.malignant <- malignant_class[index.train.malignant,]
  
  index.test.val.malignant <- data.frame(index = setdiff (seq_len (nrow (malignant_class)), index.train.malignant))
  index.val.malignant <- sample(index.test.val.malignant$index, size = 0.5*nrow(index.test.val.malignant))
  index.test.malignant <- index.test.val.malignant[!(index.test.val.malignant$index %in% index.val.malignant),]

  val.set.malignant <- malignant_class[index.val.malignant,]
  test.set.malignant <- malignant_class[index.test.malignant,]
 
  #Combine training and test set
  training.set <- rbind(training.set.benign,training.set.malignant)
  training.set <- training.set[sample(nrow(training.set)),]
 
  val.set <- rbind(val.set.benign,val.set.malignant)
  val.set <- val.set[sample(nrow(val.set)),]
 
  
  test.set <- rbind(test.set.benign,test.set.malignant)
  test.set <- test.set[sample(nrow(test.set)),]
 
  write.csv(x = training.set ,file = paste("Training/","training_data","_",idx,".csv",sep = ""), sep = "\t",row.names = F, col.names = F)
  write.csv(x = val.set ,file = paste("Validation/","validation_data","_",idx,".csv",sep = ""), sep = "\t",row.names = F, col.names = F)
  write.csv(x = test.set ,file = paste("Test/","test_data","_",idx,".csv",sep = ""), sep = "\t",row.names = F, col.names = F)

}

