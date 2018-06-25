#!/usr/bin/env Rscript
R.Version()$version.string

# Implementing Cost Sensitive learning approach to handle class imbalance.
# Build a SVM model to predcit breast cancer tissues on Bootstrapped dataset  

# Get Libraries 
rm(list = ls())
library(caret)
library(e1071)
library(kernlab)
library(dplyr)
library(plyr)
library(R.utils)
library(tictoc)


# Set working and data directory 
mainDir <- "/set_working_directory/" 
dataDir <- "/path/to/Bootstrapped_dataset/"


# Give the simulation parameters
model <- cmdArg(model = "SVM_Thresholding")
n.simulations <- cmdArg(n.simulations = 1000L)


cat("Started Support Vector Machines ...\n")
for (idx in 1:n.simulations){
  tic("Total Execution time")
  
  set.seed(idx)
  
  #Load training and validation data
  training.data <- as.data.frame(read.csv(file = paste(dataDir,"Training/","training_data_",idx,".csv",sep = "")))
  val.data <- as.data.frame(read.csv(file = paste(dataDir,"Validation/","validation_data_",idx,".csv",sep = "")))
  test.data <- as.data.frame(read.csv(file = paste(dataDir,"Test/","test_data_",idx,".csv",sep = "")))
  
  
  #Create forumala for RF
  varNames <- names(training.data)
  varNames <- varNames[!varNames %in% c("Outcome")]
  varNames.add.sign <- paste(varNames, collapse = "+")
  svm.formula <- as.formula(paste("Outcome",varNames.add.sign, sep = "~"))
  
  
  #Scaling the data 
  normalised.training.data <- cbind(as.data.frame(scale(training.data[,-length(training.data)])),training.data[,length(training.data)])
  colnames(normalised.training.data)[length(normalised.training.data)] <- "Outcome"
  
  normalised.val.data <- cbind(as.data.frame(scale(val.data[,-length(val.data)])),val.data[,length(val.data)])
  colnames(normalised.val.data)[length(normalised.val.data)] <- "Outcome"
  
  normalised.test.data <- cbind(as.data.frame(scale(test.data[,-length(test.data)])),test.data[,length(test.data)])
  colnames(normalised.test.data)[length(normalised.test.data)] <- "Outcome"
  
  
  ## Model Building
  source("/path/to/svm_model.R/")
   svm_final_model <- svm_model(formula =  svm.formula,data =  normalised.training.data)

    ### Training set prediction 
  source("/path/to/predictions.R")
  train_X <- normalised.training.data[,-which(names(normalised.training.data) %in% c("Outcome"))]
  train_Y <- normalised.training.data$Outcome
  training_predictions <- prediction(model_object = svm_final_model,data_X =  train_X,data_y =  train_Y )
  
  
  ### Validation set prediction 
  val_X <- normalised.val.data[,-which(names(normalised.val.data) %in% c("Outcome"))]
  val_Y <- normalised.val.data$Outcome
  
  #check on range of cutoff 
  cutoff_range <- seq(from = 0.1 , to = 0.9, by =0.01)
  misclass_err_mtx <- matrix(data = NA,nrow = length(cutoff_range), ncol = 1 )
  val_pred <- prediction(model_object = svm_final_model,data_X = val_X, data_y = val_Y)
  count = 1
  
  for(i_cutoff in cutoff_range){
   val_pred_cutoff <- as.factor(ifelse(val_pred$predicted_probs$M >= i_cutoff, 'M', 'B'))
   misclass_err <- (mean( val_pred_cutoff != val_Y))
   
   misclass_err_mtx[count,1] <- misclass_err
   count = count +1
   }
  
  err_cutoff <- data.frame(cutoff = cutoff_range , error = misclass_err_mtx)
  min_misclass_val <- min(misclass_err_mtx) 
  optimum_cutoff <- err_cutoff[err_cutoff$error %in% min_misclass_val,1] %>% mean()
  
  ### Test data Predictions 
  test_X <-  normalised.test.data[,-which(names(normalised.test.data) %in% c("Outcome"))]
  test_Y <- normalised.test.data$Outcome
  test_predictions <- prediction(model_object = svm_final_model,data_X = test_X,data_y = test_Y)
  
  ### Setting cutoff 
  pred_test_cutoff <- as.factor(ifelse(test_predictions$predicted_probs$M >= optimum_cutoff, 'M', 'B'))
  Confusion.Matrix.Test_Cutoff <- confusionMatrix(data = pred_test_cutoff,
                                                  reference = test_Y,
                                                  positive = "M")
  
  #Measure prediction error
  missclass.err <-  (mean(pred_test_cutoff != test_Y))
  
  # LogLoss error
  source("path/to/logLoss.R")
  prediction_test_logloss <- apply(test_predictions$predicted_probs, c(1,2), function(x) min(max(x, 1E-15), 1-1E-15)) 
  logloss_err <- logLoss(prediction_test_logloss, test_Y)
  
  #Data frame of Truth and Predicted values
  predicted.test.set <- cbind(test.data[,-length(test.data)], pred_test_cutoff)
  colnames(predicted.test.set)[length(test.data)] <- c("Pred.Outcome")
  
  #Get AUC value
  require(ROCR)
  require(pROC)
  auc.probs <- test_predictions$predicted_probs$M
  perf_ROC <- roc(test_Y ,auc.probs)
  auc_output <- pROC::auc(perf_ROC)

  toc()
}
cat("All done!!")

