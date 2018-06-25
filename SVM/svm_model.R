
# function for building svm  model with 10 fold CV 
# We train the model at different gamma and cost values and select the best model that results in high ROC-AUC

svm_model <- function(formula ,data ){
  
  ctrl <- trainControl(method = "CV",
                       number  = 10,
                       classProbs = TRUE,
                       summaryFunction = twoClassSummary)
  
  tunegrid <- expand.grid(gamma.val = 2^(-15:3),
                          cost.val =  10^(-2:7))
                           
  
  svm_model <- caret::train(formula, 
                    data = data,
                    method = "svmRadial",
                    metric = "ROC",
                    tunegrid = tunegrid,
                    trControl = ctrl)
  
  #Return the models for function output 
  svm_model$finalModel
  
 #function ends
}