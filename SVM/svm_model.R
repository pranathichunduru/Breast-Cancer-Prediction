
# function for building random forest model with 5 fold CV using caret package.
# We trained the model at different nTree and mtry values and select the best model that results in low OOB error rate

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