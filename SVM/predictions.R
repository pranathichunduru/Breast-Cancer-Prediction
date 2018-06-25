# function for getting predicted sets for training ,validation and set 

prediction <- function(model_object, data_X, data_y ){
  predicted_output <-  predict(model_object, data_X)
  
  
  predicted_probabilities <- as.data.frame(predict(model_object,
                                            data_X,
                                            type = "prob"))
  
  # Create Confusion Matrix for Validation
  Confusion_Matrix <- confusionMatrix(data= predicted_output,
                                            reference = data_y,
                                            positive = "M")
  
  list(predicted_set = predicted_output, predicted_probs = predicted_probabilities, confusion_mtx = Confusion_Matrix)
  
}

