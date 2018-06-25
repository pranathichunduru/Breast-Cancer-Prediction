#Determining Log loss 
#model.matrix generates a true probabilities matrix, where an element is either 1 or 0
#we subtract the prediction, and, if the result is bigger than 0 that's the correct class
logLoss = function(pred, actual){
  -1*mean(log(pred[model.matrix(~ actual + 0) - pred > 0]))
}

LogLossBinary = function(actual, predicted, eps = 1e-15) {
  predicted = pmin(pmax(predicted, eps), 1-eps)
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}



