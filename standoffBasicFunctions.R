library(caret)
library(ggplot2)
library(data.table)
library(progress)
library(parallel)
library(doParallel)

downloadStandoffPackages <- function() {
  #install.packages('caret', dependencies = TRUE)
  install.packages('https://cran.r-project.org/src/contrib/Archive/caret/caret_6.0-21.tar.gz', repos=NULL, type="source")
  install.packages('ggplot2', dependencies = TRUE)
  install.packages('data.table', dependencies = TRUE)
  install.packages('progress', dependencies = TRUE)
  install.packages('doParallel', dependencies = TRUE)
  install.packages('Rcpp', dependencies = TRUE)
}

didExtremeLossHappen <- function(x) {
  randomNumber <- runif(1,0,100)
  ifelse(x[x[3]] < randomNumber,0,1)
}

getBlankDataTableUsingPredictions <- function() {
  playerOneValue <- runif(100, playerOneMin, playerOneMax)
  playerTwoValue <- runif(100, playerTwoMin, playerTwoMax)
  results <- data.table(playerOneValue, playerTwoValue)
  results
}

getBaseStandoff <- function(results) {
  results$losingPlayer <- apply(results, 1, function(x){ifelse(max(x) == x[1],2,1)})
  results$extremeLoss <- apply(results, 1, didExtremeLossHappen)
  results$playerOneScoreChange <- as.numeric(results$losingPlayer == 2) * 100 - results$extremeLoss * 10000
  results$playerTwoScoreChange <- as.numeric(results$losingPlayer == 1) * 100 - results$extremeLoss * 10000
  results$playerOneTotalScore <- sum(results$playerOneScoreChange)
  results$playerTwoTotalScore <- sum(results$playerTwoScoreChange)
  results
}

setPlayerMinMaxFromPredictions <- function(results, fitForPlayerOne, fitForPlayerTwo) {
  results$predictedValueForPlayerOne <- predict(fitForPlayerOne, results[['playerOneValue']])
  results$predictedValueForPlayerTwo <- predict(fitForPlayerTwo, results[['playerTwoValue']])
  dataSortedForPlayerOne <- results[order(-results$predictedValueForPlayerOne),]
  dataSortedForPlayerTwo <- results[order(-results$predictedValueForPlayerTwo),]
  
  playerOneMin <<- max(min(dataSortedForPlayerOne[1:10,'playerOneValue']) * runif(1,.8,1), 0)
  playerOneMax <<- min(max(dataSortedForPlayerOne[1:10,'playerOneValue']) * runif(1,1,1.2), 100)
  playerTwoMin <<- max(min(dataSortedForPlayerTwo[1:10,'playerTwoValue']) * runif(1,.8,1), 0)
  playerTwoMax <<- min(max(dataSortedForPlayerTwo[1:10,'playerTwoValue']) * runif(1,1,1.2), 100)
}

setPlayerMinMaxFromPredictionsWithLinearModel <- function(results, fitForPlayerOne, fitForPlayerTwo) {
  results$predictedValueForPlayerOne <- predict(fitForPlayerOne, longTermGameStorage)
  results$predictedValueForPlayerTwo <- predict(fitForPlayerTwo, longTermGameStorage)
  dataSortedForPlayerOne <- results[order(-results$predictedValueForPlayerOne),]
  dataSortedForPlayerTwo <- results[order(-results$predictedValueForPlayerTwo),]
  
  playerOneMin <<- max(min(dataSortedForPlayerOne[1:10,'playerOneValue']) * runif(1,.8,1), 0)
  playerOneMax <<- min(max(dataSortedForPlayerOne[1:10,'playerOneValue']) * runif(1,1,1.2), 100)
  playerTwoMin <<- max(min(dataSortedForPlayerTwo[1:10,'playerTwoValue']) * runif(1,.8,1), 0)
  playerTwoMax <<- min(max(dataSortedForPlayerTwo[1:10,'playerTwoValue']) * runif(1,1,1.2), 100)
}

createMinMaxDataTable <- function(nrows = 50) {
  minMaxDataTable <- data.table(matrix(NA_real_, nrow = nrows, ncol = 4))
  colnames(minMaxDataTable) <- c('playerOneMin', 'playerOneMax', 'playerTwoMin', 'playerTwoMax')
  minMaxDataTable
}

addMinMaxObservation <- function() {
  if(!exists("minMaxObservations")) {
    minMaxObservations <<- createMinMaxDataTable()
  }
  naRows <- which(is.na(minMaxObservations$playerOneMin))
  if(length(naRows) == 0) {
    tempObservation <- createMinMaxDataTable(nrow(minMaxObservations) * 2)
    tempObservation[1:nrow(minMaxObservations)] <- minMaxObservations
    minMaxObservations <<- tempObservation
  }
  correctnaRow <- which(is.na(minMaxObservations$playerOneMin))[1]
  minMaxObservations[correctnaRow,'playerOneMin'] <<- playerOneMin
  minMaxObservations[correctnaRow,'playerOneMax'] <<- playerOneMax
  minMaxObservations[correctnaRow,'playerTwoMin'] <<- playerTwoMin
  minMaxObservations[correctnaRow,'playerTwoMax'] <<- playerTwoMax
}

setRandomMinMax <- function() {
  playerOneMax <<- 100
  playerOneMin <<- 0
  playerTwoMax <<- 100
  playerTwoMin <<- 0
}

addResultsToLongTermStorage <- function(newResults) {
  if(exists('longTermGameStorage')) {
    longTermGameStorage <<- rbind(longTermGameStorage, newResults)
  } else {
    longTermGameStorage <<- newResults
  }
}

setupDefaultData <- function() {
  setRandomMinMax()
  for(i in 1:5) {
    addResultsToLongTermStorage(getBaseStandoff(getBlankDataTableUsingPredictions()))
  }
}

runThePredictionsInALoop <- function() {
  progressBar <- progress_bar$new(total=20)
  progressBar$tick(0)
  
  fitControl <- trainControl(method = "cv")
  cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(cluster)
  
  #run with knn model, despite the fact there is only one variable
  for(i in 1:10) {
    fitForPlayerOneForLongTermThinking <- train(longTermGameStorage[,'playerOneValue'], longTermGameStorage[['playerOneTotalScore']], data=longTermGameStorage, method='knn', trControl = fitControl)
    fitForPlayerTwoForLongTermThinking <- train(longTermGameStorage[,'playerTwoValue'], longTermGameStorage[['playerTwoTotalScore']], data=longTermGameStorage, method='knn', trControl = fitControl)
    setPlayerMinMaxFromPredictions(longTermGameStorage, fitForPlayerOneForLongTermThinking, fitForPlayerTwoForLongTermThinking)
    
    addMinMaxObservation()
    
    addResultsToLongTermStorage(getBaseStandoff(getBlankDataTableUsingPredictions()))
    progressBar$tick()
  }
  
  stopCluster(cluster)
  
  #run the model with a simple linear model
  for(i in 1:10) {
    fitForPlayerOneForLongTermThinking <- lm(playerOneTotalScore ~ playerOneValue, data=longTermGameStorage)
    fitForPlayerTwoForLongTermThinking <- lm(playerTwoTotalScore ~ playerTwoValue, data=longTermGameStorage)
    setPlayerMinMaxFromPredictionsWithLinearModel(longTermGameStorage, fitForPlayerOneForLongTermThinking, fitForPlayerTwoForLongTermThinking)
    
    addMinMaxObservation()
    
    addResultsToLongTermStorage(getBaseStandoff(getBlankDataTableUsingPredictions()))
    progressBar$tick()
  }
}

setupDefaultData()
runThePredictionsInALoop()
