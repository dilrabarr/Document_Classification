# loading the data
hamilton.train=read.directory('fp_hamilton_train_clean')
hamilton.test=read.directory('fp_hamilton_test_clean')
madison.train=read.directory('fp_madison_train_clean')
madison.test=read.directory('fp_madison_test_clean')



combined1=c(hamilton.train,hamilton.test)
combined2=c(madison.train,madison.test)
combined_total=c(combined1,combined2)
dictionary=make.sorted.dictionary.df(combined_total)

# creating grain and test data

dtm.hamilton.train=make.document.term.matrix(hamilton.train,dictionary)
dtm.hamilton.test=make.document.term.matrix(hamilton.train,dictionary)
dtm.madison.train=make.document.term.matrix(madison.train,dictionary)
dtm.madison.test=make.document.term.matrix(madison.test,dictionary)

#Step Five

mu_q1=1/nrow(dictionary)

logp.hamilton.train <- make.log.pvec(dtm.hamilton.train, mu_q1)
logp.hamilton.test <- make.log.pvec(dtm.hamilton.test, mu_q1)
logp.madison.train <- make.log.pvec(dtm.madison.train, mu_q1)
logp.madison.test <- make.log.pvec(dtm.madison.test, mu_q1)




logPriorHamiltonTrain_q2 <- log(nrow(dtm.hamilton.train)/(nrow(dtm.hamilton.train)+nrow(dtm.madison.train)))
logPriorMadisonTrain_q2 <- log(nrow(dtm.madison.train)/(nrow(dtm.hamilton.train)+nrow(dtm.madison.train)))

# Naive Bayes function itself
naive.bayes = function(logp.hamilton.train, logp.madison.train, log.prior.hamilton, log.prior.madison , dtm.test){


log.post.hamilton <- log.prior.hamilton + (dtm.test %*% logp.hamilton.train)
log.post.madison <- log.prior.madison + (dtm.test %*% logp.madison.train)


prediction <- data.frame(logPostHam=log.post.hamilton,
                           logPostMad=log.post.madison)
prediction$pred <- (log.post.hamilton >= log.post.madison)
prediction$pred <- gsub(TRUE, "Hamilton", prediction$pred)
prediction$pred <- gsub(FALSE, "Madison", prediction$pred)


# return a vector of the predictions
return(prediction$pred)
}

install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)

# train the model
predictions <- naive.bayes(logp.hamilton.train,
                           logp.madison.train,
                           logPriorHamiltonTrain,
                           logPriorMadisonTrain,
                           rbind(dtm.hamilton.test, dtm.madison.test))


# make predictions
predictions <- data.frame(trueValue=c(rep("Hamilton", nrow(dtm.hamilton.test)),
                                      rep("Madison", nrow(dtm.madison.test))),
                          pred=predictions)
# evaluate the result 
confusionMatrix(data=predictions$pred,
                reference=predictions$trueValue,
                dnn=c("Prediction", "True Value"),
                positive="Hamilton")
