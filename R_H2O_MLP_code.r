library(h2o)
h2o.init(nthresds = -1)

# # The following two commands remove any previously installed H2O packages for R.
# if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
# if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# 
# # Next, we download, install and initialize the H2O package for R.
# install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1497/R", getOption("repos"))))
library(h2o)
# localH2O = h2o.init()
h2o.init(startH2O = FALSE) 
# Finally, let's run a demo to see H2O at work.
demo(h2o.glm)
# Load Predictor variable
train_y = h2o.importFile("C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2011\\first_train_data.csv")
val_y = h2o.importFile("C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2011\\first_val_data.csv")
test_y = h2o.importFile("C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2011\\first_test_data.csv")
# test_2 = h2o.importFile("C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2011\\second_test_data.csv")
# Load Data
# train = h2o.importFile("C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2012\\R_data\\Train_data.csv")
# val = h2o.importFile("C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2012\\R_data\\Val_data.csv")
# test = h2o.importFile("C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2012\\R_data\\test_data.csv")

# # summary(train, exact_quantiles = TRUE)
# # nrow(test)
# train$y = train_y[,144]
# train = train[,-1]
# val$y = val_y[,144]
# val = val[,-1]
# test$y = test_y[,144]
# test = test[,-1]

x = colnames(train_y[1:143])
y = colnames(train_y[144])
train_y[,144] = as.factor(train_y[,144])
# model = h2o.deeplearning(x=x, y=y,training_frame = train_y, validation_frame = val_y,
#                          distribution = "bernoulli",
#                          activation = "Tanh" ,
#                          hidden = c(200,200),
#                          input_dropout_ratio = 0.2,
#                          hidden_dropout_ratios = c(0.4,0.4),
#                          l2 = 1e-2, epochs = 500)
# 
# summary(model)
# 
# pred = h2o.predict(model,test_y)
# pred$actual = test_y[,144]
# pred$accuracy = pred$actual
# for (i in 1:1981){
#   if (pred$predict[i,1] == 1 && pred$actual[i,1] ==1){
#     pred$accuracy[i,1]=1}
#   else{
#     pred$accuracy[i,1]=0}
#   print (i)
# }
# h2o.exportFile(pred$predict[,1],"C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2012\\pred1", force= FALSE)
# # sum(pred$accuracy)
# # sum(pred$predict)
# # sum(pred$actual)
# # sum(pred$accuracy)/sum(pred$predict)
# 
# #### Grid tuning
# hyper_params <- list(
#   hidden=list(c(100,100),c(200,200),c(50,100), c(150,150)),
#   input_dropout_ratio=c(0,0.2,0.4),
#   hidden_dropout_ratios = list(c(0.5,0.5), c(0,0),c(0.3,0.3)),
#   rate=c(0.001,0.01,0.1),
#   rate_annealing=c(1e-5,1e-4,1e-3),
#   epochs = c(10,25,50,75,100),
#   l1 = c(1e-5,1e-4,1e-3,0),
#   l2 = c(1e-5,1e-4,1e-3,0),
#   activation=c("Tanh", "TanhWithDropout","Maxout", "MaxoutWithDropout")
# )
# search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 1000, 
#                        max_models = 600, seed=1234567)
# hyper_params
# dl_random_grid1 <- h2o.grid(
#   algorithm="deeplearning",
#   grid_id="dl_grid_random", 
#   training_frame=train_y,
#   validation_frame=val_y, 
#   x=x, 
#   y=y,
#   distribution = "bernoulli",
#   stopping_metric="AUC",
#   stopping_tolerance=1e-3,        ## stop when misclassification does not improve by >=1% for 2 scoring events
#   # stopping_rounds=2,
#   #score_validation_samples=10000, ## downsample validation set for faster scoring
#   #score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
#   adaptive_rate=F,                ## manually tuned learning rate
#   momentum_start=0.5,             ## manually tuned momentum
#   momentum_stable=0.9, 
#   momentum_ramp=1e7, 
#   max_w2=10,                      ## can help improve stability for Rectifier
#   hyper_params=hyper_params,
#   search_criteria = search_criteria
# )
# dl_random_grid1
# write.csv(dl_random_grid1@summary_table,file = "C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2012\\variations_run5.csv")
#write.csv(grid@summary_table, file = "C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2012\\variations.csv")
# Kfold cross validation
# Model 1
dlmodel = h2o.deeplearning(
  training_frame=train_y,
  # validation_frame=val_y,
  x=x,
  y=y,
  distribution = "bernoulli",
  stopping_metric="logloss",
  stopping_tolerance=1e-3,        ## stop when misclassification does not improve by >=1% for 2 scoring events
  stopping_rounds=2,
  #score_validation_samples=10000, ## downsample validation set for faster scoring
  #score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  momentum_start=0.5,             ## manually tuned momentum
  momentum_stable=0.9,
  momentum_ramp=1e7,
  max_w2=500,
  nfolds = 5,
  fold_assignment = "Modulo",
  # stopping_rounds=5,
  hidden= c(150,40),
  input_dropout_ratio= 0,
  hidden_dropout_ratios =  c(0.3,0.3),
  rate=0.05,
  rate_annealing=1e-4,
  epochs = 40,
  l1 = 0,
  l2 = 1e-4,
  activation="Rectifier",
  variable_importances = TRUE
  )
h2o.varimp(dlmodel)

# Model 2
# dlmodel = h2o.deeplearning(
#   training_frame=train_y,
#   validation_frame=val_y, 
#   x=x, 
#   y=y,
#   distribution = "bernoulli",
#   stopping_metric="AUC",
#   stopping_tolerance=1e-3,        ## stop when misclassification does not improve by >=1% for 2 scoring events
#   stopping_rounds=2,
#   #score_validation_samples=10000, ## downsample validation set for faster scoring
#   #score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
#   adaptive_rate=F,                ## manually tuned learning rate
#   momentum_start=0.5,             ## manually tuned momentum
#   momentum_stable=0.9, 
#   momentum_ramp=1e7, 
#   max_w2=10,
#   nfolds = 5,
#   fold_assignment = "Modulo",
#   # stopping_rounds=5,
#   hidden= c(100,100),
#   input_dropout_ratio= 0,
#   hidden_dropout_ratios =  c(0.3,0.3),
#   rate=0.01,
#   rate_annealing=1e-4,
#   epochs = 50,
#   l1 = 0,
#   l2 = 1e-4,
#   activation="MaxoutWithDropout"
#   
# )


pred = h2o.predict(dlmodel,test_y)
pred$actual = test_y[,144]
pred$accuracy = pred$actual
for (i in 1:nrow(pred)){
  if (pred$predict[i,1] == 1 && pred$actual[i,1] ==1){
    pred$accuracy[i,1]=1}
  else{
    pred$accuracy[i,1]=0}
  # print (i)
}
sum(pred$accuracy)
sum(pred$predict)
h2o.exportFile(pred,"C:\\Users\\Administrator\\Desktop\\Input_FUT\\NN_model\\2011\\pred10", force= FALSE)
