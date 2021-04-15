library(fastDummies)
library(mlr3)
library(mlr3tuning)
library(mlr3verse)
library(mlr3viz)
library(readr)
library(summarytools)

df <- read_csv("../Data/cardataset.csv")

View(df)
descr(df)

hist(df$MSRP)
hist(log(df$MSRP))

df$logMSRP <- df$MSRP
df$MSRP <- NULL
df <- dummy_cols(df, remove_selected_columns = TRUE)
df <- na.omit(df)
colnames(df) <- make.names(colnames(df),unique = T)

# Task 
task_reg = TaskRegr$new(id="Car MSRP Prediction", 
                        backend = df, 
                        target = "logMSRP")
print(task_reg)
autoplot(task_reg)

# Learners
learners = c("regr.featureless",
             "regr.glmnet",
             "regr.ranger",
             "regr.rpart",
             "regr.xgboost")
learners = lapply(learners, lrn, 
                  predict_type = "response", 
                  predict_sets = c("train", "test", "validation"))

# Resample
resamplings = rsmp("cv", folds = 5)

# Benchmark
lgr::get_logger("mlr3")$set_threshold("trace")

design = benchmark_grid(task_reg, learners, resamplings)
print(design)

bmr = benchmark(design)

# Measures
measures = list(
  msr("regr.rmse", id="rmse_tr", predict_sets = "train"),
  msr("regr.rmse", id="rmse_ts")
)

# Performance values
aggr = bmr$aggregate(measures)
print(aggr)

# Fine-tuning GLMNET
learner = lrn("regr.glmnet", predict_type="response", predict_sets = c("train", "test", "validation"))
learner$param_set$values$alpha = to_tune(0, 1)

at = auto_tuner(
  method = "random_search",
  learner = learner, 
  resampling = resamplings,
  measure = msr("regr.rmse"),
  # use term_evals for real cases
  # term_evals = 100, 
  term_time = 60
)

at$train(task_reg)

# Train with best params
learner = lrn("regr.glmnet", predict_type="response", predict_sets = c("train", "test", "validation"))
learner$param_set$values$alpha = at$archive$best()$alpha
learner$train(task_reg)
