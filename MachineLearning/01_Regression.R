library(dplyr)
library(mlr3)
library(mlr3tuning)
library(mlr3verse)
library(mlr3viz)
library(readr)
library(summarytools)

df <- read_csv("../Data/cardataset.csv")
df <- df %>% mutate_if(is.character, factor)

View(df)
descr(df)

hist(df$MSRP)
hist(log(df$MSRP))

df$logMSRP <- df$MSRP
df$MSRP <- NULL
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
             "regr.ranger",
             "regr.rpart")
learners = lapply(learners, lrn, 
                  predict_type = "response", 
                  predict_sets = c("train", "test"))

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
learner = lrn("regr.ranger", predict_type="response", predict_sets = c("train", "test"))
learner$param_set$values$min.node.size = to_tune(1, 10)

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
learner = lrn("regr.ranger", predict_type="response", predict_sets = c("train", "test"))
learner$param_set$values$min.node.size = at$archive$best()$min.node.size
learner$train(task_reg)

# Plot
object = learner$train(task_reg)$predict(task_reg)
autoplot(object, type="xy")
autoplot(object, type="residual")

# Save for serve with Plumber API
saveRDS(learner, "../Models/01_Regression.rds")

feature_info = list(
  feature_names = task_reg$feature_names,
  feature_types = task_reg$feature_types,
  levels = task_reg$levels()
)

saveRDS(feature_info, "../Models/01_Regression_feature_info.rds")
