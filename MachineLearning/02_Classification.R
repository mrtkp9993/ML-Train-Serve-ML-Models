library(dplyr)
library(mlr3)
library(mlr3tuning)
library(mlr3verse)
library(mlr3viz)
library(readr)
library(summarytools)

df <- read_csv("../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df <- df %>% mutate_if(is.character, factor)
df$customerID <- NULL
df <- na.omit(df)

View(df)
descr(df)
freq(df)

colnames(df) <- make.names(colnames(df),unique = T)

# Task 
task_class = TaskClassif$new(id="Churn Prediction", 
                             backend = df, 
                             target = "Churn")
print(task_class)
autoplot(task_class)

# Learners
learners = c("classif.featureless",
             "classif.lda",
             "classif.naive_bayes",
             "classif.nnet",
             "classif.ranger",
             "classif.rpart")
learners = lapply(learners, lrn, 
                  predict_type = "response", 
                  predict_sets = c("train", "test"))

# Resample
resamplings = rsmp("cv", folds = 5)

# Benchmark
lgr::get_logger("mlr3")$set_threshold("trace")

design = benchmark_grid(task_class, learners, resamplings)
print(design)

bmr = benchmark(design)

# Measures
measures = list(
  msr("classif.fbeta", id="f1_tr", predict_sets = "train"),
  msr("classif.fbeta", id="f1_ts")
)

# Performance values
aggr = bmr$aggregate(measures)
print(aggr)

# Fine-tuning GLMNET
learner = lrn("classif.ranger", predict_type="response", predict_sets = c("train", "test"))
learner$param_set$values$min.node.size = to_tune(1, 10)

at = auto_tuner(
  method = "random_search",
  learner = learner, 
  resampling = resamplings,
  measure = msr("classif.fbeta"),
  # use term_evals for real cases
  # term_evals = 100, 
  term_time = 60
)

at$train(task_class)

# Train with best params
learner = lrn("classif.ranger", predict_type="response", predict_sets = c("train", "test"))
learner$param_set$values$min.node.size = at$archive$best()$min.node.size
learner$train(task_class)

# Plot
object = learner$train(task_class)$predict(task_class)
autoplot(object, type="stacked")

# Save for serve with Plumber API
saveRDS(learner, "../Models/02_Classification.rds")

feature_info = list(
  feature_names = task_class$feature_names,
  feature_types = task_class$feature_types,
  levels = task_class$levels()
)

saveRDS(feature_info, "../Models/02_Classification_feature_info.rds")
