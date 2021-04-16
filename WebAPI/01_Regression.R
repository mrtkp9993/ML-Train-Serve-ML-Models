library(data.table)
library(dplyr)
library(jsonlite)
library(mlr3)
library(readr)

source("00_fix_feature_types.R")

learner = readRDS("../Models/01_Regression.rds")
feature_info = readRDS("../Models/01_Regression_feature_info.rds")

#* @get /feature_list
function(req) {
  data.frame(features=feature_info$feature_names, 
             types=feature_info$feature_types)
}

#* @get /feature_levels
function(req) {
  feature_info$levels
}

#* @post /predict_msrp
function(req) {
  newdata = fromJSON(req$postBody, simplifyVector = FALSE)
  newdata = as.data.table(newdata, keep.rownames = TRUE)
  #newdata = rbindlist(newdata, use.names = TRUE)
  newdata[, colnames(newdata) := mlr3misc::pmap(
    list(.SD, colnames(newdata)),
    fix_feature_types,
    feature_info = feature_info
  )]
  as.data.table(learner$predict_newdata(newdata))
}
