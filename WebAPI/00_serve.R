library(plumber)
r = plumb("01_Regression.R")
r$run(port = 1030, host = "0.0.0.0")
