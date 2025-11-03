
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(doParallel)

parallel::detectCores() 
cl <- makePSOCKcluster(50)
registerDoParallel(cl)


# Import Data


train <- vroom("/yunity/rileyw02/Stat_348/train.csv")
test <- vroom("/yunity/rileyw02/Stat_348/test.csv")


# Recipe


train <- train %>%
  mutate(ACTION = as.factor(ACTION))

amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), threshold = .6)

amazon_prep <- prep(amazon_recipe)
amazon_train <- bake(amazon_prep, new_data = NULL)
amazon_test <- bake(amazon_prep, new_data = test)


# Polynomial


svmPoly <- svm_poly(degree = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab", kpar = list(maxiter = 800))

wf_Poly <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(svmPoly)

tune_grid_poly <- grid_regular(
  degree(range = c(1, 5)),
  cost(range = c(-3, 3), trans = log10_trans()),
  levels = 3)

folds <- vfold_cv(train, v = 3)

CV_results_poly <- wf_Poly %>%
  tune_grid(
    resamples = folds,
    grid = tune_grid_poly,
    metrics = metric_set(roc_auc, accuracy),
    control = control_grid(verbose = TRUE))

bestTune_poly <- CV_results_poly %>%
  select_best(metric = "roc_auc")

final_wf_poly <- wf_Poly %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

amazon_predictions_p <- predict(final_wf_poly,
                                new_data = test,
                                type= "prob") 

kaggle_submission_p <- amazon_predictions_p |>
  bind_cols(test) |> 
  select(id, .pred_1) |>
  rename(ACTION = .pred_1) 

vroom_write(x = kaggle_submission_p,
            file = "/yunity/rileyw02/Stat_348/SVM/Kaggle_Submission_Poly.csv",
            delim = ",")


# Radial

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab", kpar = list(maxiter = 800))

wf_Rad <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(svmRadial)

tune_grid_rad <- grid_regular(
  rbf_sigma(range = c(-3, 0), trans = log10_trans()), 
  cost(range = c(-3, 3), trans = log10_trans()),
  levels = 3)

folds <- vfold_cv(train, v = 3)

CV_results_rad <- wf_Rad %>%
  tune_grid(
    resamples = folds,
    grid = tune_grid_rad,
    metrics = metric_set(roc_auc, accuracy),
    control = control_grid(verbose = TRUE)
  )

bestTune_rad <- CV_results_rad %>%
  select_best(metric = "roc_auc")

final_wf_rad <- wf_Rad %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

amazon_predictions_r <- predict(final_wf_rad,
                                new_data = test,
                                type= "prob") 

kaggle_submission_r <- amazon_predictions_r |>
  bind_cols(test) |> 
  select(id, .pred_1) |>
  rename(ACTION = .pred_1) 

vroom_write(x = kaggle_submission_r,
            file = "/yunity/rileyw02/Stat_348/SVM/Kaggle_Submission_Rad.csv",
            delim = ",")


svmLinear <- svm_linear(cost=tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

wf_Lin <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(svmLinear)

tune_grid_lin <- grid_regular(
  cost(range = c(-3, 3), trans = log10_trans()),
  levels = 3)


folds <- vfold_cv(train, v = 3)

CV_results_lin <- wf_Lin %>%
  tune_grid(
    resamples = folds,
    grid = tune_grid_lin,
    metrics = metric_set(roc_auc, accuracy),
    control = control_grid(verbose = TRUE)
  )

bestTune_lin <- CV_results_lin %>%
  select_best(metric = "roc_auc")

final_wf_lin <- wf_Lin %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

amazon_predictions_l <- predict(final_wf_lin,
                                new_data = test,
                                type= "prob") 

kaggle_submission_l <- amazon_predictions_l |>
  bind_cols(test) |> 
  select(id, .pred_1) |>
  rename(ACTION = .pred_1) 

vroom_write(x = kaggle_submission_l,
            file = "/yunity/rileyw02/Stat_348/SVM/Kaggle_Submission_linear.csv",
            delim = ",")

stopCluster(cl)

