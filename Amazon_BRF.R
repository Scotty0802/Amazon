library(tidymodels)
library(vroom)
library(embed)
library(ranger)

# Import Data


train <- vroom("/yunity/rileyw02/Stat_348/train.csv")
test <- vroom("/yunity/rileyw02/Stat_348/test.csv")


# Recipe


train <- train %>%
  mutate(ACTION = as.factor(ACTION))

amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

amazon_prep <- prep(amazon_recipe)
amazon_train <- bake(amazon_prep, new_data = NULL)
amazon_test <- bake(amazon_prep, new_data = test)
# 


#Models


model <- rand_forest(
  mtry = tune(),      # number of predictors randomly sampled at each split
  min_n = tune(),     # minimal node size
  trees = 1000         # number of trees; can increase to 1000
) %>%
  set_engine("ranger") %>%
  set_mode("classification")



#Workflow


wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(model)



#Tune


tune_grid <- grid_regular(
  mtry(range = c(1, 9)),     # adjust according to number of predictors
  min_n(range = c(1, 9)),
  levels = 9)

folds <- vfold_cv(train, v = 10)

## Run the CV
CV_results <- wf %>%
  tune_grid(
    resamples = folds,
    grid = tune_grid,
    metrics = metric_set(roc_auc, accuracy)
  )



# Best model

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")


#Predictions


final_wf <- wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

amazon_predictions <- predict(final_wf,
                              new_data = test,
                              type= "prob") 


#Print


kaggle_submission <- amazon_predictions |>
  bind_cols(test) |> 
  select(id, .pred_1) |>
  rename(ACTION = .pred_1) 

## Write out the file
vroom_write(x = kaggle_submission,
            file = "/yunity/rileyw02/Stat_348/Kaggle_Submission_BRF.csv",
            delim = ",")