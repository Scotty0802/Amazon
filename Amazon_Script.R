#Load Libraries
library(tidymodels)
library(vroom)
library(dplyr)

#Import Data

train <- vroom("/yunity/rileyw02/Stat_348/train.csv")
test <- vroom("/yunity/rileyw02/Stat_348/test.csv")


# Recipe
amazon_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate(across(everything(), as.factor)) %>%
  step_other(all_nominal_predictors(), threshold = 0.001)%>%
  step_dummy(all_nominal_predictors())

amazon_prep <- prep(amazon_recipe)
amazon_train <- bake(amazon_prep, new_data = NULL)
amazon_test <- bake(amazon_prep, new_data = test)
#View(amazon_test)


#Models


logRegModel <- logistic_reg() %>%
  set_engine("glm")


#Workflow


wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = train)

#Predictions
amazon_predictions <- predict(wf,
                              new_data = test,
                              type= "prob") 


#Print
kaggle_submission <- amazon_predictions |>
  bind_cols(test) |> 
  select(id, .pred_1) |>
  rename(ACTION = .pred_1) 

## Write out the file
vroom_write(x = kaggle_submission,
            file = "/yunity/rileyw02/Stat_348/Kaggle_Submission.csv",
            delim = ",")

