library(tidyverse)
library(tidymodels)
library(car)
library(npreg)
library(mgcv)
library(leaps)
library(factoextra)
library(pls)
library(caret)
library(MLmetrics)
library(ROCR)
library(gam)
library(plsmod)
library(AppliedPredictiveModeling)


# Import and removal of unnecessary column
df <- read.csv(file = "./full_wine.csv", stringsAsFactors = TRUE)
df <- subset(df, select = -c(Type, quality, X))

# Split data in train-test

splits <- initial_split(df, prop = .75)

data_train <- training(splits)
data_test <- testing(splits)

folds <- vfold_cv(data_train, v=5)

# ----------------------------------------------
# Logistic Regression 

recipe <- recipe(QtyBin ~ . , data = training(splits)) %>% 
      step_normalize(all_predictors(), 
                     -all_nominal()) %>%
      step_dummy(all_nominal(), 
                 -all_outcomes()) %>%
      step_nzv(all_predictors())
  
  
logreg <- logistic_reg(
  mode = "classification",
  engine = "glm",
  penalty = NULL,
  mixture = NULL
)


logreg_fit <- workflow() %>%
  add_model(logreg) %>%
  add_recipe(recipe) %>% 
  fit(training(splits))

logreg_fit

logreg_fit %>% 
  calibrate_evaluate_plot(y = "QtyBin", type = "testing", mode = "classification")

logreg_last_fit <- logreg_fit %>% 
  last_fit(splits)

# Get predictions to compute various metrics
logreg_preds <- logreg_last_fit %>% collect_predictions()# %>% select(.pred_class) 


logreg_CM <- confusionMatrix(data=logreg_preds$.pred_class, reference = data_test$QtyBin)
f1_logreg <- F1_Score(logreg_preds$.pred_class, data_test$QtyBin)


#PCA Tidymodels
  
recipe_pca <- recipe(QtyBin ~ ., data = training(splits)) %>% 
                    step_normalize(all_predictors(), 
                                   -all_nominal()) %>%
                    step_dummy(all_nominal(), 
                               -all_outcomes()) %>%
                    step_pca(all_predictors()) %>%
                    step_nzv(all_predictors()) %>%
                    step_naomit(all_predictors())


pca_fit <- workflow() %>%
  add_model(logreg) %>%
  add_recipe(recipe_pca) %>% 
  fit(training(splits))

pca_fit %>% 
  calibrate_evaluate_plot(y = "QtyBin", type = "testing", mode = "classification")

pca_last_fit <- pca_fit %>% 
  last_fit(splits)

# Get predictions to compute various metrics
pca_preds <- pca_last_fit %>% collect_predictions()# %>% select(.pred_class) 


pca_CM <- confusionMatrix(data=pca_preds$.pred_class, reference = data_test$QtyBin)
f1_pca <- F1_Score(pca_preds$.pred_class, data_test$QtyBin)

                

# Partial Least Squares
library(pls)

df$QtyBin <- ifelse(df$QtyBin == "High", 1, 0)  
splits <- initial_split(df, prop = .75)

data_train <- training(splits)
data_test <- testing(splits)

folds <- vfold_cv(data_train, v=5)
  
pls_r <- plsr(QtyBin~. , data=data_train, scale=TRUE, family="binomial", validation="CV")
summary(pls_r)

prob_preds_pls <- predict(pls_r, data_test, ncomp=8)

preds_plsr <- ifelse(prob_preds_pls > 0.5, "High", "Low")
preds_plsr <- factor(preds_plsr)
data_test$QtyBin <- ifelse(data_test$QtyBin == 1, "High", "Low")
data_test$QtyBin <- factor(data_test$QtyBin)

pls_CM <- confusionMatrix(data=preds_plsr, reference = data_test$QtyBin)
f1_pls <- F1_Score(preds_plsr,data_test$QtyBin)


prob_pls <- list(prob_preds_pls)
pred_pls <- prediction(prob_pls, data_test$QtyBin)
perf_pls = performance(pred_pls, "tpr","fpr")
plot(perf_pls)

# -----------------------------------------
  
# GAM
gam4 <- gam(QtyBin ~ s(fixed.acidity) + 
                s(volatile.acidity) + 
                s(citric.acid) + 
                s(residual.sugar) + 
                s(chlorides)+
                s(free.sulfur.dioxide)+
                s(total.sulfur.dioxide)+
                s(density)+
                s(pH)+
                s(sulphates)+
                s(alcohol), data = data_train, 
              family = binomial(link="logit"))  

gam4_pred <- predict(gam4, data_test, type="response")

pred4 <- prediction(gam4_pred, data_test$QtyBin)
perf4 = performance(pred4, "tpr","fpr")
plot(perf4)


gam4_pred_bin <- ifelse(gam4_pred > 0.5,"High","Low")
gam4_pred_bin <- factor(gam4_pred_bin)


CM <- confusionMatrix(data=gam4_pred_bin, reference = data_test$QtyBin)
f1_gam <- F1_Score(gam4_pred_bin,data_test$QtyBin)

































#####Code from lab classes for plotting and evaluation matrics - By M. Zanotti ####
calibrate_evaluate_plot <- function(
  model_fit, 
  y, 
  mode, 
  type = "testing", 
  print = TRUE
) {
  
  if (type == "testing") {
    new_data <- testing(splits)
  } else {
    new_data <- training(splits)
  }
  
  if (mode == "regression") {
    pred_res <- model_fit %>% 
      augment(new_data) %>%
      dplyr::select(all_of(y), .pred) %>% 
      set_names(c("Actual", "Pred")) %>% 
      # bind_cols(
      # 	model_fit %>% 
      # 		predict(new_data, type = "conf_int") %>%
      # 		set_names(c("Lower", "Upper")) 
      # ) %>% 
      add_column("Type" = type)
  } else {
    pred_res <- model_fit %>% 
      augment(new_data) %>%
      dplyr::select(all_of(y), contains(".pred")) %>% 
      set_names(c("Actual", "Pred", "Prob_Low", "Prob_High")) %>% 
      add_column("Type" = type)
  }
  
  pred_met <- pred_res %>% evaluate_model(mode, type) 
  
  if (mode == "regression") {
    pred_plot <- pred_res %>% plot_model(mode)
  } else {
    pred_plot <- pred_met %>% plot_model(mode)
  }
  
  if (print) {
    print(pred_met)
    print(pred_plot)	
  }
  
  res <- list(
    "pred_results" = pred_res,
    "pred_metrics" = pred_met
  )
  
  return(invisible(res))
  
}


evaluate_model <- function(prediction_results, mode, type) {
  
  if (mode == "regression") {
    res <- prediction_results %>% 
      metrics(truth = Actual, estimate = Pred) %>% 
      dplyr::select(.metric, .estimate) %>% 
      set_names(c("Metric", "Estimate")) %>% 
      add_column("Type" = type)
  } else {
    res_confmat <- prediction_results %>%
      conf_mat(truth = Actual, estimate = Pred)
    res_metrics <- bind_rows(
      prediction_results %>% metrics(truth = Actual, estimate = Pred),
      prediction_results %>% roc_auc(truth = Actual, estimate = Prob_Low)
    ) %>% 
      dplyr::select(.metric, .estimate) %>% 
      set_names(c("Metric", "Estimate")) %>% 
      add_column("Type" = type)
    res_roc <- prediction_results %>%
      roc_curve(truth = Actual, Prob_Low) %>% 
      add_column("Type" = type)
    res <- list("confusion" = res_confmat$table, "metrics" = res_metrics, "roc" = res_roc)
  }
  
  return(res)
  
}

plot_model <- function(prediction_results, mode) {
  
  if (mode == "regression") {
    p <- prediction_results %>% 
      dplyr::select(-Type) %>% 
      mutate(id = 1:n()) %>% 
      ggplot(aes(x = id)) +
      geom_point(aes(y = Actual, col = "Actual")) +
      geom_point(aes(y = Pred, col = "Pred")) +
      # geom_errorbar(aes(ymin = Lower, ymax = Upper), width = .2, col = "red") +
      scale_color_manual(values = c("black", "red")) +
      labs(x = "", y = "", col = "") +
      theme_minimal()
  } else {
    p <- prediction_results$roc %>%
      ggplot(aes(x = 1 - specificity, y = sensitivity)) +
      geom_path() +
      geom_abline(lty = 3) +
      coord_equal() + 
      theme_minimal() 
  }
  
  res <- plotly::ggplotly(p)
  return(res)
  
}