#####Zanotti####
calibrate_evaluate_plot <- function(
  model_fit, 
  y, 
  mode, 
  type = "testing", 
  print = TRUE
) {
  
  if (type == "testing") {
    new_data <- testing(split)
  } else {
    new_data <- training(split)
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

###Used libraries####
library(readxl)
library(readr)
library(dplyr)
library(magrittr)
library(ggpubr)
library(car)
library(rsample)
library(dplyr)
library(bestglm)
library(MASS)
library(xgboost)
library(ranger)

##importing and fixing the dataset####
####Data importing#####
rw = read.table("C:\\Users\\vitto\\Documents\\UNI\\1m\\StatLearning\\GW\\winequality-red.csv",sep=",",header=T)
ww = read.table("C:\\Users\\vitto\\Documents\\UNI\\1m\\StatLearning\\GW\\winequality-white.csv",sep=";",header=T)
#wfull %>% 
# DataExplorer::create_report()
rw['type'] <- c("red")
ww['type'] <- c("white")
wdb <- rbind(ww,rw)
wdb<-as.data.frame(wdb)
summary(wdb)
##PREPROCESSING####
wdb <- wdb %>%
  mutate(
    quality = ifelse(quality<=5, "low", "high"),
    quality = factor(quality)
  ) %>%
  mutate(across(where(is.character), as.factor))
wdb <- wdb[!duplicated(wdb),]

###Train/test split####
set.seed(42)
split <- initial_split(wdb, strata=type, prop = .8)
folds<-vfold_cv(training(split))


#MODELLING-------------------------------------------
###Baseline recipe####
rcp_spec <- recipe(quality ~ ., data = training(split))%>%
  step_dummy(type)
rcp_spec %>% prep() %>% juice() %>% glimpse()

#TREES####-------------------------------------------

treemod <- decision_tree(      #tree
  mode = "classification",
  tree_depth = tune(),
  min_n = tune()
) %>% set_engine('rpart')


model_spec_bag <- bag_tree(        #bagging
  mode = "classification",
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart",	times = 25) 

model_spec_rf <- rand_forest(         #random forest
  mode = "classification",
  mtry = 3, #sqrt delle features
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger") 


model_spec_xgb <- boost_tree(          #extreme gradient boosting
  mode = "classification",
  mtry = tune(),
  trees = 1000,
  min_n = 2,
  tree_depth = tune(),
  learn_rate = 0.0005
) %>%
  set_engine("xgboost")

###Workflows------------------------------------------

wrkfl_tree <- workflow() %>%       #tree
  add_model(treemod) %>%
  add_recipe(rcp_spec)

wrkfl_bag <- workflow() %>%    #bagging
  add_model(model_spec_bag) %>%
  add_recipe(rcp_spec)

wrkfl_rf <- workflow() %>%      #rf
  add_model(model_spec_rf) %>%
  add_recipe(rcp_spec)

wrkfl_xgb <- workflow() %>%     #boost
  add_model(model_spec_xgb) %>%
  add_recipe(rcp_spec)

###Grid Search------------------------------------------
set.seed(42)                #tree
tree_grid <- grid_regular(
  min_n(),
  tree_depth(range = c(1,29)),
  levels = 5
)

set.seed(42)                     #bagging
bag_grid <- grid_regular(
  min_n(),
  tree_depth(),
  cost_complexity(),
  levels = 2)

set.seed(42)               #rf
rf_grid <- grid_regular(
  min_n(),
  trees(),
  levels = 2
)  

set.seed(42)              #boosting
xgb_grid <- grid_regular(
  mtry(range = c(1,9)),
  tree_depth(),
  levels = 2
)


###Tuning-------------------------------------------

set.seed(42)                #tree
tree_res <- wrkfl_tree %>% 
  tune_grid(
    resamples = folds,
    grid = tree_grid
  )

set.seed(42)                #bagging

bag_res <- wrkfl_bag %>% 
  tune_grid(
    resamples = folds,
    grid = bag_grid
  )

set.seed(42)                    #rf
rf_res <- wrkfl_rf %>% 
  tune_grid(
    resamples = folds,
    grid = rf_grid
  )

set.seed(42)                #boosting      
xgb_res <- wrkfl_xgb %>% 
  tune_grid(
    resamples = folds,
    grid = xgb_grid
  )


###Evaluation-----------------------------------------

tree_res %>%  collect_metrics()
tree_best <- tree_res %>% select_best("roc_auc")
tree_best

bag_res %>%  collect_metrics()
bag_best <- bag_res %>% select_best("roc_auc")
bag_best

rf_res %>%  collect_metrics()
rf_best <- rf_res %>% select_best("roc_auc")
rf_best

xgb_res %>%  collect_metrics()
xgb_best <- xgb_res %>% select_best("roc_auc")
xgb_best


##Refit---------------------------------------------------------
wrkfl_tree_final <-	wrkfl_tree %>%	
  finalize_workflow(tree_best) %>% 
  last_fit(split)

wrkfl_tree_final %>%	collect_metrics()
wrkfl_tree_final %>% collect_predictions()

wrkfl_bag_final <-	wrkfl_bag %>%	
  finalize_workflow(bag_best) %>% 
  last_fit(split)

wrkfl_bag_final %>%	collect_metrics()
wrkfl_bag_final %>% collect_predictions()

wrkfl_rf_final <-	wrkfl_rf %>%	
  finalize_workflow(rf_best) %>% 
  last_fit(split)

wrkfl_rf_final %>%	collect_metrics()
wrkfl_rf_final %>% collect_predictions()

wrkfl_xgb_final <-	wrkfl_xgb %>%	
  finalize_workflow(xgb_best) %>% 
  last_fit(split)

wrkfl_xgb_final %>%	collect_metrics()
wrkfl_xgb_final %>% collect_predictions()

### Finding out more about the best model ---------------------------------------------------------------------
wrkfl_fit_tree %>%
  extract_fit_engine() %>%
  rpart.plot::rpart.plot(roundint = FALSE,5) 

wrkfl_fit_tree %>% 
  extract_fit_parsnip() %>% 
  vip::vip()
