library(maditr)

getMainFolderName <- function(mainFolder)
{
  mainFolderName <- unlist(strsplit(mainFolder, split="/"))
  mainFolderShort <- mainFolderName[length(mainFolderName)]
  mainFolderShort
}

getModelName <- function(mainFolder)
{
  # 1.a) get model's name
  resultsFolder <- file.path(mainFolder,"results")
  modelName <- list.dirs(path = resultsFolder,full.names = FALSE,recursive = FALSE)
  
  # modelId <- grep("Random", modelName)
  # modelName <- modelName[modelId]
  modelName
}

getAUC <- function(true_value, x1_probability)
{
  if(length(unique(true_value))> 1)
  {
      roc_obj <- roc(true_value, x1_probability)
      AUC <- auc(roc_obj) #0.5744
  }
  else
  {
    AUC <- NA
  }
  AUC
}

getAccuracy <- function(true_value, predicted_value)
{
  accuracy <- sum(true_value == predicted_value) / length(true_value)
  accuracy
}


getMape <- function(mainFolder, modelName)
{
  predictionFolder <- file.path(mainFolder,"predictions",modelName)
  allFiles <- list.files(path = predictionFolder, full.names = FALSE,recursive = FALSE)
    
  dfOutcome <- data.frame()
  for(resultsFile in allFiles)
  {
      df <- read_csv(file.path(predictionFolder,resultsFile))
      MAPE <- MLmetrics::MAPE(df$Predicted_value, df$True_value)
      dfOutcome <-  rbind(dfOutcome, MAPE)
  }
  
  colnames(dfOutcome) <- "MAPE"
  dfOutcome
  }

getRegressionResults <- function(mainFolder, modelName)
{
  modelFolder <- file.path(mainFolder,"results",modelName)
  
  allResults <- list.files(path = modelFolder, full.names = FALSE,recursive = FALSE)
  
  dfOutput <- data.frame()
  
  for(resultsFile in allResults)
  {
    df <- read_csv(file.path(modelFolder,resultsFile))
    dfOutput <- rbind(dfOutput,df)
  }
  
  dfOutput
}

getGridsearchResults <- function(mainFolder, modelName)
{
  modelFolder <- file.path(mainFolder,"results_per_fold",modelName)
  
  allFolds <- list.dirs(path = modelFolder, full.names = FALSE,recursive = FALSE)
  
  dfOutput <- data.frame()
  
  foldID <- 1
  for(fold in allFolds)
  {
    foldResult <- list.files(file.path(modelFolder,fold), full.names = TRUE)
    df <- read_csv(foldResult)
    df <- cbind(foldID, df)
    dfOutput <- rbind(dfOutput,df)
    foldID <- foldID + 1
  }
  
  dfOutput
}

getClassificationResults <- function(mainFolder, modelName)
{
  resultsTag <- "OverallResults"

  mainFolderName <- getMainFolderName(mainFolder)
  
  modelFolder <- file.path(mainFolder,"results",modelName)
  
  allResults <- list.files(path = modelFolder, full.names = FALSE,recursive = FALSE)
  
  dfOutput <- data.frame()
  
  for(resultsFile in allResults)
  {
    df <- read_csv(file.path(modelFolder,resultsFile))
    dfOutput <- rbind(dfOutput,df)
  }
  
  dfOutput
}

getModelParams <- function(mainFolder, modelName)
{
  hyperparametersFolder <- file.path(mainFolder,"hyperparameters",modelName)
  hyperparametersSubfolder <- list.dirs(path = hyperparametersFolder,full.names = FALSE,recursive = FALSE)
  
  hpfolder <- file.path(hyperparametersFolder,hyperparametersSubfolder[1],"1")
  hpFile <- list.files(hpfolder, full.names = TRUE)

  tryCatch(
    expr = {
      lineMaxDepth <- grep("MODEL__max_depth", readLines(hpFile), value = TRUE)
      param_maxDepth <- unlist(strsplit(lineMaxDepth, split=": "))[2]
      
      if(length(param_maxDepth) == 0)
      {
        param_maxDepth <- NA
      }
    },
    error = function(e){ 
      param_maxDepth <- NA
    })
    
  
  tryCatch(
    expr = {
      lineNEstimators <- grep("MODEL__n_estimators", readLines(hpFile), value = TRUE)
      param_NEstimators <- unlist(strsplit(lineNEstimators, split=": "))[2]
      
      if(length(param_NEstimators) == 0)
      {
        param_NEstimators <- NA
      }
    },
    error = function(e){ 
      param_NEstimators <- NA
    })
  
  params <- data.frame(param_maxDepth, param_NEstimators)
  params
}

getGridsearchModelParams <- function(mainFolder, modelName)
{
  hyperparametersFolder <- file.path(mainFolder,"hyperparameters",modelName)
  hyperparametersSubfolder <- list.dirs(path = hyperparametersFolder,full.names = TRUE,recursive = FALSE)
  
  folds <- list.dirs(path = hyperparametersSubfolder, recursive = FALSE)
  foldID <- 1
  dfParams <- data.frame()
  
  for(fold in folds)
  {
      hpFile <- list.files(fold, full.names = TRUE)
      
        tryCatch(
          expr = {
            lineMaxDepth <- grep("MODEL__max_depth", readLines(hpFile), value = TRUE)
            param_maxDepth <- unlist(strsplit(lineMaxDepth, split=": "))[2]
            
            if(length(param_maxDepth) == 0)
            {
              param_maxDepth <- NA
            }
          },
          error = function(e){ 
            param_maxDepth <- NA
          })
    
  
      tryCatch(
        expr = {
          lineNEstimators <- grep("MODEL__n_estimators", readLines(hpFile), value = TRUE)
          param_NEstimators <- unlist(strsplit(lineNEstimators, split=": "))[2]
          
          if(length(param_NEstimators) == 0)
          {
            param_NEstimators <- NA
          }
        },
        error = function(e){ 
          param_NEstimators <- NA
        })
  
      params <- data.frame(foldID,param_maxDepth, param_NEstimators)
      dfParams <- rbind(dfParams, params)
      foldID <- foldID + 1
  }

  dfParams
}

getNumberClasses <- function(mainFolder, modelName)
{
  predictionFolder <- file.path(mainFolder,"predictions",modelName)
  
  allFiles <- list.files(path = predictionFolder, full.names = FALSE,recursive = FALSE)
    
  df <- read_csv(file.path(predictionFolder,allFiles[3]))
  
  numberClasses <- length(unique(df$True_value))    
  numberClasses
}

getSubjectsCount <- function(originalFeatures)
{
  df <- read_csv(originalFeatures)
  
  subjectCount <- length(unique(df$snapshot_id))
  subjectCount
}

getOverallRegressions <- function(mainFolder, originalFeatures)
{
  resultsTag <- "OverallResults"

  mainFolderName <- getMainFolderName(mainFolder)
  
  modelName <- getModelName(mainFolder)
  
  subjectsCount <- getSubjectsCount(originalFeatures)
      
  for(i in seq(1,length(modelName)))
    {
          # 1. get overall results
      dfOverallResults <- getRegressionResults(mainFolder, modelName[i])
      
      # 2. get the model hyperparameters
      dfParams <- getModelParams(mainFolder, modelName[i])
      
      # 3. get the number of classes
      numberClasses <- getNumberClasses(mainFolder, modelName[i])
      
      # 4. get the overall MAPE
      MAPEs <- getMape(mainFolder, modelName[i])
      
      # 4. Get the overall summary
      dfOutcome <- cbind(resultsTag,mainFolderName, dfOverallResults, subjectsCount, numberClasses, dfParams[1,1], dfParams[1,2], MAPEs)
    
      # 5. write to the excel file
      if(file.exists(OUTPUTFILE)==FALSE)
      {
        write.table(dfOutcome, OUTPUTFILE, sep=",", row.names = FALSE)
    
      }else{
        write.table(dfOutcome, OUTPUTFILE, sep = ",", row.name = FALSE, col.names = !file.exists(OUTPUTFILE), append = T)

      }
  }
}

getSubjectSpecificRegressions <- function(mainFolder, originalFeatures)
{
  resultsTag <- "SubjectSpecificResults"
  mainFolderName <- getMainFolderName(mainFolder)
  modelName <- getModelName(mainFolder)
  
  dfInputFeatures <- read.table(file= originalFeatures, sep=",", header=TRUE, stringsAsFactors = TRUE)
  
  
  if(length(modelName) > 1)
  {
    for(i in seq(1,length(modelName)))
    {
      dfParams <- getModelParams(mainFolder, modelName[i])

# 1. get subject-specific results from the predictions
      predictionFolder <- file.path(mainFolder,"predictions",modelName[i])
      allFiles <- list.files(path = predictionFolder, full.names = FALSE,recursive = FALSE)
  
      dfOutcome <- data.frame()
      for(resultsFile in allFiles)
      {
        df <- read_csv(file.path(predictionFolder,resultsFile))
        merged <- merge(df, dfInputFeatures[c("id","snapshot_id")], by ="id")
        
        dfResults <- ddply(merged, ~snapshot_id, summarise,
        Spearman_correlation = cor(True_value, Predicted_value, method = "spearman", use = "pairwise.complete.obs"),            
        MAPE = MLmetrics::MAPE(Predicted_value, True_value),                   
        Total_num_examples = length(True_value), 
        numberClasses = length(unique(True_value)),
        true.mn = mean(True_value, na.rm = T), 
        pred.mn = mean(Predicted_value, na.rm = T),
        true.sd = sd(True_value, na.rm = T),
        pred.sd = sd(Predicted_value, na.rm = T))
        
        # remove subjects with NA's
        dfResults.cleaned <- dfResults[which(is.na(dfResults$Spearman_correlation) == FALSE),] # removed 11 people with NA Spearman's correlation
        

        # get the current subject count
        subjectCount = length(unique(dfResults.cleaned$snapshot_id))
        
        row <- data.frame(resultsTag,
                          mainFolderName,
                          merged$Model[1],
                          merged$Label[1],
                          mean(dfResults.cleaned$Spearman_correlation, na.rm=TRUE), 
                          sd(dfResults.cleaned$Spearman_correlation, na.rm=TRUE),
                          mean(dfResults.cleaned$MAPE, na.rm = T),
                          sd(dfResults.cleaned$MAPE, na.rm = T), 
                          mean(dfResults.cleaned$Total_num_examples),
                          sd(dfResults.cleaned$Total_num_examples),
                          mean(dfResults.cleaned$numberClasses),
                          sd(dfResults.cleaned$numberClasses),
                          subjectCount,
                          dfParams[1,1],
                          dfParams[1,2]
                          )
        
        colnames(row) <- c("resultsTag",
                             "mainFolderName",
                             "Model",
                             "Label",
                             "Spearman_correlation_mean",
                             "Spearman_correlation_sd",
                             "MAPE_mean",
                             "MAPE_sd",
                             "Total_num_examples_mean",
                             "Total_num_examples_sd",
                             "numberClasses_mean",
                             "numberClasses_sd",
                             "subjectCount",
                             "dfParams[1, 1]",
                             "dfParams[1, 2]"
                           )
        dfOutcome <- rbind(dfOutcome, row)
        }
  
          # 5. write to the excel file
        if(file.exists(OUTPUTSUBJECTREGRESSION)==FALSE)
        {
          write.table(dfOutcome, OUTPUTSUBJECTREGRESSION, sep=",", row.names = FALSE)
        }
        else
        {
          write.table(dfOutcome, OUTPUTSUBJECTREGRESSION, sep = ",", row.name = FALSE, col.names = !file.exists(OUTPUTSUBJECTREGRESSION), append = T)
        }
    }
  }else{
    dfParams <- getModelParams(mainFolder, modelName)

# 1. get subject-specific results from the predictions
    predictionFolder <- file.path(mainFolder,"predictions",modelName)
    allFiles <- list.files(path = predictionFolder, full.names = FALSE,recursive = FALSE)

    dfInputFeatures <- read.table(file= originalFeatures, sep=",", header=TRUE, stringsAsFactors = TRUE)
  
    dfOutcome <- data.frame()
    for(resultsFile in allFiles)
    {
        df <- read_csv(file.path(predictionFolder,resultsFile))
        merged <- merge(df, dfInputFeatures[c("id","snapshot_id")], by ="id")
        
        dfResults <- ddply(merged, ~snapshot_id, summarise,
        Spearman_correlation = cor(True_value, Predicted_value, method = "spearman", use = "pairwise.complete.obs"),                   
        MAPE = MLmetrics::MAPE(Predicted_value, True_value),                   
        Total_num_examples = length(True_value), 
        numberClasses = length(unique(True_value)),
        true.mn = mean(True_value, na.rm = T), 
        pred.mn = mean(Predicted_value, na.rm = T))
        
        row <- data.frame(resultsTag,
                          mainFolderName,
                          merged$Model[1],
                          merged$Label[1],
                          mean(dfResults$Spearman_correlation, na.rm=TRUE), 
                          sd(dfResults$Spearman_correlation, na.rm=TRUE),
                          mean(dfResults$MAPE, na.rm = T),
                          sd(dfResults$MAPE, na.rm = T), 
                          mean(dfResults$Total_num_examples),
                          sd(dfResults$Total_num_examples),
                          mean(dfResults$numberClasses),
                          sd(dfResults$numberClasses),
                          dfParams[1,1],
                          dfParams[1,2]
                          )
        
        colnames(row) <- c("resultsTag",
                             "mainFolderName",
                             "Model",
                             "Label",
                             "Spearman_correlation_mean",
                             "Spearman_correlation_sd",
                             "MAPE_mean",
                             "MAPE_sd",
                             "Total_num_examples_mean",
                             "Total_num_examples_sd",
                             "numberClasses_mean",
                             "numberClasses_sd",
                             "dfParams[1, 1]",
                             "dfParams[1, 2]"
                           )
        dfOutcome <- rbind(dfOutcome, row)
        }
  
          # 5. write to the excel file
        if(file.exists(OUTPUTSUBJECTREGRESSION)==FALSE)
        {
          write.table(dfOutcome, OUTPUTSUBJECTREGRESSION, sep=",", row.names = FALSE)
        }
        else
        {
          write.table(dfOutcome, OUTPUTSUBJECTREGRESSION, sep = ",", row.name = FALSE, col.names = !file.exists(OUTPUTSUBJECTREGRESSION), append = T)
        }
    }
  
  
}

getOverallClassificationResults <- function(mainFolder)
{
  resultsTag <- "OverallResults"
  mainFolderName <- getMainFolderName(mainFolder)
  modelName <- getModelName(mainFolder)

  # 1. get overall results
  dfOverallResults <- getClassificationResults(mainFolder, modelName)
  
  # 2. get the model hyperparameters
  dfParams <- getModelParams(mainFolder, modelName)
  
  # 3. get the number of classes
  numberClasses <- getNumberClasses(mainFolder, modelName)
  
  # 4. Get the overall summary
  dfOutcome <- cbind(resultsTag,mainFolderName, dfOverallResults, numberClasses, dfParams[1,1], dfParams[1,2])

  # 5. write to the excel file
  if(file.exists(OUTPUTCLASSIFICATIONFILE)==FALSE)
  {
    write.table(dfOutcome, OUTPUTCLASSIFICATIONFILE, sep=",", row.names = FALSE)

  }
  else
  {
    write.table(dfOutcome, OUTPUTCLASSIFICATIONFILE, sep = ",", row.name = FALSE, col.names = !file.exists(OUTPUTCLASSIFICATIONFILE), append = T)

  }
}

getSubjectSpecificClassifications <- function(mainFolder, originalFeatures)
{
   resultsTag <- "SubjectSpecificResults"
  mainFolderName <- getMainFolderName(mainFolder)
  modelName <- getModelName(mainFolder)
  dfParams <- getModelParams(mainFolder, modelName)

# 1. get subject-specific results from the predictions
  predictionFolder <- file.path(mainFolder,"predictions",modelName)
  allFiles <- list.files(path = predictionFolder, full.names = FALSE,recursive = FALSE)

  dfInputFeatures <- read.table(file = originalFeatures, sep=",", header=TRUE, stringsAsFactors = TRUE)
  
  dfOutcome <- data.frame()
  
  for(resultsFile in allFiles)
  {
    df <- read_csv(file.path(predictionFolder,resultsFile))
    merged <- merge(df, dfInputFeatures[c("id","snapshot_id")], by ="id")
    
    dfResults <- ddply(merged, ~snapshot_id, summarise,
      Accuracy = getAccuracy(True_value, Predicted_value),
      AUC = getAUC(True_value, `1_probability`),
      Total_num_examples = length(True_value), 
      numberClasses = length(unique(True_value)))
    
    row <- data.frame(resultsTag,
                      mainFolderName,
                      merged$Model[1],
                      merged$Label[1],
                      mean(dfResults$Accuracy, na.rm=TRUE), 
                      sd(dfResults$Accuracy, na.rm=TRUE),
                      mean(dfResults$AUC, na.rm=TRUE),
                      sd(dfResults$AUC, na.rm=TRUE), 
                      mean(dfResults$Total_num_examples),
                      sd(dfResults$Total_num_examples),
                      mean(dfResults$numberClasses),
                      sd(dfResults$numberClasses),
                      dfParams[1,1],
                      dfParams[1,2]
                      )
    
    colnames(row) <- c("resultsTag",
                         "mainFolderName",
                         "Model",
                         "Label",
                         "Accuracy_mean",
                         "Accuracy_sd",
                         "AUC_mean",
                         "AUC_sd",
                         "Total_num_examples_mean",
                         "Total_num_examples_sd",
                         "numberClasses_mean",
                         "numberClasses_sd",
                         "dfParams[1, 1]",
                         "dfParams[1, 2]"
                       )
    dfOutcome <- rbind(dfOutcome, row)
    }
  
    # 5. write to the excel file
  if(file.exists(OUTPUTSUBJECTCLASSIFICATIONFILE)==FALSE)
  {
    write.table(dfOutcome, OUTPUTSUBJECTCLASSIFICATIONFILE, sep=",", row.names = FALSE)
  }
  else
  {
    write.table(dfOutcome, OUTPUTSUBJECTCLASSIFICATIONFILE, sep = ",", row.name = FALSE, col.names = !file.exists(OUTPUTSUBJECTCLASSIFICATIONFILE), append = T)
  } 
}

getGridsearchRegressions <- function(mainFolder)
{
  resultsTag <- "Gridsearch_regression"
  
  mainFolderName <- getMainFolderName(mainFolder)
  
  modelName <- getModelName(mainFolder)
  
  # 1. get overall results
  dfOverallResults <- getGridsearchResults(mainFolder, modelName)
  
  # 2. get the model hyperparameters for each fold
  dfParams <- getGridsearchModelParams(mainFolder, modelName)
  
  # 3. get the number of classes
  numberClasses <- getNumberClasses(mainFolder, modelName)
  
  # 4. Get the overall summary
  dfOutcome <- merge(dfOverallResults, dfParams, by="foldID")
  dfOutcome <- cbind(resultsTag,mainFolderName,dfOutcome, numberClasses)
  
  
  # 5. write to the excel file
  if(file.exists(OUTPUTGRIDSEARCHREGRESSION)==FALSE)
  {
    write.table(dfOutcome, OUTPUTGRIDSEARCHREGRESSION, sep=",", row.names = FALSE)
  }else{
    write.table(dfOutcome, OUTPUTGRIDSEARCHREGRESSION, sep = ",", row.name = FALSE, col.names = !file.exists(OUTPUTGRIDSEARCHREGRESSION), append = T)
  }
}

getGridsearchClassification <- function(mainFolder)
{
  resultsTag <- "Gridsearch_classification"
  
  mainFolderName <- getMainFolderName(mainFolder)
  
  modelName <- getModelName(mainFolder)
  
  # 1. get overall results
  dfOverallResults <- getGridsearchResults(mainFolder, modelName)
  
  # 2. get the model hyperparameters for each fold
  dfParams <- getGridsearchModelParams(mainFolder, modelName)
  
  # 3. get the number of classes
  numberClasses <- getNumberClasses(mainFolder, modelName)
  
  # 4. Get the overall summary
  dfOutcome <- merge(dfOverallResults, dfParams, by="foldID")
  dfOutcome <- cbind(resultsTag,mainFolderName,dfOutcome, numberClasses)
  
  # 5. write to the excel file
  if(file.exists(OUTPUTGRIDSEARCHCLASSIFICATION)==FALSE)
  {
    write.table(dfOutcome, OUTPUTGRIDSEARCHCLASSIFICATION, sep=",", row.names = FALSE)
  }else{
    write.table(dfOutcome, OUTPUTGRIDSEARCHCLASSIFICATION, sep = ",", row.name = FALSE, col.names = !file.exists(OUTPUTGRIDSEARCHCLASSIFICATION), append = T)
  }
}

getIndividualPredictionsDF <- function(mainFolders, originalFeatures, OUTPUTPATH)
{
  dfInputFeatures <- read.table(file= originalFeatures, sep=",", header=TRUE, stringsAsFactors = FALSE)
  
  # Stress daily summary with Garmin's stress
  dfStressSummaries <- read.table(file="../../dataset/IARPA/data_surveys/stress_daily_summary.csv", header=TRUE, sep=",", stringsAsFactors = FALSE)
      
  # Original feature set
  dfFeatures.fullset <- read.csv("../../dataset/IARPA/data_surveys/merge_all_time_summary_not_blinded_outlier_treated_08_12.csv", stringsAsFactors=FALSE)
  dfFeatures.nonblinded <- dfFeatures.fullset[!is.na(dfFeatures.fullset$stress.d),]
  rownames(dfFeatures.nonblinded) <- NULL
  dfFeatures.nonblinded['id'] <- seq(1, dim(dfFeatures.nonblinded)[1])
    
  # MERGE with Garmin's stress score 
  colnames(dfStressSummaries)[colnames(dfStressSummaries) == "avg_stress"] <- "avg_garmin_stress"
  dfFeatures.nonblinded.stress <- merge(dfFeatures.nonblinded, dfStressSummaries[c("snapshot_id","date","avg_garmin_stress")],  by=c("snapshot_id","date"), all.x = TRUE)
  
  # Read the predictions
  #mainFolderName <- getMainFolderName(mainFolder)
  dfOutcome <- data.frame()
  for (mainFolder in mainFolders)
  {
    modelName <- getModelName(mainFolder)
    for(i in seq(1,length(modelName)))
    {
        # 1. get subject-specific results from the predictions
        predictionFolder <- file.path(mainFolder,"predictions",modelName[i])
        allFiles <- list.files(path = predictionFolder, full.names = FALSE,recursive = FALSE)
    
        # get all predictions and baselines
        for(resultsFile in allFiles)
        {
          df <- read.csv(file.path(predictionFolder,resultsFile))
          
          dfMerged.0 <- merge(df,  dfFeatures.nonblinded.stress[c("id","snapshot_id","date","alc_status","alc.quantity.d","anxiety.d","extraversion.d","agreeableness.d","conscientiousness.d","neuroticism.d","openness.d","work_status","ocb.d","cwb.d","total.pa.d","irb.d","itp.d","pos.affect.d","neg.affect.d","sleep.d","stress.d","tob_status","tob.quantity.d","ave_stress_5min_beforesent", "ave_stress_5min_toend","avg_garmin_stress")], by="id")
          
          dfOutcome <- rbind(dfOutcome, dfMerged.0)
        }
    }
  }
  
  # LR*RF predictions
  df.LR <- dfOutcome[((dfOutcome$Model == "ElasticNet") & (dfOutcome$Label == "stress.d")),]
  df.RF <- dfOutcome[((dfOutcome$Model == "RandomForestRegressor") & (dfOutcome$Label == "stress.d")),]  
  df.RFLR <- df.LR
  df.RFLR$Predicted_value <- (df.LR$Predicted_value * df.RF$Predicted_value)/2
  df.RFLR$Model <- "RFLR"
  df.RFLR$Label <- "stress.d"
  
  # LR*RF predictions - shuffled baseline
  df.LR.shuffled <- dfOutcome[((dfOutcome$Model == "ElasticNet") & (dfOutcome$Label == "stress.d_shuffledWithinSubject")),]
  df.RF.shuffled <- dfOutcome[((dfOutcome$Model == "RandomForestRegressor") & (dfOutcome$Label == "stress.d_shuffledWithinSubject")),]  
  df.RFLR.shuffled <- df.LR.shuffled
  df.RFLR.shuffled$Predicted_value <- (df.LR.shuffled$Predicted_value * df.RF.shuffled$Predicted_value)/2
  df.RFLR.shuffled$Model <- "RFLR"
  df.RFLR.shuffled$Label <- "stress.d_shuffledWithinSubject"
  
  dfOutcome <- rbind(dfOutcome, df.RFLR)
  dfOutcome <- rbind(dfOutcome, df.RFLR.shuffled)
  
  data.check <- dcast(dfOutcome, Model ~ Label, value.var="Label", fun.aggregate=length) 
  
  # remove the overall baseline
  # dfOutcome.2 <- dfOutcome[dfOutcome$Label != "stress_shuffledAll",]
  
  # copy the true labels back to randomized baseline (I disagree with it)
  dfOutcome$True_value[dfOutcome$Label == "stress.d_shuffledWithinSubject"] <- dfOutcome$True_value[dfOutcome$Label == "stress.d"] 
  
  # data check
  sum(dfOutcome$True_value[dfOutcome$Label == "stress.d_shuffledWithinSubject"] == dfOutcome$True_value[dfOutcome$Label == "stress.d"]) 
  data.check <- dcast(dfOutcome, Model ~ Label, value.var="Label", fun.aggregate=length) 
  
  write.table(dfOutcome, OUTPUTPATH, sep = ",", row.name = FALSE)
  dfOutcome
}

# All features, cross-subject
originalFeatures <- "../2_FeatureCorrectionAndFilter/results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_all_feats.csv"
mainFolder<- c("../4_MachineLearning/results_all_feats_cross_subj", "../4_MachineLearning/results_all_feats_cross_subj_shuffle_baseline", "../4_MachineLearning/results_all_feats_cross_subj_shufwtn_baseline")
OUTPUTPATH <- "./results/all_feats_cross_subj_regressions_LR_RF.csv"
dfResults <- getIndividualPredictionsDF(mainFolder, originalFeatures, OUTPUTPATH)

# All features, within subject
originalFeatures <- "../2_FeatureCorrectionAndFilter/results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_all_feats.csv"
mainFolder<- c("../4_MachineLearning/results_all_feats_within_subj", "../4_MachineLearning/results_all_feats_within_subj_shuffle_baseline", "../4_MachineLearning/results_all_feats_within_subj_shufwtn_baseline")
OUTPUTPATH <- "./results/all_feats_within_subj_regressions_LR_RF.csv"
dfResults <- getIndividualPredictionsDF(mainFolder, originalFeatures, OUTPUTPATH)

# No context features, cross-subject
originalFeatures <- "../2_FeatureCorrectionAndFilter/results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv"
mainFolder<- c("../4_MachineLearning/results_ncx_feats_cross_subj", "../4_MachineLearning/results_ncx_feats_cross_subj_shuffle_baseline", "../4_MachineLearning/results_ncx_feats_cross_subj_shufwtn_baseline")
OUTPUTPATH <- "./results/ncx_feats_cross_subj_regressions_LR_RF.csv"
dfResults <- getIndividualPredictionsDF(mainFolder, originalFeatures, OUTPUTPATH)

# No context features, within subject
originalFeatures <- "../2_FeatureCorrectionAndFilter/results/merge_all_time_summary_not_blinded_outlier_treated_08_12_with_baseline_corrected_non_survey_context_feats.csv"
mainFolder<- c("../4_MachineLearning/results_ncx_feats_within_subj", "../4_MachineLearning/results_ncx_feats_within_subj_shuffle_baseline", "../4_MachineLearning/results_ncx_feats_within_subj_shufwtn_baseline")
OUTPUTPATH <- "./results/ncx_feats_within_subj_regressions_LR_RF.csv"
dfResults <- getIndividualPredictionsDF(mainFolder, originalFeatures, OUTPUTPATH)

