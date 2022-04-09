# Stress in the Age of Wearables
Code for the "Stress Prediction in the Age of Wearables" 2021 paper based on Notre Dame's IARPA-MOSAIC data set.

The source code provided implements data analysis only.  Data curation and processing scripts have been omitted in this repository, which means this code is not standalone.  This repository serves as documentation of the analysis performed in the paper.  If you have any specific questions regarding this code, please contact Brandon M. Booth (brandon.m.booth@gmail.com).

# Running the Source Code
Most code is written for Python and R, so it could run on any platform.  The code was executed on a Windows 10 machine.

In the 4_MachineLearning folder, the settings files describe machine learning experiments.  These files cannot be run by themselves and require a working installation of either common-models or common-models2.  Both of these modules are provided in the src folder and can be installed following the instructions there.  Settings files whose names begin with 'settings' should be executed by common-models using the following command line (for example):

  main --settings settings_ml_within_subj_all_features.py

Settings files whose names start with "cm2" should be run using common-models2.  Once CM2 is installed, you simply run the settings file using python:

  python cm2_settings_ml_cross_subj_all_features_wo_survey_context_mlp.py
