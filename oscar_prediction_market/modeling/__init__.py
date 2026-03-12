"""Oscar Best Picture modeling package.

ML modeling pipeline for predicting Oscar Best Picture winners.

Modules:
- models: Model types, configurations, implementations, and persistence
- feature_engineering: Feature definitions, groups, and transforms
- data_loader: Load raw data, engineer features, and prepare train/test splits
- evaluation: Discrimination and calibration metrics
- cv_splitting: Cross-validation fold strategies (LOYO, expanding, sliding, bootstrap)
- hyperparameter_tuning: Nested and non-nested CV for hyperparameter selection
- calibration: Probability calibration utilities (softmax, temperature scaling)
- build_model: End-to-end model building pipeline (CV → feature selection → train → predict)
- evaluate_cv: CLI for running cross-validation experiments
- train_predict: CLI for training models and generating predictions
"""
