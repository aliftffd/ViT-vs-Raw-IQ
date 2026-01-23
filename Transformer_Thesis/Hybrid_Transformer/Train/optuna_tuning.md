
  Quick exploration (find promising regions):
  python Train/optuna_tune.py --n_trials 50 --tune_epochs 15 --subset 0.2 --study_name
  exploration_v1

  Full tuning (after exploration):
  python Train/optuna_tune.py --n_trials 100 --tune_epochs 30 --subset 0.3 --study_name
  full_tune_v1

  Hardware Utilization Tips
  ┌───────────────┬───────────────┬──────────────────────────┐
  │    Setting    │ Your Hardware │      Recommendation      │
  ├───────────────┼───────────────┼──────────────────────────┤
  │ --subset      │ 32GB RAM      │ 0.2-0.3 (20-30% of data) │
  ├───────────────┼───────────────┼──────────────────────────┤
  │ batch_size    │ 16GB VRAM     │ 512-1024 works well      │
  ├───────────────┼───────────────┼──────────────────────────┤
  │ --tune_epochs │ -             │ 15-30 for tuning         │
  ├───────────────┼───────────────┼──────────────────────────┤
  │ --n_trials    │ -             │ 50-100 for good coverage │
  └───────────────┴───────────────┴──────────────────────────┘
  After Tuning

  Once Optuna finds the best hyperparameters, train the final model with full dataset:
  # The script generates this command in best_training_command.txt
  python Train/train.py --epochs 100 --batch_size 512 ...  # with best params

  The tuning results will be saved to:
  - optuna_results/<study_name>/best_trial.json - Best hyperparameters
  - optuna_results/<study_name>/best_training_command.txt - Ready-to-run command