"""Section V-A. Stage 2 pipeline on each Stage 1 candidate (Tables IX-XIII, Figs. 3-4)."""
import sys

import pandas as pd

from config import DATA_DIR, RESULTS_DIR
from stage2.pipeline import run_pipeline

XGB_PARAMS_RF = {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7,
              'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257,
              'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362,
              'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256,
              'scale_pos_weight': 3.898438563311057}


def run_stage2_for_rf():
    rf = pd.read_csv(f"{DATA_DIR}/rf_pred_all.csv")
    res = run_pipeline('RF', rf, XGB_PARAMS_RF, n_repeats=10, out_dir=RESULTS_DIR, smoky_delta_unhealthy_only=True)
    print("RF DONE")


def run_stage2_for_lr():
    lr = pd.read_csv(f"{RESULTS_DIR}/lr_pred_all.csv")
    xgb_params = {'n_estimators': 1250, 'learning_rate': 0.028619258045166674, 'max_depth': 3, 'min_child_weight': 4.825242680421432, 'subsample': 0.7744478300213498, 'colsample_bytree': 0.8446287719436285, 'reg_lambda': 0.7294006807049335, 'reg_alpha': 0.002875698896299254, 'gamma': 3.908661632115336, 'scale_pos_weight': 1.2632973500466855}
    res = run_pipeline('LR', lr, xgb_params, n_repeats=10, out_dir=RESULTS_DIR)
    print("LR DONE")


def run_stage2_for_nn():
    nn = pd.read_csv(f"{RESULTS_DIR}/nn_pred_all.csv")
    xgb_params = {'n_estimators': 1000, 'learning_rate': 0.041694714355708345, 'max_depth': 6, 'min_child_weight': 1.3983308055380061, 'subsample': 0.7885628565972381, 'colsample_bytree': 0.7651456263915392, 'reg_lambda': 0.5466381907901013, 'reg_alpha': 9.378118632537179, 'gamma': 8.469961233886654, 'scale_pos_weight': 8.90252363337455}
    res = run_pipeline('NN', nn, xgb_params, n_repeats=10, out_dir=RESULTS_DIR)
    print("NN DONE")


def run_stage2_for_bj():
    bj = pd.read_csv(f"{DATA_DIR}/bj_pred_all.csv")
    xgb_params = {'n_estimators': 1350, 'learning_rate': 0.029454474784661306, 'max_depth': 7, 'min_child_weight': 5.482646385370849, 'subsample': 0.7646717241827257, 'colsample_bytree': 0.9926123594067943, 'reg_lambda': 19.383944005992362, 'reg_alpha': 0.00010444341231664277, 'gamma': 2.202410950657256, 'scale_pos_weight': 3.898438563311057}
    res = run_pipeline('BJ', bj, xgb_params, n_repeats=10, out_dir=RESULTS_DIR)
    print("BJ DONE")


RUNNERS = {"RF": run_stage2_for_rf, "LR": run_stage2_for_lr, "NN": run_stage2_for_nn, "BJ": run_stage2_for_bj}

if __name__ == "__main__":
    for name in (sys.argv[1:] or RUNNERS):
        RUNNERS[name]()
