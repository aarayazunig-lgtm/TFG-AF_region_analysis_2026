% AAZ_LR_region_ranking_v4.m
% =========================================================================
% Region Analysis: Z-score + Logistic Regression + Ranking
% VERSION 4 -- Train/test comparison added on top of v3 features:
%   (a) Stratified train/test split (TRAIN_RATIO, default 80/20)
%   (b) Class-weighting alternative to SMOTE
%   (c) Optimal threshold selection via ROC Youden index (train set)
%   (d) Stratified k-fold cross-validation on the training set
%   (e) Comparison mode: SMOTE vs class-weight vs no-correction
%   (f) Fixed random seed for reproducibility
%   (g) Per-region figure shows BOTH train and test ROC + confusion matrix
%   (h) Metrics panel shows train vs test side-by-side for easy comparison
%   (i) Excel and terminal output include train-set Sens/Spec/BalAcc columns
%
% PER-REGION FIGURE LAYOUT (2 rows x 3 cols):
%   [1] Train ROC     [2] Test ROC      [3] Metrics text (train vs test)
%   [4] Train ConfMat [5] Test ConfMat  [6] Z-score scatter (train+test)
%
% REQUIREMENTS
% ------------
%   Statistics and Machine Learning Toolbox (fitglm, perfcurve)
%
% =========================================================================

clear; clc; close all;

% ── Reproducibility ───────────────────────────────────────────────────────
%   Fix the random seed so train/test splits and SMOTE are identical
%   across runs. Change the integer to get a different (but stable) result.
RANDOM_SEED = 42;
rng(RANDOM_SEED, 'twister');

% =========================================================================
% [USER SETTINGS]
% =========================================================================

EXCEL_FILE = '/home/data/Documents/DATA_MAPS/RESULTS_PLUS_Mar23_2026.xlsx'; % <-- change this

% ── Train / test split ────────────────────────────────────────────────────
%   0.80 = 80% train / 20% test  (recommended for ~200 patients)
TRAIN_RATIO = 0.80;

% ── Imbalance correction ──────────────────────────────────────────────────
%   'smote'       SMOTE oversampling (applied to train split only)
%   'classweight' Weighted logistic regression (recommended for 70/30)
%   'none'        No correction (baseline)
%   'compare'     Run smote + classweight + none side-by-side
BALANCE_METHOD = 'compare';   % <-- change this

% ── Threshold selection ───────────────────────────────────────────────────
%   'fixed'    use THRESHOLD value below
%   'youden'   auto-select per region via Youden index on training ROC
THRESHOLD_MODE = 'youden';    % <-- change this
THRESHOLD      = 0.50;        % used only when THRESHOLD_MODE = 'fixed'

% ── Cross-validation (on training set only) ───────────────────────────────
%   Set CV_FOLDS = 1 to skip. Recommended: 5 or 10 for ~200 patients.
%   SMOTE is applied INSIDE each CV training fold -- no data leakage.
CV_FOLDS = 5;

ALPHA   = 0.05;   % significance level for p-value annotations
SMOTE_K = 5;      % k nearest neighbours for SMOTE

SAVE_FIGS  = true;
FIG_DIR    = '';     % '' = auto subfolder next to Excel file
FIG_FORMAT = 'png';  % 'png' | 'pdf' | 'svg'

% ── Constants ─────────────────────────────────────────────────────────────
REGION_COUNT  = 24;
region_labels = arrayfun(@(r) sprintf('R%02d',r), 1:REGION_COUNT, 'UniformOutput', false);

C0      = [0.22 0.53 0.88];   % recurrence = 0  (blue)
C1      = [0.88 0.25 0.18];   % recurrence = 1  (red)
C_CURVE = [0.10 0.10 0.10];

% =========================================================================
% [1] LOAD DATA
% =========================================================================

fprintf('Loading data from:\n  %s\n\n', EXCEL_FILE);

T_median = readtable(EXCEL_FILE, 'Sheet', 'Median');
T_gq2    = readtable(EXCEL_FILE, 'Sheet', 'Global_Q2');
T_giqr   = readtable(EXCEL_FILE, 'Sheet', 'Global_IQR');

if ~ismember('recurrence', T_median.Properties.VariableNames)
    error(['Column "recurrence" not found in the Median sheet. ' ...
           'Re-run the Python script with --student-xlsx.']);
end

y_all = T_median.recurrence;
valid = ~isnan(y_all);
if any(~valid)
    fprintf('NOTE: %d row(s) with NaN recurrence excluded.\n\n', sum(~valid));
end
y = y_all(valid);

n_patients = numel(y);
fprintf('Patients: %d  (recurrence=1: %d | recurrence=0: %d)\n', ...
    n_patients, sum(y==1), sum(y==0));
fprintf('Class balance: %.1f%% / %.1f%%\n\n', ...
    100*mean(y==0), 100*mean(y==1));

% Per-patient Q2 matrix  [n x 24]
Q2_patient = NaN(n_patients, REGION_COUNT);
for r = 1:REGION_COUNT
    Q2_patient(:,r) = T_median.(region_labels{r})(valid);
end

% Global Q2 and IQR  [1 x 24]
Q2_global  = NaN(1, REGION_COUNT);
IQR_global = NaN(1, REGION_COUNT);
for r = 1:REGION_COUNT
    Q2_global(r)  = T_gq2.(region_labels{r})(1);
    IQR_global(r) = T_giqr.(region_labels{r})(1);
end

% =========================================================================
% [2] Z-SCORES
% =========================================================================

zero_iqr = IQR_global == 0 | isnan(IQR_global);
if any(zero_iqr)
    fprintf('WARNING: IQR_global = 0 for: %s  -- Z will be NaN.\n\n', ...
        strjoin(region_labels(zero_iqr), ', '));
end

Z = (Q2_patient - Q2_global) ./ IQR_global;   % [n_patients x 24]

% =========================================================================
% [3] MODEL FITTING PER REGION
% =========================================================================

if strcmp(BALANCE_METHOD, 'compare')
    methods_to_run = {'smote', 'classweight', 'none'};
else
    methods_to_run = {BALANCE_METHOD};
end
n_methods = numel(methods_to_run);

fprintf('Balance method(s): %s\n', strjoin(methods_to_run, ', '));
fprintf('Threshold mode   : %s\n', THRESHOLD_MODE);
fprintf('CV folds         : %d\n\n', CV_FOLDS);
fprintf('Fitting logistic regression for all %d regions...\n\n', REGION_COUNT);

% Pre-allocate results structs -- one per method
for m = 1:n_methods
    RES(m).method           = methods_to_run{m};         %#ok<AGROW>
    % AUC
    RES(m).AUC_train        = NaN(1, REGION_COUNT);
    RES(m).AUC              = NaN(1, REGION_COUNT);      % TEST (primary)
    RES(m).CV_AUC_mean      = NaN(1, REGION_COUNT);
    RES(m).CV_AUC_std       = NaN(1, REGION_COUNT);
    % Test classification metrics
    RES(m).BAL_ACCURACY     = NaN(1, REGION_COUNT);
    RES(m).SENSITIVITY      = NaN(1, REGION_COUNT);
    RES(m).SPECIFICITY      = NaN(1, REGION_COUNT);
    % Train classification metrics
    RES(m).BAL_ACCURACY_TR  = NaN(1, REGION_COUNT);
    RES(m).SENSITIVITY_TR   = NaN(1, REGION_COUNT);
    RES(m).SPECIFICITY_TR   = NaN(1, REGION_COUNT);
    % Model info
    RES(m).P_VALUE          = NaN(1, REGION_COUNT);
    RES(m).COEFF            = NaN(1, REGION_COUNT);
    RES(m).THRESHOLD_OPT    = NaN(1, REGION_COUNT);
    % ROC curves
    RES(m).ROC_data         = cell(1, REGION_COUNT);     % test ROC
    RES(m).ROC_data_tr      = cell(1, REGION_COUNT);     % train ROC
    % Confusion matrices
    RES(m).CONF_MAT         = cell(1, REGION_COUNT);     % test
    RES(m).CONF_MAT_TR      = cell(1, REGION_COUNT);     % train
    % Raw predictions and data
    RES(m).P_HAT_TRAIN      = cell(1, REGION_COUNT);
    RES(m).P_HAT_TEST       = cell(1, REGION_COUNT);
    RES(m).Z_TRAIN          = cell(1, REGION_COUNT);
    RES(m).Y_TRAIN          = cell(1, REGION_COUNT);
    RES(m).Z_TEST           = cell(1, REGION_COUNT);
    RES(m).Y_TEST           = cell(1, REGION_COUNT);
    RES(m).CURVE_Z          = cell(1, REGION_COUNT);
    RES(m).CURVE_P          = cell(1, REGION_COUNT);
    RES(m).N_TRAIN          = NaN(1, REGION_COUNT);
    RES(m).N_TEST           = NaN(1, REGION_COUNT);
end

for r = 1:REGION_COUNT

    z_r = Z(:,r);

    if all(isnan(z_r))
        fprintf('  %s -- SKIPPED (all NaN Z-scores)\n\n', region_labels{r});
        continue
    end

    ok   = ~isnan(z_r);
    z_ok = z_r(ok);
    y_ok = y(ok);

    if numel(unique(y_ok)) < 2
        fprintf('  %s -- SKIPPED (only one class present)\n\n', region_labels{r});
        continue
    end

    z_grid = linspace(min(z_ok)-0.5, max(z_ok)+0.5, 300)';

    % ── Stratified train / test split ────────────────────────────────────
    idx0  = find(y_ok == 0);
    idx1  = find(y_ok == 1);
    idx0  = idx0(randperm(numel(idx0)));
    idx1  = idx1(randperm(numel(idx1)));

    n_tr0 = max(1, round(TRAIN_RATIO * numel(idx0)));
    n_tr1 = max(1, round(TRAIN_RATIO * numel(idx1)));

    train_idx   = [idx0(1:n_tr0);     idx1(1:n_tr1)];
    test_idx    = [idx0(n_tr0+1:end); idx1(n_tr1+1:end)];

    z_train_raw = z_ok(train_idx);
    y_train_raw = y_ok(train_idx);
    z_test      = z_ok(test_idx);
    y_test      = y_ok(test_idx);

    if numel(unique(y_test)) < 2
        fprintf('  %s -- SKIPPED (test set has only one class; try larger dataset)\n\n', ...
            region_labels{r});
        continue
    end

    fprintf('  %s  split: train=%d (pos=%d neg=%d)  test=%d (pos=%d neg=%d)\n', ...
        region_labels{r}, numel(y_train_raw), sum(y_train_raw==1), sum(y_train_raw==0), ...
        numel(y_test), sum(y_test==1), sum(y_test==0));

    % Class weights for 'classweight' method
    n_pos = sum(y_train_raw == 1);
    n_neg = sum(y_train_raw == 0);
    n_tot = numel(y_train_raw);
    w_pos = n_tot / (2 * n_pos);
    w_neg = n_tot / (2 * n_neg);
    weights_balanced = zeros(n_tot, 1);
    weights_balanced(y_train_raw == 1) = w_pos;
    weights_balanced(y_train_raw == 0) = w_neg;

    for m = 1:n_methods
        meth = methods_to_run{m};
        try
            % ── Build training data according to method ───────────────────
            switch meth
                case 'smote'
                    [z_train, y_train] = smote(z_train_raw, y_train_raw, SMOTE_K);
                    w_train = [];
                    fprintf('    [%s]  SMOTE: n=%d --> n=%d balanced\n', ...
                        meth, numel(y_train_raw), numel(y_train));

                case 'classweight'
                    z_train = z_train_raw;
                    y_train = y_train_raw;
                    w_train = weights_balanced;
                    fprintf('    [%s]  class weights: pos=%.2f  neg=%.2f\n', ...
                        meth, w_pos, w_neg);

                case 'none'
                    z_train = z_train_raw;
                    y_train = y_train_raw;
                    w_train = [];
                    fprintf('    [%s]  no correction\n', meth);
            end

            % ── Fit model on TRAINING data ────────────────────────────────
            if isempty(w_train)
                mdl = fitglm(z_train, y_train, ...
                    'Distribution', 'binomial', 'Link', 'logit');
            else
                mdl = fitglm(z_train, y_train, ...
                    'Distribution', 'binomial', 'Link', 'logit', ...
                    'Weights', w_train);
            end

            % ── Predict on TRAINING data ──────────────────────────────────
            p_hat_train = predict(mdl, z_train_raw);
            p_grid      = predict(mdl, z_grid);

            % ── Training ROC + threshold selection (no leakage) ──────────
            [fpr_tr, tpr_tr, thr_roc_tr, auc_train] = perfcurve(y_train_raw, p_hat_train, 1);
            switch THRESHOLD_MODE
                case 'youden'
                    youden    = tpr_tr - fpr_tr;
                    [~, best] = max(youden);
                    thr_use   = thr_roc_tr(best);
                case 'fixed'
                    thr_use   = THRESHOLD;
            end

            % ── Predict on TEST data ──────────────────────────────────────
            p_hat_test = predict(mdl, z_test);

            % ── TEST ROC / AUC ────────────────────────────────────────────
            [fpr_te, tpr_te, ~, auc_test] = perfcurve(y_test, p_hat_test, 1);

            RES(m).AUC_train(r)   = auc_train;
            RES(m).AUC(r)         = auc_test;
            RES(m).ROC_data_tr{r} = struct('fpr', fpr_tr, 'tpr', tpr_tr);
            RES(m).ROC_data{r}    = struct('fpr', fpr_te, 'tpr', tpr_te);

            % ── Classification metrics on TRAIN set ───────────────────────
            y_pred_tr = double(p_hat_train >= thr_use);
            TP_tr = sum(y_pred_tr==1 & y_train_raw==1);
            TN_tr = sum(y_pred_tr==0 & y_train_raw==0);
            FP_tr = sum(y_pred_tr==1 & y_train_raw==0);
            FN_tr = sum(y_pred_tr==0 & y_train_raw==1);

            RES(m).SENSITIVITY_TR(r)  = TP_tr / max(TP_tr+FN_tr, 1);
            RES(m).SPECIFICITY_TR(r)  = TN_tr / max(TN_tr+FP_tr, 1);
            RES(m).BAL_ACCURACY_TR(r) = (RES(m).SENSITIVITY_TR(r) + RES(m).SPECIFICITY_TR(r)) / 2;
            RES(m).CONF_MAT_TR{r}     = [TN_tr FP_tr; FN_tr TP_tr];

            % ── Classification metrics on TEST set ────────────────────────
            y_pred_te = double(p_hat_test >= thr_use);
            TP = sum(y_pred_te==1 & y_test==1);
            TN = sum(y_pred_te==0 & y_test==0);
            FP = sum(y_pred_te==1 & y_test==0);
            FN = sum(y_pred_te==0 & y_test==1);

            RES(m).SENSITIVITY(r)   = TP / max(TP+FN, 1);
            RES(m).SPECIFICITY(r)   = TN / max(TN+FP, 1);
            RES(m).BAL_ACCURACY(r)  = (RES(m).SENSITIVITY(r) + RES(m).SPECIFICITY(r)) / 2;
            RES(m).CONF_MAT{r}      = [TN FP; FN TP];

            % ── Model coefficients ────────────────────────────────────────
            RES(m).P_VALUE(r)       = mdl.Coefficients.pValue(2);
            RES(m).COEFF(r)         = mdl.Coefficients.Estimate(2);
            RES(m).THRESHOLD_OPT(r) = thr_use;

            % ── Store raw data for figures ────────────────────────────────
            RES(m).P_HAT_TRAIN{r} = p_hat_train;
            RES(m).P_HAT_TEST{r}  = p_hat_test;
            RES(m).Z_TRAIN{r}     = z_train_raw;
            RES(m).Y_TRAIN{r}     = y_train_raw;
            RES(m).Z_TEST{r}      = z_test;
            RES(m).Y_TEST{r}      = y_test;
            RES(m).CURVE_Z{r}     = z_grid;
            RES(m).CURVE_P{r}     = p_grid;
            RES(m).N_TRAIN(r)     = numel(y_train_raw);
            RES(m).N_TEST(r)      = numel(y_test);

            % ── CV on TRAINING set only ───────────────────────────────────
            if CV_FOLDS > 1
                cv_aucs = stratified_kfold_auc( ...
                    z_train_raw, y_train_raw, CV_FOLDS, meth, SMOTE_K, ...
                    weights_balanced, w_train);
                RES(m).CV_AUC_mean(r) = mean(cv_aucs, 'omitnan');
                RES(m).CV_AUC_std(r)  = std(cv_aucs,  0, 'omitnan');
                fprintf('    [%s]  CV(%d-fold, train): AUC=%.3f +/- %.3f\n', ...
                    meth, CV_FOLDS, RES(m).CV_AUC_mean(r), RES(m).CV_AUC_std(r));
            end

            fprintf(['    [%s]  trainAUC=%.3f  testAUC=%.3f' ...
                '  BalAcc_tr=%.3f  BalAcc_te=%.3f' ...
                '  Sens_te=%.3f  Spec_te=%.3f  thr=%.3f  p=%.4f\n\n'], ...
                meth, auc_train, auc_test, ...
                RES(m).BAL_ACCURACY_TR(r), RES(m).BAL_ACCURACY(r), ...
                RES(m).SENSITIVITY(r), RES(m).SPECIFICITY(r), ...
                thr_use, RES(m).P_VALUE(r));

        catch ME
            fprintf('    [%s] -- GLM ERROR: %s\n\n', meth, ME.message);
        end
    end % methods loop
end % region loop

% ── Primary method for ranking and figures ────────────────────────────────
PRIMARY_METHOD_IDX = 1;
R = RES(PRIMARY_METHOD_IDX);

% ── Sort by TEST AUC descending ───────────────────────────────────────────
[AUC_sorted, sort_idx] = sort(R.AUC, 'descend', 'MissingPlacement', 'last');
ranked_labels = region_labels(sort_idx);

% ── Figure output folder ──────────────────────────────────────────────────
[excel_folder, excel_name, ~] = fileparts(EXCEL_FILE);
if isempty(FIG_DIR)
    FIG_DIR = fullfile(excel_folder, [excel_name '_LR_figures_v4']);
end
if SAVE_FIGS && ~exist(FIG_DIR, 'dir')
    mkdir(FIG_DIR);
    fprintf('Figure folder:\n  %s\n\n', FIG_DIR);
end

% =========================================================================
% [4] PER-REGION FIGURES  (2 rows x 3 cols)
%   [1] Train ROC     [2] Test ROC      [3] Metrics (train vs test)
%   [4] Train ConfMat [5] Test ConfMat  [6] Z-score scatter
% =========================================================================

fprintf('Generating %d region figures...\n', REGION_COUNT);

cnames = {'TN','FP'; 'FN','TP'};   % reused in both confusion matrix panels

for r = 1:REGION_COUNT

    fig = figure('Name', sprintf('Region %s', region_labels{r}), ...
        'NumberTitle', 'off', ...
        'Position', [40 40 1320 720], ...
        'Color', [0.97 0.97 0.97]);

    sgtitle(sprintf('Region  %s  --  Logistic Regression  |  %s  |  Recurrence 0 vs 1', ...
        region_labels{r}, BALANCE_METHOD), 'FontSize', 13, 'FontWeight', 'bold');

    if isnan(R.AUC(r))
        axes('Position',[0.1 0.35 0.8 0.25]); axis off; %#ok<LAXES>
        text(0.5, 0.5, 'Region could not be fitted (NaN Z-scores or single class)', ...
            'HorizontalAlignment','center','FontSize',12,'Color',[0.5 0.1 0.1]);
        if SAVE_FIGS
            saveas(fig, fullfile(FIG_DIR, sprintf('%s.%s', region_labels{r}, FIG_FORMAT)));
        end
        continue
    end

    z_tr  = R.Z_TRAIN{r};
    y_tr  = R.Y_TRAIN{r};
    z_te  = R.Z_TEST{r};
    y_te  = R.Y_TEST{r};
    p_tr  = R.P_HAT_TRAIN{r};
    p_te  = R.P_HAT_TEST{r};
    thr_r = R.THRESHOLD_OPT(r);

    % Operating points
    y_pred_tr = double(p_tr >= thr_r);
    tpr_op_tr = sum(y_pred_tr==1 & y_tr==1) / max(sum(y_tr==1), 1);
    fpr_op_tr = sum(y_pred_tr==1 & y_tr==0) / max(sum(y_tr==0), 1);

    y_pred_te = double(p_te >= thr_r);
    tpr_op_te = sum(y_pred_te==1 & y_te==1) / max(sum(y_te==1), 1);
    fpr_op_te = sum(y_pred_te==1 & y_te==0) / max(sum(y_te==0), 1);

    % ── [1] Train ROC ────────────────────────────────────────────────────
    fpr_tr_v = R.ROC_data_tr{r}.fpr;
    tpr_tr_v = R.ROC_data_tr{r}.tpr;

    ax1 = subplot(2, 3, 1);
    hold(ax1, 'on');
    fill(ax1, [fpr_tr_v; flipud(fpr_tr_v)], [tpr_tr_v; zeros(size(tpr_tr_v))], ...
        [0.60 0.82 0.60], 'FaceAlpha',0.28, 'EdgeColor','none');
    plot(ax1, fpr_tr_v, tpr_tr_v, '-', 'Color',[0.08 0.48 0.08], 'LineWidth',2.2);
    plot(ax1, [0 1],[0 1], '--', 'Color',[0.55 0.55 0.55], 'LineWidth',1.0);
    plot(ax1, fpr_op_tr, tpr_op_tr, 'o', 'MarkerSize',8, ...
        'MarkerFaceColor',[0.95 0.55 0.10], 'MarkerEdgeColor','k', 'LineWidth',1.2);
    if CV_FOLDS > 1 && ~isnan(R.CV_AUC_mean(r))
        text(ax1, 0.36, 0.06, sprintf('CV AUC: %.3f +/- %.3f', ...
            R.CV_AUC_mean(r), R.CV_AUC_std(r)), ...
            'FontSize',7.5, 'Color',[0.0 0.35 0.6]);
    end
    xlim(ax1,[0 1]); ylim(ax1,[0 1]);
    xlabel(ax1, 'FPR  (1 - Specificity)');
    ylabel(ax1, 'TPR  (Sensitivity)');
    title(ax1, sprintf('ROC  TRAIN   AUC = %.3f', R.AUC_train(r)), 'FontSize',10);
    legend(ax1, {'AUC area','ROC','Chance',sprintf('Op. pt (thr=%.2f)',thr_r)}, ...
        'Location','southeast', 'FontSize',7);
    grid(ax1,'on'); hold(ax1,'off');

    % ── [2] Test ROC ─────────────────────────────────────────────────────
    fpr_te_v = R.ROC_data{r}.fpr;
    tpr_te_v = R.ROC_data{r}.tpr;

    ax2 = subplot(2, 3, 2);
    hold(ax2, 'on');
    fill(ax2, [fpr_te_v; flipud(fpr_te_v)], [tpr_te_v; zeros(size(tpr_te_v))], ...
        [0.60 0.82 0.60], 'FaceAlpha',0.28, 'EdgeColor','none');
    plot(ax2, fpr_te_v, tpr_te_v, '-', 'Color',[0.08 0.48 0.08], 'LineWidth',2.2);
    plot(ax2, [0 1],[0 1], '--', 'Color',[0.55 0.55 0.55], 'LineWidth',1.0);
    plot(ax2, fpr_op_te, tpr_op_te, 'o', 'MarkerSize',8, ...
        'MarkerFaceColor',[0.95 0.55 0.10], 'MarkerEdgeColor','k', 'LineWidth',1.2);
    xlim(ax2,[0 1]); ylim(ax2,[0 1]);
    xlabel(ax2, 'FPR  (1 - Specificity)');
    ylabel(ax2, 'TPR  (Sensitivity)');
    title(ax2, sprintf('ROC  TEST   AUC = %.3f', R.AUC(r)), 'FontSize',10);
    legend(ax2, {'AUC area','ROC','Chance',sprintf('Op. pt (thr=%.2f)',thr_r)}, ...
        'Location','southeast', 'FontSize',7);
    grid(ax2,'on'); hold(ax2,'off');

    % ── [3] Metrics text (train vs test side-by-side) ────────────────────
    ax3 = subplot(2, 3, 3);
    axis(ax3, 'off');

    pv = R.P_VALUE(r);
    if     pv < 0.001, sig_str = '*** p<0.001';
    elseif pv < 0.01,  sig_str = '**  p<0.01';
    elseif pv < ALPHA, sig_str = '*   p<0.05';
    else,              sig_str = sprintf('n.s. p=%.3f', pv);
    end

    rank_pos = find(sort_idx == r);

    if CV_FOLDS > 1 && ~isnan(R.CV_AUC_mean(r))
        cv_line = sprintf('CV AUC (%d-fold) :  %.4f +/- %.4f\n', ...
            CV_FOLDS, R.CV_AUC_mean(r), R.CV_AUC_std(r));
    else
        cv_line = '';
    end

    txt = sprintf([ ...
        'LOGISTIC REGRESSION\n'                      ...
        '---------------------------------\n'         ...
        'Method : %-10s  Thr: %s\n'                  ...
        '---------------------------------\n'         ...
        '               TRAIN      TEST\n'            ...
        'AUC         :  %.4f    %.4f\n'               ...
        'Bal. Acc.   :  %.4f    %.4f\n'               ...
        'Sensitivity :  %.4f    %.4f\n'               ...
        'Specificity :  %.4f    %.4f\n'               ...
        '---------------------------------\n'         ...
        '%s'                                          ...
        'Coeff (Z)   : %+.4f\n'                      ...
        'p-value     :  %.4f  %s\n'                   ...
        '---------------------------------\n'         ...
        'Rank : %d / %d   n: %d tr / %d te\n'        ...
        'Threshold   :  %.3f\n'                       ...
        'SMOTE k     :  %d\n'],                       ...
        BALANCE_METHOD, THRESHOLD_MODE,               ...
        R.AUC_train(r),       R.AUC(r),              ...
        R.BAL_ACCURACY_TR(r), R.BAL_ACCURACY(r),     ...
        R.SENSITIVITY_TR(r),  R.SENSITIVITY(r),       ...
        R.SPECIFICITY_TR(r),  R.SPECIFICITY(r),       ...
        cv_line,                                      ...
        R.COEFF(r), pv, sig_str,                      ...
        rank_pos, REGION_COUNT,                       ...
        R.N_TRAIN(r), R.N_TEST(r), thr_r, SMOTE_K);

    text(ax3, 0.02, 0.98, txt, ...
        'Units','normalized', 'VerticalAlignment','top', ...
        'FontName','Courier New', 'FontSize',8.5, ...
        'Color',[0.08 0.08 0.08]);

    % ── [4] Train confusion matrix ────────────────────────────────────────
    ax4   = subplot(2, 3, 4);
    cm_tr = R.CONF_MAT_TR{r};
    mv_tr = max(max(cm_tr(:)), 1);
    imagesc(ax4, cm_tr);
    colormap(ax4, flipud(gray));
    clim(ax4, [0 mv_tr+1]);
    for ri = 1:2
        for ci = 1:2
            val = cm_tr(ri,ci);
            tc  = [0.05 0.05 0.05];
            if val < mv_tr*0.55; tc = [0.95 0.95 0.95]; end
            text(ax4, ci, ri, sprintf('%s\n%d', cnames{ri,ci}, val), ...
                'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                'FontSize',12, 'FontWeight','bold', 'Color',tc);
        end
    end
    set(ax4, 'XTick',[1 2], 'XTickLabel',{'Pred 0','Pred 1'}, ...
             'YTick',[1 2], 'YTickLabel',{'Act 0','Act 1'}, 'FontSize',9);
    title(ax4, sprintf('Conf. Matrix  TRAIN  (thr=%.2f)', thr_r), 'FontSize',10);
    colorbar(ax4);

    % ── [5] Test confusion matrix ─────────────────────────────────────────
    ax5   = subplot(2, 3, 5);
    cm_te = R.CONF_MAT{r};
    mv_te = max(max(cm_te(:)), 1);
    imagesc(ax5, cm_te);
    colormap(ax5, flipud(gray));
    clim(ax5, [0 mv_te+1]);
    for ri = 1:2
        for ci = 1:2
            val = cm_te(ri,ci);
            tc  = [0.05 0.05 0.05];
            if val < mv_te*0.55; tc = [0.95 0.95 0.95]; end
            text(ax5, ci, ri, sprintf('%s\n%d', cnames{ri,ci}, val), ...
                'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                'FontSize',12, 'FontWeight','bold', 'Color',tc);
        end
    end
    set(ax5, 'XTick',[1 2], 'XTickLabel',{'Pred 0','Pred 1'}, ...
             'YTick',[1 2], 'YTickLabel',{'Act 0','Act 1'}, 'FontSize',9);
    title(ax5, sprintf('Conf. Matrix  TEST  (thr=%.2f)', thr_r), 'FontSize',10);
    colorbar(ax5);

    % ── [6] Z-score scatter: train (filled) + test (hollow) ───────────────
    ax6 = subplot(2, 3, 6);
    hold(ax6, 'on');
    scatter(ax6, z_tr(y_tr==0), p_tr(y_tr==0), 35, C0, 'filled', ...
        'MarkerFaceAlpha',0.50, 'DisplayName','Train Rec=0');
    scatter(ax6, z_tr(y_tr==1), p_tr(y_tr==1), 35, C1, 'filled', ...
        'MarkerFaceAlpha',0.50, 'DisplayName','Train Rec=1');
    scatter(ax6, z_te(y_te==0), p_te(y_te==0), 55, C0, ...
        'LineWidth',1.4, 'DisplayName','Test  Rec=0');
    scatter(ax6, z_te(y_te==1), p_te(y_te==1), 55, C1, ...
        'LineWidth',1.4, 'DisplayName','Test  Rec=1');
    plot(ax6, R.CURVE_Z{r}, R.CURVE_P{r}, '-', ...
        'Color',C_CURVE, 'LineWidth',2.0, 'DisplayName','Logistic fit (train)');
    yline(ax6, thr_r, '--', 'Color',[0.45 0.45 0.45], 'LineWidth',1.2, ...
        'Label',sprintf('thr=%.2f',thr_r), 'LabelHorizontalAlignment','left');
    xlabel(ax6, 'Z-score  [ (Q2_i - Q2_{global}) / IQR_{global} ]');
    ylabel(ax6, 'P(recurrence = 1)');
    title(ax6, 'Z-score vs Predicted Probability', 'FontSize',10);
    legend(ax6, 'Location','northwest', 'FontSize',7);
    ylim(ax6, [-0.05 1.05]);
    grid(ax6,'on'); hold(ax6,'off');

    if SAVE_FIGS
        saveas(fig, fullfile(FIG_DIR, sprintf('%s.%s', region_labels{r}, FIG_FORMAT)));
    end
    fprintf('  %s  saved.\n', region_labels{r});
end

fprintf('\nAll %d region figures done.\n\n', REGION_COUNT);

% =========================================================================
% [4b] COMPARISON FIGURE  (only in 'compare' mode)
% =========================================================================

if strcmp(BALANCE_METHOD, 'compare')
    fprintf('Generating method comparison figure...\n');

    fig_cmp = figure('Name','Method Comparison','NumberTitle','off', ...
        'Position',[50 50 1400 900],'Color',[0.97 0.97 0.97]);
    sgtitle('Method Comparison per Region: SMOTE vs ClassWeight vs None', ...
        'FontSize',13,'FontWeight','bold');

    metrics_names = {'AUC (test)','BalAcc (test)','Sensitivity (test)','Specificity (test)'};
    colors_cmp = [0.22 0.55 0.88; 0.18 0.75 0.42; 0.88 0.40 0.18];

    for mi = 1:4
        ax = subplot(2, 2, mi);
        hold(ax, 'on');
        for mm = 1:n_methods
            switch mi
                case 1; vals = RES(mm).AUC;
                case 2; vals = RES(mm).BAL_ACCURACY;
                case 3; vals = RES(mm).SENSITIVITY;
                case 4; vals = RES(mm).SPECIFICITY;
            end
            plot(ax, 1:REGION_COUNT, vals(sort_idx), '-o', ...
                'Color',colors_cmp(mm,:), 'LineWidth',1.6, ...
                'MarkerSize',5, 'DisplayName',methods_to_run{mm});
        end
        yline(ax, 0.5, '--', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
        set(ax, 'XTick',1:REGION_COUNT, 'XTickLabel',ranked_labels, ...
            'XTickLabelRotation',45, 'FontSize',8);
        ylabel(ax, metrics_names{mi});
        title(ax, metrics_names{mi}, 'FontSize',11);
        legend(ax, 'Location','southwest', 'FontSize',8);
        ylim(ax, [0 1.05]);
        grid(ax,'on'); hold(ax,'off');
    end

    if SAVE_FIGS
        saveas(fig_cmp, fullfile(FIG_DIR, sprintf('METHOD_COMPARISON.%s', FIG_FORMAT)));
        fprintf('  Comparison figure saved.\n\n');
    end
end

% =========================================================================
% [5] FINAL RANKING FIGURE
% =========================================================================

fprintf('Generating final ranking figure...\n');

fig_rank = figure('Name','Final Region Ranking','NumberTitle','off', ...
    'Position',[50 50 1280 800],'Color',[0.97 0.97 0.97]);

sgtitle(sprintf(['Region Ranking -- Logistic Regression  |  %s  |  Z-score predictor' ...
    '\nRanked best to worst by TEST AUC  |  Recurrence 0 vs 1'], BALANCE_METHOD), ...
    'FontSize',13,'FontWeight','bold');

x_pos = 1:REGION_COUNT;

% ── Top: AUC bar chart ────────────────────────────────────────────────────
ax_r1 = subplot(2,1,1);
hold(ax_r1,'on');

b1 = bar(ax_r1, x_pos, AUC_sorted, 'FaceColor','flat','EdgeColor','none','BarWidth',0.72);
for i = 1:REGION_COUNT
    auc_i = AUC_sorted(i);
    if isnan(auc_i)
        b1.CData(i,:) = [0.75 0.75 0.75];
    else
        t = max(0, min(1, (auc_i-0.5)/0.5));
        b1.CData(i,:) = [(1-t)*0.88, t*0.72+(1-t)*0.15, 0.12];
    end
end

if CV_FOLDS > 1
    cv_means = R.CV_AUC_mean(sort_idx);
    cv_stds  = R.CV_AUC_std(sort_idx);
    valid_cv = ~isnan(cv_means);
    errorbar(ax_r1, x_pos(valid_cv), cv_means(valid_cv), cv_stds(valid_cv), ...
        'k.', 'LineWidth',1.4, 'CapSize',5);
    legend(ax_r1, {'AUC test (full model)','CV AUC +/- std'}, ...
        'Location','northeast','FontSize',8);
end

yline(ax_r1, 0.5,'--','Color',[0.35 0.35 0.35],'LineWidth',1.6,'Label','Chance = 0.5');

for i = 1:REGION_COUNT
    rr    = sort_idx(i);
    pv    = R.P_VALUE(rr);
    auc_i = AUC_sorted(i);
    if isnan(pv) || isnan(auc_i); continue; end
    if     pv < 0.001, star = '***';
    elseif pv < 0.01,  star = '**';
    elseif pv < ALPHA, star = '*';
    else,              star = '';
    end
    if ~isempty(star)
        text(ax_r1, i, auc_i+0.028, star, 'HorizontalAlignment','center', ...
            'FontSize',11, 'FontWeight','bold', 'Color',[0.08 0.08 0.08]);
    end
end

for i = 1:REGION_COUNT
    auc_i = AUC_sorted(i);
    if isnan(auc_i); continue; end
    text(ax_r1, i, max(auc_i-0.07, 0.02), sprintf('%.2f',auc_i), ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',7.5,'FontWeight','bold','Color','w');
end

set(ax_r1,'XTick',x_pos,'XTickLabel',ranked_labels, ...
    'XTickLabelRotation',45,'FontSize',10,'XLim',[0.3 REGION_COUNT+0.7]);
ylabel(ax_r1,'AUC');
title(ax_r1,'TEST AUC per Region  (best to worst)   |   * p<0.05   ** p<0.01   *** p<0.001', ...
    'FontSize',11);
ylim(ax_r1,[0 1.13]);
grid(ax_r1,'on'); hold(ax_r1,'off');

% ── Bottom: Balanced Accuracy / Sensitivity / Specificity (TEST) ──────────
ax_r2 = subplot(2,1,2);
hold(ax_r2,'on');

metrics_mat = [R.BAL_ACCURACY(sort_idx); R.SENSITIVITY(sort_idx); R.SPECIFICITY(sort_idx)]';
b2 = bar(ax_r2, x_pos, metrics_mat,'grouped','EdgeColor','none','BarWidth',0.85);
b2(1).FaceColor = [0.22 0.60 0.90];
b2(2).FaceColor = [0.92 0.38 0.20];
b2(3).FaceColor = [0.18 0.78 0.42];

yline(ax_r2, 0.5,'--','Color',[0.40 0.40 0.40],'LineWidth',1.3);

for i = 1:REGION_COUNT
    rr = sort_idx(i);
    pv = R.P_VALUE(rr);
    if ~isnan(pv) && pv < ALPHA
        plot(ax_r2,[i-0.42 i+0.42],[1.04 1.04],'-k','LineWidth',1.8);
        text(ax_r2, i, 1.065,'*','HorizontalAlignment','center', ...
            'FontSize',12,'FontWeight','bold');
    end
end

set(ax_r2,'XTick',x_pos,'XTickLabel',ranked_labels, ...
    'XTickLabelRotation',45,'FontSize',10,'XLim',[0.3 REGION_COUNT+0.7]);
ylabel(ax_r2,'Score');
title(ax_r2, sprintf(['Balanced Accuracy / Sensitivity / Specificity  (TEST)' ...
    '  |  threshold mode: %s  |  bracket = p<%.2f'], THRESHOLD_MODE, ALPHA),'FontSize',11);
legend(ax_r2,{'Balanced Accuracy','Sensitivity','Specificity'}, ...
    'Location','southwest','FontSize',9);
ylim(ax_r2,[0 1.14]);
grid(ax_r2,'on'); hold(ax_r2,'off');

if SAVE_FIGS
    saveas(fig_rank, fullfile(FIG_DIR, sprintf('RANKING_ALL_REGIONS.%s', FIG_FORMAT)));
    fprintf('  Ranking figure saved.\n\n');
end

% =========================================================================
% [6] SAVE EXCEL TABLE
% =========================================================================

sig_col = cell(REGION_COUNT,1);
for i = 1:REGION_COUNT
    rr = sort_idx(i);
    pv = R.P_VALUE(rr);
    if     isnan(pv),   sig_col{i} = 'N/A';
    elseif pv < 0.001,  sig_col{i} = '***';
    elseif pv < 0.01,   sig_col{i} = '**';
    elseif pv < ALPHA,  sig_col{i} = '*';
    else,               sig_col{i} = 'n.s.';
    end
end

if CV_FOLDS > 1
    out_table = table( ...
        (1:REGION_COUNT)',              ...
        ranked_labels',                 ...
        R.AUC(sort_idx)',               ...
        R.AUC_train(sort_idx)',         ...
        R.CV_AUC_mean(sort_idx)',       ...
        R.CV_AUC_std(sort_idx)',        ...
        R.BAL_ACCURACY_TR(sort_idx)',   ...
        R.SENSITIVITY_TR(sort_idx)',    ...
        R.SPECIFICITY_TR(sort_idx)',    ...
        R.BAL_ACCURACY(sort_idx)',      ...
        R.SENSITIVITY(sort_idx)',       ...
        R.SPECIFICITY(sort_idx)',       ...
        R.THRESHOLD_OPT(sort_idx)',     ...
        R.P_VALUE(sort_idx)',           ...
        R.COEFF(sort_idx)',             ...
        R.N_TRAIN(sort_idx)',           ...
        R.N_TEST(sort_idx)',            ...
        sig_col,                        ...
        'VariableNames', { ...
            'Rank','Region', ...
            'AUC_test','AUC_train','CV_AUC_mean','CV_AUC_std', ...
            'BalAcc_train','Sensitivity_train','Specificity_train', ...
            'BalAcc_test','Sensitivity_test','Specificity_test', ...
            'Threshold_used','pValue','Coefficient_Z', ...
            'N_train','N_test','Significance'});
else
    out_table = table( ...
        (1:REGION_COUNT)',              ...
        ranked_labels',                 ...
        R.AUC(sort_idx)',               ...
        R.AUC_train(sort_idx)',         ...
        R.BAL_ACCURACY_TR(sort_idx)',   ...
        R.SENSITIVITY_TR(sort_idx)',    ...
        R.SPECIFICITY_TR(sort_idx)',    ...
        R.BAL_ACCURACY(sort_idx)',      ...
        R.SENSITIVITY(sort_idx)',       ...
        R.SPECIFICITY(sort_idx)',       ...
        R.THRESHOLD_OPT(sort_idx)',     ...
        R.P_VALUE(sort_idx)',           ...
        R.COEFF(sort_idx)',             ...
        R.N_TRAIN(sort_idx)',           ...
        R.N_TEST(sort_idx)',            ...
        sig_col,                        ...
        'VariableNames', { ...
            'Rank','Region', ...
            'AUC_test','AUC_train', ...
            'BalAcc_train','Sensitivity_train','Specificity_train', ...
            'BalAcc_test','Sensitivity_test','Specificity_test', ...
            'Threshold_used','pValue','Coefficient_Z', ...
            'N_train','N_test','Significance'});
end

out_path = fullfile(excel_folder, [excel_name '_LR_ranking_v4.xlsx']);
writetable(out_table, out_path);

% =========================================================================
% [7] TERMINAL RANKING SUMMARY
% =========================================================================

fprintf('\n-- FINAL RANKING (ranked by TEST AUC) ');
fprintf('------------------------------------------------------------------------\n');
fprintf('%-4s  %-6s  %9s  %9s  |  %8s  %8s  %8s  |  %8s  %8s  %8s  |  %6s  %5s\n', ...
    'Rank','Region','AUC_test','AUC_train', ...
    'BalAcc_tr','Sens_tr','Spec_tr', ...
    'BalAcc_te','Sens_te','Spec_te', ...
    'Thr','Sig');
fprintf('%s\n', repmat('-',1,112));
for i = 1:REGION_COUNT
    rr = sort_idx(i);
    fprintf(['%-4d  %-6s  %9.4f  %9.4f  |' ...
             '  %8.4f  %8.4f  %8.4f  |' ...
             '  %8.4f  %8.4f  %8.4f  |  %6.3f  %5s\n'], ...
        i, region_labels{rr}, ...
        R.AUC(rr), R.AUC_train(rr), ...
        R.BAL_ACCURACY_TR(rr), R.SENSITIVITY_TR(rr), R.SPECIFICITY_TR(rr), ...
        R.BAL_ACCURACY(rr),    R.SENSITIVITY(rr),    R.SPECIFICITY(rr), ...
        R.THRESHOLD_OPT(rr), sig_col{i});
end

fprintf('\nRanking table  ->  %s\n', out_path);
fprintf('Figures folder ->  %s\n', FIG_DIR);
fprintf('\nDone.\n');

% =========================================================================
% [8] LOCAL FUNCTIONS
% =========================================================================

% ── smote ────────────────────────────────────────────────────────────────
function [z_out, y_out] = smote(z_in, y_in, k)
% SMOTE  Synthetic Minority Over-sampling TEchnique (1-D)
%   Reference: Chawla et al. (2002) JMLR 3:321-357
    if nargin < 3; k = 5; end

    min_idx = find(y_in == 1);
    n_min   = numel(min_idx);
    n_maj   = numel(y_in) - n_min;

    if n_min == 0 || n_min >= n_maj
        z_out = z_in; y_out = y_in; return
    end

    n_syn  = n_maj - n_min;
    z_min  = z_in(min_idx);
    k_eff  = min(k, n_min-1);

    if k_eff < 1
        z_out = [z_in; z_min(1) + randn(n_syn,1)*1e-6];
        y_out = [y_in; ones(n_syn,1)];
        return
    end

    dist_mat = abs(z_min - z_min');
    z_syn    = zeros(n_syn, 1);
    for s = 1:n_syn
        seed_i      = randi(n_min);
        d           = dist_mat(seed_i,:);
        d(seed_i)   = Inf;
        [~, nn_idx] = sort(d);
        nbr_val     = z_min(nn_idx(randi(k_eff)));
        z_syn(s)    = z_min(seed_i) + rand()*(nbr_val - z_min(seed_i));
    end

    z_out = [z_in;  z_syn];
    y_out = [y_in;  ones(n_syn,1)];
end

% ── stratified_kfold_auc ─────────────────────────────────────────────────
function cv_aucs = stratified_kfold_auc(z_ok, y_ok, k_folds, meth, smote_k, ...
                                         weights_balanced, w_train) %#ok<INUSD>
% Stratified k-fold CV returning AUC per fold.
% SMOTE applied inside each training fold only -- no leakage.
    idx0 = find(y_ok == 0);
    idx1 = find(y_ok == 1);
    idx0 = idx0(randperm(numel(idx0)));
    idx1 = idx1(randperm(numel(idx1)));

    folds0  = kfold_split(idx0, k_folds);
    folds1  = kfold_split(idx1, k_folds);
    cv_aucs = NaN(k_folds, 1);

    for f = 1:k_folds
        test_idx  = [folds0{f}; folds1{f}];
        train_idx = setdiff(1:numel(y_ok), test_idx)';

        z_tr_cv = z_ok(train_idx);
        y_tr_cv = y_ok(train_idx);
        z_te_cv = z_ok(test_idx);
        y_te_cv = y_ok(test_idx);

        if numel(unique(y_tr_cv)) < 2 || numel(unique(y_te_cv)) < 2
            continue
        end

        n_pos_cv = sum(y_tr_cv == 1);
        n_neg_cv = sum(y_tr_cv == 0);
        n_tot_cv = numel(y_tr_cv);

        try
            switch meth
                case 'smote'
                    [z_fit, y_fit] = smote(z_tr_cv, y_tr_cv, smote_k);
                    mdl_cv = fitglm(z_fit, y_fit, ...
                        'Distribution','binomial','Link','logit');

                case 'classweight'
                    w_cv = zeros(n_tot_cv, 1);
                    w_cv(y_tr_cv==1) = n_tot_cv / (2*max(n_pos_cv,1));
                    w_cv(y_tr_cv==0) = n_tot_cv / (2*max(n_neg_cv,1));
                    mdl_cv = fitglm(z_tr_cv, y_tr_cv, ...
                        'Distribution','binomial','Link','logit', ...
                        'Weights',w_cv);

                case 'none'
                    mdl_cv = fitglm(z_tr_cv, y_tr_cv, ...
                        'Distribution','binomial','Link','logit');
            end

            p_cv = predict(mdl_cv, z_te_cv);
            [~, ~, ~, auc_f] = perfcurve(y_te_cv, p_cv, 1);
            cv_aucs(f) = auc_f;
        catch
            % fold failed -- leave as NaN
        end
    end
end

% ── kfold_split ──────────────────────────────────────────────────────────
function folds = kfold_split(idx, k)
% Splits index vector into k roughly equal cell arrays.
    n      = numel(idx);
    folds  = cell(k, 1);
    starts = round(linspace(0, n, k+1));
    for i  = 1:k
        folds{i} = idx(starts(i)+1 : starts(i+1));
    end
end