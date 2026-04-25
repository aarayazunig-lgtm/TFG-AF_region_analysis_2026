% AAZ_RF_region_ranking_v1.m
% =========================================================================
% Region Analysis: Z-score + Random Forest + Ranking
% Adapted from AAZ_LR_w_training.m (Logistic Regression v3)
%
% KEY CHANGES from LR version:
%   - fitglm  --> TreeBagger (Random Forest via ensemble of decision trees)
%   - predict returns [~, scores]; scores(:,2) used as P(recurrence=1)
%   - No p-value or coefficient per region (RF has no closed-form equivalent)
%     --> Feature importance (OOB permuted predictor importance) reported
%   - 'classweight' handled via TreeBagger 'Cost' matrix
%   - S-curve panel replaced with Z-score vs RF score scatter (no sigmoid fit)
%   - All other logic (train/test split, SMOTE, CV, threshold, ranking,
%     figures, Excel output) preserved from the LR version
%
% METHOD OVERVIEW
% ---------------
%   For each of the 24 regions independently:
%
%   1. Build the Z-score vector (one value per patient):
%        Z(i) = ( Q2_patient(i) - Q2_global ) / IQR_global
%
%   2. Stratified train/test split (TRAIN_RATIO, default 80/20)
%
%   3. Choose imbalance correction method (BALANCE_METHOD):
%        'smote'        -- SMOTE oversampling
%        'classweight'  -- Asymmetric misclassification cost in TreeBagger
%        'none'         -- No correction (baseline)
%        'compare'      -- Run all three side-by-side
%
%   4. Threshold options (THRESHOLD_MODE):
%        'fixed'    -- use THRESHOLD value directly
%        'youden'   -- maximise Youden index on TRAINING ROC
%
%   5. Stratified k-fold CV on training set (CV_FOLDS)
%
%   6. Per-region figure (2x2 layout):
%        [A] Z-score vs RF predicted probability (train + test scatter)
%        [B] ROC curve computed on TEST set
%        [C] Confusion matrix on TEST set
%        [D] Metrics text (train AUC, test AUC, CV AUC, Sens, Spec,
%            OOB error, feature importance)
%
%   7. Final ranking figure and Excel table ranked by TEST AUC
%
% REQUIREMENTS
% ------------
%   Statistics and Machine Learning Toolbox (TreeBagger, perfcurve)
%
% =========================================================================

clear; clc; close all;

% =========================================================================
% [USER SETTINGS]
% =========================================================================

EXCEL_FILE = '/home/data/Documents/DATA_MAPS/RESULTS_PLUS_Mar23_3Dtemplate.xlsx'; % <-- change this

% ── Random Forest hyperparameters ─────────────────────────────────────────
N_TREES     = 200;   % number of decision trees (increase for stability)
MIN_LEAF    = 5;     % minimum leaf size  (controls tree depth / overfitting)
MAX_FEATURES = 1;    % predictors sampled per split  (1 = only Z-score here)

% ── Train / test split ────────────────────────────────────────────────────
TRAIN_RATIO = 0.80;

% ── Imbalance correction ──────────────────────────────────────────────────
%   'smote'       SMOTE oversampling
%   'classweight' Asymmetric cost matrix in TreeBagger  (recommended 70/30)
%   'none'        No correction (baseline)
%   'compare'     Run all three and report side-by-side
BALANCE_METHOD = 'compare';   % <-- change this

% ── Threshold selection ───────────────────────────────────────────────────
%   'fixed'    use THRESHOLD value below
%   'youden'   auto-select via Youden index on TRAINING ROC
THRESHOLD_MODE = 'youden';    % <-- change this
THRESHOLD      = 0.50;        % used only when THRESHOLD_MODE = 'fixed'

% ── Cross-validation (on training set only) ───────────────────────────────
CV_FOLDS = 5;   % set to 1 to skip

ALPHA   = 0.05;   % significance level for annotations
SMOTE_K = 5;      % k nearest neighbours for SMOTE

SAVE_FIGS  = true;
FIG_DIR    = '';      % '' = auto subfolder next to Excel file
FIG_FORMAT = 'png';   % 'png' | 'pdf' | 'svg'

% ── Constants ─────────────────────────────────────────────────────────────
REGION_COUNT  = 24;
region_labels = arrayfun(@(r) sprintf('R%02d',r), 1:REGION_COUNT, 'UniformOutput', false);

C0 = [0.22 0.53 0.88];   % recurrence = 0  (blue)
C1 = [0.88 0.25 0.18];   % recurrence = 1  (red)

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
% [3] RANDOM FOREST PER REGION
%     Supports: smote | classweight | none | compare
% =========================================================================

if strcmp(BALANCE_METHOD, 'compare')
    methods_to_run = {'smote', 'classweight', 'none'};
else
    methods_to_run = {BALANCE_METHOD};
end
n_methods = numel(methods_to_run);

fprintf('Balance method(s) : %s\n', strjoin(methods_to_run, ', '));
fprintf('Threshold mode    : %s\n', THRESHOLD_MODE);
fprintf('CV folds          : %d\n', CV_FOLDS);
fprintf('N trees / MinLeaf : %d / %d\n\n', N_TREES, MIN_LEAF);
fprintf('Fitting Random Forest for all %d regions...\n\n', REGION_COUNT);

% Pre-allocate results structs -- one per method
for m = 1:n_methods
    RES(m).method           = methods_to_run{m};           %#ok<AGROW>
    RES(m).AUC_train        = NaN(1, REGION_COUNT);
    RES(m).AUC              = NaN(1, REGION_COUNT);   % TEST AUC (primary)
    RES(m).CV_AUC_mean      = NaN(1, REGION_COUNT);
    RES(m).CV_AUC_std       = NaN(1, REGION_COUNT);
    RES(m).BAL_ACCURACY     = NaN(1, REGION_COUNT);
    RES(m).SENSITIVITY      = NaN(1, REGION_COUNT);
    RES(m).SPECIFICITY      = NaN(1, REGION_COUNT);
    RES(m).OOB_ERROR        = NaN(1, REGION_COUNT);   % RF-specific
    RES(m).FEAT_IMPORTANCE  = NaN(1, REGION_COUNT);   % RF-specific
    RES(m).THRESHOLD_OPT    = NaN(1, REGION_COUNT);
    RES(m).ROC_data         = cell(1, REGION_COUNT);
    RES(m).CONF_MAT         = cell(1, REGION_COUNT);
    RES(m).P_HAT_TRAIN      = cell(1, REGION_COUNT);
    RES(m).P_HAT_TEST       = cell(1, REGION_COUNT);
    RES(m).Z_TRAIN          = cell(1, REGION_COUNT);
    RES(m).Y_TRAIN          = cell(1, REGION_COUNT);
    RES(m).Z_TEST           = cell(1, REGION_COUNT);
    RES(m).Y_TEST           = cell(1, REGION_COUNT);
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

    % ── Stratified train / test split ────────────────────────────────────
    idx0 = find(y_ok == 0);
    idx1 = find(y_ok == 1);
    idx0 = idx0(randperm(numel(idx0)));
    idx1 = idx1(randperm(numel(idx1)));

    n_tr0 = max(1, round(TRAIN_RATIO * numel(idx0)));
    n_tr1 = max(1, round(TRAIN_RATIO * numel(idx1)));

    train_idx = [idx0(1:n_tr0);      idx1(1:n_tr1)];
    test_idx  = [idx0(n_tr0+1:end);  idx1(n_tr1+1:end)];

    z_train_raw = z_ok(train_idx);
    y_train_raw = y_ok(train_idx);
    z_test      = z_ok(test_idx);
    y_test      = y_ok(test_idx);

    if numel(unique(y_test)) < 2
        fprintf('  %s -- SKIPPED (test set has only one class)\n\n', region_labels{r});
        continue
    end

    fprintf('  %s  split: train=%d (pos=%d neg=%d)  test=%d (pos=%d neg=%d)\n', ...
        region_labels{r}, numel(y_train_raw), sum(y_train_raw==1), sum(y_train_raw==0), ...
        numel(y_test), sum(y_test==1), sum(y_test==0));

    % Class counts for cost matrix (classweight branch)
    n_pos = sum(y_train_raw == 1);
    n_neg = sum(y_train_raw == 0);
    n_tot = numel(y_train_raw);
    % Cost: [correct_0 misclassify_0; misclassify_1 correct_1]
    % Penalise misclassifying the minority (pos) class more
    cost_mat = [0, (n_pos/n_neg); 1, 0];   % asymmetric cost

    for m = 1:n_methods
        meth = methods_to_run{m};
        try
            % ── Build training data / options per method ──────────────────
            switch meth
                case 'smote'
                    [z_train, y_train] = smote(z_train_raw, y_train_raw, SMOTE_K);
                    use_cost = false;
                    fprintf('    [%s]  SMOTE: n=%d --> n=%d balanced\n', ...
                        meth, numel(y_train_raw), numel(y_train));

                case 'classweight'
                    z_train  = z_train_raw;
                    y_train  = y_train_raw;
                    use_cost = true;
                    fprintf('    [%s]  cost matrix: FN-cost=%.2f  FP-cost=1.00\n', ...
                        meth, n_pos/n_neg);

                case 'none'
                    z_train  = z_train_raw;
                    y_train  = y_train_raw;
                    use_cost = false;
                    fprintf('    [%s]  no correction\n', meth);
            end

            % ── Fit Random Forest on TRAINING data ───────────────────────
            % TreeBagger expects categorical labels as strings or a
            % categorical array. We use categorical for clean class order.
            y_train_cat = categorical(y_train, [0 1], {'0','1'});

            rf_opts = { ...
                'Method',           'classification', ...
                'NumPredictorsToSample', MAX_FEATURES, ...
                'MinLeafSize',      MIN_LEAF, ...
                'OOBPrediction',    'on', ...
                'OOBPredictorImportance', 'on' };

            if use_cost
                rf_opts = [rf_opts, {'Cost', cost_mat}]; %#ok<AGROW>
            end

            mdl = TreeBagger(N_TREES, z_train, y_train_cat, rf_opts{:});

            % ── OOB error & feature importance (train-set estimates) ──────
            RES(m).OOB_ERROR(r)       = oobError(mdl, 'Mode', 'ensemble');
            RES(m).FEAT_IMPORTANCE(r) = mdl.OOBPermutedPredictorDeltaError(end);

            % ── Predict on TRAINING data ──────────────────────────────────
            [~, score_train] = predict(mdl, z_train_raw);
            p_hat_train      = score_train(:,2);   % P(class='1')

            % ── Threshold selection on TRAINING ROC (no leakage) ─────────
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
            [~, score_test] = predict(mdl, z_test);
            p_hat_test      = score_test(:,2);

            % ── TEST ROC / AUC ────────────────────────────────────────────
            [fpr_te, tpr_te, ~, auc_test] = perfcurve(y_test, p_hat_test, 1);
            RES(m).AUC_train(r) = auc_train;
            RES(m).AUC(r)       = auc_test;
            RES(m).ROC_data{r}  = struct('fpr', fpr_te, 'tpr', tpr_te);

            % ── Classification on TEST set ────────────────────────────────
            y_pred_test = double(p_hat_test >= thr_use);
            TP = sum(y_pred_test==1 & y_test==1);
            TN = sum(y_pred_test==0 & y_test==0);
            FP = sum(y_pred_test==1 & y_test==0);
            FN = sum(y_pred_test==0 & y_test==1);

            RES(m).SENSITIVITY(r)   = TP / max(TP+FN, 1);
            RES(m).SPECIFICITY(r)   = TN / max(TN+FP, 1);
            RES(m).BAL_ACCURACY(r)  = (RES(m).SENSITIVITY(r) + RES(m).SPECIFICITY(r)) / 2;
            RES(m).THRESHOLD_OPT(r) = thr_use;

            RES(m).CONF_MAT{r}    = [TN FP; FN TP];
            RES(m).P_HAT_TRAIN{r} = p_hat_train;
            RES(m).P_HAT_TEST{r}  = p_hat_test;
            RES(m).Z_TRAIN{r}     = z_train_raw;
            RES(m).Y_TRAIN{r}     = y_train_raw;
            RES(m).Z_TEST{r}      = z_test;
            RES(m).Y_TEST{r}      = y_test;
            RES(m).N_TRAIN(r)     = numel(y_train_raw);
            RES(m).N_TEST(r)      = numel(y_test);

            % ── CV on TRAINING set only ───────────────────────────────────
            if CV_FOLDS > 1
                cv_aucs = stratified_kfold_auc_rf( ...
                    z_train_raw, y_train_raw, CV_FOLDS, meth, SMOTE_K, ...
                    cost_mat, N_TREES, MIN_LEAF, MAX_FEATURES);
                RES(m).CV_AUC_mean(r) = mean(cv_aucs, 'omitnan');
                RES(m).CV_AUC_std(r)  = std(cv_aucs,  0, 'omitnan');
                fprintf('    [%s]  CV(%d-fold, train): AUC=%.3f ± %.3f\n', ...
                    meth, CV_FOLDS, RES(m).CV_AUC_mean(r), RES(m).CV_AUC_std(r));
            end

            fprintf('    [%s]  trainAUC=%.3f  testAUC=%.3f  BalAcc=%.3f  Sens=%.3f  Spec=%.3f  thr=%.3f  OOB=%.3f\n\n', ...
                meth, auc_train, auc_test, RES(m).BAL_ACCURACY(r), ...
                RES(m).SENSITIVITY(r), RES(m).SPECIFICITY(r), thr_use, ...
                RES(m).OOB_ERROR(r));

        catch ME
            fprintf('    [%s] -- RF ERROR: %s\n\n', meth, ME.message);
        end
    end % methods loop
end % region loop

% ── Primary method for ranking and figures ────────────────────────────────
PRIMARY_METHOD_IDX = 1;
R = RES(PRIMARY_METHOD_IDX);

% ── Sort by AUC descending ────────────────────────────────────────────────
[AUC_sorted, sort_idx] = sort(R.AUC, 'descend', 'MissingPlacement', 'last');
ranked_labels = region_labels(sort_idx);

% ── Figure output folder ──────────────────────────────────────────────────
[excel_folder, excel_name, ~] = fileparts(EXCEL_FILE);
if isempty(FIG_DIR)
    FIG_DIR = fullfile(excel_folder, [excel_name '_RF_figures_v1']);
end
if SAVE_FIGS && ~exist(FIG_DIR, 'dir')
    mkdir(FIG_DIR);
    fprintf('Figure folder:\n  %s\n\n', FIG_DIR);
end

% =========================================================================
% [4] PER-REGION FIGURES
% =========================================================================

fprintf('Generating %d region figures...\n', REGION_COUNT);

for r = 1:REGION_COUNT

    fig = figure('Name', sprintf('Region %s', region_labels{r}), ...
        'NumberTitle', 'off', ...
        'Position', [60 60 960 700], ...
        'Color', [0.97 0.97 0.97]);

    sgtitle(sprintf('Region  %s  --  Random Forest  |  %s  |  Recurrence 0 vs 1', ...
        region_labels{r}, BALANCE_METHOD), 'FontSize', 13, 'FontWeight', 'bold');

    if isnan(R.AUC(r))
        axes('Position',[0.1 0.35 0.8 0.25]); axis off;
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

    % ── [A] Z-score vs RF predicted probability ───────────────────────────
    ax_sc = subplot(2, 2, 1);
    hold(ax_sc, 'on');

    scatter(ax_sc, z_tr(y_tr==0), p_tr(y_tr==0), 40, C0, 'filled', ...
        'MarkerFaceAlpha', 0.55, 'DisplayName', 'Train Rec=0');
    scatter(ax_sc, z_tr(y_tr==1), p_tr(y_tr==1), 40, C1, 'filled', ...
        'MarkerFaceAlpha', 0.55, 'DisplayName', 'Train Rec=1');
    scatter(ax_sc, z_te(y_te==0), p_te(y_te==0), 60, C0, ...
        'LineWidth', 1.4, 'DisplayName', 'Test  Rec=0');
    scatter(ax_sc, z_te(y_te==1), p_te(y_te==1), 60, C1, ...
        'LineWidth', 1.4, 'DisplayName', 'Test  Rec=1');

    yline(ax_sc, thr_r, '--', 'Color',[0.45 0.45 0.45], 'LineWidth',1.3, ...
        'Label', sprintf('thr = %.2f', thr_r), 'LabelHorizontalAlignment','left');

    xlabel(ax_sc, 'Z-score  [ (Q2_i - Q2_{global}) / IQR_{global} ]');
    ylabel(ax_sc, 'RF Predicted  P(recurrence = 1)');
    title(ax_sc, 'Z-score vs RF Predicted Probability', 'FontSize', 11);
    legend(ax_sc, 'Location','northwest', 'FontSize', 8);
    ylim(ax_sc, [-0.05 1.05]);
    grid(ax_sc, 'on');
    hold(ax_sc, 'off');

    % ── [B] ROC curve ────────────────────────────────────────────────────
    ax_roc = subplot(2, 2, 2);
    hold(ax_roc, 'on');

    fpr_v = R.ROC_data{r}.fpr;
    tpr_v = R.ROC_data{r}.tpr;

    fill(ax_roc, [fpr_v; flipud(fpr_v)], [tpr_v; zeros(size(tpr_v))], ...
        [0.60 0.82 0.60], 'FaceAlpha',0.30, 'EdgeColor','none');
    plot(ax_roc, fpr_v, tpr_v, '-', 'Color',[0.08 0.48 0.08], 'LineWidth',2.5);
    plot(ax_roc, [0 1],[0 1],'--','Color',[0.55 0.55 0.55],'LineWidth',1.1);

    y_pred_te = double(p_te >= thr_r);
    tpr_op = sum(y_pred_te==1 & y_te==1) / max(sum(y_te==1),1);
    fpr_op = sum(y_pred_te==1 & y_te==0) / max(sum(y_te==0),1);
    plot(ax_roc, fpr_op, tpr_op, 'o', 'MarkerSize',8, ...
        'MarkerFaceColor',[0.95 0.55 0.10], 'MarkerEdgeColor','k','LineWidth',1.2);

    text(ax_roc, 0.55, 0.06, sprintf('Train AUC: %.3f', R.AUC_train(r)), ...
        'FontSize', 8, 'Color', [0.4 0.4 0.4]);
    if CV_FOLDS > 1 && ~isnan(R.CV_AUC_mean(r))
        cv_str = sprintf('CV AUC: %.3f ± %.3f', R.CV_AUC_mean(r), R.CV_AUC_std(r));
        text(ax_roc, 0.55, 0.13, cv_str, 'FontSize', 8, 'Color', [0.0 0.35 0.6]);
    end

    xlim(ax_roc,[0 1]); ylim(ax_roc,[0 1]);
    xlabel(ax_roc, 'False Positive Rate  (1 - Specificity)');
    ylabel(ax_roc, 'True Positive Rate  (Sensitivity)');
    title(ax_roc, sprintf('ROC Curve  (TEST)   AUC = %.3f', R.AUC(r)), 'FontSize',11);
    legend(ax_roc, {'AUC area', sprintf('ROC test (AUC=%.3f)',R.AUC(r)), ...
        'Chance', sprintf('Op. point (thr=%.2f)',thr_r)}, ...
        'Location','southeast','FontSize',8);
    grid(ax_roc,'on');
    hold(ax_roc,'off');

    % ── [C] Confusion matrix (TEST set) ──────────────────────────────────
    ax_cm = subplot(2, 2, 3);

    cm      = R.CONF_MAT{r};
    max_val = max(cm(:));
    imagesc(ax_cm, cm);
    colormap(ax_cm, flipud(gray));
    clim(ax_cm, [0 max_val+1]);

    cnames = {'TN','FP';'FN','TP'};
    for ri = 1:2
        for ci = 1:2
            val = cm(ri,ci);
            tc  = [0.05 0.05 0.05];
            if val < max_val*0.55; tc = [0.95 0.95 0.95]; end
            text(ax_cm, ci, ri, sprintf('%s\n%d', cnames{ri,ci}, val), ...
                'HorizontalAlignment','center','VerticalAlignment','middle', ...
                'FontSize',13,'FontWeight','bold','Color',tc);
        end
    end

    set(ax_cm, 'XTick',[1 2],'XTickLabel',{'Predicted 0','Predicted 1'}, ...
               'YTick',[1 2],'YTickLabel',{'Actual 0','Actual 1'},'FontSize',10);
    title(ax_cm, sprintf('Confusion Matrix  TEST  (threshold = %.2f)', thr_r), 'FontSize',11);
    colorbar(ax_cm);

    % ── [D] Metrics text box ─────────────────────────────────────────────
    ax_txt = subplot(2, 2, 4);
    axis(ax_txt, 'off');

    rank_pos = find(sort_idx == r);

    if CV_FOLDS > 1 && ~isnan(R.CV_AUC_mean(r))
        cv_line = sprintf('CV AUC (%d-fold)  :  %.3f +/- %.3f\n', ...
            CV_FOLDS, R.CV_AUC_mean(r), R.CV_AUC_std(r));
    else
        cv_line = '';
    end

    txt = sprintf([ ...
        'RANDOM FOREST METRICS\n'            ...
        '------------------------------\n'   ...
        'Balance method    :  %s\n'          ...
        'Threshold mode    :  %s\n'          ...
        'N trees / MinLeaf :  %d / %d\n'    ...
        '------------------------------\n'   ...
        'AUC  (train)      :  %.4f\n'        ...
        'AUC  (TEST)       :  %.4f  <--\n'   ...
        '%s'                                 ...
        'Balanced Accuracy :  %.4f\n'        ...
        'Sensitivity       :  %.4f\n'        ...
        'Specificity       :  %.4f\n'        ...
        '------------------------------\n'   ...
        'OOB Error (train) :  %.4f\n'        ...
        'Feat Importance Z :  %.4f\n'        ...
        '------------------------------\n'   ...
        'Rank (by testAUC) :  %d / %d\n'    ...
        'n train / n test  :  %d / %d\n'    ...
        'Threshold used    :  %.3f\n'        ...
        'SMOTE k           :  %d\n'],        ...
        BALANCE_METHOD, THRESHOLD_MODE,      ...
        N_TREES, MIN_LEAF,                   ...
        R.AUC_train(r), R.AUC(r), cv_line,  ...
        R.BAL_ACCURACY(r), R.SENSITIVITY(r), R.SPECIFICITY(r), ...
        R.OOB_ERROR(r), R.FEAT_IMPORTANCE(r), ...
        rank_pos, REGION_COUNT,              ...
        R.N_TRAIN(r), R.N_TEST(r), thr_r, SMOTE_K);

    text(ax_txt, 0.04, 0.97, txt, ...
        'Units','normalized','VerticalAlignment','top', ...
        'FontName','Courier New','FontSize',9, ...
        'Color',[0.08 0.08 0.08]);

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
    sgtitle('Method Comparison per Region: SMOTE vs ClassWeight vs None (RF)', ...
        'FontSize',13,'FontWeight','bold');

    metrics_names = {'AUC','BalAcc','Sensitivity','Specificity'};
    n_met = numel(metrics_names);
    colors_cmp = [0.22 0.55 0.88; 0.18 0.75 0.42; 0.88 0.40 0.18];

    for mi = 1:n_met
        ax = subplot(2, 2, mi);
        hold(ax, 'on');

        for mm = 1:n_methods
            switch metrics_names{mi}
                case 'AUC';          vals = RES(mm).AUC;
                case 'BalAcc';       vals = RES(mm).BAL_ACCURACY;
                case 'Sensitivity';  vals = RES(mm).SENSITIVITY;
                case 'Specificity';  vals = RES(mm).SPECIFICITY;
            end
            plot(ax, 1:REGION_COUNT, vals(sort_idx), '-o', ...
                'Color', colors_cmp(mm,:), 'LineWidth', 1.6, ...
                'MarkerSize', 5, 'DisplayName', methods_to_run{mm});
        end

        yline(ax, 0.5, '--', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
        set(ax, 'XTick',1:REGION_COUNT,'XTickLabel',ranked_labels, ...
            'XTickLabelRotation',45,'FontSize',8);
        ylabel(ax, metrics_names{mi});
        title(ax, metrics_names{mi}, 'FontSize',11);
        legend(ax, 'Location','southwest','FontSize',8);
        ylim(ax, [0 1.05]);
        grid(ax, 'on');
        hold(ax, 'off');
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

sgtitle(sprintf(['Region Ranking -- Random Forest  |  %s  |  Z-score predictor' ...
    '\nRanked best to worst by AUC  |  Recurrence 0 vs 1'], BALANCE_METHOD), ...
    'FontSize',13,'FontWeight','bold');

x_pos = 1:REGION_COUNT;

% ── Top: AUC bar chart ────────────────────────────────────────────────────
ax1 = subplot(2,1,1);
hold(ax1,'on');

b1 = bar(ax1, x_pos, AUC_sorted, 'FaceColor','flat','EdgeColor','none','BarWidth',0.72);

for i = 1:REGION_COUNT
    auc_i = AUC_sorted(i);
    if isnan(auc_i)
        b1.CData(i,:) = [0.75 0.75 0.75];
    else
        t = max(0, min(1, (auc_i-0.5)/0.5));
        b1.CData(i,:) = [(1-t)*0.88, t*0.72+(1-t)*0.15, 0.12];
    end
end

% CV error bars
if CV_FOLDS > 1
    cv_means = R.CV_AUC_mean(sort_idx);
    cv_stds  = R.CV_AUC_std(sort_idx);
    valid_cv = ~isnan(cv_means);
    errorbar(ax1, x_pos(valid_cv), cv_means(valid_cv), cv_stds(valid_cv), ...
        'k.', 'LineWidth', 1.4, 'CapSize', 5);
    legend(ax1, {'AUC (full model)', 'CV AUC ± std'}, ...
        'Location','northeast','FontSize',8);
end

yline(ax1, 0.5,'--','Color',[0.35 0.35 0.35],'LineWidth',1.6,'Label','Chance = 0.5');

% OOB error annotations above bars (RF-specific, replaces p-value stars)
for i = 1:REGION_COUNT
    rr    = sort_idx(i);
    oob_i = R.OOB_ERROR(rr);
    auc_i = AUC_sorted(i);
    if isnan(oob_i) || isnan(auc_i); continue; end
    text(ax1, i, auc_i+0.025, sprintf('%.2f', oob_i), ...
        'HorizontalAlignment','center','FontSize',7,'Color',[0.3 0.3 0.3]);
end

% AUC value inside bar
for i = 1:REGION_COUNT
    auc_i = AUC_sorted(i);
    if isnan(auc_i); continue; end
    text(ax1, i, max(auc_i-0.07, 0.02), sprintf('%.2f',auc_i), ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',7.5,'FontWeight','bold','Color','w');
end

set(ax1,'XTick',x_pos,'XTickLabel',ranked_labels, ...
    'XTickLabelRotation',45,'FontSize',10,'XLim',[0.3 REGION_COUNT+0.7]);
ylabel(ax1,'AUC');
title(ax1,'AUC per Region  (best to worst)   |   label above bar = OOB error', ...
    'FontSize',11);
ylim(ax1,[0 1.13]);
grid(ax1,'on');
hold(ax1,'off');

% ── Bottom: Balanced Accuracy / Sensitivity / Specificity ─────────────────
ax2 = subplot(2,1,2);
hold(ax2,'on');

metrics_mat = [R.BAL_ACCURACY(sort_idx); R.SENSITIVITY(sort_idx); R.SPECIFICITY(sort_idx)]';
b2 = bar(ax2, x_pos, metrics_mat,'grouped','EdgeColor','none','BarWidth',0.85);
b2(1).FaceColor = [0.22 0.60 0.90];
b2(2).FaceColor = [0.92 0.38 0.20];
b2(3).FaceColor = [0.18 0.78 0.42];

yline(ax2, 0.5,'--','Color',[0.40 0.40 0.40],'LineWidth',1.3);

set(ax2,'XTick',x_pos,'XTickLabel',ranked_labels, ...
    'XTickLabelRotation',45,'FontSize',10,'XLim',[0.3 REGION_COUNT+0.7]);
ylabel(ax2,'Score');
title(ax2, sprintf('Balanced Accuracy / Sensitivity / Specificity  (threshold mode: %s)', ...
    THRESHOLD_MODE),'FontSize',11);
legend(ax2,{'Balanced Accuracy','Sensitivity','Specificity'}, ...
    'Location','southwest','FontSize',9);
ylim(ax2,[0 1.14]);
grid(ax2,'on');
hold(ax2,'off');

if SAVE_FIGS
    saveas(fig_rank, fullfile(FIG_DIR, sprintf('RANKING_ALL_REGIONS.%s', FIG_FORMAT)));
    fprintf('  Ranking figure saved.\n\n');
end

% =========================================================================
% [6] SAVE EXCEL TABLE
% =========================================================================

if CV_FOLDS > 1
    out_table = table( ...
        (1:REGION_COUNT)',                    ...
        ranked_labels',                       ...
        R.AUC(sort_idx)',                     ...
        R.AUC_train(sort_idx)',               ...
        R.CV_AUC_mean(sort_idx)',             ...
        R.CV_AUC_std(sort_idx)',              ...
        R.BAL_ACCURACY(sort_idx)',            ...
        R.SENSITIVITY(sort_idx)',             ...
        R.SPECIFICITY(sort_idx)',             ...
        R.THRESHOLD_OPT(sort_idx)',           ...
        R.OOB_ERROR(sort_idx)',               ...
        R.FEAT_IMPORTANCE(sort_idx)',         ...
        R.N_TRAIN(sort_idx)',                 ...
        R.N_TEST(sort_idx)',                  ...
        'VariableNames', {'Rank','Region','AUC_test','AUC_train', ...
                          'CV_AUC_mean','CV_AUC_std', ...
                          'BalancedAccuracy','Sensitivity','Specificity', ...
                          'Threshold_used','OOB_Error','Feat_Importance_Z', ...
                          'N_train','N_test'});
else
    out_table = table( ...
        (1:REGION_COUNT)',                    ...
        ranked_labels',                       ...
        R.AUC(sort_idx)',                     ...
        R.AUC_train(sort_idx)',               ...
        R.BAL_ACCURACY(sort_idx)',            ...
        R.SENSITIVITY(sort_idx)',             ...
        R.SPECIFICITY(sort_idx)',             ...
        R.THRESHOLD_OPT(sort_idx)',           ...
        R.OOB_ERROR(sort_idx)',               ...
        R.FEAT_IMPORTANCE(sort_idx)',         ...
        R.N_TRAIN(sort_idx)',                 ...
        R.N_TEST(sort_idx)',                  ...
        'VariableNames', {'Rank','Region','AUC_test','AUC_train', ...
                          'BalancedAccuracy','Sensitivity','Specificity', ...
                          'Threshold_used','OOB_Error','Feat_Importance_Z', ...
                          'N_train','N_test'});
end

out_path = fullfile(excel_folder, [excel_name '_RF_ranking_v1.xlsx']);
writetable(out_table, out_path);

% Terminal ranking summary
fprintf('-- FINAL RANKING (ranked by TEST AUC) ------------------------------------------\n');
fprintf('%-4s  %-6s  %8s  %8s  %8s  %8s  %8s  %6s  %9s\n', ...
    'Rank','Region','AUC_test','AUC_train','BalAcc','Sensitiv','Specific','Thr','OOB_Err');
fprintf('%s\n', repmat('-',1,88));
for i = 1:REGION_COUNT
    rr = sort_idx(i);
    fprintf('%-4d  %-6s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %6.3f  %9.4f\n', ...
        i, region_labels{rr}, R.AUC(rr), R.AUC_train(rr), R.BAL_ACCURACY(rr), ...
        R.SENSITIVITY(rr), R.SPECIFICITY(rr), R.THRESHOLD_OPT(rr), R.OOB_ERROR(rr));
end

fprintf('\nRanking table  ->  %s\n', out_path);
fprintf('Figures folder ->  %s\n', FIG_DIR);
fprintf('\nDone.\n');

% =========================================================================
% [7] LOCAL FUNCTIONS
% =========================================================================

% ── smote ────────────────────────────────────────────────────────────────
function [z_out, y_out] = smote(z_in, y_in, k)
% SMOTE  Synthetic Minority Over-sampling TEchnique (1-D)

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

% ── stratified_kfold_auc_rf ───────────────────────────────────────────────
function cv_aucs = stratified_kfold_auc_rf(z_ok, y_ok, k_folds, meth, smote_k, ...
                                            cost_mat, n_trees, min_leaf, max_feat)
% Stratified k-fold CV using Random Forest.
% SMOTE applied INSIDE each training fold only (no leakage).

    idx0 = find(y_ok == 0);
    idx1 = find(y_ok == 1);
    idx0 = idx0(randperm(numel(idx0)));
    idx1 = idx1(randperm(numel(idx1)));

    folds0 = kfold_split(idx0, k_folds);
    folds1 = kfold_split(idx1, k_folds);

    cv_aucs = NaN(k_folds, 1);

    for f = 1:k_folds
        test_idx  = [folds0{f}; folds1{f}];
        train_idx = setdiff(1:numel(y_ok), test_idx)';

        z_tr = z_ok(train_idx);
        y_tr = y_ok(train_idx);
        z_te = z_ok(test_idx);
        y_te = y_ok(test_idx);

        if numel(unique(y_tr)) < 2 || numel(unique(y_te)) < 2
            continue
        end

        n_pos_tr = sum(y_tr == 1);
        n_neg_tr = sum(y_tr == 0);
        n_tot_tr = numel(y_tr);

        try
            switch meth
                case 'smote'
                    [z_tr_fit, y_tr_fit] = smote(z_tr, y_tr, smote_k);
                    y_cat = categorical(y_tr_fit, [0 1], {'0','1'});
                    mdl_cv = TreeBagger(n_trees, z_tr_fit, y_cat, ...
                        'Method','classification', ...
                        'NumPredictorsToSample', max_feat, ...
                        'MinLeafSize', min_leaf, ...
                        'OOBPrediction','off');

                case 'classweight'
                    cost_cv = [0, (n_pos_tr/max(n_neg_tr,1)); 1, 0];
                    y_cat   = categorical(y_tr, [0 1], {'0','1'});
                    mdl_cv  = TreeBagger(n_trees, z_tr, y_cat, ...
                        'Method','classification', ...
                        'NumPredictorsToSample', max_feat, ...
                        'MinLeafSize', min_leaf, ...
                        'Cost', cost_cv, ...
                        'OOBPrediction','off');

                case 'none'
                    y_cat  = categorical(y_tr, [0 1], {'0','1'});
                    mdl_cv = TreeBagger(n_trees, z_tr, y_cat, ...
                        'Method','classification', ...
                        'NumPredictorsToSample', max_feat, ...
                        'MinLeafSize', min_leaf, ...
                        'OOBPrediction','off');
            end

            [~, score_te] = predict(mdl_cv, z_te);
            p_te = score_te(:,2);
            [~, ~, ~, auc_f] = perfcurve(y_te, p_te, 1);
            cv_aucs(f) = auc_f;
        catch
            % fold failed -- leave as NaN
        end
    end
end

% ── kfold_split ──────────────────────────────────────────────────────────
function folds = kfold_split(idx, k)
    n      = numel(idx);
    folds  = cell(k, 1);
    starts = round(linspace(0, n, k+1));
    for i  = 1:k
        folds{i} = idx(starts(i)+1 : starts(i+1));
    end
end