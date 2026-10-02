%{
Statistical analysis of the experiment results.

This script consumes the Excel outputs of the Python experiment scripts, up to three analyses, selected via `analyses` below:
- 'mac_acc: Per-classifier analysis from dim*/result.xlsx, merged over seeds (per dimension, the accuracy vs MAC regression slope, p-value, signed R, mean/std accuracy per Cor level, a paired Wilcoxon signed-rank test of Cor = 0.0 vs. 1.0,  and the MAC vs. Accuracy scatter plots)
- 'mlp: The same statistics for the MLP baseline, read from the CNN training summary.xlsx files.
- 'mac_kta_acc' Empirical KTA computed from the stored kernel statistics, then KTA vs. MAC and Accuracy vs. KTA regressions per dimension, with the two scatter panels (MAC_vs_KTA_*.png, KTA_vs_Acc_*.png).

Classifier selection ('mac_acc' / 'mac_kta_acc'): 
- Set cfg.target_classifier_name to 'ZZ' (ZZFeatureMap QSVM), 'Z' (ZFeatureMap % QSVM), or 'rbf' (classical SVM-RBF). 
- Set cfg.norm_tail to 'Truncated' for the noiseless results or 'Truncated_NoisySim' for the noisy-simulator results ('mac_kta_acc' requires the kernel-statistics files of the ZZ map).
%}

clear; clc; close all

% ----------------------------------------------------------------------
% Configuration
% ----------------------------------------------------------------------
% Anchored to the repo root (the code/ directory two levels above this file),
% so the path holds regardless of MATLAB's working directory.
cfg.RESULT_ROOT = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results');

cfg.dim_list = 3:7;
cfg.datasets = {'Fashion_59', 'Fashion_24', 'CIFAR_02', 'CIFAR_89', 'Number_35'};
% Must match ARCH_CONFIG in training/common.py; note the capitalization
% differs between the two backbones.
cfg.CNN = {'complex'};                   % 'complex' and/or 'Lightweight'
cfg.norm_base = 'BatchNorm';
cfg.norm_tail = 'Truncated';             % 'Truncated' | 'Truncated_NoisySim'
cfg.seed_list = {'seed0', 'seed1', 'seed2'};
cfg.strengths = [0, 0.5, 1.0];           % Cor levels of the lambda = 1 runs
cfg.target_classifier_name = 'ZZ';       % 'ZZ' | 'Z' | 'rbf'

% Cor-level colors (match the correlation-heatmap palette of the paper)
cfg.colors = [0.80 0.80 0.80;            % Cor = 0.0
              0.93 0.53 0.42;            % Cor = 0.5
              0.68 0.04 0.14];           % Cor = 1.0

% Analyses to run
analyses = {'mac_acc', 'mlp', 'mac_kta_acc'};

for a = analyses
    switch a{1}
        case 'mac_acc',     run_mac_acc(cfg);
        case 'mlp',         run_mlp(cfg);
        case 'mac_kta_acc', run_mac_kta_acc(cfg);
    end
end

% ======================================================================
% Analysis 1: MAC vs. accuracy per classifier (dim*/result.xlsx)
% ======================================================================
function run_mac_acc(cfg)
num_dims = length(cfg.dim_list);

for CNN_name = cfg.CNN
    for data_name = cfg.datasets

        fprintf("\n[mac_acc] %s / %s / classifier: %s (merged seeds: %d)\n", ...
            data_name{1}, CNN_name{1}, cfg.target_classifier_name, length(cfg.seed_list));

        % Figures are saved under the last seed's folder
        last_seed = cfg.seed_list{end};
        save_norm_folder = [cfg.norm_base, '_', last_seed, '_', cfg.norm_tail];
        outdir = fullfile(cfg.RESULT_ROOT, data_name{1}, ...
            [CNN_name{1}, '_', save_norm_folder], 'fig_out');

        Cor_Acc_Data = cell(num_dims, 1);
        res_ave_acc = zeros(num_dims, length(cfg.strengths));
        res_std_acc = zeros(num_dims, length(cfg.strengths));
        res_pvals = nan(num_dims, 1);
        res_Rs = nan(num_dims, 1);
        res_non_ave_acc = zeros(num_dims, 1);
        res_wilcoxon_p = nan(num_dims, 1);

        for i = 1:num_dims
            d = cfg.dim_list(i);

            % Merge the seeds' result tables for this dimension
            combined_T = table();
            for s = 1:length(cfg.seed_list)
                norm_folder = [cfg.norm_base, '_', cfg.seed_list{s}, '_', cfg.norm_tail];
                file_path = fullfile(cfg.RESULT_ROOT, data_name{1}, ...
                    [CNN_name{1}, '_', norm_folder], ['dim', num2str(d)], 'result.xlsx');
                combined_T = vertcat(combined_T, read_xlsx(file_path));
            end

            idx_kernel = strcmp(combined_T.("Kernel/FeatureMap"), cfg.target_classifier_name);
            T_sub = combined_T(idx_kernel, :);
            if isempty(T_sub), error('No data for dim%d', d); end

            T_cor = T_sub(T_sub.("Lambda_corr") == 1, :);
            T_non = T_sub(T_sub.("Lambda_corr") == 0, :);

            Cor_Acc_Data{i} = T_cor;
            res_non_ave_acc(i) = mean(T_non.("Accuracy"));

            for s_idx = 1:length(cfg.strengths)
                val_acc = T_cor.("Accuracy")(T_cor.("Correlation_strength") == cfg.strengths(s_idx));
                if ~isempty(val_acc)
                    res_ave_acc(i, s_idx) = mean(val_acc);
                    res_std_acc(i, s_idx) = std(val_acc);
                end
            end

            res_wilcoxon_p(i) = wilcoxon_cor0_vs_cor1(T_cor, ...
                'Correlation_strength', 'Accuracy', 'Fold', d);
        end

        % Scatter panels + regression stats
        [fig, res_pvals, res_Rs] = plot_relation(Cor_Acc_Data, cfg.dim_list, ...
            'MAC', 'Accuracy', 'Lambda_corr', 'Correlation_strength', ...
            cfg.strengths, cfg.colors, [0.0 1.05]);

        if ~exist(outdir, 'dir'), mkdir(outdir); end
        exportgraphics(fig, fullfile(outdir, ...
            ['MAC_vs_Acc_', cfg.target_classifier_name, '.png']), 'Resolution', 300);

        % Text report
        fprintf('\n--- [%s] %s merged results ---\n', data_name{1}, cfg.target_classifier_name);
        disp('Regression slope p-value (Accuracy ~ MAC), per dim:');
        fprintf('%.2e\n', res_pvals);
        disp('Signed correlation R, per dim:');
        fprintf('%.2e\n', res_Rs);
        for s_idx = 1:length(cfg.strengths)
            fprintf('Mean Acc (Cor = %.1f), per dim:\n', cfg.strengths(s_idx));
            fprintf('%.2f\n', res_ave_acc(:, s_idx) * 100);
        end
        disp('Mean Acc (lambda = 0, no correlation loss), per dim:');
        fprintf('%.2f\n', res_non_ave_acc * 100);
        fprintf('Wilcoxon p (Cor 0.0 vs 1.0), per dim:\n');
        fprintf('%.2e\n', res_wilcoxon_p);
    end
end
end

% ======================================================================
% Analysis 2: MLP baseline (CNN training summary.xlsx)
% ======================================================================
function run_mlp(cfg)
num_dims = length(cfg.dim_list);

for data_name = cfg.datasets
    for CNN_name = cfg.CNN
        fprintf("\n[mlp] %s / %s (merged seeds: %d)\n", ...
            data_name{1}, CNN_name{1}, length(cfg.seed_list));

        % Merge the seeds' summary tables
        combined_T = table();
        for s = 1:length(cfg.seed_list)
            norm_folder = [cfg.norm_base, '_', cfg.seed_list{s}];
            file_path = fullfile(cfg.RESULT_ROOT, data_name{1}, ...
                [CNN_name{1}, '_', norm_folder], 'summary.xlsx');
            if ~exist(file_path, 'file')
                warning('File not found, skipping: %s', file_path);
                continue;
            end
            combined_T = vertcat(combined_T, read_xlsx(file_path));
        end

        if isempty(combined_T)
            fprintf('No merged data; skipping dataset.\n');
            continue;
        end

        res_ave_acc = zeros(num_dims, length(cfg.strengths));
        res_std_acc = zeros(num_dims, length(cfg.strengths));
        res_non_ave_acc = zeros(num_dims, 1);
        res_wilcoxon_p = nan(num_dims, 1);

        for i = 1:num_dims
            d = cfg.dim_list(i);
            T_sub = combined_T(combined_T.("dims") == d, :);
            if isempty(T_sub)
                warning('No rows for dim%d', d);
                continue;
            end

            T_cor = T_sub(T_sub.("lambda_corr") == 1, :);
            T_non = T_sub(T_sub.("lambda_corr") == 0, :);

            if ~isempty(T_non)
                res_non_ave_acc(i) = mean(T_non.("test_acc"));
            end

            for s_idx = 1:length(cfg.strengths)
                val_acc = T_cor.("test_acc")(T_cor.("strength") == cfg.strengths(s_idx));
                if ~isempty(val_acc)
                    res_ave_acc(i, s_idx) = mean(val_acc);
                    res_std_acc(i, s_idx) = std(val_acc);
                end
            end

            res_wilcoxon_p(i) = wilcoxon_cor0_vs_cor1(T_cor, ...
                'strength', 'test_acc', 'fold', d);
        end

        % Text report
        fprintf('\n--- [%s] MLP merged results (%s) ---\n', data_name{1}, CNN_name{1});
        for s_idx = 1:length(cfg.strengths)
            fprintf('Mean Acc (Cor = %.1f), per dim:\n', cfg.strengths(s_idx));
            fprintf('%.2f\n', res_ave_acc(:, s_idx) * 100);
        end
        disp('Std Acc per Cor level (mean over dims):');
        for s_idx = 1:length(cfg.strengths)
            fprintf('%.2f\n', mean(res_std_acc(:, s_idx)) * 100);
        end
        disp('Mean Acc (lambda = 0, no correlation loss), per dim:');
        fprintf('%.2f\n', res_non_ave_acc * 100);
        fprintf('Wilcoxon p (Cor 0.0 vs 1.0), per dim:\n');
        fprintf('%.2e\n', res_wilcoxon_p);
    end
end
end

% ======================================================================
% Analysis 3: empirical KTA and MAC-KTA / KTA-Acc regressions
% ======================================================================
function run_mac_kta_acc(cfg)
num_dims = length(cfg.dim_list);

for data_name = cfg.datasets
    for CNN_name = cfg.CNN
        fprintf("\n[mac_kta_acc] %s / %s / classifier: %s\n", ...
            data_name{1}, CNN_name{1}, cfg.target_classifier_name);

        last_seed = cfg.seed_list{end};
        save_norm_folder = [cfg.norm_base, '_', last_seed, '_', cfg.norm_tail];
        final_save_path = fullfile(cfg.RESULT_ROOT, data_name{1}, ...
            [CNN_name{1}, '_', save_norm_folder], 'fig_out');

        Merged_Data_All_Dims = cell(num_dims, 1);

        for i = 1:num_dims
            d = cfg.dim_list(i);
            combined_dim_table = table();

            for s = 1:length(cfg.seed_list)
                norm_folder = [cfg.norm_base, '_', cfg.seed_list{s}, '_', cfg.norm_tail];
                base_path = fullfile(cfg.RESULT_ROOT, data_name{1}, ...
                    [CNN_name{1}, '_', norm_folder]);
                result_file = fullfile(base_path, ['dim', num2str(d)], 'result.xlsx');

                % Kernel-statistics file. The current scripts write the plain
    
                kta_file = fullfile(base_path, 'qsvm_ZZ_results_trainset.xlsx');
     
                if ~exist(kta_file, 'file')
                    error('Kernel-statistics file not found under: %s', base_path);

                end

                T_kta = read_xlsx(kta_file);
                T_res = read_xlsx(result_file);

                % Filter to this dimension / classifier (order preserved)
                T_kta_dim = T_kta(T_kta.Dim == d, :);
                T_res_sub = T_res(strcmp(T_res.("Kernel/FeatureMap"), cfg.target_classifier_name), :);

                % Empirical KTA from per-pairing kernel moments
                num = T_kta_dim.Class0_mean + T_kta_dim.Class1_mean - 2 * T_kta_dim.Diff_mean_q;
                den_inner = T_kta_dim.Class0_var + T_kta_dim.Class0_mean.^2 + ...
                            T_kta_dim.Class1_var + T_kta_dim.Class1_mean.^2 + ...
                            2 * (T_kta_dim.Diff_var + T_kta_dim.Diff_mean_q.^2);
                KTA_val = num ./ (2 * sqrt(den_inner));

                T_res_sub.KTA = KTA_val;
                combined_dim_table = vertcat(combined_dim_table, T_res_sub);
            end

            Merged_Data_All_Dims{i} = combined_dim_table;
        end

        % Scatter panels + regression stats
        if ~exist(final_save_path, 'dir'), mkdir(final_save_path); end

        [f1, pvals_MAC_KTA, ~] = plot_relation(Merged_Data_All_Dims, cfg.dim_list, ...
            'MAC', 'KTA', 'Lambda_corr', 'Correlation_strength', ...
            cfg.strengths, cfg.colors, []);
        exportgraphics(f1, fullfile(final_save_path, ...
            ['MAC_vs_KTA_', cfg.target_classifier_name, '.png']), 'Resolution', 300);

        [f2, pvals_KTA_Acc, ~] = plot_relation(Merged_Data_All_Dims, cfg.dim_list, ...
            'KTA', 'Accuracy', 'Lambda_corr', 'Correlation_strength', ...
            cfg.strengths, cfg.colors, []);
        exportgraphics(f2, fullfile(final_save_path, ...
            ['KTA_vs_Acc_', cfg.target_classifier_name, '.png']), 'Resolution', 300);

        % Text report
        fprintf('\n--- [%s] merged analysis results ---\n', cfg.target_classifier_name);
        disp('1. MAC vs. KTA slope p-value, per dim:');
        fprintf('%.2e\n', pvals_MAC_KTA);
        disp('2. KTA vs. Acc slope p-value, per dim:');
        fprintf('%.2e\n', pvals_KTA_Acc);
    end
end
end

% ======================================================================
% Shared helpers
% ======================================================================
function T = read_xlsx(file_path)
% Read an Excel table with original column names preserved.
if ~exist(file_path, 'file')
    error('File not found: %s', file_path);
end
opts = detectImportOptions(file_path);
opts.VariableNamingRule = 'preserve';
T = readtable(file_path, opts);
end


function p = wilcoxon_cor0_vs_cor1(T_cor, strength_col, acc_col, fold_col, d)
% Paired Wilcoxon signed-rank test: Cor = 0.0 vs. Cor = 1.0 accuracies.
% Pairing is by fold x seed; rows are sorted by fold, and within a fold the (stable) sort preserves the seed concatenation order, so rows line up pairwise.
p = NaN;
T_c0 = T_cor(T_cor.(strength_col) == 0,   :);
T_c1 = T_cor(T_cor.(strength_col) == 1.0, :);
if height(T_c0) == height(T_c1) && height(T_c0) >= 5
    if ismember(fold_col, T_c0.Properties.VariableNames)
        T_c0 = sortrows(T_c0, fold_col);
        T_c1 = sortrows(T_c1, fold_col);
    end
    p = signrank(T_c0.(acc_col), T_c1.(acc_col));
else
    warning('dim%d: Cor 0/1 sample-count mismatch (%d vs %d); Wilcoxon skipped', ...
        d, height(T_c0), height(T_c1));
end
end


function [fig, pvals, Rs] = plot_relation(data_cells, dim_list, x_col, y_col, ...
    lambda_col, strength_col, strengths, colors, xlims)
% Scatter panel (one tile per dimension) of y_col vs. x_col over the lambda = 1 runs, with a fitted regression line per tile. Returns the figure handle and, per dimension, the slope p-value and signed R.
fig = figure('Name', [x_col ' vs ' y_col], 'Visible', 'on');
fig.Position = [853 721 1043 189];
tiledlayout(1, length(dim_list), 'Padding', 'compact', 'TileSpacing', 'compact');

pvals = nan(length(dim_list), 1);
Rs = nan(length(dim_list), 1);
fontsize = 10;

for i = 1:length(dim_list)
    T = data_cells{i};
    if isempty(T), continue; end
    nexttile; hold on;
    T_cor = T(T.(lambda_col) == 1, :);
    x = T_cor.(x_col);
    y = T_cor.(y_col);

    if length(x) > 2 && range(x) > 0
        mdl = fitlm(x, y);
        pvals(i) = mdl.Coefficients.pValue(2);
        Rs(i) = sign(mdl.Coefficients.Estimate(2)) * sqrt(mdl.Rsquared.Ordinary);
        xf = linspace(min(x), max(x), 200);
        plot(xf, predict(mdl, xf(:)), 'k-', 'LineWidth', 1.8);
    end
    for s_idx = 1:length(strengths)
        idx = T_cor.(strength_col) == strengths(s_idx);
        scatter(T_cor.(x_col)(idx), T_cor.(y_col)(idx), 45, 'Marker', 'o', ...
            'MarkerFaceColor', colors(s_idx, :), 'MarkerEdgeColor', colors(s_idx, :), ...
            'MarkerFaceAlpha', 0.5);
    end

    xlabel(x_col, 'FontWeight', 'bold', 'FontSize', fontsize);
    ylabel(y_col, 'FontWeight', 'bold', 'FontSize', fontsize);
    grid on; box on;
    if ~isempty(xlims), xlim(xlims); end
    set(gca, 'FontSize', fontsize, 'FontWeight', 'bold')
    title(sprintf('d=%d', dim_list(i)), 'FontWeight', 'bold', 'FontSize', fontsize + 2);
end
end
