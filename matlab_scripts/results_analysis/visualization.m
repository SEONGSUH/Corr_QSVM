%{
Grouped bar charts of the reported accuracies, per dataset and Cor level.

The numeric matrices below are transcribed from the analysis reports produced by analysis.m

Each matrix is with datasets ordered (a)-(e) and Cor levels. 

Outputs:
- Noiseless: results_lightweight.png, results_complex.png (ZZ QSVM), results_mlp.png, results_svm.png, results_z.png
- Noisy: `results_noisy_zz.png, results_noisy_z.png
%}

clear; clc; close all

% --- Style ---
fontWeight = 'bold';

% Cor-level colors (match the correlation-heatmap palette of the paper)
color_cor00 = [0.80 0.80 0.80];   % Cor = 0.0
color_cor05 = [0.93 0.53 0.42];   % Cor = 0.5
color_cor10 = [0.68 0.04 0.14];   % Cor = 1.0

datasets = {'(a)', '(b)', '(c)', '(d)', '(e)'};
cor_labels = {'Cor = 0.0', 'Cor = 0.5', 'Cor = 1.0'};

% ======================================================================
% Noiseless results: ZZ QSVM (main), MLP / SVM-RBF / Z QSVM (baselines)
% ======================================================================
nl_data_lightweight_zz = [
    90.53   68.00   71.36   71.52   79.76 ;
    91.68   69.89   73.52   76.32   86.48 ;
    95.28   72.43   75.76   79.23   90.75 ;
]';
nl_data_complex_zz = [
    90.62   71.92   71.57   73.81   89.92 ;
    93.71   75.95   75.04   78.75   93.39 ;
    95.25   78.56   78.80   82.64   96.85 ;
]';
nl_std_lightweight_zz = [
    4.55    6.39    6.95    6.55    7.47 ;
    3.80    6.80    5.51    5.46    7.21 ;
    3.03    7.32    5.40    6.82    5.23 ;
]';
nl_std_complex_zz = [
    3.65    6.92    7.25    6.93    4.84 ;
    3.22    6.19    6.36    5.67    4.16 ;
    3.53    6.47    6.24    5.78    2.40 ;
]';

nl_data_lightweight_mlp = [
    94.69   70.08   76.59   77.41   90.85 ;
    94.85   70.21   74.96   78.56   88.96 ;
    94.35   70.27   74.77   78.64   88.72 ;
]';
nl_data_complex_mlp = [
    95.89   77.65   77.95   79.23   96.59 ;
    96.08   77.55   77.60   80.83   97.01 ;
    95.06   77.47   78.37   80.83   97.20 ;
]';
nl_std_lightweight_mlp = [
    3.82    5.94    6.16    7.49    6.35 ;
    3.84    7.64    6.48    7.66    6.51 ;
    4.10    7.16    6.92    6.88    7.17 ;
]';
nl_std_complex_mlp = [
    4.05    7.11    7.15    5.89    3.48 ;
    3.71    7.30    5.96    6.75    2.52 ;
    3.49    7.82    6.44    6.60    3.16 ;
]';

nl_data_lightweight_svm = [
    96.16   74.00   76.99   79.63   89.92 ;
    95.97   72.40   76.75   80.56   90.24 ;
    96.00   73.12   77.04   79.60   91.68 ;
]';
nl_data_complex_svm = [
    96.99   78.61   79.40   82.11   97.94 ;
    97.23   79.60   78.45   83.01   97.94 ;
    96.45   78.91   80.02   83.01   98.27 ;
]';
nl_std_lightweight_svm = [
    3.30    7.10    6.09    5.63    6.44 ;
    3.00    6.70    4.97    5.07    7.09 ;
    3.08    7.20    5.43    6.39    5.06 ;
]';
nl_std_complex_svm = [
    2.04    7.62    4.94    5.09    2.51 ;
    2.01    5.66    5.42    4.32    2.11 ;
    2.91    6.06    6.25    5.69    1.76 ;
]';

nl_data_lightweight_z = [
    95.17   72.99   75.44   77.68   88.45 ;
    95.71   71.92   76.85   80.74   89.52 ;
    95.65   73.01   76.72   79.55   91.44 ;
]';
nl_data_complex_z = [
    95.95   77.95   78.42   80.16   96.53 ;
    96.77   79.38   78.13   82.43   98.05 ;
    95.92   79.04   79.33   81.89   98.02 ;
]';
nl_std_lightweight_z = [
    3.34    7.37    6.20    5.34    7.10 ;
    2.84    6.70    4.90    4.64    7.30 ;
    3.89    7.37    5.66    6.14    4.80 ;
]';
nl_std_complex_z = [
    2.20    7.55    5.29    5.58    2.37 ;
    2.09    5.84    5.54    4.74    2.18 ;
    3.34    5.94    6.11    5.72    1.81 ;
]';

% ======================================================================
% Noisy-simulator results: ZZ and Z QSVM
% ======================================================================
ns_data_lightweight_zz = [
    89.52   67.68   71.09   71.12   79.31 ;
    90.75   69.57   74.99   75.13   84.70 ;
    93.95   71.73   75.84   79.04   90.13 ;
]';
ns_data_complex_zz = [
    90.37   71.63   72.59   73.76   88.96 ;
    93.31   75.87   75.00   76.97   91.63 ;
    94.54   78.06   78.72   81.89   96.35 ;
]';
ns_std_lightweight_zz = [
    5.00    6.98    6.75    7.57    8.36 ;
    4.35    6.61    5.46    5.66    7.08 ;
    3.04    7.36    5.70    6.91    5.22 ;
]';
ns_std_complex_zz = [
    4.28    6.89    7.44    7.22    4.31 ;
    3.63    5.53    6.46    5.24    4.60 ;
    3.73    6.26    6.70    6.04    2.61 ;
]';

ns_data_lightweight_z = [
    94.72   72.21   75.89   77.55   88.13 ;
    96.16   72.59   78.85   80.91   89.83 ;
    94.93   72.29   77.71   80.00   92.03 ;
]';
ns_data_complex_z = [
    96.43   78.48   78.83   79.65   96.64 ;
    97.55   79.68   78.45   83.97   98.11 ;
    94.80   79.04   80.27   82.72   98.19 ;
]';
ns_std_lightweight_z = [
    2.63    6.62    5.87    5.71    7.32 ;
    2.96    6.63    5.51    4.55    6.52 ;
    3.93    7.38    5.72    6.82    4.62 ;
]';
ns_std_complex_z = [
    2.30    7.75    5.59    5.76    2.28 ;
    2.01    6.13    5.49    4.68    2.27 ;
    3.93    6.11    6.43    5.61    1.81 ;
]';

% ======================================================================
% Noiseless ZZ QSVM: single-panel figures with value labels
% (Cor = 0.0 and Cor = 1.0 bars labeled)
% ======================================================================
fontSize = 11;
y_limits = [65, 100];

fig1 = make_single_panel_bars(1, nl_data_lightweight_zz, nl_std_lightweight_zz, ...
    'Lightweight CNN', datasets, cor_labels, ...
    color_cor00, color_cor05, color_cor10, fontSize, fontWeight, y_limits);
exportgraphics(fig1, 'results_lightweight.png', 'Resolution', 300);

fig2 = make_single_panel_bars(2, nl_data_complex_zz, nl_std_complex_zz, ...
    'Complex CNN', datasets, cor_labels, ...
    color_cor00, color_cor05, color_cor10, fontSize, fontWeight, y_limits);
exportgraphics(fig2, 'results_complex.png', 'Resolution', 300);

% ======================================================================
% Noiseless baselines: MLP / SVM-RBF / Z QSVM, two-panel figures
% ======================================================================
fontSize = 9;
y_limits = [65, 101];

fig3 = make_two_panel_bars(3, nl_data_lightweight_mlp, nl_std_lightweight_mlp, ...
    nl_data_complex_mlp, nl_std_complex_mlp, datasets, cor_labels, ...
    color_cor00, color_cor05, color_cor10, fontSize, fontWeight, y_limits);
exportgraphics(fig3, 'results_mlp.png', 'Resolution', 300);

fig4 = make_two_panel_bars(4, nl_data_lightweight_svm, nl_std_lightweight_svm, ...
    nl_data_complex_svm, nl_std_complex_svm, datasets, cor_labels, ...
    color_cor00, color_cor05, color_cor10, fontSize, fontWeight, y_limits);
exportgraphics(fig4, 'results_svm.png', 'Resolution', 300);

fig5 = make_two_panel_bars(5, nl_data_lightweight_z, nl_std_lightweight_z, ...
    nl_data_complex_z, nl_std_complex_z, datasets, cor_labels, ...
    color_cor00, color_cor05, color_cor10, fontSize, fontWeight, y_limits);
exportgraphics(fig5, 'results_z.png', 'Resolution', 300);

% ======================================================================
% Noisy-simulator results: ZZ and Z QSVM, two-panel figures
% ======================================================================
fig6 = make_two_panel_bars(6, ns_data_lightweight_zz, ns_std_lightweight_zz, ...
    ns_data_complex_zz, ns_std_complex_zz, datasets, cor_labels, ...
    color_cor00, color_cor05, color_cor10, fontSize, fontWeight, [65, 100]);
exportgraphics(fig6, 'results_noisy_zz.png', 'Resolution', 300);

fig7 = make_two_panel_bars(7, ns_data_lightweight_z, ns_std_lightweight_z, ...
    ns_data_complex_z, ns_std_complex_z, datasets, cor_labels, ...
    color_cor00, color_cor05, color_cor10, fontSize, fontWeight, [65, 101]);
exportgraphics(fig7, 'results_noisy_z.png', 'Resolution', 300);

% ----------------------------------------------------------------------
% Helper: single-panel grouped bar chart with value labels on the
% Cor = 0.0 and Cor = 1.0 bars
% ----------------------------------------------------------------------
function fig = make_single_panel_bars(fig_no, data, stds, panel_title, ...
    datasets, cor_labels, c00, c05, c10, fontSize, fontWeight, y_limits)

    fig = figure(fig_no);
    fig.Position = [955 423 551 247];
    b = bar(data, 'grouped');
    hold on;
    b(1).FaceColor = c00; b(2).FaceColor = c05; b(3).FaceColor = c10;
    for i = 1:numel(b)
        errorbar(b(i).XEndPoints, b(i).YEndPoints, stds(:, i), ...
            'k', 'LineStyle', 'none', 'LineWidth', 1.0, 'CapSize', 4);
    end
    for i = 1:numel(b)
        if i == 1 || i == 3
            xtips = b(i).XEndPoints;
            ytips = b(i).YEndPoints;
            labels = string(ytips);
            offset = stds(:, i)' + 0.2;
            text(xtips, ytips + offset, labels, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', fontSize, 'FontWeight', fontWeight);
        end
    end
    set(gca, 'XTickLabel', datasets, 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    ylim(y_limits);
    ylabel('Average Accuracy (%)', 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    xlabel('Dataset', 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    title(panel_title, 'FontSize', fontSize + 3, 'FontWeight', fontWeight);
    legend(cor_labels, 'Location', 'best', 'FontSize', fontSize, 'FontWeight', fontWeight);
    grid on;
    set(findall(fig, '-property', 'FontWeight'), 'FontWeight', fontWeight);
end

% ----------------------------------------------------------------------
% Helper: two-panel (Lightweight | Complex) grouped bar chart with a
% shared bottom legend
% ----------------------------------------------------------------------
function fig = make_two_panel_bars(fig_no, data_lightweight, std_lightweight, ...
    data_complex, std_complex, datasets, cor_labels, c00, c05, c10, ...
    fontSize, fontWeight, y_limits)

    fig = figure(fig_no);
    fig.Position = [969 626 564 216];
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    b1 = bar(data_lightweight, 'grouped');
    hold on;
    b1(1).FaceColor = c00; b1(2).FaceColor = c05; b1(3).FaceColor = c10;
    for i = 1:numel(b1)
        errorbar(b1(i).XEndPoints, b1(i).YEndPoints, std_lightweight(:, i), ...
            'k', 'LineStyle', 'none', 'LineWidth', 1.0, 'CapSize', 4);
    end
    set(gca, 'XTickLabel', datasets, 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    ylim(y_limits);
    ylabel('Ave. Accuracy (%)', 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    xlabel('Dataset', 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    title('Lightweight CNN', 'FontSize', fontSize + 3, 'FontWeight', fontWeight);
    grid on;

    nexttile;
    b2 = bar(data_complex, 'grouped');
    hold on;
    b2(1).FaceColor = c00; b2(2).FaceColor = c05; b2(3).FaceColor = c10;
    for i = 1:numel(b2)
        errorbar(b2(i).XEndPoints, b2(i).YEndPoints, std_complex(:, i), ...
            'k', 'LineStyle', 'none', 'LineWidth', 1.0, 'CapSize', 4);
    end
    set(gca, 'XTickLabel', datasets, 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    ylim(y_limits);
    ylabel('Ave. Accuracy (%)', 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    xlabel('Dataset', 'FontSize', fontSize + 2, 'FontWeight', fontWeight);
    title('Complex CNN', 'FontSize', fontSize + 3, 'FontWeight', fontWeight);
    grid on;

    lgd = legend(ax1, cor_labels, 'NumColumns', 3, 'FontSize', fontSize + 1.5, 'FontWeight', fontWeight);
    lgd.Layout.Tile = 'south';

    set(findall(fig, '-property', 'FontWeight'), 'FontWeight', fontWeight);
end
