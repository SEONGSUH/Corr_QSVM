%{
Analytic KTA vs. Cor visualization script

Loads the per kernel statistics produced by compute_kta_statistics.m, averages over seeds, combines the moments into the analytic two-class KTA.
%}

clc; clear; close all;

% ----------------------------------------------------------------------
% Configuration (must match the generation sweep)
% ----------------------------------------------------------------------
dim = 3; %:6;
Cor = 0.0:0.1:0.9;
Cor = [Cor 0.99];
random_seeds = 1:5;
angles = [90, 120, 150, 180];

for angle_idx = 1:length(angles)
    angle = angles(angle_idx);

    % (seed, dim, Cor) accumulators
    cls_x_mean_total  = nan([length(random_seeds), length(dim), length(Cor)]);
    cls_y_mean_total  = cls_x_mean_total;
    cls_xy_mean_total = cls_x_mean_total;
    cls_x_var_total   = cls_x_mean_total;
    cls_y_var_total   = cls_x_mean_total;
    cls_xy_var_total  = cls_x_mean_total;

    % ------------------------------------------------------------------
    % Load: each .mat stores (dim x Cor) matrices with only the row of its
    % own dimension populated; extract that row and stack over seeds.
    % ------------------------------------------------------------------
    for random_seed = random_seeds
        for k = 1:length(dim)
            tic
            folder_name = sprintf("random_vectors_angle_%.1f", angle);
            filename = strcat(folder_name, "/kernel_statistics_", num2str(k + 2), ...
                "_seed", num2str(random_seed), ".mat");
            load(filename);

            cls_x_mean_total(random_seed, k, 1:length(Cor))  = squeeze(cls_x_mean(k, 1:length(Cor)));
            cls_y_mean_total(random_seed, k, 1:length(Cor))  = squeeze(cls_y_mean(k, 1:length(Cor)));
            cls_xy_mean_total(random_seed, k, 1:length(Cor)) = squeeze(cls_xy_mean(k, 1:length(Cor)));
            cls_x_var_total(random_seed, k, 1:length(Cor))   = squeeze(cls_x_var(k, 1:length(Cor)));
            cls_y_var_total(random_seed, k, 1:length(Cor))   = squeeze(cls_y_var(k, 1:length(Cor)));
            cls_xy_var_total(random_seed, k, 1:length(Cor))  = squeeze(cls_xy_var(k, 1:length(Cor)));
            toc
        end
    end

    % Average over seeds -> (dim x Cor)
    if length(dim) == 1
        cls_x_mean_mean  = reshape(mean(cls_x_mean_total, 1),  [size(cls_x_mean_total, 2),  size(cls_x_mean_total, 3)]);
        cls_y_mean_mean  = reshape(mean(cls_y_mean_total, 1),  [size(cls_y_mean_total, 2),  size(cls_y_mean_total, 3)]);
        cls_xy_mean_mean = reshape(mean(cls_xy_mean_total, 1), [size(cls_xy_mean_total, 2), size(cls_xy_mean_total, 3)]);
        cls_x_var_mean   = reshape(mean(cls_x_var_total, 1),   [size(cls_x_var_total, 2),   size(cls_x_var_total, 3)]);
        cls_y_var_mean   = reshape(mean(cls_y_var_total, 1),   [size(cls_y_var_total, 2),   size(cls_y_var_total, 3)]);
        cls_xy_var_mean  = reshape(mean(cls_xy_var_total, 1),  [size(cls_xy_var_total, 2),  size(cls_xy_var_total, 3)]);
    else
        cls_x_mean_mean  = squeeze(mean(cls_x_mean_total, 1));
        cls_y_mean_mean  = squeeze(mean(cls_y_mean_total, 1));
        cls_xy_mean_mean = squeeze(mean(cls_xy_mean_total, 1));
        cls_x_var_mean   = squeeze(mean(cls_x_var_total, 1));
        cls_y_var_mean   = squeeze(mean(cls_y_var_total, 1));
        cls_xy_var_mean  = squeeze(mean(cls_xy_var_total, 1));
    end

    % ------------------------------------------------------------------
    % Analytic KTA [Eq. (9)] and figure
    % ------------------------------------------------------------------
    plot_Cor = [0.0:0.1:0.9, 0.99];   % Cor values shown on the x-axis

    fig = figure();
    fig.Position = [347 333 334 175];
    hold on; box on; grid on;

    % numerator:   E[K_xx] + E[K_yy] - 2 E[K_xy]
    % denominator: 2 * sqrt( E[K_xx^2] + E[K_yy^2] + 2 E[K_xy^2] ),
    %              with E[K^2] = Var[K] + E[K]^2
    num = cls_x_mean_mean + cls_y_mean_mean - 2 * cls_xy_mean_mean;
    den = 2 * sqrt(cls_x_var_mean + cls_x_mean_mean.^2 ...
                 + cls_y_var_mean + cls_y_mean_mean.^2 ...
                 + 2 * (cls_xy_var_mean + cls_xy_mean_mean.^2));
    score = num ./ den;

    plot_idx_bool = ismember(single(Cor), single(plot_Cor));
    score_plot = score(:, plot_idx_bool);

    markers = {'o', 's', '^', 'd', 'v', 'p', 'h'};   % one marker per dimension
    colors = lines(length(dim));

    for k = 1:length(dim)
        row = score_plot(k, :);
        plot(plot_Cor, row, ...
            'LineStyle', '-', ...
            'LineWidth', 2, ...
            'Color', colors(k, :), ...
            'Marker', markers{k}, ...
            'MarkerSize', 6, ...
            'MarkerFaceColor', 'w', ...
            'DisplayName', sprintf('Dim %d', dim(k)));
    end

    font_weight = 'bold';
    fs = 15;
    xlabel('\textbf{Cor}', 'Interpreter', 'latex', 'FontSize', fs, 'FontWeight', font_weight);
    ylabel('\textbf{KTA}', 'Interpreter', 'latex', 'FontSize', fs, 'FontWeight', font_weight);

    set(gca, 'FontSize', fs - 2, 'FontWeight', font_weight, 'LineWidth', 1.2);
    xlim([0, 1]);

    legend('show', 'Location', 'NorthWest', 'FontSize', fs - 2, 'FontWeight', font_weight, 'Box', 'off');

    save_filename = sprintf('KTA_Cor_Angle_%g.jpg', angle);
    exportgraphics(fig, save_filename, 'ContentType', 'vector', 'BackgroundColor', 'none');
    fprintf('>> Saved Figure: %s\n', save_filename);
end
