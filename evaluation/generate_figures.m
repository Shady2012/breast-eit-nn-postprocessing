%% Phase 5 - generate_figures.m
% =========================================================================
% ML-Enhanced EIT for Early Breast Cancer Detection
%
% Figures are saved as .fig (editable) and .eps (vector, publication).
%
% Dependencies:
%   evaluation/evaluation_results.mat (from evaluate_all_methods.m)
%
% Output:
%   evaluation/figures/fig1_comparison_bar.fig/.eps
%   evaluation/figures/fig2a_scatter_radial.fig/.eps
%   evaluation/figures/fig2b_scatter_xyz.fig/.eps
%   evaluation/figures/fig3_noise_robustness.fig/.eps
%   evaluation/figures/fig4_size_performance.fig/.eps
%   evaluation/figures/fig5_improvement.fig/.eps
% =========================================================================

clear; clc; close all;

fprintf('============================================\n');
fprintf('  PHASE 5: FIGURE GENERATION\n');
fprintf('============================================\n\n');

%% ========================================================================
% 1. LOAD EVALUATION RESULTS
% =========================================================================
fprintf('=== Loading Evaluation Results ===\n');

search_dirs = {'.', '..', '../evaluation', 'evaluation'};
eval_path = find_file('evaluation_results.mat', search_dirs);
fprintf('  Loaded: %s\n', eval_path);
E = load(eval_path);

methods    = E.methods;
defs       = E.defs;
scatter_d  = E.scatter_data;
sig        = E.significance;
improve    = E.improvement;
n_methods  = length(methods);

% Create output directory
fig_dir = 'figures';
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
    fprintf('  Created directory: %s/\n', fig_dir);
end

fprintf('\n');

%% ========================================================================
% 2. GLOBAL FIGURE SETTINGS
% =========================================================================

% Publication text settings (22pt bold as requested)
FONT_SIZE   = 22;
FONT_WEIGHT = 'bold';
FONT_NAME   = 'Arial';

% Colour palette
% Baselines: blue/grey family
colors_baseline = [ ...
    0.55 0.70 0.90;   % 1: NOSER + fixed50     (light blue)
    0.40 0.55 0.80;   % 2: NOSER + Otsu         (medium blue)
    0.20 0.40 0.70;   % 3: Tikhonov + fixed50    (dark blue)
    0.10 0.25 0.55;   % 4: Tikhonov + Otsu       (very dark blue)
    0.60 0.60 0.60;   % 5: Laplace + fixed50     (medium grey)
    0.40 0.40 0.40;   % 6: Laplace + Otsu        (dark grey)
];

% ML approaches: warm colours
colors_ml = [ ...
    0.95 0.60 0.20;   % 7: Approach A (orange)
    0.20 0.70 0.30;   % 8: Approach B (green) - best, stands out
    0.85 0.25 0.25;   % 9: Approach C (red/coral)
];

colors_all = [colors_baseline; colors_ml];

% Subset for Figures 3 & 4 (5 methods)
fig34_idx  = defs.fig34_indices;  % [4, 3, 7, 8, 9]
fig34_names = defs.fig34_names;
colors_fig34 = colors_all(fig34_idx, :);

%% ========================================================================
% FIGURE 1: HEAD-TO-HEAD COMPARISON BAR CHART
% =========================================================================
fprintf('=== Figure 1: Head-to-Head Comparison ===\n');

fig1 = figure('Name', 'Fig 1: Comparison', ...
    'Position', [30, 100, 1800, 550], 'Color', 'w');

% Gather data for all 9 methods
pos_vals      = arrayfun(@(m) m.overall.pos_mean, methods);
size_vals     = arrayfun(@(m) m.overall.size_abs_mean, methods);
contrast_vals = arrayfun(@(m) m.overall.contrast_mean, methods);

short_names = {'NOS+f50', 'NOS+Otsu', 'Tik+f50', 'Tik+Otsu', ...
               'Lap+f50', 'Lap+Otsu', 'ML-A', 'ML-B', 'ML-C'};

% --- Panel 1: Position Error ---
subplot(1, 3, 1);
b1 = bar(pos_vals, 'FaceColor', 'flat');
b1.CData = colors_all;
set(gca, 'XTickLabel', short_names, 'XTickLabelRotation', 45);
ylabel('Mean Position Error (cm)');
title('Position');
grid on;
% Reference line: balanced baseline
hold on;
yline(methods(3).overall.pos_mean, 'k--', 'LineWidth', 1.5);
% Star on best
[~, best_pos_idx] = min(pos_vals);
text(best_pos_idx, pos_vals(best_pos_idx) + 0.02, '*', ...
    'FontSize', FONT_SIZE, 'FontWeight', FONT_WEIGHT, ...
    'HorizontalAlignment', 'center', 'Color', [0.8 0.1 0.1]);
hold off;

% --- Panel 2: Size Error ---
subplot(1, 3, 2);
b2 = bar(size_vals, 'FaceColor', 'flat');
b2.CData = colors_all;
set(gca, 'XTickLabel', short_names, 'XTickLabelRotation', 45);
ylabel('Mean Size Error (mm)');
title('Size');
grid on;
hold on;
yline(methods(3).overall.size_abs_mean, 'k--', 'LineWidth', 1.5);
[~, best_size_idx] = min(size_vals);
text(best_size_idx, size_vals(best_size_idx) + 0.3, '*', ...
    'FontSize', FONT_SIZE, 'FontWeight', FONT_WEIGHT, ...
    'HorizontalAlignment', 'center', 'Color', [0.8 0.1 0.1]);
hold off;

% --- Panel 3: Contrast Error ---
subplot(1, 3, 3);
b3 = bar(contrast_vals, 'FaceColor', 'flat');
b3.CData = colors_all;
set(gca, 'XTickLabel', short_names, 'XTickLabelRotation', 45);
ylabel('Mean Contrast Error');
title('Contrast');
grid on;
hold on;
yline(methods(3).overall.contrast_mean, 'k--', 'LineWidth', 1.5);
[~, best_con_idx] = min(contrast_vals);
text(best_con_idx, contrast_vals(best_con_idx) + 0.05, '*', ...
    'FontSize', FONT_SIZE, 'FontWeight', FONT_WEIGHT, ...
    'HorizontalAlignment', 'center', 'Color', [0.8 0.1 0.1]);
hold off;

sgtitle('Comparative Performance: Baselines vs ML Approaches', ...
    'FontSize', FONT_SIZE, 'FontWeight', FONT_WEIGHT);

format_figure(fig1, FONT_SIZE, FONT_WEIGHT, FONT_NAME);
%save_figure(fig1, fullfile(fig_dir, 'fig1_comparison_bar'));
%fprintf('  Saved.\n\n');


%% ========================================================================
% FIGURE 2a: PREDICTED-VS-TRUE SCATTER (RADIAL DISTANCE) - FOR PAPER
% =========================================================================
fprintf('=== Figure 2a: Scatter Plots (Radial) ===\n');

pred = scatter_d.approach_B.predictions;   % 864 x 5
gt   = scatter_d.approach_B.ground_truth;  % 864 x 5
nl   = E.noise_levels;

% Radial distance from breast centre (assumed origin)
pred_r = sqrt(pred(:,1).^2 + pred(:,2).^2 + pred(:,3).^2);
true_r = sqrt(gt(:,1).^2 + gt(:,2).^2 + gt(:,3).^2);

% Noise level colour mapping
noise_cmap = [0.2 0.7 0.3;   % Clean: green
              0.2 0.5 0.8;   % 80 dB: blue
              0.9 0.6 0.2;   % 60 dB: orange
              0.8 0.2 0.2];  % 40 dB: red
noise_vals = [Inf, 80, 60, 40];
noise_lbls = {'Clean', '80 dB', '60 dB', '40 dB'};

fig2a = figure('Name', 'Fig 2a: Scatter (Radial)', ...
    'Position', [30, 100, 1800, 550], 'Color', 'w');

% --- Panel 1: Position (radial distance) ---
subplot(1, 3, 1);
hold on;
for i = 1:4
    mask = nl == noise_vals(i);
    scatter(true_r(mask), pred_r(mask), 25, noise_cmap(i,:), 'filled', ...
        'MarkerFaceAlpha', 0.6);
end
max_r = max([true_r; pred_r]) * 1.05;
plot([0, max_r], [0, max_r], 'k--', 'LineWidth', 1.5);
hold off;
xlabel('True Radial Distance (cm)');
ylabel('Predicted Radial Distance (cm)');
title('Position');
R2_r = 1 - sum((pred_r - true_r).^2) / sum((true_r - mean(true_r)).^2);
text(0.05, 0.92, sprintf('R^2 = %.3f', R2_r), 'Units', 'normalized', ...
    'FontSize', FONT_SIZE-4, 'FontWeight', FONT_WEIGHT);
legend(noise_lbls, 'Location', 'southeast', 'FontSize', FONT_SIZE-6);
grid on; axis equal;

% --- Panel 2: Diameter ---
subplot(1, 3, 2);
hold on;
for i = 1:4
    mask = nl == noise_vals(i);
    scatter(gt(mask,4), pred(mask,4), 25, noise_cmap(i,:), 'filled', ...
        'MarkerFaceAlpha', 0.6);
end
diam_range = [8 32];
plot(diam_range, diam_range, 'k--', 'LineWidth', 1.5);
hold off;
xlabel('True Diameter (mm)');
ylabel('Predicted Diameter (mm)');
title('Size');
R2_d = 1 - sum((pred(:,4) - gt(:,4)).^2) / sum((gt(:,4) - mean(gt(:,4))).^2);
text(0.05, 0.92, sprintf('R^2 = %.3f', R2_d), 'Units', 'normalized', ...
    'FontSize', FONT_SIZE-4, 'FontWeight', FONT_WEIGHT);
grid on; axis equal; xlim(diam_range); ylim(diam_range);

% --- Panel 3: Contrast ---
subplot(1, 3, 3);
hold on;
for i = 1:4
    mask = nl == noise_vals(i);
    scatter(gt(mask,5), pred(mask,5), 25, noise_cmap(i,:), 'filled', ...
        'MarkerFaceAlpha', 0.6);
end
con_range = [1.5 5.5];
plot(con_range, con_range, 'k--', 'LineWidth', 1.5);
hold off;
xlabel('True Contrast Ratio');
ylabel('Predicted Contrast Ratio');
title('Contrast');
R2_c = 1 - sum((pred(:,5) - gt(:,5)).^2) / sum((gt(:,5) - mean(gt(:,5))).^2);
text(0.05, 0.92, sprintf('R^2 = %.3f', R2_c), 'Units', 'normalized', ...
    'FontSize', FONT_SIZE-4, 'FontWeight', FONT_WEIGHT);
grid on; axis equal; xlim(con_range); ylim(con_range);

sgtitle('Approach B: Predicted vs True (All Test Samples)', ...
    'FontSize', FONT_SIZE, 'FontWeight', FONT_WEIGHT);

format_figure(fig2a, FONT_SIZE, FONT_WEIGHT, FONT_NAME);
%save_figure(fig2a, fullfile(fig_dir, 'fig2a_scatter_radial'));
%fprintf('  Saved.\n\n');


%% ========================================================================
% FIGURE 2b: PREDICTED-VS-TRUE SCATTER (X, Y, Z SEPARATE) - DIAGNOSTIC
% =========================================================================
fprintf('=== Figure 2b: Scatter Plots (X, Y, Z) ===\n');

fig2b = figure('Name', 'Fig 2b: Scatter (XYZ)', ...
    'Position', [30, 100, 1800, 550], 'Color', 'w');

axis_labels = {'X Position (cm)', 'Y Position (cm)', 'Z Position (cm)'};

for ax = 1:3
    subplot(1, 3, ax);
    hold on;
    for i = 1:4
        mask = nl == noise_vals(i);
        scatter(gt(mask, ax), pred(mask, ax), 25, noise_cmap(i,:), ...
            'filled', 'MarkerFaceAlpha', 0.6);
    end
    ax_range = [min([gt(:,ax); pred(:,ax)])-0.5, ...
                max([gt(:,ax); pred(:,ax)])+0.5];
    plot(ax_range, ax_range, 'k--', 'LineWidth', 1.5);
    hold off;
    xlabel(sprintf('True %s', axis_labels{ax}));
    ylabel(sprintf('Predicted %s', axis_labels{ax}));
    title(axis_labels{ax});
    R2_ax = 1 - sum((pred(:,ax) - gt(:,ax)).^2) / ...
            sum((gt(:,ax) - mean(gt(:,ax))).^2);
    text(0.05, 0.92, sprintf('R^2 = %.3f', R2_ax), 'Units', 'normalized', ...
        'FontSize', FONT_SIZE-4, 'FontWeight', FONT_WEIGHT);
    grid on; axis equal;
end

sgtitle('Approach B: Per-Axis Position Prediction', ...
    'FontSize', FONT_SIZE, 'FontWeight', FONT_WEIGHT);

format_figure(fig2b, FONT_SIZE, FONT_WEIGHT, FONT_NAME);
%save_figure(fig2b, fullfile(fig_dir, 'fig2b_scatter_xyz'));
%fprintf('  Saved.\n\n');


%% ========================================================================
% FIGURE 3: NOISE ROBUSTNESS (5 METHODS)
% =========================================================================
fprintf('=== Figure 3: Noise Robustness ===\n');

fig3 = figure('Name', 'Fig 3: Noise Robustness', ...
    'Position', [50, 150, 1000, 600], 'Color', 'w');

n_noise = length(defs.noise_labels);
n_fig34 = length(fig34_idx);
bar_data_noise = zeros(n_noise, n_fig34);

for i = 1:n_fig34
    mi = fig34_idx(i);
    bar_data_noise(:, i) = methods(mi).by_noise.pos(:);
end

b3 = bar(bar_data_noise, 'grouped');
for i = 1:n_fig34
    b3(i).FaceColor = colors_fig34(i, :);
end

set(gca, 'XTickLabel', defs.noise_labels);
ylabel('Mean Position Error (cm)');
title('Position Error Across Noise Levels');

% Clean legend labels
legend_labels_34 = {'Tik + Otsu', 'Tik + fixed50', ...
                    'ML-A (Direct)', 'ML-B (Post-Proc)', 'ML-C (Hybrid)'};
legend(legend_labels_34, 'Location', 'northeastoutside');
grid on;

format_figure(fig3, FONT_SIZE, FONT_WEIGHT, FONT_NAME);
%save_figure(fig3, fullfile(fig_dir, 'fig3_noise_robustness'));
%fprintf('  Saved.\n\n');


%% ========================================================================
% FIGURE 4: SIZE-DEPENDENT PERFORMANCE (5 METHODS)
% =========================================================================
fprintf('=== Figure 4: Size-Dependent Performance ===\n');

fig4 = figure('Name', 'Fig 4: Size Performance', ...
    'Position', [50, 100, 1600, 600], 'Color', 'w');

n_size = size(defs.size_bands, 1);
short_size_labels = {'Small', 'Medium', 'Large'};

% --- Panel 1: Position error by size band ---
subplot(1, 2, 1);
bar_data_pos = zeros(n_size, n_fig34);
for i = 1:n_fig34
    mi = fig34_idx(i);
    bar_data_pos(:, i) = methods(mi).by_size.pos(:);
end

b4a = bar(bar_data_pos, 'grouped');
for i = 1:n_fig34
    b4a(i).FaceColor = colors_fig34(i, :);
end
set(gca, 'XTickLabel', short_size_labels);
xlabel('Tumour Size Band');
ylabel('Mean Position Error (cm)');
title('Position Error by Tumour Size');
legend(legend_labels_34, 'Location', 'northeastoutside');
grid on;

% --- Panel 2: Size error by size band ---
subplot(1, 2, 2);
bar_data_size = zeros(n_size, n_fig34);
for i = 1:n_fig34
    mi = fig34_idx(i);
    bar_data_size(:, i) = methods(mi).by_size.size_abs(:);
end

b4b = bar(bar_data_size, 'grouped');
for i = 1:n_fig34
    b4b(i).FaceColor = colors_fig34(i, :);
end
set(gca, 'XTickLabel', short_size_labels);
xlabel('Tumour Size Band');
ylabel('Mean Size Error (mm)');
title('Size Error by Tumour Size');
legend(legend_labels_34, 'Location', 'northeastoutside');
grid on;

sgtitle('Performance by Tumour Diameter', ...
    'FontSize', FONT_SIZE, 'FontWeight', FONT_WEIGHT);

format_figure(fig4, FONT_SIZE, FONT_WEIGHT, FONT_NAME);
%save_figure(fig4, fullfile(fig_dir, 'fig4_size_performance'));
%fprintf('  Saved.\n\n');


%% ========================================================================
% FIGURE 5: IMPROVEMENT SUMMARY (HORIZONTAL BAR)
% =========================================================================
fprintf('=== Figure 5: Improvement Summary ===\n');

fig5 = figure('Name', 'Fig 5: Improvement', ...
    'Position', [100, 200, 900, 500], 'Color', 'w');

improve_vals  = [improve.pos_pct, improve.size_pct, improve.contrast_pct];
improve_names = {'Position', 'Size', 'Contrast'};
improve_colors = [0.20 0.70 0.30;   % green for all (all improvements)
                  0.20 0.70 0.30;
                  0.20 0.70 0.30];

% If position improvement is negative, colour it differently
if improve_vals(1) < 0
    improve_colors(1, :) = [0.85 0.25 0.25];  % red for degradation
end

bh = barh(improve_vals, 'FaceColor', 'flat');
bh.CData = improve_colors;

set(gca, 'YTickLabel', improve_names);
xlabel('Improvement Over Balanced Baseline (%)');
title('Approach B vs Tikhonov + fixed50');
grid on;

% Add value labels on each bar
for i = 1:3
    if improve_vals(i) >= 0
        x_pos = improve_vals(i) + 1.5;
        halign = 'left';
    else
        x_pos = improve_vals(i) - 1.5;
        halign = 'right';
    end
    text(x_pos, i, sprintf('%+.1f%%', improve_vals(i)), ...
        'FontSize', FONT_SIZE-2, 'FontWeight', FONT_WEIGHT, ...
        'HorizontalAlignment', halign, 'VerticalAlignment', 'middle');
end

% Zero reference line
hold on;
xline(0, 'k-', 'LineWidth', 1.5);
hold off;

format_figure(fig5, FONT_SIZE, FONT_WEIGHT, FONT_NAME);
%save_figure(fig5, fullfile(fig_dir, 'fig5_improvement'));
%fprintf('  Saved.\n\n');


%% ========================================================================
fprintf('============================================\n');
fprintf('  ALL FIGURES GENERATED\n');
fprintf('  Output directory: %s/\n', fig_dir);
fprintf('============================================\n');


%% ========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function fpath = find_file(filename, search_dirs)
% FIND_FILE  Search for a file in multiple directories.
    for i = 1:length(search_dirs)
        candidate = fullfile(search_dirs{i}, filename);
        if exist(candidate, 'file')
            fpath = candidate;
            return;
        end
    end
    error('Cannot find %s. Searched: %s', filename, strjoin(search_dirs, ', '));
end


function format_figure(fig, font_size, font_weight, font_name)
% FORMAT_FIGURE  Apply 22pt bold formatting to all text elements in a figure.
%
%   This walks through all axes, labels, titles, legends, and tick labels
%   and sets them to the specified font size and weight.

    all_axes = findall(fig, 'Type', 'axes');
    for i = 1:length(all_axes)
        ax = all_axes(i);

        % Axis tick labels
        set(ax, 'FontSize', font_size, 'FontWeight', font_weight, ...
                'FontName', font_name);

        % Axis labels
        if ~isempty(ax.XLabel)
            set(ax.XLabel, 'FontSize', font_size, 'FontWeight', font_weight, ...
                           'FontName', font_name);
        end
        if ~isempty(ax.YLabel)
            set(ax.YLabel, 'FontSize', font_size, 'FontWeight', font_weight, ...
                           'FontName', font_name);
        end

        % Title
        if ~isempty(ax.Title)
            set(ax.Title, 'FontSize', font_size, 'FontWeight', font_weight, ...
                          'FontName', font_name);
        end
    end

    % Legends
    all_legends = findall(fig, 'Type', 'legend');
    for i = 1:length(all_legends)
        set(all_legends(i), 'FontSize', font_size - 6, ...
            'FontWeight', font_weight, 'FontName', font_name);
    end

    % Text annotations
    all_text = findall(fig, 'Type', 'text');
    for i = 1:length(all_text)
        if all_text(i).FontSize < font_size
            set(all_text(i), 'FontWeight', font_weight, ...
                             'FontName', font_name);
        end
    end
end


function save_figure(fig, filepath)
% SAVE_FIGURE  Save figure as .fig (editable) and .eps (publication vector).

    % Save as MATLAB .fig
    savefig(fig, [filepath, '.fig']);
    fprintf('    -> %s.fig\n', filepath);

    % Save as .eps (vector format for publication)
    % Use painters renderer for clean vector output
    print(fig, [filepath, '.eps'], '-depsc', '-painters', '-r300');
    fprintf('    -> %s.eps\n', filepath);
end
