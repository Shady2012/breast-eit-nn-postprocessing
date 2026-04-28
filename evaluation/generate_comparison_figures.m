% =========================================================================
%   GENERATE VISUAL COMPARISON FIGURE (v2 - Cross-Sectional Slices)
%   Ground Truth vs EIDORS Reconstruction vs ML Prediction
%
%   Uses 2D cross-sectional slices at the tumour z-plane instead of 3D
%   mesh renders. This produces larger, cleaner panels suitable for
%   IEEE TBME double-column format.
%
%   Layout: 3 rows x 2 columns
%     Column 1: Medium tumour, clean
%     Column 2: Small tumour, 40 dB noise
%     Row 1: Ground truth
%     Row 2: Tikhonov reconstruction (+ true boundary overlay)
%     Row 3: ML-predicted parameters (+ true boundary overlay)
%
%   Dependencies:
%     - geometry/build_breast_model.m
%     - baseline/baseline_results.mat
%     - networks/results_approach_B.mat
%     - dataset/training_dataset.mat
%
%   Place in: evaluation/generate_visual_comparison.m
%   Project: ML-Enhanced EIT for Early Breast Cancer Detection
% =========================================================================

clear; clc; close all;

fprintf('=== Visual Comparison Figure Generator (v2) ===\n\n');

%% ========================================================================
% 1. LOAD ALL REQUIRED DATA
% =========================================================================
fprintf('--- Loading data files ---\n');

find_file = @(fname, dirs) find_file_impl(fname, dirs);

dataset_path  = find_file('training_dataset.mat', {'../dataset', '.', 'dataset'});
baseline_path = find_file('baseline_results.mat', {'../baseline', '.', 'baseline'});
ml_path       = find_file('results_approach_B.mat', {'../networks', '.', 'networks'});

fprintf('  Loading dataset...\n');
data = load(dataset_path);
fprintf('  Loading baseline results...\n');
bl = load(baseline_path);
fprintf('  Loading ML results (Approach B)...\n');
ml = load(ml_path);
fprintf('  All files loaded.\n\n');

%% ========================================================================
% 2. REBUILD FORWARD MODEL AND PRECOMPUTE GEOMETRY
% =========================================================================
fprintf('--- Rebuilding forward model ---\n');

fmdl = build_breast_model();
background_cond = 0.3;

node_coords = fmdl.nodes;
elem_nodes  = fmdl.elems;
n_elems     = size(elem_nodes, 1);
n_verts     = size(elem_nodes, 2);

% Element centres
elem_centres = zeros(n_elems, 3);
for d = 1:3
    col_sum = zeros(n_elems, 1);
    for v = 1:n_verts
        col_sum = col_sum + node_coords(elem_nodes(:, v), d);
    end
    elem_centres(:, d) = col_sum / n_verts;
end

% Element volumes (for size extraction in Tikhonov)
elem_volumes = get_elem_volume(fmdl);

fprintf('  Model: %d elements, %d nodes\n\n', n_elems, size(node_coords, 1));

%% ========================================================================
% 3. EXTRACT TEST SET DATA
% =========================================================================
fprintf('--- Extracting test set info ---\n');

test_idx    = data.split.test;
test_params = data.tumour_params(test_idx, :);
test_noise  = data.noise_levels(test_idx, :);

% ML predictions (top-level field in results_approach_B.mat)
if isfield(ml, 'predictions')
    ml_predictions = ml.predictions;
elseif isfield(ml, 'results') && isfield(ml.results, 'predictions')
    ml_predictions = ml.results.predictions;
else
    error('Cannot find ML predictions field.');
end

fprintf('  Test samples: %d\n', size(test_params, 1));

%% ========================================================================
% 4. SELECT TWO REPRESENTATIVE TEST SAMPLES
% =========================================================================
fprintf('\n--- Selecting representative cases ---\n');

% Position errors for median-based selection (typical, not cherry-picked)
pos_errors_B = sqrt(sum((ml_predictions(:,1:3) - test_params(:,1:3)).^2, 2));

% CASE 1: Medium tumour (18-22 mm), clean, moderate contrast
candidates_1 = find( ...
    test_params(:,4) >= 18 & test_params(:,4) <= 22 & ...
    test_noise == Inf & ...
    test_params(:,5) >= 2.5 & test_params(:,5) <= 4.0);
if isempty(candidates_1)
    candidates_1 = find(test_params(:,4) >= 15 & test_params(:,4) <= 25 & test_noise == Inf);
end
median_err = median(pos_errors_B(candidates_1));
[~, ri] = min(abs(pos_errors_B(candidates_1) - median_err));
case1_idx = candidates_1(ri);

% CASE 2: Small tumour (10-14 mm), 40 dB noise
candidates_2 = find( ...
    test_params(:,4) >= 10 & test_params(:,4) <= 14 & ...
    test_noise == 40);
if isempty(candidates_2)
    candidates_2 = find(test_params(:,4) < 15 & test_noise == 40);
end
median_err_2 = median(pos_errors_B(candidates_2));
[~, ri2] = min(abs(pos_errors_B(candidates_2) - median_err_2));
case2_idx = candidates_2(ri2);

case_indices = [case1_idx, case2_idx];
case_labels  = {'Medium tumour, clean', 'Small tumour, 40 dB SNR'};

for k = 1:2
    idx = case_indices(k);
    fprintf('  Case %d (%s): sample #%d\n', k, case_labels{k}, idx);
    fprintf('    True: pos=[%.2f, %.2f, %.2f] cm, d=%.1f mm, c=%.2fx\n', ...
        test_params(idx,1:3), test_params(idx,4), test_params(idx,5));
    fprintf('    ML:   pos=[%.2f, %.2f, %.2f] cm, d=%.1f mm, c=%.2fx\n', ...
        ml_predictions(idx,1:3), ml_predictions(idx,4), ml_predictions(idx,5));
end

%% ========================================================================
% 5. LOCATE TIKHONOV RECONSTRUCTIONS
% =========================================================================
fprintf('\n--- Locating Tikhonov reconstructions ---\n');

if isfield(bl, 'recon_images')
    recon_all = bl.recon_images;
    tik_prior_idx = 2;  % NOSER=1, Tikhonov=2, Laplace=3
    fprintf('  Found recon_images: [%d x %d x %d]\n', size(recon_all));
else
    error('Cannot find recon_images in baseline_results.mat.');
end

%% ========================================================================
% 6. BUILD INTERPOLATION GRID
% =========================================================================
fprintf('\n--- Building interpolation grid ---\n');

grid_res = 200;  % pixels per axis
breast_radius = 6.25;  % cm

x_range = linspace(-breast_radius, breast_radius, grid_res);
y_range = linspace(-breast_radius, breast_radius, grid_res);
[X_grid, Y_grid] = meshgrid(x_range, y_range);

% Breast boundary mask
breast_mask = (X_grid.^2 + Y_grid.^2) <= breast_radius^2;

fprintf('  Grid: %d x %d pixels\n', grid_res, grid_res);

%% ========================================================================
% 7. GENERATE THE FIGURE
% =========================================================================
fprintf('\n--- Generating figure ---\n');

% Figure sized for IEEE double-column: 18 cm wide, 22 cm tall
fig = figure('Name', 'Visual Comparison (Cross-Section)', ...
    'Color', 'w', 'Units', 'centimeters', 'Position', [2, 2, 18, 22]);

row_titles = {'Ground truth', 'Tikhonov + fixed50%', 'Approach B (ML prediction)'};
cmap = jet(256);

% Circle coordinates for boundary overlays
theta = linspace(0, 2*pi, 100);

for col = 1:2
    idx = case_indices(col);

    % --- Extract parameters ---
    true_pos      = test_params(idx, 1:3);
    true_diam_mm  = test_params(idx, 4);
    true_diam_cm  = true_diam_mm / 10;
    true_contrast = test_params(idx, 5);
    true_r_cm     = true_diam_cm / 2;

    ml_pos      = ml_predictions(idx, 1:3);
    ml_diam_mm  = ml_predictions(idx, 4);
    ml_diam_cm  = ml_diam_mm / 10;
    ml_contrast = ml_predictions(idx, 5);
    ml_r_cm     = ml_diam_cm / 2;

    tik_recon = recon_all(:, idx, tik_prior_idx);

    % --- Slice at tumour z-plane ---
    z_slice = true_pos(3);
    z_tol = 0.3;  % cm slice thickness
    z_mask = abs(elem_centres(:,3) - z_slice) < z_tol;
    slice_elem_idx = find(z_mask);

    if isempty(slice_elem_idx)
        z_tol = 0.6;
        z_mask = abs(elem_centres(:,3) - z_slice) < z_tol;
        slice_elem_idx = find(z_mask);
    end

    slice_xy = elem_centres(slice_elem_idx, 1:2);

    % --- Build element data arrays ---
    % Ground truth
    truth_data = ones(n_elems, 1) * background_cond;
    in_tumour = sqrt(sum((elem_centres - true_pos).^2, 2)) < true_r_cm;
    truth_data(in_tumour) = background_cond * true_contrast;

    % ML prediction
    ml_data = ones(n_elems, 1) * background_cond;
    ml_con_clamped = max(min(ml_contrast, 10), 1);
    in_ml = sqrt(sum((elem_centres - ml_pos).^2, 2)) < ml_r_cm;
    ml_data(in_ml) = background_cond * ml_con_clamped;

    % --- Interpolate onto grid (nearest neighbour) ---
    grid_truth = NaN(grid_res);
    grid_recon = NaN(grid_res);
    grid_ml    = NaN(grid_res);

    for ix = 1:grid_res
        for iy = 1:grid_res
            if ~breast_mask(iy, ix), continue; end

            gx = X_grid(iy, ix);
            gy = Y_grid(iy, ix);

            dd = (slice_xy(:,1) - gx).^2 + (slice_xy(:,2) - gy).^2;
            [~, mi] = min(dd);
            ei = slice_elem_idx(mi);

            grid_truth(iy, ix) = truth_data(ei);
            grid_recon(iy, ix) = tik_recon(ei);
            grid_ml(iy, ix)    = ml_data(ei);
        end
    end

    % --- Tikhonov extracted parameters ---
    abs_r = abs(tik_recon);
    thr = 0.50 * max(abs_r);
    abv = abs_r > thr;
    if any(abv)
        w = abs_r(abv);
        tik_pos = sum(elem_centres(abv,:) .* w, 1) / sum(w);
        tv = sum(elem_volumes(abv));
        tik_d_cm = 2 * (3*tv/(4*pi))^(1/3);
        tik_r_cm = tik_d_cm / 2;
        tik_pe = sqrt(sum((tik_pos - true_pos).^2)) * 10;  % mm
        tik_se = abs(tik_d_cm*10 - true_diam_mm);           % mm
    else
        tik_pos = [NaN NaN NaN]; tik_pe = NaN; tik_se = NaN;
        tik_r_cm = 0;
    end

    ml_pe = sqrt(sum((ml_pos - true_pos).^2)) * 10;
    ml_se = abs(ml_diam_mm - true_diam_mm);

    % --- Colour limits ---
    clim_truth = [background_cond * 0.9, background_cond * true_contrast * 1.05];
    clim_recon = [min(tik_recon) * 1.1, max(tik_recon) * 1.1];
    if clim_recon(1) >= clim_recon(2)
        clim_recon = [-0.1, 0.1];
    end

    % =================================================================
    % ROW 1: GROUND TRUTH
    % =================================================================
    subplot(3, 2, col);
    imagesc(x_range, y_range, grid_truth, 'AlphaData', breast_mask);
    set(gca, 'YDir', 'normal', 'Color', [0.95 0.95 0.95]);
    hold on;

    % True tumour boundary (black, solid)
    plot(true_pos(1) + true_r_cm * cos(theta), ...
         true_pos(2) + true_r_cm * sin(theta), 'k-', 'LineWidth', 1.8);
    plot(true_pos(1), true_pos(2), 'k+', 'MarkerSize', 10, 'LineWidth', 1.5);

    % Breast outline
    plot(breast_radius * cos(theta), breast_radius * sin(theta), ...
        'k-', 'LineWidth', 0.8);

    axis equal; xlim([-breast_radius breast_radius]*1.05);
    ylim([-breast_radius breast_radius]*1.05);
    colormap(gca, cmap); caxis(clim_truth);
    cb = colorbar; ylabel(cb, '\sigma (S/m)', 'FontSize', 7);

    title(sprintf('%s\nPos: (%.0f, %.0f, %.0f) mm, Size: %.1f mm, c = %.1fx', ...
        case_labels{col}, true_pos(1)*10, true_pos(2)*10, true_pos(3)*10, ...
        true_diam_mm, true_contrast), ...
        'FontSize', 9, 'FontWeight', 'bold');

    if col == 1
        ylabel({'Ground truth'; ''; 'y (cm)'}, 'FontSize', 9, 'FontWeight', 'bold');
    end
    set(gca, 'FontSize', 8);

    % =================================================================
    % ROW 2: TIKHONOV RECONSTRUCTION
    % =================================================================
    subplot(3, 2, 2 + col);
    imagesc(x_range, y_range, grid_recon, 'AlphaData', breast_mask);
    set(gca, 'YDir', 'normal', 'Color', [0.95 0.95 0.95]);
    hold on;

    % True tumour boundary (green, solid)
    plot(true_pos(1) + true_r_cm * cos(theta), ...
         true_pos(2) + true_r_cm * sin(theta), 'g-', 'LineWidth', 2);
    plot(true_pos(1), true_pos(2), 'g+', 'MarkerSize', 10, 'LineWidth', 1.5);

    % EIDORS estimated centre (red x) and size (red dashed)
    if ~isnan(tik_pos(1))
        plot(tik_pos(1), tik_pos(2), 'rx', 'MarkerSize', 12, 'LineWidth', 2.5);
        plot(tik_pos(1) + tik_r_cm * cos(theta), ...
             tik_pos(2) + tik_r_cm * sin(theta), 'r--', 'LineWidth', 1.5);
    end

    plot(breast_radius * cos(theta), breast_radius * sin(theta), ...
        'k-', 'LineWidth', 0.8);

    axis equal; xlim([-breast_radius breast_radius]*1.05);
    ylim([-breast_radius breast_radius]*1.05);
    colormap(gca, cmap); caxis(clim_recon);
    cb = colorbar; ylabel(cb, '\Delta\sigma', 'FontSize', 7);

    title(sprintf('Tikhonov + fixed50%%\nPos err: %.1f mm, Size err: %.1f mm', ...
        tik_pe, tik_se), 'FontSize', 9, 'FontWeight', 'bold');

    if col == 1
        ylabel({'Tikhonov recon.'; ''; 'y (cm)'}, 'FontSize', 9, 'FontWeight', 'bold');
    end
    set(gca, 'FontSize', 8);

    % =================================================================
    % ROW 3: ML PREDICTION
    % =================================================================
    subplot(3, 2, 4 + col);
    imagesc(x_range, y_range, grid_ml, 'AlphaData', breast_mask);
    set(gca, 'YDir', 'normal', 'Color', [0.95 0.95 0.95]);
    hold on;

    % True tumour boundary (green, solid)
    plot(true_pos(1) + true_r_cm * cos(theta), ...
         true_pos(2) + true_r_cm * sin(theta), 'g-', 'LineWidth', 2);
    plot(true_pos(1), true_pos(2), 'g+', 'MarkerSize', 10, 'LineWidth', 1.5);

    % ML predicted centre (red circle) and size (red dashed)
    plot(ml_pos(1), ml_pos(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    plot(ml_pos(1) + ml_r_cm * cos(theta), ...
         ml_pos(2) + ml_r_cm * sin(theta), 'r--', 'LineWidth', 1.5);

    plot(breast_radius * cos(theta), breast_radius * sin(theta), ...
        'k-', 'LineWidth', 0.8);

    axis equal; xlim([-breast_radius breast_radius]*1.05);
    ylim([-breast_radius breast_radius]*1.05);
    colormap(gca, cmap); caxis(clim_truth);
    cb = colorbar; ylabel(cb, '\sigma (S/m)', 'FontSize', 7);

    title(sprintf('Approach B (NOSER + PCA + NN)\nPos err: %.1f mm, Size err: %.1f mm', ...
        ml_pe, ml_se), 'FontSize', 9, 'FontWeight', 'bold');

    xlabel('x (cm)', 'FontSize', 9);
    if col == 1
        ylabel({'ML prediction'; ''; 'y (cm)'}, 'FontSize', 9, 'FontWeight', 'bold');
    end
    set(gca, 'FontSize', 8);
end

% --- Legend annotation ---
annotation('textbox', [0.02, 0.002, 0.96, 0.022], ...
    'String', ['Green solid/+: true boundary and centre.  ' ...
               'Red x: EIDORS estimated centre.  ' ...
               'Red o: ML predicted centre.  ' ...
               'Red dashed: estimated boundary.'], ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', ...
    'FontSize', 7.5, 'FontAngle', 'italic');

%% ========================================================================
% 8. PRINT QUANTITATIVE SUMMARY
% =========================================================================
fprintf('\n--- Quantitative Summary ---\n\n');

for col = 1:2
    idx = case_indices(col);
    tp = test_params(idx,:);
    mp = ml_predictions(idx,:);

    mpe = sqrt(sum((mp(1:3) - tp(1:3)).^2)) * 10;
    mse_ = abs(mp(4) - tp(4));
    mce = abs(mp(5) - tp(5));

    tr = recon_all(:, idx, tik_prior_idx);
    ar = abs(tr); th = 0.50*max(ar); ab = ar > th;
    if any(ab)
        w_ = ar(ab);
        ep = sum(elem_centres(ab,:).*w_,1)/sum(w_);
        tpe = sqrt(sum((ep - tp(1:3)).^2))*10;
        tv = sum(elem_volumes(ab));
        td = 2*(3*tv/(4*pi))^(1/3)*10;
        tse = abs(td - tp(4));
    else
        tpe = NaN; tse = NaN;
    end

    fprintf('  %s\n', case_labels{col});
    fprintf('    Ground truth:  d = %.1f mm, contrast = %.2fx\n', tp(4), tp(5));
    fprintf('    Tikhonov:      pos err = %.1f mm, size err = %.1f mm\n', tpe, tse);
    fprintf('    Approach B:    pos err = %.1f mm, size err = %.1f mm, con err = %.2f\n', ...
        mpe, mse_, mce);
    fprintf('\n');
end

%% ========================================================================
% 9. SAVE FIGURE
% =========================================================================
fprintf('--- Saving figure ---\n');

fig_dir = 'figures';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

set(fig, 'PaperPositionMode', 'auto');

%savefig(fig, fullfile(fig_dir, 'fig_visual_comparison.fig'));
%print(fig, fullfile(fig_dir, 'fig_visual_comparison.eps'), '-depsc', '-r300');
%print(fig, fullfile(fig_dir, 'fig_visual_comparison.png'), '-dpng', '-r300');

%fprintf('  Saved: fig_visual_comparison.fig, .eps, .png\n');
%fprintf('\nDone.\n');


%% ========================================================================
% LOCAL FUNCTION
% =========================================================================
function fpath = find_file_impl(filename, search_dirs)
    for i = 1:length(search_dirs)
        candidate = fullfile(search_dirs{i}, filename);
        if exist(candidate, 'file')
            fpath = candidate;
            return;
        end
    end
    error('Cannot find %s. Searched: %s', filename, strjoin(search_dirs, ', '));
end
