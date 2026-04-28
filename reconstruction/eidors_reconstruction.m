%% Phase 3: Baseline Reconstruction
% =========================================================================
% ML-Enhanced EIT for Early Breast Cancer Detection
% Phase 3 of 6: Traditional Reconstruction Baseline
%
% This script runs the complete baseline pipeline:
%   Stage A - Hyperparameter optimisation on the validation set
%   Stage B - Reconstruct the full test set with 3 optimised priors
%   Stage C - Extract tumour parameters using 2 threshold methods
%   + Error metric computation, summary statistics, and figures
%
% Six baseline combinations: 3 priors x 2 thresholds
%   Priors:     NOSER, Tikhonov, Laplace
%   Thresholds: Fixed 50%, Otsu's method
%
% Dependencies:
%   geometry/build_breast_model.m   (Phase 1)
%   dataset/training_dataset.mat    (Phase 2)
%   baseline/extract_tumour_params.m (this phase)
%
% Output:
%   baseline_results.mat  - All metrics, extracted parameters, recon images
%   Figures 1-4           - Diagnostic and comparison plots
% =========================================================================

clear; clc; close all;
eidors_cache('clear_all');

fprintf('============================================\n');
fprintf('  PHASE 3: BASELINE RECONSTRUCTION\n');
fprintf('============================================\n\n');

%% ========================================================================
% 1. CONFIGURATION
% =========================================================================

% Background conductivity (must match Phase 1/2)
background_cond = 0.3;  % S/m

% Prior definitions
prior_names   = {'NOSER', 'Tikhonov', 'Laplace'};
prior_handles = {@prior_noser, @prior_tikhonov, @prior_laplace};
n_priors = length(prior_names);

% Threshold method definitions
threshold_names = {'fixed50', 'otsu'};
n_thresholds = length(threshold_names);

% Hyperparameter sweep range (log-spaced from 0.01 to 10)
hp_values = logspace(-2, 1, 15);
n_hp = length(hp_values);

% Combined optimisation metric weights
% Position is our top ML priority (70%), size is secondary (30%)
% Size error is converted from mm to cm inside the loop for unit consistency
w_position = 0.7;
w_size     = 0.3;

% Tumour size bands for breakdown analysis (mm)
size_bands = struct();
size_bands.small  = [10, 15];   % 10-15 mm (early detection target)
size_bands.medium = [15, 20];   % 15-20 mm
size_bands.large  = [20, 30];   % 20-30 mm

fprintf('Configuration:\n');
fprintf('  Priors:      %s\n', strjoin(prior_names, ', '));
fprintf('  Thresholds:  %s\n', strjoin(threshold_names, ', '));
fprintf('  HP range:    %.2f to %.2f (%d values)\n', hp_values(1), hp_values(end), n_hp);
fprintf('  Metric:      %.0f%% position + %.0f%% size\n', w_position*100, w_size*100);
fprintf('\n');

%% ========================================================================
% 2. REBUILD FORWARD MODEL
% =========================================================================
fprintf('--- Building Forward Model ---\n');

fmdl = build_breast_model();

% Compute reference voltages (homogeneous breast)
img_h = mk_image(fmdl, background_cond);
vh_data = fwd_solve(img_h);

fprintf('  Reference voltages: %d measurements\n', length(vh_data.meas));
fprintf('  Mean |vh| = %.4e V\n', mean(abs(vh_data.meas)));
fprintf('\n');

%% ========================================================================
% 3. LOAD TRAINING DATASET
% =========================================================================
fprintf('--- Loading Training Dataset ---\n');

% Adjust path as needed for your directory structure
dataset_path = fullfile('..', 'dataset', 'training_dataset.mat');
if ~exist(dataset_path, 'file')
    % Try current directory
    dataset_path = 'training_dataset.mat';
end
if ~exist(dataset_path, 'file')
    error('Cannot find training_dataset.mat. Place it in ../dataset/ or the current directory.');
end

data = load(dataset_path);

% Extract what we need
measurements_clean = data.measurements_clean;   % 208 x 1440
measurements_all   = data.measurements;          % 208 x 5760
tumour_params      = data.tumour_params;         % 5760 x 5
noise_levels       = data.noise_levels;          % 5760 x 1
split              = data.split;

% Validation set (clean only, for HP optimisation)
val_clean_idx  = split.clean_val;                % indices into 1440 clean configs
val_meas       = measurements_clean(:, val_clean_idx);  % 208 x 216
n_val = size(val_meas, 2);

% Map clean config indices to augmented indices to get ground truth
% Each clean config k has 4 augmented samples at indices (k-1)*4 + [1,2,3,4]
% The first augmented sample is the clean version
val_aug_idx   = (val_clean_idx - 1) * 4 + 1;
val_params    = tumour_params(val_aug_idx, :);  % 216 x 5

% Test set (all noise levels)
test_idx       = split.test;                     % indices into 5760
test_meas      = measurements_all(:, test_idx);  % 208 x 864
test_params    = tumour_params(test_idx, :);      % 864 x 5
test_noise     = noise_levels(test_idx);          % 864 x 1
n_test = size(test_meas, 2);

fprintf('  Validation: %d clean samples\n', n_val);
fprintf('  Test:       %d samples (all noise levels)\n', n_test);
fprintf('  Noise levels in test: ');
unique_noise = unique(test_noise);
for i = 1:length(unique_noise)
    if isinf(unique_noise(i))
        fprintf('clean ');
    else
        fprintf('%ddB ', unique_noise(i));
    end
end
fprintf('\n\n');

%% ========================================================================
% 4. PRECOMPUTE ELEMENT GEOMETRY
% =========================================================================
fprintf('--- Precomputing Element Geometry ---\n');

% Element volumes (using EIDORS built-in)
elem_volumes = get_elem_volume(fmdl);
n_elems = length(elem_volumes);

% Element centres (vectorised computation for speed)
node_coords = fmdl.nodes;     % [n_nodes x 3]
elem_nodes  = fmdl.elems;     % [n_elems x 4] for tetrahedra
n_nodes_per_elem = size(elem_nodes, 2);

elem_centres = zeros(n_elems, 3);
for d = 1:3
    col_sum = zeros(n_elems, 1);
    for v = 1:n_nodes_per_elem
        col_sum = col_sum + node_coords(elem_nodes(:, v), d);
    end
    elem_centres(:, d) = col_sum / n_nodes_per_elem;
end

fprintf('  Elements: %d\n', n_elems);
fprintf('  Volume range: %.4e to %.4e cm^3\n', min(elem_volumes), max(elem_volumes));
fprintf('  Total breast volume: %.2f cm^3\n', sum(elem_volumes));
fprintf('\n');

%% ========================================================================
% 5. STAGE A: HYPERPARAMETER OPTIMISATION
% =========================================================================
fprintf('=== STAGE A: Hyperparameter Optimisation ===\n');
fprintf('  %d priors x %d HP values x %d validation samples = %d solves\n', ...
    n_priors, n_hp, n_val, n_priors * n_hp * n_val);

% Storage for sweep results
hp_sweep = struct();
hp_sweep.hp_values = hp_values;
hp_sweep.position_error = zeros(n_priors, n_hp);  % mean pos error per (prior, hp)
hp_sweep.size_error     = zeros(n_priors, n_hp);  % mean size error per (prior, hp)
hp_sweep.combined_error = zeros(n_priors, n_hp);  % combined metric
hp_sweep.detection_rate = zeros(n_priors, n_hp);  % fraction detected

optimal_hp = zeros(1, n_priors);

stage_a_start = tic;

for p = 1:n_priors
    fprintf('\n  Prior: %s\n', prior_names{p});

    for h = 1:n_hp
        % Build inverse model for this (prior, hp) pair
        imdl = eidors_obj('inv_model', sprintf('baseline_%s_hp%d', prior_names{p}, h));
        imdl.reconst_type          = 'difference';
        imdl.solve                 = @inv_solve_diff_GN_one_step;
        imdl.fwd_model             = fmdl;
        imdl.jacobian_bkgnd.value  = background_cond;
        imdl.RtR_prior             = prior_handles{p};
        imdl.hyperparameter.value  = hp_values(h);

        % Reconstruct all validation samples and extract parameters
        pos_errors = zeros(n_val, 1);
        size_errors = zeros(n_val, 1);
        detected = zeros(n_val, 1);

        for s = 1:n_val
            % Build inhomogeneous data struct
            vi_data = vh_data;
            vi_data.meas = vh_data.meas + val_meas(:, s);

            % Reconstruct
            img_recon = inv_solve(imdl, vh_data, vi_data);

            % Extract parameters (using fixed 50% for optimisation stability)
            result = extract_tumour_params(img_recon.elem_data, ...
                elem_centres, elem_volumes, 'fixed50', background_cond);

            detected(s) = result.detected;

            if result.detected
                % Position error (Euclidean distance in cm)
                true_pos = val_params(s, 1:3);
                pos_errors(s) = sqrt(sum((result.position - true_pos).^2));

                % Size error (absolute, in mm)
                true_diam = val_params(s, 4);  % already in mm
                size_errors(s) = abs(result.diameter_mm - true_diam);
            else
                pos_errors(s) = NaN;
                size_errors(s) = NaN;
            end
        end

        % Compute mean errors (excluding detection failures)
        valid = detected == 1;
        if any(valid)
            mean_pos  = mean(pos_errors(valid));
            mean_size = mean(size_errors(valid));
        else
            mean_pos  = Inf;
            mean_size = Inf;
        end

        % Combined metric: position in cm, size converted from mm to cm
        combined = w_position * mean_pos + w_size * (mean_size / 10);

        hp_sweep.position_error(p, h) = mean_pos;
        hp_sweep.size_error(p, h)     = mean_size;
        hp_sweep.combined_error(p, h) = combined;
        hp_sweep.detection_rate(p, h) = mean(detected);

        % Progress
        if mod(h, 5) == 0 || h == n_hp
            fprintf('    HP %2d/%d (%.3f): pos=%.2f cm, size=%.1f mm, det=%.0f%%\n', ...
                h, n_hp, hp_values(h), mean_pos, mean_size, mean(detected)*100);
        end
    end

    % Select optimal hyperparameter (lowest combined error)
    [~, best_h] = min(hp_sweep.combined_error(p, :));
    optimal_hp(p) = hp_values(best_h);

    fprintf('  >> Optimal HP for %s: %.4f (pos=%.2f cm, size=%.1f mm, det=%.0f%%)\n', ...
        prior_names{p}, optimal_hp(p), ...
        hp_sweep.position_error(p, best_h), ...
        hp_sweep.size_error(p, best_h), ...
        hp_sweep.detection_rate(p, best_h) * 100);
end

stage_a_time = toc(stage_a_start);
fprintf('\n  Stage A complete: %.1f minutes\n\n', stage_a_time / 60);

%% ========================================================================
% 6. STAGE B: RECONSTRUCT TEST SET
% =========================================================================
fprintf('=== STAGE B: Test Set Reconstruction ===\n');
fprintf('  %d priors x %d test samples = %d solves\n', n_priors, n_test, n_priors * n_test);

% Storage for reconstructed images (19403 x 864 x 3 priors)
% This will be ~400 MB in memory
recon_images = zeros(n_elems, n_test, n_priors);

stage_b_start = tic;

for p = 1:n_priors
    fprintf('\n  Reconstructing with %s (HP = %.4f)...\n', prior_names{p}, optimal_hp(p));

    % Build optimised inverse model
    imdl = eidors_obj('inv_model', sprintf('baseline_%s_optimal', prior_names{p}));
    imdl.reconst_type          = 'difference';
    imdl.solve                 = @inv_solve_diff_GN_one_step;
    imdl.fwd_model             = fmdl;
    imdl.jacobian_bkgnd.value  = background_cond;
    imdl.RtR_prior             = prior_handles{p};
    imdl.hyperparameter.value  = optimal_hp(p);

    for s = 1:n_test
        % Build inhomogeneous data struct
        vi_data = vh_data;
        vi_data.meas = vh_data.meas + test_meas(:, s);

        % Reconstruct
        img_recon = inv_solve(imdl, vh_data, vi_data);
        recon_images(:, s, p) = img_recon.elem_data;

        % Progress every 200 samples
        if mod(s, 200) == 0
            fprintf('    [%d/%d] samples\n', s, n_test);
        end
    end

    fprintf('    Done: %d reconstructions\n', n_test);
end

stage_b_time = toc(stage_b_start);
fprintf('\n  Stage B complete: %.1f minutes\n\n', stage_b_time / 60);

%% ========================================================================
% 7. STAGE C: PARAMETER EXTRACTION
% =========================================================================
fprintf('=== STAGE C: Parameter Extraction ===\n');
fprintf('  %d priors x %d thresholds x %d samples = %d extractions\n', ...
    n_priors, n_thresholds, n_test, n_priors * n_thresholds * n_test);

% Storage: one struct array per (prior, threshold) combination
% Each entry holds the extracted parameters for one test sample
extracted = struct();

for p = 1:n_priors
    for t = 1:n_thresholds
        combo_name = sprintf('%s_%s', prior_names{p}, threshold_names{t});

        % Pre-allocate arrays for this combination
        positions    = NaN(n_test, 3);
        diameters    = NaN(n_test, 1);
        contrasts    = NaN(n_test, 1);
        detected     = false(n_test, 1);
        n_elements   = zeros(n_test, 1);

        for s = 1:n_test
            result = extract_tumour_params(recon_images(:, s, p), ...
                elem_centres, elem_volumes, threshold_names{t}, background_cond);

            positions(s, :)  = result.position;
            diameters(s)     = result.diameter_mm;
            contrasts(s)     = result.contrast_est;
            detected(s)      = result.detected;
            n_elements(s)    = result.n_elements;
        end

        extracted.(combo_name).positions   = positions;
        extracted.(combo_name).diameters   = diameters;
        extracted.(combo_name).contrasts   = contrasts;
        extracted.(combo_name).detected    = detected;
        extracted.(combo_name).n_elements  = n_elements;

        det_rate = sum(detected) / n_test * 100;
        fprintf('  %s: detection rate = %.1f%%\n', combo_name, det_rate);
    end
end

fprintf('\n');

%% ========================================================================
% 8. COMPUTE ERROR METRICS
% =========================================================================
fprintf('=== Computing Error Metrics ===\n');

combo_names = fieldnames(extracted);
n_combos = length(combo_names);

errors = struct();

for c = 1:n_combos
    name = combo_names{c};
    ext  = extracted.(name);

    % Position error: Euclidean distance (cm)
    pos_err = sqrt(sum((ext.positions - test_params(:, 1:3)).^2, 2));

    % Size error: absolute difference (mm)
    size_err_abs = abs(ext.diameters - test_params(:, 4));

    % Size error: relative (%)
    size_err_rel = size_err_abs ./ test_params(:, 4) * 100;

    % Contrast error: absolute difference (dimensionless)
    contrast_err = abs(ext.contrasts - test_params(:, 5));

    % Store per-sample errors (NaN where not detected)
    errors.(name).position_err   = pos_err;
    errors.(name).size_err_abs   = size_err_abs;
    errors.(name).size_err_rel   = size_err_rel;
    errors.(name).contrast_err   = contrast_err;
    errors.(name).detected       = ext.detected;
end

%% ========================================================================
% 9. SUMMARY STATISTICS
% =========================================================================
fprintf('\n=== SUMMARY STATISTICS ===\n');

% Define noise level labels for display
noise_labels = {'Clean', '80dB', '60dB', '40dB'};
noise_values = [Inf, 80, 60, 40];

% ------------------------------------------------------------------
% 9.1 Overall summary table
% ------------------------------------------------------------------
fprintf('\n--- Overall Performance (All Test Samples) ---\n');
fprintf('%-25s  %8s  %8s  %8s  %8s  %8s\n', ...
    'Combination', 'Pos(cm)', 'Size(mm)', 'Size(%)', 'Contrast', 'Det(%)');
fprintf('%s\n', repmat('-', 1, 80));

perf_summary = struct();

for c = 1:n_combos
    name = combo_names{c};
    err  = errors.(name);
    det  = err.detected;

    s = struct();  % Explicitly initialise as struct

    if any(det)
        s.pos_mean      = mean(err.position_err(det), 'omitnan');
        s.pos_median    = median(err.position_err(det), 'omitnan');
        s.pos_std       = std(err.position_err(det), 'omitnan');
        s.size_abs_mean = mean(err.size_err_abs(det), 'omitnan');
        s.size_rel_mean = mean(err.size_err_rel(det), 'omitnan');
        s.contrast_mean = mean(err.contrast_err(det), 'omitnan');
        s.detection_rate = sum(det) / n_test * 100;
    else
        s.pos_mean      = NaN;
        s.pos_median    = NaN;
        s.pos_std       = NaN;
        s.size_abs_mean = NaN;
        s.size_rel_mean = NaN;
        s.contrast_mean = NaN;
        s.detection_rate = 0;
    end

    perf_summary.(name) = s;

    fprintf('%-25s  %8.2f  %8.1f  %7.1f%%  %8.2f  %7.1f%%\n', ...
        name, s.pos_mean, s.size_abs_mean, s.size_rel_mean, ...
        s.contrast_mean, s.detection_rate);
end

% ------------------------------------------------------------------
% 9.2 Breakdown by noise level
% ------------------------------------------------------------------
fprintf('\n--- Breakdown by Noise Level (Mean Position Error, cm) ---\n');
fprintf('%-25s', 'Combination');
for nl = 1:length(noise_labels)
    fprintf('  %8s', noise_labels{nl});
end
fprintf('\n%s\n', repmat('-', 1, 70));

noise_breakdown = struct();

for c = 1:n_combos
    name = combo_names{c};
    err  = errors.(name);
    fprintf('%-25s', name);

    for nl = 1:length(noise_values)
        mask = test_noise == noise_values(nl) & err.detected;
        if any(mask)
            val = mean(err.position_err(mask), 'omitnan');
        else
            val = NaN;
        end
        noise_breakdown.(name).pos_by_noise(nl) = val;
        fprintf('  %8.2f', val);
    end
    fprintf('\n');
end

% ------------------------------------------------------------------
% 9.3 Breakdown by tumour size band
% ------------------------------------------------------------------
fprintf('\n--- Breakdown by Tumour Size (Mean Position Error, cm) ---\n');
band_names = fieldnames(size_bands);
fprintf('%-25s', 'Combination');
for b = 1:length(band_names)
    bnd = size_bands.(band_names{b});
    fprintf('  %5d-%dmm', bnd(1), bnd(2));
end
fprintf('\n%s\n', repmat('-', 1, 70));

size_breakdown = struct();

for c = 1:n_combos
    name = combo_names{c};
    err  = errors.(name);
    fprintf('%-25s', name);

    for b = 1:length(band_names)
        bnd = size_bands.(band_names{b});
        true_diam = test_params(:, 4);
        mask = true_diam >= bnd(1) & true_diam < bnd(2) & err.detected;
        if any(mask)
            val = mean(err.position_err(mask), 'omitnan');
        else
            val = NaN;
        end
        size_breakdown.(name).pos_by_size(b) = val;
        fprintf('  %10.2f', val);
    end
    fprintf('\n');
end

% ------------------------------------------------------------------
% 9.4 Identify best combination
% ------------------------------------------------------------------
fprintf('\n--- Best Performing Combination ---\n');
best_pos = Inf;
best_name = '';
for c = 1:n_combos
    name = combo_names{c};
    if perf_summary.(name).pos_mean < best_pos
        best_pos = perf_summary.(name).pos_mean;
        best_name = name;
    end
end
fprintf('  Best overall position accuracy: %s (%.2f cm)\n', best_name, best_pos);
fprintf('  This becomes the baseline target for the neural network in Phase 4.\n');

%% ========================================================================
% 10. SAVE RESULTS
% =========================================================================
fprintf('\n=== Saving Results ===\n');

baseline_results = struct();

% Stage A: hyperparameter sweep
baseline_results.hp_sweep   = hp_sweep;
baseline_results.optimal_hp = optimal_hp;

% Stage B: reconstructed images (large)
baseline_results.recon_images = recon_images;

% Stage C: extracted parameters
baseline_results.extracted = extracted;

% Error metrics
baseline_results.errors = errors;

% Summary tables
baseline_results.perf_summary    = perf_summary;
baseline_results.noise_breakdown = noise_breakdown;
baseline_results.size_breakdown  = size_breakdown;

% Ground truth and metadata
baseline_results.test_params      = test_params;
baseline_results.test_noise       = test_noise;
baseline_results.test_indices     = test_idx;

baseline_results.metadata = struct();
baseline_results.metadata.date           = datestr(now);
baseline_results.metadata.prior_names    = {prior_names};
baseline_results.metadata.threshold_names = {threshold_names};
baseline_results.metadata.optimal_hp     = optimal_hp;
baseline_results.metadata.hp_values      = hp_values;
baseline_results.metadata.background_cond = background_cond;
baseline_results.metadata.w_position     = w_position;
baseline_results.metadata.w_size         = w_size;
baseline_results.metadata.n_test         = n_test;
baseline_results.metadata.n_val          = n_val;
baseline_results.metadata.best_combination = best_name;

% Save with -v7.3 for large file support
save('baseline_results.mat', '-struct', 'baseline_results', '-v7.3');

finfo = dir('baseline_results.mat');
fprintf('  Saved: baseline_results.mat (%.1f MB)\n', finfo.bytes / 1e6);

%% ========================================================================
% 11. GENERATE FIGURES
% =========================================================================
fprintf('\n=== Generating Figures ===\n');

% Colour scheme for the three priors
prior_colors = [0.2 0.4 0.8;   % NOSER: blue
                0.8 0.2 0.2;   % Tikhonov: red
                0.2 0.7 0.3];  % Laplace: green

% ------------------------------------------------------------------
% FIGURE 1: Hyperparameter Sweep Curves
% ------------------------------------------------------------------
figure('Name', 'Fig 1: HP Sweep', 'Position', [50, 400, 900, 400], 'Color', 'w');

% Left: Position error
subplot(1, 2, 1);
for p = 1:n_priors
    semilogx(hp_values, hp_sweep.position_error(p, :), '-o', ...
        'Color', prior_colors(p, :), 'LineWidth', 1.5, 'MarkerSize', 4);
    hold on;
    % Mark optimal point
    [~, best_h] = min(hp_sweep.combined_error(p, :));
    plot(hp_values(best_h), hp_sweep.position_error(p, best_h), ...
        'p', 'Color', prior_colors(p, :), 'MarkerSize', 14, ...
        'MarkerFaceColor', prior_colors(p, :));
end
xlabel('Hyperparameter (\lambda)');
ylabel('Mean Position Error (cm)');
title('Position Error vs Hyperparameter');
legend(prior_names, 'Location', 'best');
grid on;

% Right: Combined metric
subplot(1, 2, 2);
for p = 1:n_priors
    semilogx(hp_values, hp_sweep.combined_error(p, :), '-o', ...
        'Color', prior_colors(p, :), 'LineWidth', 1.5, 'MarkerSize', 4);
    hold on;
    [~, best_h] = min(hp_sweep.combined_error(p, :));
    plot(hp_values(best_h), hp_sweep.combined_error(p, best_h), ...
        'p', 'Color', prior_colors(p, :), 'MarkerSize', 14, ...
        'MarkerFaceColor', prior_colors(p, :));
end
xlabel('Hyperparameter (\lambda)');
ylabel('Combined Error (70% pos + 30% size)');
title('Combined Metric vs Hyperparameter');
legend(prior_names, 'Location', 'best');
grid on;

sgtitle('Figure 1: Hyperparameter Optimisation');

% ------------------------------------------------------------------
% FIGURE 2: Position Error by Method and Noise Level
% ------------------------------------------------------------------
figure('Name', 'Fig 2: Position Error', 'Position', [50, 300, 1000, 450], 'Color', 'w');

bar_data = zeros(n_combos, length(noise_values));
bar_labels = cell(1, n_combos);

for c = 1:n_combos
    name = combo_names{c};
    bar_labels{c} = strrep(name, '_', ' + ');
    for nl = 1:length(noise_values)
        bar_data(c, nl) = noise_breakdown.(name).pos_by_noise(nl);
    end
end

b = bar(bar_data, 'grouped');
% Assign colours cycling through noise levels
noise_colors = [0.3 0.7 0.3; 0.3 0.5 0.8; 0.9 0.6 0.2; 0.8 0.2 0.2];
for nl = 1:length(noise_values)
    b(nl).FaceColor = noise_colors(nl, :);
end

set(gca, 'XTickLabel', bar_labels, 'XTickLabelRotation', 25);
ylabel('Mean Position Error (cm)');
title('Figure 2: Position Error by Method and Noise Level');
legend(noise_labels, 'Location', 'northeastoutside');
grid on;

% ------------------------------------------------------------------
% FIGURE 3: Size Error by Method and Noise Level
% ------------------------------------------------------------------
figure('Name', 'Fig 3: Size Error', 'Position', [50, 200, 1000, 450], 'Color', 'w');

bar_data_size = zeros(n_combos, length(noise_values));

for c = 1:n_combos
    name = combo_names{c};
    err  = errors.(name);
    for nl = 1:length(noise_values)
        mask = test_noise == noise_values(nl) & err.detected;
        if any(mask)
            bar_data_size(c, nl) = mean(err.size_err_abs(mask), 'omitnan');
        else
            bar_data_size(c, nl) = NaN;
        end
    end
end

b2 = bar(bar_data_size, 'grouped');
for nl = 1:length(noise_values)
    b2(nl).FaceColor = noise_colors(nl, :);
end

set(gca, 'XTickLabel', bar_labels, 'XTickLabelRotation', 25);
ylabel('Mean Size Error (mm)');
title('Figure 3: Size Error by Method and Noise Level');
legend(noise_labels, 'Location', 'northeastoutside');
grid on;

% ------------------------------------------------------------------
% FIGURE 4: Example Reconstructions
% ------------------------------------------------------------------
fprintf('  Selecting example test cases for Figure 4...\n');

% Pick 4 example cases spanning a range of difficulties:
%   1. Large tumour, clean (easy case)
%   2. Small tumour, clean (challenging)
%   3. Medium tumour, 60dB noise
%   4. Small tumour, 40dB noise (hardest case)
example_idx = [];
example_labels = {};

% Case 1: Large tumour, clean
candidates = find(test_params(:,4) > 22 & test_noise == Inf);
if ~isempty(candidates)
    example_idx(end+1) = candidates(1);
    example_labels{end+1} = sprintf('Large (%.0fmm), Clean', test_params(candidates(1),4));
end

% Case 2: Small tumour, clean
candidates = find(test_params(:,4) < 13 & test_noise == Inf);
if ~isempty(candidates)
    example_idx(end+1) = candidates(1);
    example_labels{end+1} = sprintf('Small (%.0fmm), Clean', test_params(candidates(1),4));
end

% Case 3: Medium tumour, 60dB
candidates = find(test_params(:,4) > 15 & test_params(:,4) < 20 & test_noise == 60);
if ~isempty(candidates)
    example_idx(end+1) = candidates(1);
    example_labels{end+1} = sprintf('Medium (%.0fmm), 60dB', test_params(candidates(1),4));
end

% Case 4: Small tumour, 40dB
candidates = find(test_params(:,4) < 14 & test_noise == 40);
if ~isempty(candidates)
    example_idx(end+1) = candidates(1);
    example_labels{end+1} = sprintf('Small (%.0fmm), 40dB', test_params(candidates(1),4));
end

n_examples = length(example_idx);

if n_examples > 0
    figure('Name', 'Fig 4: Example Reconstructions', ...
        'Position', [50, 50, 350*n_priors, 280*n_examples], 'Color', 'w');

    for e = 1:n_examples
        idx = example_idx(e);
        true_pos = test_params(idx, 1:3);
        true_diam = test_params(idx, 4) / 10;  % Convert mm to cm for plotting

        for p = 1:n_priors
            subplot(n_examples, n_priors, (e-1)*n_priors + p);

            % Create a temporary image struct for show_fem
            img_show = mk_image(fmdl, background_cond);
            img_show.elem_data = recon_images(:, idx, p);

            show_fem(img_show, [1]);
            hold on;

            % Overlay true tumour position as a circle
            theta = linspace(0, 2*pi, 50);
            % Plot circle at the tumour's z-height
            x_circle = true_pos(1) + (true_diam/2) * cos(theta);
            y_circle = true_pos(2) + (true_diam/2) * sin(theta);
            z_circle = true_pos(3) * ones(size(theta));
            plot3(x_circle, y_circle, z_circle, 'g-', 'LineWidth', 2);

            % Mark true centre
            plot3(true_pos(1), true_pos(2), true_pos(3), 'gx', ...
                'MarkerSize', 10, 'LineWidth', 2);

            hold off;
            view(0, 90);  % Top-down view
            axis equal; axis tight;

            if e == 1
                title(prior_names{p}, 'FontWeight', 'bold');
            end
            if p == 1
                ylabel(example_labels{e}, 'FontWeight', 'bold', 'FontSize', 9);
            end
        end
    end

    sgtitle('Figure 4: Example Reconstructions (green = true tumour)');
end

%% ========================================================================
% DONE
% =========================================================================
total_time = stage_a_time + stage_b_time;
fprintf('\n============================================\n');
fprintf('  PHASE 3 COMPLETE\n');
fprintf('  Total reconstruction time: %.1f minutes\n', total_time / 60);
fprintf('  Best combination: %s\n', best_name);
fprintf('  Best position error: %.2f cm\n', best_pos);
fprintf('  Results saved: baseline_results.mat\n');
fprintf('============================================\n');
fprintf('\nReview Figures 1-4, then proceed to Phase 4.\n');
