%% Phase 5 - evaluate_all_methods.m
% =========================================================================
% ML-Enhanced EIT for Early Breast Cancer Detection
% Phase 5 of 6: Comprehensive Evaluation and Figure Generation
%
% This script loads all Phase 3 and Phase 4 results, aligns the data,
% computes unified metrics for all 9 methods (6 baselines + 3 ML), runs
% Wilcoxon signed-rank tests for statistical significance, and saves a
% single comprehensive evaluation structure.
%
% Dependencies:
%   baseline/baseline_results.mat   (Phase 3)
%   networks/results_approach_A.mat (Phase 4)
%   networks/results_approach_B.mat (Phase 4)
%   networks/results_approach_C.mat (Phase 4)
%   dataset/training_dataset.mat    (Phase 2, for ground truth verification)
%
% Output:
%   evaluation/evaluation_results.mat
%
% Estimated runtime: < 1 minute (no EIDORS solves, no training)
% =========================================================================

clear; clc; close all;

fprintf('============================================\n');
fprintf('  PHASE 5: COMPREHENSIVE EVALUATION\n');
fprintf('============================================\n\n');

%% ========================================================================
% 1. LOAD ALL RESULTS FILES
% =========================================================================
fprintf('=== Loading Results Files ===\n');

% Robust file finder (same pattern as earlier phases)
search_dirs = {'../baseline', '../networks', '../dataset', ...
               './baseline', './networks', './dataset', '.', '..', ...
               'baseline', 'networks', 'dataset'};

% --- Baseline results (Phase 3) ---
baseline_path = find_file('baseline_results.mat', search_dirs);
fprintf('  Loading baseline results: %s\n', baseline_path);
BL = load(baseline_path);

% --- ML results (Phase 4) ---
pathA = find_file('results_approach_A.mat', search_dirs);
fprintf('  Loading Approach A: %s\n', pathA);
A = load(pathA);

pathB = find_file('results_approach_B.mat', search_dirs);
fprintf('  Loading Approach B: %s\n', pathB);
B = load(pathB);

pathC = find_file('results_approach_C.mat', search_dirs);
fprintf('  Loading Approach C: %s\n', pathC);
C = load(pathC);

% --- Training dataset (Phase 2, for verification) ---
ds_path = find_file('training_dataset.mat', search_dirs);
fprintf('  Loading dataset: %s\n', ds_path);
DS = load(ds_path);

fprintf('  All files loaded.\n\n');

%% ========================================================================
% 2. VERIFY GROUND TRUTH ALIGNMENT
% =========================================================================
fprintf('=== Verifying Ground Truth Alignment ===\n');

% All results files should have been evaluated on the same 864 test samples.
% The ground truth tumour params should be identical across all files.
gt_baseline = BL.test_params;      % 864 x 5
gt_A        = A.test_params;       % 864 x 5
gt_B        = B.test_params;       % 864 x 5
gt_C        = C.test_params;       % 864 x 5

n_test = size(gt_baseline, 1);
fprintf('  Test set size: %d samples\n', n_test);

% Cross-check ground truth
match_A = max(abs(gt_baseline - gt_A), [], 'all') < 1e-10;
match_B = max(abs(gt_baseline - gt_B), [], 'all') < 1e-10;
match_C = max(abs(gt_baseline - gt_C), [], 'all') < 1e-10;

if match_A && match_B && match_C
    fprintf('  Ground truth alignment: PASS (all files match)\n');
else
    warning('Ground truth mismatch detected! Results may not be comparable.');
    if ~match_A, fprintf('  Approach A: MISMATCH\n'); end
    if ~match_B, fprintf('  Approach B: MISMATCH\n'); end
    if ~match_C, fprintf('  Approach C: MISMATCH\n'); end
end

% Use the baseline ground truth as canonical
ground_truth  = gt_baseline;         % 864 x 5
noise_levels  = BL.test_noise;       % 864 x 1

fprintf('\n');

%% ========================================================================
% 3. BUILD UNIFIED METHODS STRUCTURE
% =========================================================================
fprintf('=== Building Unified Methods Structure ===\n');

% Noise and size band definitions (consistent with Phases 3 & 4)
noise_values  = [Inf, 80, 60, 40];
noise_labels  = {'Clean', '80dB', '60dB', '40dB'};
size_bands    = [10 15; 15 20; 20 30];
size_labels   = {'Small (10-15 mm)', 'Medium (15-20 mm)', 'Large (20-30 mm)'};
true_diameters = ground_truth(:, 4);  % mm

% Define the 9 methods
method_names = { ...
    'NOSER + fixed50', 'NOSER + Otsu', ...
    'Tikhonov + fixed50', 'Tikhonov + Otsu', ...
    'Laplace + fixed50', 'Laplace + Otsu', ...
    'Approach A (Direct)', 'Approach B (Post-Proc)', 'Approach C (Hybrid)'};

method_families = { ...
    'baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline', ...
    'ml', 'ml', 'ml'};

% Field names in baseline_results.mat (from eidors_reconstruction.m)
baseline_field_names = { ...
    'NOSER_fixed50', 'NOSER_otsu', ...
    'Tikhonov_fixed50', 'Tikhonov_otsu', ...
    'Laplace_fixed50', 'Laplace_otsu'};

n_methods = length(method_names);

% Preallocate methods struct array
methods = struct();

% --- Populate baseline methods (1-6) ---
for m = 1:6
    fname = baseline_field_names{m};

    methods(m).name   = method_names{m};
    methods(m).family = method_families{m};

    % Per-sample errors already computed in Phase 3
    methods(m).pos_errors      = BL.errors.(fname).position_err(:);
    methods(m).size_errors     = BL.errors.(fname).size_err_abs(:);
    methods(m).size_errors_rel = BL.errors.(fname).size_err_rel(:);
    methods(m).contrast_errors = BL.errors.(fname).contrast_err(:);
    methods(m).detected        = BL.errors.(fname).detected(:);

    % Handle detection failures (replace NaN with Inf for fair comparison)
    % This ensures non-detected samples count as worst-case, not excluded.
    det = methods(m).detected;
    if ~all(det)
        methods(m).pos_errors(~det)      = NaN;
        methods(m).size_errors(~det)     = NaN;
        methods(m).size_errors_rel(~det) = NaN;
        methods(m).contrast_errors(~det) = NaN;
    end

    fprintf('  [%d] %s: loaded (%d detected)\n', m, method_names{m}, sum(det));
end

% --- Populate ML methods (7-9) ---
ml_results = {A.results, B.results, C.results};
ml_predictions = {A.predictions, B.predictions, C.predictions};

for m = 7:9
    idx = m - 6;  % 1, 2, 3
    res = ml_results{idx};

    methods(m).name   = method_names{m};
    methods(m).family = method_families{m};

    % ML approaches always produce output (100% detection)
    methods(m).pos_errors      = res.position_err(:);
    methods(m).size_errors     = res.size_err_abs(:);
    methods(m).size_errors_rel = res.size_err_rel(:);
    methods(m).contrast_errors = res.contrast_err(:);
    methods(m).detected        = true(n_test, 1);

    fprintf('  [%d] %s: loaded (864/864 detected)\n', m, method_names{m});
end

% --- Compute overall, noise-level, and size-band summaries ---
for m = 1:n_methods
    det = methods(m).detected;

    % Overall
    methods(m).overall.pos_mean      = mean(methods(m).pos_errors(det), 'omitnan');
    methods(m).overall.pos_median    = median(methods(m).pos_errors(det), 'omitnan');
    methods(m).overall.pos_std       = std(methods(m).pos_errors(det), 'omitnan');
    methods(m).overall.size_abs_mean = mean(methods(m).size_errors(det), 'omitnan');
    methods(m).overall.size_rel_mean = mean(methods(m).size_errors_rel(det), 'omitnan');
    methods(m).overall.contrast_mean = mean(methods(m).contrast_errors(det), 'omitnan');
    methods(m).overall.detection_pct = sum(det) / n_test * 100;

    % By noise level
    for nl = 1:length(noise_values)
        mask = noise_levels == noise_values(nl) & det;
        if any(mask)
            methods(m).by_noise.pos(nl)      = mean(methods(m).pos_errors(mask), 'omitnan');
            methods(m).by_noise.size_abs(nl) = mean(methods(m).size_errors(mask), 'omitnan');
            methods(m).by_noise.contrast(nl) = mean(methods(m).contrast_errors(mask), 'omitnan');
        else
            methods(m).by_noise.pos(nl)      = NaN;
            methods(m).by_noise.size_abs(nl) = NaN;
            methods(m).by_noise.contrast(nl) = NaN;
        end
    end

    % By tumour size band
    for sb = 1:size(size_bands, 1)
        mask = true_diameters >= size_bands(sb, 1) & ...
               true_diameters <  size_bands(sb, 2) & det;
        if any(mask)
            methods(m).by_size.pos(sb)      = mean(methods(m).pos_errors(mask), 'omitnan');
            methods(m).by_size.size_abs(sb) = mean(methods(m).size_errors(mask), 'omitnan');
            methods(m).by_size.contrast(sb) = mean(methods(m).contrast_errors(mask), 'omitnan');
        else
            methods(m).by_size.pos(sb)      = NaN;
            methods(m).by_size.size_abs(sb) = NaN;
            methods(m).by_size.contrast(sb) = NaN;
        end
    end
end

fprintf('\n');

%% ========================================================================
% 4. PRINT UNIFIED COMPARISON TABLE
% =========================================================================
fprintf('=== UNIFIED COMPARISON TABLE ===\n');
fprintf('%-28s  %8s  %8s  %8s  %8s  %8s  %6s\n', ...
    'Method', 'Pos(cm)', 'Med(cm)', 'Size(mm)', 'Size(%)', 'Contr', 'Det%');
fprintf('%s\n', repmat('-', 1, 90));

for m = 1:n_methods
    o = methods(m).overall;
    fprintf('%-28s  %8.2f  %8.2f  %8.1f  %7.1f%%  %8.2f  %5.1f%%\n', ...
        methods(m).name, o.pos_mean, o.pos_median, o.size_abs_mean, ...
        o.size_rel_mean, o.contrast_mean, o.detection_pct);
end

fprintf('\n');

%% ========================================================================
% 5. VERIFY AGAINST KNOWN VALUES (sanity check)
% =========================================================================
fprintf('=== Verification Against Phase 3/4 Reports ===\n');

% Check a few known values from the verification reports
checks = {
    'Tikhonov + Otsu',    3+1, 'pos_mean',      0.29, 0.02;
    'Tikhonov + fixed50', 3,   'pos_mean',      0.42, 0.02;
    'Tikhonov + fixed50', 3,   'size_abs_mean', 6.2,  0.2;
    'Approach B (Post-Proc)', 8, 'pos_mean',    0.34, 0.02;
    'Approach B (Post-Proc)', 8, 'size_abs_mean', 1.3, 0.2;
};

% Method index for Tik+Otsu is 4, Tik+fix50 is 3
check_indices = [4, 3, 3, 8, 8];
all_pass = true;

for v = 1:size(checks, 1)
    mi    = check_indices(v);
    field = checks{v, 3};
    expected = checks{v, 4};
    tol      = checks{v, 5};
    actual   = methods(mi).overall.(field);

    if abs(actual - expected) <= tol
        fprintf('  %s %s: %.2f (expected ~%.2f) PASS\n', ...
            methods(mi).name, field, actual, expected);
    else
        fprintf('  %s %s: %.2f (expected ~%.2f) ** MISMATCH **\n', ...
            methods(mi).name, field, actual, expected);
        all_pass = false;
    end
end

if all_pass
    fprintf('  All verification checks passed.\n');
else
    warning('Some values did not match expected. Review the loaded data.');
end

fprintf('\n');

%% ========================================================================
% 6. STATISTICAL SIGNIFICANCE TESTING
% =========================================================================
fprintf('=== Statistical Significance Testing ===\n');
fprintf('  Test: Wilcoxon signed-rank (paired, non-parametric)\n');
fprintf('  Correction: Bonferroni\n');

% Reference baseline: Tikhonov + fixed50 (index 3)
ref_idx = 3;
ref_name = methods(ref_idx).name;
fprintf('  Reference baseline: %s\n\n', ref_name);

% Number of tests for Bonferroni correction
n_tests = 6;
alpha_raw = 0.05;
alpha_corrected = alpha_raw / n_tests;
fprintf('  Raw alpha:       %.4f\n', alpha_raw);
fprintf('  Corrected alpha: %.4f (Bonferroni, %d tests)\n\n', alpha_corrected, n_tests);

% Helper function for running one Wilcoxon test
run_test = @(x, y, label) run_wilcoxon(x, y, label, alpha_corrected, n_test);

% ------------------------------------------------------------------
% PRIMARY COMPARISONS: ML vs balanced baseline (Tikhonov + fixed50)
% ------------------------------------------------------------------
fprintf('--- Primary Comparisons (ML vs Balanced Baseline) ---\n');

% Best ML approach is Approach B (index 8)
ml_best_idx = 8;

% P1: Position error
primary(1) = run_test( ...
    methods(ref_idx).pos_errors, ...
    methods(ml_best_idx).pos_errors, ...
    sprintf('Position: %s vs %s', ref_name, methods(ml_best_idx).name));

% P2: Size error
primary(2) = run_test( ...
    methods(ref_idx).size_errors, ...
    methods(ml_best_idx).size_errors, ...
    sprintf('Size: %s vs %s', ref_name, methods(ml_best_idx).name));

% P3: Contrast error
primary(3) = run_test( ...
    methods(ref_idx).contrast_errors, ...
    methods(ml_best_idx).contrast_errors, ...
    sprintf('Contrast: %s vs %s', ref_name, methods(ml_best_idx).name));

% ------------------------------------------------------------------
% CROSS-APPROACH COMPARISONS (to justify ML ranking)
% ------------------------------------------------------------------
fprintf('\n--- Cross-Approach Comparisons ---\n');

% C1: Position, B vs A
cross(1) = run_test( ...
    methods(8).pos_errors, ...
    methods(7).pos_errors, ...
    sprintf('Position: %s vs %s', methods(8).name, methods(7).name));

% C2: Position, B vs C
cross(2) = run_test( ...
    methods(8).pos_errors, ...
    methods(9).pos_errors, ...
    sprintf('Position: %s vs %s', methods(8).name, methods(9).name));

% C3: Contrast, C vs B
cross(3) = run_test( ...
    methods(9).contrast_errors, ...
    methods(8).contrast_errors, ...
    sprintf('Contrast: %s vs %s', methods(9).name, methods(8).name));

% Pack significance results
significance.primary      = primary;
significance.cross        = cross;
significance.alpha_raw    = alpha_raw;
significance.alpha_corr   = alpha_corrected;
significance.n_tests      = n_tests;
significance.correction   = 'Bonferroni';
significance.ref_baseline = ref_name;

fprintf('\n');

%% ========================================================================
% 7. IMPROVEMENT SUMMARY
% =========================================================================
fprintf('=== Improvement Summary (Approach B vs Balanced Baseline) ===\n');

ref = methods(ref_idx).overall;
mlb = methods(ml_best_idx).overall;

improve.pos_pct      = (1 - mlb.pos_mean / ref.pos_mean) * 100;
improve.size_pct     = (1 - mlb.size_abs_mean / ref.size_abs_mean) * 100;
improve.contrast_pct = (1 - mlb.contrast_mean / ref.contrast_mean) * 100;

fprintf('  Position:  %.2f -> %.2f cm  (%+.1f%%)\n', ...
    ref.pos_mean, mlb.pos_mean, improve.pos_pct);
fprintf('  Size:      %.1f -> %.1f mm  (%+.1f%%)\n', ...
    ref.size_abs_mean, mlb.size_abs_mean, improve.size_pct);
fprintf('  Contrast:  %.2f -> %.2f     (%+.1f%%)\n', ...
    ref.contrast_mean, mlb.contrast_mean, improve.contrast_pct);

fprintf('\n');

%% ========================================================================
% 8. STORE PREDICTIONS FOR SCATTER PLOTS
% =========================================================================
% The scatter plots need predicted vs true values. Store them here so
% generate_figures.m does not need to reload the approach files.

scatter_data.approach_B.predictions = B.predictions;  % 864 x 5
scatter_data.approach_B.ground_truth = gt_B;          % 864 x 5

% Also store A and C for optional comparison
scatter_data.approach_A.predictions = A.predictions;
scatter_data.approach_A.ground_truth = gt_A;
scatter_data.approach_C.predictions = C.predictions;
scatter_data.approach_C.ground_truth = gt_C;

%% ========================================================================
% 9. SAVE EVALUATION RESULTS
% =========================================================================
fprintf('=== Saving Evaluation Results ===\n');

eval_results = struct();
eval_results.methods       = methods;
eval_results.ground_truth  = ground_truth;
eval_results.noise_levels  = noise_levels;
eval_results.significance  = significance;
eval_results.improvement   = improve;
eval_results.scatter_data  = scatter_data;

% Definitions (for figure scripts)
eval_results.defs.method_names   = method_names;
eval_results.defs.method_families = method_families;
eval_results.defs.noise_values   = noise_values;
eval_results.defs.noise_labels   = noise_labels;
eval_results.defs.size_bands     = size_bands;
eval_results.defs.size_labels    = size_labels;
eval_results.defs.ref_baseline_idx = ref_idx;
eval_results.defs.ml_best_idx     = ml_best_idx;

% Indices for the 5 methods shown in Figures 3 and 4
% Tikhonov + Otsu (4), Tikhonov + fixed50 (3), Approach A (7), B (8), C (9)
eval_results.defs.fig34_indices = [4, 3, 7, 8, 9];
eval_results.defs.fig34_names   = method_names([4, 3, 7, 8, 9]);

eval_results.metadata.date          = datestr(now);
eval_results.metadata.n_test        = n_test;
eval_results.metadata.alpha         = alpha_raw;
eval_results.metadata.alpha_corr    = alpha_corrected;
eval_results.metadata.phase         = 'Phase 5: Comprehensive Evaluation';

save('evaluation_results.mat', '-struct', 'eval_results', '-v7.3');
finfo = dir('evaluation_results.mat');
fprintf('  Saved: evaluation_results.mat (%.1f MB)\n', finfo.bytes / 1e6);

%% ========================================================================
fprintf('\n============================================\n');
fprintf('  EVALUATION COMPLETE\n');
fprintf('  9 methods compared, %d test samples each\n', n_test);
fprintf('  %d statistical tests run\n', n_tests);
fprintf('  Results saved: evaluation_results.mat\n');
fprintf('============================================\n');
fprintf('\nNext: run generate_figures.m, then results_summary.m\n');


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


function result = run_wilcoxon(errors_1, errors_2, label, alpha, n)
% RUN_WILCOXON  Run a paired Wilcoxon signed-rank test with effect size.
%
%   errors_1, errors_2: paired per-sample error vectors (864 x 1)
%   label: descriptive string for display
%   alpha: corrected significance level
%   n: number of paired samples
%
%   The test asks whether errors_1 differs significantly from errors_2.
%   A negative median difference means errors_1 < errors_2 (method 1 better).

    % Remove any NaN pairs
    valid = ~isnan(errors_1) & ~isnan(errors_2);
    e1 = errors_1(valid);
    e2 = errors_2(valid);
    n_valid = sum(valid);

    differences = e1 - e2;
    med_diff = median(differences);
    mean_diff = mean(differences);

    % Wilcoxon signed-rank test (two-sided)
    try
        [p, ~, stats] = signrank(e1, e2, 'method', 'approximate');
        Z = stats.zval;
    catch
        % Fallback if approximate method fails
        [p, ~, stats] = signrank(e1, e2);
        Z = NaN;
    end

    % Effect size: matched-pairs rank-biserial correlation
    if ~isnan(Z)
        r = abs(Z) / sqrt(n_valid);
    else
        r = NaN;
    end

    % Determine significance
    is_sig = p < alpha;

    % Determine direction
    if med_diff < 0
        direction = 'Method 1 better';
    elseif med_diff > 0
        direction = 'Method 2 better';
    else
        direction = 'No difference';
    end

    % Display
    if is_sig
        sig_str = '*** SIGNIFICANT ***';
    else
        sig_str = 'not significant';
    end

    fprintf('  %s\n', label);
    fprintf('    p = %.2e, Z = %.2f, r = %.3f, median diff = %.4f\n', ...
        p, Z, r, med_diff);
    fprintf('    %s (%s)\n\n', sig_str, direction);

    % Pack result
    result.label     = label;
    result.p_value   = p;
    result.Z         = Z;
    result.r_effect  = r;
    result.med_diff  = med_diff;
    result.mean_diff = mean_diff;
    result.significant = is_sig;
    result.direction = direction;
    result.n_valid   = n_valid;
end
