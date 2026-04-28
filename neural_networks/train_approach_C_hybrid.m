%% Phase 4 - Approach C: Hybrid (Measurements + Reconstruction Features)
% =========================================================================
% ML-Enhanced EIT for Early Breast Cancer Detection
% Phase 4 of 6: Neural Network Development
%
% Approach C feeds the network BOTH raw boundary measurements AND
% PCA-compressed EIDORS reconstruction features. This gives the network
% access to all available information: the raw signal (for contrast) and
% the physics-informed spatial prior (for position).
%
% Tests both Tikhonov and NOSER PCA features (matching Approach B).
%
% Input:  208 measurements + K PCA scores = (208 + K) features
% Output: 5 tumour parameters [x_cm, y_cm, z_cm, diameter_mm, contrast]
%
% Dependencies:
%   networks/network_utils.m
%   dataset/training_dataset.mat            (Phase 2)
%   networks/results_approach_B.mat         (Phase 4B, for PCA models)
%
% Output:
%   results_approach_C.mat
%   Figures C1-C3
%
% Recommended execution: Run AFTER Approach B (needs PCA models from B).
% =========================================================================

clear; clc; close all;
rng(42, 'twister');

fprintf('============================================\n');
fprintf('  PHASE 4 - APPROACH C: HYBRID\n');
fprintf('============================================\n\n');

%% ========================================================================
% 1. SETUP
% =========================================================================
utils = network_utils();
cfg   = utils.default_config();
archs = utils.architectures();
n_archs = length(archs);

prior_names = {'NOSER', 'Tikhonov'};
n_priors    = length(prior_names);

fprintf('Configuration:\n');
fprintf('  Input: 208 measurements + K PCA scores (concatenated)\n');
fprintf('  Priors tested: %s\n', strjoin(prior_names, ', '));
fprintf('  Architectures: %d candidates per prior\n', n_archs);
fprintf('\n');

%% ========================================================================
% 2. LOAD DATA
% =========================================================================
fprintf('--- Loading Datasets ---\n');

% Training dataset (Phase 2)
dataset_path = utils.find_file('training_dataset.mat', ...
    {'../dataset', '.', 'dataset'});
data = load(dataset_path);

train_idx  = data.split.train;
val_idx    = data.split.val;
test_idx   = data.split.test;

meas_train = data.measurements(:, train_idx);   % 208 x 4032
meas_val   = data.measurements(:, val_idx);      % 208 x 864
meas_test  = data.measurements(:, test_idx);     % 208 x 864

Y_train_raw = data.tumour_params(train_idx, :);  % 4032 x 5
Y_val_raw   = data.tumour_params(val_idx, :);    % 864 x 5
Y_test_raw  = data.tumour_params(test_idx, :);   % 864 x 5
test_noise  = data.noise_levels(test_idx);

n_train = size(meas_train, 2);
n_val   = size(meas_val, 2);
n_test  = size(meas_test, 2);

fprintf('  Train: %d    Val: %d    Test: %d\n', n_train, n_val, n_test);

% PCA models from Approach B
approach_B_path = utils.find_file('results_approach_B.mat', {'.', '../networks'});
B_data = load(approach_B_path, 'pca_models');
pca_models = B_data.pca_models;

fprintf('  PCA models loaded from Approach B.\n');
for p = 1:n_priors
    fprintf('    %s: K=%d components\n', pca_models(p).prior_name, pca_models(p).K);
end
fprintf('\n');

%% ========================================================================
% 3. BUILD CONCATENATED FEATURES
% =========================================================================
fprintf('--- Building Hybrid Feature Vectors ---\n');

% For each prior, concatenate: [measurements; PCA_scores]
% Measurements are the same for all priors; PCA scores differ.

hybrid_data = struct();

for p = 1:n_priors
    K = pca_models(p).K;

    % PCA scores: already computed by Approach B (n_samples x K)
    pca_train = pca_models(p).scores_train';  % K x n_train
    pca_val   = pca_models(p).scores_val';    % K x n_val
    pca_test  = pca_models(p).scores_test';   % K x n_test

    % Concatenate: [208 measurements; K PCA scores]
    X_train_cat = [meas_train; pca_train];    % (208+K) x n_train
    X_val_cat   = [meas_val;   pca_val];      % (208+K) x n_val
    X_test_cat  = [meas_test;  pca_test];     % (208+K) x n_test

    % Normalise (z-score from training stats)
    X_mu  = mean(X_train_cat, 2);
    X_sig = std(X_train_cat, 0, 2);
    X_sig(X_sig < eps) = 1;

    hybrid_data(p).prior      = prior_names{p};
    hybrid_data(p).input_size = 208 + K;
    hybrid_data(p).X_train    = (X_train_cat - X_mu) ./ X_sig;
    hybrid_data(p).X_val      = (X_val_cat   - X_mu) ./ X_sig;
    hybrid_data(p).X_test     = (X_test_cat  - X_mu) ./ X_sig;
    hybrid_data(p).X_mu       = X_mu;
    hybrid_data(p).X_sig      = X_sig;

    fprintf('  %s: input_size = 208 + %d = %d\n', ...
        prior_names{p}, K, hybrid_data(p).input_size);
end

%% ========================================================================
% 4. NORMALISE OUTPUTS
% =========================================================================
Y_mu  = mean(Y_train_raw, 1);
Y_sig = std(Y_train_raw, 0, 1);
Y_sig(Y_sig < eps) = 1;

Y_train_norm = ((Y_train_raw - Y_mu) ./ Y_sig)';  % 5 x n_train
Y_val_norm   = ((Y_val_raw   - Y_mu) ./ Y_sig)';  % 5 x n_val

%% ========================================================================
% 5. ARCHITECTURE SEARCH (for each prior)
% =========================================================================
fprintf('\n=== ARCHITECTURE SEARCH ===\n');
fprintf('  %d priors x %d architectures = %d trainings\n\n', ...
    n_priors, n_archs, n_priors * n_archs);

all_results = struct();
output_size = 5;

for p = 1:n_priors
    fprintf('--- Prior: %s (input_size=%d) ---\n', ...
        prior_names{p}, hybrid_data(p).input_size);

    for a = 1:n_archs
        fprintf('\n  [%s / %s] layers = [%s]\n', ...
            prior_names{p}, archs(a).name, ...
            strjoin(arrayfun(@num2str, archs(a).layers, 'Uni', false), ', '));

        net = utils.build_network(hybrid_data(p).input_size, ...
            archs(a).layers, output_size, cfg.dropout_rate);

        t_start = tic;
        [trained_net, history] = utils.train_network(net, ...
            hybrid_data(p).X_train, Y_train_norm, ...
            hybrid_data(p).X_val, Y_val_norm, cfg);
        t_elapsed = toc(t_start);

        idx = (p - 1) * n_archs + a;
        all_results(idx).prior      = prior_names{p};
        all_results(idx).arch       = archs(a).name;
        all_results(idx).layers     = archs(a).layers;
        all_results(idx).net        = trained_net;
        all_results(idx).history    = history;
        all_results(idx).val_loss   = history.best_val_loss;
        all_results(idx).prior_idx  = p;

        fprintf('    >> val_loss=%.4f  best_epoch=%d  time=%.1fs\n', ...
            history.best_val_loss, history.best_epoch, t_elapsed);
    end
end

% Select overall best
val_losses = [all_results.val_loss];
[best_val, best_idx] = min(val_losses);
best = all_results(best_idx);

fprintf('\n--- Best Combination: %s + %s (val_loss=%.4f) ---\n\n', ...
    best.prior, best.arch, best_val);

% Best per prior
for p = 1:n_priors
    prior_mask = strcmp({all_results.prior}, prior_names{p});
    prior_losses = val_losses;
    prior_losses(~prior_mask) = Inf;
    [pv, pi] = min(prior_losses);
    fprintf('  Best for %s: %s (val_loss=%.4f)\n', ...
        prior_names{p}, all_results(pi).arch, pv);
end

%% ========================================================================
% 6. EVALUATE ON TEST SET
% =========================================================================
fprintf('\n=== EVALUATING ON TEST SET ===\n');

% Use the winning prior's test features
bp = best.prior_idx;
X_test_n = hybrid_data(bp).X_test;

Y_pred_norm = double(extractdata(predict(best.net, dlarray(X_test_n, 'CB'))));
Y_pred = (Y_pred_norm' .* Y_sig + Y_mu);  % 864 x 5

results_C = utils.evaluate(Y_pred, Y_test_raw, test_noise);

utils.print_summary(results_C, sprintf('Approach C (%s + %s)', best.prior, best.arch));
utils.print_breakdown(results_C, 'Approach C');

% Per-prior test results
fprintf('\n--- Per-Prior Test Results ---\n');
prior_test_results = struct();
for p = 1:n_priors
    prior_mask = strcmp({all_results.prior}, prior_names{p});
    prior_losses = val_losses;
    prior_losses(~prior_mask) = Inf;
    [~, pi] = min(prior_losses);
    pr = all_results(pi);

    X_te = hybrid_data(p).X_test;
    Yp = double(extractdata(predict(pr.net, dlarray(X_te, 'CB'))));
    Yp = (Yp' .* Y_sig + Y_mu);

    prior_test_results(p).results = utils.evaluate(Yp, Y_test_raw, test_noise);
    prior_test_results(p).prior   = prior_names{p};
    prior_test_results(p).arch    = pr.arch;

    utils.print_summary(prior_test_results(p).results, ...
        sprintf('%s + %s', prior_names{p}, pr.arch));
end

% Comparison with baseline and other approaches
fprintf('\n--- Comparison with Phase 3 Baseline ---\n');
fprintf('                          Baseline     Approach C   Improvement\n');
fprintf('  Position (cm):          0.29         %.2f         %+.1f%%\n', ...
    results_C.pos_mean, (1 - results_C.pos_mean / 0.29) * 100);
fprintf('  Size (mm):              6.2          %.1f         %+.1f%%\n', ...
    results_C.size_abs_mean, (1 - results_C.size_abs_mean / 6.2) * 100);
fprintf('  Contrast:               ~2.2         %.2f         (baseline failed)\n', ...
    results_C.contrast_mean);

%% ========================================================================
% 7. SAVE RESULTS
% =========================================================================
fprintf('\n=== Saving Results ===\n');

save_data = struct();
save_data.approach          = 'C_hybrid';
save_data.best_prior        = best.prior;
save_data.best_arch_name    = best.arch;
save_data.best_arch_layers  = best.layers;
save_data.best_net          = best.net;
save_data.best_history      = best.history;
save_data.arch_results      = rmfield(all_results, 'net');
save_data.predictions       = Y_pred;
save_data.test_params       = Y_test_raw;
save_data.test_noise        = test_noise;
save_data.results           = results_C;
save_data.prior_test_results = prior_test_results;

save_data.normalisation.Y_mu  = Y_mu;
save_data.normalisation.Y_sig = Y_sig;

save_data.metadata.date   = datestr(now);
save_data.metadata.config = cfg;

save('results_approach_C.mat', '-struct', 'save_data', '-v7.3');
finfo = dir('results_approach_C.mat');
fprintf('  Saved: results_approach_C.mat (%.1f MB)\n', finfo.bytes / 1e6);

%% ========================================================================
% 8. GENERATE FIGURES
% =========================================================================
fprintf('\n=== Generating Figures ===\n');

% --- Figure C1: Training Curves (best combination) ---
figure('Name', 'Fig C1: Training Curves', ...
    'Position', [50, 400, 800, 350], 'Color', 'w');

plot(1:best.history.total_epochs, best.history.train_loss, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:best.history.total_epochs, best.history.val_loss, 'r-', 'LineWidth', 1.5);
xline(best.history.best_epoch, 'k--', 'LineWidth', 1);
hold off;

xlabel('Epoch');
ylabel('Weighted MSE Loss');
title(sprintf('Approach C: %s + %s - Training Curves', best.prior, best.arch));
legend('Train', 'Validation', 'Best Epoch', 'Location', 'northeast');
grid on;

% --- Figure C2: Architecture Comparison (grouped by prior) ---
figure('Name', 'Fig C2: Architecture Search', ...
    'Position', [50, 300, 900, 400], 'Color', 'w');

bar_data = reshape(val_losses, n_archs, n_priors)';
b = bar(bar_data, 'grouped');
set(gca, 'XTickLabel', prior_names);
ylabel('Best Validation Loss');
title('Approach C: Architecture Search by Prior');
legend({archs.name}, 'Location', 'northeastoutside');
grid on;

% --- Figure C3: Prior Comparison (test set) ---
figure('Name', 'Fig C3: Prior Comparison', ...
    'Position', [50, 200, 600, 400], 'Color', 'w');

prior_pos  = [prior_test_results(1).results.pos_mean, ...
              prior_test_results(2).results.pos_mean];
prior_size = [prior_test_results(1).results.size_abs_mean, ...
              prior_test_results(2).results.size_abs_mean];

subplot(1, 2, 1);
bar(prior_pos);
set(gca, 'XTickLabel', prior_names);
ylabel('Mean Position Error (cm)');
title('Position Accuracy by Prior');
grid on;

subplot(1, 2, 2);
bar(prior_size);
set(gca, 'XTickLabel', prior_names);
ylabel('Mean Size Error (mm)');
title('Size Accuracy by Prior');
grid on;

sgtitle('Approach C: Hybrid Feature Prior Comparison');

%% ========================================================================
% FINAL CROSS-APPROACH SUMMARY
% =========================================================================
fprintf('\n============================================\n');
fprintf('  APPROACH C COMPLETE\n');
fprintf('  Best prior: %s\n', best.prior);
fprintf('  Best architecture: %s\n', best.arch);
fprintf('  Position error: %.2f cm\n', results_C.pos_mean);
fprintf('  Size error:     %.1f mm\n', results_C.size_abs_mean);
fprintf('  Contrast error: %.2f\n', results_C.contrast_mean);
fprintf('============================================\n');

% Load Approach A results if available for quick comparison
fprintf('\n--- Quick Cross-Approach Comparison ---\n');
fprintf('%-25s  %8s  %8s  %8s\n', 'Method', 'Pos(cm)', 'Size(mm)', 'Contrast');
fprintf('%s\n', repmat('-', 1, 60));
fprintf('%-25s  %8s  %8s  %8s\n', 'Baseline (Tik+Otsu)', '0.29', '20.4', '~2.2');
fprintf('%-25s  %8s  %8s  %8s\n', 'Baseline (Tik+fix50)', '0.42', '6.2', '~2.2');

if exist('results_approach_A.mat', 'file')
    A = load('results_approach_A.mat', 'results');
    fprintf('%-25s  %8.2f  %8.1f  %8.2f\n', 'Approach A (Direct)', ...
        A.results.pos_mean, A.results.size_abs_mean, A.results.contrast_mean);
end

if exist('results_approach_B.mat', 'file')
    B = load('results_approach_B.mat', 'results');
    fprintf('%-25s  %8.2f  %8.1f  %8.2f\n', 'Approach B (Post-Proc)', ...
        B.results.pos_mean, B.results.size_abs_mean, B.results.contrast_mean);
end

fprintf('%-25s  %8.2f  %8.1f  %8.2f\n', 'Approach C (Hybrid)', ...
    results_C.pos_mean, results_C.size_abs_mean, results_C.contrast_mean);

fprintf('\nAll three approach results are saved. Proceed to Phase 5.\n');
