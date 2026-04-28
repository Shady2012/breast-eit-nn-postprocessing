%% Phase 4 - Approach A: Direct Parameter Estimation
% =========================================================================
% ML-Enhanced EIT for Early Breast Cancer Detection
% Phase 4 of 6: Neural Network Development
%
% Approach A learns the mapping directly from 208 boundary voltage
% differences to the 5 tumour parameters, bypassing image reconstruction.
%
% Input:  208 boundary voltage differences (dV = vi - vh)
% Output: 5 tumour parameters [x_cm, y_cm, z_cm, diameter_mm, contrast]
%
% Dependencies:
%   networks/network_utils.m
%   dataset/training_dataset.mat  (Phase 2)
%
% Output:
%   results_approach_A.mat
%   Figures A1-A2
%
% Recommended execution: Run this FIRST (no EIDORS dependency, fastest).
% =========================================================================

clear; clc; close all;
rng(42, 'twister');  % Reproducibility

fprintf('============================================\n');
fprintf('  PHASE 4 - APPROACH A: DIRECT ESTIMATION\n');
fprintf('============================================\n\n');

%% ========================================================================
% 1. SETUP
% =========================================================================
utils = network_utils();
cfg   = utils.default_config();
archs = utils.architectures();
n_archs = length(archs);

fprintf('Configuration:\n');
fprintf('  Input:        208 boundary voltage differences\n');
fprintf('  Output:       5 tumour parameters\n');
fprintf('  Architectures: %d candidates\n', n_archs);
fprintf('  Loss weights:  pos=%.0f%% size=%.0f%% contrast=%.0f%%\n', ...
    cfg.loss_weights * 100);
fprintf('\n');

%% ========================================================================
% 2. LOAD DATA
% =========================================================================
fprintf('--- Loading Training Dataset ---\n');

dataset_path = utils.find_file('training_dataset.mat', ...
    {'../dataset', '.', 'dataset'});
data = load(dataset_path);

% Training set (all noise levels)
train_idx   = data.split.train;
X_train_raw = data.measurements(:, train_idx);      % 208 x 4032
Y_train_raw = data.tumour_params(train_idx, :);      % 4032 x 5
n_train     = size(X_train_raw, 2);

% Validation set (all noise levels)
val_idx     = data.split.val;
X_val_raw   = data.measurements(:, val_idx);          % 208 x 864
Y_val_raw   = data.tumour_params(val_idx, :);          % 864 x 5
n_val       = size(X_val_raw, 2);

% Test set (all noise levels)
test_idx    = data.split.test;
X_test_raw  = data.measurements(:, test_idx);          % 208 x 864
Y_test_raw  = data.tumour_params(test_idx, :);          % 864 x 5
test_noise  = data.noise_levels(test_idx);              % 864 x 1
n_test      = size(X_test_raw, 2);

fprintf('  Train: %d    Val: %d    Test: %d\n', n_train, n_val, n_test);

%% ========================================================================
% 3. NORMALISE DATA
% =========================================================================
fprintf('--- Normalising Data ---\n');

% Input: z-score from training statistics
X_mu  = mean(X_train_raw, 2);
X_sig = std(X_train_raw, 0, 2);
X_sig(X_sig < eps) = 1;  % Prevent division by zero

X_train = (X_train_raw - X_mu) ./ X_sig;
X_val   = (X_val_raw   - X_mu) ./ X_sig;
X_test  = (X_test_raw  - X_mu) ./ X_sig;

% Output: z-score from training statistics (transpose for [5 x N] format)
Y_mu  = mean(Y_train_raw, 1);     % 1 x 5
Y_sig = std(Y_train_raw, 0, 1);   % 1 x 5
Y_sig(Y_sig < eps) = 1;

Y_train = ((Y_train_raw - Y_mu) ./ Y_sig)';  % 5 x 4032
Y_val   = ((Y_val_raw   - Y_mu) ./ Y_sig)';  % 5 x 864
Y_test  = ((Y_test_raw  - Y_mu) ./ Y_sig)';  % 5 x 864

fprintf('  Input features: %d    Output parameters: %d\n', ...
    size(X_train, 1), size(Y_train, 1));
fprintf('  Y ranges (raw):  x=[%.1f,%.1f]  diam=[%.0f,%.0f]  contrast=[%.1f,%.1f]\n', ...
    min(Y_train_raw(:,1)), max(Y_train_raw(:,1)), ...
    min(Y_train_raw(:,4)), max(Y_train_raw(:,4)), ...
    min(Y_train_raw(:,5)), max(Y_train_raw(:,5)));

%% ========================================================================
% 4. ARCHITECTURE SEARCH
% =========================================================================
fprintf('\n=== ARCHITECTURE SEARCH ===\n');
fprintf('  Testing %d architectures, selecting by validation loss.\n\n', n_archs);

arch_results = struct();
input_size   = size(X_train, 1);  % 208
output_size  = 5;

for a = 1:n_archs
    fprintf('  [%d/%d] %s: layers = [%s]\n', a, n_archs, archs(a).name, ...
        strjoin(arrayfun(@num2str, archs(a).layers, 'Uni', false), ', '));

    % Build network
    net = utils.build_network(input_size, archs(a).layers, ...
        output_size, cfg.dropout_rate);

    % Train
    t_start = tic;
    [trained_net, history] = utils.train_network(net, X_train, Y_train, ...
        X_val, Y_val, cfg);
    t_elapsed = toc(t_start);

    % Store results
    arch_results(a).name       = archs(a).name;
    arch_results(a).layers     = archs(a).layers;
    arch_results(a).net        = trained_net;
    arch_results(a).history    = history;
    arch_results(a).val_loss   = history.best_val_loss;
    arch_results(a).best_epoch = history.best_epoch;
    arch_results(a).time       = t_elapsed;

    fprintf('    >> val_loss=%.4f  best_epoch=%d  time=%.1fs\n\n', ...
        history.best_val_loss, history.best_epoch, t_elapsed);
end

% Select best architecture
val_losses = [arch_results.val_loss];
[best_val, best_idx] = min(val_losses);
best_arch = arch_results(best_idx);

fprintf('--- Best Architecture: %s (val_loss=%.4f) ---\n\n', ...
    best_arch.name, best_val);

%% ========================================================================
% 5. EVALUATE ON TEST SET
% =========================================================================
fprintf('=== EVALUATING ON TEST SET ===\n');

% Forward pass (inference mode, no dropout)
Y_test_pred_norm = double(extractdata(predict(best_arch.net, ...
    dlarray(X_test, 'CB'))));  % 5 x 864

% Denormalise predictions back to physical units
Y_test_pred = (Y_test_pred_norm' .* Y_sig + Y_mu);  % 864 x 5

% Compute error metrics
results_A = utils.evaluate(Y_test_pred, Y_test_raw, test_noise);

% Print summary
utils.print_summary(results_A, 'Approach A (Direct Estimation)');
utils.print_breakdown(results_A, 'Approach A');

% Comparison with Phase 3 baseline
fprintf('\n--- Comparison with Phase 3 Baseline ---\n');
fprintf('                          Baseline     Approach A   Improvement\n');
fprintf('  Position (cm):          0.29         %.2f         %+.1f%%\n', ...
    results_A.pos_mean, (1 - results_A.pos_mean / 0.29) * 100);
fprintf('  Size (mm):              6.2          %.1f         %+.1f%%\n', ...
    results_A.size_abs_mean, (1 - results_A.size_abs_mean / 6.2) * 100);
fprintf('  Contrast:               ~2.2         %.2f         (baseline failed)\n', ...
    results_A.contrast_mean);

%% ========================================================================
% 6. SAVE RESULTS
% =========================================================================
fprintf('\n=== Saving Results ===\n');

save_data = struct();
save_data.approach        = 'A_direct_estimation';
save_data.best_arch_name  = best_arch.name;
save_data.best_arch_layers = best_arch.layers;
save_data.best_net        = best_arch.net;
save_data.best_history    = best_arch.history;
save_data.arch_results    = rmfield(arch_results, 'net');  % Save space
save_data.predictions     = Y_test_pred;
save_data.test_params     = Y_test_raw;
save_data.test_noise      = test_noise;
save_data.results         = results_A;
save_data.normalisation.X_mu  = X_mu;
save_data.normalisation.X_sig = X_sig;
save_data.normalisation.Y_mu  = Y_mu;
save_data.normalisation.Y_sig = Y_sig;
save_data.metadata.date   = datestr(now);
save_data.metadata.config = cfg;

save('results_approach_A.mat', '-struct', 'save_data', '-v7.3');
finfo = dir('results_approach_A.mat');
fprintf('  Saved: results_approach_A.mat (%.1f MB)\n', finfo.bytes / 1e6);

%% ========================================================================
% 7. GENERATE FIGURES
% =========================================================================
fprintf('\n=== Generating Figures ===\n');

% --- Figure A1: Training Curves (best architecture) ---
figure('Name', 'Fig A1: Training Curves', ...
    'Position', [50, 400, 800, 350], 'Color', 'w');

plot(1:best_arch.history.total_epochs, best_arch.history.train_loss, ...
    'b-', 'LineWidth', 1.5);
hold on;
plot(1:best_arch.history.total_epochs, best_arch.history.val_loss, ...
    'r-', 'LineWidth', 1.5);
xline(best_arch.history.best_epoch, 'k--', 'LineWidth', 1);
hold off;

xlabel('Epoch');
ylabel('Weighted MSE Loss');
title(sprintf('Approach A: %s - Training Curves', best_arch.name));
legend('Train', 'Validation', 'Best Epoch', 'Location', 'northeast');
grid on;

% --- Figure A2: Architecture Comparison ---
figure('Name', 'Fig A2: Architecture Comparison', ...
    'Position', [50, 300, 600, 350], 'Color', 'w');

bar_vals = [arch_results.val_loss];
bar_colors = repmat([0.4 0.6 0.8], n_archs, 1);
bar_colors(best_idx, :) = [0.2 0.7 0.3];  % Highlight best in green

b = bar(bar_vals);
b.FaceColor = 'flat';
b.CData = bar_colors;

set(gca, 'XTickLabel', {arch_results.name}, 'XTickLabelRotation', 15);
ylabel('Best Validation Loss');
title('Approach A: Architecture Search');
grid on;

% Mark best
hold on;
text(best_idx, bar_vals(best_idx), sprintf('  %.4f', bar_vals(best_idx)), ...
    'VerticalAlignment', 'bottom', 'FontWeight', 'bold');
hold off;

%% ========================================================================
fprintf('\n============================================\n');
fprintf('  APPROACH A COMPLETE\n');
fprintf('  Best architecture: %s\n', best_arch.name);
fprintf('  Position error:    %.2f cm\n', results_A.pos_mean);
fprintf('  Size error:        %.1f mm\n', results_A.size_abs_mean);
fprintf('  Contrast error:    %.2f\n', results_A.contrast_mean);
fprintf('============================================\n');
