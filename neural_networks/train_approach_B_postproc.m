%% Phase 4 - Approach B: Post-Processing (Reconstruction Enhancement)
% =========================================================================
% ML-Enhanced EIT for Early Breast Cancer Detection
% Phase 4 of 6: Neural Network Development
%
% Approach B uses EIDORS reconstructions as physics-informed features.
% PCA compresses the 19,403-element reconstruction into K components,
% and the network maps these to tumour parameters.
%
% Tests BOTH Tikhonov and NOSER priors as reconstruction input (Q5).
%
% Input:  K PCA scores from EIDORS reconstruction
% Output: 5 tumour parameters [x_cm, y_cm, z_cm, diameter_mm, contrast]
%
% Dependencies:
%   networks/network_utils.m
%   dataset/training_dataset.mat            (Phase 2)
%   baseline/baseline_results.mat           (Phase 3)
%   geometry/build_breast_model.m           (Phase 1)
%
% Output:
%   results_approach_B.mat  (includes PCA models for Approach C)
%   Figures B1-B4
%
% Recommended execution: Run AFTER Approach A.
% =========================================================================

clear; clc; close all;
rng(42, 'twister');

fprintf('============================================\n');
fprintf('  PHASE 4 - APPROACH B: POST-PROCESSING\n');
fprintf('============================================\n\n');

%% ========================================================================
% 1. SETUP
% =========================================================================
utils = network_utils();
cfg   = utils.default_config();
archs = utils.architectures();
n_archs = length(archs);

% Background conductivity (must match Phase 1/2/3)
background_cond = 0.3;

% Priors to test for reconstruction input
prior_names   = {'NOSER', 'Tikhonov'};
prior_handles = {@prior_noser, @prior_tikhonov};
n_priors      = length(prior_names);

% PCA variance threshold
pca_var_threshold = 95;

fprintf('Configuration:\n');
fprintf('  Priors tested:    %s\n', strjoin(prior_names, ', '));
fprintf('  PCA threshold:    %d%% variance retained\n', pca_var_threshold);
fprintf('  Architectures:    %d candidates per prior\n', n_archs);
fprintf('\n');

%% ========================================================================
% 2. LOAD DATA
% =========================================================================
fprintf('--- Loading Datasets ---\n');

% Training dataset (Phase 2)
dataset_path = utils.find_file('training_dataset.mat', ...
    {'../dataset', '.', 'dataset'});
data = load(dataset_path);

train_idx   = data.split.train;
val_idx     = data.split.val;
test_idx    = data.split.test;

train_meas  = data.measurements(:, train_idx);   % 208 x 4032
val_meas    = data.measurements(:, val_idx);      % 208 x 864
test_meas   = data.measurements(:, test_idx);     % 208 x 864

Y_train_raw = data.tumour_params(train_idx, :);   % 4032 x 5
Y_val_raw   = data.tumour_params(val_idx, :);     % 864 x 5
Y_test_raw  = data.tumour_params(test_idx, :);    % 864 x 5
test_noise  = data.noise_levels(test_idx);

n_train = size(train_meas, 2);
n_val   = size(val_meas, 2);
n_test  = size(test_meas, 2);

fprintf('  Train: %d    Val: %d    Test: %d\n', n_train, n_val, n_test);

% Baseline results (Phase 3) - for test reconstructions and optimal HPs
baseline_path = utils.find_file('baseline_results.mat', ...
    {'../baseline', '.', 'baseline'});
baseline = load(baseline_path, 'recon_images', 'optimal_hp', 'test_indices');

% Verify test indices match
assert(isequal(baseline.test_indices(:), test_idx(:)), ...
    'Test indices mismatch between Phase 2 split and Phase 3 results.');

% Test set reconstructions from Phase 3: recon_images is [19403 x 864 x 3]
% Prior ordering: 1=NOSER, 2=Tikhonov, 3=Laplace
recon_test_noser = baseline.recon_images(:, :, 1);   % 19403 x 864
recon_test_tik   = baseline.recon_images(:, :, 2);   % 19403 x 864
optimal_hp       = baseline.optimal_hp;               % [hp_noser, hp_tik, hp_lap]

fprintf('  Test reconstructions loaded from Phase 3.\n');
fprintf('  Optimal HPs: NOSER=%.4f, Tikhonov=%.4f\n', ...
    optimal_hp(1), optimal_hp(2));
fprintf('\n');

%% ========================================================================
% 3. RECONSTRUCT TRAINING AND VALIDATION SETS
% =========================================================================
fprintf('=== Reconstructing Training/Validation Sets ===\n');
fprintf('  This requires EIDORS. Using single inverse model per prior\n');
fprintf('  to avoid cache thrashing (lesson from Phase 3 Stage A).\n\n');

% Rebuild forward model
fprintf('  Building forward model...\n');
fmdl = build_breast_model();
img_h   = mk_image(fmdl, background_cond);
vh_data = fwd_solve(img_h);
n_elems = size(fmdl.elems, 1);

fprintf('  Forward model: %d elements, %d measurements\n', ...
    n_elems, length(vh_data.meas));

% Storage for reconstructions
recon_train = zeros(n_elems, n_train, n_priors);
recon_val   = zeros(n_elems, n_val, n_priors);

for p = 1:n_priors
    fprintf('\n  Reconstructing with %s (HP=%.4f)...\n', ...
        prior_names{p}, optimal_hp(p));

    % Build ONE inverse model (cache-friendly)
    imdl = eidors_obj('inv_model', sprintf('phase4_%s', prior_names{p}));
    imdl.reconst_type         = 'difference';
    imdl.solve                = @inv_solve_diff_GN_one_step;
    imdl.fwd_model            = fmdl;
    imdl.jacobian_bkgnd.value = background_cond;
    imdl.RtR_prior            = prior_handles{p};
    imdl.hyperparameter.value = optimal_hp(p);

    % --- Training set ---
    t_start = tic;
    for s = 1:n_train
        vi_data = vh_data;
        vi_data.meas = vh_data.meas + train_meas(:, s);
        img_recon = inv_solve(imdl, vh_data, vi_data);
        recon_train(:, s, p) = img_recon.elem_data;

        if mod(s, 500) == 0
            fprintf('    Train: [%d/%d]\n', s, n_train);
        end
    end
    t_train = toc(t_start);
    fprintf('    Training set: %.1f min (%d solves)\n', t_train/60, n_train);

    % --- Validation set ---
    t_start = tic;
    for s = 1:n_val
        vi_data = vh_data;
        vi_data.meas = vh_data.meas + val_meas(:, s);
        img_recon = inv_solve(imdl, vh_data, vi_data);
        recon_val(:, s, p) = img_recon.elem_data;
    end
    t_val = toc(t_start);
    fprintf('    Validation set: %.1f min (%d solves)\n', t_val/60, n_val);
end

fprintf('\n  Reconstruction complete.\n\n');

%% ========================================================================
% 4. FIT PCA (separately for each prior)
% =========================================================================
fprintf('=== Fitting PCA ===\n');

pca_models = struct();

% Collect test reconstructions per prior
recon_test_all = zeros(n_elems, n_test, n_priors);
recon_test_all(:, :, 1) = recon_test_noser;
recon_test_all(:, :, 2) = recon_test_tik;

for p = 1:n_priors
    fprintf('\n  PCA for %s reconstructions...\n', prior_names{p});

    % Fit PCA on training data (samples x features)
    train_data = recon_train(:, :, p)';   % n_train x n_elems
    [coeff, scores, ~, ~, explained] = pca(train_data, 'Economy', true);

    cum_var = cumsum(explained);
    K = find(cum_var >= pca_var_threshold, 1);

    fprintf('    Components for %d%% variance: K = %d\n', pca_var_threshold, K);
    fprintf('    Variance captured: %.1f%%\n', cum_var(K));

    % Store PCA model
    pca_models(p).prior_name = prior_names{p};
    pca_models(p).coeff      = coeff(:, 1:K);         % n_elems x K
    pca_models(p).data_mean  = mean(train_data, 1);    % 1 x n_elems
    pca_models(p).K          = K;
    pca_models(p).explained  = explained;
    pca_models(p).cum_var    = cum_var;

    % Compute PCA scores for all splits
    pca_models(p).scores_train = scores(:, 1:K);       % n_train x K

    val_centered = recon_val(:, :, p)' - pca_models(p).data_mean;
    pca_models(p).scores_val = val_centered * coeff(:, 1:K);  % n_val x K

    test_centered = recon_test_all(:, :, p)' - pca_models(p).data_mean;
    pca_models(p).scores_test = test_centered * coeff(:, 1:K);  % n_test x K
end

%% ========================================================================
% 5. NORMALISE OUTPUTS
% =========================================================================
Y_mu  = mean(Y_train_raw, 1);
Y_sig = std(Y_train_raw, 0, 1);
Y_sig(Y_sig < eps) = 1;

Y_train_norm = ((Y_train_raw - Y_mu) ./ Y_sig)';  % 5 x n_train
Y_val_norm   = ((Y_val_raw   - Y_mu) ./ Y_sig)';  % 5 x n_val

%% ========================================================================
% 6. ARCHITECTURE SEARCH (for each prior)
% =========================================================================
fprintf('\n=== ARCHITECTURE SEARCH ===\n');
fprintf('  %d priors x %d architectures = %d trainings\n\n', ...
    n_priors, n_archs, n_priors * n_archs);

all_results = struct();
output_size = 5;

for p = 1:n_priors
    fprintf('--- Prior: %s (K=%d PCA components) ---\n', ...
        prior_names{p}, pca_models(p).K);

    % Normalise PCA scores (input)
    X_tr = pca_models(p).scores_train';  % K x n_train
    X_va = pca_models(p).scores_val';    % K x n_val

    X_mu  = mean(X_tr, 2);
    X_sig = std(X_tr, 0, 2);
    X_sig(X_sig < eps) = 1;

    X_tr_n = (X_tr - X_mu) ./ X_sig;
    X_va_n = (X_va - X_mu) ./ X_sig;

    input_size = pca_models(p).K;

    for a = 1:n_archs
        fprintf('\n  [%s / %s] layers = [%s]\n', ...
            prior_names{p}, archs(a).name, ...
            strjoin(arrayfun(@num2str, archs(a).layers, 'Uni', false), ', '));

        net = utils.build_network(input_size, archs(a).layers, ...
            output_size, cfg.dropout_rate);

        t_start = tic;
        [trained_net, history] = utils.train_network(net, ...
            X_tr_n, Y_train_norm, X_va_n, Y_val_norm, cfg);
        t_elapsed = toc(t_start);

        idx = (p - 1) * n_archs + a;
        all_results(idx).prior     = prior_names{p};
        all_results(idx).arch      = archs(a).name;
        all_results(idx).layers    = archs(a).layers;
        all_results(idx).net       = trained_net;
        all_results(idx).history   = history;
        all_results(idx).val_loss  = history.best_val_loss;
        all_results(idx).input_size = input_size;
        all_results(idx).X_mu     = X_mu;
        all_results(idx).X_sig    = X_sig;

        fprintf('    >> val_loss=%.4f  best_epoch=%d  time=%.1fs\n', ...
            history.best_val_loss, history.best_epoch, t_elapsed);
    end
end

% Select overall best (across all priors and architectures)
val_losses = [all_results.val_loss];
[best_val, best_idx] = min(val_losses);
best = all_results(best_idx);

fprintf('\n--- Best Combination: %s + %s (val_loss=%.4f) ---\n\n', ...
    best.prior, best.arch, best_val);

% Also find best per prior (for comparison)
for p = 1:n_priors
    prior_mask = strcmp({all_results.prior}, prior_names{p});
    prior_losses = val_losses;
    prior_losses(~prior_mask) = Inf;
    [pv, pi] = min(prior_losses);
    fprintf('  Best for %s: %s (val_loss=%.4f)\n', ...
        prior_names{p}, all_results(pi).arch, pv);
end

%% ========================================================================
% 7. EVALUATE ON TEST SET
% =========================================================================
fprintf('\n=== EVALUATING ON TEST SET ===\n');

% Determine which prior index won
best_prior_idx = find(strcmp(prior_names, best.prior));

% Normalise test PCA scores using the winning prior's statistics
X_test_pca = pca_models(best_prior_idx).scores_test';  % K x n_test
X_test_n   = (X_test_pca - best.X_mu) ./ best.X_sig;

% Forward pass
Y_pred_norm = double(extractdata(predict(best.net, dlarray(X_test_n, 'CB'))));
Y_pred = (Y_pred_norm' .* Y_sig + Y_mu);  % 864 x 5

% Compute errors
results_B = utils.evaluate(Y_pred, Y_test_raw, test_noise);

utils.print_summary(results_B, sprintf('Approach B (%s + %s)', best.prior, best.arch));
utils.print_breakdown(results_B, 'Approach B');

% Also evaluate the other prior's best for comparison
fprintf('\n--- Per-Prior Test Results ---\n');
prior_test_results = struct();
for p = 1:n_priors
    prior_mask = strcmp({all_results.prior}, prior_names{p});
    prior_losses = val_losses;
    prior_losses(~prior_mask) = Inf;
    [~, pi] = min(prior_losses);
    pr = all_results(pi);

    X_te = pca_models(p).scores_test';
    X_te_n = (X_te - pr.X_mu) ./ pr.X_sig;
    Yp = double(extractdata(predict(pr.net, dlarray(X_te_n, 'CB'))));
    Yp = (Yp' .* Y_sig + Y_mu);

    prior_test_results(p).results = utils.evaluate(Yp, Y_test_raw, test_noise);
    prior_test_results(p).prior   = prior_names{p};
    prior_test_results(p).arch    = pr.arch;

    utils.print_summary(prior_test_results(p).results, ...
        sprintf('%s + %s', prior_names{p}, pr.arch));
end

% Comparison with baseline
fprintf('\n--- Comparison with Phase 3 Baseline ---\n');
fprintf('                          Baseline     Approach B   Improvement\n');
fprintf('  Position (cm):          0.29         %.2f         %+.1f%%\n', ...
    results_B.pos_mean, (1 - results_B.pos_mean / 0.29) * 100);
fprintf('  Size (mm):              6.2          %.1f         %+.1f%%\n', ...
    results_B.size_abs_mean, (1 - results_B.size_abs_mean / 6.2) * 100);
fprintf('  Contrast:               ~2.2         %.2f         (baseline failed)\n', ...
    results_B.contrast_mean);

%% ========================================================================
% 8. SAVE RESULTS
% =========================================================================
fprintf('\n=== Saving Results ===\n');

save_data = struct();
save_data.approach          = 'B_post_processing';
save_data.best_prior        = best.prior;
save_data.best_arch_name    = best.arch;
save_data.best_arch_layers  = best.layers;
save_data.best_net          = best.net;
save_data.best_history      = best.history;
save_data.arch_results      = rmfield(all_results, 'net');
save_data.predictions       = Y_pred;
save_data.test_params       = Y_test_raw;
save_data.test_noise        = test_noise;
save_data.results           = results_B;
save_data.prior_test_results = prior_test_results;

% PCA models (needed by Approach C)
save_data.pca_models        = pca_models;

% Normalisation
save_data.normalisation.Y_mu  = Y_mu;
save_data.normalisation.Y_sig = Y_sig;
save_data.normalisation.X_mu  = best.X_mu;
save_data.normalisation.X_sig = best.X_sig;

save_data.metadata.date            = datestr(now);
save_data.metadata.config          = cfg;
save_data.metadata.pca_threshold   = pca_var_threshold;
save_data.metadata.optimal_hp_used = optimal_hp(1:2);
save_data.metadata.background_cond = background_cond;

save('results_approach_B.mat', '-struct', 'save_data', '-v7.3');
finfo = dir('results_approach_B.mat');
fprintf('  Saved: results_approach_B.mat (%.1f MB)\n', finfo.bytes / 1e6);

%% ========================================================================
% 9. GENERATE FIGURES
% =========================================================================
fprintf('\n=== Generating Figures ===\n');

% --- Figure B1: Training Curves (best combination) ---
figure('Name', 'Fig B1: Training Curves', ...
    'Position', [50, 400, 800, 350], 'Color', 'w');

plot(1:best.history.total_epochs, best.history.train_loss, 'b-', 'LineWidth', 1.5);
hold on;
plot(1:best.history.total_epochs, best.history.val_loss, 'r-', 'LineWidth', 1.5);
xline(best.history.best_epoch, 'k--', 'LineWidth', 1);
hold off;

xlabel('Epoch');
ylabel('Weighted MSE Loss');
title(sprintf('Approach B: %s + %s - Training Curves', best.prior, best.arch));
legend('Train', 'Validation', 'Best Epoch', 'Location', 'northeast');
grid on;

% --- Figure B2: Architecture Comparison (grouped by prior) ---
figure('Name', 'Fig B2: Architecture Search', ...
    'Position', [50, 300, 900, 400], 'Color', 'w');

bar_data = reshape(val_losses, n_archs, n_priors)';  % n_priors x n_archs
b = bar(bar_data, 'grouped');
prior_colors = [0.2 0.4 0.8; 0.8 0.2 0.2];
for a = 1:n_archs
    % Color doesn't vary by arch here, so just use default grouping
end
set(gca, 'XTickLabel', prior_names);
ylabel('Best Validation Loss');
title('Approach B: Architecture Search by Prior');
legend({archs.name}, 'Location', 'northeastoutside');
grid on;

% --- Figure B3: Prior Comparison (test set) ---
figure('Name', 'Fig B3: Prior Comparison', ...
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

sgtitle('Approach B: Does Reconstruction Prior Matter?');

% --- Figure B4: PCA Variance Explained ---
figure('Name', 'Fig B4: PCA Analysis', ...
    'Position', [50, 100, 800, 350], 'Color', 'w');

for p = 1:n_priors
    subplot(1, 2, p);
    plot(pca_models(p).cum_var(1:min(100, length(pca_models(p).cum_var))), ...
        '-', 'LineWidth', 1.5);
    hold on;
    yline(pca_var_threshold, 'r--', sprintf('%d%%', pca_var_threshold));
    xline(pca_models(p).K, 'k--', sprintf('K=%d', pca_models(p).K));
    hold off;
    xlabel('Number of Components');
    ylabel('Cumulative Variance (%)');
    title(sprintf('%s: K=%d for %d%%', prior_names{p}, ...
        pca_models(p).K, pca_var_threshold));
    grid on;
    xlim([1, min(100, length(pca_models(p).cum_var))]);
end

sgtitle('PCA Dimensionality Reduction');

%% ========================================================================
fprintf('\n============================================\n');
fprintf('  APPROACH B COMPLETE\n');
fprintf('  Best prior: %s\n', best.prior);
fprintf('  Best architecture: %s\n', best.arch);
fprintf('  PCA components: %d\n', pca_models(find(strcmp(prior_names, best.prior))).K);
fprintf('  Position error: %.2f cm\n', results_B.pos_mean);
fprintf('  Size error:     %.1f mm\n', results_B.size_abs_mean);
fprintf('  Contrast error: %.2f\n', results_B.contrast_mean);
fprintf('============================================\n');
fprintf('\nPCA models saved for Approach C.\n');
