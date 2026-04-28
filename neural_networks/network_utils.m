function utils = network_utils()
% NETWORK_UTILS  Shared utilities for Phase 4 neural network training.
%
% Returns a struct of function handles used by all three approach scripts.
%
% Usage:
%   utils = network_utils();
%   net   = utils.build_network(208, [256, 128, 64], 5, 0.2);
%   cfg   = utils.default_config();
%   [net, history] = utils.train_network(net, X_tr, Y_tr, X_val, Y_val, cfg);
%   results = utils.evaluate(pred_params, true_params, noise_levels);
%   utils.print_summary(results, 'Approach A');
%
% Part of: Phase 4 - Neural Network Development
% Project: ML-Enhanced EIT for Early Breast Cancer Detection

    utils.build_network   = @build_network;
    utils.default_config  = @default_config;
    utils.train_network   = @train_network;
    utils.evaluate        = @evaluate_predictions;
    utils.print_summary   = @print_summary;
    utils.print_breakdown = @print_breakdown;
    utils.architectures   = @get_architectures;
    utils.find_file       = @find_file;
end


% =====================================================================
% FILE FINDER (robust path search, same pattern as Phase 3)
% =====================================================================
function fpath = find_file(filename, search_dirs)
% FIND_FILE  Search for a file in multiple directories.
%   fpath = find_file('training_dataset.mat', {'../dataset', '.'})

    for i = 1:length(search_dirs)
        candidate = fullfile(search_dirs{i}, filename);
        if exist(candidate, 'file')
            fpath = candidate;
            return;
        end
    end
    error('Cannot find %s. Searched: %s', filename, strjoin(search_dirs, ', '));
end


% =====================================================================
% ARCHITECTURE DEFINITIONS
% =====================================================================
function archs = get_architectures()
% GET_ARCHITECTURES  Return the four architecture candidates.

    archs = struct();
    archs(1).name   = 'Shallow';
    archs(1).layers = [128, 64];

    archs(2).name   = 'Medium';
    archs(2).layers = [256, 128, 64];

    archs(3).name   = 'Deep';
    archs(3).layers = [512, 256, 128, 64];

    archs(4).name   = 'Wide';
    archs(4).layers = [512, 512];
end


% =====================================================================
% DEFAULT TRAINING CONFIGURATION
% =====================================================================
function cfg = default_config()
% DEFAULT_CONFIG  Return the locked training hyperparameters.

    cfg.lr              = 1e-3;     % Initial learning rate
    cfg.max_epochs      = 300;      % Maximum epochs
    cfg.patience        = 30;       % Early stopping patience
    cfg.batch_size      = 64;       % Mini-batch size
    cfg.lr_drop_factor  = 0.5;     % LR multiplier at each drop
    cfg.lr_drop_period  = 50;       % Epochs between LR drops
    cfg.l2_lambda       = 1e-4;     % L2 weight decay
    cfg.dropout_rate    = 0.2;      % Dropout probability
    cfg.loss_weights    = [0.50, 0.30, 0.20];  % position, size, contrast
end


% =====================================================================
% BUILD NETWORK
% =====================================================================
function net = build_network(input_size, hidden_sizes, output_size, dropout_rate)
% BUILD_NETWORK  Construct a dlnetwork with ReLU and dropout.
%
%   net = build_network(208, [256, 128, 64], 5, 0.2)
%
% Architecture: input -> [FC -> ReLU -> Dropout] x N -> FC(output)

    layers = featureInputLayer(input_size, 'Name', 'input', ...
        'Normalization', 'none');

    for i = 1:length(hidden_sizes)
        h = hidden_sizes(i);
        layers = [layers
            fullyConnectedLayer(h, 'Name', sprintf('fc%d', i))
            reluLayer('Name', sprintf('relu%d', i))
            dropoutLayer(dropout_rate, 'Name', sprintf('drop%d', i))
        ]; %#ok<AGROW>
    end

    layers = [layers
        fullyConnectedLayer(output_size, 'Name', 'output')
    ];

    net = dlnetwork(layers);
end


% =====================================================================
% TRAIN NETWORK (custom training loop with Adam)
% =====================================================================
function [trained_net, history] = train_network(net, X_train, Y_train, ...
        X_val, Y_val, config)
% TRAIN_NETWORK  Train a dlnetwork with Adam, LR scheduling, early stopping.
%
%   [net, history] = train_network(net, X_tr, Y_tr, X_val, Y_val, cfg)
%
%   X_train: [n_features x n_train] double
%   Y_train: [5 x n_train] double (normalised outputs)
%   X_val:   [n_features x n_val] double
%   Y_val:   [5 x n_val] double (normalised outputs)
%   config:  struct from default_config()
%
%   Returns trained_net (best validation) and history struct.

    n_train  = size(X_train, 2);
    n_batches = ceil(n_train / config.batch_size);

    % Preallocate history
    history.train_loss = zeros(config.max_epochs, 1);
    history.val_loss   = zeros(config.max_epochs, 1);

    % Adam state
    avg_grad    = [];
    avg_sq_grad = [];
    iteration   = 0;

    % Early stopping state
    best_val_loss    = Inf;
    best_epoch       = 0;
    best_net         = net;
    patience_counter = 0;

    lr = config.lr;

    for epoch = 1:config.max_epochs

        % --- Shuffle training data ---
        perm = randperm(n_train);
        X_shuf = X_train(:, perm);
        Y_shuf = Y_train(:, perm);

        epoch_loss = 0;

        for b = 1:n_batches
            iteration = iteration + 1;

            idx_s = (b - 1) * config.batch_size + 1;
            idx_e = min(b * config.batch_size, n_train);

            X_b = dlarray(X_shuf(:, idx_s:idx_e), 'CB');
            Y_b = dlarray(Y_shuf(:, idx_s:idx_e), 'CB');

            % Compute gradients (forward with dropout)
            [loss_val, grads] = dlfeval(@model_loss, net, X_b, Y_b, ...
                config.loss_weights, config.l2_lambda);

            % Adam update
            [net, avg_grad, avg_sq_grad] = adamupdate(net, grads, ...
                avg_grad, avg_sq_grad, iteration, lr);

            epoch_loss = epoch_loss + double(extractdata(loss_val));
        end

        history.train_loss(epoch) = epoch_loss / n_batches;

        % --- Validation loss (predict mode: no dropout, no L2) ---
        Y_val_pred = predict(net, dlarray(X_val, 'CB'));
        val_diff   = double(extractdata(Y_val_pred)) - Y_val;

        v_pos  = mean(val_diff(1:3, :).^2, 'all');
        v_size = mean(val_diff(4, :).^2, 'all');
        v_con  = mean(val_diff(5, :).^2, 'all');
        val_loss = config.loss_weights(1) * v_pos + ...
                   config.loss_weights(2) * v_size + ...
                   config.loss_weights(3) * v_con;

        history.val_loss(epoch) = val_loss;

        % --- Early stopping ---
        if val_loss < best_val_loss
            best_val_loss    = val_loss;
            best_epoch       = epoch;
            best_net         = net;
            patience_counter = 0;
        else
            patience_counter = patience_counter + 1;
        end

        if patience_counter >= config.patience
            fprintf('    Early stopping at epoch %d (best: %d, val=%.4f)\n', ...
                epoch, best_epoch, best_val_loss);
            break;
        end

        % --- Learning rate schedule ---
        if mod(epoch, config.lr_drop_period) == 0
            lr = lr * config.lr_drop_factor;
        end

        % --- Progress ---
        if mod(epoch, 25) == 0 || epoch == 1
            fprintf('    Epoch %3d: train=%.4f  val=%.4f  lr=%.1e\n', ...
                epoch, history.train_loss(epoch), val_loss, lr);
        end
    end

    % Trim history to actual epochs
    actual = min(epoch, config.max_epochs);
    history.train_loss   = history.train_loss(1:actual);
    history.val_loss     = history.val_loss(1:actual);
    history.best_epoch   = best_epoch;
    history.best_val_loss = best_val_loss;
    history.final_lr     = lr;
    history.total_epochs = actual;

    trained_net = best_net;
end


% =====================================================================
% MODEL LOSS (called inside dlfeval by train_network)
% =====================================================================
function [loss, grads] = model_loss(net, X, Y, loss_weights, l2_lambda)
% MODEL_LOSS  Weighted MSE with L2 regularisation (differentiable).
%
%   Outputs 1-3: position (x, y, z)   -> weight = loss_weights(1)
%   Output  4:   diameter              -> weight = loss_weights(2)
%   Output  5:   contrast              -> weight = loss_weights(3)

    Y_pred = forward(net, X);
    diff   = Y_pred - Y;

    pos_loss      = mean(diff(1:3, :).^2, 'all');
    size_loss     = mean(diff(4, :).^2, 'all');
    contrast_loss = mean(diff(5, :).^2, 'all');

    data_loss = loss_weights(1) * pos_loss + ...
                loss_weights(2) * size_loss + ...
                loss_weights(3) * contrast_loss;

    % L2 penalty on fully connected weights (not biases)
    reg_loss = 0;
    learnables = net.Learnables;
    for i = 1:height(learnables)
        if strcmp(learnables.Parameter{i}, 'Weights')
            reg_loss = reg_loss + sum(learnables.Value{i}.^2, 'all');
        end
    end

    loss  = data_loss + l2_lambda * reg_loss;
    grads = dlgradient(loss, net.Learnables);
end


% =====================================================================
% EVALUATE PREDICTIONS
% =====================================================================
function results = evaluate_predictions(pred_params, true_params, noise_levels)
% EVALUATE_PREDICTIONS  Compute all error metrics (matches Phase 3 format).
%
%   pred_params:  [n_test x 5]  predicted [x, y, z, diam_mm, contrast]
%   true_params:  [n_test x 5]  ground truth
%   noise_levels: [n_test x 1]  SNR labels (Inf, 80, 60, 40)

    % --- Per-sample errors ---
    pos_err      = sqrt(sum((pred_params(:,1:3) - true_params(:,1:3)).^2, 2));
    size_err_abs = abs(pred_params(:,4) - true_params(:,4));
    size_err_rel = size_err_abs ./ true_params(:,4) * 100;
    contrast_err = abs(pred_params(:,5) - true_params(:,5));

    results.position_err = pos_err;
    results.size_err_abs = size_err_abs;
    results.size_err_rel = size_err_rel;
    results.contrast_err = contrast_err;

    % --- Overall summary ---
    results.pos_mean      = mean(pos_err);
    results.pos_median    = median(pos_err);
    results.pos_std       = std(pos_err);
    results.size_abs_mean = mean(size_err_abs);
    results.size_rel_mean = mean(size_err_rel);
    results.contrast_mean = mean(contrast_err);

    % --- Breakdown by noise level ---
    noise_values = [Inf, 80, 60, 40];
    results.pos_by_noise  = NaN(1, 4);
    results.size_by_noise = NaN(1, 4);
    for i = 1:length(noise_values)
        mask = noise_levels == noise_values(i);
        if any(mask)
            results.pos_by_noise(i)  = mean(pos_err(mask));
            results.size_by_noise(i) = mean(size_err_abs(mask));
        end
    end

    % --- Breakdown by tumour size band ---
    size_bands = [10 15; 15 20; 20 30];
    results.pos_by_size  = NaN(1, 3);
    results.size_by_size = NaN(1, 3);
    for i = 1:size(size_bands, 1)
        mask = true_params(:,4) >= size_bands(i,1) & ...
               true_params(:,4) <  size_bands(i,2);
        if any(mask)
            results.pos_by_size(i)  = mean(pos_err(mask));
            results.size_by_size(i) = mean(size_err_abs(mask));
        end
    end
end


% =====================================================================
% PRINT SUMMARY (mirrors Phase 3 output format)
% =====================================================================
function print_summary(results, name)
% PRINT_SUMMARY  Display overall performance metrics.

    fprintf('\n  %s:\n', name);
    fprintf('    Position error:  %.2f cm (median: %.2f, std: %.2f)\n', ...
        results.pos_mean, results.pos_median, results.pos_std);
    fprintf('    Size error:      %.1f mm (%.1f%%)\n', ...
        results.size_abs_mean, results.size_rel_mean);
    fprintf('    Contrast error:  %.2f\n', results.contrast_mean);
end


% =====================================================================
% PRINT BREAKDOWN TABLES
% =====================================================================
function print_breakdown(results, name)
% PRINT_BREAKDOWN  Display noise and size breakdowns.

    noise_labels = {'Clean', '80dB', '60dB', '40dB'};
    size_labels  = {'10-15mm', '15-20mm', '20-30mm'};

    fprintf('\n  %s - Position Error by Noise Level:\n', name);
    fprintf('    ');
    for i = 1:4
        fprintf('%8s', noise_labels{i});
    end
    fprintf('\n    ');
    for i = 1:4
        fprintf('%8.2f', results.pos_by_noise(i));
    end
    fprintf(' cm\n');

    fprintf('\n  %s - Position Error by Tumour Size:\n', name);
    fprintf('    ');
    for i = 1:3
        fprintf('%10s', size_labels{i});
    end
    fprintf('\n    ');
    for i = 1:3
        fprintf('%10.2f', results.pos_by_size(i));
    end
    fprintf(' cm\n');
end
