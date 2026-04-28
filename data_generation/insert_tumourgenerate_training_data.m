function dataset = generate_training_data()
% GENERATE_TRAINING_DATA  Create full training dataset for EIT-ML project
%
%   dataset = generate_training_data()
%
%   Generates a comprehensive dataset of simulated EIT measurements
%   for training neural networks. Combines structured grid sampling
%   with random sampling, injects noise at multiple SNR levels, and
%   saves everything in a single .mat file.
%
%   OUTPUT:
%       dataset - Structure containing all data and metadata
%                 Also saved to disk as 'training_dataset.mat'
%
%   REQUIRES:
%       build_breast_model.m, insert_tumour.m, dataset_config.m
%       Phase 1 must be verified (5/5 tests passed)
%
%   ESTIMATED TIME: 2-3 hours for ~2500 configurations
%
%   Part of: EIT Breast Cancer ML Project - Phase 2
%   Date:    March 2026

    total_timer = tic;
    
    %% ================================================================
    %  STEP 1: LOAD CONFIGURATION AND BUILD MODEL
    % =================================================================
    fprintf('============================================\n');
    fprintf('  TRAINING DATA GENERATION\n');
    fprintf('============================================\n\n');
    
    cfg = dataset_config();
    
    fprintf('Building forward model...\n');
    [fmdl, vh_vec, elec_pos_centres] = build_breast_model();
    
    n_elems = size(fmdl.elems, 1);
    n_meas = length(vh_vec);
    
    % We need the vh as a struct for fwd_solve comparison
    img_homog = mk_image(fmdl, cfg.background_conductivity);
    vh_struct = fwd_solve(img_homog);
    
    %% ================================================================
    %  STEP 2: GENERATE VALID TUMOUR POSITIONS (STRUCTURED GRID)
    % =================================================================
    fprintf('Generating structured grid positions...\n');
    
    grid_configs = [];  % Will store [x, y, z, radius, contrast]
    
    % Create 3D grid inside the breast volume
    x_range = -cfg.base_radius + 1 : cfg.grid_spacing : cfg.base_radius - 1;
    y_range = -cfg.base_radius + 1 : cfg.grid_spacing : cfg.base_radius - 1;
    z_range = 0.5 : cfg.grid_spacing : cfg.height - 0.3;
    
    sphere_centre = [0, 0, cfg.parent_sphere_z_offset];
    
    for xi = 1:length(x_range)
        for yi = 1:length(y_range)
            for zi = 1:length(z_range)
                pos = [x_range(xi), y_range(yi), z_range(zi)];
                
                % For each grid position, try all size/contrast combinations
                for ri = 1:length(cfg.grid_radii)
                    for ci = 1:length(cfg.grid_contrasts)
                        r = cfg.grid_radii(ri);
                        c = cfg.grid_contrasts(ci);
                        
                        % Check if this tumour fits inside the breast
                        if is_valid_tumour(pos, r, cfg)
                            grid_configs = [grid_configs; ...
                                pos, r, c];
                        end
                    end
                end
            end
        end
    end
    
    n_grid = size(grid_configs, 1);
    fprintf('  Valid grid configurations: %d\n', n_grid);
    
    %% ================================================================
    %  STEP 3: GENERATE RANDOM TUMOUR POSITIONS
    % =================================================================
    fprintf('Generating random sample positions...\n');
    
    rng(cfg.split_seed + 1);  % Reproducible randomness
    random_configs = [];
    attempts = 0;
    max_attempts = cfg.n_random_samples * 10;  % Prevent infinite loop
    
    while size(random_configs, 1) < cfg.n_random_samples && attempts < max_attempts
        attempts = attempts + 1;
        
        % Random position within a bounding box
        x = (2 * rand - 1) * (cfg.base_radius - 1);
        y = (2 * rand - 1) * (cfg.base_radius - 1);
        z = rand * (cfg.height - 0.5) + 0.3;
        pos = [x, y, z];
        
        % Random size (continuous within range)
        r = cfg.min_radius + rand * (cfg.max_radius - cfg.min_radius);
        
        % Random contrast (continuous within range)
        c = cfg.min_contrast + rand * (cfg.max_contrast - cfg.min_contrast);
        
        % Validate
        if is_valid_tumour(pos, r, cfg)
            random_configs = [random_configs; pos, r, c];
        end
    end
    
    n_random = size(random_configs, 1);
    fprintf('  Valid random configurations: %d (from %d attempts)\n', ...
        n_random, attempts);
    
    %% ================================================================
    %  STEP 4: COMBINE ALL CONFIGURATIONS
    % =================================================================
    all_configs = [grid_configs; random_configs];
    n_total_clean = size(all_configs, 1);
    
    fprintf('\n  Total unique configurations: %d\n', n_total_clean);
    fprintf('    Grid-based:  %d (%.0f%%)\n', n_grid, 100*n_grid/n_total_clean);
    fprintf('    Random:      %d (%.0f%%)\n', n_random, 100*n_random/n_total_clean);
    
    %% ================================================================
    %  STEP 5: RUN FORWARD SOLVES
    % =================================================================
    fprintf('\n--- Running Forward Solves ---\n');
    fprintf('  This will take approximately %.0f to %.0f minutes...\n', ...
        n_total_clean * 2 / 60, n_total_clean * 5 / 60);
    
    % Pre-allocate storage for clean data
    measurements_clean = zeros(n_meas, n_total_clean);
    elem_data_all = zeros(n_elems, n_total_clean);
    nearest_electrode = zeros(n_total_clean, 1);
    electrode_distance = zeros(n_total_clean, 1);
    valid_flags = true(n_total_clean, 1);
    
    solve_timer = tic;
    
    for i = 1:n_total_clean
        % Extract parameters
        pos = all_configs(i, 1:3);
        r   = all_configs(i, 4);
        c   = all_configs(i, 5);
        
        % Insert tumour (suppress output for speed)
        [img, is_valid] = insert_tumour(fmdl, pos, r, c, cfg.background_conductivity);
        
        if ~is_valid
            valid_flags(i) = false;
            continue;
        end
        
        % Forward solve
        vi = fwd_solve(img);
        
        % Store difference measurements
        measurements_clean(:, i) = vi.meas - vh_vec;
        
        % Store ground truth conductivity map
        if cfg.save_elem_data
            elem_data_all(:, i) = img.elem_data;
        end
        
        % Compute nearest electrode
        distances_to_elecs = sqrt(sum((elec_pos_centres - pos).^2, 2));
        [min_dist, min_idx] = min(distances_to_elecs);
        nearest_electrode(i) = min_idx;
        electrode_distance(i) = min_dist;
        
        % Progress reporting
        if mod(i, 50) == 0 || i == n_total_clean
            elapsed = toc(solve_timer);
            rate = i / elapsed;
            remaining = (n_total_clean - i) / rate;
            fprintf('  [%d/%d] %.1f samples/sec | Elapsed: %.1f min | Remaining: ~%.1f min\n', ...
                i, n_total_clean, rate, elapsed/60, remaining/60);
        end
    end
    
    % Remove invalid configurations
    valid_idx = find(valid_flags);
    n_valid = length(valid_idx);
    
    if n_valid < n_total_clean
        fprintf('\n  Removed %d invalid configurations. Keeping %d.\n', ...
            n_total_clean - n_valid, n_valid);
        
        all_configs = all_configs(valid_idx, :);
        measurements_clean = measurements_clean(:, valid_idx);
        elem_data_all = elem_data_all(:, valid_idx);
        nearest_electrode = nearest_electrode(valid_idx);
        electrode_distance = electrode_distance(valid_idx);
    end
    
    fprintf('\nForward solves complete. %d valid samples.\n', n_valid);
    
    %% ================================================================
    %  STEP 6: INJECT NOISE
    % =================================================================
    fprintf('\n--- Injecting Noise ---\n');
    
    n_snr = length(cfg.snr_levels);
    n_total_noisy = n_valid * n_snr;
    
    fprintf('  Creating %d noisy copies (%d clean x %d noise levels)\n', ...
        n_total_noisy, n_valid, n_snr);
    
    % Pre-allocate
    measurements_all = zeros(n_meas, n_total_noisy);
    params_all = zeros(n_total_noisy, 5);  % [x, y, z, diameter_mm, contrast]
    noise_level_all = zeros(n_total_noisy, 1);
    nearest_elec_all = zeros(n_total_noisy, 1);
    elec_dist_all = zeros(n_total_noisy, 1);
    clean_index_all = zeros(n_total_noisy, 1);  % Maps back to clean sample
    
    rng(cfg.noise_seed);  % Reproducible noise
    
    idx = 0;
    for i = 1:n_valid
        dv_clean = measurements_clean(:, i);
        signal_norm = norm(dv_clean);
        
        for s = 1:n_snr
            idx = idx + 1;
            snr = cfg.snr_levels(s);
            
            if isinf(snr)
                % Clean copy (no noise)
                measurements_all(:, idx) = dv_clean;
            else
                % Add Gaussian noise at specified SNR
                noise_power = signal_norm / (10^(snr / 20));
                noise = randn(n_meas, 1) * (noise_power / sqrt(n_meas));
                measurements_all(:, idx) = dv_clean + noise;
            end
            
            % Store parameters
            % Convert radius to diameter in mm for convenience
            diameter_mm = all_configs(i, 4) * 20;  % radius(cm) * 2 * 10(mm/cm)
            params_all(idx, :) = [all_configs(i, 1:3), diameter_mm, all_configs(i, 5)];
            noise_level_all(idx) = snr;
            nearest_elec_all(idx) = nearest_electrode(i);
            elec_dist_all(idx) = electrode_distance(i);
            clean_index_all(idx) = i;
        end
    end
    
    fprintf('  Total samples with noise: %d\n', idx);
    
    %% ================================================================
    %  STEP 7: SPLIT INTO TRAIN / VALIDATION / TEST
    % =================================================================
    fprintf('\n--- Splitting Dataset ---\n');
    
    % Split on UNIQUE configurations (all noise versions go together)
    rng(cfg.split_seed);
    perm = randperm(n_valid);
    
    n_train = round(cfg.train_fraction * n_valid);
    n_val   = round(cfg.val_fraction * n_valid);
    n_test  = n_valid - n_train - n_val;
    
    % Indices into the CLEAN sample array
    clean_train_idx = sort(perm(1:n_train));
    clean_val_idx   = sort(perm(n_train+1 : n_train+n_val));
    clean_test_idx  = sort(perm(n_train+n_val+1 : end));
    
    % Expand to include all noise versions of each sample
    % Each clean sample i maps to noisy indices: (i-1)*n_snr+1 : i*n_snr
    expand = @(clean_idx) reshape( ...
        ((clean_idx(:)-1)*n_snr)' + (1:n_snr)', 1, []);
    
    train_idx = sort(expand(clean_train_idx));
    val_idx   = sort(expand(clean_val_idx));
    test_idx  = sort(expand(clean_test_idx));
    
    fprintf('  Unique configs:  Train=%d, Val=%d, Test=%d\n', ...
        n_train, n_val, n_test);
    fprintf('  With noise:      Train=%d, Val=%d, Test=%d\n', ...
        length(train_idx), length(val_idx), length(test_idx));
    
    % Verify no overlap
    assert(isempty(intersect(train_idx, val_idx)), 'Train/Val overlap!');
    assert(isempty(intersect(train_idx, test_idx)), 'Train/Test overlap!');
    assert(isempty(intersect(val_idx, test_idx)),   'Val/Test overlap!');
    assert(length(train_idx) + length(val_idx) + length(test_idx) == n_total_noisy, ...
        'Split size mismatch!');
    
    fprintf('  No data leakage: VERIFIED\n');
    
    %% ================================================================
    %  STEP 8: BUILD OUTPUT STRUCTURE AND SAVE
    % =================================================================
    fprintf('\n--- Saving Dataset ---\n');
    
    % Build the dataset structure
    dataset.measurements      = measurements_all;         % [208 x N_total]
    dataset.measurements_clean = measurements_clean;      % [208 x N_clean]
    dataset.tumour_params     = params_all;               % [N_total x 5]
    dataset.noise_levels      = noise_level_all;          % [N_total x 1]
    dataset.nearest_electrode = nearest_elec_all;         % [N_total x 1]
    dataset.electrode_distance = elec_dist_all;           % [N_total x 1]
    dataset.clean_index       = clean_index_all;          % [N_total x 1]
    
    if cfg.save_elem_data
        dataset.elem_data = elem_data_all;                % [n_elems x N_clean]
    end
    
    % Split indices
    dataset.split.train = train_idx;
    dataset.split.val   = val_idx;
    dataset.split.test  = test_idx;
    dataset.split.clean_train = clean_train_idx;
    dataset.split.clean_val   = clean_val_idx;
    dataset.split.clean_test  = clean_test_idx;
    
    % Metadata
    dataset.info.date_generated    = datestr(now);
    dataset.info.n_clean_samples   = n_valid;
    dataset.info.n_total_samples   = n_total_noisy;
    dataset.info.n_measurements    = n_meas;
    dataset.info.n_elements        = n_elems;
    dataset.info.background_cond   = cfg.background_conductivity;
    dataset.info.snr_levels        = cfg.snr_levels;
    dataset.info.tumour_radii_range = [cfg.min_radius, cfg.max_radius];
    dataset.info.contrast_range    = [cfg.min_contrast, cfg.max_contrast];
    dataset.info.param_columns     = {'x_cm', 'y_cm', 'z_cm', 'diameter_mm', 'contrast_ratio'};
    dataset.info.electrode_positions = elec_pos_centres;
    
    % Save to disk
    fprintf('  Saving to %s...\n', cfg.output_filename);
    save(cfg.output_filename, '-struct', 'dataset', '-v7.3');
    
    file_info = dir(cfg.output_filename);
    fprintf('  File size: %.1f MB\n', file_info.bytes / 1e6);
    
    %% ================================================================
    %  STEP 9: PRINT SUMMARY STATISTICS
    % =================================================================
    total_time = toc(total_timer);
    
    fprintf('\n============================================\n');
    fprintf('  DATASET GENERATION COMPLETE\n');
    fprintf('============================================\n');
    fprintf('  Clean samples:     %d\n', n_valid);
    fprintf('  With noise:        %d (x%d noise levels)\n', n_total_noisy, n_snr);
    fprintf('  Training set:      %d samples (%d unique)\n', ...
        length(train_idx), n_train);
    fprintf('  Validation set:    %d samples (%d unique)\n', ...
        length(val_idx), n_val);
    fprintf('  Test set:          %d samples (%d unique)\n', ...
        length(test_idx), n_test);
    fprintf('  Data leakage:      None (verified)\n');
    fprintf('  Total time:        %.1f minutes\n', total_time / 60);
    fprintf('  Saved to:          %s\n', cfg.output_filename);
    fprintf('============================================\n\n');
    
    %% ================================================================
    %  STEP 10: QUICK VISUALISATION
    % =================================================================
    fprintf('Generating dataset overview plots...\n');
    
    % Plot 1: Tumour position distribution
    clean_params = params_all(1:n_snr:end, :);  % Take only clean copies
    
    figure('Name', 'Dataset Overview', 'Position', [100, 100, 1400, 500]);
    
    subplot(1, 3, 1);
    scatter3(clean_params(:,1), clean_params(:,2), clean_params(:,3), ...
        clean_params(:,4)*2, clean_params(:,5), 'filled', 'MarkerFaceAlpha', 0.5);
    colorbar; colormap(jet);
    xlabel('X (cm)'); ylabel('Y (cm)'); zlabel('Z (cm)');
    title('Tumour Positions (colour=contrast, size=diameter)');
    axis equal; grid on; view(3);
    
    % Plot 2: Parameter distributions
    subplot(1, 3, 2);
    histogram(clean_params(:,4), 20);
    xlabel('Diameter (mm)'); ylabel('Count');
    title('Size Distribution');
    
    subplot(1, 3, 3);
    histogram(clean_params(:,5), 20);
    xlabel('Contrast Ratio'); ylabel('Count');
    title('Contrast Distribution');
    
    % Plot 3: Signal strength distribution
    figure('Name', 'Signal Statistics', 'Position', [100, 650, 800, 400]);
    
    signal_norms = sqrt(sum(measurements_clean.^2, 1));
    histogram(signal_norms, 30);
    xlabel('Signal Energy (L2 norm of dv)');
    ylabel('Count');
    title('Distribution of Signal Strength Across All Configurations');
    xline(mean(signal_norms), 'r--', sprintf('Mean: %.4f', mean(signal_norms)), 'LineWidth', 2);
    
    fprintf('Done. Review plots before proceeding to Phase 3.\n');
end


%% ====================================================================
%  LOCAL HELPER FUNCTION
% =====================================================================

function valid = is_valid_tumour(centre, radius, cfg)
% IS_VALID_TUMOUR  Check if a tumour fits inside the breast volume
%
%   Checks:
%   1. Tumour stays above the base plane (z > 0) with margin
%   2. Tumour stays inside the spherical cap surface with margin
%   3. Tumour centre is inside the breast volume

    margin = cfg.boundary_margin;
    sphere_centre = [0, 0, cfg.parent_sphere_z_offset];
    
    % Check 1: Above base plane
    if centre(3) - radius < margin
        valid = false;
        return;
    end
    
    % Check 2: Inside spherical surface
    dist_to_sphere_centre = norm(centre - sphere_centre);
    if dist_to_sphere_centre + radius > cfg.parent_sphere_radius - margin
        valid = false;
        return;
    end
    
    % Check 3: Centre is inside the volume
    if centre(3) < 0 || dist_to_sphere_centre > cfg.parent_sphere_radius
        valid = false;
        return;
    end
    
    valid = true;
end
