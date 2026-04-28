function cfg = dataset_config()
% DATASET_CONFIG  Central configuration for training data generation
%
%   cfg = dataset_config()
%
%   Returns a structure containing all parameter ranges and settings
%   for the training dataset. Edit values here to modify the dataset
%   without changing the generation script.
%
%   Part of: EIT Breast Cancer ML Project - Phase 2
%   Date:    March 2026

    %% Geometry (must match build_breast_model.m)
    cfg.base_radius = 6.25;          % cm
    cfg.height = 4.0;                % cm
    cfg.parent_sphere_radius = (cfg.base_radius^2 + cfg.height^2) / (2 * cfg.height);
    cfg.parent_sphere_z_offset = cfg.height - cfg.parent_sphere_radius;
    cfg.background_conductivity = 0.3;  % S/m
    
    %% Tumour Size Range
    cfg.min_radius = 0.5;            % cm (10 mm diameter)
    cfg.max_radius = 1.5;            % cm (30 mm diameter)
    cfg.grid_radii = [0.5, 0.75, 1.0, 1.25, 1.5];  % For structured grid
    
    %% Conductivity Contrast Range
    cfg.min_contrast = 2.0;          % 2x background
    cfg.max_contrast = 5.0;          % 5x background
    cfg.grid_contrasts = [2.0, 3.0, 4.0, 5.0];      % For structured grid
    
    %% Position Sampling
    % Grid spacing for structured sampling (cm)
    cfg.grid_spacing = 1.5;          % Spacing between grid points
    cfg.boundary_margin = 0.2;       % cm, safety margin from boundary
    
    % Number of additional random samples
    cfg.n_random_samples = 1200;
    
    %% Noise Configuration
    % SNR levels in dB (Inf = no noise)
    cfg.snr_levels = [Inf, 80, 60, 40];
    cfg.noise_seed = 42;             % For reproducibility
    
    %% Dataset Split
    cfg.train_fraction = 0.70;
    cfg.val_fraction   = 0.15;
    cfg.test_fraction  = 0.15;
    cfg.split_seed = 123;            % For reproducible splits
    
    %% Output
    cfg.output_filename = 'training_dataset.mat';
    cfg.save_elem_data = true;       % Save full conductivity maps (large!)
    
    %% Display summary
    fprintf('=== Dataset Configuration ===\n');
    fprintf('  Tumour radius:   %.1f to %.1f cm (%.0f to %.0f mm diameter)\n', ...
        cfg.min_radius, cfg.max_radius, cfg.min_radius*20, cfg.max_radius*20);
    fprintf('  Contrast ratio:  %.1f to %.1f\n', cfg.min_contrast, cfg.max_contrast);
    fprintf('  Grid spacing:    %.1f cm\n', cfg.grid_spacing);
    fprintf('  Random samples:  %d\n', cfg.n_random_samples);
    fprintf('  Noise levels:    clean, %d dB, %d dB, %d dB\n', ...
        cfg.snr_levels(2), cfg.snr_levels(3), cfg.snr_levels(4));
    fprintf('  Split:           %.0f%% / %.0f%% / %.0f%%\n', ...
        cfg.train_fraction*100, cfg.val_fraction*100, cfg.test_fraction*100);
    fprintf('=============================\n\n');
end
