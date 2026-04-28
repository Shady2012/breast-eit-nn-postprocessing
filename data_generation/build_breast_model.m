function [fmdl, vh, elec_pos_centres] = build_breast_model()
% BUILD_BREAST_MODEL  Construct 3D spherical cap breast model for EIT
%
%   [fmdl, vh, elec_pos_centres] = build_breast_model()
%
%   This function builds a 3D finite element model of a female breast
%   represented as a spherical cap. The model includes 16 circular
%   surface electrodes arranged in 3 concentric rings and a stimulation
%   protocol producing 208 boundary voltage measurements.
%
%   OUTPUTS:
%       fmdl              - EIDORS forward model structure
%       vh                - Reference (homogeneous) voltage measurements
%       elec_pos_centres  - [16 x 3] matrix of electrode centre positions
%
%   GEOMETRY:
%       Base radius:    6.25 cm
%       Height:         4.0 cm
%       Electrode size: 1.5 cm diameter
%       Electrodes:     16 (4 inner + 6 middle + 6 outer ring)
%
%   PROTOCOL:
%       8 injection pairs, 26 measurements each = 208 total
%       Matches hardware acquisition protocol V5
%
%   USAGE:
%       [fmdl, vh, elec_centres] = build_breast_model();
%       figure; show_fem(fmdl); title('Breast Model');
%
%   Part of: EIT Breast Cancer ML Project - Phase 1
%   Date:    March 2026
    fprintf('=== Building 3D Breast Model ===\n');
    
    %% ----------------------------------------------------------------
    %  SECTION 1: PHYSICAL DIMENSIONS
    % -----------------------------------------------------------------
    base_radius = 6.25;    % cm
    height      = 4.0;     % cm
    
    % Mesh control parameters
    mesh_remote = 0.5;     % Bulk mesh density (coarser = faster)
    mesh_elec   = 0.15;    % Electrode region density (finer = more accurate)
    elec_diam   = 1.5;     % Electrode diameter in cm
    
    % Derive the parent sphere that creates our spherical cap
    % From the cap geometry: R = (r^2 + h^2) / (2h)
    parent_sphere_radius = (base_radius^2 + height^2) / (2 * height);
    parent_sphere_z_offset = height - parent_sphere_radius;
    
    fprintf('  Parent sphere: R = %.3f cm, z-offset = %.3f cm\n', ...
        parent_sphere_radius, parent_sphere_z_offset);
    
    %% ----------------------------------------------------------------
    %  SECTION 2: NETGEN GEOMETRY STRING
    % -----------------------------------------------------------------
    shape_str = sprintf([ ...
        'solid sph = sphere(0,0,%.4f;%.4f) -maxh=%.4f;\n' ...
        'solid mainobj = sph and plane(0,0,0; 0,0,-1);\n' ...
        ], parent_sphere_z_offset, parent_sphere_radius, mesh_remote);
    
    %% ----------------------------------------------------------------
    %  SECTION 3: ELECTRODE POSITIONS
    % -----------------------------------------------------------------
    % 16 electrodes in 3 concentric rings on the spherical cap surface
    %   Ring 1 (inner):  4 electrodes at radius ~1.5 cm
    %   Ring 2 (middle): 6 electrodes at radius ~3.5 cm
    %   Ring 3 (outer):  6 electrodes at radius ~5.7 cm
    
    xy_pos = [
        % Ring 1: Inner (4 electrodes, 90 deg spacing)
         0.0,  1.5;   1.5,  0.0;   0.0, -1.5;  -1.5,  0.0;
        % Ring 2: Middle (6 electrodes, 60 deg spacing)
         0.0,  3.5;   3.03,  1.75;  3.03, -1.75;
         0.0, -3.5;  -3.03, -1.75; -3.03,  1.75;
        % Ring 3: Outer (6 electrodes, 60 deg spacing)
         2.85,  4.94;  5.7,  0.0;   2.85, -4.94;
        -2.85, -4.94; -5.7,  0.0;  -2.85,  4.94
    ];
    
    nElecs = size(xy_pos, 1);
    elec_pos_full = zeros(nElecs, 6);  % [x, y, z, nx, ny, nz]
    
    % Map 2D electrode positions onto the 3D spherical cap surface
    for i = 1:nElecs
        x = xy_pos(i, 1);
        y = xy_pos(i, 2);
        
        % Find z-coordinate on the sphere surface
        z_sq = parent_sphere_radius^2 - (x^2 + y^2);
        if z_sq < 0
            error('Electrode %d at (%.2f, %.2f) lies outside the model!', i, x, y);
        end
        z = sqrt(z_sq) + parent_sphere_z_offset;
        
        % Outward normal vector (points away from sphere centre)
        nv = [x, y, z - parent_sphere_z_offset];
        unv = nv / norm(nv);
        
        elec_pos_full(i, :) = [x, y, z, unv];
    end
    
    elec_pos_centres = elec_pos_full(:, 1:3);
    
    %% ----------------------------------------------------------------
    %  SECTION 4: GENERATE MESH WITH LOCAL REFINEMENT
    % -----------------------------------------------------------------
    fprintf('  Generating mesh (this may take a moment)...\n');
    
    elec_shape_params = repmat([elec_diam/2, 0, mesh_elec], nElecs, 1);
    
    fmdl = ng_mk_gen_models(shape_str, elec_pos_full, ...
        elec_shape_params, ...
        repmat({'mainobj'}, nElecs, 1));
    
    % Set contact impedance for all electrodes
    for i = 1:nElecs
        fmdl.electrode(i).z_contact = 0.01;
    end
    
    fprintf('  Mesh: %d nodes, %d elements\n', ...
        size(fmdl.nodes, 1), size(fmdl.elems, 1));
    
    %% ----------------------------------------------------------------
    %  SECTION 5: STIMULATION PROTOCOL
    % -----------------------------------------------------------------
    % 8 injection pairs matching hardware protocol V5
    % Each pair generates 26 measurement combinations = 208 total
    
    fprintf('  Configuring stimulation protocol...\n');
    
    injection_pairs = [
        1, 3;   2, 4;          % Ring 1 opposites
        5, 8;   6, 9;  7, 10;  % Ring 2 opposites
        11, 14; 12, 15; 13, 16 % Ring 3 opposites
    ];
    
    stim = [];
    total_meas_count = 0;
    
    for i = 1:size(injection_pairs, 1)
        stim(i).stimulation = 'mA';
        
        % Current injection pattern
        inj_p = injection_pairs(i, 1);
        inj_m = injection_pairs(i, 2);
        stim(i).stim_pattern = sparse([inj_p; inj_m], 1, [1; -1], nElecs, 1);
        
        % Measurement selection (replicating V5 logic)
        meas_list = [];
        all_el = 1:16;
        rem_el = setdiff(all_el, [inj_p, inj_m]);
        n_rem = length(rem_el);
        target_count = 26;
        
        % Step 1: Adjacent fill
        for j = 1:n_rem
            vp = rem_el(j);
            vm = rem_el(mod(j, n_rem) + 1);
            meas_list = [meas_list; vp, vm];
            if size(meas_list, 1) >= target_count, break; end
        end
        
        % Step 2: Spaced fill (skip-1, skip-2, ...) to reach 26
        if size(meas_list, 1) < target_count
            needed = target_count - size(meas_list, 1);
            for spacing = 2:7
                for j = 1:n_rem
                    if needed <= 0, break; end
                    vp = rem_el(j);
                    vm_idx = mod(j + spacing - 1, n_rem) + 1;
                    vm = rem_el(vm_idx);
                    
                    % Check for duplicates
                    is_dup = false;
                    for k = 1:size(meas_list, 1)
                        if meas_list(k,1)==vp && meas_list(k,2)==vm
                            is_dup = true; break;
                        end
                    end
                    
                    if ~is_dup
                        meas_list = [meas_list; vp, vm];
                        needed = needed - 1;
                    end
                end
                if needed <= 0, break; end
            end
        end
        
        % Build measurement pattern matrix
        n_m = size(meas_list, 1);
        meas_mat = sparse(n_m, nElecs);
        for m = 1:n_m
            meas_mat(m, meas_list(m,1)) =  1;
            meas_mat(m, meas_list(m,2)) = -1;
        end
        stim(i).meas_pattern = meas_mat;
        total_meas_count = total_meas_count + n_m;
    end
    
    fmdl.stimulation = stim;
    
    % Verify measurement count
    if total_meas_count ~= 208
        error('Protocol error: expected 208 measurements, got %d', total_meas_count);
    end
    fprintf('  Protocol: %d injection pairs, %d measurements total\n', ...
        size(injection_pairs, 1), total_meas_count);
    
    %% ----------------------------------------------------------------
    %  SECTION 6: COMPUTE REFERENCE (HOMOGENEOUS) DATA
    % -----------------------------------------------------------------
    fprintf('  Computing reference voltages...\n');
    
    background_conductivity = 0.3;  % S/m
    img_homog = mk_image(fmdl, background_conductivity);
    vh_data = fwd_solve(img_homog);
    vh = vh_data.meas;
    
    fprintf('  Reference voltages: %d measurements computed\n', length(vh));
    fprintf('  Mean |vh| = %.4e V\n', mean(abs(vh)));
    fprintf('=== Model Build Complete ===\n\n');
end