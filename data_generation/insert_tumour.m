function [img, is_valid] = insert_tumour(fmdl, centre, radius, contrast_ratio, background_conductivity)
% INSERT_TUMOUR  Place a spherical tumour in the breast EIT model
%
%   [img, is_valid] = insert_tumour(fmdl, centre, radius, contrast_ratio, background_conductivity)
%
%   Creates an EIDORS image structure with a spherical inclusion
%   representing a tumour. The function validates that the tumour
%   fits inside the breast geometry before inserting it.
%
%   INPUTS:
%       fmdl                    - EIDORS forward model (from build_breast_model)
%       centre                  - [x, y, z] tumour centre in cm
%       radius                  - Tumour radius in cm (e.g. 0.5 for 10mm diameter)
%       contrast_ratio          - Conductivity ratio vs background (e.g. 3.0 = 3x)
%       background_conductivity - Background tissue conductivity in S/m (default: 0.3)
%
%   OUTPUTS:
%       img       - EIDORS image structure with tumour conductivity assigned
%       is_valid  - Logical flag: true if tumour fits inside geometry
%
%   EXAMPLES:
%       % 10mm diameter tumour at (2, 1, 2), 3x contrast
%       [img, valid] = insert_tumour(fmdl, [2, 1, 2], 0.5, 3.0, 0.3);
%
%       % 20mm diameter tumour near electrode 5, 4x contrast
%       [img, valid] = insert_tumour(fmdl, [0, 3, 2.5], 1.0, 4.0, 0.3);
%
%   Part of: EIT Breast Cancer ML Project - Phase 1
%   Date:    March 2026

    % Default background conductivity
    if nargin < 5
        background_conductivity = 0.3;
    end
    
    %% ----------------------------------------------------------------
    %  SECTION 1: GEOMETRY VALIDATION
    % -----------------------------------------------------------------
    % Reconstruct the spherical cap parameters to check containment
    base_radius = 6.25;
    height = 4.0;
    parent_sphere_radius = (base_radius^2 + height^2) / (2 * height);
    parent_sphere_z_offset = height - parent_sphere_radius;
    sphere_centre = [0, 0, parent_sphere_z_offset];
    
    % Safety margin: tumour should not touch the boundary
    boundary_margin = 0.2;  % 2mm margin
    
    % Check 1: Tumour centre must be above z = 0 (base plane)
    if centre(3) - radius < boundary_margin
        warning('Tumour extends below base plane (z < 0). Adjusting.');
        is_valid = false;
    else
        is_valid = true;
    end
    
    % Check 2: Tumour must fit within the spherical cap
    % The farthest point of the tumour from the sphere centre must
    % be less than the sphere radius
    dist_to_sphere_centre = norm(centre - sphere_centre);
    if dist_to_sphere_centre + radius > parent_sphere_radius - boundary_margin
        warning('Tumour extends outside breast surface. Centre: [%.2f, %.2f, %.2f], R: %.2f', ...
            centre(1), centre(2), centre(3), radius);
        is_valid = false;
    end
    
    % Check 3: Tumour centre must lie within the breast volume
    % (above base plane and below spherical surface)
    if centre(3) < 0 || norm(centre - sphere_centre) > parent_sphere_radius
        warning('Tumour centre is outside the breast volume.');
        is_valid = false;
    end
    
    %% ----------------------------------------------------------------
    %  SECTION 2: COMPUTE ELEMENT CENTROIDS
    % -----------------------------------------------------------------
    n_elems = size(fmdl.elems, 1);
    centroids = zeros(n_elems, 3);
    
    for i = 1:n_elems
        elem_node_indices = fmdl.elems(i, :);
        node_coords = fmdl.nodes(elem_node_indices, :);
        centroids(i, :) = mean(node_coords, 1);
    end
    
    %% ----------------------------------------------------------------
    %  SECTION 3: ASSIGN CONDUCTIVITY VALUES
    % -----------------------------------------------------------------
    % Start with homogeneous background
    img = mk_image(fmdl, background_conductivity);
    
    % Find elements whose centroids fall within the tumour sphere
    distances = sqrt(sum((centroids - centre).^2, 2));
    tumour_elements = distances < radius;
    n_tumour_elems = sum(tumour_elements);
    
    % Assign tumour conductivity
    tumour_conductivity = background_conductivity * contrast_ratio;
    img.elem_data(tumour_elements) = tumour_conductivity;
    
    %% ----------------------------------------------------------------
    %  SECTION 4: REPORT
    % -----------------------------------------------------------------
    if n_tumour_elems == 0
        warning('No mesh elements captured by tumour at [%.2f, %.2f, %.2f] with radius %.2f cm.', ...
            centre(1), centre(2), centre(3), radius);
        warning('The tumour may be too small for the current mesh density.');
        is_valid = false;
    else
        fprintf('  Tumour inserted: centre=[%.2f, %.2f, %.2f], diameter=%.1fmm, contrast=%.1fx\n', ...
            centre(1), centre(2), centre(3), radius*20, contrast_ratio);
        fprintf('  Elements affected: %d / %d (%.1f%% of mesh)\n', ...
            n_tumour_elems, n_elems, 100*n_tumour_elems/n_elems);
    end
end
