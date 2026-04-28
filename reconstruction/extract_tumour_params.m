function result = extract_tumour_params(elem_data, elem_centres, elem_volumes, threshold_method, background_cond)
% EXTRACT_TUMOUR_PARAMS  Extract tumour parameters from a reconstructed EIT image.
%
% This function takes the output of an EIDORS difference reconstruction
% and estimates the tumour's position, diameter, and conductivity contrast.
% It identifies the tumour region by thresholding the reconstructed
% conductivity change distribution, then computes geometric and physical
% parameters from the thresholded region.
%
% Inputs:
%   elem_data        - [n_elements x 1] reconstructed conductivity change values
%   elem_centres     - [n_elements x 3] XYZ coordinates of element centroids (cm)
%   elem_volumes     - [n_elements x 1] element volumes (cm^3)
%   threshold_method - 'fixed50' or 'otsu'
%   background_cond  - background conductivity in S/m (default: 0.3)
%
% Output:
%   result - struct with fields:
%     .position      [1x3]   Estimated tumour centre in cm (NaN if not detected)
%     .diameter_mm   [1x1]   Estimated tumour diameter in mm (NaN if not detected)
%     .contrast_est  [1x1]   Estimated contrast ratio (NaN if not detected)
%     .detected      logical True if a tumour region was identified
%     .n_elements    [1x1]   Number of elements in detected region
%     .threshold     [1x1]   The threshold value that was applied
%
% Usage:
%   result = extract_tumour_params(img.elem_data, centres, vols, 'fixed50', 0.3);
%
% Part of: Phase 3 - Baseline Reconstruction
% Project: ML-Enhanced EIT for Early Breast Cancer Detection

    % Default background conductivity
    if nargin < 5 || isempty(background_cond)
        background_cond = 0.3;
    end

    % Ensure column vector
    elem_data = elem_data(:);
    n_elems = length(elem_data);

    % Work with absolute values for thresholding.
    % Difference EIT reconstructs conductivity CHANGES. For conductive
    % tumours (contrast > 1), the change is positive. However,
    % reconstruction artefacts can produce negative values elsewhere.
    % Using absolute values ensures we capture the strongest anomaly
    % regardless of sign.
    abs_data = abs(elem_data);

    % =====================================================================
    % STEP 1: Compute threshold
    % =====================================================================
    switch lower(threshold_method)
        case 'fixed50'
            % Fixed 50% of peak: widely used in EIT literature.
            % Simple and deterministic, but sensitive to artefacts
            % (a strong artefact anywhere raises the threshold).
            peak_val = max(abs_data);
            threshold = 0.50 * peak_val;

        case 'otsu'
            % Otsu's method: automatically selects a threshold that
            % maximises the between-class variance. Adaptive to each
            % individual reconstruction. Assumes a bimodal distribution
            % (background vs tumour), which may not hold for noisy or
            % heavily smoothed reconstructions.
            threshold = otsu_threshold(abs_data);

        otherwise
            error('Unknown threshold method: %s. Use ''fixed50'' or ''otsu''.', ...
                threshold_method);
    end

    % =====================================================================
    % STEP 2: Identify tumour elements
    % =====================================================================
    tumour_mask = abs_data > threshold;
    tumour_idx = find(tumour_mask);
    n_tumour = length(tumour_idx);

    % =====================================================================
    % STEP 3: Handle detection failure
    % =====================================================================
    if n_tumour == 0
        result.position     = [NaN, NaN, NaN];
        result.diameter_mm  = NaN;
        result.contrast_est = NaN;
        result.detected     = false;
        result.n_elements   = 0;
        result.threshold    = threshold;
        return;
    end

    % =====================================================================
    % STEP 4: Estimate tumour position (weighted centre of mass)
    % =====================================================================
    % Weight by absolute conductivity change so that the peak of the
    % reconstructed anomaly contributes more than the fringes.
    weights = abs_data(tumour_idx);
    weight_sum = sum(weights);

    position = sum(elem_centres(tumour_idx, :) .* weights, 1) / weight_sum;

    % =====================================================================
    % STEP 5: Estimate tumour diameter (volume-equivalent sphere)
    % =====================================================================
    % Sum the volumes of all elements in the tumour region, then
    % back-calculate the diameter of a sphere with the same volume.
    % This is more robust than using the maximum spatial extent,
    % which is sensitive to outlier elements.
    tumour_volume = sum(elem_volumes(tumour_idx));  % cm^3
    diameter_cm = 2 * (3 * tumour_volume / (4 * pi))^(1/3);
    diameter_mm = diameter_cm * 10;

    % =====================================================================
    % STEP 6: Estimate contrast ratio
    % =====================================================================
    % Difference reconstruction gives conductivity CHANGES (delta_sigma).
    % The estimated conductivity at the tumour is:
    %   sigma_tumour = background + delta_sigma
    % So the contrast ratio is:
    %   contrast = sigma_tumour / background = 1 + delta_sigma / background
    %
    % We use the actual (signed) values here, not absolute, because
    % the sign carries physical meaning.
    mean_change = mean(elem_data(tumour_idx));
    contrast_est = 1 + mean_change / background_cond;

    % Clamp to physically reasonable range. A contrast below 1.0 would
    % mean the tumour is less conductive than background, which is not
    % expected for malignant tissue in our model. A contrast above 10
    % is unrealistic. These clamps catch reconstruction artefacts.
    contrast_est = max(contrast_est, 0.5);
    contrast_est = min(contrast_est, 10.0);

    % =====================================================================
    % STEP 7: Pack results
    % =====================================================================
    result.position     = position;
    result.diameter_mm  = diameter_mm;
    result.contrast_est = contrast_est;
    result.detected     = true;
    result.n_elements   = n_tumour;
    result.threshold    = threshold;

end


% =========================================================================
% LOCAL FUNCTION: Otsu's threshold
% =========================================================================
function threshold = otsu_threshold(data)
% OTSU_THRESHOLD  Compute optimal threshold using Otsu's method.
%
% Finds the threshold that maximises the between-class variance of the
% data distribution, effectively separating it into two classes
% (background and tumour).
%
% This is a standalone implementation that does not require the Image
% Processing Toolbox.

    data = data(:);
    min_val = min(data);
    max_val = max(data);

    % Edge case: uniform data (no variation to threshold on)
    if max_val - min_val < eps
        threshold = max_val;
        return;
    end

    % Build normalised histogram
    n_bins = 256;
    edges = linspace(min_val, max_val, n_bins + 1);
    counts = histcounts(data, edges);
    p = counts / sum(counts);

    % Bin centres
    bin_centres = (edges(1:end-1) + edges(2:end)) / 2;

    % Cumulative sums
    omega = cumsum(p);           % Class probability (class 0)
    mu = cumsum(p .* bin_centres); % Cumulative mean
    mu_total = mu(end);           % Total mean

    % Between-class variance for each possible threshold
    % sigma_b^2 = [mu_total * omega - mu]^2 / [omega * (1 - omega)]
    numerator = (mu_total * omega - mu).^2;
    denominator = omega .* (1 - omega);

    % Avoid division by zero at the extremes
    denominator(denominator < eps) = eps;
    sigma_b_sq = numerator ./ denominator;

    % Find the bin with maximum between-class variance
    [~, best_idx] = max(sigma_b_sq);
    threshold = bin_centres(best_idx);

end
