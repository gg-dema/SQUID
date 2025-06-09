

function Vout = v_multivar(y1, y2)
    % Define goals in 3D
    goals = [-1, -4, 0;   % goal 1
            1, 4, 0]';  % goal 2 --> transpose for column-wise access

    % Flatten input grids
    y1_flat = y1(:);
    y2_flat = y2(:);
    z_fixed = 0;   % z is fixed (2D slice through z = 0)

    % Stack input points into [N x 3]
    y_t = [y1_flat, y2_flat, z_fixed * ones(size(y1_flat))];

    potential = zeros(size(y_t));  % same shape as y_t (N x 3)

    for i = 1:size(goals, 2)
        g = goals(:, i)';  % 1 x 3
        diff = y_t - g;    % N x 3
        dist_sq = sum(diff.^2, 2);  % N x 1
        weight = exp(-dist_sq);    % N x 1

        for d = 1:3
            potential(:, d) = potential(:, d) + weight * 2 .* (g(d) - y_t(:, d));
        end
    end

    % Optionally visualize the magnitude or a component of the vector field
    V_mag = sqrt(sum(potential.^2, 2));  % scalar potential from vector field

    % Reshape back to grid
    Vout = reshape(V_mag, size(y1));
end
