%{
Analytic kernel statistics for the ZZ feature map vs. feature correlation (Cor) 

Model:
- mu_x, mu_y: random unit vectors with a predifined angle between them.
- Sigma = (1 - Cor) * I + Cor * ones(d): equicorrelated covariance whose uniform off-diagonal value is the swept correlation level Cor.
- cls_x_*  : (x, x') both from class x
- cls_y_*  : (y, y') both from class y
- cls_xy_* : mixed pairs
%}


clc; clear; close all;

% ----------------------------------------------------------------------
% Sweep configuration
% ----------------------------------------------------------------------
dim = 3; %:6;
Cor = 0.0:0.1:0.9;
Cor = [Cor 0.99];      

random_seeds = 1:5;
angles = [90, 120, 150, 180];      % angle (deg) between the class means

% ----------------------------------------------------------------------
% Sweep
% ----------------------------------------------------------------------
for k = 1:length(dim)
    d = dim(k);

    for angle_idx = 1:length(angles)
        angle = angles(angle_idx);

        for random_seed = random_seeds
            rng(random_seed)   % deterministic mean-vector draws per (d, angle, seed)
            tic

            folder_name = sprintf("random_vectors_angle_%.1f", angle);
            file_name = sprintf("kernel_statistics_%d_seed%d.mat", d, random_seed);
            save_path = fullfile(folder_name, file_name);

            % Resume support: skip combinations that are already computed
            if isfile(save_path)
                fprintf('>> Skipping (exists): %s\n', save_path);
                continue
            end

            fprintf('\n>> Starting: Angle %g | Dim %d | Seed %d\n', angle, d, random_seed);

            % Fresh accumulators for this (d, angle, seed) combination;
            % only row k (this dimension) is populated.
            cls_x_mean  = nan([length(dim), length(Cor)]);
            cls_y_mean  = nan([length(dim), length(Cor)]);
            cls_xy_mean = nan([length(dim), length(Cor)]);
            cls_x_var   = nan([length(dim), length(Cor)]);
            cls_y_var   = nan([length(dim), length(Cor)]);
            cls_xy_var  = nan([length(dim), length(Cor)]);


            theta = deg2rad(angle);

            % 1. Random unit vector v1
            v1 = randn(d, 1);
            v1 = v1 / norm(v1);

            % 2. Random unit vector orthogonal to v1
            v_temp = randn(d, 1);
            v_orth = v_temp - (v1' * v_temp) * v1;
            v_orth = v_orth / norm(v_orth);

            % 3. v2 at the prescribed angle to v1
            v2 = cos(theta) * v1 + sin(theta) * v_orth;
            actual_angle = acosd(max(min(v1' * v2, 1), -1));

            for c = 1:length(Cor)

                fprintf('[Sweep] Angle: %5.1f | Dim: %2d | Seed: %d | Cor: %.4f | Calc_Angle: %6.2f\n', ...
                    angle, d, random_seed, Cor(c), actual_angle);

                mu_x = v1;
                mu_y = v2;

                % Equicorrelated covariance with off-diagonal level Cor
                Sigma = (1 - Cor(c)) * eye(d) + Cor(c) * ones(d);

                % Full-entanglement ZZ coupling: W(i,j) = 1 for all i ~= j
                pairs = nchoosek(1:d, 2);
                W = zeros(d);
                for t = 1:size(pairs, 1)
                    i = pairs(t, 1); j = pairs(t, 2);
                    W(i, j) = 1; W(j, i) = 1;
                end
                W(1:d+1:end) = 0;

                % Closed-form kernel moments:
                %   Eabs2_uniformZZ_two_means -> E[K]
                %   EK2_uniformZZ_two_means   -> E[K^2]
                EK_xx = Eabs2_uniformZZ_two_means(Sigma, mu_x, mu_x, W);
                EK_yy = Eabs2_uniformZZ_two_means(Sigma, mu_y, mu_y, W);
                EK_xy = Eabs2_uniformZZ_two_means(Sigma, mu_x, mu_y, W);

                EK_2_xx = EK2_uniformZZ_two_means(Sigma, mu_x, mu_x, W);
                EK_2_yy = EK2_uniformZZ_two_means(Sigma, mu_y, mu_y, W);
                EK_2_xy = EK2_uniformZZ_two_means(Sigma, mu_x, mu_y, W);

                cls_x_mean(k, c)  = EK_xx;
                cls_y_mean(k, c)  = EK_yy;
                cls_xy_mean(k, c) = EK_xy;
                cls_x_var(k, c)   = EK_2_xx - EK_xx^2;   % Var[K] = E[K^2] - E[K]^2
                cls_y_var(k, c)   = EK_2_yy - EK_yy^2;
                cls_xy_var(k, c)  = EK_2_xy - EK_xy^2;
            end

            if ~isfolder(folder_name)
                mkdir(folder_name);
                fprintf('--- New folder created: %s ---\n', folder_name);
            end

            save(save_path, "cls_x_mean", "cls_y_mean", "cls_xy_mean", ...
                "cls_x_var", "cls_y_var", "cls_xy_var");
            fprintf('[Success] Data saved to: %s\n', save_path);
            toc
        end
    end
end

%% ======================= Local helpers =======================
function EK2 = EK2_uniformZZ_two_means(Sigma, mu_x, mu_y, W)
% E[K^2] = E[|A|^4] for the ZZ feature map with two Gaussian means.
% K(x,y) = |A(x,y)|^2,  A(x,y) = 2^{-d} sum_s e^{i phi_s(x)} e^{-i phi_s(y)}
% => K^2(x,y) = 4^{-2d} sum_{s,t,u,v} e^{i(phi_s-phi_t+phi_u-phi_v)(x)}
%                                     e^{-i(phi_s-phi_t+phi_u-phi_v)(y)}
% Each term is a Gaussian characteristic-function integral of a quadratic
% phase, evaluated in closed form below.

Sigma = double(Sigma);
d = size(Sigma, 1);
I = eye(d);
alpha = ones(d, 1);   % RZ rotation on every qubit (uniform data scaling)

% enumerate sign patterns {+1,-1}^d
S = all_pm1_patterns(d);
Ns = size(S, 1);

% precompute A_s, b_s for all s
As_list = zeros(d, d, Ns);
b_list = zeros(d, Ns);
for i = 1:Ns
    s = S(i, :).';
    ssT = s * s.';

    % A_s = 0.5 * (W .* s s^T), diag = 0
    As = 0.5 * (W .* ssT);
    As(1:d+1:end) = 0;

    % b_s = s .* (alpha - pi W s)
    b = s .* (alpha - pi * (W * s));

    As_list(:, :, i) = As;
    b_list(:, i) = b;
end

E_sum = 0.0;
for i = 1:Ns
    As_i = As_list(:, :, i); b_i = b_list(:, i);
    for j = 1:Ns
        As_j = As_list(:, :, j); b_j = b_list(:, j);

        % pair (s,t): A_st, b_st
        A_st = As_i - As_j;
        b_st = b_i - b_j;

        for u = 1:Ns
            As_u = As_list(:, :, u); b_u = b_list(:, u);
            for v = 1:Ns
                As_v = As_list(:, :, v); b_v = b_list(:, v);

                % quadruple (s,t,u,v): A = A_st + A_uv, b = b_st + b_uv
                A = A_st + (As_u - As_v);
                b = b_st + (b_u - b_v);

                % v(mu) = b + 2 A mu
                vx = b + 2 * (A * mu_x);
                vy = b + 2 * (A * mu_y);

                % B = I - 2 i Sigma A,  B* = I + 2 i Sigma A
                B = I - 2 * (1i) * (Sigma * A);
                Bc = conj(B);

                % quadratic terms v^T B^{-1} Sigma v
                yx = B  \ (Sigma * vx);
                yy = Bc \ (Sigma * vy);
                quad_x = vx.' * yx;
                quad_y = vy.' * yy;

                AbsDetB = abs(det(B));

                % phase term (mu_x^T b + mu_x^T A mu_x) - (mu_y^T b + mu_y^T A mu_y)
                phase = (mu_x.' * b + mu_x.' * (A * mu_x)) ...
                      - (mu_y.' * b + mu_y.' * (A * mu_y));

                term = exp(1i * phase) ...
                     * exp(-0.5 * quad_x) ...
                     * exp(-0.5 * conj(quad_y)) ...
                     / AbsDetB;

                E_sum = E_sum + term;
            end
        end
    end
end

% K has prefactor 4^{-d}, so K^2 has prefactor 4^{-2d}
EK2 = real(E_sum) / (4^(2 * d));   % real by construction
end


function Eabs2 = Eabs2_uniformZZ_two_means(Sigma, mu_x, mu_y, W)
% E[K] = E[|A|^2] = 2^{-2d} sum_{s,t} M_st(mu_x) * conj(M_st(mu_y)),
% with each term a closed-form Gaussian integral (same structure as above,
% over pairs (s,t) instead of quadruples).

Sigma = double(Sigma);
d = size(Sigma, 1);
I = eye(d);
alpha = ones(d, 1);

S = all_pm1_patterns(d);
Ns = size(S, 1);

As_list = zeros(d, d, Ns);
b_list = zeros(d, Ns);
for i = 1:Ns
    s = S(i, :).';
    ssT = s * s.';
    As = 0.5 * (W .* ssT);
    As(1:d+1:end) = 0;
    b = s .* (alpha - pi * (W * s));
    As_list(:, :, i) = As;
    b_list(:, i) = b;
end

E_sum = 0.0;
for i = 1:Ns
    As_i = As_list(:, :, i); b_i = b_list(:, i);
    for j = 1:Ns
        As_t = As_list(:, :, j); b_t = b_list(:, j);

        A = As_i - As_t;
        b = b_i - b_t;

        vx = b + 2 * (A * mu_x);
        vy = b + 2 * (A * mu_y);

        B = I - 2 * (1i) * (Sigma * A);
        Bc = conj(B);

        yx = B  \ (Sigma * vx);
        yy = Bc \ (Sigma * vy);

        quad_x = vx.' * yx;
        quad_y = vy.' * yy;
        AbsDetB = abs(det(B));
        phase = (mu_x.' * b + mu_x.' * (A * mu_x)) ...
              - (mu_y.' * b + mu_y.' * (A * mu_y));

        term = exp(1i * phase) * exp(-0.5 * quad_x) * exp(-0.5 * conj(quad_y)) ...
             / AbsDetB;

        E_sum = E_sum + term;
    end
end

Eabs2 = real(E_sum) / (2^(2 * d));  % >= 0
end


function S = all_pm1_patterns(d)
% All 2^d sign patterns in {+1, -1}^d.
n = 2^d;
S = zeros(n, d);
for k = 0:n-1
    bits = bitget(k, 1:d);      % LSB -> MSB
    S(k+1, :) = 1 - 2 * bits;   % 0 -> +1, 1 -> -1
end
end
