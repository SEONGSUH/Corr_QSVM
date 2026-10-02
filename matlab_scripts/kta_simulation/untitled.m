%{
Feature correlation vs. kernel-bandwidth control for ZZ-kernel concentration
============================================================================

Question: does raising the equicorrelation Cor reduce Var[K] (concentration)
in a way that is *different* from simply shrinking the input scale
(kernel bandwidth)?

Equicorrelated Sigma = (1-Cor) I + Cor 11^T has eigenvalues
    1 + (d-1) Cor   (common-mode direction)      x 1
    1 - Cor         (independent directions)     x (d-1)
So raising Cor shrinks the d-1 independent directions to variance (1-Cor)
while keeping a common mode. Three conditions are compared at each Cor:

  'corr'      : Sigma = (1-Cor) I + Cor 11^T,   mu unchanged   <- proposed
  'diag'      : Sigma = (1-Cor) I,              mu unchanged   <- common mode removed
  'bandwidth' : Sigma = (1-Cor) I,              mu -> sqrt(1-Cor) mu
                                                <- global rescale x -> sqrt(1-Cor) x

'corr' vs 'diag'      isolates the common-mode contribution.
'corr' vs 'bandwidth' is the classic bandwidth baseline
                      (Shaydulin & Wild 2022; Canatar et al. 2022).

Outputs per (condition, d, Cor, seed):
  M_xx, M_yy, M_xy   : E[K] within class x, within class y, between classes
  V_xx, V_yy, V_xy   : Var[K]  (concentration metric)
  KTA                : analytic two-class KTA, Eq. (9) of the AAAI paper

Helpers are the same closed-form Gaussian integrals as compute_KTA.m; the
E[K^2] loop is rewritten over precomputed (s,t) pairs with the
(p,q) <-> (q,p) symmetry, which halves the 2^{4d} cost. d <= 6 is practical
in MATLAB; use the Monte-Carlo statevector script for larger d.
%}

clc; clear; close all;

% ----------------------------------------------------------------------
% Sweep configuration
% ----------------------------------------------------------------------
dim          = 3:6;
Cor          = [0.0 0.25 0.5 0.75 0.9 1.0];
random_seeds = 1:3;
angle        = 90;                       % V_xx does not depend on angle
conditions   = {'corr', 'diag', 'bandwidth'};

out_dir = 'concentration_control';
if ~isfolder(out_dir), mkdir(out_dir); end

% ----------------------------------------------------------------------
% Sweep
% ----------------------------------------------------------------------
for k = 1:numel(dim)
    d = dim(k);

    % full-entanglement ZZ coupling graph
    W = ones(d) - eye(d);

    for random_seed = random_seeds
        save_path = fullfile(out_dir, sprintf('stats_d%d_seed%d.mat', d, random_seed));
        if isfile(save_path)
            fprintf('>> skipping (exists): %s\n', save_path); continue
        end

        % class means: unit vectors at prescribed angle (same draw as compute_KTA.m)
        rng(random_seed)
        theta  = deg2rad(angle);
        v1     = randn(d,1); v1 = v1/norm(v1);
        v_tmp  = randn(d,1); v_orth = v_tmp - (v1'*v_tmp)*v1; v_orth = v_orth/norm(v_orth);
        v2     = cos(theta)*v1 + sin(theta)*v_orth;

        nC = numel(conditions); nR = numel(Cor);
        M_xx = nan(nC,nR); M_yy = M_xx; M_xy = M_xx;
        V_xx = M_xx;       V_yy = M_xx; V_xy = M_xx;
        KTA  = M_xx;

        for ci = 1:nC
            cond = conditions{ci};
            for c = 1:nR
                r = Cor(c);
                tic
                switch cond
                    case 'corr'
                        Sigma = (1-r)*eye(d) + r*ones(d);
                        mu_x = v1;            mu_y = v2;
                    case 'diag'
                        Sigma = (1-r)*eye(d);
                        mu_x = v1;            mu_y = v2;
                    case 'bandwidth'
                        Sigma = (1-r)*eye(d);
                        mu_x = sqrt(1-r)*v1;  mu_y = sqrt(1-r)*v2;
                end

                % first moments
                EK_xx = EK_ZZ(Sigma, mu_x, mu_x, W);
                EK_yy = EK_ZZ(Sigma, mu_y, mu_y, W);
                EK_xy = EK_ZZ(Sigma, mu_x, mu_y, W);
                % second moments
                EK2_xx = EK2_ZZ(Sigma, mu_x, mu_x, W);
                EK2_yy = EK2_ZZ(Sigma, mu_y, mu_y, W);
                EK2_xy = EK2_ZZ(Sigma, mu_x, mu_y, W);

                M_xx(ci,c) = EK_xx;  M_yy(ci,c) = EK_yy;  M_xy(ci,c) = EK_xy;
                V_xx(ci,c) = EK2_xx - EK_xx^2;
                V_yy(ci,c) = EK2_yy - EK_yy^2;
                V_xy(ci,c) = EK2_xy - EK_xy^2;

                % analytic KTA, Eq. (9)
                num = EK_xx + EK_yy - 2*EK_xy;
                den = 2*sqrt(EK2_xx + EK2_yy + 2*EK2_xy);
                KTA(ci,c) = num/den;

                fprintf('[d=%d seed=%d %-9s Cor=%.2f]  Var_xx=%.3e  KTA=%.4f  (%.1fs)\n', ...
                    d, random_seed, cond, r, V_xx(ci,c), KTA(ci,c), toc);
            end
        end

        save(save_path, 'conditions','Cor','d','angle', ...
            'M_xx','M_yy','M_xy','V_xx','V_yy','V_xy','KTA');
        fprintf('[saved] %s\n', save_path);
    end
end

% ----------------------------------------------------------------------
% Aggregate over seeds and plot
% ----------------------------------------------------------------------
nC = numel(conditions); nR = numel(Cor); nD = numel(dim);
Vxx_all = nan(nD,nC,nR,numel(random_seeds));
Vxy_all = Vxx_all; KTA_all = Vxx_all; SEP_all = Vxx_all;
for k = 1:nD
    for si = 1:numel(random_seeds)
        S = load(fullfile(out_dir, sprintf('stats_d%d_seed%d.mat', dim(k), random_seeds(si))));
        Vxx_all(k,:,:,si) = S.V_xx;
        Vxy_all(k,:,:,si) = S.V_xy;
        KTA_all(k,:,:,si) = S.KTA;
        SEP_all(k,:,:,si) = S.M_xx + S.M_yy - 2*S.M_xy;   % numerator of KTA
    end
end
Vxx = mean(Vxx_all,4); Vxy = mean(Vxy_all,4);
KTAm = mean(KTA_all,4); SEP = mean(SEP_all,4);

cols  = lines(nR);
lstyle = {'-','--',':'};      % corr, diag, bandwidth
mk     = {'o','s','^'};

% --- Fig 1: log Var[K] vs d, per Cor; line style = condition -----------
figure('Position',[100 100 900 340]);
subplot(1,2,1); hold on; box on; grid on;
for c = 1:nR
    for ci = 1:nC
        semilogy(dim, squeeze(Vxx(:,ci,c)), 'LineStyle',lstyle{ci}, 'Marker',mk{ci}, ...
            'Color',cols(c,:), 'LineWidth',1.5, ...
            'DisplayName',sprintf('%s, Cor=%.2f',conditions{ci},Cor(c)));
    end
end
set(gca,'YScale','log'); xlabel('d (qubits)'); ylabel('Var[K] intra-class');
title('Concentration: solid=corr, dashed=diag, dotted=bandwidth');
legend('show','Location','southwest','FontSize',7,'NumColumns',2);

% --- decay rate beta(Cor): slope of log2 Var vs d ---------------------
subplot(1,2,2); hold on; box on; grid on;
for ci = 1:nC
    beta = nan(1,nR);
    for c = 1:nR
        y = log2(squeeze(Vxx(:,ci,c)));
        ok = isfinite(y);
        if nnz(ok) >= 2, p = polyfit(dim(ok), y(ok), 1); beta(c) = -p(1); end
    end
    plot(Cor, beta, 'LineStyle',lstyle{ci}, 'Marker',mk{ci}, 'LineWidth',1.5, ...
        'DisplayName',conditions{ci});
end
xlabel('Cor'); ylabel('\beta  (Var[K] \propto 2^{-\beta d})');
title('Decay rate vs Cor'); legend('show','Location','best');

% --- Fig 2: KTA and class separation vs Cor, largest d ---------------
figure('Position',[100 500 900 340]);
kk = nD;
subplot(1,2,1); hold on; box on; grid on;
for ci = 1:nC
    plot(Cor, squeeze(KTAm(kk,ci,:)), 'LineStyle',lstyle{ci}, 'Marker',mk{ci}, ...
        'LineWidth',1.5, 'DisplayName',conditions{ci});
end
xlabel('Cor'); ylabel('KTA'); title(sprintf('KTA vs Cor (d=%d)',dim(kk)));
legend('show','Location','best');

subplot(1,2,2); hold on; box on; grid on;
for ci = 1:nC
    plot(Cor, squeeze(SEP(kk,ci,:)), 'LineStyle',lstyle{ci}, 'Marker',mk{ci}, ...
        'LineWidth',1.5, 'DisplayName',conditions{ci});
end
xlabel('Cor'); ylabel('M_0 + M_1 - 2M_{01}');
title(sprintf('Class separation vs Cor (d=%d)',dim(kk)));
legend('show','Location','best');

% --- Table: same Var[K], different KTA? ------------------------------
fprintf('\n=== d=%d: corr vs bandwidth at matched Cor ===\n', dim(kk));
fprintf('%6s | %10s %10s | %8s %8s | %8s %8s\n','Cor','Vxx corr','Vxx bw','KTA corr','KTA bw','SEP corr','SEP bw');
for c = 1:nR
    fprintf('%6.2f | %10.3e %10.3e | %8.4f %8.4f | %8.4f %8.4f\n', Cor(c), ...
        Vxx(kk,1,c), Vxx(kk,3,c), KTAm(kk,1,c), KTAm(kk,3,c), SEP(kk,1,c), SEP(kk,3,c));
end

%% ======================= Local helpers =======================
function [As_list, b_list] = precompute_sign_terms(W, d)
% A_s = 0.5 (W .* s s^T), diag 0 ;  b_s = s .* (1 - pi W s)
S  = all_pm1_patterns(d);
Ns = size(S,1);
As_list = zeros(d,d,Ns); b_list = zeros(d,Ns);
for i = 1:Ns
    s  = S(i,:).';
    As = 0.5*(W .* (s*s.')); As(1:d+1:end) = 0;
    As_list(:,:,i) = As;
    b_list(:,i)    = s .* (1 - pi*(W*s));
end
end

function [Ap, bp] = precompute_pairs(As_list, b_list)
% all (s,t) difference pairs: A_st = A_s - A_t, b_st = b_s - b_t   (Ns^2 of them)
[d,~,Ns] = size(As_list);
Np = Ns^2;
Ap = zeros(d,d,Np); bp = zeros(d,Np);
n = 0;
for i = 1:Ns
    for j = 1:Ns
        n = n+1;
        Ap(:,:,n) = As_list(:,:,i) - As_list(:,:,j);
        bp(:,n)   = b_list(:,i)   - b_list(:,j);
    end
end
end

function term = gauss_term(A, b, Sigma, mu_x, mu_y, I)
% closed-form Gaussian integral of exp(i(x^T A x + b^T x)) * conj(same in y)
vx = b + 2*(A*mu_x);
vy = b + 2*(A*mu_y);
B  = I - 2i*(Sigma*A);
Bc = conj(B);
quad_x = vx.' * (B  \ (Sigma*vx));
quad_y = vy.' * (Bc \ (Sigma*vy));
phase  = (mu_x.'*b + mu_x.'*(A*mu_x)) - (mu_y.'*b + mu_y.'*(A*mu_y));
term   = exp(1i*phase) * exp(-0.5*quad_x) * exp(-0.5*conj(quad_y)) / abs(det(B));
end

function EK = EK_ZZ(Sigma, mu_x, mu_y, W)
% E[K] = 2^{-2d} sum_{s,t} (...)
d = size(Sigma,1); I = eye(d);
[As_list, b_list] = precompute_sign_terms(W, d);
[Ap, bp] = precompute_pairs(As_list, b_list);
Np = size(bp,2);
E = 0;
for p = 1:Np
    E = E + gauss_term(Ap(:,:,p), bp(:,p), Sigma, mu_x, mu_y, I);
end
EK = real(E) / 2^(2*d);
end

function EK2 = EK2_ZZ(Sigma, mu_x, mu_y, W)
% E[K^2] = 4^{-2d} sum_{(s,t),(u,v)} f(A_st + A_uv, b_st + b_uv)
% f is symmetric in the two pairs -> sum over p<=q with weight 2 off-diagonal.
d = size(Sigma,1); I = eye(d);
[As_list, b_list] = precompute_sign_terms(W, d);
[Ap, bp] = precompute_pairs(As_list, b_list);
Np = size(bp,2);
partial = zeros(Np,1);
parfor p = 1:Np          % runs serially without Parallel Computing Toolbox
    Apn = Ap(:,:,p); bpn = bp(:,p);
    acc = gauss_term(2*Apn, 2*bpn, Sigma, mu_x, mu_y, I);     % q = p
    for q = p+1:Np
        acc = acc + 2*gauss_term(Apn + Ap(:,:,q), bpn + bp(:,q), Sigma, mu_x, mu_y, I);
    end
    partial(p) = real(acc);
end
EK2 = sum(partial) / 4^(2*d);
end

function S = all_pm1_patterns(d)
n = 2^d; S = zeros(n,d);
for k = 0:n-1
    S(k+1,:) = 1 - 2*bitget(k,1:d);
end
end