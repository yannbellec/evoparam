% SPDX-License-Identifier: GPL-3.0-or-later
% Copyright (C) 2026 Yann V. Bellec. See LICENSE in this folder.
%
% crp_wrapper(in_file, out_file)
%
% Runs CRP_method (Miller et al. 2023) on every channel of in_file and saves, per channel,
% the p-value and t-value of the projection test at the response duration and the
% canonical shape scaled by the mean projection weight (NaN after the response duration).
%
% in_file holds V (samples x trials x channels) and t_win (samples, s); out_file gets
% p_value, t_value (channels x 1) and shape (channels x samples).
%
% NOT VERIFIED against the CRP scripts: the output fields read here (C, al, tR,
% t_value_tR, p_value_tR, in crp_parms or crp_projs) must be checked with the installed
% scripts before a result is published.

function crp_wrapper(in_file, out_file)
    s = load(in_file);
    V = s.V;
    t_win = s.t_win(:);
    n_samples = size(V, 1);
    n_channels = size(V, 3);
    p_value = nan(n_channels, 1);
    t_value = nan(n_channels, 1);
    shape = nan(n_channels, n_samples);
    for k = 1:n_channels
        [crp_parms, crp_projs] = CRP_method(V(:, :, k), t_win);
        p_value(k) = first_field(crp_parms, crp_projs, 'p_value_tR');
        t_value(k) = first_field(crp_parms, crp_projs, 't_value_tR');
        if isfield(crp_parms, 'C') && isfield(crp_parms, 'al')
            C = crp_parms.C(:);
            shape(k, 1:numel(C)) = C' * mean(crp_parms.al);
        end
    end
    save(out_file, 'p_value', 't_value', 'shape', '-v7');
end

function value = first_field(a, b, name)
    value = NaN;
    if isfield(a, name)
        value = double(a.(name));
    elseif isfield(b, name)
        value = double(b.(name));
    end
    value = value(1);
end
