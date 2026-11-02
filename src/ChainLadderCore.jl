module ChainLadderCore

using Statistics
using Distributions
using Random

export last_observed_col, dev_factors, cdf_to_ultimate, project_chain_ladder,
       fitted_retro_triangle, pearson_residuals, mack_mseps, bootstrap_odp,
       bornhuetter_ferguson, calendar_effect_test

function last_observed_col(tri::AbstractMatrix{<:Real}, i::Int)
    k = 0
    for j in 1:size(tri, 2)
        if !isnan(tri[i, j])
            k = j
        end
    end
    return k
end

"""
    dev_factors(tri)

Fatores de desenvolvimento ponderados por volume (chain ladder clássico),
devolvendo `f` (vetor de fatores f_1..f_{maxdev-1}), `s` (soma usada como
denominador em cada coluna, necessária para o erro padrão de Mack) e `n`
(número de safras usadas em cada coluna).
"""
function dev_factors(tri::AbstractMatrix{<:Real})
    n_cohorts, max_dev = size(tri)
    f = fill(NaN, max_dev - 1)
    s = fill(NaN, max_dev - 1)
    n = fill(0, max_dev - 1)
    for k in 1:(max_dev - 1)
        num = 0.0
        den = 0.0
        cnt = 0
        for i in 1:n_cohorts
            if !isnan(tri[i, k]) && !isnan(tri[i, k + 1])
                num += tri[i, k + 1]
                den += tri[i, k]
                cnt += 1
            end
        end
        if cnt > 0 && den > 0
            f[k] = num / den
            s[k] = den
            n[k] = cnt
        end
    end
    return f, s, n
end

"""
    cdf_to_ultimate(f)

`CDF[j]` é o fator cumulativo que leva a coluna `j` até a última coluna
(ultimate). `CDF[maxdev] = 1`.
"""
function cdf_to_ultimate(f::Vector{Float64})
    max_dev = length(f) + 1
    cdf = fill(1.0, max_dev)
    for j in (max_dev - 1):-1:1
        cdf[j] = (isnan(f[j]) ? 1.0 : f[j]) * cdf[j + 1]
    end
    return cdf
end

function project_chain_ladder(tri::AbstractMatrix{<:Real}, f::Vector{Float64})
    n_cohorts, max_dev = size(tri)
    proj = Matrix{Float64}(tri)
    for i in 1:n_cohorts
        k_i = last_observed_col(tri, i)
        k_i == 0 && continue
        for j in (k_i + 1):max_dev
            proj[i, j] = proj[i, j - 1] * f[j - 1]
        end
    end
    return proj
end

"""
    fitted_retro_triangle(tri, f)

Triângulo ajustado retrospectivamente: para cada safra, usa o último
valor observado e os fatores de desenvolvimento para reconstruir uma
curva suave em todas as colunas (observadas e futuras). É a base para os
resíduos de Pearson (bootstrap) e coincide com a projeção chain ladder
nas colunas futuras.
"""
function fitted_retro_triangle(tri::AbstractMatrix{<:Real}, f::Vector{Float64})
    n_cohorts, max_dev = size(tri)
    cdf = cdf_to_ultimate(f)
    fitted = fill(NaN, n_cohorts, max_dev)
    for i in 1:n_cohorts
        k_i = last_observed_col(tri, i)
        k_i == 0 && continue
        ultimate_i = tri[i, k_i] * cdf[k_i]
        for j in 1:max_dev
            fitted[i, j] = ultimate_i / cdf[j]
        end
    end
    return fitted
end

function to_incremental(cum::AbstractMatrix{<:Real})
    n_cohorts, max_dev = size(cum)
    incr = fill(NaN, n_cohorts, max_dev)
    for i in 1:n_cohorts, j in 1:max_dev
        isnan(cum[i, j]) && continue
        incr[i, j] = j == 1 ? cum[i, j] : cum[i, j] - cum[i, j - 1]
    end
    return incr
end

"""
    pearson_residuals(tri, f)

Resíduos de Pearson do chain ladder visto como GLM Poisson sobredisperso
(England; Verrall, 1999/2002), calculados só nas células observadas.
Devolve a matriz de resíduos, o incremento ajustado e o parâmetro de
dispersão de Pearson (phi).
"""
function pearson_residuals(tri::AbstractMatrix{<:Real}, f::Vector{Float64})
    n_cohorts, max_dev = size(tri)
    fitted_cum = fitted_retro_triangle(tri, f)
    fitted_incr = to_incremental(fitted_cum)
    actual_incr = to_incremental(tri)
    resid = fill(NaN, n_cohorts, max_dev)
    n_obs = 0
    sum_sq = 0.0
    for i in 1:n_cohorts, j in 1:max_dev
        isnan(actual_incr[i, j]) && continue
        fm = fitted_incr[i, j]
        fm = max(fm, 1e-6)
        r = (actual_incr[i, j] - fm) / sqrt(fm)
        resid[i, j] = r
        n_obs += 1
        sum_sq += r^2
    end
    n_params = n_cohorts + (max_dev - 1)
    dof = max(n_obs - n_params, 1)
    phi = sum_sq / dof
    return resid, fitted_incr, phi, n_obs, dof
end

"""
    mack_mseps(tri, f)

Erro quadrático médio de previsão (MSEP) de Mack (1993), decomposto em
variância de processo e de estimação, por safra e para a reserva total.
Devolve `(mseps_por_safra, msep_total, sigma2)`.
"""
function mack_mseps(tri::AbstractMatrix{<:Real}, f::Vector{Float64})
    n_cohorts, max_dev = size(tri)
    _, s, n = dev_factors(tri)
    sigma2 = fill(NaN, max_dev - 1)
    for k in 1:(max_dev - 1)
        n[k] < 2 && continue
        acc = 0.0
        for i in 1:n_cohorts
            if !isnan(tri[i, k]) && !isnan(tri[i, k + 1])
                acc += tri[i, k] * (tri[i, k + 1] / tri[i, k] - f[k])^2
            end
        end
        sigma2[k] = acc / (n[k] - 1)
    end
    # extrapolação simples de Mack para a última coluna sem graus de liberdade
    for k in (max_dev - 1):-1:1
        if isnan(sigma2[k])
            if k < max_dev - 1 && !isnan(sigma2[k + 1])
                sigma2[k] = min(sigma2[k + 1], sigma2[k + 1]^2)
            elseif k > 1 && !isnan(sigma2[k - 1])
                sigma2[k] = sigma2[k - 1]
            else
                sigma2[k] = 0.0
            end
        end
    end

    proj = project_chain_ladder(tri, f)
    k_vec = [last_observed_col(tri, i) for i in 1:n_cohorts]
    ultimate = proj[:, max_dev]

    msep_i = fill(0.0, n_cohorts)
    for i in 1:n_cohorts
        k_i = k_vec[i]
        (k_i == 0 || k_i >= max_dev) && continue
        acc = 0.0
        for k in k_i:(max_dev - 1)
            fk = f[k]
            (isnan(fk) || fk == 0 || isnan(sigma2[k]) || isnan(s[k]) || s[k] <= 0) && continue
            acc += (sigma2[k] / fk^2) * (1 / proj[i, k] + 1 / s[k])
        end
        msep_i[i] = ultimate[i]^2 * acc
    end

    msep_total = sum(msep_i)
    for i in 1:n_cohorts, ip in 1:n_cohorts
        i >= ip && continue
        k_i, k_ip = k_vec[i], k_vec[ip]
        (k_i == 0 || k_ip == 0 || k_i >= max_dev || k_ip >= max_dev) && continue
        kmin = max(k_i, k_ip)
        acc = 0.0
        for k in kmin:(max_dev - 1)
            fk = f[k]
            (isnan(fk) || fk == 0 || isnan(sigma2[k]) || isnan(s[k]) || s[k] <= 0) && continue
            acc += sigma2[k] / fk^2 / s[k]
        end
        msep_total += 2 * ultimate[i] * ultimate[ip] * acc
    end

    return msep_i, msep_total, sigma2
end

"""
    bootstrap_odp(tri, f; n_boot=1000, seed=20261102)

Bootstrap ODP (England; Verrall, 1999/2002): reamostra resíduos de
Pearson para gerar incerteza de parâmetro, e soma incerteza de processo
via distribuição Gama com a dispersão de Pearson estimada. Devolve o
vetor de reservas totais simuladas (distribuição preditiva completa).
"""
function bootstrap_odp(tri::AbstractMatrix{<:Real}, f::Vector{Float64};
                        n_boot::Int=1000, seed::Int=20261102)
    n_cohorts, max_dev = size(tri)
    resid, fitted_incr, phi, n_obs, dof = pearson_residuals(tri, f)
    adj = sqrt(n_obs / dof)

    valid_idx = [(i, j) for i in 1:n_cohorts, j in 1:max_dev if !isnan(resid[i, j])]
    valid_resid = [resid[i, j] for (i, j) in valid_idx]

    k_vec = [last_observed_col(tri, i) for i in 1:n_cohorts]
    latest = [k_vec[i] == 0 ? 0.0 : tri[i, k_vec[i]] for i in 1:n_cohorts]

    rng = MersenneTwister(seed)
    reservas = fill(0.0, n_boot)

    for b in 1:n_boot
        pseudo_incr = fill(NaN, n_cohorts, max_dev)
        for (idx, (i, j)) in enumerate(valid_idx)
            e_star = valid_resid[rand(rng, 1:length(valid_resid))]
            fm = max(fitted_incr[i, j], 1e-6)
            pseudo_incr[i, j] = max(fm + e_star * adj * sqrt(fm), 0.0)
        end
        pseudo_cum = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            acc = 0.0
            for j in 1:k_vec[i]
                acc += pseudo_incr[i, j]
                pseudo_cum[i, j] = acc
            end
        end
        f_star, = dev_factors(pseudo_cum)
        for k in 1:(max_dev - 1)
            isnan(f_star[k]) && (f_star[k] = f[k])
        end

        total_reserva = 0.0
        for i in 1:n_cohorts
            k_i = k_vec[i]
            (k_i == 0 || k_i >= max_dev) && continue
            cdf_star = cdf_to_ultimate(f_star)
            mean_ultimate = latest[i] * cdf_star[k_i]
            mu_reserva = max(mean_ultimate - latest[i], 1e-6)
            # processo ODP: Var = phi * média, casado por momentos a uma Gama(shape=mu/phi, scale=phi)
            shape_gamma = mu_reserva / phi
            scale_gamma = phi
            ultimate_proc = latest[i] + rand(rng, Gamma(shape_gamma, scale_gamma))
            total_reserva += ultimate_proc - latest[i]
        end
        reservas[b] = total_reserva
    end

    return reservas, phi
end

"""
    bornhuetter_ferguson(tri, f, prior_rate, volume)

Reserva de Bornhuetter-Ferguson (Bornhuetter; Ferguson, 1972): mistura o
desenvolvimento observado com um `prior_rate` externo (taxa de perda
esperada a priori, independente da própria safra), ponderado pela fração
já madura da safra (1/CDF). Devolve a reserva por safra.
"""
function bornhuetter_ferguson(tri::AbstractMatrix{<:Real}, f::Vector{Float64},
                               prior_rate::Vector{Float64}, volume::Vector{Float64})
    n_cohorts, max_dev = size(tri)
    cdf = cdf_to_ultimate(f)
    reserva = fill(0.0, n_cohorts)
    for i in 1:n_cohorts
        k_i = last_observed_col(tri, i)
        (k_i == 0 || k_i >= max_dev) && continue
        w = 1 / cdf[k_i]
        reserva[i] = prior_rate[i] * volume[i] * (1 - w)
    end
    return reserva
end

"""
    calendar_effect_test(tri, f)

Teste de efeito de calendário via ANOVA de um fator sobre os resíduos de
Pearson agrupados por diagonal (idade + safra constante = mesmo
trimestre-calendário). Devolve a estatística F e o p-valor.
"""
function calendar_effect_test(tri::AbstractMatrix{<:Real}, f::Vector{Float64})
    n_cohorts, max_dev = size(tri)
    resid, = pearson_residuals(tri, f)
    grupos = Dict{Int, Vector{Float64}}()
    for i in 1:n_cohorts, j in 1:max_dev
        isnan(resid[i, j]) && continue
        d = i + j
        push!(get!(grupos, d, Float64[]), resid[i, j])
    end
    all_vals = vcat(values(grupos)...)
    grand_mean = mean(all_vals)
    n_total = length(all_vals)
    k_groups = length(grupos)
    ss_between = sum(length(v) * (mean(v) - grand_mean)^2 for v in values(grupos))
    ss_within = sum(sum((v .- mean(v)) .^ 2) for v in values(grupos))
    df_between = k_groups - 1
    df_within = n_total - k_groups
    ms_between = ss_between / df_between
    ms_within = ss_within / max(df_within, 1)
    f_stat = ms_within > 0 ? ms_between / ms_within : NaN
    p_valor = isnan(f_stat) ? NaN : 1 - cdf(FDist(df_between, df_within), f_stat)
    return f_stat, p_valor, df_between, df_within
end

end # module
