module ChainLadderCore

using Statistics
using Distributions
using Random

export last_observed_col, dev_factors, extrapolate_factor_tail, extrapolate_sigma2_tail,
       mack_classical_last_sigma2, observed_sigma2, cdf_to_ultimate, project_chain_ladder,
       fitted_retro_triangle, pearson_residuals, mack_mseps, bootstrap_reserves,
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
devolvendo `f` (vetor de fatores f_1..f_{maxdev-1}, `NaN` onde não há par
observado), `s` (soma usada como denominador em cada coluna, necessária
para o erro padrão de Mack) e `n` (número de safras usadas em cada
coluna). Não extrapola cauda: isso é feito por `extrapolate_factor_tail`.
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
    extrapolate_factor_tail(f, n; min_pairs=2)

Extrapolação de cauda por curva ajustada em log(f_k - 1) contra a idade k
(forma de Sherman, padrão em literatura atuarial de cauda), usando só as
colunas com pelo menos `min_pairs` safras observadas para o ajuste de
mínimos quadrados. Colunas sem fator observável recebem o valor
extrapolado pela curva; colunas observadas não são alteradas. Devolve o
vetor de fatores completo e os coeficientes `(a, b)` da reta ajustada
(`f_k = 1 + exp(a + b*k)`), ou `(NaN, NaN)` se não houver pontos
suficientes para ajustar (caso em que a cauda usa o último fator
observado, sem decaimento adicional, e isso deve ser declarado).
"""
function extrapolate_factor_tail(f::Vector{Float64}, n::Vector{Int}; min_pairs::Int=2)
    K = length(f)
    ks = [k for k in 1:K if !isnan(f[k]) && n[k] >= min_pairs && f[k] > 1.0]
    f_full = copy(f)
    if length(ks) < 2
        last_obs = findlast(!isnan, f)
        fallback = last_obs === nothing ? 1.0 : f[last_obs]
        for k in 1:K
            isnan(f_full[k]) && (f_full[k] = fallback)
        end
        return f_full, (NaN, NaN)
    end
    xs = Float64.(ks)
    ys = [log(f[k] - 1.0) for k in ks]
    xbar, ybar = mean(xs), mean(ys)
    b = sum((xs .- xbar) .* (ys .- ybar)) / sum((xs .- xbar) .^ 2)
    a = ybar - b * xbar
    for k in 1:K
        isnan(f_full[k]) && (f_full[k] = 1.0 + exp(a + b * k))
    end
    return f_full, (a, b)
end

"""
    extrapolate_sigma2_tail(sigma2, n; min_pairs=2)

Mesma lógica de `extrapolate_factor_tail`, aplicada a σ²_k: ajusta
log(σ²_k) contra a idade k nas colunas com `n[k] >= min_pairs`, e usa a
curva para preencher as colunas sem dado suficiente. Consistente com a
extrapolação de cauda dos próprios fatores, em vez da regra
min(σ²[k+1], σ²[k+1]²) (que não é a extrapolação de Mack e zera a cauda
prematuramente).
"""
function extrapolate_sigma2_tail(sigma2::Vector{Float64}, n::Vector{Int}; min_pairs::Int=2)
    K = length(sigma2)
    ks = [k for k in 1:K if !isnan(sigma2[k]) && n[k] >= min_pairs && sigma2[k] > 0]
    sigma2_full = copy(sigma2)
    if length(ks) < 2
        last_obs = findlast(k -> !isnan(sigma2[k]) && sigma2[k] > 0, 1:K)
        fallback = last_obs === nothing ? 0.0 : sigma2[last_obs]
        for k in 1:K
            isnan(sigma2_full[k]) && (sigma2_full[k] = fallback)
        end
        return sigma2_full, (NaN, NaN)
    end
    xs = Float64.(ks)
    ys = [log(sigma2[k]) for k in ks]
    xbar, ybar = mean(xs), mean(ys)
    b = sum((xs .- xbar) .* (ys .- ybar)) / sum((xs .- xbar) .^ 2)
    a = ybar - b * xbar
    for k in 1:K
        isnan(sigma2_full[k]) && (sigma2_full[k] = exp(a + b * k))
    end
    return sigma2_full, (a, b)
end

"""
    mack_classical_last_sigma2(sigma2)

Regra original de Mack (1993) para a única coluna final sem grau de
liberdade suficiente (só uma safra no par), usada no próprio exemplo de
Taylor/Ashe do artigo. Recebe σ²_1, ..., σ²_K (as colunas já conhecidas) e
devolve σ²_{K+1} = min(σ²_K, σ²_{K-1}, σ²_K²/σ²_{K-1}). Só resolve esse
caso específico (uma coluna faltante, a última), não o caso deste
pipeline, em que várias colunas seguidas podem não ter nenhum par
observável numa avaliação antecipada. `extrapolate_sigma2_tail`, usada no
restante deste módulo, é uma generalização por curva log-linear para esse
caso, declarada como alternativa à regra de Mack, não uma implementação
dela.
"""
function mack_classical_last_sigma2(sigma2::Vector{Float64})
    K = length(sigma2)
    @assert K >= 2 "A regra de Mack para a coluna seguinte precisa de pelo menos duas colunas conhecidas"
    s_last = sigma2[K]
    s_prev = sigma2[K - 1]
    return min(s_prev, s_last, s_last^2 / s_prev)
end

"""
    cdf_to_ultimate(f)

`CDF[j]` é o fator cumulativo que leva a coluna `j` até a última coluna
(ultimate). `CDF[maxdev] = 1`. Espera um vetor de fatores já com a cauda
extrapolada (sem `NaN`).
"""
function cdf_to_ultimate(f::Vector{Float64})
    max_dev = length(f) + 1
    cdf = fill(1.0, max_dev)
    for j in (max_dev - 1):-1:1
        cdf[j] = f[j] * cdf[j + 1]
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
valor observado e os fatores de desenvolvimento (já com cauda
extrapolada) para reconstruir uma curva suave em todas as colunas. É a
base para os resíduos de Pearson (bootstrap) e coincide com a projeção
chain ladder nas colunas futuras.
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
(Renshaw; Verrall, 1998), calculados só nas células observadas. Devolve a
matriz de resíduos, o incremento ajustado e o parâmetro de dispersão de
Pearson (phi). `f` já deve vir com a cauda extrapolada.
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
    observed_sigma2(tri, f, n)

σ²_k observado de Mack (1993) por coluna: a variância dos incrementos
relativos ponderada pelo valor acumulado, só onde há pelo menos 2 safras
no par (k, k+1). Colunas sem dado suficiente ficam `NaN`, para
extrapolação posterior por `extrapolate_sigma2_tail`.
"""
function observed_sigma2(tri::AbstractMatrix{<:Real}, f::Vector{Float64}, n::Vector{Int})
    n_cohorts, max_dev = size(tri)
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
    return sigma2
end

"""
    mack_mseps(tri, f, sigma2; target_col=nothing)

Erro quadrático médio de previsão (MSEP) de Mack (1993), somado só até
`target_col[i]` (por padrão, a última coluna do triângulo, ou seja, o
ultimate completo). `f` e `sigma2` já devem vir com a cauda extrapolada
(`extrapolate_factor_tail`, `extrapolate_sigma2_tail`), para que a
incerteza da cauda entre na soma em vez de ser tratada como zero.
Devolve `(mseps_por_safra, msep_total)`.
"""
function mack_mseps(tri::AbstractMatrix{<:Real}, f::Vector{Float64}, sigma2::Vector{Float64};
                     target_col::Union{Nothing, Vector{Int}}=nothing)
    n_cohorts, max_dev = size(tri)
    _, s, n = dev_factors(tri)
    # para colunas sem par observado, a soma s[k] (exposição) é aproximada pela
    # última exposição observada, já que a extrapolação de cauda assume o mesmo
    # tipo de comportamento agregado das colunas vizinhas
    s_full = copy(s)
    last_s = findlast(!isnan, s)
    for k in eachindex(s_full)
        isnan(s_full[k]) && (s_full[k] = last_s === nothing ? 0.0 : s[last_s])
    end

    proj = project_chain_ladder(tri, f)
    k_vec = [last_observed_col(tri, i) for i in 1:n_cohorts]
    tcol = target_col === nothing ? fill(max_dev, n_cohorts) : target_col

    msep_i = fill(0.0, n_cohorts)
    for i in 1:n_cohorts
        k_i = k_vec[i]
        tc = tcol[i]
        (k_i == 0 || k_i >= tc) && continue
        acc = 0.0
        for k in k_i:(tc - 1)
            fk = f[k]
            (fk == 0 || s_full[k] <= 0) && continue
            acc += (sigma2[k] / fk^2) * (1 / proj[i, k] + 1 / s_full[k])
        end
        msep_i[i] = proj[i, tc]^2 * acc
    end

    msep_total = sum(msep_i)
    for i in 1:n_cohorts, ip in 1:n_cohorts
        i >= ip && continue
        k_i, k_ip = k_vec[i], k_vec[ip]
        tc_i, tc_ip = tcol[i], tcol[ip]
        (k_i == 0 || k_ip == 0 || k_i >= tc_i || k_ip >= tc_ip) && continue
        kmin = max(k_i, k_ip)
        tcmin = min(tc_i, tc_ip)
        kmin >= tcmin && continue
        acc = 0.0
        for k in kmin:(tcmin - 1)
            fk = f[k]
            (fk == 0 || s_full[k] <= 0) && continue
            acc += sigma2[k] / fk^2 / s_full[k]
        end
        msep_total += 2 * proj[i, tc_i] * proj[ip, tc_ip] * acc
    end

    return msep_i, msep_total
end

"""
    bootstrap_reserves(tri, f; n_boot=1000, seed=20261102, target_col=nothing,
                        prior_rate=nothing, volume=nothing)

Bootstrap ODP (Renshaw; Verrall, 1998; England; Verrall, 1999/2002):
reamostra resíduos de Pearson para gerar incerteza de parâmetro, e soma
incerteza de processo via distribuição Gama com a dispersão de Pearson
estimada. Projeta até `target_col[i]` (ultimate por padrão). Se
`prior_rate` e `volume` forem passados, cada réplica bootstrap também
gera uma reserva Bornhuetter-Ferguson usando os mesmos fatores
reamostrados, o que dá a esse método um intervalo de incerteza próprio
(ausente na formulação determinística usual de BF), reaproveitando o
mesmo laço de reamostragem. Devolve `(reservas_cl, reservas_bf_ou_nothing, phi)`.
"""
function bootstrap_reserves(tri::AbstractMatrix{<:Real}, f::Vector{Float64};
                             n_boot::Int=1000, seed::Int=20261102,
                             target_col::Union{Nothing, Vector{Int}}=nothing,
                             prior_rate::Union{Nothing, Vector{Float64}}=nothing,
                             volume::Union{Nothing, Vector{Float64}}=nothing)
    n_cohorts, max_dev = size(tri)
    resid, fitted_incr, phi, n_obs, dof = pearson_residuals(tri, f)
    adj = sqrt(n_obs / dof)

    valid_idx = [(i, j) for i in 1:n_cohorts, j in 1:max_dev if !isnan(resid[i, j])]
    valid_resid = [resid[i, j] for (i, j) in valid_idx]

    k_vec = [last_observed_col(tri, i) for i in 1:n_cohorts]
    latest = [k_vec[i] == 0 ? 0.0 : tri[i, k_vec[i]] for i in 1:n_cohorts]
    tcol = target_col === nothing ? fill(max_dev, n_cohorts) : target_col
    com_bf = prior_rate !== nothing && volume !== nothing

    rng = MersenneTwister(seed)
    reservas_cl = fill(0.0, n_boot)
    reservas_bf = com_bf ? fill(0.0, n_boot) : nothing

    for b in 1:n_boot
        pseudo_incr = fill(NaN, n_cohorts, max_dev)
        for (i, j) in valid_idx
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
        f_star_raw, s_star, n_star = dev_factors(pseudo_cum)
        f_star, = extrapolate_factor_tail(f_star_raw, n_star)
        cdf_star = cdf_to_ultimate(f_star)

        total_cl = 0.0
        total_bf = 0.0
        for i in 1:n_cohorts
            k_i = k_vec[i]
            tc = tcol[i]
            (k_i == 0 || k_i >= tc) && continue
            mean_cl = latest[i] * cdf_star[k_i] / cdf_star[tc]
            mu_cl = max(mean_cl - latest[i], 1e-6)
            shape_cl = mu_cl / phi
            ultimate_proc_cl = latest[i] + rand(rng, Gamma(shape_cl, phi))
            total_cl += ultimate_proc_cl - latest[i]

            if com_bf
                # fração esperada de emergência entre k_i e tc, no mesmo CDF reamostrado
                frac = (1 / cdf_star[tc] - 1 / cdf_star[k_i])
                mean_bf = latest[i] + prior_rate[i] * volume[i] * frac
                mu_bf = max(mean_bf - latest[i], 1e-6)
                shape_bf = mu_bf / phi
                ultimate_proc_bf = latest[i] + rand(rng, Gamma(shape_bf, phi))
                total_bf += ultimate_proc_bf - latest[i]
            end
        end
        reservas_cl[b] = total_cl
        com_bf && (reservas_bf[b] = total_bf)
    end

    return reservas_cl, reservas_bf, phi
end

"""
    bornhuetter_ferguson(tri, f, prior_rate, volume; target_col=nothing)

Reserva de Bornhuetter-Ferguson (Bornhuetter; Ferguson, 1972): mistura o
desenvolvimento observado com um `prior_rate` externo (taxa de perda
esperada a priori, independente da própria safra), ponderado pela fração
da perda esperada entre a idade atual e `target_col[i]` (ultimate por
padrão). Devolve a reserva por safra.
"""
function bornhuetter_ferguson(tri::AbstractMatrix{<:Real}, f::Vector{Float64},
                               prior_rate::Vector{Float64}, volume::Vector{Float64};
                               target_col::Union{Nothing, Vector{Int}}=nothing)
    n_cohorts, max_dev = size(tri)
    cdf = cdf_to_ultimate(f)
    tcol = target_col === nothing ? fill(max_dev, n_cohorts) : target_col
    reserva = fill(0.0, n_cohorts)
    for i in 1:n_cohorts
        k_i = last_observed_col(tri, i)
        tc = tcol[i]
        (k_i == 0 || k_i >= tc) && continue
        frac = 1 / cdf[tc] - 1 / cdf[k_i]
        reserva[i] = prior_rate[i] * volume[i] * frac
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
