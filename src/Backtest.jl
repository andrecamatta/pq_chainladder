module Backtest

using Statistics
using ..ChainLadderCore
using ..VintageSim

export truncate_at_valuation, run_backtest_cut, run_rolling_backtest

"""
    truncate_at_valuation(sim, valuation_idx)

Recorta o triângulo completo simulado (`sim.cum`) para o que seria
observável numa data de avaliação `valuation_idx` (índice na série real
do SCR.data). Safras ainda não originadas ficam com todas as colunas
`NaN`. Devolve a matriz truncada (triângulo superior).
"""
function truncate_at_valuation(sim::SimTriangle, valuation_idx::Int)
    tri = fill(NaN, sim.n_cohorts, sim.max_dev)
    for i in 1:sim.n_cohorts, j in 1:sim.max_dev
        if sim.calendar_index[i, j] <= valuation_idx
            tri[i, j] = sim.cum[i, j]
        end
    end
    return tri
end

"""
    run_backtest_cut(sim, valuation_idx, scr_taxa, dilution_factor; n_boot=500)

Roda um corte de backtest: trunca o triângulo na diagonal de
`valuation_idx`, projeta a reserva (perda ainda não emergida das safras
em desenvolvimento) por chain ladder, Mack, bootstrap ODP e
Bornhuetter-Ferguson, e compara com a reserva realizada de verdade
(conhecida porque o triângulo completo foi simulado). O prior do BF usa
a taxa agregada real do SCR.data na época de originação de cada safra
(a mesma regra de calibração usada para gerar a simulação, representando
a expectativa a priori de subscrição), não o desenvolvimento já
observado da própria safra. Por não depender da experiência emergida, o
prior do BF está disponível mesmo na primeira avaliação, sem o problema
de partida a frio de usar só safras internas já maduras.
"""
function run_backtest_cut(sim::SimTriangle, valuation_idx::Int, scr_taxa::Vector{Float64},
                           dilution_factor::Float64; n_boot::Int=500)
    tri = truncate_at_valuation(sim, valuation_idx)
    f, s, n = dev_factors(tri)
    for k in eachindex(f)
        isnan(f[k]) && (f[k] = 1.0)
    end

    k_vec = [last_observed_col(tri, i) for i in 1:sim.n_cohorts]
    ativo(i) = k_vec[i] > 0 && k_vec[i] < sim.max_dev
    proj_cl = project_chain_ladder(tri, f)
    reserva_cl_i = [ativo(i) ? proj_cl[i, sim.max_dev] - tri[i, k_vec[i]] : 0.0
                    for i in 1:sim.n_cohorts]
    reserva_cl = sum(reserva_cl_i)

    msep_i, msep_total, sigma2 = mack_mseps(tri, f)
    se_mack = sqrt(max(msep_total, 0.0))

    reservas_boot, phi = bootstrap_odp(tri, f; n_boot=n_boot, seed=20261102 + valuation_idx)
    boot_mean = mean(reservas_boot)
    boot_sd = std(reservas_boot)
    boot_q = quantile(reservas_boot, [0.025, 0.975])

    prior_rate = fill(0.0, sim.n_cohorts)
    for i in 1:sim.n_cohorts
        orig_idx = sim.start_idx + i - 1
        janela = max(1, orig_idx - 7):orig_idx
        prior_rate[i] = dilution_factor * mean(scr_taxa[janela])
    end
    reserva_bf_i = bornhuetter_ferguson(tri, f, prior_rate, sim.volume)
    reserva_bf = sum(reserva_bf_i)
    maduras = [i for i in 1:sim.n_cohorts if k_vec[i] >= sim.max_dev]

    reserva_real_i = [ativo(i) ? sim.cum[i, sim.max_dev] - tri[i, k_vec[i]] : 0.0
                       for i in 1:sim.n_cohorts]
    reserva_real = sum(reserva_real_i)

    f_stat, p_valor, df_b, df_w = calendar_effect_test(tri, f)

    return (
        valuation_idx = valuation_idx,
        reserva_real = reserva_real,
        reserva_cl = reserva_cl,
        reserva_mack_se = se_mack,
        reserva_boot_mean = boot_mean,
        reserva_boot_sd = boot_sd,
        reserva_boot_q025 = boot_q[1],
        reserva_boot_q975 = boot_q[2],
        reserva_bf = reserva_bf,
        prior_rate_bf = mean(prior_rate),
        erro_cl_pct = 100 * (reserva_cl - reserva_real) / reserva_real,
        erro_bf_pct = 100 * (reserva_bf - reserva_real) / reserva_real,
        erro_boot_pct = 100 * (boot_mean - reserva_real) / reserva_real,
        coverage_mack_95 = (reserva_cl - 1.96 * se_mack <= reserva_real <= reserva_cl + 1.96 * se_mack),
        coverage_boot_95 = (boot_q[1] <= reserva_real <= boot_q[2]),
        cal_f_stat = f_stat,
        cal_p_valor = p_valor,
        n_maduras_prior = length(maduras),
    )
end

function run_rolling_backtest(sim::SimTriangle, valuation_pontos::Vector{Int}, scr_taxa::Vector{Float64},
                               dilution_factor::Float64; n_boot::Int=500)
    return [run_backtest_cut(sim, v, scr_taxa, dilution_factor; n_boot=n_boot) for v in valuation_pontos]
end

end # module
