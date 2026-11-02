module Backtest

using Statistics
using ..ChainLadderCore
using ..VintageSim

export truncate_at_valuation, prior_rate_externo, run_backtest_cut, run_rolling_backtest

"""
    truncate_at_valuation(sim, valuation_idx)

Recorta o triângulo completo simulado (`sim.cum`) para o que seria
observável numa data de avaliação `valuation_idx` (índice na série real
do SCR.data). Safras ainda não originadas, ou idades fora do horizonte
real de calibração (`calendar_index == 0`), ficam `NaN`.
"""
function truncate_at_valuation(sim::SimTriangle, valuation_idx::Int)
    tri = fill(NaN, sim.n_cohorts, sim.max_dev)
    for i in 1:sim.n_cohorts, j in 1:sim.max_dev
        ci = sim.calendar_index[i, j]
        if ci > 0 && ci <= valuation_idx
            tri[i, j] = sim.cum[i, j]
        end
    end
    return tri
end

"""
    prior_rate_externo(sim, scr_taxa, dilution_factor; janela=8, vies=1.0)

Prior a priori do Bornhuetter-Ferguson para cada safra: a taxa real do
SCR.data numa janela móvel de `janela` trimestres terminando no
trimestre de originação da safra (não no trimestre de avaliação, então
não lê a conjuntura corrente, só a conjuntura que já era conhecida
quando a safra foi concedida), multiplicada por `dilution_factor` (o
mesmo fator usado para gerar a simulação, o que torna este prior bem
especificado por construção) e por `vies` (1,0 = prior bem especificado;
valores diferentes simulam um prior mal calibrado, para medir
sensibilidade).
"""
function prior_rate_externo(sim::SimTriangle, scr_taxa::Vector{Float64},
                             dilution_factor::Float64; janela::Int=8, vies::Float64=1.0)
    prior = fill(0.0, sim.n_cohorts)
    for i in 1:sim.n_cohorts
        orig_idx = sim.start_idx + i - 1
        janela_idx = max(1, orig_idx - janela + 1):orig_idx
        prior[i] = vies * dilution_factor * mean(scr_taxa[janela_idx])
    end
    return prior
end

"""
    run_backtest_cut(sim, valuation_idx, scr_taxa, dilution_factor; n_boot=1000, prior_vies=1.0)

Roda um corte de backtest: trunca o triângulo na diagonal de
`valuation_idx`, extrapola a cauda de fatores e de σ² além da última
idade observada, projeta a reserva por chain ladder, Mack, bootstrap ODP
e Bornhuetter-Ferguson, e compara com a reserva realizada. A comparação
não vai até a maturidade completa (`sim.max_dev`): vai só até a idade
`sim.max_age_within_horizon[i]`, a última para a qual existe dado real
de calendário simulável para aquela safra. Para safras originadas perto
do fim da série real, essa idade é menor que `sim.max_dev`, e o déficit
de trimestres até a maturidade completa é reportado, não escondido.
`prior_vies` controla o viés do prior do BF (1,0 = bem especificado).
"""
function run_backtest_cut(sim::SimTriangle, valuation_idx::Int, scr_taxa::Vector{Float64},
                           dilution_factor::Float64; n_boot::Int=1000, prior_vies::Float64=1.0)
    tri = truncate_at_valuation(sim, valuation_idx)
    f_raw, s, n = dev_factors(tri)
    f, tail_coef = extrapolate_factor_tail(f_raw, n)
    sigma2_raw = observed_sigma2(tri, f_raw, n)
    sigma2, sigma2_tail_coef = extrapolate_sigma2_tail(sigma2_raw, n)
    n_colunas_extrapoladas = count(isnan, f_raw)

    k_vec = [last_observed_col(tri, i) for i in 1:sim.n_cohorts]
    tcol = sim.max_age_within_horizon
    ativo(i) = k_vec[i] > 0 && k_vec[i] < tcol[i]
    n_ativas = count(ativo, 1:sim.n_cohorts)
    n_safras_jovens = count(i -> ativo(i) && k_vec[i] <= 4, 1:sim.n_cohorts)

    proj_cl = project_chain_ladder(tri, f)
    reserva_cl_i = [ativo(i) ? proj_cl[i, tcol[i]] - tri[i, k_vec[i]] : 0.0 for i in 1:sim.n_cohorts]
    reserva_cl = sum(reserva_cl_i)

    msep_i, msep_total = mack_mseps(tri, f, sigma2; target_col=tcol)
    se_mack = sqrt(max(msep_total, 0.0))

    prior_rate = prior_rate_externo(sim, scr_taxa, dilution_factor; vies=prior_vies)
    reserva_bf_i = bornhuetter_ferguson(tri, f, prior_rate, sim.volume; target_col=tcol)
    reserva_bf = sum(reserva_bf_i)

    reservas_boot_cl, reservas_boot_bf, phi = bootstrap_reserves(
        tri, f; n_boot=n_boot, seed=20261102 + valuation_idx, target_col=tcol,
        prior_rate=prior_rate, volume=sim.volume)
    boot_cl_mean = mean(reservas_boot_cl)
    boot_cl_q = quantile(reservas_boot_cl, [0.025, 0.975])
    boot_bf_mean = mean(reservas_boot_bf)
    boot_bf_q = quantile(reservas_boot_bf, [0.025, 0.975])

    reserva_real_i = [ativo(i) ? sim.cum[i, tcol[i]] - tri[i, k_vec[i]] : 0.0 for i in 1:sim.n_cohorts]
    reserva_real = sum(reserva_real_i)

    # déficit de maturidade: para as safras ativas, quantos trimestres faltam
    # entre o alvo de comparação (tcol) e a maturidade completa (max_dev)
    deficits = [ativo(i) ? sim.max_dev - tcol[i] : 0 for i in 1:sim.n_cohorts]
    deficit_max = isempty(deficits) ? 0 : maximum(deficits)
    n_safras_truncadas_horizonte = count(>(0), deficits)

    f_stat, p_valor, df_b, df_w = calendar_effect_test(tri, f)

    erro_cl_pct = 100 * (reserva_cl - reserva_real) / reserva_real
    erro_bf_pct = 100 * (reserva_bf - reserva_real) / reserva_real
    erro_boot_cl_pct = 100 * (boot_cl_mean - reserva_real) / reserva_real
    erro_boot_bf_pct = 100 * (boot_bf_mean - reserva_real) / reserva_real

    return (
        valuation_idx = valuation_idx,
        n_safras_ativas = n_ativas,
        n_safras_jovens = n_safras_jovens,
        n_colunas_extrapoladas = n_colunas_extrapoladas,
        deficit_maturidade_max = deficit_max,
        n_safras_truncadas_horizonte = n_safras_truncadas_horizonte,
        reserva_real = reserva_real,
        reserva_cl = reserva_cl,
        reserva_mack_se = se_mack,
        reserva_boot_cl_mean = boot_cl_mean,
        reserva_boot_cl_q025 = boot_cl_q[1],
        reserva_boot_cl_q975 = boot_cl_q[2],
        reserva_bf = reserva_bf,
        reserva_boot_bf_mean = boot_bf_mean,
        reserva_boot_bf_q025 = boot_bf_q[1],
        reserva_boot_bf_q975 = boot_bf_q[2],
        prior_rate_bf = mean(prior_rate),
        erro_cl_pct = erro_cl_pct,
        erro_bf_pct = erro_bf_pct,
        erro_boot_cl_pct = erro_boot_cl_pct,
        erro_boot_bf_pct = erro_boot_bf_pct,
        sub_cl = erro_cl_pct < 0,
        sub_bf = erro_bf_pct < 0,
        coverage_mack_95 = (reserva_cl - 1.96 * se_mack <= reserva_real <= reserva_cl + 1.96 * se_mack),
        coverage_boot_cl_95 = (boot_cl_q[1] <= reserva_real <= boot_cl_q[2]),
        coverage_boot_bf_95 = (boot_bf_q[1] <= reserva_real <= boot_bf_q[2]),
        cal_f_stat = f_stat,
        cal_p_valor = p_valor,
    )
end

function run_rolling_backtest(sim::SimTriangle, valuation_pontos::Vector{Int}, scr_taxa::Vector{Float64},
                               dilution_factor::Float64; n_boot::Int=1000, prior_vies::Float64=1.0)
    return [run_backtest_cut(sim, v, scr_taxa, dilution_factor; n_boot=n_boot, prior_vies=prior_vies)
            for v in valuation_pontos]
end

end # module
