import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Statistics
using JSON3
using Dates
using Random
using LinearAlgebra

# A seção 5 paraleliza por semente com Threads.@threads, e cada semente chama
# fit_apc_calendar (IRLS, resolve `\` via QR do LAPACK) muitas vezes por réplica. BLAS
# multi-thread chamado de dentro de várias threads Julia simultâneas (threading
# aninhado) é instável -- pode travar ou (como ocorreu numa execução anterior desta
# fase, terminada por SIGTERM externo em pleno GC/BLAS) tornar o diagnóstico de uma
# interrupção externa mais difícil de distinguir de um crash genuíno. Fixar BLAS em 1
# thread aqui é a prática padrão ao combinar Threads.@threads com álgebra linear.
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito
using .ChainLadderCredito.ScrCalibration
using .ChainLadderCredito.VintageSim
using .ChainLadderCredito.ChainLadderCore
using .ChainLadderCredito.Backtest
using .ChainLadderCredito.ApcModel

const HERE = @__DIR__
const DATA = joinpath(HERE, "..", "data")
const RESULTS = joinpath(HERE, "..", "results")
mkpath(RESULTS)

println("== 0. Carregando série real e reconstruindo a simulação de produção ==")
scr = load_scr_series(joinpath(DATA, "scr_credito_pessoal_sem_consignacao.csv"))
MAX_DEV = 24
START_IDX = 8
idx_fim = findfirst(==(Date(2024, 12, 31)), scr.data_base)
N_COHORTS = idx_fim - START_IDX + 1
VP = [18, 22, 26, 30, 34, 38, 42, 46, 50]  # mesmos 9 cortes do pipeline de produção

sim = simulate_vintage_triangle(scr.taxa; n_cohorts=N_COHORTS, max_dev=MAX_DEV, start_idx=START_IDX,
                                 dilution_factor=2.2, shape_a=1.6, shape_b=5.0,
                                 base_volume_r=scr.carteira_ativa, origination_share=0.15,
                                 cv_incremento=0.15, seed=20261102)

# fator de calendário: variação homóloga (4 trimestres) da série real, padronizada --
# a mesma transformação usada em VintageSim.simulate_multi_segment_triangles (seção 3
# do plano, caminho (c)). NÃO é o multiplicador de nível (scr_taxa/mean(scr_taxa)) que
# gerou a simulação de produção -- ver nota na seção 1 abaixo.
Z = padronizar(fator_calendario_bcb(scr.taxa))

println("Taxa real -- média: ", round(100 * mean(scr.taxa), digits=2), "%")
println("Safras simuladas: ", sim.n_cohorts, " | idade máxima: ", sim.max_dev)

# ---------------------------------------------------------------------------
println()
println("== 1. Diagnóstico de diagonal ANTES/DEPOIS do termo de calendário ==")
println("(triângulo COMPLETO simulado, não um corte truncado -- é um diagnóstico sobre")
println("o dado inteiro, não uma previsão; usa o fator derivado só da série do BCB, não")
println("o multiplicador de nível que de fato gerou a simulação -- por isso o termo de")
println("calendário não precisa 'ganhar' por construção.)")

f_raw_full, s_full, n_full = dev_factors(sim.cum)
f_full, = extrapolate_factor_tail(f_raw_full, n_full)
resid_antes, = pearson_residuals(sim.cum, f_full)
fstat_antes, pval_antes, dfb_antes, dfw_antes, frac_antes =
    diagonal_anova(sim.n_cohorts, sim.max_dev, resid_antes)

fit_full = fit_apc_calendar(sim.cum, sim.calendar_index, Z)
fstat_depois, pval_depois, dfb_depois, dfw_depois, frac_depois =
    diagonal_anova(sim.n_cohorts, sim.max_dev, fit_full.resid)

println("Antes  (idade+safra, chain ladder clássico): F=", round(fstat_antes, digits=2),
        " p=", pval_antes, " frac_sistemática=", round(frac_antes, digits=3))
println("Depois (idade+safra+β·fator_BCB):            F=", round(fstat_depois, digits=2),
        " p=", pval_depois, " frac_sistemática=", round(frac_depois, digits=3))
println("beta_hat=", round(fit_full.beta, digits=4), " (se=", round(fit_full.se_beta, digits=4), ")")

df_diag_antes_depois = DataFrame(
    etapa=["antes_idade_safra", "depois_idade_safra_calendario"],
    f_stat=[fstat_antes, fstat_depois],
    p_valor=[pval_antes, pval_depois],
    df_entre=[dfb_antes, dfb_depois],
    df_dentro=[dfw_antes, dfw_depois],
    frac_sistematica=[frac_antes, frac_depois],
    beta_hat=[NaN, fit_full.beta],
    se_beta=[NaN, fit_full.se_beta],
)
CSV.write(joinpath(RESULTS, "apc_diagonal_antes_depois.csv"), df_diag_antes_depois)

# ---------------------------------------------------------------------------
println()
println("== 2. Recuperação do efeito de período LIVRE (só do triângulo) vs. série do BCB ==")
println("(exercício de recuperação declarado: a 'verdade' simulada já é construída a")
println("partir da série real, então correlação alta aqui é checagem de consistência,")
println("não descoberta -- ver plano de reenquadramento, riscos (i)/(ii).)")

diag_ids, delta_hat, n_por_diag = estimate_free_period_effects(sim.cum, f_full)
indice_calendario = START_IDX .+ diag_ids .- 2
validos = [1 <= ci <= length(Z) && !isnan(Z[ci]) for ci in indice_calendario]
data_calendario = [1 <= ci <= length(scr.data_base) ? scr.data_base[ci] : missing for ci in indice_calendario]
z_bcb = [1 <= ci <= length(Z) ? Z[ci] : NaN for ci in indice_calendario]

cor_recuperacao = cor(delta_hat[validos], z_bcb[validos])
println("Diagonais estimadas: ", length(diag_ids), " | correlação δ̂ vs. fator do BCB: ",
        round(cor_recuperacao, digits=3))

df_periodo_livre = DataFrame(
    diagonal_id=diag_ids,
    indice_calendario=indice_calendario,
    data_calendario=data_calendario,
    delta_hat=delta_hat,
    n_celulas=n_por_diag,
    fator_bcb_padronizado=z_bcb,
)
CSV.write(joinpath(RESULTS, "apc_recuperacao_periodo_livre.csv"), df_periodo_livre)

# ---------------------------------------------------------------------------
println()
println("== 3. Decomposição do desvio de dezembro de 2021 ==")
V_2021 = 38
@assert scr.data_base[V_2021] == Date(2021, 12, 31)
dec = decompose_calendar_deviation(sim, V_2021)
cuts_ref = run_rolling_backtest(sim, [V_2021], scr.taxa, 2.2; n_boot=1000, prior_vies=1.0)
r2021 = cuts_ref[1]
println("reserva_real=", round(r2021.reserva_real / 1e9, digits=2), " bi | reserva_cl=",
        round(r2021.reserva_cl / 1e9, digits=2), " bi | erro_cl_pct=", round(r2021.erro_cl_pct, digits=1), "%")
println("Soma dos resíduos célula-a-célula no leque de previsão (realizado - ajustado): ",
        round(dec.erro_total / 1e9, digits=3), " bi (deve bater com reserva_real - reserva_cl = ",
        round((r2021.reserva_real - r2021.reserva_cl) / 1e9, digits=3), " bi)")

por_diag = Dict{Int, Float64}()
cont_diag = Dict{Int, Int}()
for k in eachindex(dec.residuo_dolar)
    ci = dec.indice_calendario[k]
    por_diag[ci] = get(por_diag, ci, 0.0) + dec.residuo_dolar[k]
    cont_diag[ci] = get(cont_diag, ci, 0) + 1
end
ids_ordenados = sort(collect(keys(por_diag)); by=ci -> -abs(por_diag[ci]))
df_dez2021 = DataFrame(
    indice_calendario=ids_ordenados,
    data_calendario=[scr.data_base[ci] for ci in ids_ordenados],
    contribuicao_bi=[por_diag[ci] / 1e9 for ci in ids_ordenados],
    n_celulas=[cont_diag[ci] for ci in ids_ordenados],
)
CSV.write(joinpath(RESULTS, "apc_decomposicao_dez2021.csv"), df_dez2021)

# matriz de resíduos em R$ para a ANOVA de diagonal, restrita ao leque de previsão
resid_matrix_dez2021 = fill(NaN, sim.n_cohorts, sim.max_dev)
for k in eachindex(dec.residuo_dolar)
    resid_matrix_dez2021[dec.cohort[k], dec.idade[k]] = dec.residuo_dolar[k]
end
fstat_2021, pval_2021, dfb_2021, dfw_2021, frac_2021 =
    diagonal_anova(sim.n_cohorts, sim.max_dev, resid_matrix_dez2021)
soma_idiossincratica = sum(dec.residuo_dolar[k] - por_diag[dec.indice_calendario[k]] / cont_diag[dec.indice_calendario[k]]
                            for k in eachindex(dec.residuo_dolar))
println("Checagem de cancelamento (deve ser ~0): soma dos desvios intra-diagonal = ",
        round(soma_idiossincratica / 1e9, digits=6), " bi")
println("Fração da VARIÂNCIA célula-a-célula do resíduo atribuível à diagonal (não à soma): ",
        round(frac_2021, digits=3), " (F=", round(fstat_2021, digits=2), ", p=", pval_2021, ")")
println("As ", min(5, length(ids_ordenados)), " diagonais de maior contribuição absoluta:")
for ci in ids_ordenados[1:min(5, length(ids_ordenados))]
    println("  ", scr.data_base[ci], ": ", round(por_diag[ci] / 1e9, digits=3), " bi (n=", cont_diag[ci], " safras)")
end

# ---------------------------------------------------------------------------
println()
println("== 4. Cobertura do IC95% com/sem termo de calendário -- 9 cortes (1 semente) ==")
println("(Fase 1b: montecarlo_apc_reserve agora reestima idade+safra+beta por réplica")
println("(bootstrap de resíduos de Pearson) e NUNCA usa o fator de calendário realizado")
println("depois do corte dentro do Monte Carlo -- o valor futuro é simulado por um AR(1)")
println("ajustado só com dado até o corte (ApcModel.fit_ar1_fator/simulate_fator_future).")
println("'reserva_apc_bi'/'erro_apc_pct' continuam usando o fator REALIZADO (quase-nowcast,")
println("ver docstring de montecarlo_apc_reserve) só como ponto determinístico; o intervalo")
println("de cobertura nunca usa esse valor realizado.)")

cuts9 = run_rolling_backtest(sim, VP, scr.taxa, 2.2; n_boot=1000, prior_vies=1.0)
rows9 = NamedTuple[]
for (idx, v) in enumerate(VP)
    tri = truncate_at_valuation(sim, v)
    k_vec = [last_observed_col(tri, i) for i in 1:sim.n_cohorts]
    tcol = sim.max_age_within_horizon
    r = cuts9[idx]
    local fit, reservas, pontual, cobre_apc, beta_hat, se_beta, erro_apc_pct, largura_apc
    try
        fit = fit_apc_calendar(tri, sim.calendar_index, Z)
        reservas, pontual = montecarlo_apc_reserve(tri, sim.calendar_index, Z, v, k_vec, tcol;
                                                     n_boot=1000, seed=20261102 + v)
        q = quantile(reservas, [0.025, 0.975])
        cobre_apc = q[1] <= r.reserva_real <= q[2]
        largura_apc = q[2] - q[1]
        beta_hat, se_beta = fit.beta, fit.se_beta
        erro_apc_pct = 100 * (pontual - r.reserva_real) / r.reserva_real
        push!(rows9, (data=scr.data_base[v], reserva_real_bi=r.reserva_real / 1e9,
                       reserva_cl_bi=r.reserva_cl / 1e9, reserva_apc_bi=pontual / 1e9,
                       erro_cl_pct=r.erro_cl_pct, erro_apc_pct=erro_apc_pct,
                       cobertura_mack_95=r.coverage_mack_95, cobertura_boot_cl_95=r.coverage_boot_cl_95,
                       cobertura_apc_95=cobre_apc,
                       largura_mack_bi=2 * 1.96 * r.reserva_mack_se / 1e9,
                       largura_boot_cl_bi=(r.reserva_boot_cl_q975 - r.reserva_boot_cl_q025) / 1e9,
                       largura_apc_bi=largura_apc / 1e9,
                       beta_hat=beta_hat, se_beta=se_beta,
                       n_obs_fit=fit.n_obs, ajuste_ok=true))
    catch e
        push!(rows9, (data=scr.data_base[v], reserva_real_bi=r.reserva_real / 1e9,
                       reserva_cl_bi=r.reserva_cl / 1e9, reserva_apc_bi=NaN,
                       erro_cl_pct=r.erro_cl_pct, erro_apc_pct=NaN,
                       cobertura_mack_95=r.coverage_mack_95, cobertura_boot_cl_95=r.coverage_boot_cl_95,
                       cobertura_apc_95=false,
                       largura_mack_bi=2 * 1.96 * r.reserva_mack_se / 1e9,
                       largura_boot_cl_bi=(r.reserva_boot_cl_q975 - r.reserva_boot_cl_q025) / 1e9,
                       largura_apc_bi=NaN,
                       beta_hat=NaN, se_beta=NaN,
                       n_obs_fit=0, ajuste_ok=false))
    end
end
df_9cortes = DataFrame(rows9)
CSV.write(joinpath(RESULTS, "apc_cobertura_9cortes.csv"), df_9cortes)
println(df_9cortes)
println("Cobertura (1 semente) -- Mack: ", sum(df_9cortes.cobertura_mack_95), "/9 | bootstrap CL: ",
        sum(df_9cortes.cobertura_boot_cl_95), "/9 | modelo (c), Monte Carlo com reestimação de parâmetro: ",
        sum(df_9cortes.cobertura_apc_95), "/9")
println("Largura média do IC95% (bi) -- Mack: ", round(mean(df_9cortes.largura_mack_bi), digits=2),
        " | bootstrap CL: ", round(mean(df_9cortes.largura_boot_cl_bi), digits=2),
        " | modelo (c): ", round(mean(filter(!isnan, df_9cortes.largura_apc_bi)), digits=2))

# ---------------------------------------------------------------------------
println()
println("== 5. Experimento de 200 sementes: cobertura média com/sem termo de calendário ==")
println("(n_boot do modelo (c) reduzido para 150 réplicas por corte/semente -- em vez de")
println("500 na Fase 1 -- porque a Fase 1b reestima idade+safra+beta por réplica (ajuste")
println("IRLS completo), não só soma ruído de processo; custo computacional declarado, não")
println("escondido. Paralelizado por semente em Threads.nthreads()=", Threads.nthreads(),
        " threads.)")
NSEEDS = 200
N_BOOT_200 = 150
cobertura_mack = zeros(NSEEDS, length(VP))
cobertura_bootcl = zeros(NSEEDS, length(VP))
cobertura_apc = fill(NaN, NSEEDS, length(VP))
largura_apc = fill(NaN, NSEEDS, length(VP))

Threads.@threads for sd in 1:NSEEDS
    sim_s = simulate_vintage_triangle(scr.taxa; n_cohorts=N_COHORTS, max_dev=MAX_DEV, start_idx=START_IDX,
                                       dilution_factor=2.2, shape_a=1.6, shape_b=5.0,
                                       base_volume_r=scr.carteira_ativa, origination_share=0.15,
                                       cv_incremento=0.15, seed=1000 + sd)
    cuts_s = run_rolling_backtest(sim_s, VP, scr.taxa, 2.2; n_boot=1000, prior_vies=1.0)
    for (k, v) in enumerate(VP)
        r = cuts_s[k]
        cobertura_mack[sd, k] = r.coverage_mack_95
        cobertura_bootcl[sd, k] = r.coverage_boot_cl_95
        tri = truncate_at_valuation(sim_s, v)
        k_vec = [last_observed_col(tri, i) for i in 1:sim_s.n_cohorts]
        tcol = sim_s.max_age_within_horizon
        try
            reservas_s, = montecarlo_apc_reserve(tri, sim_s.calendar_index, Z, v, k_vec, tcol;
                                                   n_boot=N_BOOT_200, seed=20261102 + v + 1000 * sd)
            q_s = quantile(reservas_s, [0.025, 0.975])
            cobertura_apc[sd, k] = q_s[1] <= r.reserva_real <= q_s[2]
            largura_apc[sd, k] = q_s[2] - q_s[1]
        catch
            cobertura_apc[sd, k] = NaN
        end
    end
end
ajuste_falhou = [count(isnan, cobertura_apc[:, k]) for k in eachindex(VP)]

df_200 = DataFrame(
    data=[scr.data_base[v] for v in VP],
    cobertura_mack_media=[mean(cobertura_mack[:, k]) for k in eachindex(VP)],
    cobertura_boot_cl_media=[mean(cobertura_bootcl[:, k]) for k in eachindex(VP)],
    cobertura_apc_media=[mean(filter(!isnan, cobertura_apc[:, k])) for k in eachindex(VP)],
    largura_apc_media_bi=[mean(filter(!isnan, largura_apc[:, k])) / 1e9 for k in eachindex(VP)],
    n_sementes_ajuste_falhou=ajuste_falhou,
)
CSV.write(joinpath(RESULTS, "apc_cobertura_200sementes.csv"), df_200)
println(df_200)
println("Médias agregadas (todas as datas, 200 sementes) -- Mack: ", round(mean(df_200.cobertura_mack_media), digits=3),
        " | bootstrap CL: ", round(mean(df_200.cobertura_boot_cl_media), digits=3),
        " | modelo (c): ", round(mean(df_200.cobertura_apc_media), digits=3))

# ---------------------------------------------------------------------------
println()
println("== 6. Multi-segmento: recuperação de β_s (paralelo Vasicek/IRB) ==")
BETAS_TRUE = [0.0, 0.25, 0.5, 0.75, 1.0]
NSEEDS_SEG = 50
beta_hats = zeros(NSEEDS_SEG, length(BETAS_TRUE))
frac_classica = zeros(NSEEDS_SEG, length(BETAS_TRUE))

for sd in 1:NSEEDS_SEG
    segs, z_seg = simulate_multi_segment_triangles(scr.taxa; betas=BETAS_TRUE, n_cohorts=N_COHORTS,
        max_dev=MAX_DEV, start_idx=START_IDX, base_volume_r=scr.carteira_ativa,
        origination_share=0.15, cv_incremento=0.15, seed=2000 + sd)
    for (s, beta_true) in enumerate(BETAS_TRUE)
        seg = segs[s]
        fit_s = fit_apc_calendar(seg.cum, seg.calendar_index, z_seg)
        beta_hats[sd, s] = fit_s.beta
        f_raw_s, _, n_s = dev_factors(seg.cum)
        f_s, = extrapolate_factor_tail(f_raw_s, n_s)
        resid_s, = pearson_residuals(seg.cum, f_s)
        _, _, _, _, frac_s = diagonal_anova(seg.n_cohorts, seg.max_dev, resid_s)
        frac_classica[sd, s] = frac_s
    end
end

df_betas = DataFrame(
    beta_verdadeiro=BETAS_TRUE,
    beta_verdadeiro_ao_quadrado=BETAS_TRUE .^ 2,
    beta_hat_medio=[mean(beta_hats[:, s]) for s in eachindex(BETAS_TRUE)],
    beta_hat_dp=[std(beta_hats[:, s]) for s in eachindex(BETAS_TRUE)],
    frac_sistematica_anova_media=[mean(frac_classica[:, s]) for s in eachindex(BETAS_TRUE)],
    frac_sistematica_anova_dp=[std(frac_classica[:, s]) for s in eachindex(BETAS_TRUE)],
    n_sementes=fill(NSEEDS_SEG, length(BETAS_TRUE)),
)
CSV.write(joinpath(RESULTS, "apc_beta_segmentos.csv"), df_betas)
println(df_betas)

# ---------------------------------------------------------------------------
println()
println("== Gravando resumo agregado ==")
resumo = Dict(
    "diagonal_f_antes" => fstat_antes,
    "diagonal_p_antes" => pval_antes,
    "diagonal_frac_sistematica_antes" => frac_antes,
    "diagonal_f_depois" => fstat_depois,
    "diagonal_p_depois" => pval_depois,
    "diagonal_frac_sistematica_depois" => frac_depois,
    "beta_hat_full" => fit_full.beta,
    "se_beta_full" => fit_full.se_beta,
    "correlacao_periodo_livre_vs_bcb" => cor_recuperacao,
    "dez2021_reserva_real_bi" => r2021.reserva_real / 1e9,
    "dez2021_reserva_cl_bi" => r2021.reserva_cl / 1e9,
    "dez2021_erro_cl_pct" => r2021.erro_cl_pct,
    "dez2021_frac_sistematica_variancia" => frac_2021,
    "dez2021_cancelamento_idiossincratico_bi" => soma_idiossincratica / 1e9,
    "cobertura_9cortes_mack" => sum(df_9cortes.cobertura_mack_95),
    "cobertura_9cortes_boot_cl" => sum(df_9cortes.cobertura_boot_cl_95),
    "cobertura_9cortes_apc" => sum(df_9cortes.cobertura_apc_95),
    "cobertura_200sem_mack_media" => mean(df_200.cobertura_mack_media),
    "cobertura_200sem_boot_cl_media" => mean(df_200.cobertura_boot_cl_media),
    "cobertura_200sem_apc_media" => mean(df_200.cobertura_apc_media),
    "largura_9cortes_mack_media_bi" => mean(df_9cortes.largura_mack_bi),
    "largura_9cortes_boot_cl_media_bi" => mean(df_9cortes.largura_boot_cl_bi),
    "largura_9cortes_apc_media_bi" => mean(filter(!isnan, df_9cortes.largura_apc_bi)),
    "largura_200sem_apc_media_bi" => mean(df_200.largura_apc_media_bi),
    "n_boot_200sem_apc" => N_BOOT_200,
    "n_sementes_200" => NSEEDS,
    "n_sementes_segmentos" => NSEEDS_SEG,
    "betas_verdadeiros" => BETAS_TRUE,
    "beta_hat_medio" => df_betas.beta_hat_medio,
    "beta_hat_dp" => df_betas.beta_hat_dp,
)
open(joinpath(RESULTS, "apc_resumo.json"), "w") do io
    JSON3.write(io, resumo)
end

println()
println("Saídas gravadas em ", RESULTS)
