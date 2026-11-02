import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Statistics
using JSON3
using Dates

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito
using .ChainLadderCredito.ScrCalibration
using .ChainLadderCredito.VintageSim
using .ChainLadderCredito.ChainLadderCore
using .ChainLadderCredito.Backtest

const HERE = @__DIR__
const DATA = joinpath(HERE, "..", "data")
const OUT = joinpath(DATA, "output")
mkpath(OUT)

println("== 1. Carregando série real do SCR.data (crédito pessoal sem consignação, PF) ==")
scr = load_scr_series(joinpath(DATA, "scr_credito_pessoal_sem_consignacao.csv"))
println("Trimestres carregados: ", length(scr.data_base), " (", scr.data_base[1], " a ", scr.data_base[end], ")")
println("Taxa média no período: ", round(100 * mean(scr.taxa), digits=2), "%")
println("Taxa mínima/máxima: ", round(100 * minimum(scr.taxa), digits=2), "% / ", round(100 * maximum(scr.taxa), digits=2), "%")

CSV.write(joinpath(OUT, "scr_serie_real.csv"),
          DataFrame(data_base=scr.data_base, carteira_ativa=scr.carteira_ativa,
                    carteira_inadimplencia=scr.carteira_inadimplencia, taxa=scr.taxa))

println()
println("== 2. Simulando triângulo de safras (proxy calibrado) ==")
N_COHORTS = 24
MAX_DEV = 24
START_IDX = 8
sim = simulate_vintage_triangle(scr.taxa; n_cohorts=N_COHORTS, max_dev=MAX_DEV, start_idx=START_IDX,
                                 dilution_factor=2.2, shape_a=1.6, shape_b=5.0,
                                 base_volume_r=scr.carteira_ativa, origination_share=0.15,
                                 cv_incremento=0.15, seed=20261102)

println("Safras simuladas: ", sim.n_cohorts, " | idade máxima: ", sim.max_dev, " trimestres")
origem_datas = [scr.data_base[sim.start_idx + i - 1] for i in 1:sim.n_cohorts]
println("Safras de ", origem_datas[1], " a ", origem_datas[end])
println("Volume originado total (todas as safras): R\$ ", round(sum(sim.volume) / 1e9, digits=1), " bilhões")
println("Perda final média simulada (ultimate / volume): ",
        round(100 * mean(sim.cum[:, end] ./ sim.volume), digits=2), "%")

df_tri = DataFrame()
for i in 1:sim.n_cohorts
    df_tri[!, Symbol("safra_", i)] = sim.cum[i, :]
end
CSV.write(joinpath(OUT, "triangulo_simulado_cum.csv"), df_tri)

df_calendar = DataFrame(safra=Int[], idade=Int[], indice_calendario=Int[], data_calendario=Date[], cum=Float64[])
for i in 1:sim.n_cohorts, j in 1:sim.max_dev
    push!(df_calendar, (i, j - 1, sim.calendar_index[i, j], scr.data_base[sim.calendar_index[i, j]], sim.cum[i, j]))
end
CSV.write(joinpath(OUT, "triangulo_simulado_long.csv"), df_calendar)

println()
println("== 3. Backtest por truncamento de diagonal ==")
valuation_pontos = [18, 22, 26, 30, 34, 38, 42, 46, 50]
println("Datas de avaliação testadas: ", [scr.data_base[v] for v in valuation_pontos])

resultados = run_rolling_backtest(sim, valuation_pontos, scr.taxa, 2.2; n_boot=1000)

df_bt = DataFrame(
    data_avaliacao = [scr.data_base[r.valuation_idx] for r in resultados],
    reserva_real_bi = [r.reserva_real / 1e9 for r in resultados],
    reserva_cl_bi = [r.reserva_cl / 1e9 for r in resultados],
    reserva_bf_bi = [r.reserva_bf / 1e9 for r in resultados],
    reserva_boot_media_bi = [r.reserva_boot_mean / 1e9 for r in resultados],
    reserva_boot_sd_bi = [r.reserva_boot_sd / 1e9 for r in resultados],
    erro_cl_pct = [r.erro_cl_pct for r in resultados],
    erro_bf_pct = [r.erro_bf_pct for r in resultados],
    erro_boot_pct = [r.erro_boot_pct for r in resultados],
    mack_se_bi = [r.reserva_mack_se / 1e9 for r in resultados],
    cobertura_mack_95 = [r.coverage_mack_95 for r in resultados],
    cobertura_boot_95 = [r.coverage_boot_95 for r in resultados],
    cal_f_stat = [r.cal_f_stat for r in resultados],
    cal_p_valor = [r.cal_p_valor for r in resultados],
    n_maduras_prior_bf = [r.n_maduras_prior for r in resultados],
)
CSV.write(joinpath(OUT, "backtest_resultados.csv"), df_bt)
println(df_bt)

println()
println("== 4. Resumo agregado (todos os cortes) ==")
mae_cl = mean(abs.(df_bt.erro_cl_pct))
mae_bf = mean(abs.(df_bt.erro_bf_pct))
mae_boot = mean(abs.(df_bt.erro_boot_pct))
println("Erro absoluto médio (% da reserva real) -- chain ladder: ", round(mae_cl, digits=1),
        "% | Bornhuetter-Ferguson: ", round(mae_bf, digits=1), "% | bootstrap ODP (média): ", round(mae_boot, digits=1), "%")
println("Cobertura do intervalo de 95% de Mack: ", sum(df_bt.cobertura_mack_95), "/", nrow(df_bt))
println("Cobertura do intervalo de 95% do bootstrap: ", sum(df_bt.cobertura_boot_95), "/", nrow(df_bt))
println("Cortes com efeito de calendário significativo (p<0.05): ", sum(df_bt.cal_p_valor .< 0.05), "/", nrow(df_bt))

vencedor = mae_cl <= mae_bf ? "chain ladder" : "Bornhuetter-Ferguson"
println("Método com menor erro absoluto médio nos cortes testados: ", vencedor)

resumo = Dict(
    "n_cortes" => nrow(df_bt),
    "mae_cl_pct" => mae_cl,
    "mae_bf_pct" => mae_bf,
    "mae_boot_pct" => mae_boot,
    "cobertura_mack_95" => sum(df_bt.cobertura_mack_95),
    "cobertura_boot_95" => sum(df_bt.cobertura_boot_95),
    "cortes_efeito_calendario_sig" => sum(df_bt.cal_p_valor .< 0.05),
    "n_cortes_total" => nrow(df_bt),
    "vencedor" => vencedor,
    "taxa_media_real_pct" => 100 * mean(scr.taxa),
    "taxa_min_real_pct" => 100 * minimum(scr.taxa),
    "taxa_max_real_pct" => 100 * maximum(scr.taxa),
    "ultimate_medio_simulado_pct" => 100 * mean(sim.cum[:, end] ./ sim.volume),
    "volume_total_bi" => sum(sim.volume) / 1e9,
    "safra_inicio" => string(origem_datas[1]),
    "safra_fim" => string(origem_datas[end]),
)
open(joinpath(OUT, "resumo.json"), "w") do io
    JSON3.write(io, resumo)
end

println()
println("Saídas gravadas em ", OUT)
