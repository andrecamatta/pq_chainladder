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

println()
println("== 2. Simulando triângulo de safras (proxy, originação contínua até 2024-Q4) ==")
MAX_DEV = 24
START_IDX = 8
origem_fim = Date(2024, 12, 31)
idx_fim = findfirst(==(origem_fim), scr.data_base)
@assert idx_fim !== nothing "Data de fim de originação não encontrada na série real"
N_COHORTS = idx_fim - START_IDX + 1

sim = simulate_vintage_triangle(scr.taxa; n_cohorts=N_COHORTS, max_dev=MAX_DEV, start_idx=START_IDX,
                                 dilution_factor=2.2, shape_a=1.6, shape_b=5.0,
                                 base_volume_r=scr.carteira_ativa, origination_share=0.15,
                                 cv_incremento=0.15, seed=20261102)

origem_datas = [scr.data_base[sim.start_idx + i - 1] for i in 1:sim.n_cohorts]
println("Safras simuladas: ", sim.n_cohorts, " (originação trimestral contínua, ", origem_datas[1], " a ", origem_datas[end], ")")
println("Idade máxima alvo (maturidade completa): ", sim.max_dev, " trimestres")
println("Horizonte real de calibração (último trimestre da série real usada): ", scr.data_base[sim.horizon_idx])
n_safras_truncadas = count(<(sim.max_dev), sim.max_age_within_horizon)
println("Safras cuja maturidade completa não cabe no horizonte real: ", n_safras_truncadas, " de ", sim.n_cohorts,
        " (déficit máximo: ", sim.max_dev - minimum(sim.max_age_within_horizon), " trimestres, na safra mais recente)")
println("Volume originado total (todas as safras): R\$ ", round(sum(sim.volume) / 1e9, digits=1), " bilhões")
println("ATENÇÃO: dilution_factor, shape_a, shape_b, cv_incremento e origination_share são hipóteses")
println("assumidas para a simulação, modulada pela série real de inadimplência, não calibradas nela.")

df_tri = DataFrame()
for i in 1:sim.n_cohorts
    df_tri[!, Symbol("safra_", i)] = sim.cum[i, :]
end
CSV.write(joinpath(OUT, "triangulo_simulado_cum.csv"), df_tri)

df_calendar = DataFrame(safra=Int[], idade=Int[], indice_calendario=Int[], data_calendario=Union{Date,Missing}[],
                         cum=Union{Float64,Missing}[], volume_safra=Float64[])
for i in 1:sim.n_cohorts, j in 1:sim.max_dev
    ci = sim.calendar_index[i, j]
    data_cal = ci > 0 ? scr.data_base[ci] : missing
    cum_val = ci > 0 ? sim.cum[i, j] : missing
    push!(df_calendar, (i, j - 1, ci, data_cal, cum_val, sim.volume[i]))
end
CSV.write(joinpath(OUT, "triangulo_simulado_long.csv"), df_calendar)

println()
println("== 3. Backtest por truncamento de diagonal (prior bem especificado) ==")
valuation_pontos = [18, 22, 26, 30, 34, 38, 42, 46, 50]
println("Datas de avaliação testadas: ", [scr.data_base[v] for v in valuation_pontos])

resultados = run_rolling_backtest(sim, valuation_pontos, scr.taxa, 2.2; n_boot=1000, prior_vies=1.0)

function monta_df(resultados)
    DataFrame(
        data_avaliacao = [scr.data_base[r.valuation_idx] for r in resultados],
        n_safras_ativas = [r.n_safras_ativas for r in resultados],
        n_safras_jovens = [r.n_safras_jovens for r in resultados],
        n_colunas_extrapoladas = [r.n_colunas_extrapoladas for r in resultados],
        deficit_maturidade_max = [r.deficit_maturidade_max for r in resultados],
        n_safras_truncadas_horizonte = [r.n_safras_truncadas_horizonte for r in resultados],
        idade_alvo_minima = [r.idade_alvo_minima for r in resultados],
        idade_alvo_maxima = [r.idade_alvo_maxima for r in resultados],
        reserva_real_bi = [r.reserva_real / 1e9 for r in resultados],
        reserva_cl_bi = [r.reserva_cl / 1e9 for r in resultados],
        reserva_bf_bi = [r.reserva_bf / 1e9 for r in resultados],
        reserva_boot_cl_media_bi = [r.reserva_boot_cl_mean / 1e9 for r in resultados],
        reserva_boot_bf_media_bi = [r.reserva_boot_bf_mean / 1e9 for r in resultados],
        erro_cl_pct = [r.erro_cl_pct for r in resultados],
        erro_bf_pct = [r.erro_bf_pct for r in resultados],
        erro_boot_cl_pct = [r.erro_boot_cl_pct for r in resultados],
        erro_boot_bf_pct = [r.erro_boot_bf_pct for r in resultados],
        sub_cl = [r.sub_cl for r in resultados],
        sub_bf = [r.sub_bf for r in resultados],
        mack_se_bi = [r.reserva_mack_se / 1e9 for r in resultados],
        cobertura_mack_95 = [r.coverage_mack_95 for r in resultados],
        cobertura_boot_cl_95 = [r.coverage_boot_cl_95 for r in resultados],
        cobertura_boot_bf_95 = [r.coverage_boot_bf_95 for r in resultados],
        cal_f_stat = [r.cal_f_stat for r in resultados],
        cal_p_valor = [r.cal_p_valor for r in resultados],
    )
end

df_bt = monta_df(resultados)
CSV.write(joinpath(OUT, "backtest_resultados.csv"), df_bt)
println(df_bt)

function resumo_metricas(df_bt; rotulo="")
    mae_cl = mean(abs.(df_bt.erro_cl_pct))
    mae_bf = mean(abs.(df_bt.erro_bf_pct))
    med_cl = median(abs.(df_bt.erro_cl_pct))
    med_bf = median(abs.(df_bt.erro_bf_pct))
    w = df_bt.reserva_real_bi
    mae_cl_pond = sum(w .* abs.(df_bt.erro_cl_pct)) / sum(w)
    mae_bf_pond = sum(w .* abs.(df_bt.erro_bf_pct)) / sum(w)
    println()
    println("-- Resumo$(isempty(rotulo) ? "" : " ($rotulo)") --")
    println("Erro absoluto médio simples -- CL: ", round(mae_cl, digits=2), "% | BF: ", round(mae_bf, digits=2), "%")
    println("Mediana do erro absoluto -- CL: ", round(med_cl, digits=2), "% | BF: ", round(med_bf, digits=2), "%")
    println("Erro absoluto médio ponderado pela reserva -- CL: ", round(mae_cl_pond, digits=2), "% | BF: ", round(mae_bf_pond, digits=2), "%")
    println("Subestimação -- CL: ", sum(df_bt.sub_cl), "/", nrow(df_bt), " | BF: ", sum(df_bt.sub_bf), "/", nrow(df_bt))
    println("Cortes em que CL tem menor |erro| que BF: ", sum(abs.(df_bt.erro_cl_pct) .< abs.(df_bt.erro_bf_pct)), "/", nrow(df_bt),
            " (", join(Dates.format.(df_bt.data_avaliacao[abs.(df_bt.erro_cl_pct) .< abs.(df_bt.erro_bf_pct)], "yyyy"), ", "), ")")
    println("Cobertura IC95% -- Mack (CL): ", sum(df_bt.cobertura_mack_95), "/", nrow(df_bt),
            " | bootstrap CL: ", sum(df_bt.cobertura_boot_cl_95), "/", nrow(df_bt),
            " | bootstrap BF: ", sum(df_bt.cobertura_boot_bf_95), "/", nrow(df_bt))
    println("Cortes com efeito de calendário significativo (p<0,05): ", sum(df_bt.cal_p_valor .< 0.05), "/", nrow(df_bt))
    return (mae_cl=mae_cl, mae_bf=mae_bf, med_cl=med_cl, med_bf=med_bf, mae_cl_pond=mae_cl_pond, mae_bf_pond=mae_bf_pond)
end

resumo_bem_especificado = resumo_metricas(df_bt; rotulo="prior bem especificado")

println()
println("== 3b. Robustez: a vantagem média do BF depende de um corte só? ==")
idx_2021 = findfirst(==(Date(2021, 12, 31)), df_bt.data_avaliacao)
sem_2021 = df_bt[setdiff(1:nrow(df_bt), [idx_2021]), :]
mae_cl_sem21 = mean(abs.(sem_2021.erro_cl_pct))
mae_bf_sem21 = mean(abs.(sem_2021.erro_bf_pct))
println("Erro de dez/2021 isolado -- CL: ", round(df_bt.erro_cl_pct[idx_2021], digits=1), "% | BF: ", round(df_bt.erro_bf_pct[idx_2021], digits=1), "%")
println("MAE simples excluindo dez/2021 -- CL: ", round(mae_cl_sem21, digits=2), "% | BF: ", round(mae_bf_sem21, digits=2), "%")

maduros = df_bt[df_bt.deficit_maturidade_max .== 0, :]
truncados = df_bt[df_bt.deficit_maturidade_max .> 0, :]
mae_cl_maduro = mean(abs.(maduros.erro_cl_pct))
mae_bf_maduro = mean(abs.(maduros.erro_bf_pct))
w_mad = maduros.reserva_real_bi
mae_cl_maduro_pond = sum(w_mad .* abs.(maduros.erro_cl_pct)) / sum(w_mad)
mae_bf_maduro_pond = sum(w_mad .* abs.(maduros.erro_bf_pct)) / sum(w_mad)
mae_cl_truncado = mean(abs.(truncados.erro_cl_pct))
mae_bf_truncado = mean(abs.(truncados.erro_bf_pct))
println("Cortes medidos até a maturidade completa (", nrow(maduros), " cortes, ",
        join(Dates.format.(maduros.data_avaliacao, "yyyy"), "-"), "): MAE CL ",
        round(mae_cl_maduro, digits=1), "% | BF ", round(mae_bf_maduro, digits=1),
        "% (ponderado: CL ", round(mae_cl_maduro_pond, digits=1), "% | BF ", round(mae_bf_maduro_pond, digits=1), "%)")
println("Cortes com horizonte truncado (", nrow(truncados), " cortes, ",
        join(Dates.format.(truncados.data_avaliacao, "yyyy"), "-"), "): MAE CL ",
        round(mae_cl_truncado, digits=1), "% | BF ", round(mae_bf_truncado, digits=1), "%")

println()
println("-- Separando cortes pela proporção de safras jovens sobre ativas (achado nulo, reportado) --")
proporcao_jovens = df_bt.n_safras_jovens ./ df_bt.n_safras_ativas
mediana_prop = median(proporcao_jovens)
println("Proporção de safras jovens varia de ", round(100 * minimum(proporcao_jovens), digits=1), "% a ",
        round(100 * maximum(proporcao_jovens), digits=1), "%, mediana ", round(100 * mediana_prop, digits=1),
        "%, sem variação suficiente para separar cortes de forma informativa nesta simulação.")

println()
println("== 4. Sensibilidade do prior do BF (viés de +25% e -25% no nível) ==")
resultados_mais = run_rolling_backtest(sim, valuation_pontos, scr.taxa, 2.2; n_boot=1000, prior_vies=1.25)
df_bt_mais = monta_df(resultados_mais)
CSV.write(joinpath(OUT, "backtest_resultados_prior_mais25.csv"), df_bt_mais)
resumo_mais25 = resumo_metricas(df_bt_mais; rotulo="prior mal especificado, +25%")

resultados_menos = run_rolling_backtest(sim, valuation_pontos, scr.taxa, 2.2; n_boot=1000, prior_vies=0.75)
df_bt_menos = monta_df(resultados_menos)
CSV.write(joinpath(OUT, "backtest_resultados_prior_menos25.csv"), df_bt_menos)
resumo_menos25 = resumo_metricas(df_bt_menos; rotulo="prior mal especificado, -25%")

resumo = Dict(
    "n_cortes" => nrow(df_bt),
    "mae_cl_pct" => resumo_bem_especificado.mae_cl,
    "mae_bf_pct" => resumo_bem_especificado.mae_bf,
    "mediana_cl_pct" => resumo_bem_especificado.med_cl,
    "mediana_bf_pct" => resumo_bem_especificado.med_bf,
    "mae_cl_pct_ponderado" => resumo_bem_especificado.mae_cl_pond,
    "mae_bf_pct_ponderado" => resumo_bem_especificado.mae_bf_pond,
    "mae_cl_pct_sem_2021" => mae_cl_sem21,
    "mae_bf_pct_sem_2021" => mae_bf_sem21,
    "erro_cl_pct_2021" => df_bt.erro_cl_pct[idx_2021],
    "erro_bf_pct_2021" => df_bt.erro_bf_pct[idx_2021],
    "mae_cl_pct_maduro" => mae_cl_maduro,
    "mae_bf_pct_maduro" => mae_bf_maduro,
    "mae_cl_pct_maduro_ponderado" => mae_cl_maduro_pond,
    "mae_bf_pct_maduro_ponderado" => mae_bf_maduro_pond,
    "mae_cl_pct_truncado" => mae_cl_truncado,
    "mae_bf_pct_truncado" => mae_bf_truncado,
    "n_cortes_maduros" => nrow(maduros),
    "n_cortes_truncados" => nrow(truncados),
    "mae_bf_pct_prior_mais25" => resumo_mais25.mae_bf,
    "mae_bf_pct_ponderado_prior_mais25" => resumo_mais25.mae_bf_pond,
    "mae_bf_pct_prior_menos25" => resumo_menos25.mae_bf,
    "mae_bf_pct_ponderado_prior_menos25" => resumo_menos25.mae_bf_pond,
    "sub_cl" => sum(df_bt.sub_cl),
    "sub_bf" => sum(df_bt.sub_bf),
    "cortes_cl_vence" => sum(abs.(df_bt.erro_cl_pct) .< abs.(df_bt.erro_bf_pct)),
    "cobertura_mack_95" => sum(df_bt.cobertura_mack_95),
    "cobertura_boot_cl_95" => sum(df_bt.cobertura_boot_cl_95),
    "cobertura_boot_bf_95" => sum(df_bt.cobertura_boot_bf_95),
    "cortes_efeito_calendario_sig" => sum(df_bt.cal_p_valor .< 0.05),
    "n_cortes_total" => nrow(df_bt),
    "taxa_media_real_pct" => 100 * mean(scr.taxa),
    "taxa_min_real_pct" => 100 * minimum(scr.taxa),
    "taxa_max_real_pct" => 100 * maximum(scr.taxa),
    "volume_total_bi" => sum(sim.volume) / 1e9,
    "safra_inicio" => string(origem_datas[1]),
    "safra_fim" => string(origem_datas[end]),
    "n_safras_truncadas_horizonte_total" => n_safras_truncadas,
    "deficit_maturidade_maximo" => sim.max_dev - minimum(sim.max_age_within_horizon),
    "idade_alvo_minima_2024" => df_bt.idade_alvo_minima[end],
)
open(joinpath(OUT, "resumo.json"), "w") do io
    JSON3.write(io, resumo)
end

println()
println("Saídas gravadas em ", OUT)
