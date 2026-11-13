"""
Testa a hipótese do André: "os ladders são diferentes entre setores, algumas operações
são mais curtas que outras" -- será que a diferença de PRAZO típico de carteira entre
modalidades explica (sozinha) o ranking de β pós-2020 medido na Fase 1c, em vez de (ou
além de) um fator sistemático genuíno?

Duas fontes de prazo, documentadas e comparadas:
1. `data/prazo_modalidades.csv`: série oficial SGS "prazo médio da carteira de crédito"
   por modalidade (códigos documentados em `scripts/prep_macro_sgs.py`), classificação
   livres/direcionados do BCB -- não exatamente a mesma predicação de modalidade usada
   no resto deste projeto.
2. `data/scr_modalidades_prazo.csv`: construído AQUI dos microdados brutos do SCR.data
   (`scripts/build_scr_prazo.py`), usando as MESMAS sete predicações de modalidade de
   `src/ScrModalidades.jl` -- diretamente comparável ao β de `fator_comum_beta_r2.csv`
   porque vem da mesma base e do mesmo agrupamento.

Grava em `results/`:
- `prazo_vs_beta.csv`: prazo médio (dias, pré/pós-2020, as duas fontes) ao lado do β
  pré/pós-2020 já medido, com RANKING de prazo e de β, e a correlação de Spearman
  entre eles.
- `correlacao_cruzada_lags.csv`: correlação defasada (-4 a +4 trimestres) entre a
  média do grupo de varejo "rápido" (cartão, veículos, pessoal sem consignação) e
  cada uma das modalidades de ladder mais longo (imobiliário, rural, consignado) e
  capital de giro -- testa se o comovimento aparece com atraso nas carteiras longas
  em vez de genuinamente ausente.
"""

using CSV
using DataFrames
using Statistics
using StatsBase: corspearman
using Dates

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito.ScrModalidades
using .ChainLadderCredito.FatorComum

dados = load_scr_modalidades(joinpath(@__DIR__, "..", "data", "scr_modalidades.csv"))
nomes = dados.nomes
variacoes = Dict(nome => variacao_log(dados.taxa[nome]) for nome in nomes)
datas_var = dados.datas[2:end]
corte_2020 = Date(2020, 1, 1)
idx_pre, idx_pos = indices_subperiodo(datas_var, corte_2020)

beta_df = CSV.read(joinpath(@__DIR__, "..", "results", "fator_comum_beta_r2.csv"), DataFrame)

println("== 1. Prazo médio por modalidade (dias), SCR.data bruto, pré/pós-2020 ==")
prazo_scr = CSV.read(joinpath(@__DIR__, "..", "data", "scr_modalidades_prazo.csv"), DataFrame; delim=';')
prazo_scr.data_base = Date.(prazo_scr.data_base)

prazo_medio_scr = Dict{String,NamedTuple}()
for nome in nomes
    sub = prazo_scr[prazo_scr.modalidade .== nome, :]
    sort!(sub, :data_base)
    pre = mean(sub.prazo_medio_dias[sub.data_base .< corte_2020])
    pos = mean(sub.prazo_medio_dias[sub.data_base .>= corte_2020])
    prazo_medio_scr[nome] = (prazo_pre_dias=pre, prazo_pos_dias=pos)
    println(rpad(nome, 25), " prazo pré=", round(pre / 30.44, digits=1), " meses",
            "  prazo pós=", round(pos / 30.44, digits=1), " meses")
end

println("\n== 2. Prazo médio por modalidade (meses), série oficial SGS, pré/pós-2020 ==")
prazo_sgs = CSV.read(joinpath(@__DIR__, "..", "data", "prazo_modalidades.csv"), DataFrame; delim=';')
prazo_sgs.data_base = Date.(prazo_sgs.data_base)
# mapeamento de nomes: a coluna "rural" do CSV SGS já é a média simples rural_pf/rural_pj
col_sgs = Dict("cartao_credito" => :cartao_credito, "veiculos" => :veiculos,
               "pessoal_sem_consignacao" => :pessoal_sem_consignacao,
               "pessoal_consignado" => :pessoal_consignado, "capital_giro" => :capital_giro,
               "imobiliario" => :imobiliario, "rural" => :rural)
prazo_medio_sgs = Dict{String,NamedTuple}()
for nome in nomes
    col = col_sgs[nome]
    pre = mean(skipmissing(prazo_sgs[prazo_sgs.data_base .< corte_2020, col]))
    pos = mean(skipmissing(prazo_sgs[prazo_sgs.data_base .>= corte_2020, col]))
    prazo_medio_sgs[nome] = (prazo_pre_meses=pre, prazo_pos_meses=pos)
    println(rpad(nome, 25), " prazo pré=", round(pre, digits=1), " meses",
            "  prazo pós=", round(pos, digits=1), " meses")
end

println("\n== 3. Ranking prazo (SCR.data) vs ranking β pós-2020, correlação de Spearman ==")
linhas = NamedTuple[]
for nome in nomes
    brow = beta_df[beta_df.modalidade .== nome, :]
    push!(linhas, (modalidade=nome,
                    prazo_scr_pre_meses=prazo_medio_scr[nome].prazo_pre_dias / 30.44,
                    prazo_scr_pos_meses=prazo_medio_scr[nome].prazo_pos_dias / 30.44,
                    prazo_sgs_pre_meses=prazo_medio_sgs[nome].prazo_pre_meses,
                    prazo_sgs_pos_meses=prazo_medio_sgs[nome].prazo_pos_meses,
                    beta_pre2020=brow.beta_pre2020[1], beta_pos2020=brow.beta_pos2020[1],
                    beta_full=brow.beta_pca[1]))
end
df_pb = DataFrame(linhas)
sort!(df_pb, :prazo_scr_pos_meses)
df_pb.rank_prazo_pos = 1:nrow(df_pb)
sort!(df_pb, order(:beta_pos2020, rev=true))
df_pb.rank_beta_pos = 1:nrow(df_pb)
sort!(df_pb, :modalidade)

rho_scr = corspearman(df_pb.prazo_scr_pos_meses, df_pb.beta_pos2020)
rho_sgs = corspearman(df_pb.prazo_sgs_pos_meses, df_pb.beta_pos2020)
rho_scr_pre = corspearman(df_pb.prazo_scr_pre_meses, df_pb.beta_pre2020)
println(df_pb[:, [:modalidade, :prazo_scr_pos_meses, :rank_prazo_pos, :beta_pos2020, :rank_beta_pos]])
println("\nCorrelação de Spearman entre prazo (SCR.data) e β, pós-2020: ", round(rho_scr, digits=3))
println("Correlação de Spearman entre prazo (SGS) e β, pós-2020: ", round(rho_sgs, digits=3))
println("Correlação de Spearman entre prazo (SCR.data) e β, pré-2020: ", round(rho_scr_pre, digits=3))

open(joinpath(@__DIR__, "..", "results", "prazo_vs_beta_spearman.txt"), "w") do io
    println(io, "spearman_prazo_scr_pos_vs_beta_pos,", rho_scr)
    println(io, "spearman_prazo_sgs_pos_vs_beta_pos,", rho_sgs)
    println(io, "spearman_prazo_scr_pre_vs_beta_pre,", rho_scr_pre)
end
CSV.write(joinpath(@__DIR__, "..", "results", "prazo_vs_beta.csv"), df_pb)

println("\n== 4. Correlação cruzada com defasagem: grupo rápido (cartão+veículos+pessoal s/ consig) vs carteiras longas ==")
grupo_rapido = padronizar_serie((variacoes["cartao_credito"] .+ variacoes["veiculos"] .+
                                  variacoes["pessoal_sem_consignacao"]) ./ 3)
candidatos_longos = ["imobiliario", "rural", "pessoal_consignado", "capital_giro"]

linhas_lag = NamedTuple[]
for nome in candidatos_longos
    y = padronizar_serie(variacoes[nome])
    for lag in -4:4
        r = correlacao_defasada(y, grupo_rapido, lag)
        push!(linhas_lag, (modalidade_longa=nome, lag=lag, corr=r))
    end
    sub = DataFrame([l for l in linhas_lag if l.modalidade_longa == nome])
    melhor = sub[argmax(abs.(sub.corr)), :]
    println(rpad(nome, 25), " melhor |corr| em lag=", melhor.lag, " (corr=", round(melhor.corr, digits=3), ")",
            "  corr(lag=0)=", round(sub[sub.lag .== 0, :corr][1], digits=3))
end
CSV.write(joinpath(@__DIR__, "..", "results", "correlacao_cruzada_lags.csv"), DataFrame(linhas_lag))

println("\nConcluído.")
