"""
Exercício de identificação que a Fase 1c (`apc_fase1c_resumo.md`, seção 5) declarou
explicitamente como NÃO feito: correlacionar o fator comum estatístico (PCA, sem
exclusão -- aqui não há circularidade a evitar, porque nenhuma série macro externa
entra na construção do fator) e a variação de cada modalidade com séries
macroeconômicas oficiais, no período completo e separadamente antes/depois de
2020-01-01. Também testa a ROBUSTEZ da data de quebra (2020 vs 2021 vs 2022,
recortes 2020-21 vs 2022-25, leave-one-year-out) e a PERSISTÊNCIA do comovimento via
β/correlação em janela móvel -- ver `apc_explicacao_2020.md` para a leitura.

Fonte das séries macro: `data/macro_trimestral.csv`, já agregado por
`scripts/prep_macro_sgs.py` a partir de `data/sgs_raw/*.json` (API SGS do Banco
Central, códigos e datas de acesso documentados no cabeçalho desse script).

Grava em `results/`:
- `identificacao_macro_fator.csv`: correlação (e regressão simples) do fator comum
  global (PCA e média ponderada, sem leave-one-out) com cada série macro, full/pré/pós.
- `identificacao_macro_modalidades.csv`: o mesmo, mas para a variação de CADA
  modalidade (não o fator), full/pré/pós.
- `robustez_data_quebra.csv`: β pré/pós recalculado para cortes em 2020, 2021 e 2022.
- `robustez_subperiodos_pos2020.csv`: β nos recortes 2020-2021 vs 2022-2025, e
  leave-one-year-out dentro do pós-2020 (exclui um ano por vez).
- `rolling_beta_fator.csv`: β em janela móvel (12 e 16 trimestres) de cada modalidade
  contra o fator leave-one-out FIXO da amostra completa (o mesmo da Fase 1c).
- `rolling_correlacao_pares.csv`: correlação em janela móvel (12 e 16 trimestres)
  entre pares de modalidades de varejo (cartão, veículos, pessoal sem consignação).
"""

using CSV
using DataFrames
using Statistics
using Dates

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito.ScrModalidades
using .ChainLadderCredito.FatorComum

dados = load_scr_modalidades(joinpath(@__DIR__, "..", "data", "scr_modalidades.csv"))
nomes = dados.nomes
n_mod = length(nomes)
println("Modalidades (", n_mod, "): ", join(nomes, ", "))

variacoes = Dict(nome => variacao_log(dados.taxa[nome]) for nome in nomes)
datas_var = dados.datas[2:end]
Z = hcat([padronizar_serie(variacoes[nome]) for nome in nomes]...)
pesos_carteira = [mean(dados.carteira_ativa[nome]) for nome in nomes]
println("Trimestres (variação): ", datas_var[1], " a ", datas_var[end], " (", length(datas_var), ")")

fator_global_pca = fator_pca(Z)
fator_global_pond = fator_media_ponderada(Z, pesos_carteira)
println("Correlação fator_pca global vs fator_media_ponderada global: ",
        round(cor(fator_global_pca, fator_global_pond), digits=3))

# -- Carrega macro, alinha pelas datas de datas_var (inner join) --
macro_df = CSV.read(joinpath(@__DIR__, "..", "data", "macro_trimestral.csv"), DataFrame;
                     delim=';', decimal='.')
macro_df.data_base = Date.(macro_df.data_base)
filter!(r -> r.data_base in Set(datas_var), macro_df)
sort!(macro_df, :data_base)
macro_df.data_base == datas_var || error("macro_trimestral.csv não cobre exatamente as datas de datas_var")

VARS_MACRO = [:selic_nivel, :selic_var, :ipca_trim, :desemprego_nivel, :desemprego_var,
              :massa_renda_var, :ibcbr_var, :auxilio_emergencial_dummy]

corte_2020 = Date(2020, 1, 1)
idx_pre, idx_pos = indices_subperiodo(datas_var, corte_2020)
println("Pré-2020: ", length(idx_pre), " trimestres | Pós-2020: ", length(idx_pos), " trimestres")

function corr_e_slope(x::Vector{Float64}, y::Vector{Float64})
    # x: série macro bruta; y: fator/variação já padronizado. slope = quantos
    # desvios-padrão de y por unidade de x (regressão simples OLS, y ~ a + slope*x).
    r = cor(x, y)
    slope = cov(x, y) / var(x)
    return r, slope
end

println("\n== 1. Fator comum global × macro (full / pré-2020 / pós-2020) ==")
linhas_fator = NamedTuple[]
for vsym in VARS_MACRO
    xv = Float64.(macro_df[:, vsym])
    for (fname, fser) in (("pca", fator_global_pca), ("media_ponderada", fator_global_pond))
        r_full, s_full = corr_e_slope(xv, fser)
        r_pre, s_pre = corr_e_slope(xv[idx_pre], fser[idx_pre])
        r_pos, s_pos = corr_e_slope(xv[idx_pos], fser[idx_pos])
        push!(linhas_fator, (variavel_macro=String(vsym), fator=fname,
                              corr_full=r_full, slope_full=s_full, n_full=length(xv),
                              corr_pre2020=r_pre, slope_pre2020=s_pre, n_pre2020=length(idx_pre),
                              corr_pos2020=r_pos, slope_pos2020=s_pos, n_pos2020=length(idx_pos)))
        println(rpad(String(vsym), 24), " fator=", rpad(fname, 16),
                " corr(full)=", round(r_full, digits=3),
                " corr(pré)=", round(r_pre, digits=3),
                " corr(pós)=", round(r_pos, digits=3))
    end
end
CSV.write(joinpath(@__DIR__, "..", "results", "identificacao_macro_fator.csv"), DataFrame(linhas_fator))

println("\n== 2. Variação de CADA modalidade × macro (full / pré-2020 / pós-2020) ==")
linhas_mod = NamedTuple[]
for nome in nomes
    yv = padronizar_serie(variacoes[nome])
    for vsym in VARS_MACRO
        xv = Float64.(macro_df[:, vsym])
        r_full, s_full = corr_e_slope(xv, yv)
        r_pre, s_pre = corr_e_slope(xv[idx_pre], yv[idx_pre])
        r_pos, s_pos = corr_e_slope(xv[idx_pos], yv[idx_pos])
        push!(linhas_mod, (modalidade=nome, variavel_macro=String(vsym),
                            corr_full=r_full, corr_pre2020=r_pre, corr_pos2020=r_pos))
    end
end
mod_df = DataFrame(linhas_mod)
CSV.write(joinpath(@__DIR__, "..", "results", "identificacao_macro_modalidades.csv"), mod_df)
for nome in nomes
    sub = mod_df[mod_df.modalidade .== nome, :]
    println(rpad(nome, 25), " maior |corr| full: ",
            sub.variavel_macro[argmax(abs.(sub.corr_full))], " = ",
            round(sub.corr_full[argmax(abs.(sub.corr_full))], digits=3))
end

println("\n== 3. Robustez da data de quebra (2020 / 2021 / 2022) ==")
function beta_r2_subperiodo_corte(corte::Date)
    idx_a, idx_b = indices_subperiodo(datas_var, corte)
    out = Dict{String,NamedTuple}()
    Za, Zb = Z[idx_a, :], Z[idx_b, :]
    for (i, nome) in enumerate(nomes)
        fa = fator_pca_loo(Za, i)
        fb = fator_pca_loo(Zb, i)
        ba, r2a = estimar_beta_r2(variacoes[nome][idx_a], fa)
        bb, r2b = estimar_beta_r2(variacoes[nome][idx_b], fb)
        out[nome] = (beta_antes=ba, r2_antes=r2a, n_antes=length(idx_a),
                     beta_depois=bb, r2_depois=r2b, n_depois=length(idx_b))
    end
    return out
end

linhas_corte = NamedTuple[]
for corte in (Date(2020, 1, 1), Date(2021, 1, 1), Date(2022, 1, 1))
    res = beta_r2_subperiodo_corte(corte)
    println("-- corte em $corte --")
    for nome in nomes
        r = res[nome]
        push!(linhas_corte, (corte=corte, modalidade=nome, beta_antes=r.beta_antes,
                              r2_antes=r.r2_antes, n_antes=r.n_antes,
                              beta_depois=r.beta_depois, r2_depois=r.r2_depois, n_depois=r.n_depois))
        println(rpad(nome, 25), " β antes=", round(r.beta_antes, digits=3),
                " β depois=", round(r.beta_depois, digits=3))
    end
end
CSV.write(joinpath(@__DIR__, "..", "results", "robustez_data_quebra.csv"), DataFrame(linhas_corte))

println("\n== 4. Dentro do pós-2020: 2020-21 vs 2022-25, e leave-one-year-out ==")
anos_pos = year.(datas_var[idx_pos])
idx_2020_21 = idx_pos[findall(a -> a in (2020, 2021), anos_pos)]
idx_2022_25 = idx_pos[findall(a -> a in (2022, 2023, 2024, 2025), anos_pos)]

linhas_sub = NamedTuple[]
for (rotulo, idx) in (("2020_2021", idx_2020_21), ("2022_2025", idx_2022_25))
    Zs = Z[idx, :]
    for (i, nome) in enumerate(nomes)
        f = fator_pca_loo(Zs, i)
        b, r2 = estimar_beta_r2(variacoes[nome][idx], f)
        push!(linhas_sub, (recorte=rotulo, modalidade=nome, beta=b, r2=r2, n=length(idx)))
    end
end

for ano_excluido in 2020:2025
    idx_loyo = idx_pos[findall(a -> a != ano_excluido, anos_pos)]
    length(idx_loyo) < 4 && continue
    Zs = Z[idx_loyo, :]
    for (i, nome) in enumerate(nomes)
        f = fator_pca_loo(Zs, i)
        b, r2 = estimar_beta_r2(variacoes[nome][idx_loyo], f)
        push!(linhas_sub, (recorte="pos2020_sem_$ano_excluido", modalidade=nome, beta=b, r2=r2, n=length(idx_loyo)))
    end
end
sub_df = DataFrame(linhas_sub)
CSV.write(joinpath(@__DIR__, "..", "results", "robustez_subperiodos_pos2020.csv"), sub_df)
for nome in ("cartao_credito", "veiculos", "pessoal_sem_consignacao", "pessoal_consignado", "rural")
    sub = sub_df[sub_df.modalidade .== nome, :]
    println(rpad(nome, 25), join([" $(r.recorte)=$(round(r.beta, digits=2))" for r in eachrow(sub)]))
end

println("\n== 5. β em janela móvel contra o fator leave-one-out FIXO (12 e 16 trimestres) ==")
linhas_roll_beta = NamedTuple[]
for (i, nome) in enumerate(nomes)
    f_fixo = fator_pca_loo(Z, i)   # mesmo fator da Fase 1c (amostra completa), não reestimado por janela
    y = variacoes[nome]
    for janela in (12, 16)
        fim_idx, betas, r2s = beta_r2_janela_movel(y, f_fixo, janela)
        for (k, fi) in enumerate(fim_idx)
            push!(linhas_roll_beta, (modalidade=nome, janela=janela, data_fim=datas_var[fi],
                                      beta=betas[k], r2=r2s[k]))
        end
    end
end
CSV.write(joinpath(@__DIR__, "..", "results", "rolling_beta_fator.csv"), DataFrame(linhas_roll_beta))
println("Gravado rolling_beta_fator.csv (", length(linhas_roll_beta), " linhas)")

println("\n== 6. Correlação em janela móvel entre pares de modalidades de varejo ==")
pares = [("cartao_credito", "veiculos"), ("cartao_credito", "pessoal_sem_consignacao"),
         ("veiculos", "pessoal_sem_consignacao"), ("cartao_credito", "pessoal_consignado"),
         ("veiculos", "rural")]
linhas_roll_corr = NamedTuple[]
for (a, b) in pares
    ya, yb = variacoes[a], variacoes[b]
    n = length(ya)
    for janela in (12, 16)
        for fim in janela:n
            inicio = fim - janela + 1
            r = cor(ya[inicio:fim], yb[inicio:fim])
            push!(linhas_roll_corr, (par="$(a)__$(b)", janela=janela, data_fim=datas_var[fim], corr=r))
        end
    end
end
CSV.write(joinpath(@__DIR__, "..", "results", "rolling_correlacao_pares.csv"), DataFrame(linhas_roll_corr))
println("Gravado rolling_correlacao_pares.csv (", length(linhas_roll_corr), " linhas)")

println("\nConcluído.")
