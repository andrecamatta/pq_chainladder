"""
Mede, nas séries REAIS do SCR.data por modalidade (`data/scr_modalidades.csv`, gerado
por `scripts/build_scr_modalidades.jl`), quanto da variação trimestral da taxa de
inadimplência de cada modalidade é explicada por um fator comum (sistemático) e
quanto é idiossincrático (próprio da modalidade) -- o paralelo empírico, com dado real,
do `beta_s`/`beta_s²` sintético já medido em `results/apc_beta_segmentos.csv` (Fase 1).

Grava:
- `results/fator_comum_beta_r2.csv`: β e R² por modalidade (PCA leave-one-out,
  ponto + IC95% bootstrap em blocos), a alternativa de robustez (média ponderada pela
  carteira, leave-one-out) e a quebra por subperíodo (antes/depois de 2020-01-01).
- `results/fator_comum_sensibilidade_bloco.csv`: IC95% do bootstrap sob três
  comprimentos de bloco (4, 8, 12 trimestres), para checar sensibilidade da LARGURA do
  intervalo (o ponto β não depende do bootstrap, só o IC) a essa escolha.

O bootstrap resample os PARES `(variação_t, fator_t)` em blocos, com o fator leave-
one-out mantido FIXO no valor calculado uma vez sobre a amostra completa (não
reestimado por réplica) -- captura a incerteza amostral da correlação medida numa
série curta, não a incerteza de reestimar a própria PCA a cada réplica. Simplificação
declarada, não escondida -- ver `apc_fase1c_resumo.md`, seção 3.
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
println("Trimestres (nível): ", dados.datas[1], " a ", dados.datas[end], " (", length(dados.datas), ")")

# -- Transformação estacionária: variação trimestral do log da taxa (QoQ) --
variacoes = Dict(nome => variacao_log(dados.taxa[nome]) for nome in nomes)
datas_var = dados.datas[2:end]
Z = hcat([padronizar_serie(variacoes[nome]) for nome in nomes]...)  # T x N
pesos_carteira = [mean(dados.carteira_ativa[nome]) for nome in nomes]
println("Trimestres (variação): ", datas_var[1], " a ", datas_var[end], " (", length(datas_var), ")")

println("\n== 1. β / R² (PCA leave-one-out) e robustez (média ponderada pela carteira), amostra completa ==")
linhas_principais = NamedTuple[]
fatores_pca = Dict{String,Vector{Float64}}()
for (i, nome) in enumerate(nomes)
    y = variacoes[nome]
    f_pca = fator_pca_loo(Z, i)
    fatores_pca[nome] = f_pca
    beta_pca, r2_pca = estimar_beta_r2(y, f_pca)

    f_pond = fator_media_ponderada_loo(Z, pesos_carteira, i)
    beta_pond, r2_pond = estimar_beta_r2(y, f_pond)

    betas_boot, r2_boot = bootstrap_beta_r2(y, f_pca; n_boot=2000, bloco=8)
    lo, hi = intervalo_percentil(betas_boot; alpha=0.05)
    lo_r2, hi_r2 = intervalo_percentil(r2_boot; alpha=0.05)

    println(rpad(nome, 25), " β(PCA)=", round(beta_pca, digits=3),
            " IC95%=[", round(lo, digits=3), ", ", round(hi, digits=3), "]",
            "  R²=", round(r2_pca, digits=3), " IC95%=[", round(lo_r2, digits=3), ", ", round(hi_r2, digits=3), "]",
            "  | β(ponderado)=", round(beta_pond, digits=3), " R²=", round(r2_pond, digits=3))

    push!(linhas_principais, (modalidade=nome, beta_pca=beta_pca, r2_pca=r2_pca,
                               beta_ic95_lo=lo, beta_ic95_hi=hi, r2_ic95_lo=lo_r2, r2_ic95_hi=hi_r2,
                               beta_media_ponderada=beta_pond, r2_media_ponderada=r2_pond,
                               n_trimestres=length(y)))
end

println("\n== 2. Estabilidade por subperíodo (antes / depois de 2020-01-01) ==")
corte = Date(2020, 1, 1)
idx_pre, idx_pos = indices_subperiodo(datas_var, corte)
println("Pré-2020: ", length(idx_pre), " trimestres (", datas_var[idx_pre[1]], " a ", datas_var[idx_pre[end]], ")")
println("Pós-2020: ", length(idx_pos), " trimestres (", datas_var[idx_pos[1]], " a ", datas_var[idx_pos[end]], ")")

function beta_r2_subperiodo(Z_sub::AbstractMatrix{Float64}, variacoes_sub::Dict{String,Vector{Float64}}, nomes)
    out = Dict{String,Tuple{Float64,Float64}}()
    for (i, nome) in enumerate(nomes)
        f = fator_pca_loo(Z_sub, i)
        beta, r2 = estimar_beta_r2(variacoes_sub[nome], f)
        out[nome] = (beta, r2)
    end
    return out
end

Z_pre = Z[idx_pre, :]
Z_pos = Z[idx_pos, :]
variacoes_pre = Dict(nome => variacoes[nome][idx_pre] for nome in nomes)
variacoes_pos = Dict(nome => variacoes[nome][idx_pos] for nome in nomes)
sub_pre = beta_r2_subperiodo(Z_pre, variacoes_pre, nomes)
sub_pos = beta_r2_subperiodo(Z_pos, variacoes_pos, nomes)

linhas_finais = NamedTuple[]
for (k, nome) in enumerate(nomes)
    base = linhas_principais[k]
    beta_pre, r2_pre = sub_pre[nome]
    beta_pos, r2_pos = sub_pos[nome]
    println(rpad(nome, 25), " β pré-2020=", round(beta_pre, digits=3), " (R²=", round(r2_pre, digits=3), ")",
            "  | β pós-2020=", round(beta_pos, digits=3), " (R²=", round(r2_pos, digits=3), ")")
    push!(linhas_finais, merge(base, (beta_pre2020=beta_pre, r2_pre2020=r2_pre, n_pre2020=length(idx_pre),
                                        beta_pos2020=beta_pos, r2_pos2020=r2_pos, n_pos2020=length(idx_pos))))
end

out_path = joinpath(@__DIR__, "..", "results", "fator_comum_beta_r2.csv")
CSV.write(out_path, DataFrame(linhas_finais))
println("\nGravado em $out_path")

println("\n== 3. Sensibilidade do IC95% ao comprimento do bloco do bootstrap (4, 8, 12 trimestres) ==")
linhas_bloco = NamedTuple[]
for (i, nome) in enumerate(nomes)
    y = variacoes[nome]
    f_pca = fatores_pca[nome]
    for bloco in (4, 8, 12)
        betas_boot, = bootstrap_beta_r2(y, f_pca; n_boot=2000, bloco=bloco)
        lo, hi = intervalo_percentil(betas_boot; alpha=0.05)
        push!(linhas_bloco, (modalidade=nome, bloco=bloco, beta_ic95_lo=lo, beta_ic95_hi=hi,
                              largura=hi - lo))
    end
end
out_bloco = DataFrame(linhas_bloco)
for nome in nomes
    sub = out_bloco[out_bloco.modalidade .== nome, :]
    println(rpad(nome, 25), " larguras IC95% (bloco=4,8,12): ", round.(sub.largura, digits=3))
end
out_bloco_path = joinpath(@__DIR__, "..", "results", "fator_comum_sensibilidade_bloco.csv")
CSV.write(out_bloco_path, out_bloco)
println("\nGravado em $out_bloco_path")
