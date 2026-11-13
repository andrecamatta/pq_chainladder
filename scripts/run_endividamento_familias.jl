"""
Testa a hipótese (b) do memo `apc_explicacao_2020_deep.md`: será que o endividamento e
o comprometimento de renda das famílias -- em níveis recordes no pós-2020 -- explicam
o fator comum de inadimplência melhor do que as séries macro já testadas em
`run_identificacao_macro.jl` (Selic, desemprego, atividade, massa de renda, Auxílio
Emergencial)?

Fonte: SGS do Banco Central, API `https://api.bcb.gov.br/dados/serie/bcdata.sgs.<codigo>`,
acesso em 12/11/2026:
- SGS 29037, "Endividamento das famílias com o SFN em relação à renda acumulada dos
  últimos 12 meses" (RNDBF), % -- nível de ESTOQUE de dívida sobre renda, série desde
  jan/2005.
- SGS 29034, "Comprometimento de renda das famílias com o serviço da dívida com o SFN,
  com ajuste sazonal" (RNDBF), % -- FLUXO de pagamento de dívida sobre renda do mês,
  série desde mar/2005.
Agregadas por média dos 3 meses do trimestre (mesma convenção de Selic/IBC-Br em
`prep_macro_sgs.py`), em `data/macro_endividamento_trimestral.csv` (JSON brutos fora do
commit em `data/sgs_raw/`, mesma convenção do restante do projeto).

Mesmo método de `run_identificacao_macro.jl`: correlação (Pearson) do fator comum
global (sem leave-one-out, sem circularidade -- nenhuma destas duas séries entra na
construção do fator) e da variação de cada modalidade com os NÍVEIS e as VARIAÇÕES
trimestrais de endividamento/comprometimento, full / pré-2020 / pós-2020.

Grava em `results/`:
- `identificacao_endividamento_fator.csv`
- `identificacao_endividamento_modalidades.csv`
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

variacoes = Dict(nome => variacao_log(dados.taxa[nome]) for nome in nomes)
datas_var = dados.datas[2:end]
Z = hcat([padronizar_serie(variacoes[nome]) for nome in nomes]...)
pesos_carteira = [mean(dados.carteira_ativa[nome]) for nome in nomes]

fator_global_pca = fator_pca(Z)
fator_global_pond = fator_media_ponderada(Z, pesos_carteira)

endiv_df = CSV.read(joinpath(@__DIR__, "..", "data", "macro_endividamento_trimestral.csv"),
                     DataFrame; delim=';', decimal='.')
endiv_df.data_base = Date.(endiv_df.data_base)
filter!(r -> r.data_base in Set(datas_var), endiv_df)
sort!(endiv_df, :data_base)
endiv_df.data_base == datas_var || error("macro_endividamento_trimestral.csv não cobre exatamente as datas de datas_var")

function diff1(x::Vector{Float64})
    out = fill(NaN, length(x))
    out[2:end] = x[2:end] .- x[1:(end-1)]
    return out
end

endiv_df.endividamento_var = diff1(Float64.(endiv_df.endividamento_nivel))
endiv_df.comprometimento_renda_var = diff1(Float64.(endiv_df.comprometimento_renda_nivel))

VARS = [:endividamento_nivel, :endividamento_var, :comprometimento_renda_nivel, :comprometimento_renda_var]

corte_2020 = Date(2020, 1, 1)
idx_pre, idx_pos = indices_subperiodo(datas_var, corte_2020)
println("Pré-2020: ", length(idx_pre), " trimestres | Pós-2020: ", length(idx_pos), " trimestres")

function corr_valido(x::Vector{Float64}, y::Vector{Float64})
    ok = .!isnan.(x) .& .!isnan.(y)
    sum(ok) < 3 && return NaN, sum(ok)
    return cor(x[ok], y[ok]), sum(ok)
end

println("\n== 1. Fator comum global × endividamento/comprometimento (full / pré / pós) ==")
linhas_fator = NamedTuple[]
for vsym in VARS
    xv = Float64.(endiv_df[:, vsym])
    for (fname, fser) in (("pca", fator_global_pca), ("media_ponderada", fator_global_pond))
        r_full, n_full = corr_valido(xv, fser)
        r_pre, n_pre = corr_valido(xv[idx_pre], fser[idx_pre])
        r_pos, n_pos = corr_valido(xv[idx_pos], fser[idx_pos])
        push!(linhas_fator, (variavel=String(vsym), fator=fname,
                              corr_full=r_full, n_full=n_full,
                              corr_pre2020=r_pre, n_pre2020=n_pre,
                              corr_pos2020=r_pos, n_pos2020=n_pos))
        println(rpad(String(vsym), 28), " fator=", rpad(fname, 16),
                " corr(full)=", round(r_full, digits=3),
                " corr(pré)=", round(r_pre, digits=3),
                " corr(pós)=", round(r_pos, digits=3))
    end
end
CSV.write(joinpath(@__DIR__, "..", "results", "identificacao_endividamento_fator.csv"), DataFrame(linhas_fator))

println("\n== 2. Variação de CADA modalidade × endividamento/comprometimento (full / pré / pós) ==")
linhas_mod = NamedTuple[]
for nome in nomes
    yv = padronizar_serie(variacoes[nome])
    for vsym in VARS
        xv = Float64.(endiv_df[:, vsym])
        r_full, n_full = corr_valido(xv, yv)
        r_pre, n_pre = corr_valido(xv[idx_pre], yv[idx_pre])
        r_pos, n_pos = corr_valido(xv[idx_pos], yv[idx_pos])
        push!(linhas_mod, (modalidade=nome, variavel=String(vsym),
                            corr_full=r_full, corr_pre2020=r_pre, corr_pos2020=r_pos))
    end
end
mod_df = DataFrame(linhas_mod)
CSV.write(joinpath(@__DIR__, "..", "results", "identificacao_endividamento_modalidades.csv"), mod_df)
for nome in nomes
    sub = mod_df[mod_df.modalidade .== nome, :]
    i = argmax(abs.(sub.corr_pos2020))
    println(rpad(nome, 25), " maior |corr| pós-2020: ", sub.variavel[i], " = ", round(sub.corr_pos2020[i], digits=3))
end

println("\n== 3. Nível de endividamento no pós-2020: sobe, platô ou recua? ==")
niv_pos = Float64.(endiv_df.endividamento_nivel[idx_pos])
println("endividamento_nivel pós-2020: ", round(niv_pos[1], digits=1), " (", datas_var[idx_pos[1]], ") -> ",
        round(maximum(niv_pos), digits=1), " (máximo) -> ", round(niv_pos[end], digits=1), " (", datas_var[idx_pos[end]], ")")
compr_pos = Float64.(endiv_df.comprometimento_renda_nivel[idx_pos])
println("comprometimento_renda_nivel pós-2020: ", round(compr_pos[1], digits=1), " -> ",
        round(maximum(compr_pos), digits=1), " (máximo) -> ", round(compr_pos[end], digits=1))

println("\nConcluído.")
