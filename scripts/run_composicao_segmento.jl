"""
Testa a hipótese de mudança de COMPOSIÇÃO de carteira (candidata (a)/(b) do memo
`apc_explicacao_2020_deep.md`): será que a entrada de fintechs de crédito (Sociedades
de Crédito Direto / SCD e Sociedades de Empréstimo entre Pessoas / SEP, Resolução CMN
4.656/2018) e de instituições de pagamento com crédito agregado ("bundling", REB 2021)
foi desproporcionalmente maior nas três modalidades de maior β pós-2020 (cartão,
veículos, pessoal sem consignação) do que em consignado e rural?

Fonte: microdados BRUTOS do SCR.data (Portal de Dados Abertos do Banco Central,
`scrdata_<trimestre>.csv`), já baixados para OUTRO projeto deste repositório de
artigos (`pq_panjer`) -- lidos aqui em modo SOMENTE LEITURA, nenhum arquivo copiado ou
baixado de novo nesta sessão. Caminho fora deste repositório por design: os brutos têm
~100MB por trimestre (~5,4GB ao todo), não cabem no controle de versão deste projeto
(mesma convenção já declarada em `ScrModalidades.build_scr_modalidades_csv`).

O campo `segmento` do SCR.data (tipo de instituição) só passa a ter a categoria
"Fintech" a partir do trimestre de referência jun/2019 (confirmado por inspeção direta
dos arquivos: set/2012 a mar/2019 têm só Banco/Financeira/Cooperativa/Arrendamento/
Desenvolvimento-Fomento/Outros; "Instituição de pagamento" aparece já em mar/2019;
"Fintech" aparece a partir de jun/2019) -- consistente com a Resolução CMN 4.656/2018,
que criou as figuras de SCD/SEP e cuja primeira autorização efetiva ocorreu em 2019.

Grava em `results/composicao_segmento_modalidades.csv`: carteira_ativa total e por
segmento (Fintech, Instituição de pagamento, Banco) de cada uma das 7 modalidades, por
trimestre, de set/2012 a dez/2025 (mesma predicação de modalidade de
`ScrModalidades.MODALIDADES_PADRAO`, aplicada linha a linha aos microdados brutos, não
reaproveitando `data/scr_modalidades.csv`, que não tem a coluna segmento).
"""

using CSV
using DataFrames
using Dates

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito.ScrModalidades: MODALIDADES_PADRAO

RAW_DIR = "/workspace/pq/artigos/panjer/pq_panjer/data/raw"
isdir(RAW_DIR) || error("diretório de microdados brutos do SCR.data não encontrado: $RAW_DIR")

arquivos = sort(filter(f -> endswith(f, ".csv"), readdir(RAW_DIR; join=true)))
println("Arquivos trimestrais encontrados: ", length(arquivos))

SEGMENTOS_INTERESSE = ["Fintech", "Instituição de pagamento", "Banco"]

linhas = NamedTuple[]
for (k, f) in enumerate(arquivos)
    t0 = time()
    df = CSV.read(f, DataFrame; delim=';', decimal=',',
                   select=[:data_base, :cliente, :segmento, :modalidade, :submodalidade,
                           :carteira_ativa],
                   pool=true, stringtype=String)
    data_base = df.data_base[1]
    for def in MODALIDADES_PADRAO
        sub = def.cliente === nothing ? df : df[df.cliente .== def.cliente, :]
        mask = [def.pred(String(m), String(s)) for (m, s) in zip(sub.modalidade, sub.submodalidade)]
        bloco = sub[mask, :]
        total = sum(bloco.carteira_ativa)
        porseg = Dict(seg => sum(bloco.carteira_ativa[bloco.segmento .== seg]; init=0.0)
                       for seg in SEGMENTOS_INTERESSE)
        push!(linhas, (data_base=data_base, modalidade=def.nome, carteira_ativa_total=total,
                        carteira_ativa_fintech=porseg["Fintech"],
                        carteira_ativa_ip=porseg["Instituição de pagamento"],
                        carteira_ativa_banco=porseg["Banco"]))
    end
    println("  [$k/$(length(arquivos))] $(basename(f)) ($data_base) -- $(round(time()-t0, digits=1))s")
end

out = DataFrame(linhas)
sort!(out, [:modalidade, :data_base])
out.share_fintech = out.carteira_ativa_fintech ./ out.carteira_ativa_total
out.share_fintech_ip = (out.carteira_ativa_fintech .+ out.carteira_ativa_ip) ./ out.carteira_ativa_total
out.share_banco = out.carteira_ativa_banco ./ out.carteira_ativa_total

out_path = joinpath(@__DIR__, "..", "results", "composicao_segmento_modalidades.csv")
CSV.write(out_path, out)
println("\nGravado $out_path ($(nrow(out)) linhas)")

println("\n== Share fintech+instituição de pagamento, por modalidade, dez/2019 vs dez/2025 ==")
for def in MODALIDADES_PADRAO
    nome = def.nome
    sub = out[out.modalidade .== nome, :]
    sort!(sub, :data_base)
    i_2019 = findfirst(==(Date(2019, 12, 31)), sub.data_base)
    i_2025 = findfirst(==(Date(2025, 12, 31)), sub.data_base)
    if i_2019 === nothing || i_2025 === nothing
        println(rpad(nome, 25), " datas de referência não encontradas")
        continue
    end
    s19 = sub.share_fintech_ip[i_2019]
    s25 = sub.share_fintech_ip[i_2025]
    println(rpad(nome, 25), " dez/2019=", round(100*s19, digits=2), "%  dez/2025=",
            round(100*s25, digits=2), "%  Δp.p.=", round(100*(s25-s19), digits=2))
end

println("\nConcluído.")
