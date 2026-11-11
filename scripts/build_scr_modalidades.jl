"""
Agrega os microdados brutos trimestrais do SCR.data (Banco Central do Brasil,
`https://www.bcb.gov.br/pda/desig/scrdata_<ano>.zip`) em `data/scr_modalidades.csv`:
uma linha por (data-base, modalidade), com `carteira_ativa` e `carteira_inadimplencia`
somadas sobre todas as outras dimensões do SCR.data (UF, tipo de instituição,
ocupação/CNAE, porte, origem, indexador).

Os CSVs brutos (~65-100MB cada, um por trimestre) NÃO estão neste repositório -- são
grandes demais e são dados públicos, então reproduzir exige rebaixá-los. Este script
lê de `SCR_RAW_DIR` (variável de ambiente) ou, se não definida, do caminho usado nesta
sessão de trabalho (já baixado para outro artigo da mesma série). Roda uma única vez;
o artefato pequeno gerado (`data/scr_modalidades.csv`) é o que fica versionado.

Uso: `julia --project=. scripts/build_scr_modalidades.jl`
"""

using DataFrames
include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito.ScrModalidades

raw_dir = get(ENV, "SCR_RAW_DIR", "/workspace/pq/artigos/panjer/pq_panjer/data/raw")
out_path = joinpath(@__DIR__, "..", "data", "scr_modalidades.csv")

println("== Agregando SCR.data bruto de $raw_dir ==")
println("Modalidades: ", join([d.nome for d in MODALIDADES_PADRAO], ", "))
out = build_scr_modalidades_csv(raw_dir, out_path)
println("Gravado em $out_path: ", nrow(out), " linhas (", length(unique(out.modalidade)),
        " modalidades × ", length(unique(out.data_base)), " trimestres)")
