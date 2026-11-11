module ScrModalidades

using CSV
using DataFrames
using Dates
using Statistics

export ModalidadeDef, MODALIDADES_PADRAO, build_scr_modalidades_csv, load_scr_modalidades,
       ScrModalidadesData

"""
Definição de uma modalidade agregada do SCR.data: `nome` (chave curta usada nos
resultados), `cliente` (`"PF"`, `"PJ"` ou `nothing` para somar os dois) e `pred`, um
predicado `(modalidade, submodalidade) -> Bool` sobre as colunas brutas do SCR.data que
decide se uma linha pertence a essa modalidade agregada.

A correspondência é por SUBSTRING (via `occursin`, case-insensitive), não por string
exata, porque a própria nomenclatura do SCR.data mudou dentro da amostra (set/2012 a
dez/2025) sem mudar o que a modalidade significa -- ex.: as faixas de prazo de "capital
de giro" eram "inferior/superior a 30 dias" até ~2016 e passaram a "até/superior a 365
dias" depois, com uma terceira categoria ("teto rotativo") aparecendo só depois de
~2018. Checado manualmente ano a ano (2012, 2015, 2018, 2020, 2022, 2024) antes de
escrever estes predicados -- ver `apc_fase1c_resumo.md`, seção 1.
"""
struct ModalidadeDef
    nome::String
    cliente::Union{String,Nothing}
    pred::Function
end

_norm(s::AbstractString) = lowercase(s)

"""
Sete modalidades: cinco de pessoa física (cartão de crédito, crédito pessoal
consignado, crédito pessoal sem consignação, veículos, financiamento imobiliário), uma
de pessoa jurídica (capital de giro) e uma combinando PF+PJ (rural, porque o crédito
rural do SCR.data inclui tanto produtor pessoa física quanto agronegócio pessoa
jurídica, e a série agregada usual do BCB não separa as duas).

Exclusões declaradas:
- "Antecipação de fatura de cartão de crédito" (modalidade "Direitos creditórios
  descontados") NÃO entra em `cartao_credito`: é um adiantamento sobre recebível de
  fatura (produto de desconto), não o saldo rotativo/parcelado do próprio cartão.
- `veiculos` usa só "Aquisição de bens - veículos automotores" (Financiamentos),
  excluindo arrendamento/leasing de veículos (produto contábil diferente, com
  submodalidade própria em "Operações de arrendamento").
- `imobiliario` usa só as duas submodalidades "Financiamento habitacional" (SFH e
  fora do SFH), excluindo "Financiamento imobiliário - empreendim., exceto habitac."
  (financiamento a incorporador/construtor, natureza de crédito diferente do
  financiamento ao mutuário final).
"""
const MODALIDADES_PADRAO = ModalidadeDef[
    ModalidadeDef("cartao_credito", "PF",
        (m, s) -> (m != "Direitos creditórios descontados" && occursin("cartão de crédito", _norm(s))) ||
                   s == "Crédito rotativo vinculado a cartão de crédito"),
    ModalidadeDef("pessoal_consignado", "PF",
        (m, s) -> m == "Empréstimos" && occursin("com consignação", _norm(s))),
    ModalidadeDef("pessoal_sem_consignacao", "PF",
        (m, s) -> m == "Empréstimos" && occursin("sem consignação", _norm(s))),
    ModalidadeDef("veiculos", "PF",
        (m, s) -> m == "Financiamentos" && s == "Aquisição de bens - veículos automotores"),
    ModalidadeDef("imobiliario", "PF",
        (m, s) -> m == "Financiamentos imobiliários" && occursin("habitacional", _norm(s))),
    ModalidadeDef("capital_giro", "PJ",
        (m, s) -> m == "Empréstimos" && occursin("capital de giro", _norm(s))),
    ModalidadeDef("rural", nothing,
        (m, s) -> occursin("financiamentos rurais", _norm(m))),
]

"""
    build_scr_modalidades_csv(raw_dir, out_path; modalidades=MODALIDADES_PADRAO)

Constrói a série agregada por modalidade a partir dos microdados BRUTOS do SCR.data
(um CSV por trimestre, ~65-100MB cada, baixados de
`https://www.bcb.gov.br/pda/desig/scrdata_<ano>.zip`, Portal de Dados Abertos do Banco
Central). `raw_dir` é o diretório onde esses CSVs já foram extraídos -- não incluído
neste repositório (grande demais; são dados públicos, basta rebaixar). Para cada
arquivo (um trimestre) e cada `ModalidadeDef`, filtra as linhas por `cliente` e `pred`
e soma `carteira_ativa`/`carteira_inadimplencia` sobre todas as outras dimensões (UF,
tipo de instituição, ocupação/CNAE, porte, origem, indexador) -- a mesma soma que o
BCB usaria para publicar uma série agregada por modalidade.

Grava um CSV pequeno (trimestre × modalidade, ~400 linhas) em `out_path`, o único
artefato desta função que entra no repositório.
"""
function build_scr_modalidades_csv(raw_dir::AbstractString, out_path::AbstractString;
        modalidades::Vector{ModalidadeDef}=MODALIDADES_PADRAO)
    isdir(raw_dir) || error(
        "diretório de dados brutos do SCR.data não encontrado: $raw_dir -- baixe os " *
        "arquivos trimestrais em https://www.bcb.gov.br/pda/desig/scrdata_<ano>.zip " *
        "(Portal de Dados Abertos do Banco Central)")
    arquivos = sort(filter(f -> endswith(f, ".csv"), readdir(raw_dir; join=true)))
    isempty(arquivos) && error("nenhum CSV encontrado em $raw_dir")

    linhas = NamedTuple{(:data_base, :modalidade, :carteira_ativa, :carteira_inadimplencia),
                         Tuple{Date,String,Float64,Float64}}[]
    for (k, f) in enumerate(arquivos)
        df = CSV.read(f, DataFrame; delim=';', decimal=',',
                       select=[:data_base, :cliente, :modalidade, :submodalidade,
                               :carteira_ativa, :carteira_inadimplencia], pool=true)
        data_base = df.data_base[1]
        for def in modalidades
            sub = def.cliente === nothing ? df : df[df.cliente .== def.cliente, :]
            mask = [def.pred(String(m), String(s)) for (m, s) in zip(sub.modalidade, sub.submodalidade)]
            n_matches = count(mask)
            n_matches == 0 && error(
                "modalidade '$(def.nome)' não casou nenhuma linha em $(basename(f)) " *
                "(data-base $data_base) -- a nomenclatura do SCR.data pode ter mudado " *
                "de forma não coberta pelo predicado; revise MODALIDADES_PADRAO")
            ca = sum(sub.carteira_ativa[mask])
            ci = sum(sub.carteira_inadimplencia[mask])
            push!(linhas, (data_base=data_base, modalidade=def.nome,
                            carteira_ativa=ca, carteira_inadimplencia=ci))
        end
        println("  [$k/$(length(arquivos))] $(basename(f)) ($data_base) agregado")
    end

    out = DataFrame(linhas)
    sort!(out, [:modalidade, :data_base])
    CSV.write(out_path, out; delim=';', decimal='.')
    return out
end

"""
Painel carregado de `data/scr_modalidades.csv`: `datas` (vetor de `Date`, comum a
todas as modalidades) e, por modalidade, `carteira_ativa`, `carteira_inadimplencia` e
`taxa = carteira_inadimplencia / carteira_ativa` -- a mesma métrica e definição usada
em `ScrCalibration.load_scr_series` (estoque, inadimplência acima de 90 dias sobre
carteira ativa, não uma taxa de formação/entrada em atraso).
"""
struct ScrModalidadesData
    datas::Vector{Date}
    nomes::Vector{String}
    carteira_ativa::Dict{String,Vector{Float64}}
    carteira_inadimplencia::Dict{String,Vector{Float64}}
    taxa::Dict{String,Vector{Float64}}
end

function load_scr_modalidades(path::AbstractString)
    df = CSV.read(path, DataFrame; delim=';', decimal='.')
    if eltype(df.data_base) != Date
        df.data_base = Date.(df.data_base)
    end
    datas = sort(unique(df.data_base))
    nomes = sort(unique(df.modalidade))
    ativa = Dict{String,Vector{Float64}}()
    inad = Dict{String,Vector{Float64}}()
    taxa = Dict{String,Vector{Float64}}()
    for nome in nomes
        sub = df[df.modalidade .== nome, :]
        sort!(sub, :data_base)
        sub.data_base == datas || error(
            "modalidade '$nome' não tem exatamente as mesmas datas-base que as demais " *
            "-- painel desbalanceado, não suportado por este carregador")
        ativa[nome] = sub.carteira_ativa
        inad[nome] = sub.carteira_inadimplencia
        taxa[nome] = sub.carteira_inadimplencia ./ sub.carteira_ativa
    end
    return ScrModalidadesData(datas, nomes, ativa, inad, taxa)
end

end # module
