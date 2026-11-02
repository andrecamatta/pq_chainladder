module ScrCalibration

using CSV
using DataFrames
using Dates

export load_scr_series, ScrSeries

"""
Série trimestral real de inadimplência agregada (crédito pessoal sem
consignação, PF) extraída do SCR.data do Banco Central do Brasil
(set/2012 em diante, primeira data-base trimestral disponível na amostra
baixada). Usada só como índice de estresse macro para modular a curva de
maturação da safra simulada, não como triângulo de safra em si.
"""
struct ScrSeries
    data_base::Vector{Date}
    carteira_ativa::Vector{Float64}
    carteira_inadimplencia::Vector{Float64}
    numero_operacoes::Vector{Float64}
    taxa::Vector{Float64}
end

function load_scr_series(path::AbstractString)
    df = CSV.read(path, DataFrame; delim=';', decimal='.')
    if eltype(df.data_base) != Date
        df.data_base = Date.(df.data_base)
    end
    sort!(df, :data_base)
    taxa = df.carteira_inadimplencia ./ df.carteira_ativa
    return ScrSeries(df.data_base, df.carteira_ativa, df.carteira_inadimplencia,
                      df.numero_operacoes, taxa)
end

end # module
