module VintageSim

using Random
using Distributions
using Statistics

export SimTriangle, simulate_vintage_triangle

"""
Triângulo de safra simulado (proxy). Linha = safra de originação (coorte
trimestral), coluna = trimestres desde a originação. `cum` guarda o saldo
inadimplente (90+ dias) acumulado por coorte-idade; `incr` guarda os
incrementos trimestrais; `volume` guarda o volume originado de cada
coorte; `calendar_index` guarda, para cada célula, o índice na série real
do SCR.data usado para modular aquela célula.
"""
struct SimTriangle
    n_cohorts::Int
    max_dev::Int
    start_idx::Int
    cum::Matrix{Float64}
    incr::Matrix{Float64}
    volume::Vector{Float64}
    calendar_index::Matrix{Int}
end

"""
    maturation_hazard(max_dev, shape_a, shape_b)

Curva paramétrica de maturação (forma log-normal de hazard, tipicamente
usada para vintage curves de crédito de varejo): o incremento de
inadimplência nova, como fração do saldo originado, sobe nos primeiros
trimestres e decai depois. Normalizada para somar 1 ao longo de
`max_dev` trimestres (a escala final é aplicada separadamente via
`h_ultimate`).
"""
function maturation_hazard(max_dev::Int, shape_a::Float64, shape_b::Float64)
    j = 1:max_dev
    raw = (j .^ shape_a) .* exp.(-(j) ./ shape_b)
    return raw ./ sum(raw)
end

"""
    simulate_vintage_triangle(scr_taxa; kwargs...)

Constrói o triângulo de safra proxy. `scr_taxa` é a série real de
inadimplência agregada (SCR.data) usada para calcular o multiplicador de
estresse macro de cada trimestre-calendário. O nível de maturação final
(`h_ultimate`) é calibrado como `dilution_factor` vezes a taxa média da
série real, para refletir que a taxa de estoque (SCR.data) dilui a perda
acumulada de safra (efeito de diluição discutido no texto).
"""
function simulate_vintage_triangle(scr_taxa::Vector{Float64};
        n_cohorts::Int=24, max_dev::Int=24, start_idx::Int=8,
        dilution_factor::Float64=2.2, shape_a::Float64=1.6, shape_b::Float64=5.0,
        base_volume_r::Vector{Float64}, origination_share::Float64=0.15,
        cv_incremento::Float64=0.15, seed::Int=20261102)

    last_idx = start_idx + (n_cohorts - 1) + (max_dev - 1)
    @assert last_idx <= length(scr_taxa) "Série real curta demais para a janela pedida"

    rng = MersenneTwister(seed)

    mean_rate = mean(scr_taxa[start_idx:last_idx])
    multiplicador = scr_taxa ./ mean_rate
    h_ultimate = dilution_factor * mean_rate
    hazard = maturation_hazard(max_dev, shape_a, shape_b) .* h_ultimate

    volume = [origination_share * base_volume_r[start_idx + i - 1] for i in 1:n_cohorts]

    cum = fill(NaN, n_cohorts, max_dev)
    incr = fill(NaN, n_cohorts, max_dev)
    calendar_index = fill(0, n_cohorts, max_dev)

    for i in 1:n_cohorts
        acc = 0.0
        for j in 1:max_dev
            t = start_idx + (i - 1) + (j - 1)
            calendar_index[i, j] = t
            media = volume[i] * hazard[j] * multiplicador[t]
            media = max(media, 1e-6)
            shape_gamma = 1 / cv_incremento^2
            scale_gamma = media / shape_gamma
            inc = rand(rng, Gamma(shape_gamma, scale_gamma))
            acc += inc
            incr[i, j] = inc
            cum[i, j] = acc
        end
    end

    return SimTriangle(n_cohorts, max_dev, start_idx, cum, incr, volume, calendar_index)
end

end # module
