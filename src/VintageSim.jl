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
do SCR.data usado para modular aquela célula. Células fora do horizonte
real disponível (depois do último trimestre da série de calibração)
ficam como `NaN` em `cum`/`incr`: a safra existe, mas não há dado real
para simular aquela idade, e isso não é opcional, é consequência de a
safra ter sido originada tarde demais para madurar inteiramente dentro
da janela de calendário coberta pela série real.
"""
struct SimTriangle
    n_cohorts::Int
    max_dev::Int
    start_idx::Int
    horizon_idx::Int
    cum::Matrix{Float64}
    incr::Matrix{Float64}
    volume::Vector{Float64}
    calendar_index::Matrix{Int}
    max_age_within_horizon::Vector{Int}
end

"""
    maturation_hazard(max_dev, shape_a, shape_b)

Curva paramétrica de maturação: o incremento de inadimplência nova, como
fração do saldo originado, sobe nos primeiros trimestres e decai depois,
na forma h(j) ∝ j^a · exp(-j/b) (uma gama não normalizada no índice de
idade, não uma log-normal). Normalizada para somar 1 ao longo de
`max_dev` trimestres (a escala final é aplicada separadamente via
`h_ultimate`). Os parâmetros `shape_a` e `shape_b` são hipóteses de
formato assumidas para este exercício, não valores calibrados em dado de
safra real (que não existe publicamente, como o artigo discute).
"""
function maturation_hazard(max_dev::Int, shape_a::Float64, shape_b::Float64)
    j = 1:max_dev
    raw = (j .^ shape_a) .* exp.(-(j) ./ shape_b)
    return raw ./ sum(raw)
end

"""
    simulate_vintage_triangle(scr_taxa; kwargs...)

Constrói o triângulo de safra proxy, com originação trimestral contínua
de `start_idx` até `start_idx + n_cohorts - 1` (ambos índices na série
real `scr_taxa`). `scr_taxa` é a série real de inadimplência agregada
(SCR.data) usada para calcular o multiplicador de estresse macro de cada
trimestre-calendário. O nível de maturação final (`h_ultimate`) é
hipotetizado como `dilution_factor` vezes a taxa média da série real
completa, para refletir que a taxa de estoque (SCR.data) dilui a perda
acumulada de safra (efeito de diluição discutido no texto). Nenhum dos
parâmetros (`dilution_factor`, `shape_a`, `shape_b`, `cv_incremento`,
`origination_share`) foi estimado a partir de dado de safra real: são
hipóteses declaradas, escolhidas para produzir uma trajetória plausível,
não uma calibração no sentido estatístico do termo. O saldo inadimplente
simulado também não é perda final: não há modelo de LGD (loss given
default) nem de recuperação aplicado, só o saldo que entrou em atraso
acima de 90 dias.

Cada safra só é simulada até a idade `min(max_dev, length(scr_taxa) -
orig_idx + 1)`: não há dado real de calendário além do último ponto de
`scr_taxa`, então safras originadas perto do fim da série não alcançam a
maturidade completa dentro do horizonte simulado. `max_age_within_horizon`
guarda essa idade máxima por safra.
"""
function simulate_vintage_triangle(scr_taxa::Vector{Float64};
        n_cohorts::Int, max_dev::Int=24, start_idx::Int=8,
        dilution_factor::Float64=2.2, shape_a::Float64=1.6, shape_b::Float64=5.0,
        base_volume_r::Vector{Float64}, origination_share::Float64=0.15,
        cv_incremento::Float64=0.15, seed::Int=20261102)

    horizon_idx = length(scr_taxa)
    @assert start_idx + n_cohorts - 1 <= horizon_idx "Safras originadas além da série real disponível"

    rng = MersenneTwister(seed)

    mean_rate = mean(scr_taxa)
    multiplicador = scr_taxa ./ mean_rate
    h_ultimate = dilution_factor * mean_rate
    hazard = maturation_hazard(max_dev, shape_a, shape_b) .* h_ultimate

    volume = [origination_share * base_volume_r[start_idx + i - 1] for i in 1:n_cohorts]

    cum = fill(NaN, n_cohorts, max_dev)
    incr = fill(NaN, n_cohorts, max_dev)
    calendar_index = fill(0, n_cohorts, max_dev)
    max_age = fill(0, n_cohorts)

    for i in 1:n_cohorts
        orig_idx = start_idx + i - 1
        age_cap = min(max_dev, horizon_idx - orig_idx + 1)
        max_age[i] = age_cap
        acc = 0.0
        for j in 1:age_cap
            t = orig_idx + (j - 1)
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

    return SimTriangle(n_cohorts, max_dev, start_idx, horizon_idx, cum, incr, volume,
                        calendar_index, max_age)
end

end # module
