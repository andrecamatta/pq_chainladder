module VintageSim

using Random
using Distributions
using Statistics

export SimTriangle, simulate_vintage_triangle, fator_calendario_bcb, padronizar,
       simulate_multi_segment_triangles

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

"""
    fator_calendario_bcb(scr_taxa; defasagem=4)

Variação homóloga (ano sobre ano, 4 trimestres) da série real do SCR.data:
`g_t = log(scr_taxa_t / scr_taxa_{t-defasagem})`. Por construção remove nível e
tendência de baixa frequência da série -- ao contrário do multiplicador `scr_taxa_t /
mean(scr_taxa)` usado em `simulate_vintage_triangle` (um desvio de nível, não de
variação), o que torna esta a covariável adequada para representar choque de
calendário (ciclo), não deslocamento permanente de nível. É a variável proposta no
plano de reenquadramento (seção 3, caminho (c)) para ancorar o termo de calendário do
modelo idade-safra-período. As primeiras `defasagem` posições ficam `NaN` (sem par
defasado disponível).
"""
function fator_calendario_bcb(scr_taxa::Vector{Float64}; defasagem::Int=4)
    T = length(scr_taxa)
    g = fill(NaN, T)
    for t in (defasagem + 1):T
        g[t] = log(scr_taxa[t] / scr_taxa[t - defasagem])
    end
    return g
end

"""
    padronizar(x)

Padroniza um vetor (possivelmente com `NaN`) para média 0 e desvio-padrão 1, calculados
só sobre as entradas não-`NaN`. As entradas `NaN` permanecem `NaN`.
"""
function padronizar(x::Vector{Float64})
    obs = filter(!isnan, x)
    mu, sd = mean(obs), std(obs)
    return [isnan(v) ? NaN : (v - mu) / sd for v in x]
end

"""
    simulate_multi_segment_triangles(scr_taxa; betas, n_cohorts, max_dev=24, start_idx=8,
                                      kwargs...)

Extensão multi-segmento de `simulate_vintage_triangle`: um `SimTriangle` por segmento
`s`, todos compartilhando o mesmo fator comum de calendário padronizado `z_t`
(`fator_calendario_bcb` seguido de `padronizar`), cada um com carga `betas[s]` sobre
esse fator -- paralelo direto com a correlação de ativos do modelo de Vasicek (1987,
2002) e da fórmula *Internal Ratings-Based* (IRB) de Basileia (Gordy, 2003):

    log(multiplicador_{s,t}) = beta_s * z_t + sqrt(1 - beta_s^2) * epsilon_{s,t}

com `epsilon_{s,t} ~ N(0,1)` i.i.d. por segmento e por trimestre, independente entre
segmentos e independente de `z_t`. Por construção `Var(log multiplicador_{s,t}) = 1`
para qualquer `beta_s` em `[0,1]` (decomposição exata de variância, não aproximada), e
a fração da variância do log-multiplicador explicada pelo fator comum é `beta_s^2` --
o número comparado contra o `beta_s` estimado por `ApcModel.fit_apc_calendar` em cada
segmento sintético.

Diferença deliberada em relação a `simulate_vintage_triangle`: lá o multiplicador é o
desvio de NÍVEL `scr_taxa_t / mean(scr_taxa)` (a mesma série real, sem transformação de
variação); aqui o fator comum é a variação homóloga PADRONIZADA. A razão é evitar uma
incompatibilidade de escala entre o que é simulado e o que é depois estimado: a mesma
transformação (`fator_calendario_bcb` + `padronizar`) usada para gerar o choque de
calendário é a usada como covariável em `ApcModel.fit_apc_calendar`, para que o
exercício de recuperação de `beta_s` (seção 5 do plano) compare o estimado com o
mesmo objeto que gerou os dados, não com uma proxy transformada de outra forma.
`beta_s=1,0`: segmento inteiramente dirigido pelo fator comum, sem ruído
idiossincrático. `beta_s=0,0`: segmento sem nenhuma exposição ao calendário comum.

Devolve `(segmentos, z)`, em que `segmentos` é um `Vector{SimTriangle}` (um por
`betas[s]`, na mesma ordem) e `z` é o fator comum padronizado usado (vetor do mesmo
comprimento de `scr_taxa`, com `NaN` substituído por `0,0` -- a média do fator
padronizado -- nos primeiros `defasagem_calendario` trimestres sem par homólogo
disponível; irrelevante no pipeline de produção porque `start_idx` é sempre maior que
`defasagem_calendario`, mas precisa de um valor definido para não propagar `NaN`).
"""
function simulate_multi_segment_triangles(scr_taxa::Vector{Float64};
        betas::Vector{Float64}, n_cohorts::Int, max_dev::Int=24, start_idx::Int=8,
        dilution_factor::Float64=2.2, shape_a::Float64=1.6, shape_b::Float64=5.0,
        base_volume_r::Vector{Float64}, origination_share::Float64=0.15,
        cv_incremento::Float64=0.15, defasagem_calendario::Int=4, seed::Int=20261102)

    horizon_idx = length(scr_taxa)
    @assert start_idx + n_cohorts - 1 <= horizon_idx "Safras originadas além da série real disponível"
    @assert all(b -> 0.0 <= b <= 1.0, betas) "beta_s precisa estar em [0,1] para a decomposição de variância ser exata"

    g = fator_calendario_bcb(scr_taxa; defasagem=defasagem_calendario)
    z = padronizar(g)
    z_full = [isnan(v) ? 0.0 : v for v in z]

    mean_rate = mean(scr_taxa)
    h_ultimate = dilution_factor * mean_rate
    hazard = maturation_hazard(max_dev, shape_a, shape_b) .* h_ultimate
    volume = [origination_share * base_volume_r[start_idx + i - 1] for i in 1:n_cohorts]

    rng = MersenneTwister(seed)
    segmentos = SimTriangle[]
    for beta_s in betas
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
                eps = randn(rng)
                log_mult = beta_s * z_full[t] + sqrt(1 - beta_s^2) * eps
                multiplicador = exp(log_mult)
                media = volume[i] * hazard[j] * multiplicador
                media = max(media, 1e-6)
                shape_gamma = 1 / cv_incremento^2
                scale_gamma = media / shape_gamma
                inc = rand(rng, Gamma(shape_gamma, scale_gamma))
                acc += inc
                incr[i, j] = inc
                cum[i, j] = acc
            end
        end
        push!(segmentos, SimTriangle(n_cohorts, max_dev, start_idx, horizon_idx, cum, incr,
                                      volume, calendar_index, max_age))
    end
    return segmentos, z_full
end

end # module
