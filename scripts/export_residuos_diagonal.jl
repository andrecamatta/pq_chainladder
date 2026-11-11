import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Statistics
using Dates
using LinearAlgebra

# Reaproveita exatamente a mesma reconstrução e o mesmo ajuste da seção 1 de
# run_apc_pipeline.jl (mesma semente, mesmos parâmetros) para exportar, além do
# F-stat já salvo em apc_diagonal_antes_depois.csv, o resíduo médio por diagonal de
# calendário antes/depois do termo beta*fator -- insumo da figura nova (seção 3.4 do
# esqueleto do artigo). Não reestima nada, não muda nenhum resultado já gravado.
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito
using .ChainLadderCredito.ScrCalibration
using .ChainLadderCredito.VintageSim
using .ChainLadderCredito.ChainLadderCore
using .ChainLadderCredito.ApcModel

const HERE = @__DIR__
const DATA = joinpath(HERE, "..", "data")
const RESULTS = joinpath(HERE, "..", "results")

scr = load_scr_series(joinpath(DATA, "scr_credito_pessoal_sem_consignacao.csv"))
MAX_DEV = 24
START_IDX = 8
idx_fim = findfirst(==(Date(2024, 12, 31)), scr.data_base)
N_COHORTS = idx_fim - START_IDX + 1

sim = simulate_vintage_triangle(scr.taxa; n_cohorts=N_COHORTS, max_dev=MAX_DEV, start_idx=START_IDX,
                                 dilution_factor=2.2, shape_a=1.6, shape_b=5.0,
                                 base_volume_r=scr.carteira_ativa, origination_share=0.15,
                                 cv_incremento=0.15, seed=20261102)

Z = padronizar(fator_calendario_bcb(scr.taxa))

f_raw_full, s_full, n_full = dev_factors(sim.cum)
f_full, = extrapolate_factor_tail(f_raw_full, n_full)
resid_antes, = pearson_residuals(sim.cum, f_full)

fit_full = fit_apc_calendar(sim.cum, sim.calendar_index, Z)
resid_depois = fit_full.resid

por_diag_antes = Dict{Int, Vector{Float64}}()
por_diag_depois = Dict{Int, Vector{Float64}}()
for i in 1:sim.n_cohorts, j in 1:sim.max_dev
    ci = sim.calendar_index[i, j]
    ra = resid_antes[i, j]
    rd = resid_depois[i, j]
    if !isnan(ra)
        push!(get!(por_diag_antes, ci, Float64[]), ra)
    end
    if !isnan(rd)
        push!(get!(por_diag_depois, ci, Float64[]), rd)
    end
end

ids_ordenados = sort(collect(keys(por_diag_antes)))
df_resid_diag = DataFrame(
    indice_calendario=ids_ordenados,
    data_calendario=[1 <= ci <= length(scr.data_base) ? scr.data_base[ci] : missing for ci in ids_ordenados],
    n_celulas=[length(por_diag_antes[ci]) for ci in ids_ordenados],
    resid_medio_antes=[mean(por_diag_antes[ci]) for ci in ids_ordenados],
    resid_rms_antes=[sqrt(mean(por_diag_antes[ci] .^ 2)) for ci in ids_ordenados],
    resid_medio_depois=[haskey(por_diag_depois, ci) ? mean(por_diag_depois[ci]) : NaN for ci in ids_ordenados],
    resid_rms_depois=[haskey(por_diag_depois, ci) ? sqrt(mean(por_diag_depois[ci] .^ 2)) : NaN for ci in ids_ordenados],
)
CSV.write(joinpath(RESULTS, "apc_residuos_diagonal.csv"), df_resid_diag)
println("Gravado ", joinpath(RESULTS, "apc_residuos_diagonal.csv"), " (", nrow(df_resid_diag), " diagonais)")
println(df_resid_diag)
