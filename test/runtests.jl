import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using Statistics

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito
using .ChainLadderCredito.ChainLadderCore

@testset "ChainLadderCore" begin

    @testset "Chain ladder num triângulo didático (resultado à mão)" begin
        # safra 1 madura (idades 1,2,3), safra 2 com idade 3 a projetar,
        # safra 3 com idades 2 e 3 a projetar. f1 e f2 calculados à mão:
        # f1 = (150+165)/(100+110) = 315/210 = 1.5
        # f2 = 175/150 = 1.1666...
        tri = [100.0 150.0 175.0;
               110.0 165.0 NaN;
               105.0 NaN   NaN]
        f_raw, s, n = dev_factors(tri)
        @test f_raw[1] ≈ 1.5 atol=1e-9
        @test f_raw[2] ≈ 175 / 150 atol=1e-9
        f, = extrapolate_factor_tail(f_raw, n)
        @test f == f_raw  # sem coluna faltante aqui, a cauda não deve alterar nada

        proj = project_chain_ladder(tri, f)
        @test proj[2, 3] ≈ 165 * (175 / 150) atol=1e-6
        @test proj[3, 2] ≈ 105 * 1.5 atol=1e-6
        @test proj[3, 3] ≈ 105 * 1.5 * (175 / 150) atol=1e-6
    end

    @testset "Chain ladder no triângulo de Taylor/Ashe (Mack, 1993, Tabela 1)" begin
        # Dados acumulados exatos da Tabela 1 de Mack (1993), ASTIN Bulletin 23(2).
        raw_rows = [
            [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
            [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
            [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
            [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
            [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
            [396132, 1333217, 2180715, 2985752, 3691712],
            [440832, 1288463, 2419861, 3483130],
            [359480, 1421128, 2864498],
            [376686, 1363294],
            [344014],
        ]
        n_cohorts = 10
        tri = fill(NaN, n_cohorts, n_cohorts)
        for (i, row) in enumerate(raw_rows), (j, v) in enumerate(row)
            tri[i, j] = Float64(v)
        end

        f_raw, s, n = dev_factors(tri)
        f_publicado = [3.49, 1.75, 1.46, 1.174, 1.104, 1.086, 1.054, 1.077, 1.018]
        @test all(isapprox.(f_raw, f_publicado; atol=0.006))
        @test all(n .== [9, 8, 7, 6, 5, 4, 3, 2, 1])  # triângulo completo, sem coluna faltando

        f, tail_coef = extrapolate_factor_tail(f_raw, n)
        @test f == f_raw  # nenhuma coluna NaN neste triângulo completo

        proj = project_chain_ladder(tri, f)
        reservas_publicadas = Dict(2 => 95, 3 => 470, 4 => 710, 5 => 985, 6 => 1419,
                                    7 => 2178, 8 => 3920, 9 => 4279, 10 => 4626)
        reserva_total = 0.0
        for i in 2:n_cohorts
            k_i = last_observed_col(tri, i)
            r = proj[i, n_cohorts] - tri[i, k_i]
            reserva_total += r
            @test r / 1000 ≈ reservas_publicadas[i] rtol=0.02
        end
        @test reserva_total / 1000 ≈ 18681 rtol=0.002
    end

    @testset "Erro padrão de Mack no triângulo de Taylor/Ashe (σ² publicado)" begin
        raw_rows = [
            [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
            [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
            [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
            [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
            [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
            [396132, 1333217, 2180715, 2985752, 3691712],
            [440832, 1288463, 2419861, 3483130],
            [359480, 1421128, 2864498],
            [376686, 1363294],
            [344014],
        ]
        n_cohorts = 10
        tri = fill(NaN, n_cohorts, n_cohorts)
        for (i, row) in enumerate(raw_rows), (j, v) in enumerate(row)
            tri[i, j] = Float64(v)
        end
        f_raw, s, n = dev_factors(tri)
        # σ² publicado por Mack (1993, Tabela 1), em vez da extrapolação deste
        # pacote: isola o teste da aritmética do MSEP, sem depender de como a
        # cauda de σ² é extrapolada (que Mack resolveu com uma regra própria
        # só para a última coluna, não generalizável ao caso de várias colunas
        # faltantes que este pacote precisa tratar no backtest real).
        sigma2_publicado = [160.0, 37.7, 42.0, 15.2, 13.7, 8.19, 0.447, 1.15, 0.477] .* 1000

        msep_i, msep_total = mack_mseps(tri, f_raw, sigma2_publicado)
        se_total = sqrt(msep_total)
        # Mack (1993, Tabela 3) reporta erro padrão total = 13% da reserva total (18681 mil);
        # replicações usuais do exemplo (pacote R ChainLadder) chegam a ~2447 mil.
        @test 2200_000 <= se_total <= 2700_000
    end

    @testset "Extrapolação de cauda recupera padrão log-linear sem ruído" begin
        # gera um triângulo determinístico cujo fator verdadeiro é
        # f_k = 1 + 5*exp(-0.3k), e só revela os pares das primeiras colunas
        # (como acontece numa avaliação antecipada de verdade): as colunas
        # sem nenhum par devem ser recuperadas pela extrapolação.
        a_true, b_true = log(5.0), -0.3
        f_true = [1 + exp(a_true + b_true * k) for k in 1:8]

        n_cohorts = 5
        max_dev = 9
        tri = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            idade_max = n_cohorts + 1 - i  # 5,4,3,2,1: forma de triângulo padrão
            tri[i, 1] = 100.0
            for j in 1:(idade_max - 1)
                tri[i, j + 1] = tri[i, j] * f_true[j]
            end
        end

        f_raw, s, n = dev_factors(tri)
        @test count(isnan, f_raw) == 4  # colunas 5,6,7,8 sem nenhum par observado

        f_hat, (a_hat, b_hat) = extrapolate_factor_tail(f_raw, n; min_pairs=2)
        @test a_hat ≈ a_true atol=1e-6
        @test b_hat ≈ b_true atol=1e-6
        for k in 5:8
            @test f_hat[k] ≈ f_true[k] atol=1e-6
        end
    end

end

println("Todos os testes passaram.")
