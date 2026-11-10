using Test
using Statistics
using Random

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito
using .ChainLadderCredito.ChainLadderCore
using .ChainLadderCredito.VintageSim
using .ChainLadderCredito.ApcModel

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
        # σ²_1..σ²_8 publicados por Mack (1993, Tabela 1). σ²_9 (última coluna,
        # só uma safra no par) é o próprio resultado da regra de Mack para a
        # última coluna aplicada a σ²_7 e σ²_8: 0,447 (ver teste seguinte).
        sigma2_publicado = [160.0, 37.7, 42.0, 15.2, 13.7, 8.19, 0.447, 1.15, 0.447] .* 1000

        msep_i, msep_total = mack_mseps(tri, f_raw, sigma2_publicado)
        se_total = sqrt(msep_total)
        # Mack (1993, Tabela 3) reporta erro padrão total = 13% da reserva total (18681 mil);
        # replicações usuais do exemplo (pacote R ChainLadder) chegam a ≈2.447.095.
        @test se_total ≈ 2_447_095 rtol=1e-3
    end

    @testset "Regra de Mack (1993) para a última coluna de σ² (Taylor/Ashe)" begin
        # σ²_7 = 0,447 e σ²_8 = 1,15 (Mack, 1993, Tabela 1, em milhares).
        # Regra de Mack para a coluna final sem grau de liberdade:
        # σ²_9 = min(σ²_8, σ²_7, σ²_8²/σ²_7) = min(1,15; 0,447; 2,959) = 0,447.
        sigma2_parcial = [160.0, 37.7, 42.0, 15.2, 13.7, 8.19, 0.447, 1.15]
        sigma2_9 = mack_classical_last_sigma2(sigma2_parcial)
        @test sigma2_9 ≈ 0.447 atol=1e-6
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

    @testset "Cauda log-linear ajustada só nas últimas colunas (viés de avaliação antecipada)" begin
        # Triângulo determinístico gerado por uma curva de maturação em
        # sino (h(j) ∝ j^1.6 exp(-j/5), igual à usada na simulação de
        # safras), sem ruído. O fator f_k verdadeiro, nesse caso, NÃO é
        # log-linear globalmente: log(f_k - 1) decai muito mais rápido nas
        # idades iniciais (hazard ainda subindo até o pico em j=8) do que
        # nas idades maduras (hazard já em queda). Um triângulo com poucas
        # safras maduras só revela as colunas 1-8 (ainda no trecho que
        # antecede o pico): ajustar a reta de cauda com TODO esse trecho
        # extrapola a inclinação íngreme do início, subestimando a cauda.
        # Restringir o ajuste às colunas mais próximas da fronteira
        # (`max_pontos_ajuste`) reduz esse viés.
        shape_a, shape_b = 1.6, 5.0
        maxk = 16
        hazard_raw = [(j^shape_a) * exp(-j / shape_b) for j in 1:maxk]
        hazard = hazard_raw ./ sum(hazard_raw)
        cum_teorico = cumsum(hazard)
        f_teorico = [cum_teorico[k + 1] / cum_teorico[k] for k in 1:(maxk - 1)]

        n_cohorts = 9
        max_dev = 14
        tri = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            idade_max = n_cohorts + 1 - i  # 9,8,...,1: triângulo padrão
            tri[i, 1] = 100.0
            for k in 1:(idade_max - 1)
                tri[i, k + 1] = tri[i, k] * f_teorico[k]
            end
        end
        f_raw, s, n = dev_factors(tri)
        @test count(isnan, f_raw) == 5  # colunas 9-13 sem nenhum par observado

        f_full, = extrapolate_factor_tail(f_raw, n; max_pontos_ajuste=100)  # equivalente ao comportamento antigo (todo o histórico)
        f_janela, = extrapolate_factor_tail(f_raw, n; max_pontos_ajuste=4)
        f_padrao, = extrapolate_factor_tail(f_raw, n)  # max_pontos_ajuste=6 por padrão

        for k in 9:12
            erro_full = abs(f_full[k] - f_teorico[k])
            erro_janela = abs(f_janela[k] - f_teorico[k])
            @test erro_janela < 0.5 * erro_full
        end
        @test f_padrao != f_full  # o padrão (janela de 6) já restringe o ajuste, não é mais "todo o histórico"
    end

    @testset "Bootstrap no triângulo de Taylor/Ashe reproduz erro padrão publicado" begin
        # Mesmo triângulo do teste de Mack acima. A projeção de cada réplica
        # bootstrap deve partir da diagonal reamostrada (pseudo_cum[i, k_i]),
        # não da diagonal observada do triângulo original (England; Verrall,
        # 1999/2002) -- usar a diagonal observada descarta a incerteza de
        # parâmetro que a reamostragem do corpo gerou para a última idade
        # conhecida de cada safra e sub-estima o erro padrão da reserva.
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
        f, s, n = dev_factors(tri)
        reservas_cl, = bootstrap_reserves(tri, f; n_boot=20000, seed=1)

        # England (2002) / pacote R ChainLadder (BootChainLadder) relatam um
        # erro de previsão do bootstrap ODP em torno de 2,95-3,0 milhões
        # (~16% da reserva de 18.681 mil) para este triângulo.
        @test std(reservas_cl) ≈ 2_950_000 rtol=0.05
        @test mean(reservas_cl) ≈ 18_681_000 rtol=0.02
    end

    @testset "Bootstrap não explode com triângulo raso (cauda extrapolada fixa por réplica)" begin
        # Triângulo bem raso: só 10 das 23 colunas têm par observado (13
        # extrapoladas), com apenas 2 safras na coluna mais frágil -- o
        # mesmo padrão do corte de 2016-12-31 do backtest, onde reajustar a
        # curva de cauda a cada réplica reamostrada fazia a reserva
        # bootstrap explodir (média de 56,1bi contra 20,45bi do CL
        # determinístico, cauda direita de centenas de bilhões).
        shape_a, shape_b = 1.6, 5.0
        max_dev = 24
        hazard_raw = [(j^shape_a) * exp(-j / shape_b) for j in 1:max_dev]
        hazard = hazard_raw ./ sum(hazard_raw)
        cum_teorico = cumsum(hazard)
        f_teorico = [cum_teorico[k + 1] / cum_teorico[k] for k in 1:(max_dev - 1)]

        n_cohorts = 10
        tri = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            idade_max = n_cohorts + 1 - i  # 10,9,...,1
            tri[i, 1] = 1e9
            for k in 1:(idade_max - 1)
                tri[i, k + 1] = tri[i, k] * f_teorico[k]
            end
        end

        f_raw, s, n = dev_factors(tri)
        @test count(!isnan, f_raw) == 9  # só colunas 1-9 têm par observado
        @test n[8] == 2  # a coluna mais frágil usada no ajuste (min_pairs=2)
        @test n[9] == 1  # só 1 par, abaixo de min_pairs -- observada mas fora do ajuste

        f, = extrapolate_factor_tail(f_raw, n)
        k_vec = [last_observed_col(tri, i) for i in 1:n_cohorts]
        tcol = fill(max_dev, n_cohorts)
        reservas_cl, = bootstrap_reserves(tri, f; n_boot=500, seed=1, target_col=tcol)

        proj = project_chain_ladder(tri, f)
        reserva_cl_det = sum(proj[i, max_dev] - tri[i, k_vec[i]] for i in 1:n_cohorts if k_vec[i] < max_dev)

        # a média bootstrap deve ficar na vizinhança do CL determinístico, não
        # ordens de magnitude acima (o bug fazia a média passar de 2,5x o
        # determinístico, com réplicas isoladas chegando a centenas de vezes)
        @test mean(reservas_cl) / reserva_cl_det < 1.5
        @test maximum(reservas_cl) / reserva_cl_det < 10
    end

    @testset "ApcModel: fit_apc_calendar recupera idade+safra+beta exatos sem ruído" begin
        # Triângulo determinístico: incremento = exp(intercepto + safra_efeito[i] +
        # idade_efeito[j] + beta*fator[t]), sem ruído algum. `fator` é uma função
        # não-linear do índice de calendário (seno + tendência linear fraca), escolhida
        # deliberadamente para não ser combinação linear de idade/safra -- é essa
        # não-linearidade que evita a colinearidade exata idade+safra=período (ver
        # docstring de `fit_apc_calendar`). Com dado gerado exatamente pelo modelo, o
        # IRLS deve recuperar os parâmetros a menos de erro numérico de ponto flutuante.
        n_cohorts = 7
        max_dev = 7
        cohort_true = [0.0, 0.3, -0.2, 0.5, -0.4, 0.1, 0.2]
        age_true = [0.0, 0.6, 0.9, 0.5, 0.2, -0.1, -0.3]
        beta_true = 0.8
        intercept_true = log(1000.0)
        fator = [sin(t / 2.3) + 0.1 * t for t in 1:20]

        calendar_index = zeros(Int, n_cohorts, max_dev)
        incr = fill(NaN, n_cohorts, max_dev)
        cum = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts, j in 1:(n_cohorts + 1 - i)
            t = i + j - 1
            calendar_index[i, j] = t
            incr[i, j] = exp(intercept_true + cohort_true[i] + age_true[j] + beta_true * fator[t])
        end
        for i in 1:n_cohorts
            acc = 0.0
            for j in 1:max_dev
                isnan(incr[i, j]) && continue
                acc += incr[i, j]
                cum[i, j] = acc
            end
        end

        fit = fit_apc_calendar(cum, calendar_index, fator)
        @test fit.converged
        @test fit.beta ≈ beta_true atol=1e-6
        @test fit.intercept ≈ intercept_true atol=1e-6
        @test all(isapprox.(fit.cohort_effect, cohort_true; atol=1e-6))
        @test all(isapprox.(fit.age_effect, age_true; atol=1e-6))
    end

    @testset "ApcModel: recuperação de beta_s multi-segmento (Vasicek/IRB)" begin
        # simulate_multi_segment_triangles com beta_s nos dois extremos: beta=0 (sem
        # exposição ao fator comum, só ruído idiossincrático) e beta=1 (inteiramente
        # dirigido pelo fator comum, sem ruído). O beta estimado por fit_apc_calendar,
        # usando o MESMO fator padronizado usado na simulação, deve ficar perto dos
        # extremos verdadeiros -- não exatamente, porque ainda há ruído Gama no
        # incremento (cv_incremento), mas a ordem de grandeza e o sinal não podem falhar.
        scr_taxa = 0.05 .+ 0.02 .* sin.((1:60) ./ 5) .+ 0.001 .* (1:60)
        n_cohorts = 20
        betas_true = [0.0, 1.0]
        segs, z = simulate_multi_segment_triangles(scr_taxa; betas=betas_true, n_cohorts=n_cohorts,
            max_dev=16, start_idx=8, base_volume_r=fill(1e9, 60), seed=4242)

        fit0 = fit_apc_calendar(segs[1].cum, segs[1].calendar_index, z)
        fit1 = fit_apc_calendar(segs[2].cum, segs[2].calendar_index, z)

        @test abs(fit0.beta) < 0.25          # beta_true = 0,0: estimado deve ficar perto de zero
        @test fit1.beta > 0.8                 # beta_true = 1,0: estimado deve ficar perto de um
        @test fit1.beta > fit0.beta + 0.5      # ordenação correta entre os dois segmentos
    end

    @testset "ApcModel: diagonal_anova mantém calendar_effect_test (regressão)" begin
        # calendar_effect_test refatorado para delegar a diagonal_anova -- confere que
        # os quatro valores devolvidos continuam idênticos aos calculados diretamente
        # pela ANOVA, no mesmo triângulo determinístico do teste de cauda acima.
        shape_a, shape_b = 1.6, 5.0
        maxk = 16
        hazard_raw = [(j^shape_a) * exp(-j / shape_b) for j in 1:maxk]
        hazard = hazard_raw ./ sum(hazard_raw)
        cum_teorico = cumsum(hazard)
        f_teorico = [cum_teorico[k + 1] / cum_teorico[k] for k in 1:(maxk - 1)]
        n_cohorts = 9
        max_dev = 14
        tri = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            idade_max = n_cohorts + 1 - i
            tri[i, 1] = 100.0
            for k in 1:(idade_max - 1)
                tri[i, k + 1] = tri[i, k] * f_teorico[k]
            end
        end
        f_raw, s, n = dev_factors(tri)
        f, = extrapolate_factor_tail(f_raw, n)
        resid, = pearson_residuals(tri, f)

        f1, p1, dfb1, dfw1 = calendar_effect_test(tri, f)
        f2, p2, dfb2, dfw2, frac2 = diagonal_anova(n_cohorts, max_dev, resid)
        @test f1 ≈ f2
        @test isnan(p1) == isnan(p2) || p1 ≈ p2
        @test dfb1 == dfb2
        @test dfw1 == dfw2
        @test 0.0 <= frac2 <= 1.0
    end

end

println("Todos os testes passaram.")
