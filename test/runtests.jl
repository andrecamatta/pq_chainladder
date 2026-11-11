using Test
using Statistics
using Random

include(joinpath(@__DIR__, "..", "src", "ChainLadderCredito.jl"))
using .ChainLadderCredito
using .ChainLadderCredito.ChainLadderCore
using .ChainLadderCredito.VintageSim
using .ChainLadderCredito.Backtest
using .ChainLadderCredito.ApcModel
using .ChainLadderCredito.FatorComum

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

    @testset "ApcModel: extrapolate_age_effect_tail recupera decaimento linear sem ruído" begin
        # age_effect verdadeiro EXATAMENTE linear em j (b_true=-0.3, com age_true[1]=0
        # -- a categoria de referência de `fit_apc_calendar` é sempre fixada em 0, então
        # escolher a "verdade" já com age_true[1]=0 evita qualquer ambiguidade de
        # deslocamento, igual ao teste de recuperação exata acima) sobre TODO o
        # max_dev, mas só as idades 1..7 são observadas (triângulo padrão, n_cohorts=7
        # < max_dev=10) -- a mesma situação de um corte antecipado do backtest, em que
        # idades 8,9,10 ainda não foram observadas por nenhuma safra.
        n_cohorts = 7
        max_dev = 10
        b_true = -0.3
        age_true = [b_true * (j - 1) for j in 1:max_dev]  # age_true[1] = 0
        cohort_true = [0.0, 0.3, -0.2, 0.5, -0.4, 0.1, 0.2]
        beta_true = 0.8
        intercept_true = log(1000.0)
        fator = [sin(t / 2.3) + 0.1 * t for t in 1:20]

        calendar_index = zeros(Int, n_cohorts, max_dev)
        for i in 1:n_cohorts, j in 1:max_dev
            calendar_index[i, j] = i + j - 1
        end
        full_incr = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts, j in 1:max_dev
            t = calendar_index[i, j]
            full_incr[i, j] = exp(intercept_true + cohort_true[i] + age_true[j] + beta_true * fator[t])
        end
        k_vec = [n_cohorts + 1 - i for i in 1:n_cohorts]  # triângulo padrão: só idades 1..(n_cohorts+1-i) observadas
        cum = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            acc = 0.0
            for j in 1:k_vec[i]
                acc += full_incr[i, j]
                cum[i, j] = acc
            end
        end

        fit = fit_apc_calendar(cum, calendar_index, fator)
        @test fit.converged
        @test fit.idades_presentes == collect(1:n_cohorts)  # idade máxima observada = 7 (safra 1)
        @test all(isapprox.(fit.age_effect[1:n_cohorts], age_true[1:n_cohorts]; atol=1e-6))

        age_full, (a_hat, b_hat) = extrapolate_age_effect_tail(fit.age_effect, fit.idades_presentes, max_dev)
        @test b_hat ≈ b_true atol=1e-6
        @test a_hat ≈ -b_true atol=1e-6  # age_effect[j] = b*(j-1) = b*j - b => intercepto da reta = -b
        for j in (n_cohorts + 1):max_dev
            @test age_full[j] ≈ age_true[j] atol=1e-6
        end
        # e o ponto projetado herda essa extrapolação (não mais o fallback plano)
        proj = project_apc_calendar(fit, calendar_index, fator)
        @test proj[1, max_dev] ≈ exp(intercept_true + cohort_true[1] + age_true[max_dev] + beta_true * fator[max_dev]) atol=1e-3
    end

    @testset "ApcModel: extrapolate_age_effect_tail cai no fallback plano (poucos pontos ou inclinação não-negativa)" begin
        # (a) menos de 2 idades elegíveis: cai no fallback plano, devolve (NaN,NaN).
        # `age_effect` precisa ter comprimento `max_dev` (como `fit.age_effect` sempre
        # tem, por construção de `fit_apc_calendar`) -- só a idade 2 está "presente".
        age_effect_1pt = [0.0, 0.7, 0.0, 0.0, 0.0]
        age_full_1pt, coefs_1pt = extrapolate_age_effect_tail(age_effect_1pt, [2], 5)
        @test isnan(coefs_1pt[1]) && isnan(coefs_1pt[2])
        @test age_full_1pt[3] == age_full_1pt[4] == age_full_1pt[5] == age_effect_1pt[2]  # plano, repete a idade 2

        # (b) age_effect CRESCENTE nas últimas idades (inclinação b>0): mesmo guarda de
        # sinal de `extrapolate_factor_tail` (um efeito que cresce para sempre, em vez
        # de decair, contradiz a forma de hazard em sino que motiva a extrapolação) --
        # cai no fallback plano em vez de extrapolar um crescimento sem fundamento.
        idades = collect(1:6)
        age_effect_crescente = vcat([0.1 * j for j in 1:6], fill(0.0, 4))  # estritamente crescente até a idade 6
        age_full_cresc, coefs_cresc = extrapolate_age_effect_tail(age_effect_crescente, idades, 10)
        @test isnan(coefs_cresc[1]) && isnan(coefs_cresc[2])
        @test all(age_full_cresc[j] == age_effect_crescente[6] for j in 7:10)
    end

    @testset "ApcModel: fit_ar1_fator e simulate_fator_future nunca leem o fator depois do corte" begin
        # Duas séries idênticas até valuation_idx=15, mas com os valores DEPOIS do
        # corte completamente diferentes (uma crescente, outra com ruído extremo) --
        # se a função lesse esses valores (vazamento), os parâmetros do AR(1) ou o
        # caminho simulado variariam entre as duas; devem ser IDÊNTICOS, porque nenhuma
        # das duas funções deve olhar além de `valuation_idx`.
        valuation_idx = 15
        historico = [0.05 * sin(t / 3) + 0.01 * t for t in 1:valuation_idx]
        fator_a = vcat(historico, [100.0 * t for t in 1:10])         # futuro absurdamente alto
        fator_b = vcat(historico, [-100.0 * t for t in 1:10])        # futuro absurdamente baixo (e diferente de a)
        @test fator_a[1:valuation_idx] == fator_b[1:valuation_idx]   # mesma história, por construção
        @test fator_a[(valuation_idx + 1):end] != fator_b[(valuation_idx + 1):end]

        ca, phia, siga = fit_ar1_fator(fator_a, valuation_idx)
        cb, phib, sigb = fit_ar1_fator(fator_b, valuation_idx)
        @test ca ≈ cb atol=1e-12
        @test phia ≈ phib atol=1e-12
        @test siga ≈ sigb atol=1e-12

        rng_a = MersenneTwister(2026)
        rng_b = MersenneTwister(2026)
        path_a = simulate_fator_future(fator_a, valuation_idx, 22, ca, phia, siga, rng_a)
        path_b = simulate_fator_future(fator_b, valuation_idx, 22, cb, phib, sigb, rng_b)
        @test path_a[(valuation_idx + 1):end] ≈ path_b[(valuation_idx + 1):end] atol=1e-12
        # e nenhum dos dois caminhos simulados chega perto da escala do "futuro" de
        # entrada (100*t ou -100*t) -- confirma que o caminho é GERADO, não copiado.
        @test maximum(abs.(path_a[(valuation_idx + 1):end])) < 50.0
    end

    @testset "ApcModel: Monte Carlo com reestimação de parâmetro é mais largo que o de só processo" begin
        # Triângulo realista com ruído de processo genuíno (VintageSim, cv_incremento
        # padrão), um corte intermediário. O Monte Carlo antigo (só processo, mantido
        # em `montecarlo_apc_reserve_processo` para esta comparação) soma só a
        # variabilidade Gama em torno da média ajustada; o novo (`montecarlo_apc_reserve`)
        # também reestima idade+safra+beta por réplica e simula o fator futuro -- deve
        # produzir um intervalo SEMPRE mais largo (ou igual), nunca mais estreito.
        scr_taxa = 0.05 .+ 0.02 .* sin.((1:60) ./ 5) .+ 0.001 .* (1:60)
        n_cohorts = 20
        sim = simulate_vintage_triangle(scr_taxa; n_cohorts=n_cohorts, max_dev=16, start_idx=8,
            dilution_factor=2.2, shape_a=1.6, shape_b=5.0, base_volume_r=fill(1e9, 60),
            origination_share=0.15, cv_incremento=0.15, seed=777)
        Z = padronizar(fator_calendario_bcb(scr_taxa))
        v = 20
        tri = truncate_at_valuation(sim, v)
        k_vec = [last_observed_col(tri, i) for i in 1:sim.n_cohorts]
        tcol = sim.max_age_within_horizon
        fit0 = fit_apc_calendar(tri, sim.calendar_index, Z)
        @test fit0.phi > 0  # ruído de processo genuíno nesta simulação

        reservas_proc, = montecarlo_apc_reserve_processo(fit0, sim.calendar_index, Z, k_vec, tcol;
                                                           n_boot=500, seed=1)
        reservas_novo, = montecarlo_apc_reserve(tri, sim.calendar_index, Z, v, k_vec, tcol;
                                                 n_boot=500, seed=1)
        @test std(reservas_novo) > 2 * std(reservas_proc)
    end

    @testset "ApcModel: Monte Carlo colapsa para o ponto quando não há ruído nenhum" begin
        # Triângulo determinístico (incremento = exp(apc exato), sem ruído de processo)
        # e fator de calendário CONSTANTE na história pré-corte (sem variância -- o AR(1)
        # cai no fallback de ruído branco com sigma=0, ver `fit_ar1_fator`): nem o
        # bootstrap de resíduos (todos ~0) nem a simulação do fator futuro (sigma=0) têm
        # de onde tirar variabilidade. O Monte Carlo deve colapsar para um ponto, só com
        # ruído de arredondamento de ponto flutuante -- prova de que a dispersão medida
        # no teste anterior vem genuinamente do ruído dos dados, não de algum artefato
        # numérico interno (ex.: o bug de floor inconsistente do Gamma corrigido nesta
        # fase, que fazia a reserva EXPLODIR mesmo sem ruído nenhum).
        n_cohorts = 7
        max_dev = 10
        cohort_true = [0.0, 0.3, -0.2, 0.5, -0.4, 0.1, 0.2]
        age_true = [0.0, 0.6, 0.9, 0.5, 0.2, -0.1, -0.3, -0.5, -0.7, -0.9]
        beta_true = 0.0
        intercept_true = log(1000.0)
        fator = fill(0.1, 20)  # história constante -> AR(1) com sigma=0

        calendar_index = zeros(Int, n_cohorts, max_dev)
        for i in 1:n_cohorts, j in 1:max_dev
            calendar_index[i, j] = i + j - 1
        end
        full_incr = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts, j in 1:max_dev
            t = calendar_index[i, j]
            full_incr[i, j] = exp(intercept_true + cohort_true[i] + age_true[j] + beta_true * fator[t])
        end
        k_vec = [n_cohorts + 1 - i for i in 1:n_cohorts]
        cum = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            acc = 0.0
            for j in 1:k_vec[i]
                acc += full_incr[i, j]
                cum[i, j] = acc
            end
        end
        tcol = fill(max_dev, n_cohorts)
        valuation_idx = n_cohorts

        reservas, pontual = montecarlo_apc_reserve(cum, calendar_index, fator, valuation_idx, k_vec, tcol;
                                                     n_boot=300, seed=1)
        @test mean(reservas) > 0
        @test std(reservas) / mean(reservas) < 1e-4  # colapso: só ruído de ponto flutuante
        @test pontual ≈ mean(reservas) rtol=1e-4
    end

    @testset "FatorComum: estimar_beta_r2 recupera beta e R²=beta² em casos exatos" begin
        rng = MersenneTwister(99)
        f = randn(rng, 200)
        @test estimar_beta_r2(f, f) == (1.0, 1.0)              # y = f exatamente
        @test estimar_beta_r2(-f, f) == (-1.0, 1.0)            # y = -f: beta=-1, R²=1 (sinal não afeta R²)
        beta_hat, r2_hat = estimar_beta_r2(2.0 .* f .+ 5.0, f)  # transformação afim não muda beta/R² (ambos padronizados)
        @test beta_hat ≈ 1.0 atol=1e-9
        @test r2_hat ≈ 1.0 atol=1e-9
        @test beta_hat^2 ≈ r2_hat atol=1e-12                    # R² = beta² é identidade algébrica, não aproximação
    end

    @testset "FatorComum: fator_pca_loo recupera beta_s em série sintética com fator comum conhecido" begin
        # Seis séries com o MESMO fator comum f e betas conhecidos nos dois extremos e
        # no meio -- paralelo direto do teste de beta_s multi-segmento já existente em
        # ApcModel/VintageSim, agora para o fator construído aqui (PCA leave-one-out)
        # em vez do beta_s usado para SIMULAR o triângulo.
        rng = MersenneTwister(2026)
        T = 300
        betas_true = [0.0, 0.3, 0.6, 0.9, 0.6, 0.3]
        f_true = randn(rng, T)
        Y = hcat([betas_true[i] .* f_true .+ sqrt(1 - betas_true[i]^2) .* randn(rng, T)
                  for i in eachindex(betas_true)]...)
        Z = hcat([padronizar_serie(Y[:, i]) for i in axes(Y, 2)]...)

        beta_hat = Float64[]
        for i in eachindex(betas_true)
            f_loo = fator_pca_loo(Z, i)
            bh, r2h = estimar_beta_r2(Z[:, i], f_loo)
            push!(beta_hat, bh)
            @test r2h ≈ bh^2 atol=1e-12
        end
        @test abs(beta_hat[1]) < 0.25           # beta_true = 0,0
        # beta_hat[4] (beta_true=0,9) fica ATENUADO em relação ao verdadeiro: o fator
        # leave-one-out é a PCA de só 5 colunas (as outras), não o f_true exato, então
        # correlaciona com f_true de forma imperfeita (corr < 1) -- um viés de
        # atenuação esperado e estrutural (quanto menos colunas restantes e mais fraco
        # o beta delas, maior a atenuação), não um bug de implementação.
        @test beta_hat[4] > 0.55
        @test issorted(abs.(beta_hat[[1, 2, 3]]))  # ordem preservada entre 0, 0.3, 0.6
    end

    @testset "FatorComum: fator_pca_loo / fator_media_simples_loo / fator_media_ponderada_loo nunca leem a coluna i (sem vazamento)" begin
        # Garantia de ausência de vazamento, mesmo padrão já usado para fit_ar1_fator:
        # substituir a coluna i por lixo (valores arbitrários, inclusive NaN-livres mas
        # absurdos) não pode mudar o fator leave-one-out calculado para aquele i, porque
        # a função nunca deveria olhar para essa coluna.
        rng = MersenneTwister(7)
        T, N = 120, 5
        Z1 = hcat([padronizar_serie(randn(rng, T)) for _ in 1:N]...)
        Z2 = copy(Z1)
        Z2[:, 2] = padronizar_serie(1000.0 .* randn(rng, T) .+ 1e6)  # lixo arbitrário na coluna 2
        pesos = [1.0, 999.0, 2.0, 3.0, 4.0]  # peso da coluna 2 também irrelevante para o fator de i=2

        @test fator_pca_loo(Z1, 2) ≈ fator_pca_loo(Z2, 2)
        @test fator_media_simples_loo(Z1, 2) ≈ fator_media_simples_loo(Z2, 2)
        @test fator_media_ponderada_loo(Z1, pesos, 2) ≈ fator_media_ponderada_loo(Z2, pesos, 2)

        # e, inversamente, mudar qualquer OUTRA coluna muda o fator de i=2 (checagem de
        # que o teste acima não passa trivialmente porque a função ignora tudo)
        Z3 = copy(Z1)
        Z3[:, 3] = padronizar_serie(randn(rng, T))
        @test !isapprox(fator_pca_loo(Z1, 2), fator_pca_loo(Z3, 2))
    end

    @testset "FatorComum: bootstrap_beta_r2 cobre o beta verdadeiro com frequência razoável" begin
        # Checagem de cobertura leve (não um Monte Carlo de 200 sementes como o do
        # modelo APC): 40 réplicas independentes de uma série curta (T=50, comparável à
        # amostra real de 53 variações trimestrais), beta verdadeiro fixo, IC90% via
        # bootstrap em blocos -- a cobertura empírica não deveria ficar muito abaixo do
        # nominal, nem o IC deveria colapsar para largura zero.
        rng = MersenneTwister(555)
        T = 50
        beta_true = 0.5
        n_rep = 40
        acertos = 0
        larguras = Float64[]
        for _ in 1:n_rep
            f = randn(rng, T)
            y = beta_true .* f .+ sqrt(1 - beta_true^2) .* randn(rng, T)
            betas_b, r2s_b = bootstrap_beta_r2(y, f; n_boot=1000, bloco=8, seed=rand(rng, 1:10^6))
            lo, hi = intervalo_percentil(betas_b; alpha=0.10)
            push!(larguras, hi - lo)
            (lo <= beta_true <= hi) && (acertos += 1)
        end
        @test acertos / n_rep > 0.5   # bem abaixo do nominal 0,90, só para travar regressão grosseira
        @test minimum(larguras) > 0.01
    end

    @testset "FatorComum: variacao_log e padronizar_serie" begin
        taxa = [0.05, 0.055, 0.05, 0.06]
        v = variacao_log(taxa)
        @test length(v) == 3
        @test v[1] ≈ log(0.055 / 0.05) atol=1e-12
        @test v[3] ≈ log(0.06 / 0.05) atol=1e-12

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        xs = padronizar_serie(x)
        @test mean(xs) ≈ 0.0 atol=1e-12
        @test std(xs) ≈ 1.0 atol=1e-12
    end

end

println("Todos os testes passaram.")
