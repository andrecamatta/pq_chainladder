module ApcModel

using Statistics
using LinearAlgebra
using Random
using Distributions
using ..ChainLadderCore
using ..VintageSim
using ..Backtest

export fit_apc_calendar, project_apc_calendar, estimate_free_period_effects,
       decompose_calendar_deviation, montecarlo_apc_reserve

"""
    fit_apc_calendar(tri, calendar_index, fator; max_iter=50, tol=1e-8)

Ajuste log-linear idade + safra + β·fator de calendário, via quase-Poisson
(GLM Poisson sobredisperso, log link, IRLS) sobre o incremento observado do
triângulo -- a mesma família já usada no bootstrap ODP de `ChainLadderCore`
(Renshaw; Verrall, 1998; England; Verrall, 1999/2002), estendida com um terceiro
termo: `log(mu[i,j]) = intercepto + safra_efeito[i] + idade_efeito[j] + beta *
fator[calendar_index[i,j]]`.

**Identificação.** Idade + safra = período é colinearidade exata só quando o termo de
período é um conjunto de efeitos fixos LIVRES, um por trimestre-calendário (caminho
(a)/(b) do plano de reenquadramento). Aqui o termo de período é reduzido a um único
parâmetro (`beta`) multiplicando uma função conhecida e não-linear de uma série externa
(`fator`, tipicamente `VintageSim.fator_calendario_bcb` padronizado -- a variação
homóloga da série do BCB, não um efeito fixo por trimestre). Como essa função não é uma
combinação linear dos efeitos de idade e de safra (não é da forma `a*i + b*j`), a
matriz de desenho não fica exatamente colinear, e `beta` é identificável junto com os
efeitos de idade e de safra (cada um com uma categoria de referência, a usual para
variáveis categóricas). Isto é uma ESCOLHA DE MODELAGEM (plano, seção 1: "não é o
procedimento canônico de identificação em APC"), não a parametrização canônica de
Kuang-Nielsen-Nielsen -- ela resolve o problema de estimação ao custo de só conseguir
atribuir ao calendário a parte da variação que covaria com `fator`; um choque de
calendário ortogonal a `fator` não seria capturado por `beta` e apareceria como
resíduo, o que é exatamente o que `calendar_effect_test`/`diagonal_anova` sobre os
resíduos deste ajuste (o "depois") testa.

Usa só as células observadas (incremento não-`NaN`, positivo, com
`calendar_index[i,j] > 0` e `fator` definido naquele índice). Devolve um `NamedTuple`
com `intercept`, `cohort_effect` (vetor de comprimento `n_cohorts`, primeira entrada
fixada em 0 como categoria de referência), `age_effect` (vetor de comprimento
`max_dev`, idem), `beta`, `se_beta` (erro-padrão aproximado via
`phi * inv(X'WX)`, W = diag(mu) da última iteração), `fitted_incr`, `resid`
(resíduos de Pearson, só nas células observadas), `phi`, `n_obs`, `dof`, `converged`.

Só cria uma coluna de referência (*dummy*) para safras e idades que de fato têm pelo
menos uma célula observada (`cohorts_presentes`, `idades_presentes` no retorno) --
não para as `n_cohorts`/`max_dev` dimensões completas do triângulo. Isso evita
desperdiçar graus de liberdade com níveis sem nenhum dado (relevante nos cortes mais
rasos do backtest, como 2016-12-31, em que a maior parte das `n_cohorts` safras do
triângulo completo ainda não foi originada ou não tem nenhuma célula observável
naquela data de avaliação). `cohort_effect`/`age_effect` continuam vetores de
comprimento `n_cohorts`/`max_dev` para indexação direta por `i`/`j`, com `0,0` nos
níveis sem dado -- ver `project_apc_calendar` para a extrapolação (plana, mais simples
que a do chain ladder clássico) usada nesses níveis ao projetar para frente.
"""
function fit_apc_calendar(tri::AbstractMatrix{<:Real}, calendar_index::Matrix{Int},
                           fator::Vector{Float64}; max_iter::Int=50, tol::Float64=1e-8)
    n_cohorts, max_dev = size(tri)
    incr = to_incremental(tri)
    obs = Tuple{Int,Int}[]
    for i in 1:n_cohorts, j in 1:max_dev
        isnan(incr[i, j]) && continue
        incr[i, j] <= 0 && continue
        ci = calendar_index[i, j]
        ci == 0 && continue
        (ci > length(fator) || isnan(fator[ci])) && continue
        push!(obs, (i, j))
    end
    n_obs = length(obs)
    cohorts_presentes = sort(unique(i for (i, _) in obs))
    idades_presentes = sort(unique(j for (_, j) in obs))
    # 1º nível (k=1) de cada lista é a categoria de referência (efeito 0,0 por
    # definição); os demais (k=2,3,...) ocupam as colunas de dummy 1,2,... (k-1).
    cohort_col = Dict(c => k - 1 for (k, c) in enumerate(cohorts_presentes) if k > 1)
    age_col = Dict(a => k - 1 for (k, a) in enumerate(idades_presentes) if k > 1)
    n_cohort_dummy = max(length(cohorts_presentes) - 1, 0)
    n_age_dummy = max(length(idades_presentes) - 1, 0)
    p = 1 + n_cohort_dummy + n_age_dummy + 1
    n_obs <= p && error("poucas observações ($n_obs) para $p parâmetros (idade+safra+calendário)")

    X = zeros(n_obs, p)
    y = zeros(n_obs)
    for (row, (i, j)) in enumerate(obs)
        X[row, 1] = 1.0
        haskey(cohort_col, i) && (X[row, 1 + cohort_col[i]] = 1.0)
        haskey(age_col, j) && (X[row, 1 + n_cohort_dummy + age_col[j]] = 1.0)
        X[row, end] = fator[calendar_index[i, j]]
        y[row] = incr[i, j]
    end

    beta_coef = zeros(p)
    beta_coef[1] = log(mean(y))
    mu = exp.(X * beta_coef)
    converged = false
    for _ in 1:max_iter
        eta = X * beta_coef
        mu = exp.(eta)
        w_sqrt = sqrt.(mu)
        z = eta .+ (y .- mu) ./ mu
        beta_new = (X .* w_sqrt) \ (z .* w_sqrt)
        delta = maximum(abs.(beta_new .- beta_coef))
        beta_coef = beta_new
        if delta < tol
            converged = true
            break
        end
    end
    eta = X * beta_coef
    mu = exp.(eta)
    resid_obs = (y .- mu) ./ sqrt.(max.(mu, 1e-6))
    dof = max(n_obs - p, 1)
    phi = sum(resid_obs .^ 2) / dof

    XtWX = X' * (mu .* X)
    cov_approx = phi .* pinv(XtWX)
    se_beta = sqrt(max(cov_approx[end, end], 0.0))

    cohort_effect = fill(0.0, n_cohorts)
    for (k, c) in enumerate(cohorts_presentes)
        k > 1 && (cohort_effect[c] = beta_coef[1 + cohort_col[c]])
    end
    age_effect = fill(0.0, max_dev)
    for (k, a) in enumerate(idades_presentes)
        k > 1 && (age_effect[a] = beta_coef[1 + n_cohort_dummy + age_col[a]])
    end

    fitted_incr = fill(NaN, n_cohorts, max_dev)
    resid = fill(NaN, n_cohorts, max_dev)
    for (row, (i, j)) in enumerate(obs)
        fitted_incr[i, j] = mu[row]
        resid[i, j] = resid_obs[row]
    end

    return (intercept=beta_coef[1], cohort_effect=cohort_effect, age_effect=age_effect,
            beta=beta_coef[end], se_beta=se_beta, fitted_incr=fitted_incr, resid=resid,
            phi=phi, n_obs=n_obs, dof=dof, converged=converged,
            cohorts_presentes=cohorts_presentes, idades_presentes=idades_presentes)
end

"""
    project_apc_calendar(fit, calendar_index, fator)

Projeta o incremento esperado em todas as células (`exp(intercepto + safra_efeito[i] +
idade_efeito[j] + beta*fator[t])`), inclusive fora da janela usada para ajustar `fit`
-- é o que permite ao chamador decidir se usa valores de `fator` realizados (a
situação real: a série do SCR.data é publicada quase em tempo real, então os
trimestres entre a data de avaliação e a data de corte do calendário já são
conhecidos quando chegam, não precisam ser previstos) ou hipotéticos. Células com
`calendar_index == 0` ou `fator` indefinido (`NaN`) ficam `NaN`.

Para idades `j` fora de `fit.idades_presentes` (idades ainda não observadas em
nenhuma safra na data do ajuste -- a mesma situação que `extrapolate_factor_tail`
resolve para o chain ladder clássico), usa o efeito de idade da idade mais próxima
DENTRO de `fit.idades_presentes` (extrapolação plana, mais simples que a extrapolação
log-linear do chain ladder clássico -- declarado aqui como simplificação da Fase 1,
não como escolha definitiva: uma cauda log-linear análoga para o modelo (c) é
trabalho futuro). Para safras `i` fora de `fit.cohorts_presentes`, o efeito fica 0,0 por
definição (não deveria ocorrer em uso normal: toda safra com célula a projetar tem,
por construção, pelo menos uma célula observada mais nova).
"""
function project_apc_calendar(fit, calendar_index::Matrix{Int}, fator::Vector{Float64})
    n_cohorts, max_dev = size(calendar_index)
    proj = fill(NaN, n_cohorts, max_dev)
    idades = fit.idades_presentes
    age_min, age_max = isempty(idades) ? (1, max_dev) : (minimum(idades), maximum(idades))
    for i in 1:n_cohorts, j in 1:max_dev
        t = calendar_index[i, j]
        t == 0 && continue
        (t > length(fator) || isnan(fator[t])) && continue
        j_efetivo = clamp(j, age_min, age_max)
        proj[i, j] = exp(fit.intercept + fit.cohort_effect[i] + fit.age_effect[j_efetivo] + fit.beta * fator[t])
    end
    return proj
end

"""
    estimate_free_period_effects(tri, f)

Efeito de período LIVRE (um por diagonal, sem amarrar a nenhuma série externa),
estimado só a partir do triângulo: agrupa os resíduos de Pearson do chain ladder
clássico (idade+safra, `ChainLadderCore.pearson_residuals`) por diagonal
(`d = i + j`) e devolve a média de cada grupo, centrada na média geral dos resíduos
(restrição de identificação de soma zero sobre as diagonais observadas -- caminho (a)
do plano de reenquadramento). Esta é a estimativa a comparar DEPOIS, como exercício de
recuperação declarado, contra a série do BCB -- não usa `fator` nem qualquer série
externa na própria estimação, evitando a circularidade de amarrar o efeito de
calendário à mesma série usada para construir a simulação (ver `VintageSim`).

Devolve `(diag_ids, delta_hat, n_por_diagonal)`: `diag_ids[k] = d` é o índice local de
diagonal (`i + j`); a conversão para o índice real na série de calendário é
`indice_calendario = start_idx + d - 2` (ver `VintageSim.SimTriangle`, já que
`calendar_index[i,j] = start_idx + i + j - 2`).
"""
function estimate_free_period_effects(tri::AbstractMatrix{<:Real}, f::Vector{Float64})
    n_cohorts, max_dev = size(tri)
    resid, = pearson_residuals(tri, f)
    grupos = Dict{Int, Vector{Float64}}()
    for i in 1:n_cohorts, j in 1:max_dev
        isnan(resid[i, j]) && continue
        d = i + j
        push!(get!(grupos, d, Float64[]), resid[i, j])
    end
    diag_ids = sort(collect(keys(grupos)))
    grand_mean = mean(vcat(values(grupos)...))
    delta_hat = [mean(grupos[d]) - grand_mean for d in diag_ids]
    n_por_diagonal = [length(grupos[d]) for d in diag_ids]
    return diag_ids, delta_hat, n_por_diagonal
end

"""
    decompose_calendar_deviation(sim, valuation_idx)

Decomposição contábil (não uma nova estimação) do desvio de um corte do backtest:
projeta o chain ladder clássico (idade+safra, cauda extrapolada, igual à produção) a
partir da diagonal de `valuation_idx`, e compara célula a célula com o valor
REALIZADO na simulação (`sim.cum`/`sim.incr`, conhecido porque isto é um exercício de
diagnóstico sobre dado simulado, não uma previsão) para cada célula do leque de
previsão (idade acima da última observada, dentro de `sim.max_age_within_horizon`).
O resíduo em R\$ de cada célula (`realizado - ajustado`) é devolvido junto com o índice
real de calendário da célula. `erro_total` (a soma de todos os resíduos) é exatamente
`reserva_real - reserva_cl` (identidade contábil, sinal oposto ao de
`erro_cl_pct = 100*(reserva_cl - reserva_real)/reserva_real` já reportado pelo
pipeline de produção -- confira o sinal ao comparar os dois).

Agrupando esses resíduos por diagonal (`indice_calendario`), a soma de TODOS os
resíduos de uma diagonal já é, por definição, a média da diagonal vezes o número de
células -- ou seja, ao somar o erro agregado de uma safra para outra dentro da mesma
avaliação, o componente idiossincrático de cada célula (o desvio em torno da média da
própria diagonal) se CANCELA exatamente na soma (´sum_j (x_j - mean) = 0` por
definição), sobrando só o componente compartilhado entre safras. Isso não é uma
limitação da decomposição: é o próprio ponto que a Seção 3 do plano de reenquadramento
quer tornar visível -- o erro AGREGADO de uma reserva (somada sobre muitas safras) só
pode vir do componente sistemático, porque ruído idiossincrático genuíno se diversifica
na soma (o mesmo mecanismo da correlação de ativos de Vasicek/IRB: risco idiossincrático
não é precificado porque se diversifica numa carteira granular, só o fator comum
permanece). O que VARIA de uma célula para outra dentro da mesma diagonal (medido por
`ChainLadderCore.diagonal_anova` sobre a matriz de resíduos célula-a-célula, não sobre a
soma) é informativo sobre o tamanho relativo dos dois componentes na variância, mesmo
que não seja na soma agregada.
"""
function decompose_calendar_deviation(sim::SimTriangle, valuation_idx::Int)
    tri = truncate_at_valuation(sim, valuation_idx)
    f_raw, s, n = dev_factors(tri)
    f, tail_coef = extrapolate_factor_tail(f_raw, n)
    fitted_cum = fitted_retro_triangle(tri, f)
    fitted_incr = to_incremental(fitted_cum)
    actual_incr = to_incremental(sim.cum)
    k_vec = [last_observed_col(tri, i) for i in 1:sim.n_cohorts]
    tcol = sim.max_age_within_horizon

    cohort = Int[]
    idade = Int[]
    indice_calendario = Int[]
    residuo_dolar = Float64[]
    for i in 1:sim.n_cohorts
        k_vec[i] == 0 && continue
        for j in (k_vec[i] + 1):tcol[i]
            ci = sim.calendar_index[i, j]
            ci == 0 && continue
            (isnan(actual_incr[i, j]) || isnan(fitted_incr[i, j])) && continue
            push!(cohort, i); push!(idade, j); push!(indice_calendario, ci)
            push!(residuo_dolar, actual_incr[i, j] - fitted_incr[i, j])
        end
    end

    return (cohort=cohort, idade=idade, indice_calendario=indice_calendario,
            residuo_dolar=residuo_dolar, erro_total=sum(residuo_dolar))
end

"""
    montecarlo_apc_reserve(fit, calendar_index, fator, k_vec, tcol; n_boot=1000, seed=20261102)

Distribuição preditiva Monte Carlo da reserva do modelo (c) (idade+safra+β·fator),
projetando cada célula futura (`j` entre `k_vec[i]+1` e `tcol[i]`) com
`mu[i,j] = project_apc_calendar(fit, calendar_index, fator)[i,j]` e somando uma
amostra `Gamma(mu[i,j]/phi, phi)` por célula (mesma família usada na etapa de
incerteza de processo de `ChainLadderCore.bootstrap_reserves`).

**Limitação declarada, importante para a leitura da cobertura medida:** os parâmetros
do modelo (`intercept`, `cohort_effect`, `age_effect`, `beta`) ficam FIXOS no valor
estimado da amostra observada -- esta função soma só incerteza de PROCESSO (a
variabilidade Gama em torno da média ajustada), não a incerteza de ESTIMAÇÃO dos
próprios parâmetros, ao contrário do bootstrap do chain ladder clássico
(`bootstrap_reserves`), que reestima os fatores de desenvolvimento a cada réplica
reamostrando o corpo do triângulo. Isso tende a SUBESTIMAR a incerteza total do
modelo (c) e, portanto, a INFLAR a cobertura medida do intervalo resultante -- a
comparação "com calendário" vs. "sem calendário" (`bootstrap_reserves`,
que soma incerteza de processo E de parâmetro) não está em igualdade de condições, e
deve ser lida como um limite otimista para o modelo (c), não uma comparação justa.
Fazer o modelo (c) reestimar os parâmetros a cada réplica (como o chain ladder
clássico já faz) é a extensão natural para uma Fase 2, não implementada aqui por
custo computacional (um ajuste IRLS por réplica, multiplicado por milhares de
réplicas e centenas de sementes).
"""
function montecarlo_apc_reserve(fit, calendar_index::Matrix{Int}, fator::Vector{Float64},
                                 k_vec::Vector{Int}, tcol::Vector{Int};
                                 n_boot::Int=1000, seed::Int=20261102)
    mu = project_apc_calendar(fit, calendar_index, fator)
    n_cohorts = length(k_vec)
    mus = Float64[]
    for i in 1:n_cohorts
        k_vec[i] == 0 && continue
        for j in (k_vec[i] + 1):tcol[i]
            isnan(mu[i, j]) && continue
            push!(mus, mu[i, j])
        end
    end
    phi = fit.phi
    shapes = max.(mus ./ phi, 1e-8)
    rng = MersenneTwister(seed)
    reservas = fill(0.0, n_boot)
    for b in 1:n_boot
        total = 0.0
        for shp in shapes
            total += rand(rng, Gamma(shp, phi))
        end
        reservas[b] = total
    end
    return reservas, sum(mus)
end

end # module
