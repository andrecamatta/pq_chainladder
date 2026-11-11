module ApcModel

using Statistics
using LinearAlgebra
using Random
using Distributions
using ..ChainLadderCore
using ..VintageSim
using ..Backtest

export fit_apc_calendar, project_apc_calendar, estimate_free_period_effects,
       decompose_calendar_deviation, montecarlo_apc_reserve, montecarlo_apc_reserve_processo,
       extrapolate_age_effect_tail, fit_ar1_fator, simulate_fator_future

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
    extrapolate_age_effect_tail(age_effect, idades_presentes, max_dev; max_pontos_ajuste=6)

Extrapolação de cauda do efeito de idade do modelo (c), para idades `j` MAIORES que
`maximum(idades_presentes)` -- a mesma situação que `ChainLadderCore.extrapolate_factor_tail`
resolve para os fatores de desenvolvimento do chain ladder clássico, com o mesmo
critério (Fase 1b, corrigindo a simplificação plana da Fase 1 -- ver
`apc_fase1_resumo.md`, seção 2.4): ajusta uma reta em `age_effect[j]` contra `j`,
usando só as últimas `max_pontos_ajuste` idades elegíveis (as mais próximas da
fronteira de extrapolação, igual à janela de `extrapolate_factor_tail`), e força a
reta a ser não-crescente (`b <= 0`, mesmo guarda de sinal). Com menos de dois pontos
elegíveis ou inclinação positiva, cai no fallback plano (repete o efeito da idade
máxima observada) -- o mesmo fallback que ERA o comportamento único da Fase 1, usado
agora só como caso-limite, não como regra geral.

Diferença em relação a `extrapolate_factor_tail`: ali a reta é ajustada em
`log(f_k - 1)`, porque `f_k` é um fator multiplicativo e só o excesso sobre 1 decai
geometricamente. Aqui `age_effect[j]` já está na escala log (é o termo aditivo de
`log(mu[i,j])` no GLM), então a reta é ajustada DIRETAMENTE sobre `age_effect[j]`, sem
precisar de outro log: um `age_effect` que decai linearmente em `j` já corresponde a
um incremento esperado que decai geometricamente em `j` (`mu_{j+1}/mu_j = exp(b)`
constante), o análogo em escala log do que `f_k -> 1` representa em escala
multiplicativa.

Idades `j` MENORES que `minimum(idades_presentes)` (não deveria ocorrer em uso
normal -- ver docstring de `fit_apc_calendar`) continuam usando o fallback plano
(efeito da idade mínima observada): não há razão para supor decaimento geométrico
"para trás" no tempo de maturação, então não se estende a mesma extrapolação.

Devolve `(age_effect_full, (a, b))`, com `(a, b) = (NaN, NaN)` se o fallback plano foi
usado (incluindo o caso em que não há nada para extrapolar, `age_max >= max_dev`).
"""
function extrapolate_age_effect_tail(age_effect::Vector{Float64}, idades_presentes::Vector{Int},
                                      max_dev::Int; max_pontos_ajuste::Int=6)
    age_full = copy(age_effect)
    isempty(idades_presentes) && return age_full, (NaN, NaN)
    age_min, age_max = minimum(idades_presentes), maximum(idades_presentes)
    for j in 1:(age_min - 1)
        age_full[j] = age_effect[age_min]
    end
    age_max >= max_dev && return age_full, (NaN, NaN)

    flat_fallback = () -> begin
        for j in (age_max + 1):max_dev
            age_full[j] = age_effect[age_max]
        end
        return age_full, (NaN, NaN)
    end
    ks = length(idades_presentes) > max_pontos_ajuste ?
         idades_presentes[(end - max_pontos_ajuste + 1):end] : idades_presentes
    length(ks) < 2 && return flat_fallback()
    xs = Float64.(ks)
    ys = [age_effect[j] for j in ks]
    xbar, ybar = mean(xs), mean(ys)
    sxx = sum((xs .- xbar) .^ 2)
    sxx <= 0 && return flat_fallback()
    b = sum((xs .- xbar) .* (ys .- ybar)) / sxx
    b > 0 && return flat_fallback()
    a = ybar - b * xbar
    for j in (age_max + 1):max_dev
        age_full[j] = a + b * j
    end
    return age_full, (a, b)
end

"""
    project_apc_calendar(fit, calendar_index, fator)

Projeta o incremento esperado em todas as células (`exp(intercepto + safra_efeito[i] +
idade_efeito[j] + beta*fator[t])`), inclusive fora da janela usada para ajustar `fit`
-- é o que permite ao chamador decidir se usa valores de `fator` realizados (a
situação real: a série do SCR.data é publicada quase em tempo real, então os
trimestres entre a data de avaliação e a data de corte do calendário já são
conhecidos quando chegam, não precisam ser previstos) ou hipotéticos, inclusive
simulados (ver `montecarlo_apc_reserve`). Células com `calendar_index == 0` ou
`fator` indefinido (`NaN`) ficam `NaN`.

Para idades `j` fora de `fit.idades_presentes`, usa `extrapolate_age_effect_tail`
(decaimento log-linear acima da idade máxima observada, Fase 1b; fallback plano abaixo
da mínima). Para safras `i` fora de `fit.cohorts_presentes`, o efeito fica 0,0 por
definição (não deveria ocorrer em uso normal: toda safra com célula a projetar tem,
por construção, pelo menos uma célula observada mais nova).
"""
function project_apc_calendar(fit, calendar_index::Matrix{Int}, fator::Vector{Float64})
    n_cohorts, max_dev = size(calendar_index)
    proj = fill(NaN, n_cohorts, max_dev)
    age_effect_full, = extrapolate_age_effect_tail(fit.age_effect, fit.idades_presentes, max_dev)
    for i in 1:n_cohorts, j in 1:max_dev
        t = calendar_index[i, j]
        t == 0 && continue
        (t > length(fator) || isnan(fator[t])) && continue
        proj[i, j] = exp(fit.intercept + fit.cohort_effect[i] + age_effect_full[j] + fit.beta * fator[t])
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
    montecarlo_apc_reserve_processo(fit, calendar_index, fator, k_vec, tcol; n_boot=1000, seed=20261102)

Versão da Fase 1 (mantida só para comparação/regressão -- ver teste "Monte Carlo com
reestimação de parâmetro é mais largo que o de só processo"; não usada no pipeline de
produção do relatório de cobertura). Distribuição preditiva Monte Carlo da reserva do
modelo (c), projetando cada célula futura (`j` entre `k_vec[i]+1` e `tcol[i]`) com
`mu[i,j] = project_apc_calendar(fit, calendar_index, fator)[i,j]` e somando uma amostra
`Gamma(mu[i,j]/phi, phi)` por célula (mesma família usada na etapa de incerteza de
processo de `ChainLadderCore.bootstrap_reserves`).

**Limitação declarada (Fase 1, corrigida em `montecarlo_apc_reserve`, Fase 1b -- ver
`apc_fase1b_resumo.md`):** os parâmetros do modelo (`intercept`, `cohort_effect`,
`age_effect`, `beta`) ficam FIXOS no valor estimado da amostra observada -- esta
função soma só incerteza de PROCESSO, não a incerteza de ESTIMAÇÃO dos próprios
parâmetros nem a incerteza sobre o valor futuro de `fator` (que aqui é tratado como se
fosse conhecido com certeza, mesmo nas células do leque de previsão). Isso SUBESTIMA a
incerteza total do modelo (c) -- é exatamente a causa da cobertura de 0/9 medida na
Fase 1 (1,2% em 200 sementes), documentada em `apc_fase1_resumo.md`, seção 2.4.
"""
function montecarlo_apc_reserve_processo(fit, calendar_index::Matrix{Int}, fator::Vector{Float64},
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

"""
    fit_ar1_fator(fator, valuation_idx; min_obs=8)

Ajusta um AR(1) ingênuo (`fator_t = c + phi·fator_{t-1} + erro_t`) à série do fator de
calendário, por mínimos quadrados ordinários, usando SÓ os pares `(fator_{t-1},
fator_t)` com `t <= valuation_idx` -- nunca um valor realizado depois do corte. É essa
restrição que `simulate_fator_future` usa para simular o valor futuro do fator sem
vazamento (ver docstring de `montecarlo_apc_reserve`): o processo é ajustado só com
dado que já seria conhecido na data de avaliação.

Com menos de `min_obs` pares disponíveis (corte muito raso), variância amostral nula
nos regressores, ou `phi` fora de `[-0.98, 0.98]` (processo quase não-estacionário --
ajuste instável com poucos pontos, o mesmo tipo de guarda de sinal usado em
`extrapolate_factor_tail` para a inclinação da cauda), cai no fallback de ruído
branco: `phi=0`, `c` = média do histórico até o corte, `sigma` = desvio-padrão do
histórico até o corte. Isso é uma simplificação deliberada sobre a dinâmica do fator,
declarada, não escondida: um AR(1) não é o processo verdadeiro do ciclo de
inadimplência agregada (que não precisa ser identificado aqui) -- o objetivo é só
propagar incerteza genuína sobre o valor futuro do fator para o intervalo do modelo
(c), não prever o SCR.data.

Devolve `(c, phi, sigma)`. `sigma` é o desvio-padrão dos resíduos do ajuste (ou do
histórico, no fallback) -- usado por `simulate_fator_future` como a escala do ruído
gaussiano por passo.
"""
function fit_ar1_fator(fator::Vector{Float64}, valuation_idx::Int; min_obs::Int=8)
    hist = [fator[t] for t in 1:min(valuation_idx, length(fator)) if !isnan(fator[t])]
    fallback = () -> (isempty(hist) ? 0.0 : mean(hist), 0.0, isempty(hist) ? 0.0 : std(hist))
    pares = Tuple{Float64,Float64}[]
    for t in 2:min(valuation_idx, length(fator))
        (isnan(fator[t - 1]) || isnan(fator[t])) && continue
        push!(pares, (fator[t - 1], fator[t]))
    end
    length(pares) < min_obs && return fallback()
    xs = [p[1] for p in pares]
    ys = [p[2] for p in pares]
    xbar, ybar = mean(xs), mean(ys)
    sxx = sum((xs .- xbar) .^ 2)
    sxx <= 0 && return fallback()
    phi = sum((xs .- xbar) .* (ys .- ybar)) / sxx
    (phi < -0.98 || phi > 0.98) && return fallback()
    c = ybar - phi * xbar
    resid = ys .- (c .+ phi .* xs)
    dof = max(length(pares) - 2, 1)
    sigma = sqrt(sum(resid .^ 2) / dof)
    return c, phi, sigma
end

"""
    simulate_fator_future(fator, valuation_idx, max_t, c, phi, sigma, rng)

Simula um caminho do fator de calendário para os índices `valuation_idx+1` até
`max_t`, ancorado no último valor REALIZADO e conhecido na data de corte
(`fator[valuation_idx]`, ou o mais recente anterior não-`NaN`) e propagado pelo AR(1)
`(c, phi, sigma)` -- tipicamente o devolvido por `fit_ar1_fator(fator, valuation_idx)`,
ajustado só com dado até o corte. Cada chamada usa innovações aleatórias novas (via
`rng`), então cada réplica do Monte Carlo (`montecarlo_apc_reserve`) simula um caminho
futuro DIFERENTE -- é isso que captura a incerteza genuína sobre o valor futuro do
fator, desconhecido no momento do corte.

Os índices `<= valuation_idx` do vetor devolvido são cópias de `fator` (não usados na
projeção futura, mas preenchidos para nunca devolver `NaN` onde o valor é na verdade
conhecido). **Nunca lê `fator[t]` para `t > valuation_idx`** -- mesmo que o vetor
completo (com os valores realmente realizados depois do corte) seja passado por
conveniência pelo chamador, esta função ignora essas entradas por construção: é a
garantia de ausência de vazamento exigida no plano de reenquadramento (Fase 1b).
Índices maiores que `max_t` não são preenchidos (ficam `NaN`): o chamador só precisa
do caminho até a maior idade de calendário do leque de previsão daquele corte.
"""
function simulate_fator_future(fator::Vector{Float64}, valuation_idx::Int, max_t::Int,
                                c::Float64, phi::Float64, sigma::Float64, rng::AbstractRNG)
    path = fill(NaN, max_t)
    limite_hist = min(valuation_idx, length(fator), max_t)
    for t in 1:limite_hist
        path[t] = fator[t]
    end
    last_known = min(valuation_idx, length(fator))
    while last_known > 0 && isnan(fator[last_known])
        last_known -= 1
    end
    anterior = last_known > 0 ? fator[last_known] : 0.0
    for t in (valuation_idx + 1):max_t
        anterior = c + phi * anterior + sigma * randn(rng)
        path[t] = anterior
    end
    return path
end

"""
    montecarlo_apc_reserve(tri, calendar_index, fator, valuation_idx, k_vec, tcol;
                            n_boot=1000, seed=20261102)

Distribuição preditiva Monte Carlo da reserva do modelo (c) (idade+safra+β·fator),
corrigindo as duas simplificações da Fase 1 declaradas em `apc_fase1_resumo.md`,
seção 2.4 (cobertura medida de 0/9 cortes, 1,2% em 200 sementes). Cada réplica soma
TRÊS fontes de incerteza, nunca lendo o valor realizado do fator depois do corte:

1. **Incerteza de parâmetro**: os resíduos de Pearson do ajuste original
   (`fit_apc_calendar(tri, calendar_index, fator)`) são reamostrados para gerar um
   pseudo-triângulo (`pseudo_incr[i,j] = max(fitted[i,j] + e*_[i,j]·ajuste·sqrt(fitted[i,j]),
   1e-6)`, mesmo esquema de `ChainLadderCore.bootstrap_reserves`), e o modelo
   (intercepto, idade, safra, β) é REESTIMADO por IRLS a cada réplica, sobre esse
   pseudo-triângulo -- só com dado até `valuation_idx` (o pseudo-triângulo é
   construído célula a célula só onde `tri` tinha dado observado).
2. **Incerteza sobre o valor FUTURO do fator de calendário**: um AR(1) ingênuo
   (`fit_ar1_fator`) é ajustado UMA VEZ, só com `fator[t]` para `t <= valuation_idx`
   (nunca o valor realizado depois do corte). Cada réplica simula um caminho diferente
   (`simulate_fator_future`, innovações novas) a partir do último valor conhecido --
   paralelo ao que `bootstrap_reserves` faz ao fixar a extrapolação de cauda dos
   fatores de desenvolvimento UMA VEZ fora do laço (reajustar uma curva instável a
   cada réplica, com poucos pontos, é o que causava a explosão corrigida em
   `extrapolate_factor_tail`): aqui o que fica fixo são os PARÂMETROS do AR(1), não o
   caminho -- o caminho simulado varia por réplica, que é o ponto central desta
   correção.
3. **Incerteza de processo**: amostra `Gamma(mu[i,j]/phi_b, phi_b)` por célula futura
   (mesma família de antes), com `phi_b` o parâmetro de dispersão REESTIMADO na
   réplica (não o original).

Se o reajuste IRLS de uma réplica falhar (`fit_apc_calendar` lança erro -- ex.:
colinearidade degenerada num pseudo-triângulo raro), a réplica usa o ajuste original
(`fit0`) em vez de descartar a réplica, para não enviesar a cauda da distribuição
resultante por exclusão seletiva.

`valuation_idx` é o índice de calendário da diagonal de corte: necessário para saber
até onde `fator` é conhecido (nunca depois disso, nem na reestimação dos parâmetros do
modelo nem no ajuste do AR(1)) e a partir de onde a simulação do caminho futuro deve
começar. Internamente, `fator` é copiado para `fator_seguro` com as entradas em
`t > valuation_idx` substituídas por `NaN` antes de qualquer ajuste -- uma segunda
camada de proteção contra vazamento, redundante com a verificação de índice em
`fit_apc_calendar` (que só lê `fator` nas células observadas de `tri`, todas com
`calendar_index <= valuation_idx` por construção de `Backtest.truncate_at_valuation`),
mas explícita e auditável por si só.

Devolve `(reservas, pontual)`: `reservas` é o vetor de `n_boot` reservas simuladas --
este é o objeto a comparar contra Mack/bootstrap CL na cobertura, e NUNCA usa o fator
realizado em nenhuma réplica. `pontual` é a reserva projetada com os parâmetros
estimados na amostra original (sem reestimar) e o valor REALIZADO do fator no leque de
previsão (a mesma projeção pontual "quase-nowcast" já usada no pipeline da Fase 1,
porque o SCR.data é publicado perto de tempo real -- ver comentário em
`scripts/run_apc_pipeline.jl`): serve só para o erro percentual determinístico do
modelo (c) nas tabelas do pipeline, nunca para o intervalo.
"""
function montecarlo_apc_reserve(tri::AbstractMatrix{<:Real}, calendar_index::Matrix{Int},
                                 fator::Vector{Float64}, valuation_idx::Int,
                                 k_vec::Vector{Int}, tcol::Vector{Int};
                                 n_boot::Int=1000, seed::Int=20261102)
    n_cohorts, max_dev = size(tri)
    fator_seguro = [t <= valuation_idx ? fator[t] : NaN for t in eachindex(fator)]

    fit0 = fit_apc_calendar(tri, calendar_index, fator_seguro)
    proj0_oracle = project_apc_calendar(fit0, calendar_index, fator)
    pontual = 0.0
    for i in 1:n_cohorts
        k_vec[i] == 0 && continue
        for j in (k_vec[i] + 1):tcol[i]
            isnan(proj0_oracle[i, j]) && continue
            pontual += proj0_oracle[i, j]
        end
    end

    max_t_needed = valuation_idx
    for i in 1:n_cohorts
        k_vec[i] == 0 && continue
        for j in (k_vec[i] + 1):tcol[i]
            ci = calendar_index[i, j]
            ci > max_t_needed && (max_t_needed = ci)
        end
    end

    c_ar, phi_ar, sigma_ar = fit_ar1_fator(fator_seguro, valuation_idx)

    valid_idx = [(i, j) for i in 1:n_cohorts, j in 1:max_dev if !isnan(fit0.resid[i, j])]
    valid_resid = [fit0.resid[i, j] for (i, j) in valid_idx]
    adj = sqrt(fit0.n_obs / fit0.dof)

    rng = MersenneTwister(seed)
    reservas = fill(0.0, n_boot)
    for b in 1:n_boot
        pseudo_incr = fill(NaN, n_cohorts, max_dev)
        for (i, j) in valid_idx
            e_star = valid_resid[rand(rng, 1:length(valid_resid))]
            fm = max(fit0.fitted_incr[i, j], 1e-6)
            pseudo_incr[i, j] = max(fm + e_star * adj * sqrt(fm), 1e-6)
        end
        pseudo_cum = fill(NaN, n_cohorts, max_dev)
        for i in 1:n_cohorts
            k_vec[i] == 0 && continue
            acc = 0.0
            for j in 1:k_vec[i]
                acc += pseudo_incr[i, j]
                pseudo_cum[i, j] = acc
            end
        end

        fit_b = try
            fit_apc_calendar(pseudo_cum, calendar_index, fator_seguro)
        catch
            fit0
        end
        # fit_b.phi é usado como shape (mu/phi_b) E como scale do mesmo Gamma: tem que
        # ser o MESMO valor floorado nos dois lugares (ver chamada abaixo). Floorar só a
        # scale e deixar o shape usar o phi_b não-floorado (bug corrigido nesta fase)
        # desacopla média=shape*scale do mu pretendido por várias ordens de grandeza
        # quando phi_b é minúsculo (ex.: triângulo quase sem ruído de processo).
        phi_b = max(fit_b.phi > 0 ? fit_b.phi : fit0.phi, 1e-10)

        fator_sim = simulate_fator_future(fator_seguro, valuation_idx, max_t_needed,
                                           c_ar, phi_ar, sigma_ar, rng)
        proj_b = project_apc_calendar(fit_b, calendar_index, fator_sim)

        total = 0.0
        for i in 1:n_cohorts
            k_vec[i] == 0 && continue
            for j in (k_vec[i] + 1):tcol[i]
                mu = proj_b[i, j]
                (isnan(mu) || mu <= 0) && continue
                shp = max(mu / phi_b, 1e-8)
                total += rand(rng, Gamma(shp, phi_b))
            end
        end
        reservas[b] = total
    end
    return reservas, pontual
end

end # module
