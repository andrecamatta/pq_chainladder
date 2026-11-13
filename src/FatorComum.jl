module FatorComum

using Statistics
using LinearAlgebra
using Random
using Dates

export variacao_log, padronizar_serie, fator_pca_loo, fator_media_simples_loo,
       fator_media_ponderada_loo, estimar_beta_r2, bootstrap_beta_r2, intervalo_percentil,
       indices_subperiodo, fator_pca, fator_media_ponderada, correlacao_defasada,
       beta_r2_janela_movel

"""
    variacao_log(taxa)

Variação trimestre-sobre-trimestre do log da taxa: `diff(log.(taxa))`, comprimento
`length(taxa) - 1`. Transformação estacionária usada neste módulo para o fator comum e
para β/R² por modalidade -- diferente da variação HOMÓLOGA (4 trimestres,
`VintageSim.fator_calendario_bcb`) usada no modelo idade-safra-período: com só 54
trimestres na amostra (set/2012 a dez/2025), a variação de 1 trimestre preserva 53
pontos em vez de 50 e não exige escolher uma defasagem adicional. É uma escolha
declarada, não a única defensável -- ver `apc_fase1c_resumo.md`, seção 2, para a
discussão de robustez.
"""
function variacao_log(taxa::Vector{Float64})
    lt = log.(taxa)
    return lt[2:end] .- lt[1:(end - 1)]
end

"""
    padronizar_serie(x)

Padroniza para média 0 e desvio-padrão 1 (população amostral, `Statistics.std`).
"""
function padronizar_serie(x::Vector{Float64})
    return (x .- mean(x)) ./ std(x)
end

"""
    fator_pca_loo(Z, i)

Primeiro componente principal (PC1) das colunas de `Z` (T×N; cada coluna já
padronizada para média 0/desvio 1) EXCLUINDO a coluna `i` -- fator "leave-one-out", que
nunca usa a série da própria modalidade `i` para construir o fator contra o qual `i` é
depois testada. Evita a circularidade óbvia de testar se a modalidade `i` "carrega" num
fator que foi parcialmente construído a partir dela mesma.

Calculado via autovetores da matriz de correlação das N-1 colunas restantes
(equivalente a SVD de `Z[:, outras]`, já que as colunas têm variância 1). O sinal do
autovetor é escolhido para que o fator correlacione positivamente com a média simples
das mesmas colunas -- convenção de sinal (β pode trocar de sinal com a convenção
contrária), sem efeito sobre `β²`/R². O fator devolvido tem média 0 e desvio 1.

Exige `size(Z, 2) >= 3` (pelo menos duas colunas restantes depois de excluir `i`, senão
"primeiro componente principal" de uma única série é a própria série, sem conteúdo de
redução de dimensão).
"""
function fator_pca_loo(Z::AbstractMatrix{Float64}, i::Int)
    n_cols = size(Z, 2)
    n_cols >= 3 || error("fator_pca_loo exige pelo menos 3 colunas (2 restantes após excluir i)")
    outras = setdiff(1:n_cols, i)
    Zo = Z[:, outras]
    C = cor(Zo)
    autovals, autovecs = eigen(Symmetric(C))
    k = argmax(autovals)
    v = autovecs[:, k]
    f = Zo * v
    f = f .- mean(f)
    f = f ./ std(f)
    media_simples = vec(mean(Zo, dims=2))
    cor(f, media_simples) < 0 && (f = -f)
    return f
end

"""
    fator_media_simples_loo(Z, i)

Média simples (não ponderada) das colunas de `Z` exceto `i`, padronizada. Mesma lógica
leave-one-out de `fator_pca_loo`, mais simples (sem decomposição espectral) -- usada
como alternativa de robustez à PCA.
"""
function fator_media_simples_loo(Z::AbstractMatrix{Float64}, i::Int)
    outras = setdiff(1:size(Z, 2), i)
    f = vec(mean(Z[:, outras], dims=2))
    return padronizar_serie(f)
end

"""
    fator_media_ponderada_loo(Z, pesos, i)

Média das colunas de `Z` exceto `i`, ponderada por `pesos` (vetor de comprimento
`size(Z, 2)`, tipicamente a carteira ativa média de cada modalidade na amostra --
pesos FIXOS no tempo, não uma ponderação período a período) e renormalizada para somar
1 só sobre as colunas restantes. Alternativa de robustez pedida no plano ("média
ponderada pela carteira"): modalidades maiores pesam mais na construção do fator
comum, o que `fator_pca_loo`/`fator_media_simples_loo` não fazem (dão peso igual, ou
peso ditado só pela covariância, a cada modalidade).
"""
function fator_media_ponderada_loo(Z::AbstractMatrix{Float64}, pesos::Vector{Float64}, i::Int)
    length(pesos) == size(Z, 2) || error("pesos precisa ter um valor por coluna de Z")
    outras = setdiff(1:size(Z, 2), i)
    w = pesos[outras] ./ sum(pesos[outras])
    f = Z[:, outras] * w
    return padronizar_serie(f)
end

"""
    estimar_beta_r2(y, f)

β (carga no fator comum) e R² (fração de variância explicada) de uma série `y` sobre o
fator `f`, pela mesma convenção usada em
`VintageSim.simulate_multi_segment_triangles`/`ApcModel` para o paralelo com a
correlação de ativos de Vasicek/IRB: padroniza `y` e `f` (média 0, desvio 1) e usa a
inclinação da regressão OLS de `y` padronizado sobre `f` padronizado como β. Com as
duas séries padronizadas, essa inclinação é algebricamente igual à correlação de
Pearson entre elas, e `R² = β²` exatamente (não uma aproximação) -- o análogo direto de
`beta_s²` = fração de variância do log-multiplicador no modelo sintético multi-segmento
da Fase 1.
"""
function estimar_beta_r2(y::Vector{Float64}, f::Vector{Float64})
    length(y) == length(f) || error("y e f precisam ter o mesmo comprimento")
    beta = cor(padronizar_serie(y), padronizar_serie(f))
    return beta, beta^2
end

"""
    bootstrap_blocos(n, bloco, rng)

Índices de um bootstrap em blocos móveis (*moving block bootstrap*, Künsch, 1989):
sorteia blocos contíguos de comprimento `bloco` (início uniforme em `1:(n-bloco+1)`,
com reposição) até acumular pelo menos `n` índices, e corta para exatamente `n`.
Preserva a autocorrelação de curto prazo dentro de cada bloco -- diferente de um
bootstrap i.i.d. (que embaralharia os pontos e destruiria qualquer dependência
temporal), apropriado porque `y` e `f` são séries temporais, não amostras
independentes.
"""
function bootstrap_blocos(n::Int, bloco::Int, rng::AbstractRNG)
    bloco = clamp(bloco, 1, n)
    idx = Int[]
    while length(idx) < n
        inicio = rand(rng, 1:(n - bloco + 1))
        append!(idx, inicio:(inicio + bloco - 1))
    end
    return idx[1:n]
end

"""
    bootstrap_beta_r2(y, f; n_boot=2000, bloco=8, seed=20261111)

Distribuição bootstrap de β e R² (`estimar_beta_r2`) via bootstrap em blocos móveis
sobre os PARES `(y_t, f_t)` -- cada réplica reamostra os mesmos índices de tempo para
`y` e `f` juntos, preservando a relação contemporânea entre as duas séries (o que
importa para β/R²) enquanto reamostra a dependência temporal de cada uma. `bloco=8`
(2 anos de trimestres) é uma escolha declarada, não calibrada nos dados -- ver
`apc_fase1c_resumo.md`, seção 3, para sensibilidade a `bloco=4` e `bloco=12`.

Devolve `(betas, r2s)`, vetores de comprimento `n_boot`.
"""
function bootstrap_beta_r2(y::Vector{Float64}, f::Vector{Float64};
                            n_boot::Int=2000, bloco::Int=8, seed::Int=20261111)
    n = length(y)
    rng = MersenneTwister(seed)
    betas = fill(0.0, n_boot)
    r2s = fill(0.0, n_boot)
    for b in 1:n_boot
        idx = bootstrap_blocos(n, bloco, rng)
        betas[b], r2s[b] = estimar_beta_r2(y[idx], f[idx])
    end
    return betas, r2s
end

"""
    intervalo_percentil(dist; alpha=0.05)

Intervalo de confiança por percentil (não-paramétrico) de uma distribuição bootstrap:
`(quantil(alpha/2), quantil(1 - alpha/2))`.
"""
function intervalo_percentil(dist::Vector{Float64}; alpha::Float64=0.05)
    s = sort(dist)
    n = length(s)
    lo_k = clamp(round(Int, (alpha / 2) * n), 1, n)
    hi_k = clamp(round(Int, (1 - alpha / 2) * n), 1, n)
    return s[lo_k], s[hi_k]
end

"""
    indices_subperiodo(datas, corte)

Índices de `datas` estritamente antes de `corte` e a partir de `corte` (inclusive) --
usado para checar estabilidade de β/R² antes/depois de um corte (ex.: 2020-01-01, para
isolar o choque da pandemia). `datas` é o vetor de datas DA SÉRIE DE VARIAÇÃO (já
após `variacao_log`, um elemento mais curto que a série de nível original).
"""
function indices_subperiodo(datas::Vector{Date}, corte::Date)
    return findall(d -> d < corte, datas), findall(d -> d >= corte, datas)
end

"""
    fator_pca(Z)

Primeiro componente principal (PC1) de TODAS as colunas de `Z` (T×N, cada coluna já
padronizada) -- diferente de `fator_pca_loo`, que exclui uma coluna por vez. Usado
quando o fator comum em si (não o β de uma modalidade específica contra ele) é o
objeto de interesse -- por exemplo, para correlacionar a SÉRIE do fator com variáveis
macroeconômicas externas, onde não há nenhuma circularidade a evitar (nenhuma das
colunas de `Z` é a série macro). Mesma convenção de sinal de `fator_pca_loo` (positivo
com a média simples das colunas).
"""
function fator_pca(Z::AbstractMatrix{Float64})
    size(Z, 2) >= 2 || error("fator_pca exige pelo menos 2 colunas")
    C = cor(Z)
    autovals, autovecs = eigen(Symmetric(C))
    k = argmax(autovals)
    v = autovecs[:, k]
    f = Z * v
    f = f .- mean(f)
    f = f ./ std(f)
    media_simples = vec(mean(Z, dims=2))
    cor(f, media_simples) < 0 && (f = -f)
    return f
end

"""
    fator_media_ponderada(Z, pesos)

Média ponderada de TODAS as colunas de `Z`, análogo de `fator_media_ponderada_loo`
sem exclusão -- mesma lógica de `fator_pca` acima (fator comum como objeto de
interesse, não β leave-one-out de uma modalidade específica).
"""
function fator_media_ponderada(Z::AbstractMatrix{Float64}, pesos::Vector{Float64})
    length(pesos) == size(Z, 2) || error("pesos precisa ter um valor por coluna de Z")
    w = pesos ./ sum(pesos)
    return padronizar_serie(vec(Z * w))
end

"""
    correlacao_defasada(x, y, lag)

Correlação de Pearson entre `x[t]` e `y[t-lag]` (lag positivo: `y` DEFASADA, ou seja,
`y` no passado em relação a `x` -- testa se `x` segue `y` com atraso de `lag`
períodos). Lag negativo testa o sentido contrário (`x` atrasada em relação a `y`).
Usa a interseção de índices válidos; `length(x) == length(y)` é exigido (mesma
contagem de períodos nas duas séries, alinhadas tempo a tempo antes de chamar).
"""
function correlacao_defasada(x::Vector{Float64}, y::Vector{Float64}, lag::Int)
    length(x) == length(y) || error("x e y precisam ter o mesmo comprimento")
    n = length(x)
    if lag >= 0
        xi = x[(1 + lag):n]
        yi = y[1:(n - lag)]
    else
        xi = x[1:(n + lag)]
        yi = y[(1 - lag):n]
    end
    length(xi) >= 3 || error("poucos pontos restantes para este lag")
    return cor(xi, yi)
end

"""
    beta_r2_janela_movel(y, f, janela)

β e R² (`estimar_beta_r2`) recalculados numa janela móvel de `janela` observações
sobre `y` e `f` (mesmo comprimento) -- o fator `f` é passado FIXO (calculado uma vez
na amostra completa, não reestimado por janela), então isto mede se a RELAÇÃO entre
`y` e um fator comum já conhecido varia no tempo, não se o próprio fator mudaria se
reestimado numa janela curta (seria ruidoso demais com janelas de 12-16 trimestres e
~6 colunas). Devolve `(datas_fim_idx, betas, r2s)`: `datas_fim_idx[k]` é o índice,
dentro de `y`/`f`, do ÚLTIMO ponto da k-ésima janela (use para indexar o vetor de
datas correspondente).
"""
function beta_r2_janela_movel(y::Vector{Float64}, f::Vector{Float64}, janela::Int)
    length(y) == length(f) || error("y e f precisam ter o mesmo comprimento")
    n = length(y)
    n >= janela || error("série mais curta que a janela")
    fim_idx = Int[]
    betas = Float64[]
    r2s = Float64[]
    for fim in janela:n
        inicio = fim - janela + 1
        b, r2 = estimar_beta_r2(y[inicio:fim], f[inicio:fim])
        push!(fim_idx, fim)
        push!(betas, b)
        push!(r2s, r2)
    end
    return fim_idx, betas, r2s
end

end # module
