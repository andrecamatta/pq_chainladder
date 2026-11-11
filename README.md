# pq_chainladder

Código de apoio ao artigo "Atuária para quants, parte 3" (série [Pílulas de
Quant](https://pilulasdequant.com.br)), em reenquadramento (branch
`reenquadramento/apc`): de "qual método de reserva prevê melhor" para "quanto do erro
de uma projeção de chain ladder vem de um choque comum de calendário (sistemático,
correlacionado entre safras) e quanto vem da própria safra (idiossincrático)".

O artigo aplica o chain ladder atuarial (reserva de sinistros ainda não avisados,
IBNR) a safras de inadimplência de crédito de varejo no Brasil, compara chain ladder
puro (com cauda de fatores e de σ² extrapolada), o erro padrão de Mack, o bootstrap de
reamostragem de resíduos e o Bornhuetter-Ferguson, e diagnostica -- não promete
corrigir -- o efeito de um fator de calendário correlacionado entre safras (um ciclo
de inadimplência agregada) ausente das três premissas clássicas, via um modelo
idade-safra-período (*age-period-cohort*, APC) com o termo de período ancorado a uma
variável observável do Banco Central do Brasil (BCB), e uma extensão multi-segmento
com carga `beta_s` por segmento sobre esse fator comum, paralela à correlação de
ativos do modelo de Vasicek/fórmula *Internal Ratings-Based* (IRB) de Basileia.

Nenhuma fonte pública brasileira disponibiliza um triângulo de safra genuíno
(microdado de coorte de originação de crédito). Este repositório simula uma
carteira de safras trimestrais calibrada em parâmetros reais, usando como índice
de estresse macro a série trimestral real de inadimplência agregada do SCR.data
(Banco Central do Brasil) para crédito pessoal sem consignação de pessoa física.
A simulação é declarada como tal no artigo, não como dado de safra real, e os
parâmetros da curva de maturação são hipóteses assumidas, não valores calibrados
em dado de safra real.

A Fase 1c do reenquadramento acrescenta uma pergunta empírica separada, com dado real
(não simulado): das modalidades de crédito do SCR.data, quais têm sua inadimplência
mais explicada por um fator comum entre modalidades (sistemático) e quais são mais
idiossincráticas -- o mesmo paralelo com a correlação de ativos de Vasicek/IRB, mas
medido, não simulado com `beta_s` fixado. Ver `apc_fase1c_resumo.md`.

## Estrutura

- `src/ScrCalibration.jl`: carrega a série real de calibração (`data/scr_credito_pessoal_sem_consignacao.csv`).
- `src/ScrModalidades.jl`: agrega os microdados brutos do SCR.data em sete modalidades
  reais (cartão de crédito, crédito pessoal consignado, crédito pessoal sem
  consignação, veículos, financiamento imobiliário, capital de giro, rural) e carrega
  a série pequena já agregada (`data/scr_modalidades.csv`).
- `src/FatorComum.jl`: fator comum leave-one-out (PCA e média ponderada pela
  carteira), β e R² por modalidade (paralelo direto do `beta_s`/`beta_s²` de
  `VintageSim`), bootstrap em blocos para IC e quebra por subperíodo.
- `src/VintageSim.jl`: simula o triângulo de safras (originação trimestral contínua
  até o fim de 2024), modulado pela série macro real, com cada safra limitada ao
  horizonte de calendário realmente coberto pela série.
- `src/ChainLadderCore.jl`: fatores de desenvolvimento, extrapolação de cauda
  (fatores e σ² de Mack), projeção, erro padrão de Mack, bootstrap (chain ladder e
  Bornhuetter-Ferguson), Bornhuetter-Ferguson e teste de efeito de calendário.
- `src/Backtest.jl`: backtest por truncamento de diagonal, comparando contra a
  idade realmente observável dentro do horizonte real (não sempre a maturidade
  completa), com prior do Bornhuetter-Ferguson calibrado só na informação
  disponível até a originação de cada safra.
- `src/ApcModel.jl`: modelo idade-safra-período com o termo de calendário ancorado
  numa covariável externa (`fit_apc_calendar`, quase-Poisson log-linear via IRLS,
  identificação discutida na docstring), projeção para frente (`project_apc_calendar`),
  efeito de período LIVRE estimado só do triângulo para comparação declarada contra a
  série do BCB (`estimate_free_period_effects`), decomposição contábil do desvio de
  um corte do backtest em componente compartilhado entre safras vs. idiossincrático
  (`decompose_calendar_deviation`) e Monte Carlo de incerteza de processo do modelo
  (c) (`montecarlo_apc_reserve`, com a limitação de incerteza de parâmetro ausente
  declarada na própria docstring).
- `scripts/run_pipeline.jl`: roda a simulação, o backtest completo (prior bem
  especificado e as duas variantes mal especificadas, ±25%) e as checagens de
  robustez (mediana, exclusão do corte de dez/2021, cortes maduros vs.
  truncados), grava os resultados em `data/output/`.
- `scripts/run_apc_pipeline.jl`: mede o diagnóstico de diagonal antes/depois do
  termo de calendário, a recuperação do efeito de período livre vs. a série do BCB,
  a decomposição do desvio de dez/2021, a cobertura do IC95% com/sem termo de
  calendário (9 cortes e 200 sementes) e a recuperação de `beta_s` na extensão
  multi-segmento (50 sementes); grava os resultados em `results/`.
- `scripts/build_scr_modalidades.jl`: agrega os microdados BRUTOS do SCR.data
  (não incluídos neste repositório -- ver seção de dados abaixo) em
  `data/scr_modalidades.csv`. Roda uma única vez; precisa de `SCR_RAW_DIR` apontando
  para os CSVs trimestrais brutos já baixados.
- `scripts/run_fator_comum_pipeline.jl`: mede β/R² por modalidade (PCA leave-one-out
  e a alternativa de robustez por média ponderada pela carteira), IC95% via bootstrap
  em blocos, estabilidade por subperíodo (antes/depois de 2020) e sensibilidade do IC
  ao comprimento do bloco; grava os resultados em `results/fator_comum_*.csv`.
- `test/runtests.jl`: testes automatizados, incluindo a reprodução do exemplo
  de Taylor/Ashe usado por Mack (1993), a regra de Mack para a última coluna de
  σ², a recuperação exata da extrapolação de cauda num triângulo sintético sem
  ruído, a recuperação exata de idade+safra+beta do modelo (c) num triângulo
  determinístico sem ruído, a recuperação de `beta_s` nos extremos (0 e 1) da
  extensão multi-segmento e a recuperação de β/R² (e a ausência de vazamento do
  fator leave-one-out) em séries sintéticas com fator comum conhecido. Roda com
  `julia --project=. test/runtests.jl` ou com `julia --project=. -e 'import Pkg; Pkg.test()'`.
- `data/scr_credito_pessoal_sem_consignacao.csv`: série real já agregada (set/2012 a
  dez/2025), extraída do SCR.data do Banco Central, para o segmento de crédito
  pessoal sem consignação (PF). Usada tanto para calibrar a simulação quanto para o
  prior do Bornhuetter-Ferguson.
- `data/scr_modalidades.csv`: painel pequeno (trimestre × modalidade, 378 linhas)
  agregado a partir dos microdados BRUTOS do SCR.data (sete modalidades, set/2012 a
  dez/2025, mesmas colunas `carteira_ativa`/`carteira_inadimplencia`). Reproduz
  bit-a-bit a série de `scr_credito_pessoal_sem_consignacao.csv` para a modalidade em
  comum (checado). Os CSVs brutos (~65-100MB por trimestre, baixados de
  `https://www.bcb.gov.br/pda/desig/scrdata_<ano>.zip`, Portal de Dados Abertos do
  Banco Central) não entram no repositório -- grandes demais e reproduzíveis a
  qualquer momento a partir da fonte pública.
- `data/output/`: resultados do backtest clássico (CSV e JSON), gerados por
  `scripts/run_pipeline.jl` e versionados para referência, não dado bruto.
- `results/`: resultados do reenquadramento idade-safra-período (CSV e JSON),
  gerados por `scripts/run_apc_pipeline.jl`, e do fator comum por modalidade real
  (CSV), gerados por `scripts/run_fator_comum_pipeline.jl`.

## Como reproduzir

```
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. scripts/run_pipeline.jl
julia --project=. scripts/run_apc_pipeline.jl
julia --project=. test/runtests.jl
```

`data/scr_modalidades.csv` já está versionado -- `scripts/run_fator_comum_pipeline.jl`
roda direto a partir dele. Para reconstruí-lo do zero (precisa dos CSVs brutos do
SCR.data, não incluídos neste repositório):

```
SCR_RAW_DIR=/caminho/para/csvs/brutos/do/scrdata julia --project=. scripts/build_scr_modalidades.jl
julia --project=. scripts/run_fator_comum_pipeline.jl
```
