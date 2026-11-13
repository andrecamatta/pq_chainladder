# pq_chainladder

Código de apoio ao artigo
[Atuária para quants, parte 3: o que seu chain ladder não vê quando as carteiras andam juntas](https://pilulasdequant.com.br),
da série [Pílulas de Quant](https://pilulasdequant.com.br).

O chain ladder é um método atuarial clássico para reservar sinistros ocorridos mas
ainda não avisados (incurred but not reported, IBNR). Aplicado a safras de
inadimplência de crédito, ele projeta o saldo inadimplente futuro a partir só da
idade da safra, sem um termo para o trimestre-calendário em que a célula é observada.
Este repositório mede o que essa omissão custa e se o comovimento entre carteiras
reais de crédito no Brasil ficou mais forte depois de 2020.

Há duas frentes, com papéis distintos:

1. **Triângulo simulado.** Nenhuma fonte pública brasileira disponibiliza um
   triângulo de safra genuíno (microdado de coorte de originação). O repositório
   simula uma carteira de safras trimestrais, calibrada em volume e em um índice
   de estresse macro pela série real de inadimplência agregada do Sistema de
   Informações de Créditos (SCR.data) do Banco Central do Brasil, para crédito
   pessoal sem consignação de pessoa física. A simulação é declarada como tal, e
   os parâmetros da curva de maturação são hipóteses assumidas, não valores
   calibrados em dado de safra real. Nesse triângulo entram o teste de diagonal,
   o modelo idade-período-safra (age-period-cohort) com termo de calendário
   β · fator, a decomposição do erro de reserva em parte idiossincrática (própria
   da safra) e parte compartilhada entre safras, e a cobertura dos intervalos de
   95%.
2. **Painel real de modalidades.** Com as sete séries agregadas do SCR.data
   (cartão de crédito, veículos, crédito pessoal sem consignação, crédito pessoal
   consignado, capital de giro, financiamento imobiliário e rural, de set/2012 a
   dez/2025), o repositório constrói um fator comum leave-one-out por análise de
   componentes principais (principal component analysis, PCA) e estima o
   coeficiente β e o R² de cada modalidade com o fator das outras seis.

## Resultados principais

No triângulo simulado, o teste de diagonal detecta o fator comum de calendário
com folga (estatística F = 12,90 antes do termo de calendário). Depois do ajuste
com β · fator, a estatística cai para 6,46 e a fração sistemática da variância
residual cai de 42,7% para 27,2%. No corte de dez/2021, o componente
idiossincrático se cancela na soma das safras e o componente compartilhado
responde por todo o erro agregado da reserva (o chain ladder clássico subestima
a reserva em 35,9%), embora explique só 19,6% da variância de cada célula. O
efeito de período estimado só do triângulo (sem a série externa) tem correlação
0,73 com a série do Banco Central usada como fator. Em 200 sementes, a cobertura
do intervalo de 95% passa de 34,9% no método de Mack para 38,6% com o termo de
calendário.

Nas sete modalidades reais, o β leave-one-out por PCA de cartão de crédito,
veículos e crédito pessoal sem consignação fica entre −0,01 e 0,16 até 2019
(respectivamente 0,16, 0,07 e −0,01) e sobe para o intervalo de 0,77 a 0,82
depois de 2020 (0,77, 0,77 e 0,82). O crédito consignado fica fora do movimento
conjunto nos dois períodos (0,16 antes de 2020 e −0,09 depois). Selic, prazo
médio de carteira, composição por tipo de instituição (fintechs e instituições
de pagamento) e endividamento das famílias foram testados como explicação da
virada; nenhuma dessas variáveis, sozinha, dá conta da mudança de regime.

Os números acima estão em `results/` (e, para o backtest clássico, em
`data/output/`), gerados pelos scripts abaixo.

## Estrutura

- `src/VintageSim.jl`: simula o triângulo de safras (originação trimestral
  contínua até o fim de 2024), modulado pela série macro real.
- `src/ChainLadderCore.jl`: fatores de desenvolvimento, extrapolação de cauda
  (fatores e σ² de Mack), projeção, erro padrão de Mack, bootstrap, teste de
  efeito de calendário e Bornhuetter-Ferguson (este último, material legado;
  ver abaixo).
- `src/ApcModel.jl`: modelo idade-safra-período com o termo de calendário
  ancorado numa covariável externa (`fit_apc_calendar`), projeção
  (`project_apc_calendar`), efeito de período livre estimado só do triângulo
  (`estimate_free_period_effects`), decomposição do desvio de um corte em
  componente compartilhado versus idiossincrático
  (`decompose_calendar_deviation`) e Monte Carlo da reserva
  (`montecarlo_apc_reserve`).
- `src/FatorComum.jl`: fator comum leave-one-out (PCA e média ponderada pela
  carteira), β e R² por modalidade, bootstrap em blocos e quebra por
  subperíodo.
- `src/ScrCalibration.jl`: carrega a série real de calibração
  (`data/scr_credito_pessoal_sem_consignacao.csv`).
- `src/ScrModalidades.jl`: predicação das sete modalidades e carga do painel
  já agregado (`data/scr_modalidades.csv`).
- `src/Backtest.jl`: backtest por truncamento de diagonal, material legado da
  comparação entre chain ladder e Bornhuetter-Ferguson, mantido porque o
  pipeline clássico e os testes ainda o utilizam.
- `scripts/run_apc_pipeline.jl`: teste de diagonal antes e depois do termo de
  calendário, recuperação do efeito de período livre, decomposição do desvio
  de dez/2021, cobertura do intervalo de 95% (9 cortes e 200 sementes) e
  recuperação de β por segmento; grava em `results/apc_*.csv` e
  `results/apc_resumo.json`.
- `scripts/export_residuos_diagonal.jl`: exporta o resíduo médio e o RMS por
  diagonal de calendário, antes e depois do termo β · fator
  (`results/apc_residuos_diagonal.csv`).
- `scripts/run_fator_comum_pipeline.jl`: β e R² por modalidade real, intervalos
  por bootstrap em blocos, estabilidade antes e depois de 2020 e correlação
  móvel entre pares; grava `results/fator_comum_*.csv`,
  `results/rolling_*.csv` e `results/robustez_*.csv`.
- `scripts/run_identificacao_macro.jl`: correlação do fator comum e das
  modalidades com Selic, inflação, desemprego, massa de renda, atividade
  (Índice de Atividade Econômica do Banco Central, IBC-Br) e a dummy do
  Auxílio Emergencial.
- `scripts/run_prazo_ladders.jl`: confronta o prazo médio de carteira (série
  do Sistema Gerenciador de Séries Temporais, SGS, e prazo reconstruído do
  SCR.data) com o β pós-2020, mais correlação defasada entre modalidades.
- `scripts/run_endividamento_familias.jl`: mesma lógica de identificação, com
  endividamento e comprometimento de renda das famílias (SGS).
- `scripts/run_composicao_segmento.jl`: composição da carteira por tipo de
  instituição (fintech, instituição de pagamento, banco). Exige os microdados
  brutos do SCR.data, que não estão neste repositório.
- `scripts/build_scr_modalidades.jl`: agrega os microdados brutos do SCR.data
  em `data/scr_modalidades.csv`. Roda uma única vez, com a variável de
  ambiente `SCR_RAW_DIR` apontando para os CSVs trimestrais já baixados. O
  painel agregado já está versionado, então este script não é necessário para
  reproduzir os demais resultados.
- `scripts/run_pipeline.jl`: pipeline legado do backtest clássico (chain
  ladder, Mack, bootstrap e Bornhuetter-Ferguson, com prior bem especificado e
  variantes ±25%); grava em `data/output/`.
- `test/runtests.jl`: reprodução do exemplo de Taylor e Ashe usado por Mack
  (1993), regra de Mack para a última coluna de σ², recuperação da cauda
  log-linear, recuperação exata de idade, safra e β do modelo com termo de
  calendário, recuperação de β e R² (e ausência de vazamento do fator
  leave-one-out) em séries sintéticas, entre outros.

## Dados

Nenhum microdado bruto entra no repositório. O que está versionado são séries
trimestrais já agregadas e os CSVs e JSON de resultado:

- `data/scr_credito_pessoal_sem_consignacao.csv`: série real (set/2012 a
  dez/2025) de crédito pessoal sem consignação, pessoa física, usada para
  calibrar a simulação e, no pipeline legado, o prior do Bornhuetter-Ferguson.
- `data/scr_modalidades.csv`: painel trimestre × modalidade (378 linhas, sete
  modalidades, set/2012 a dez/2025), com carteira ativa e carteira em
  inadimplência. Reproduz a série de crédito pessoal sem consignação no
  recorte em comum.
- `data/scr_modalidades_prazo.csv`: prazo médio reconstruído do SCR.data, com
  a mesma predicação das sete modalidades.
- `data/prazo_modalidades.csv`: prazo médio oficial do SGS, classificação de
  modalidades do Banco Central (não idêntica à predicação do SCR.data).
- `data/macro_trimestral.csv`: Selic, inflação, desemprego, massa de renda,
  IBC-Br e dummy do Auxílio Emergencial, agregados por trimestre.
- `data/macro_endividamento_trimestral.csv`: endividamento e comprometimento
  de renda das famílias (SGS), agregados por trimestre.
- `data/output/`: resultados do backtest clássico, gerados por
  `scripts/run_pipeline.jl`.
- `results/`: resultados do modelo com termo de calendário, do fator comum
  por modalidade e dos exercícios de identificação e robustez.

Os CSVs brutos do SCR.data (cerca de 65 a 100 MB por trimestre, em
`https://www.bcb.gov.br/pda/desig/scrdata_<ano>.zip`, Portal de Dados Abertos
do Banco Central) ficam de fora de propósito: são grandes e reproduzíveis a
partir da fonte pública. O mesmo vale para os JSON brutos do SGS.

## Como reproduzir

```
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
julia --project=. scripts/run_apc_pipeline.jl
julia --project=. scripts/export_residuos_diagonal.jl
julia --project=. scripts/run_fator_comum_pipeline.jl
julia --project=. scripts/run_identificacao_macro.jl
julia --project=. scripts/run_prazo_ladders.jl
julia --project=. scripts/run_endividamento_familias.jl
```

Os painéis em `data/*.csv` já estão versionados; os scripts acima rodam a
partir deles. O pipeline legado do backtest, se for de interesse, é:

```
julia --project=. scripts/run_pipeline.jl
```

Para reconstruir `data/scr_modalidades.csv` a partir dos CSVs brutos do
SCR.data (não incluídos neste repositório):

```
SCR_RAW_DIR=/caminho/para/csvs/brutos/do/scrdata julia --project=. scripts/build_scr_modalidades.jl
```

`scripts/run_composicao_segmento.jl` também depende desses microdados brutos.
