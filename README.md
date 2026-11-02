# pq_chainladder

Código de apoio ao artigo "Atuária para quants, parte 3: chain ladder para safras de
inadimplência de crédito", publicado em [Pílulas de Quant](https://pilulasdequant.com.br).

O artigo aplica o chain ladder atuarial (reserva de sinistros ainda não avisados,
IBNR) a safras de inadimplência de crédito de varejo no Brasil, e testa a projeção
contra o que de fato apareceu depois, comparando chain ladder puro (com cauda de
fatores e de σ² extrapolada), o erro padrão de Mack, o bootstrap de reamostragem de
resíduos e o Bornhuetter-Ferguson, incluindo uma variante do prior deliberadamente
mal especificada para medir sensibilidade.

Nenhuma fonte pública brasileira disponibiliza um triângulo de safra genuíno
(microdado de coorte de originação de crédito). Este repositório simula uma
carteira de safras trimestrais calibrada em parâmetros reais, usando como índice
de estresse macro a série trimestral real de inadimplência agregada do SCR.data
(Banco Central do Brasil) para crédito pessoal sem consignação de pessoa física.
A simulação é declarada como tal no artigo, não como dado de safra real, e os
parâmetros da curva de maturação são hipóteses assumidas, não valores calibrados
em dado de safra real.

## Estrutura

- `src/ScrCalibration.jl`: carrega a série real de calibração (`data/scr_credito_pessoal_sem_consignacao.csv`).
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
- `scripts/run_pipeline.jl`: roda a simulação, o backtest completo (prior bem
  especificado e as duas variantes mal especificadas, ±25%) e as checagens de
  robustez (mediana, exclusão do corte de dez/2021, cortes maduros vs.
  truncados), grava os resultados em `data/output/`.
- `test/runtests.jl`: testes automatizados, incluindo a reprodução do exemplo
  de Taylor/Ashe usado por Mack (1993), a regra de Mack para a última coluna de
  σ² e a recuperação exata da extrapolação de cauda num triângulo sintético sem
  ruído. Roda com `julia --project=. test/runtests.jl` ou com
  `julia --project=. -e 'import Pkg; Pkg.test()'`.
- `data/scr_credito_pessoal_sem_consignacao.csv`: **única** série real usada no
  repositório (set/2012 a dez/2025), extraída do SCR.data do Banco Central,
  agregada para o segmento de crédito pessoal sem consignação (PF). Usada tanto
  para calibrar a simulação quanto para o prior do Bornhuetter-Ferguson; não há
  uma segunda cópia agregada em `data/output/`.
- `data/output/`: resultados do backtest (CSV e JSON), gerados por
  `scripts/run_pipeline.jl` e versionados para referência, não dado bruto.

## Como reproduzir

```
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. scripts/run_pipeline.jl
julia --project=. test/runtests.jl
```
