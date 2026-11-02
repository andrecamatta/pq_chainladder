# pq_chainladder

Código de apoio ao artigo "Atuária para quants, parte 3: chain ladder para safras de
inadimplência de crédito", publicado em [Pílulas de Quant](https://pilulasdequant.com.br).

O artigo aplica o chain ladder atuarial (reserva de sinistros ainda não avisados,
IBNR) a safras de inadimplência de crédito de varejo no Brasil, e testa a projeção
contra o que de fato apareceu depois, comparando chain ladder puro, o erro padrão
de Mack, o bootstrap ODP de reamostragem de resíduos e o Bornhuetter-Ferguson.

Nenhuma fonte pública brasileira disponibiliza um triângulo de safra genuíno
(microdado de coorte de originação de crédito). Este repositório simula uma
carteira de safras trimestrais calibrada em parâmetros reais, usando como índice
de estresse macro a série trimestral real de inadimplência agregada do SCR.data
(Banco Central do Brasil) para crédito pessoal sem consignação de pessoa física.
A simulação é declarada como tal no artigo, não como dado de safra real.

## Estrutura

- `src/ScrCalibration.jl`: carrega a série real de calibração (`data/scr_credito_pessoal_sem_consignacao.csv`).
- `src/VintageSim.jl`: simula o triângulo de safras, modulado pela série macro real.
- `src/ChainLadderCore.jl`: fatores de desenvolvimento, projeção, erro padrão de Mack,
  bootstrap ODP, Bornhuetter-Ferguson e teste de efeito de calendário.
- `src/Backtest.jl`: backtest por truncamento de diagonal, com prior do Bornhuetter-Ferguson
  calibrado só na informação disponível até cada data de avaliação.
- `scripts/run_pipeline.jl`: roda a simulação e o backtest completo, grava os resultados
  em `data/output/`.
- `data/scr_credito_pessoal_sem_consignacao.csv`: série trimestral real (set/2012 a
  dez/2025) extraída do SCR.data do Banco Central, agregada para o segmento de
  crédito pessoal sem consignação (PF), usada para calibrar a simulação.

## Como reproduzir

```
julia --project=. -e 'import Pkg; Pkg.instantiate()'
julia --project=. scripts/run_pipeline.jl
```

Os resultados (CSV e JSON) são gravados em `data/output/`.
