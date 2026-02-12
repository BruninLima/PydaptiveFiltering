# Benchmark Plan — PydaptiveFiltering

Este documento descreve **como os benchmarks do PydaptiveFiltering são estruturados**, o que já foi feito, o que falta, e quais critérios usamos para comparar algoritmos de forma justa e reprodutível.

---

## 1) Objetivo

Criar um pipeline de benchmark que seja:

- **Reprodutível** (seeds controladas, scripts versionados, outputs salvos)
- **Comparável** (mesmo cenário por família de algoritmos)
- **Escalável** (modo rápido para desenvolvimento e modo robusto para release)
- **Útil para engenharia** (decisão por acurácia × custo computacional)

---

## 2) Estado atual (já implementado)

Atualmente já temos benchmarking funcional para parte da família linear (ex.: LMS, NLMS, AffineProjection, RLS, FastRLS, QRRLS etc.), com:

1. **Varredura de hiperparâmetros** via grid
2. Execução em múltiplas **seeds**
3. Agregação por `(algoritmo, params)` com estatísticas:
   - `mse_mean`
   - `mse_std`
   - `emse_mean`
   - `misadj_mean`
   - `us_per_sample_mean`
4. Ranking dos melhores resultados e export para:
   - `bench_grid_*.json` (detalhado)
   - `bench_grid_*.csv` (agregado)

Também foi incorporado o uso do catálogo:
- `pydaptivefiltering/_utils/algo_param_catalog.json`
para filtrar parâmetros inválidos por algoritmo, evitando quebra por incompatibilidade de assinatura.

---

## 3) Princípios de comparação (fair benchmarking)

Para manter justiça na comparação:

- Comparar algoritmos **da mesma família/suite** no **mesmo cenário**
- Evitar misturar métricas incompatíveis (ex.: sistema identificado vs equalização cega)
- Sempre reportar **acurácia e custo**:
  - qualidade (`mse`, `nmse`, `rmse`, etc.)
  - eficiência (`runtime`, `us/sample`)
- Usar várias seeds para reduzir conclusões baseadas em sorte

---

## 4) Estrutura geral dos testes

Os testes são organizados por **suites**, porque nem todos os algoritmos resolvem o mesmo tipo de problema.

## 4.1 Suites

### A) `linear_system_id`
Ex.: LMS, NLMS, AffineProjection, RLS, RLSAlt, FastRLS, StabFastRLS, QRRLS, LMSNewton, TDomain*, LRLS*, SM*

**Cenário**  
Identificação de sistema FIR (real ou complexo), com ruído aditivo.

**Métricas principais**
- `mse_final`
- `emse_final`
- `misadjustment`
- `us_per_sample`

---

### B) `sign_variants`
Ex.: SignData, SignError, DualSign

**Cenário**  
Mesmo de system ID, incluindo versão com ruído impulsivo.

**Métricas**
- MSE/NMSE
- robustez a outliers
- custo por amostra

---

### C) `blind_equalization`
Ex.: CMA, Godard, Sato, AffineProjectionCM

**Cenário**  
Equalização sem referência de treino explícita.

**Métricas**
- ISI residual
- EVM (quando aplicável)
- BER após decisão (quando aplicável)
- custo por amostra

> Observação: não usar MSE tradicional direto contra `d` de forma ingênua.

---

### D) `iir_adaptive`
Ex.: ErrorEquation, GaussNewton, GaussNewtonGradient, RLSIIR, SteiglitzMcBride

**Cenário**  
Sistema alvo IIR conhecido.

**Métricas**
- taxa de convergência estável
- MSE nos runs estáveis
- custo por amostra

---

### E) `nonlinear`
Ex.: BilinearRLS, RBF, ComplexRBF, MultilayerPerceptron, VolterraLMS, VolterraRLS

**Cenário**  
Sistema não-linear sintético conhecido.

**Métricas**
- NMSE
- erro final (janela tail)
- custo por amostra

---

### F) `subband`
Ex.: OLSBLMS, DLCLLMS, CFDLMS

**Cenário**  
Entrada colorida / cenário em que subband faz sentido.

**Métricas**
- velocidade de convergência
- erro final
- custo por amostra

---

### G) `kalman_tracking`
Ex.: Kalman

**Cenário**  
Tracking CV (posição/velocidade) com manobras e ruído de medição.

**Métricas**
- RMSE de posição
- RMSE de velocidade
- estatística da inovação
- custo por amostra

---

## 5) Modos de execução (budget)

Para equilibrar tempo de execução:

## 5.1 `smoke` (CI / sanity check)
- seeds: `1`
- ensemble: `5–10`
- K: `400–800`
- objetivo: detectar quebra rápida

## 5.2 `dev` (desenvolvimento)
- seeds: `3`
- ensemble: `20–40`
- K: `1200–3000`
- objetivo: tuning e comparação rápida

## 5.3 `release` (resultado publicável)
- seeds: `5–10`
- ensemble: `80–200`
- K: `3000–8000`
- objetivo: números robustos para docs/paper

---

## 6) Reprodutibilidade

Regras adotadas:

1. Seed mestre por run e seed derivada por ensemble
2. Configuração completa salva junto com resultados
3. Timestamp no nome dos arquivos
4. Scripts versionados no repositório (`scripts/`)
5. Outputs armazenados em `bench_reports/`

---

## 7) Formato dos resultados

## 7.1 JSON detalhado
Contém:
- linhas brutas por execução (`rows`)
- agregados (`agg`)

Campos típicos por linha:
- algoritmo
- família
- parâmetros
- seed
- `mse_final`, `msemin_final`, `emse_final`, `misadjustment`
- `runtime_s`, `runtime_per_sample_us`

## 7.2 CSV agregado
Uma linha por `(algoritmo, params)` contendo médias e desvio padrão entre seeds.

Campos:
- `algo`, `family`, `is_complex`, `n_runs`
- `mse_mean`, `mse_std`, `emse_mean`, `misadj_mean`
- `us_per_sample_mean`
- `params` (JSON serializado)

---

## 8) Critérios de seleção de melhores configurações

Usamos três visões:

1. **Top por acurácia** (menor `mse_mean`)
2. **Top por velocidade** (menor `us_per_sample_mean`)
3. **Pareto frontier** (trade-off acurácia × custo)

A decisão final depende do caso de uso:
- produção em tempo real: priorizar velocidade
- modelagem offline/acurácia crítica: priorizar erro

---

## 9) Fluxo recomendado de trabalho

1. Rodar `smoke` após mudanças de API/algoritmo
2. Rodar `dev` para ajustar grids
3. Rodar `release` antes de publicar versão
4. Atualizar `docs`/README com:
   - melhores configs por família
   - tabela resumida
   - gráfico Pareto (quando disponível)

---

## 10) Riscos e cuidados

- **Incompatibilidade de parâmetros**: mitigado via catálogo de parâmetros
- **Comparação injusta entre famílias**: mitigado por suites separadas
- **Overfitting em seed única**: mitigado com múltiplas seeds
- **Tempo muito alto**: mitigado por modos (`smoke/dev/release`)
- **Métricas erradas para tarefa**: usar métricas específicas por suite

---

## 11) Roadmap de benchmark

## Fase 1 (concluída/parcial)
- Pipeline base com grid + seeds + agregação
- Export JSON/CSV
- Ranking e leitura de resultados

## Fase 2 (em andamento)
- Separação por suites
- Modo `smoke/dev/release`
- Salvamento periódico (checkpoint)

## Fase 3 (próxima)
- Métricas especializadas para blind/IIR/nonlinear/kalman
- Gráficos automáticos (convergência + Pareto)
- Relatório único por execução (`summary.md`)

---

## 12) Comandos exemplo

> Ajuste os algoritmos conforme a suite ativa.

```bash
# Dev rápido (linear)
python scripts/benchmark_grid.py \
  --algos LMS,NLMS,AffineProjection,RLS,RLSAlt,FastRLS,QRRLS \
  --seeds 0,1,2 \
  --ensemble 40 \
  --K 2000 \
  --save-every 50

# Release (mais robusto)
python scripts/benchmark_grid.py \
  --algos LMS,NLMS,AffineProjection,RLS,RLSAlt,FastRLS,QRRLS \
  --seeds 0,1,2,3,4 \
  --ensemble 100 \
  --K 4000 \
  --save-every 100

```

13) Checklist antes de publicar versão

 Smoke passou em todas as suites habilitadas

 Dev benchmark sem erros críticos

 Release benchmark finalizado

 Top configs documentadas

 Pareto atualizado

 README/docs atualizados com números da release

 Arquivos de benchmark arquivados em bench_reports/


 