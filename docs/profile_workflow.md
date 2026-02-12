# Profiling Workflow (PydaptiveFiltering)

## 1) Run raw profiling
```bash
python scripts/benchmark_profile.py \
  --algos LMS,NLMS,AffineProjection,RLS,RLSAlt,FastRLS,QRRLS,SMNLMS,SMAffineProjection,LRLSPosteriori,NormalizedLRLS,OLSBLMS,CMA,Kalman \
  --seeds 0,1,2 \
  --ensemble 40 \
  --K 2000
