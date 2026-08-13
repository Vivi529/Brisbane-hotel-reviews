Five-algorithm unified comparison
=================================

Files
-----
1. run_five_algorithm_comparison.py
   Single entry point for M0, M1, M1-I, modified MOEA/D and SHAMODE.
2. MOO_MO_SHADE_IPEA_priority_initialization_fixed.py
   Common hotel MOO problem definition and MO-SHADE core.
3. ipea_m1_extensions.py
   M0/M1/M1-I initialization switches.
4. modified_moead_comparison_adapter.py
   Modified MOEA/D mechanism under an exact FE budget.
5. shamode_comparison_adapter.py
   Literature SHAMODE mechanism under an exact FE budget.

Run
---
Place all five files in the same directory. Confirm BASE_DIR and input file names
in MOO_MO_SHADE_IPEA_priority_initialization_fixed.py, then execute:

    python run_five_algorithm_comparison.py

Start with RUN_MODE = "pilot". After checking logs and outputs, change to
RUN_MODE = "formal".

Default formal design
---------------------
- Algorithms: 5
- K: 10, 15, 20
- rho: 0.30, 0.40, 0.50
- Seeds: 42...61 (20 runs)
- MO-SHADE reference generations: 300
- All algorithms receive exactly the same objective-function-evaluation budget.

Main performance outputs
------------------------
- HV, IGD+, additive epsilon+, spacing and objective extent
- HV-FE curve and normalized HV-FE AUC
- feasible and nontrivial-feasible rates
- pairwise C(A,B) coverage indicator
- Friedman tests and algorithm ranks
- pairwise Wilcoxon tests with Holm correction across all 10 algorithm pairs
  within the same scenario and metric
- common robust representative-plan diagnostics and recommendation stability

Important interpretation
------------------------
- Compare HV/IGD+ only within the same (K, rho) scenario.
- Representative robust scores are internally normalized per Pareto front and
  are not compared across algorithms.
- Runtime is implementation-dependent and should be treated as secondary to
  equal-FE Pareto-quality measures.
