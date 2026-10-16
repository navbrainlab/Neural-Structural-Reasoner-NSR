
## WN18RR

- C-layer train time: 0.90s
- W_RR train time (best trial): 1.5085s
- Search time: 363.16s (20 trials)
- Best valid MRR: 0.4727
- Test no-C MRR: 0.3714
- Test with-C MRR: 0.4795
- Test Hits@1/3/10: 46.72% / 48.84% / 50.07%
- Test no-C eval time: 8.35s
- Test with-C eval time: 17.78s
- Best params: `{"W_RR_learning_algorithm": "cooccurrence", "T_thresh": 0.12008684679993031, "W_RR_normalize": "sigmoid", "W_RR_sigmoid_k": 0.8993716843652962, "agg_mode": "max", "alpha": 1.7066873546728665, "tail_fallback_weight": 0.0033623836779215046, "concepts_fallback_weight": 0.27702845088762973, "concepts_fallback_k": 10, "C_activity_threshold": 0.9015053810712772, "C_rule_threshold": 0.45836666277786614, "C_min_support": 24}`
