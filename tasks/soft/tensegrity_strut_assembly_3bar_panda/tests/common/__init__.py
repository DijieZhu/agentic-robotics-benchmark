"""shared/chain/common: grader helpers shared by every chain task (CHAIN_PLAN.md sections 2.1, 4.0).

  seeds           OS-entropy hidden seeds logged as hidden_seed_hi / hidden_seed_lo, optional pinned list (D6)
  anchored_mean   live-anchored episode-mean criterion: clamp(mean_agent / mean_oracle, 0, 1) with the zero-oracle guard
  mjcf_whitelist  "MJCF is data": element / attribute whitelist and bands (rule F1; 05 B3 / A1)
  mjcf_canonical  canonical form of a robot subtree for identity check C3 (1e-9 numeric, attribute order free)

Destined for shared/robotics_grader/ in the benchmark repo (seeds.py replaces the kit's seeds.py,
anchored_mean lands in scoring.py); until then tasks vendor these files into tests/.
stdlib only in seeds / anchored_mean / mjcf_whitelist / mjcf_canonical (the kit rule: a verify
stage cannot assume numpy).
"""
