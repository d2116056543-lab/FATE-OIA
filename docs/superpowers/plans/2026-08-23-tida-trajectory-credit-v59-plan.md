# TIDA trajectory credit V5.9 implementation plan

1. Add RED tests for order-independent bias cancellation, antisymmetric control, and
   saturating support.
2. Implement control-centered credit and the bounded support transform in
   `TIDATrafficTrajectoryHead`.
3. Run the focused trajectory-head tests, then the full targeted TIDA suite remotely.
4. Run a persistent one-epoch head-only probe over all train-core clips and all 885 test
   clips; compare against the V5.8 and frozen EMA baselines.
5. Continue only when full-test metrics and causal diagnostics show useful traffic
   contribution; never substitute a test-oracle threshold for deploy performance.
