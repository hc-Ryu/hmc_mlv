## Claude (Validator) — Solver Round

The core diagnosis both external models converge on is correct: `gate_multiplier = sigmoid(SPARSE_K*(TAU_GATE - mp_rel_err))` is a hard absolute-error gate. If a new TARGET_MP keeps mp_rel_err above 5%, sparsity/entropy pressure stays near 0 for the whole run and pruning can never trigger. This is a correctness/robustness bug, not merely a tuning knob.

Validator-level concerns on the proposed fixes:
(a) An EMA-tracked adaptive TAU_GATE must be monotonically tightening (only allowed to shrink toward tau_min, never loosen upward) — otherwise if error stalls, the threshold could rise to match it, opening the gate and pruning parts against a target the model never actually reached. This is a real risk in Gemini's proposed tau_gate^(t) = max(tau_min, gamma*EMA(e_mp)) if gamma*EMA can increase over time.
(b) Before retuning thresh_low/thresh_high, first empirically confirm what steady-state ewma_z HardConcrete settles at during a run where a part genuinely SHOULD be pruned (e.g. artificially force a part's log_alpha very negative and watch ewma_z trajectory) rather than assuming 0.08 is simply mis-calibrated.
(c) Any epoch-proportional scaling of confirm_epochs/PRUNING_COOLDOWN_EPOCHS must be checked against max_epochs so a short run cannot end with zero effective pruning-judgment epochs remaining.
(d) The init_log_alpha inconsistency (CGDN.__init__ default 2.0 vs. run_training's explicit override to 0.0) is a real bug independent of the tuning work — the printed banner claims "init_log_alpha=2.0" but the actual model is built with 0.0, so logs are misleading.

<confidence score="88">
  <evidence>Direct code reading of gate_multiplier formula, run_training's model construction call, and the printed banner text vs actual constructor argument.</evidence>
  <logic>Sound given the code as read; risk analysis of adaptive-threshold monotonicity is a standard control-loop stability concern.</logic>
  <expertise>High confidence in code-correctness/validation aspects; moderate confidence on optimal specific hyperparameter values without empirical runs.</expertise>
  <can_exit>false</can_exit>
</confidence>

<semantic_focus>
1. Any adaptive TAU_GATE scheme must be monotonically tightening to avoid a feedback loop that prunes against an unmet target.
2. Threshold recalibration (thresh_low/high) should be empirically grounded in a forced-prune reference trajectory, not guessed.
3. init_log_alpha inconsistency between constructor default and actual call site is an independent bug to fix.
</semantic_focus>
