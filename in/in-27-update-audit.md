---
doc_revision: 1
reader_reintern: "Reader-only: re-intern if doc_revision changed since you last read this doc."
---

# in-27 Update Audit (Delta + Correctness Impacts)

This note captures the impact of the edits made to `in/in-27.md` after it was staged. It focuses on semantic correctness, testability, and alignment with existing Prism contracts (`q`/denotation invariance, BSPˢ gauge symmetry).

---

## Summary
The edits make the document more visionary but less formally actionable. The main risks are:

1. **New semantic claim without definition** (`Holographic Collapse`).
2. **Weakened proof/justification** for blind packing (removed Haar/Z-order rationale).
3. **Loss of explicit hysteresis guard-band** in the lung controller.
4. **Test protocol ambiguity** (no numeric thresholds, dangling variables).
5. **Mismatch between m5 entropy model and m4 damage metric** (`damage_rate` still used as pass condition).
6. **Stable-sort requirement weakened/implicit**, risking ordering drift without clarity.

---

## Delta Highlights (What Changed)

1) **Fractal → Holographic**
- Reframed the model as “Holographic Geometric” and added “Holographic Collapse.”
- Implies a unification of logic and locality via interning + Morton ordering.

2) **Host tuning dropped**
- “Host via env vars” removed; now “no longer tuned by host.”

3) **Blind packing justification rewritten**
- Removed explicit Haar/Z-order reasoning; replaced with “data/space collapse guarantees optimal packing.”

4) **Lung properties reduced**
- Removed explicit “Guard Band” statement for hysteresis stability.

5) **Tests made less precise**
- “Peak at bit 10” / “Smeared entropy” are now vague.
- “Minor Spill ()” leaves dangling variable.
- “Interleave perfectly” lacks a measurable criterion.

6) **Pass condition still uses `damage_rate`**
- That is the **m4 linear boundary metric**, while m5 defines spectral entropy as the primary signal.

---

## Correctness & Consistency Impacts

### A) “Holographic Collapse” is undefined elsewhere
- **Risk:** Can be read as a semantic identity claim (space == meaning), which conflicts with `q`/denotation invariance.
- **Need:** Define it explicitly as a **performance-only gauge framing** (BSPˢ only, erased by `q`).

### B) “Space is a projection of data” is too strong as written
- **Risk:** Implies uniqueness of topology from address; not true under arbitrary scheduling. 
- **Need:** Qualify as “projection *in the scheduling gauge*; erased by `q`.”

### C) Blind-packing now asserts optimality without proof
- **Risk:** Strong claim without the Z-order/Haar rationale.
- **Need:** Either restore the justification or downgrade to a design target validated by tests.

### D) Removal of guard-band/hysteresis clause
- **Risk:** Without explicit guard-band, oscillations are plausible. 
- **Need:** Add back a stability constraint or explicit hysteresis invariant.

### E) Test protocol no longer enforceable
- **Risk:** Tests can pass without proving the intended invariants.
- **Need:** Reintroduce explicit thresholds (bit index expectations, entropy dispersion constraints, buffer/active densities).

### F) `damage_rate` still used as m5 pass condition
- **Risk:** Measures the **wrong signal** under m5. 
- **Need:** Include entropy-spectrum pass criteria, and optionally keep damage_rate as a legacy check.

### G) Stable sort requirement is weakened
- **Risk:** Masked Morton equivalence depends on stability if “super-particles” are meant to be preserved.
- **Need:** Declare stability required (or explicitly allow reordering, with a denotation invariance test).

---

## Minimal Repair Recommendations

1. **Define “Holographic Collapse”** in glossary or in-27 as **performance-only**, BSPˢ gauge, erased by `q`.
2. **Qualify the “space = projection of data” claim** as gauge-only.
3. **Reintroduce a guard-band / hysteresis constraint** in the lung section.
4. **Tighten test criteria** with numeric/structural expectations (bit peaks, entropy thresholds, buffer/active densities).
5. **Update pass condition** to include entropy-spectrum bounds; keep `damage_rate` only as a legacy check.
6. **Clarify sort stability** for masked Morton (required or explicitly not required).

---

## Recommendation on Next Step
If the edits are meant to stand, they should be followed by a short appendix or glossary addition that normalizes the new “holographic” language to the existing contract:

- **Performance only.**
- **Erased by `q`.**
- **Must commute with denotation.**

Otherwise, the document should revert to the prior version’s concrete justifications and explicit test thresholds.

---

## Update 2 (Post‑revision: “Holographic Architecture”)
This section evaluates the *latest* edits (title change, explicit gauge warning, guard band restored, stable sort requirement, numeric test thresholds, glossary appendix).

### What improved
1. **Guard band / hysteresis restored**  
   The “Else = elastic zone” text reintroduces the stability invariant; this fixes one of the earlier regressions.
2. **Stable sort requirement made explicit**  
   This clarifies the intended invariant for “super‑particles” and lets us write a direct test/contract.
3. **Numeric test assertions added**  
   The probe/controller/packing tests now have thresholds, making the plan more actionable.
4. **Glossary appendix added**  
   “Holographic Collapse” and “Super‑Particle” are defined (though not yet merged into `in/glossary.md`).

### New issues introduced
1. **Definitions still placeholders / undefined variables**  
   - Table entries for “Damage Rate” and “Spectral Energy” are blank.  
   - Holographic Collapse steps (interning/morton/collapse) are blank.  
   - Controller variables , ,  and energy density in  are undefined.  
   - Test thresholds reference `histogram[10]`, `total_edges`, `threshold`, and `spectral_entropy` without defining how these are computed or normalized.
2. **Markdown formatting errors**  
   - “Scheduling Gauge” line has a stray space and mismatched backticks: `erased by the projection `q**` (unbalanced).  
   - Several inline equations are missing variables (now rendered as empty italics).
3. **Stable sort requirement conflicts with current code path**  
   - The implementation uses `jnp.argsort` (unstable by default) or lexsort without a tie‑breaker.  
   - If stability is required, the sort key must include a deterministic tiebreak (original index) or use a stable sort API.  
   - Without this, the new requirement is *not implementable* in the current system.
4. **“Guaranteeing optimal packing” claim still too strong**  
   - The text asserts optimal packing “via 2:1 Morton recursion invariant” without a formal bound.  
   - This remains a design target, not a proven property; it should be downgraded or qualified.
5. **Probe threshold may be unrealistic**  
   - `histogram[10] > 0.9 * total_edges` assumes near‑perfect concentration at one bit.  
   - For realistic ID distributions, this may be too strict or dependent on interning order.  
6. **Legacy damage metric kept as a hard pass**  
   - `damage_rate(tile=512) == 0.0` is fragile under any scheduling nondeterminism.  
   - If this is only a sanity check, treat as “≤ epsilon” or “expected‑not” rather than a hard gate.

### Consequences / impact
1. **Implementation scope expanded**  
   The “servo” field in `Arena` is now a breaking change that must be propagated through all constructors and tests.
2. **Test suite is now sensitive to signal calibration**  
   Without defined normalization for entropy and histogram, tests may become flaky or meaningless.
3. **Docs now imply stable ordering as a contract**  
   This will force either a stable sort implementation or a precise definition that allows instability.

### Minimal repair actions (recommended)
1. **Fill all placeholders**  
   Explicitly define spectral energy, histogram bins, `total_edges`, and normalization in §2.1/§2.2/§3.3.
2. **Fix markdown and grammar**  
   - “will implements” → “will implement”  
   - fix the `q` backtick mismatch  
3. **Reconcile stability with implementation**  
   - Add a tie‑breaker (`key = (masked_morton, original_index)`), or  
   - declare stability as a *goal* for m5 with a concrete plan in `IMPLEMENTATION_PLAN.md`.
4. **Calibrate thresholds**  
   Replace absolute values with “≤/≥ relative to baseline” if thresholds are unknown.
5. **Sync glossary**  
   Move the Appendix entries into `in/glossary.md` and add test obligations.

### Correctness verdict
The revision improves clarity and reintroduces necessary stability constraints, but it is not *yet* internally consistent or testable. The biggest correctness blockers are **missing definitions** and the **stable sort requirement** that the current implementation cannot satisfy without change.

---

## Update 3 (Latest edits: tightened language + partial repairs)
This pass adds explicit thresholds, relaxes some claims, and introduces a stable‑sort implementation hint. It is a net improvement, but still incomplete.

### What improved
1. **Claims softened to “design target”**  
   “Guaranteeing optimal packing” is downgraded to a target, which is more defensible.
2. **Gauge warning tightened**  
   Now explicitly scoped to BSPˢ and `q` erasure with correct emphasis.
3. **Normalization requirement added**  
   The probe now declares a normalized histogram, which makes entropy thresholds meaningful.
4. **Stable‑sort strategy suggested**  
   Provides a concrete implementation path (masked key + original index).
5. **Test thresholds relaxed**  
   `histogram[10]` threshold lowered and `damage_rate` softened (<0.01) to reduce brittleness.

### Remaining correctness gaps
1. **Placeholder math still unresolved**  
   - “Logic: .” in the probe section is still blank.  
   - Controller variables and ranges remain undefined (`K`, spill/active zones).  
   - Holographic Collapse steps still have blank equations.
2. **Quantities lack formal definitions**  
   `total_edges`, `entropy(histogram)`, and `spectral_entropy` aren’t defined or normalized.
3. **Sort stability still a contract mismatch**  
   The doc now *requires* stability. The code currently uses `jnp.argsort` without a stable tie‑break, so the implementation still violates the stated requirement.
4. **Test feasibility unclear**  
   The proposed thresholds might still be unachievable without a specified data distribution and explicit normalization scheme.

### Consequences
The document is closer to testable, but still not executable as a spec. It now *demands* specific implementation changes (stable sort) and test machinery (entropy computation), which must be reflected in `IMPLEMENTATION_PLAN.md` and the codebase before m5 can be considered “frozen.”

### Minimal follow‑ups to resolve this revision
1. Fill all placeholder formulas and variables (`K`, spill/active zones, entropy definition).  
2. Add a stable‑sort tie‑break in the scheduler or downgrade stability to a goal with a plan.  
3. Define exact computation of `spectral_entropy` and histogram normalization.  
4. Move “Glossary Updates” into `in/glossary.md` and add test obligations there.

---

## Update 4 (Latest edits: stability clarified, tests de‑specified)
This update introduces a more accurate statement about JAX sort stability on GPU, but it *removes* the concrete numeric assertions and re‑introduces placeholder math.

### Improvements
1. **Sort stability guidance now accurate**  
   The doc now explicitly warns that `jax.lax.sort` is unstable on GPU and requires a composite key `(masked_key, original_index)`. This is a real implementation constraint and should be kept.
2. **Probe logic slightly clearer**  
   “Calculate Hamming distance magnitude” is a clearer phrasing than the prior empty “Logic: .”

### Regressions / new gaps
1. **Verification thresholds removed**  
   - The earlier concrete thresholds (e.g., `histogram[10] > 0.8 * total_edges`, `entropy > 3.0`, `spectral_entropy < 1.5`) are now blank placeholders.  
   - This makes the verification plan non‑actionable again.
2. **Controller calculations blanked**  
   - The formulas for buffer/active densities are now empty.  
   - This un‑specifies the controller, undermining the Schmitt trigger definition.
3. **Normalization formula missing**  
   - Normalization is required but the exact normalization expression is blank.

### Consequence
The document is now more accurate about stability requirements but **less testable**. It regresses on the “design freeze” intent by removing numeric assertions and controller formulas.

### Minimal repairs for this revision
1. Restore concrete thresholds **or** define them as parameters with a measurement procedure.  
2. Fill in the explicit formulas for buffer/active densities and histogram normalization.  
3. Keep the GPU sort‑stability warning and add a corresponding implementation plan item.

---

## Update 5 (Cleanup + fully specified formulas)
The latest revision converts the pasted/corrupted Markdown into a clean, executable spec and restores all key formulas and thresholds. This materially improves correctness and implementability.

### What improved
1. **Spec is now parseable and unambiguous**  
   Headings, tables, and code fences are fixed. The document is now valid Markdown and safe for downstream tooling.
2. **All placeholders resolved**  
   - `H` normalization is defined.  
   - Buffer pressure (`P_buffer`) and active density (`D_active`) are defined.  
   - Control law thresholds are specified.  
   - Test assertions are explicit.
3. **Stable sort requirement is concrete**  
   The composite key `(masked_key, original_index)` is now the explicit GPU-safe path.

### Remaining correctness risks
1. **Metric definition mismatch (MSB vs entropy target)**  
   The spectral metric uses MSB (log2 of XOR distance). This is fine, but test thresholds (e.g., `H[10] > 0.8`) presume a very concentrated distribution; may be sensitive to ID allocation order and interning paths.
2. **Dilation/Contraction criteria slightly asymmetric**  
   - Dilation uses `P_buffer > 0.25`.  
   - Contraction uses `P_buffer < 0.10` *and* `D_active < 0.30`.  
   This is a reasonable hysteresis band, but it should be stated as a design choice rather than derived necessity.
3. **“Entropy < 1.5 bits” may be optimistic**  
   The packing test presumes a very low entropy after sorting. Might need calibration runs to validate.
4. **Naming: BSP^* vs BSPᵗ/BSPˢ**  
   The doc uses “BSP^* scheduling gauge.” That’s acceptable but should be aligned with glossary naming to avoid ambiguity.

### Consequences / impact
1. **Implementation can now proceed without additional design meetings**  
   The formulas are sufficient to code the probe and lung controller.
2. **Test plan is now strict and likely to surface calibration work**  
   These tests will fail if spectral energy behaves differently under real interning; expect an initial tuning phase.
3. **Stable sort contract forces code changes**  
   A composite key or stable sort variant must be added to the scheduling path, or the spec must explicitly allow instability.

### Correctness verdict
This revision is **substantively correct and actionable** as a design‑freeze candidate. The remaining risks are **calibration and alignment**, not semantic ambiguity. If the thresholds are validated empirically, the spec is ready to implement.

---

## Update 6 (Glossary-compliant BSPˢ framing + commutation laws)
This revision cleans the pasted text and explicitly aligns the servo with the glossary’s BSPᵗ/BSPˢ split, adding axis/commutation fields for new terms.

### What improved
1. **BSPˢ explicitly named as the axis**
   The servo is now scoped to BSPˢ (layout), removing the ambiguous “BSP*” notation.
2. **Commutation laws are declared**
   `q ∘ Servo = q`, `q ∘ Lung = q`, and `q ∘ Sort = q` are explicitly stated, matching glossary requirements.
3. **New terms get axis/erasure annotations**
   “Holographic Collapse” now has Axis and Erasure fields; the sensor/controller/sort list axis/commutation.
4. **Document is clean and executable**
   The corrupted Markdown is fully repaired, preserving formulas, thresholds, and implementation constraints.

### Remaining considerations
1. **Glossary alignment still pending**
   The appendix introduces BSPˢ / Entropyₐ / Super‑Particle updates but those need to be merged into `in/glossary.md` to be normative.
2. **Calibration risk remains**
   Thresholds (0.8, 3.0 bits, 1.5 bits) may still need empirical tuning; that’s a test‑calibration phase rather than a spec defect.

### Correctness verdict
This update is **fully compliant with the glossary’s BSPᵗ/BSPˢ distinction** and is now fit for implementation. Remaining work is glossary sync and calibration, not semantic correction.

---

## Update 7 (Final glossary‑compliant BSPˢ version)
This revision fully conforms to the Glossary Contract by explicitly tagging BSPˢ vs BSPᵗ axes, declaring commutation laws, and naming Entropyₐ as the driving signal.

### What improved
1. **Axis + commutation explicitly stated**
   - Servo/Lung/Sort now declare BSPˢ axis and `q ∘ (...) = q` commutation.
2. **Entropyₐ explicitly named**
   The signal is now unambiguously Arena microstate entropy (Glossary §28).
3. **BSPˢ gauge clarity**
   “Gauge Warning” now uses BSPˢ language and gives the exact commutation law.

### Remaining considerations
1. **Glossary appendix is still a duplicate**
   The appended BSPˢ/Entropyₐ/Super‑Particle definitions should be merged into `in/glossary.md` (already done in this working tree) and optionally removed from the appendix to avoid drift.
2. **Calibration remains empirical**
   Thresholds are explicit but still need empirical validation.

### Correctness verdict
This version is **normatively compliant** with the glossary (BSPˢ, Renormˢ, Gauge, Entropyₐ). It is suitable for implementation and test development, with remaining work focused on calibration and glossary duplication cleanup.