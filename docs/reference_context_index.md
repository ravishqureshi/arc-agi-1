# Documentation Context Index

Purpose: A fast map for humans and AIs to locate the right document quickly. This index lists every first‑class doc and what questions it answers.

Last updated: Post-M10 (API/SDKs, contract drift protection)

---

## Top‑level

- README.md
  - What: Project intro, high‑level goals, quickstart pointers.
  - When to read: First contact with the repo.

- docs/OVERVIEW.md
  - What: Product/system overview, major components and their roles.
  - Answers: “What is Opoch‑OO?” “Which modules exist and why?”

- docs/ARCHITECTURE.md
  - What: Normative architecture (ownership, flows, state machines, repo layout).
  - Answers: “Who owns what?”, “Where do checks happen?”, “What does each folder contain?”
  - Notes: Live C‑ABI header is thin; extended ABI is in docs/ENGINE_ABI.md.

- docs/IMPLEMENTATION_PLAN.md
  - What: Milestones and sub‑milestones (M0..M5x) with acceptance criteria.
  - Answers: “What’s shipped?”, “What’s next?”, “Why split the work this way?”

- docs/ENGINE_ABI.md
  - What: Draft/full C‑ABI reference surface (planning). Live header stays minimal.
  - Answers: “Which C symbols and structs exist or are planned?”
  - Read with: engine/ffi/include/opoch_engine.h (thin runtime header).

- docs/core/
  - Universal_theorem.pdf / .txt, Technical_paper.txt, RBT_abstract_V1.txt
  - What: Theory background and research context.
  - Answers: "What theory informs receipts/passports and verification?"
  - Indifference_as_the_Simulator.md
  - What: Universal RBT engine semantics (Φ = E + iD, sequences, NG proof).
  - Answers: "How does IAS work?", "What are free/paid operations?"

- docs/IAS_INTEGRATION_BRIEF.md
  - What: Practical integration guide for IAS overlay on M3-M5.
  - Answers: "How to add Φ/sequences to planner?", "How to tag beats?", "Where to compute NG proof?"

---

## Contract & Schema (docs/contract/)

- docs/contract/atlas.yaml
  - What: Machine-readable field atlas for QP.eq and PDE.poisson2d (91 paths, all gates).
  - Answers: "What JSON paths does code actually use?", "Which fields are required/optional?", "What's the V1 capability status?"
  - Cross-reference: contract_plan.md (WO-ATLAS-01), v1_capabilities/v1_solver_capabillity_view.md.

- docs/contract/prx.profile.v1_1.json
  - What: Canonical PRX schema v1.1 (JSON Schema Draft 7, frozen). Post-compile shape for all gates.
  - Answers: "How do I validate canonical PRX?", "What's the strict contract?", "Which fields are required per kind?"
  - Cross-reference: CANONICAL_SCHEMA_GUIDE.md (usage), validate_canonical.sh (validation), contract_plan.md (WO-CANONICAL-01).

- docs/contract/views/
  - What: Minimal/common/advanced views per subtype (qp.eq, pde.poisson2d). LLM-friendly field slices.
  - Answers: "What fields do I need for minimal QP/PDE?", "What's truly required vs optional?"
  - Cross-reference: atlas.yaml (source), contract_plan.md (WO-VIEWS-01).

- docs/contract/pcs/
  - What: Authoring templates (qp.eq.v1, pde.poisson2d.v1). Authors fill, compiler derives remaining fields.
  - Answers: "How do I author QP/PDE?", "Which fields are user-provided vs compiler-derived?"
  - Cross-reference: pcs_authoring.schema.json (validation), contract_plan.md (WO-PCS-01).

- tools/derive/
  - What: 6 deterministic scripts compute derived fields (blake3, shape, bytes, nnz, tolerances, mesh metadata).
  - Answers: "How to compute derived fields?", "What are Tier-1 tolerance values?"
  - Cross-reference: contract_plan.md (WO-DERIVE-PACK-01), run_all_self_tests.sh (validation).

- prx-compiler/
  - What: PCS → Canonical PRX compiler. S4 bit-repro, calls derive scripts, generates compile-receipt.
  - Answers: "How to compile PCS?", "What fields are derived vs defaults?"
  - Cross-reference: contract_plan.md (WO-COMPILER-01), tests/integration_t1_*.rs (T1 validation).

When to read: Authoring PRX, validating canonical PRX, implementing gates, generating schemas.

---

## Normative specs (spec/)

- spec/PRX_SCHEMA.md
  - What: PRX envelope; kinds, inputs, units, conservation, policy.
- spec/PROBLEM_PRX_SCHEMA.md
  - What: Problem‑level schema for nodes (QP/LP/DEC/PDE...).
- spec/RECEIPTS_SPEC.md
  - What: Receipt envelope, determinism stanza, required‑by‑kind matrix.

Use when: validating specs, writing adapters, or implementing verifiers.

---

## Bounds & Determinism (docs/bounds/)

- docs/bounds/S2_DETERMINISM.md
  - What: Formal specification of S2 (bounded) determinism for GPU operations.
  - Answers: "What is S2 determinism?", "How do bounded operations work?", "What goes in bounded_info?"
  - Use when: Implementing GPU kernels, writing S2 receipts, validating bounded operations.
  - Cross-references: GPU_BOUNDS_MODEL.md, RECEIPTS_SPEC.md Section 3.1

## Mathematical Constants (opoch-constants/)

- opoch-constants/src/lib.rs, opoch-constants/src/*.rs
  - What: Single source of truth for all mathematical constants (IEEE-754, FFT, PCG, gamma constants).
  - Answers: "Where are mathematical constants defined?", "How to avoid bare literals?", "What constants are available?"
  - Use when: Need any mathematical constant (unit roundoff, tolerances, algorithm factors), replacing bare literals.
  - Cross-references: engine/src/bounds/*.rs (imports these), receipts/src/bounds_helpers.rs (re-exports for receipts).

## Engine (engine/)

- engine/DETERMINISM.md, engine/DETERMINISM_BOUNDS.md, engine/GPU_BOUNDS_MODEL.md
  - What: Determinism policy (bit‑repro vs bounded), bounds math, GPU S2 bounds model.
  - Answers: "What is S2 bounded determinism?", "How are GPU bounds computed?", "What goes in bounded receipts?"
  - Cross-reference: docs/bounds/S2_DETERMINISM.md for formal S2 specification.
- engine/src/bounds/*.rs
  - What: Error bounds computation modules (FFT, PCG, QP, PDE, gamma_n, operation counts).
  - Answers: "How are error bounds computed?", "What is the bounds model for each operation?", "Where are norm factors defined?"
  - Files: mod.rs (exports), fft.rs, pcg.rs, qp.rs, pde.rs, gamma.rs, ieee.rs, tolerances.rs, norm_map.rs, opcounts.rs, linalg.rs (RRQR/rank detection), obstruction.rs (pricing)
  - Use when: Computing error bounds for operations, implementing bounded determinism.
  - Cross-references: Imports constants from opoch-constants/, used by planner/src/bounds.rs for composition; engine/src/qp/reduced_kkt.rs uses linalg RRQR.
- engine/src/gpu/
  - What: GPU kernels with bounded determinism (M7).
  - Answers: "How do GPU operations maintain S2 bounds?", "How is FFT deterministic?", "GPU PCG implementation?"
  - Files: env.rs (detection), reduce.rs (pairwise), dot.rs (two-pass), spmv.rs, pcg.rs, fft.rs
- engine/tests/m7_*_test.rs, gpu_*_harness.rs
  - What: 100-run S2 verification harnesses, Monte Carlo bounds validation.
  - Answers: "How to test bounded determinism?", "How to verify GPU error bounds?"
- engine/ffi/include/opoch_engine.h
  - What: Thin live C‑ABI header (only implemented symbols).
  - Pair with: docs/ENGINE_ABI.md for the planned/full ABI.

---

## Receipts Core (receipts/)

- receipts/src/envelope.rs, model.rs, canonical.rs
  - What: Core receipt infrastructure - envelope structure, canonical JSON serialization, BLAKE3 IDs.
  - Answers: "How are receipts structured?", "How is canonical JSON implemented?", "How are receipt IDs generated?"
- receipts/src/bounds_helpers.rs
  - What: Re-exports mathematical constants from opoch-constants for receipt generation.
  - Answers: "Where do receipts get mathematical constants?", "How to avoid circular dependencies?"
  - Cross-references: Imports from opoch-constants/, used by policy_admissibility.rs.
- receipts/src/determinism_bounds.rs
  - What: S2 bounded determinism implementation for receipts.
  - Answers: "How are S2 bounds tracked in receipts?", "What determinism info goes in receipts?"
  - Cross-references: docs/bounds/S2_DETERMINISM.md, engine/src/bounds/*.rs.
- receipts/src/policy_admissibility.rs
  - What: Policy admissibility validation - corridor bounds, gamma_n overflow protection.
  - Answers: "How are PRX tolerances validated?", "What are corridor bounds?", "How is gamma_n overflow prevented?"
- receipts/src/obstruction.rs, solver_obstruction.rs
  - What: Mathematical and API obstruction handling.
  - Answers: "How are obstructions reported?", "What causes mathematical obstructions?"
- receipts/src/specs/kkt.rs
  - What: KKT receipt structure (thresholds, metrics, determinism).
  - Cross-references: engine/src/qp/reduced_kkt.rs (generator), spec/RECEIPTS_SPEC.md (schema).
- receipts/src/passport.rs, signing.rs
  - What: Run passport generation and cryptographic signing.
  - Answers: "How are passports created?", "How is signing implemented?"

---

## Runtime (opochrt/)

- opochrt/README.md
  - What: Runtime overview (beats, capabilities, receipts writer, fail‑closed).
- opochrt/src/
  - beat.rs: Beat scheduler and timing (Instant for durations, UTC stamps).
  - capability.rs: Capability tokens (IoRead/IoWrite), enforcement, usage logging.
  - engine.rs: Engine bridge, KKT thresholds extraction.
  - receipt.rs: ReceiptCollector, validation (hashes, schema, ids).
  - writer.rs: Canonical JSON writes, atomic rename, Windows‑safe filenames.
  - validator.rs: Fail‑closed rules; required receipts per problem kind.
  - session/: Session lifecycle modules (modularized in M6.5-4)
    - mod.rs: Main session module, DO/STORE/UNDO/RELEASE orchestration.
    - pde.rs: PDE-specific receipt generation, Stokes/Energy verification.
  - types.rs, error.rs, lib.rs: Shared types, errors, and exports.
- opochrt/tests/
  - receipt_integration_tests.rs: End‑to‑end receipt pipeline.
  - capability_integration_tests.rs: Capability enforcement + audit trail.
  - pde_receipt_integration_test.rs: PDE receipt generation and verification tests (M6.5-4).
  - m5f5_finance_qp_e2e_test.rs: Complete finance QP pipeline (M5f-5).
  - m5f6_*_test.rs: Integration testing suite (M5f-6): performance, error handling, CI verification.

When to read: implementing features around beats, receipts, or capability control; debugging runtime behavior; adding tests.

---

## API & SDKs (engined/, python-sdk/, tests/e2e/)

- engined/openapi.yaml
  - What: OpenAPI 3.0 contract (4 endpoints, fail-closed 422 responses).
  - Cross-reference: python-sdk/OPENAPI_LOCK.b3 (BLAKE3 lock), .github/workflows/ci.yml:15-36 (drift gate).
- engined/src/handlers/*.rs
  - health, spec_lint, plan, solve, verify, passport: Endpoint implementations.
- python-sdk/README.md
  - SDK usage, canonical methods (lint_spec/plan_and_run/get_receipts/pack_passport), no-policy principle.
- python-sdk/opoch_sdk.py
  - OpochSyncClient, OpochAsyncClient: 1:1 API mirroring.
- python-sdk/tools/check_openapi_lock.py
  - BLAKE3 drift detection.
- tests/e2e/*.py
  - E2E against live engined, CPU S0 determinism validation.

When to read: Implementing REST endpoints, using SDK, validating API contracts, E2E testing.

---

## Planner (planner/)

- planner/PLAN_RULES.md
  - What: Planning/fusion rules (legality, boundaries).
- planner/DEFAULT_SOP.md
  - What: Default operating procedures/policies for planning.
- planner/src/
  - plan.rs: Plan DAG structure, Φ summary, sequences (M5.5 IAS).
  - bounds.rs: Pipeline bounds composition (M7.7) - `bound(B∘A) = bound(B) + ‖J_B‖ * bound(A)`.
  - receipt.rs: Plan receipt generation with placement and fusion info.
  - placement.rs: CPU/GPU placement engine with rationale.
  - costs.rs: Cost prediction models for budgets.

---

## Examples (examples/)

- examples/README.md
  - What: How golden runs are structured.
- examples/golden/finance_qp/README.md, examples/golden/pde_poisson/README.md
  - What: Golden artifacts and expected receipts.
- examples/test_fixtures/README.md
  - What: Fixture conventions.

---

## Scripts

- scripts/check-no-stubs.sh
  - What: CI guard to prevent stubbed symbols from slipping in.
- scripts/add-license-headers.sh
  - What: Utility for license headers.
- scripts/gpu/ (M7.8)
  - check_bounded_determinism.sh: Run GPU tests N times, verify max_spread/bound ≤ 1.
  - compare_gpu_cpu_perf.sh: Measure GPU vs CPU speedups, enforce ≥2x gate.
  - collect_bound_utilization.sh: Gather bound tightness metrics.
  - generate_ci_summary.py: GitHub Actions summary generator.
- scripts/hardening/
  - run_all_scanners.sh: Master script to run all mathematical quality scanners.
  - scan_*.sh: Individual scanners for placeholders, thresholds, determinism, etc.

---

## Mathematical Quality (docs/hardening/)

- docs/hardening/README.md
  - What: Research-grade mathematical quality process documentation.
  - Answers: "How to ensure mathematical correctness?", "What is the hardening process?"
- docs/hardening/HARDENING_MANIFEST.yaml
  - What: Defines closure packs (logical file groupings) for systematic review.
  - Answers: "Which files are reviewed together?", "What scanners apply to each pack?"
- docs/hardening/acceptance/
  - What: Acceptance criteria for each closure pack (IAS_core, QP_KKT, PDE_DEC, etc.).
  - Answers: "What requirements must each pack meet?", "What constitutes PASS vs FAIL?"
- docs/hardening/reviews/
  - What: Stored review results from math-correctness and determinism reviewers.
  - Answers: "What issues were found historically?", "How were they resolved?"

Use when: Implementing new mathematical features, ensuring deterministic execution, achieving research-grade quality.

---

## How to navigate by task

- "I need to validate a PRX file": spec/PRX_SCHEMA.md, spec/PROBLEM_PRX_SCHEMA.md, linter/ (M1b) [not all files present].
- "I need to understand fail‑closed receipts": docs/ARCHITECTURE.md §6.4, spec/RECEIPTS_SPEC.md, opochrt/src/validator.rs.
- "I need to handle mathematical obstructions": receipts/src/solver_obstruction.rs (receipt generation), engine/src/obstruction.rs (detection), spec/RECEIPTS_SPEC.md §5.5.
- "I need HTTP API error obstructions": receipts/src/obstruction.rs (API layer), engined/src/response.rs (mapping), engined/openapi.yaml (contract).
- "I need to enforce file capabilities": docs/ARCHITECTURE.md §6.3, opochrt/src/capability.rs, opochrt/src/session/mod.rs (DO + write paths).
- "I need the C‑ABI reference": engine/ffi/include/opoch_engine.h (thin), docs/ENGINE_ABI.md (full draft).
- "I need repo map": docs/ARCHITECTURE.md Appendix B (updated), this context_index.md.
- "I need to understand IAS/Φ integration": docs/IAS_INTEGRATION_BRIEF.md, docs/core/Indifference_as_the_Simulator.md.
- "I need to add Φ/sequences to planner": planner/src/plan.rs, planner/src/receipt.rs (M5.5).
- "I need to tag beats with phi_op": opochrt/src/beat.rs, opochrt/src/session/mod.rs, receipts-core envelope (M5.5).
- "I need to compute NG proof": engine/src/pde/natural_gradient.rs (implementation), opochrt/src/session/pde.rs:769-875 (PDE NG proof generation).
- "I need to implement PDE receipt generation": opochrt/src/session/pde.rs, prx_io/src/pde.rs (M6.5-4).
- "I need to verify Stokes/Energy for PDE": opochrt/src/session/pde.rs verify_stokes_and_get_residuals(), verify_energy_and_get_conservation().
- "I need to ensure mathematical correctness": docs/hardening/README.md (process), docs/hardening/HARDENING_MANIFEST.yaml (closure packs).
- "I need to run quality scanners": scripts/hardening/run_all_scanners.sh (all), scripts/hardening/scan_*.sh (individual).
- "I need mathematical constants": opoch-constants/ (IEEE-754, gamma, FFT, PCG constants), use via imports not bare literals.
- "I need to compute error bounds": engine/src/bounds/*.rs (FFT, PCG, QP, PDE bounds), imports constants from opoch-constants/.
- "I need bounds for receipts": receipts/src/bounds_helpers.rs (re-exports), receipts/src/determinism_bounds.rs (S2 bounds).
- "I need deterministic execution": engine/DETERMINISM.md (policy), scan_determinism_maps.sh (check HashMap/HashSet).
- "I need S2 bounded determinism": docs/bounds/S2_DETERMINISM.md (formal spec), engine/GPU_BOUNDS_MODEL.md (primitive bounds).
- "I need to write S2 receipts": spec/RECEIPTS_SPEC.md Section 3.1 (structure), docs/bounds/S2_DETERMINISM.md Section 5 (receipt fields).
- "I need to validate bounded operations": engine/tests/monte_carlo_bounds.rs (reference harness), docs/bounds/S2_DETERMINISM.md Section 6.2 (requirements).
- "I need to review mathematical quality": docs/hardening/acceptance/ (criteria), use math-correctness-reviewer and determinism-code-reviewer agents.
- "I need to fix hardcoded constants": opoch-constants/ (single source of truth), scan_threshold_sources.sh (detect), ensure all tolerances from PRX policy or opoch-constants.
- "I need GPU bounded determinism": engine/src/gpu/ (kernels), engine/tests/m7_*_test.rs (harnesses), scripts/gpu/ (CI).
- "I need pipeline bounds composition": planner/src/bounds.rs (composition formula), opochrt/src/session/mod.rs (runtime validation).
- "I need GPU CI infrastructure": .github/workflows/gpu.yml (CI matrix), scripts/gpu/ (test scripts).
- "I need to call API from Python": python-sdk/README.md, python-sdk/examples/*.py.
- "I need API contract": engined/openapi.yaml, docs/ARCHITECTURE.md §10.
- "I need contract drift detection": python-sdk/tools/check_openapi_lock.py, .github/workflows/ci.yml:15-36.
- "I need fail-closed API semantics": engined/src/response.rs, docs/ARCHITECTURE.md §10.
- "I need canonical SDK methods": python-sdk/README.md:28-33.
- "I need rank-deficient QP solver": engine/src/qp/reduced_kkt.rs (solver), engine/src/bounds/linalg.rs (RRQR), engine/tests/qp_rank_*_tests.rs (validation).
- "I need to solve QP/PDE end-to-end": engined/src/handlers/solve.rs (POST /solve pipeline), python-sdk solve() method, engined/openapi.yaml (contract).
- "I need to know what PRX fields exist": docs/contract/atlas.yaml (91 paths, consumed_by/tier/v1_status for each).
- "I need V1 capability boundaries": docs/v1_capabilities/v1_solver_capabillity_view.md, docs/contract/atlas.yaml (v1_status field).
- "I need to validate PRX against canonical schema": docs/contract/prx.profile.v1_1.json (schema), docs/contract/validate_canonical.sh (validator), docs/contract/CANONICAL_SCHEMA_GUIDE.md (guide).
- "I need to author a QP/PDE problem": docs/contract/pcs/ (templates), docs/contract/views/ (minimal fields).
- "I need to compute derived fields": tools/derive/ (hash_inputs, compute_q_dims, mesh_counts, suggest tolerances).
- "I need to compile PCS to canonical PRX": prx-compiler/ (opoch prx-compile), tests/integration_t1_*.rs (T1-QP-01, T1-PDE-01.5).

---

## Conventions

- Deterministic timing: durations via std::time::Instant; UTC via SystemTime/chrono.
- Canonical JSON: receipts serialized via receipts-core canonical::CanonicalJson.
- Atomic writes: write‑to‑temp then rename; Windows filename sanitization for receipt ids.
- Capability logging: always log operations; in permissive mode token_id="permissive".

---

If a document appears out of date or a folder is missing from Appendix B, update both `docs/ARCHITECTURE.md` (repo layout) and this index in the same PR.
