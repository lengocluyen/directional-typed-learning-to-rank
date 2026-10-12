# DT-LTR experiment protocol

Protocol date: 2026-10-10. This is the bounded development experiment for
*Directional typed learning to rank for heterogeneous RDF entities*. The
archival datasets and generated mechanisms have already been examined;
consequently this is a development reanalysis, not a preregistered independent
confirmation. Create the immutable run manifest before fitting. Resume is
rejected after code, data, configuration or runtime changes.

The standalone entry point is `scripts/run_experiment.py`, implemented in
`dtltr_vehicle.experiments`. See the [run guide](../README.md) for commands and
the [method pipeline](../README.md#method-pipeline) for the construction and outputs.

## Questions and comparisons

1. Does direction improve over matched typed symmetric utility? Compare DT with
   ST within each dataset, reference, encoding policy and support budget.
2. What benefit comes from reference information? Compare reference-free
   direct/spline controls with equally informed direct_ref/spline_ref controls.
   The latter receive raw candidate/reference coordinates and typed reference
   differences. CatBoost receives the direct_ref encoded features and learns
   PairLogit item utilities as a nonlinear capacity control.
3. How sensitive are outcomes to reference completion and numerical clipping?
   Cross raw/default completion with clipped/unclipped coordinates. Keep test
   rows and labels identical across methods and these four blocks.
4. How do calibration, selected penalties and property contributions vary
   across refits? Separate selection and calibration roles; report boundary
   flags, solver convergence, calibration and common probes.
5. Do analytical records faithfully correspond to the declared RDF registry?
   Compare every declared value and missing annotation, including IRI-valued
   sets. This does not test arbitrary schema inference or graph reasoning.

No hypothesis promises that DT must beat direct controls. Retain all methods,
blocks, failures and unfavorable results. Do not select the best test block as
a new headline result. Main DT contrasts are ST, direct_ref and spline_ref; TD
and reference-free controls describe representation and information gaps.

## Representations and learners

Eight representations are implemented: TD, ST, DT, direct, direct_ref, spline,
spline_ref and catboost_ref. Linear additive learners use convex logistic or
multinomial loss, ridge, optional property-group regularization and, in ANU,
quadratic population anchoring. TD learns nonnegative simplex property weights
over fixed fitted typed distances. Its group penalty is constant on that
simplex and is fixed to zero; it is not a sparsity intervention. The sum-one
constraint fixes TD's overall coefficient scale, while ST/DT coefficient
magnitudes are fitted under regularization. This utility-scale distinction is
separate from numerical preprocessing scales, which remain fixed after typed
map fitting. A positive postfit probability temperature rescales utilities
while preserving ranks. No learned utility-scale TD comparator or learned
latent-reference-profile benchmark is included in this experiment.
CatBoost uses PairLogit item utilities and leaf regularization, which is not
equivalent to linear ridge or multinomial loss.

Scales, vocabularies, defaults and spline knots are fitted from training
entities only in generated, McFadden and external experiments. ANU uses its
known fixed catalogue of 10 cars in exp1 and 20 in exp2 for label-free
preprocessing. This is a transductive known-item experiment, not cold-item
transfer.

Reference-free controls have no reference access. Reference-aware controls
share the same reference source and completion/clipping policy. Human spline
complexity is fixed; generated and external complexity is selected from the
bounded knot grid. CatBoost depth and human iteration count are fixed, and its
leaf penalty is tuned. Generated/external iteration counts use the declared
bounded grid. These capacity choices do not imply equal numbers of
hyperparameter candidates.

Missingness is represented by the declared feature map. ST/DT set features
distinguish observed empty sets from absent values; missing-reason identities
retained in RDF are collapsed by the encoder. The registry audit reports that
collapse. Additive property contributions sum exactly to an uncalibrated
pairwise margin. For McFadden predictions the declared contrast is chosen
utility minus mean offered utility. These are model explanations, not causal
effects or human preference ground truth. CatBoost has no intrinsic
property-additive decomposition here.

## Archival ANU

Run both experiments. Outer folds hold out entire elicitation sessions. Each
remaining partition has disjoint population-training, development and
calibration sessions. Repeated seeds refit the procedure. Sessions are the
statistical units; repetition does not multiply independent respondents.

Within each development/calibration/test session reserve the maximum support
pool once. Draws permute that pool, budgets use nested prefixes, and every
budget evaluates exactly the same remaining comparison groups. Unused
reserved support rows never enter evaluation, including at budget zero.
Population penalties and adaptation strengths are selected using development
evaluation labels. Calibration evaluation labels select one probability
temperature per budget. Test evaluation labels enter evaluation only.

Each development, calibration or test session adapts using only its own
reserved support and the selected population coefficient anchor. No adapted
coefficients or support labels are transferred between roles. The standard
profiles use the catalogue reference. If the supported support-reference mode
is enabled, a session reference is derived only from its own support; training
references use a separately reserved subset of each training session's choices.
A support-derived predictive reference is not a directly observed
psychological reference.

Full support budgets are 0, 2, 5, 10 and 20. CatBoost is excluded from anchored
ANU adaptation because tree warm starts do not implement the same quadratic
population anchor; do not silently substitute another adaptation method.
Fixed label-free catalogue probes support descriptive contribution comparisons.

## Archival McFadden--Train

Split choice groups into four disjoint roles. If the archive exposes a
verified respondent identifier, keep the whole respondent in one role and use
that identifier as the reporting unit. Otherwise retain entire choice sets
and report that panel dependence cannot be resolved from the archived
interface. Anonymous choice sets must not be treated as verified independent
people. Fit all eight models under catalogue and design_proxy references.

The design proxy is entangled with presentation/body-type structure. Its test
performance cannot establish a causal reference effect or a directly elicited
human reference. Preserve that caveat in reports. Fixed label-free choice-set
probes compare contributions across refits.

## Generated catalogue mechanisms

Use the 2,309-entity ADEME catalogue and eight mechanism definitions:
additive_similarity, sparse_similarity, signed_linear, gain_loss,
offset_ideal_point, lexicographic, informative_missingness and query_interaction.
For each seed, hash-partition entities into train/development/calibration/test
at 60/15/10/15 percent. Each raw query and its candidate pair belong wholly to
one role; no entity crosses roles. Use the exact same triples and noisy labels
for all methods and policies.

The generating utility uses a truth map fixed from the raw catalogue, separate
from every learner's training-only map. It produces labels and clean
evaluation truth, and is never supplied to a fitted encoder, model selection
or probability calibration. Query-dependent utility is therefore evaluation
truth, not an additional learner input.

Five mechanisms draw utility parameters by seed: additive_similarity,
sparse_similarity, signed_linear, gain_loss and offset_ideal_point. The other
three utility definitions are fixed, while entity pools, sampled comparisons
and noise vary by seed. Repeats in those first five mechanisms vary both
utility parameters and data; they do not isolate refitting under one identical
mechanism.

Evaluate strict clean generating preferences, excluding generating ties and
giving prediction ties half credit. Evaluate log loss and Brier score against
observed noisy labels separately. These are different endpoints. Report
excluded ties, incomplete queries, missing candidates and values outside the
training scale. Strata never enter overall averages as extra samples.
Generated seeds are descriptive mechanism replications, not independent human
respondents, and receive no population confidence interval.

Common raw catalogue probe pairs may include entities from all pools, but have
no choice labels or mechanism truth. They are used only for descriptive
stability and never affect preprocessing, fitting or selection.

## RDF correspondence

Audit declared predicates against analytical records, entity identifiers,
scalar datatypes, set IRIs and missing-reason annotations. Respect the
catalogue's six-significant-digit numerical serialization and report rounding
explicitly. A correspondence failure aborts the task. The registry audit covers
2,309 entities and 14 properties; broad graph reasoning remains outside scope.

## Independent observed validation interface

The external suite does not create new observed choices. A supplied bundle
must contain genuine permitted observations, source/licence/reference
provenance and preassigned train/development/calibration/test roles. A cold
transfer bundle must keep query and candidate item IDs disjoint across all
roles. User/group overlap is forbidden by default; a warm protocol permitting
it must state the reason. Software validates internal consistency and optional
hashes, not the truth of a researcher's claim of previous exposure.

Use separate sealed warm and cold bundles if both are needed; relabeling a
warm run does not create cold evidence. Freeze methods before opening
independent test labels. Repeated seeds on a preassigned bundle only vary
learner initialization and do not create independent splits. Independent
human reference/presentation validation and a new domain remain pending until
suitable observations exist.

## Bounded profiles and inference

The files `configs/smoke.yaml`, `configs/pilot.yaml`, `configs/full.yaml` and
`configs/ablation-smoke.yaml` define separate studies. Smoke checks execution
on small samples. Pilot checks the four policies and repeated fits on bounded
samples. The ablation smoke profile checks repeated fits across the four
policies on smaller samples. Full uses all archived observations, five ANU
folds, five seeds, three support draws, all eight mechanisms and the declared
bounded grids: 101 tasks without an external bundle. Restricting smoke to
audit/generated creates three tasks and needs only bundled data.

Profile test outcomes must not be used to expand the full grid. Report
selected boundary values as limitations. ANU and external pairwise outcomes
are averaged within each session/group after averaging support draws and
repeated test appearances, then equally across units. McFadden units follow
the available grouping. Matched differences use identical saved cases.
Conditional paired unit bootstraps describe the saved fitted procedure, not
full training uncertainty. Repeat ranges and standard deviations describe
partition/refit sensitivity; generated sensitivity also includes seed-varying
mechanism parameters as specified above. No multiplicity-adjusted familywise
claim is implemented, and individual contrasts are not all confirmatory tests.

Store per-task predictions, splits, tuning candidates, calibrated temperatures,
encoder metadata, convergence diagnostics, timings and source/data/runtime
hashes. Save finite JSON atomically, verify completed tasks on resume and
reject changed identities. Reports contain Markdown, numerical CSVs, saved
fits and diagnostics. No figures are generated. Compact exports omit bulky
full task files; retain `tasks/` for recomputation and audit.
