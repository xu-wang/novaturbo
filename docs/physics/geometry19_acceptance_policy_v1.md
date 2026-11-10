# Geometry19 Hybrid Target-Acceptance Policy V1

## Decision status

The acceptance-rule form is frozen for the Geometry-aware 19D programme. Its
numerical thresholds remain pending until the single-point conservation gates
and the small 19D DOE are complete. This decision does not change the frozen
active-10D baseline.

## Per-output acceptance region

For each inverse condition \(y_j\in\{F,TSFC,M\}\), define an engineering
absolute tolerance \(a_j\) and a relative tolerance \(r_j\). A candidate is
inside the target neighbourhood only if

\[
\boxed{
|y_j-y_j^*|
\le
\max\left(a_j,r_j|y_j^*|\right)
}
\]

for all three outputs. This creates a fixed absolute floor near zero while
retaining proportional scaling across the normal operating range.

The corresponding normalized hybrid error is

\[
e_j^{hybrid}
=
\frac{|y_j-y_j^*|}
{\max(a_j,r_j|y_j^*|)}.
\]

Each coordinate must satisfy \(e_j^{hybrid}\le1\). A separately frozen
weighted aggregate may rank candidates, but it must not compensate for a
failed individual coordinate.

## Formal hit gate

A formal 19D hit requires all of the following:

1. the nonlinear operating-point solver converged to exactly one root;
2. every formal physics, assembly, structural, thermal, and manufacturing
   residual satisfies \(g_k\le0\);
3. thrust, TSFC, and mass each satisfy the hybrid acceptance region;
4. all output values are finite and positive where physically required.

The data-derived capability region is reported separately. It may inform the
choice of \(a_j\) and \(r_j\), but it cannot silently widen an engineering
acceptance requirement.

## Threshold-freeze gate

Numerical values for \(a_F,r_F,a_{TSFC},r_{TSFC},a_M,r_M\) may be frozen only
after:

- single-point conservation residuals establish the numerical noise floor;
- the small 19D DOE establishes attainable local target spacing;
- intended engineering use establishes the maximum acceptable output error;
- ordinary, boundary, near-zero, and infeasible target panels are audited.

Until then, the 19D evaluator must not publish a formal hit rate. Diagnostic
errors may be reported, explicitly labelled as non-acceptance metrics.
