# Introduction drafting prompt

Draft the Introduction section of our MACE manuscript in LaTeX. Read the latest
`docs/paper/MACE_Introduction_Methodology_Outline.md` and
`docs/paper/MACE_Verified_Introduction_Sources.md` on the
`mace-floor-sensitivity` branch of `LoganAlexanderBurnett/RABL`. Use the original
`main.tex` and `refs.bib` I supplied as reusable material, checking citation
metadata against the verified source list. Use my supplied papers *Variational
digital twins* and *High-fidelity Modeling of Pressurized Water Reactor Flow
Fields for Data-Driven Applications* to inform the scientific tone, rhythm,
diction, syntax, and paragraph structure. If those attachments are unavailable,
use the outline's documented scientific writing profile and identify the missing
inputs briefly; do not claim to have read them.

Write approximately 1,400–1,800 words, with an opening motivation passage,
`\subsection{Literature review}`, and
`\subsection{Objectives and contributions}`. Follow this narrative:
temporally structured control-trajectory sampling; accurate autoregressive
ensemble forecasting; MACE combining forecast-dependent ensemble spread with
conformal calibration; calibration and recalibration for different evaluation
distributions; and operation-informed microreactor transients as the final
application test. Give the sampling, forecasting, and uncertainty-method
contributions substantive treatment. Operational transients should not become
the entire story.

Use the agreed terms: control trajectory; output variables; multivariate response
trajectory; trajectory sample; forecast trajectory; calibration distribution;
evaluation distribution; MACE-Horizon; MACE-Trajectory; Horizon complete
multivariate coverage; and Trajectory complete multivariate coverage. Use x(t)
for the physical system state. Explain that MACE-Horizon calibrates each
horizon–output pair, MACE-Trajectory calibrates each output's maximum normalized
error over time, and joint trajectory calibration targets every output at every
evaluated horizon. Per-output objectives do not automatically establish complete
multivariate coverage. Adaptive width does not mean online model retraining.

Build the literature review from connected themes and representative verified
sources, rather than one summary per paper. Distinguish MACE's specific
construction and evaluated workflow from established normalized conformal
prediction, ensemble methods, and multi-step forecasting. Use the suggested
BibTeX keys in `\cite{...}` commands; label preprints accurately. Do not invent
citations, claim unsupported priority, or import another paper's numerical
performance into claims about our models.

State four concrete contributions following the outline. Explain that the models
and training scaler remain fixed during recalibration, and that training on a
different distribution can be compatible with conformal calibration when the
calibration and evaluation trajectory samples are exchangeable. The original
calibration arm evaluated on operational transients is a distribution-transfer
diagnostic, not an unchanged 95% guarantee under shift. Recalibration does not
repair the forecast mean.

Use the current Modelica/Dymola implementation as the plant authority: outlet
temperature and quality are calculated, not fixed. Describe a reduced,
simulator-based offline forecasting demonstration; do not imply experimental
plant validation, closed-loop control deployment, or an established speedup.
Exclude the broader ±30° study completely. Mention low-spread output variables
only briefly; steam quality must not become a separate introductory theme.

Use restrained engineering prose and explicit transitions. Introduce the practical
problem before methods; avoid promotional language and repetitive gap claims.
Do not quote numerical results that have unresolved recurrence/provenance issues.
The introduction needs no new figure; leave detailed equations, settings, and
figure specifications to the methodology. End with a concise manuscript roadmap.

Return the complete Introduction LaTeX, followed by a short list of citation or
evidence items that still need confirmation. Do not draft other manuscript sections.
