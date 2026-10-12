# MACE: detailed introduction and methodology outline

Prepared 12 October 2026 for the retained study on `mace-floor-sensitivity`. Implementation reference: commit `5b0f9f6a2702ab4b1b88a723d40cdba293d3c02a`. This is a writing and figure-production plan, not a completed manuscript or a request for additional model training. Citation IDs and proposed BibTeX keys refer to [MACE_Verified_Introduction_Sources.md](MACE_Verified_Introduction_Sources.md).

## 1. Narrative, scope, and writing approach

The central argument is that temporally structured control-trajectory sampling can support accurate autoregressive surrogates, and that ensemble disagreement can supply a useful local scale for conformal prediction. The resulting bounds require a stated coverage event and a calibration distribution appropriate to the evaluation distribution. Operation-informed transients provide the final application test of this argument.

Give each part of that argument substantive treatment: sampling, recursive prediction, the MACE formulation, and calibration/recalibration. Do not build the introduction around operational coverage alone. The broader ±30° study is excluded from the manuscript, figures, tables, and supplementary results. Keep its repository assets separate. Steam quality receives a short, appropriately qualified discussion as an example of low ensemble spread; it does not warrant a dedicated introductory theme or main-text figure.

### Writing profile from the supplied papers

The scientific writing references are Burnett et al., *Variational digital twins* (Energy and AI 24, 100756, 2026), especially its introduction/contributions on pp. 1–4 and formulation on pp. 5–7; and *High-fidelity Modeling of Pressurized Water Reactor Flow Fields for Data-Driven Applications*, especially its introduction on pp. 1–4 and modeling/methodology on pp. 4–7. These papers inform the presentation, not the physical assumptions of this plant model.

- **Opening and progression:** begin with a concrete physical modeling need, explain why existing approaches leave a practical gap, then state the work undertaken. Narrow from physical simulation to data design, forecasting, and uncertainty before naming the contributions.
- **Paragraph construction:** use a topic sentence, a mechanism or representative study, and its implication for the present problem. Group related studies rather than writing one paragraph per citation. A literature paragraph should lead somewhere.
- **Diction:** retain precise engineering terms such as governing equations, operating conditions, transient response, predictive accuracy, calibration, and computational framework. Define specialized terms on first use. Avoid promotional adjectives and unsupported “first” claims.
- **Syntax and rhythm:** alternate explanatory sentences with shorter statements of purpose or limitation. Introduce equations in connected prose and define their symbols immediately afterward. Use restrained connectors such as “In this work,” “For example,” “However,” and “Accordingly” where the relationship actually warrants them.
- **Authorial voice:** use “we” for implemented choices and “this study” for scope. Use present tense for equations and general relationships; use past tense for completed simulations, training, and evaluation. Reserve assertions about performance for measured results.
- **Contributions and boundaries:** state contributions in a short numbered list, then explain what is evaluated. Distinguish an offline simulator-based forecasting study from a deployed digital twin or closed-loop controller. Follow the supplied papers' practice of pairing a capability with its practical limits, while keeping repeated limitations out of the introduction.
- **Visual presentation:** use figures to explain mechanisms and tables for exact settings. Captions should identify the physical system, variables, units, sample source, and meaning of shaded regions; they should be understandable without searching the body text.

### Reuse from the RABL manuscript

| Supplied material | Treatment in the MACE manuscript |
|---|---|
| Introduction opening paragraph | Retain the physical motivation and useful citations; revise the destination from active learning/branching to sampling, autoregressive forecasting, and calibrated adaptive bounds. |
| Surrogate modeling and data generation; Data generation strategies | Merge into the first literature-review theme. Separate prespecified stochastic sampling from active simulation acquisition. |
| Deep learning for physical time series | Retain the sequence-model context; emphasize recursive deployment rather than one-step fit alone. |
| Uncertainty quantification and ensemble methods | Retain the ensemble motivation and replace the uncertainty-selection argument with normalized conformal calibration and clearly defined coverage events. |
| Artificial intelligence for control | Keep a short application-motivation paragraph. Do not imply that the present study learns or validates a control policy. |
| LSTMs for control-driven dynamical systems; Time-series forecasting with LSTM ensembles | Reorganize into §§2.1 and 2.3 below; correct dimensions, control timing, member-specific feedback, and scaler coordinates. |
| Data generation via Gaussian processes | Retain the velocity-GP/integrated-control construction; distinguish its design covariance from conditioning, numerical regularization, and bounded-path rejection in the code. |
| Demonstration on Nuclear Microreactor Plant Dynamics | Update from the executed Modelica model and experiment wrapper; see §2.5. |
| Appendix B, Reactor Plant Physics and Modeling Details | Retain and update the kinetics, thermal equations, and parameter tables. Add the dynamic steam-generator node and actual outlet closures. |
| RABL workflow, branching/acquisition sections, original complexity appendix | Exclude from this paper's method. Their existence in the repository does not make them part of the evaluated MACE workflow. |

## 2. Introduction: proposed manuscript structure

Suggested main-text length: approximately 1,400–1,800 words, adjusted to the journal. Use one opening passage, a literature-review subsection with five connected themes, and a contributions subsection. No introduction figure is needed; introduce the mechanism figures with their equations in the methodology.

### 1. Introduction — physical motivation and problem statement

**Paragraph 1: physical modeling need.** Adapt the original opening. Explain that control-driven energy systems exhibit coupled responses on different time scales, and that evaluating many candidate control trajectories with a physics simulator motivates surrogate forecasting. Introduce the microreactor as the demonstration without making nuclear safety, autonomous operation, or real-time speedup an evaluated claim. A simulator is the reference for surrogate error here; the real plant is not experimentally validated by this study.

**Paragraph 2: three linked requirements.** Explain why useful forecasting requires (i) training data that represent temporally coherent excitation, (ii) accurate recursive forecast trajectories under known future control, and (iii) uncertainty bounds whose coverage objective is explicit. Introduce the distinction between obtaining useful ensemble disagreement and obtaining calibrated prediction intervals. End with the central question: can a fixed autoregressive ensemble provide adaptive conformal bounds, and can those bounds be recalibrated for a different evaluation distribution when the mean forecasts remain useful?

### 1.1. Literature review

Use the following themes as paragraph groups, or as five short subsubsections if the journal permits. The source document supplies verified metadata and narrower claim boundaries.

| Theme | Paragraph content and intended transition | Primary citation pool |
|---|---|---|
| **Surrogate modeling and control-trajectory data generation** | Start with the dependence of learned physical surrogates on simulation-data selection. Contrast snapshot/parameter acquisition with excitation of a dynamical system through entire control trajectories. Introduce covariance-based control sampling as a way to specify temporal correlation and integrated variation. End by separating our prespecified construction from active learning and safety-constrained exploration. | S02 `pestourie2020surrogates`; S03 `kapadia2024surrogate`; S04 `zimmer2018safe`; S01 `porcu2024matern`. Use S05 `lundby2023identification` or S06 `xie2025design` as a labeled preprint only when needed. |
| **Deep learning for physical time series** | Explain why LSTMs can represent temporal dependencies and illustrate established nuclear/energy applications. Then distinguish training with observed histories from recursively feeding model predictions back over a full transient. Motivate testing full forecast trajectories and individual output variables. Do not infer our performance from these applications. | S07 `hochreiter1997lstm`; S08 `radaideh2020loca`; S09 `appiah2024coldtrap`. |
| **Ensembles and calibrated predictive uncertainty** | Explain that independently trained members provide forecast-dependent disagreement, which can vary over time and between output variables. State that disagreement alone does not establish a nominal coverage level, particularly under distribution shift. Introduce conformal prediction and prior normalized nonconformity scores; explain that the combination of adaptation and calibration is established context. | S10 `lakshminarayanan2017deep`; S11 `ovadia2019shift`; S12 `angelopoulos2023conformal`; S13 `lei2018regression`; S14 `papadopoulos2008normalized`; S15 `romano2019cqr`. |
| **Conformal prediction for forecasts and distribution changes** | Review multi-step forecasting, ensemble-based conformal methods, and multivariate/multi-horizon coverage. Compare calibration units and events, not just model names. Introduce whole trajectory samples as the exchangeability units in this study. End with the calibration/evaluation distribution distinction and the purpose of recalibration. | S16 `stankeviciute2021forecasting`; S17 `xu2021enbpi`; S18 `sun2024copulacpts`; S19 `barber2023exchangeability`. S28 is optional context for sequential recalibration, not a method implemented here. |
| **AI for control and microreactor application context** | Briefly establish why control-drum actions motivate control-conditioned forecasting. Explain that operation-informed maneuvers can differ temporally from stochastic training/calibration trajectories. Transition to an offline test of forecast accuracy and uncertainty after that change. Do not claim that literature control profiles are a measured probability distribution of actual operation. | S20 `price2022microreactor`; S21 `tunkle2025microreactor`. |

Suggested allocation: two paragraphs for sampling, one to two for sequence models, two for ensembles/conformal foundations, two for forecast coverage/distribution change, and one short control paragraph. Cite representative sources where they support a concrete claim; the introduction need not cite all 28 verified papers.

**Closing synthesis paragraph.** Connect the strands without declaring them absent from the literature: existing research provides physical surrogates, sequence models, ensemble uncertainty, and conformal forecasting. The present study brings a specified control-sampling construction and fixed autoregressive ensemble together with ensemble-normalized calibration, distinct coverage objectives, and a controlled calibration-transfer comparison. State that the contribution is the precise construction and evaluated workflow, not the invention of LSTMs, Gaussian processes, normalized conformal scores, or ensemble-conformal prediction.

### 1.2. Objectives and contributions

Use four contributions, ordered to match the eventual results:

1. **Control-trajectory sampling:** formulate and implement temporally correlated velocity sampling and integration to control trajectories, with explicit correlation, dispersion, initialization, and admissibility settings for simulation-data generation. Present this as the study's specific data-generation contribution; avoid an unsupported priority claim for integrated Gaussian processes.
2. **Autoregressive surrogate forecasting:** train a ten-member LSTM ensemble on trajectory-disjoint stochastic data and evaluate complete forecast trajectories using the correct input/output coordinate mapping. Establish accuracy separately from uncertainty calibration.
3. **MACE formulation:** use ensemble spread plus a common positive floor to normalize surrogate errors; define MACE-Horizon, MACE-Trajectory, and joint calibration for Trajectory complete multivariate coverage. Explain the selected score reductions and finite-sample calibration argument.
4. **Calibration and recalibration evaluation:** freeze the models and training scaler while comparing calibration distributions on the same operation-informed evaluation trajectories. Quantify accuracy, coverage, width, and interval score with paired whole-trajectory bootstrap uncertainty and maneuver-phase diagnostics.

Do not promise successful calibration transfer without evidence. Recalibration changes interval quantiles, not the ensemble mean. End with a compact roadmap: methodology follows this same sequence, results first establish stochastic-distribution prediction/UQ, then examine operation-informed forecasts and recalibration, and discussion interprets the coverage assumptions and method limits.

An optional literature comparison table can accompany §1.1 if useful: columns **approach**, **calibration/data unit**, **adaptation mechanism**, **coverage event**, and **relationship to this study**. Rows should cover normalized regression CP, CQR, EnbPI, multi-step conformal forecasting, CopulaCPTS, and the present formulation. Use descriptive entries rather than unverified yes/no capability scores.

## 3. Methodology: proposed manuscript structure

Suggested length: approximately 2,500–3,500 words in the main manuscript, with detailed physics, pseudocode, and audit records in appendices/supplement. The outline below is deliberately more detailed than that target: it specifies what the writer and figure creator must preserve.

### 2.1. Control-driven dynamical systems and forecast problem

**Purpose.** Define the general problem before the plant-specific example. Separate physical states from predicted output variables.

Write the reference dynamics and observation map as

$$
\dot{\mathbf x}(t)=f(\mathbf x(t),\mathbf u(t);\boldsymbol\eta),\qquad
\mathbf y(t)=g(\mathbf x(t),\mathbf u(t);\boldsymbol\eta),
$$

where $\mathbf x$ is the physical state, $\mathbf u$ the control, $\mathbf y\in\mathbb R^J$ the output variables, and $\boldsymbol\eta$ the fixed simulator parameters. The forecasting method uses a finite history of output variables and prescribed control; it does not require every output to be an independent differential state.

Define a **trajectory sample** as the paired control trajectory and simulator-generated multivariate response trajectory, together with initialization and sampling metadata. Define a **forecast trajectory** as the sequence of model predictions on the specified future grid. Use $i$ for trajectory sample, $h$ for forecast-grid index, $j$ for output variable, and $m$ for ensemble member. Use $N_{\rm cal}$, $N_{\rm eval}$, $H$, $J$, and $M$ for counts. Avoid using $M$ again for the number of GP velocity intervals; use $L$ there.

For this application, $\mathbf u(t)=\theta(t)$ is the common angle prescribed to twelve drums, $M=10$, $J=15$, and the saved response grid is $t_h=0.4h$ s for $h=0,\ldots,500$. Thus $H=501$ evaluated points span 0–200 s. Explain that “horizon” denotes a position in this fixed-origin forecast grid, not a new measured-history restart at every plotted time. The initial history is prepared from the configured steady condition; future simulated outputs are withheld during recursive prediction.

**Table 1:** compact notation/terminology table. Include all seven agreed trajectory/distribution terms and all four coverage events defined in §2.4. Do not label the 15 predicted quantities collectively as 15 dynamical states.

### 2.2. Stochastic control-trajectory sampling

**Purpose and paragraphs.** First explain why independently sampled control values ignore temporal structure. Then define the velocity covariance, integration, parameter interpretation, initialization, and admissibility. Finally describe how the generated controls enter Dymola and the trajectory-disjoint dataset.

On a control grid with spacing $\Delta_c=0.1$ s, sample interval velocities $v_r$, $r=0,\ldots,L-1$, with $L=2000$. Before initialization conditioning, their covariance is

$$
K_{rs}=s_v\left(1+a_{rs}+\frac{a_{rs}^2}{3}\right)e^{-a_{rs}}
+n_v\delta_{rs},\qquad
a_{rs}=\frac{\sqrt5\,|t_r-t_s|}{\ell}.
$$

Here $s_v$ is the velocity variance/sill in (degree/s)$^2$, $n_v$ is the nugget in the same units, and $\ell$ is the correlation length in seconds. The stationary, unconditioned velocity semivariogram is $\gamma_v(r)=K(0)-K(r)$, with the appropriate nugget convention at zero lag. The integrated angle is generally nonstationary; do not apply this stationary velocity semivariogram directly to angle.

Define $B\in\mathbb R^{(L+1)\times L}$ by $B_{kr}=\Delta_c$ for $r<k$, zero otherwise. Then

$$
\boldsymbol\theta=\theta_0\mathbf1+B\mathbf v,\qquad
\operatorname{Cov}(\boldsymbol\theta)=BKB^{\mathsf T}.
$$

This is left-endpoint interval integration, not trapezoidal integration. The Matérn regularity describes the GP prior; the exported angle table is discretized and interpolated by the Modelica input block.

Explain the terminal-dispersion design helper:

$$
A(\ell)=L+2\sum_{r=1}^{L-1}(L-r)\rho_v(r\Delta_c),\qquad
s_v=\frac{\sigma_{\theta,\mathrm{design}}^2/\Delta_c^2-n_vL}{A(\ell)}.
$$

**Important implementation qualification:** the helper solves this for the unconditioned prior. Actual generation conditions $v_0=0$; ideally its velocity covariance becomes

$$
K^{(0)}=K-K_{:0}K_{0:}/K_{00},\qquad
\operatorname{Cov}(\boldsymbol\theta\mid v_0=0)=BK^{(0)}B^{\mathsf T}.
$$

The implementation adds small numerical jitter, uses a regularized conditioning denominator, and explicitly sets the first velocity to zero. For exact numerical reproduction, use that actual linear transformation of the regularized covariance. Consequently, the configured 2.5° terminal standard deviation is an **unconditioned design target**, not an exact claim about the conditioned or accepted trajectory population. Reject generated paths outside 0–180°; acceptance further conditions the distribution. Neither a GP variance nor the code's union-bound rate helper imposes a hard rate limit on the accepted sampling procedure.

Current batch-3 configuration: Matérn-5/2, $\ell=10$ s, $\sigma_{\theta,\mathrm{design}}=2.5^\circ$, $n_v=0$, $\theta_0=45^\circ$, and seed 123. Verify the original generation records for every batch used in training before claiming all batches used these exact settings. Record generation count separately from the number of selected training/validation/calibration/evaluation trajectories.

The selected original data are split by trajectory sample: 5,000 training, 1,000 validation, 1,000 calibration, and 2,000 evaluation samples. No time window from one trajectory crosses these partitions. Confirm independence/lineage from the manifests; if shared-parent branching entered a partition, trajectory-ID disjointness alone would not establish independent trajectory samples. The evaluated manuscript workflow does not use conditional branching or uncertainty-driven sample acquisition.

**Figure 1:** new stochastic-control sampling figure, specified in §4.1. It should show the mechanism and parameter meanings, not a model-performance result.

### 2.3. LSTM ensemble training and recursive forecasting

**Data coordinates.** Explain train-only min–max fitting and separate native input/output coordinates:

$$
\widetilde \xi_j=(\xi_j-a_j^X)/b_j^X,\qquad
\widetilde y_j=(y_j-a_j^Y)/b_j^Y.
$$

Here $\xi_j$ denotes a neural-network input feature, distinct from the physical state vector $\mathbf x(t)$. The superscript $X$ identifies input coordinates. Here $a$ is the training minimum and $b$ the implemented span; the scaler replaces a zero span with 1.0. Reuse the saved scaler for all calibration/evaluation distributions. No new-data refitting and no clipping of scaled extrapolation are performed. Reconstruct input and output training arrays separately, including the prepared histories and window alignment, when independently auditing their minima and spans.

**Window and control timing.** The configured lookback is $K=12$, but the dataset builder includes both endpoints and forms **13 rows**. For a training prediction of $\mathbf y_{k+1}$, row $r=0,\ldots,K$ contains

$$
\big[\widetilde{\mathbf y}^{\,X}_{k-K+r},\;
\widetilde{\mathbf u}^{\,X}_{k-K+1+r}\big].
$$

Thus output history covers $k-12,\ldots,k$, and the prescribed controls cover $k-11,\ldots,k+1$. The tensor per sample is 13×16: 15 history features and one control feature. Superscript $X$ marks input coordinates, not a different physical output. Prepared steady rows support the first forecast at time zero.

**Architecture and training.** One 64-unit LSTM layer; the last sequence output enters Linear(64→64), ReLU, then Linear(64→15). Ten members use independently selected 70% subsets of training trajectories without replacement within each member; these subsets can overlap between members. Describe this accurately as subsample ensemble training rather than a classical with-replacement bootstrap of the training set. Train one-step mean-squared error with Adam, learning rate $3\times10^{-4}$, batch size 64, 100 epochs, StepLR every 30 epochs with factor 0.5, and restore the best validation checkpoint. The configured early-stopping patience is 1,000, so it does not ordinarily terminate a 100-epoch run. Confirm member seeds and checkpoint metadata for the final reproducibility table.

**Recursive inference.** Start each member from the same prepared history and let each member feed back its own output predictions. Known controls continue to enter the shifted window. Do not feed back the ensemble mean and do not refresh with future reference outputs. Before feedback, map each prediction to input coordinates:

$$
\widehat{\widetilde y}^{\,X}_{m,h,j}
=\widehat{\widetilde y}^{\,Y}_{m,h,j}\frac{b_j^Y}{b_j^X}
+\frac{a_j^Y-a_j^X}{b_j^X}.
$$

Cast at the model input boundary as implemented. The mapping is necessary even when input/output metadata look similar. A direct copy is justified only when the coordinates are actually identical.

Define the frozen ensemble mean and population spread in output-scaled coordinates:

$$
\mu_{i,h,j}=\frac1M\sum_m\widehat{\widetilde y}^{\,Y}_{m,i,h,j},\qquad
\sigma_{i,h,j}=\sqrt{\frac1M\sum_m
(\widehat{\widetilde y}^{\,Y}_{m,i,h,j}-\mu_{i,h,j})^2}.
$$

This uses `ddof=0`. Spread is not the standard error $\sigma/\sqrt M$ and is not automatically a Gaussian predictive standard deviation. Inverse-transform the mean with the native output scaler for physical-unit accuracy.

**Table 2:** settings table with dataset counts, 13×16 input, 15 outputs, ensemble size, member subsampling, architecture, optimizer, schedule, validation selection, saved-scaler provenance, and forecast recurrence. Include settings actually used, not all unused configuration options.

**Figure 2:** combine the supplied LSTM Architecture and NeuralAutoregression assets into a corrected two-panel figure; see §4.2. Move the corrected Rolling Forecast algorithm to Appendix A.

### 2.4. MACE: ensemble-normalized conformal prediction

**Paragraph 1: motivation and scale.** Use ensemble spread to adapt bounds to each forecast while conformal scores calibrate the scale empirically. With a common floor $\tau=10^{-3}$ in output-scaled units,

$$
d_{i,h,j}=\sigma_{i,h,j}+\tau,\qquad
s_{i,h,j}=\frac{|\widetilde y_{i,h,j}-\mu_{i,h,j}|}{d_{i,h,j}}.
$$

Keep one floor across output variables and horizons. “Adaptive” means forecast-dependent width through $d$; it does not mean online updating of the nominal coverage level or a learned conditional-quantile model.

**Paragraph 2: finite-sample calibration rule.** For $N_{\rm cal}=1000$ and $\alpha=0.05$, use the $k=\lceil(N_{\rm cal}+1)(1-\alpha)\rceil=951$st smallest calibration score, without quantile interpolation. Define the general convention as $Q_k$; if the required rank exceeds the calibration count, the usual theoretical construction uses an infinite threshold, while the current finite-threshold implementation rejects that configuration. This edge case does not arise in the reported study.

**Paragraphs 3–5: calibrated events.** State the score reduction, quantile dimensions, interval, and intended coverage event together:

| Construction | Calibration score and quantile | Intended event for a new trajectory sample |
|---|---|---|
| **MACE-Horizon** | For every $(h,j)$, $q_{h,j}=Q_k(\{s_{i,h,j}\}_i)$. Quantile array: $H\times J$. | $\Pr\{\widetilde y_{h,j}\in I_{h,j}\}\ge1-\alpha$ separately for each fixed horizon–output pair. |
| **MACE-Trajectory** | $R_{i,j}=\max_h s_{i,h,j}$, then $q_j=Q_k(\{R_{i,j}\}_i)$. Quantile vector: $J$. | $\Pr\{\forall h:\widetilde y_{h,j}\in I_{h,j}\}\ge1-\alpha$ separately for each output variable. |
| **Joint calibration for Trajectory complete multivariate coverage** | $R_i=\max_{h,j}s_{i,h,j}$, then $q_{\rm all}=Q_k(\{R_i\}_i)$. One scalar quantile. | $\Pr\{\forall h,j:\widetilde y_{h,j}\in I_{h,j}\}\ge1-\alpha$. |

For the appropriate quantile $q$, form $I_{i,h,j}=[\mu_{i,h,j}-q d_{i,h,j},\mu_{i,h,j}+q d_{i,h,j}]$. Joint calibration takes each calibration sample's maximum over both axes **before** the across-sample order statistic; it is not obtained by maximizing the per-output quantiles.

Define **Horizon complete multivariate coverage** as $\Pr\{\forall j:\widetilde y_{h,j}\in I_{h,j}\}$ at a fixed $h$. This is a useful diagnostic of the existing bands, but ordinary MACE-Horizon calibration does not target this joint event. A separate horizon-wise joint score $\max_j s_{i,h,j}$ would be needed to calibrate that event directly; that construction is not among the three evaluated procedures above. Do not silently relabel current horizon intervals as having 95% complete multivariate coverage.

Likewise, the average of the 15 per-output trajectory coverage rates is not Trajectory complete multivariate coverage. Keep the three constructions and their matching comparisons in separate result groups.

**Paragraph 6: calibration assumption and short justification.** Conditional on the frozen predictor, scaler, floor, and chosen score function, exchangeable calibration/evaluation trajectory samples yield exchangeable scalar scores for the selected event. The order-statistic argument then supplies the corresponding marginal coverage statement. Dependence among times or output variables within a trajectory is permitted. A training distribution different from calibration/evaluation does not by itself invalidate this split-calibration argument; a change between calibration and evaluation distributions removes the ordinary exchangeability guarantee. These statements concern the saved 501-point grid, not every continuous instant or every conditional family/phase.

**Paragraph 7: matched baselines and floor interpretation.** Define absolute CP using absolute residual scores, with the same horizon, per-output trajectory, or complete multivariate trajectory maximum and rank. Match its calibration arm and coverage event to each MACE comparison. If shown, label raw ensemble $\mu\pm2\sigma$ as an uncalibrated diagnostic.

When $\tau$ dominates spread, normalized scores approach absolute errors divided by $\tau$, and recalibrated MACE approaches matching absolute CP. Increasing the floor reduces spread-based adaptation; it need not widen every interval because the quantile changes as well. The floor was not selected by optimization of evaluation-set performance.

If the floor sensitivity study is included, specify the actual seven values
$\{10^{-6},10^{-5},10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}$.
Hold models, scaler, trajectory partitions, and forecasts fixed within that sweep; recompute horizon-pair quantiles and per-output trajectory maxima/quantiles at every floor. Record $\tau/(\sigma+\tau)$, per-output objective coverage, physical-unit widths, scaled aggregate widths, matched width ratios, and interval scores. Include empirical floor robustness only for forecasts generated with the same recurrence as the primary paper; see §5. The existing horizon/trajectory floor sweep does not by itself establish joint-calibration floor robustness.

**Figure 3:** new MACE construction/coverage-event figure, specified in §4.3. Add a short algorithm box if required by the journal; do not repeat the same equations in three diagrams.

### 2.5. Demonstration on nuclear microreactor plant dynamics

**Authority and model scope.** Use the executable Modelica equations, `RunOneProfile.mo` parameter bindings, and Dymola runner as the authority. Describe a deterministic reduced coupled point-kinetics/thermal-network model, not a high-fidelity full-plant simulator. Twelve synchronized drums impose a single control angle. Reactor parameters and other boundary inputs are fixed in this study.

**Main equations.** Retain the six-group point kinetics and drum/temperature feedback explanation from the supplied Appendix B:

$$
\dot n=\frac{\rho-\beta}{\Lambda}n+\sum_{r=1}^6\lambda_r c_r,\qquad
\dot c_r=\frac{\beta_r}{\Lambda}n-\lambda_r c_r,\qquad
P=P_r n,\quad P_r=6\;\mathrm{MW}.
$$

Define $\rho$ in $\Delta k/k$, and $\rho_{\$}=\rho/\beta$. Sum drum reactivity and fuel/moderator feedback using the actual cosine worth and reference temperatures. A concise main-text thermal balance can illustrate heat storage and transport; put the complete set and parameters in updated Appendix B.

There are **12 differential states**: $n$, six precursor quantities, and five thermal nodes $T_f,T_m,T_{hp},T_{N2},T_{sg}$. The **15 output variables** add algebraic $\rho_{\$}$, $T_{\rm steam,out}$, and $x_{\rm steam,out}$. Native storage order is
$T_{N2},T_m,T_{hp},T_f,T_{sg},c_1,\ldots,c_6,n,\rho_{\$},T_{\rm steam,out},x_{\rm steam,out}$.

**Current steam-generator behavior, briefly.** Explicitly include the active metal node:

$$
C_{sg}\dot T_{sg}=Q_{N2\to sg}-Q_{sg\to fw},\quad
Q_{N2\to sg}=G_{N2,sg}(T_{N2}-T_{sg}),\quad
Q_{sg\to fw}=\max\{G_{sg,fw}(T_{sg}-T_{sat}),0\}.
$$

The nitrogen-node energy balance loses $Q_{N2\to sg}$. With specific added heat $q=Q_{sg\to fw}/\dot m_{fw}$, the model uses

$$
x_{\rm steam,out}=\begin{cases}0&q\le0,\\q/h_{fg}&0<q<h_{fg},\\1&q\ge h_{fg},\end{cases}\qquad
T_{\rm steam,out}=\begin{cases}T_{sat}&q<h_{fg},\\T_{sat}+(q-h_{fg})/c_{p,sh}&q\ge h_{fg}.\end{cases}
$$

Outlet pressure and feedwater mass flow are prescribed (1.5 MPa and 2.95 kg/s); outlet temperature and quality are calculated, not fixed. Explain this as a simplified near-nominal closure. Preserve that distinction even when individual trajectories exhibit a temperature plateau or quality saturation.

For executed defaults with no overrides, the wrapper binds `C_sg=2.55e5` J/K; the plant class default is `8.775e5` J/K. Report the wrapper/configuration value actually executed, and enumerate any run overrides. Do not copy a class default or stale package comment into the manuscript parameter table. Response sampling is 0.4 s over 200 s and the wrapper specifies tolerance $10^{-8}$; obtain solver name/version and final run settings from execution metadata rather than guessing.

**Table 3:** output variable, symbol, physical interpretation, units, differential/algebraic status, and stored column. Include that the forecast output $n$ is normalized power, while the supplied dataset figure shows $P$ in MW. Report precursor units according to the normalization in the code rather than inventing physical concentration units.

**Figures 4 and 5:** supplied plant schematic and supplied 4×4 generated-transient dataset plot, with the corrections/captions in §§4.4–4.5. The figures establish physical meaning and dataset structure. They are not forecast-performance evidence.

### 2.6. Calibration distributions and operation-informed evaluation

**Purpose.** Describe two evaluation settings: the stochastic-distribution study and the retained operation-informed study. The second is a stress test of calibration distribution change with the same predictor, not a substitute for first establishing recursive accuracy and uncertainty on the original evaluation distribution.

For the operation-informed setting, generate independent calibration and evaluation sets from the same prespecified equal-probability B/C/D/F family mixture: 1,000 calibration and 2,000 evaluation trajectories. Draw maneuver parameters independently; store family, move times, endpoints, realized velocities, and profile IDs. The saved configuration uses seed 20261010 and independent child RNG streams for calibration and evaluation. Labels B/C/D/F refer to the previously defined families, not standardized reactor operating categories.

| Family | Definition in the retained recipe | Nominal sampling parameters before 0.4-s schedule rounding |
|---|---|---|
| **B: single maneuver and hold** | One constant-velocity move upward from 45°, followed by a hold. | Start 24–56 s; increase 1–3°; speed magnitude 0.05–0.15°/s. |
| **C: successive maneuvers and holds** | Three upward increments with intermediate holds. | Starts 24–36, 74–86, 130–142 s; each increment 0.5–1.25°; each speed 0.08–0.16°/s. |
| **D: maneuver and return** | Upward move, hold, and constant-velocity return to 45°. | Increase 2–3°; starts 20–32 and 116–124 s; each duration 40–60 s. |
| **F: bidirectional maneuvers** | Move above 45°, then below 45°, then return to 45°. | Starts 32–48, 84–100, 140–152 s; first level 45.25–45.75°, second 44.25–44.75°; speeds 0.04–0.12°/s. |

Draw times/rates/durations as implemented, round times to the response grid, and compute realized constant velocities from rounded endpoints. Distinguish requested rate distributions from realized rates. Across the retained recipe the possible angles span 44.25–48.75°; report the observed extrema separately from the allowed envelope. The controls have finite velocity during moves and idealized velocity jumps at corners; no finite-acceleration actuator model is evaluated. Their operation-informed structure is motivated by control literature, but their specific parameter ranges are study choices, not empirically established operational frequencies.

Compare two arms on exactly the same operation-informed evaluation forecasts:

- **Original calibration distribution:** original 1,000 stochastic calibration trajectories, forecast using the paper's corrected recurrence, with freshly recalculated quantiles. This is a distribution-transfer diagnostic; exchangeability with the operation-informed evaluation set is not assumed.
- **Operation-informed calibration distribution:** new 1,000 independent B/C/D/F calibration trajectories, using the same frozen models, scaler, floor, and recurrence. Calibration/evaluation samples share the prespecified mixture, supporting the corresponding pooled marginal objective.

Absolute CP must be recalibrated under each matching arm too. Recalibration does not retrain the LSTM ensemble or change its mean forecasts on the common evaluation set. Training on broader stochastic excitation can be compatible with calibration on a different distribution, but adequate mean forecasting in that distribution must be demonstrated; recalibration alone cannot repair the mean.

Define motion as $[t_{start},t_{end})$, early settling as $[t_{end},t_{end}+20\,\mathrm{s})$ with new motion taking precedence, initial hold as times before the first maneuver, and late holds as remaining times. Report all-time results plus motion, early settling, and late holds separately; report initial hold as supporting context. Phase definitions are prespecified and do not use forecast errors. Quantiles are calibrated on the full trajectory, not recalibrated separately by phase in this study.

**Table 4:** study partitions/arms, calibration and evaluation distributions/counts, what is frozen, and the applicable nominal coverage claim. Add the maneuver recipe table above if space permits; otherwise move its numeric ranges to the supplement.

**Figure 6:** new calibration-distribution study design plus control-family illustrations, specified in §4.6. Do not put operational performance numbers in this method figure.

### 2.7. Accuracy, coverage, efficiency, and comparison uncertainty

**Accuracy.** Compute MAE and RMSE for each output variable in physical units from the ensemble mean and native physical simulator truth. Normalize aggregate errors by the saved output spans. The retained operational implementation computes aggregate MAE as the equal-output mean of normalized MAEs and aggregate RMSE as the square root of the equal-output mean of normalized MSEs; the latter is not the mean of per-output RMSEs. Report all-time, family, and phase results with support counts. Do not substitute one-step validation loss for recursive accuracy.

**Coverage.** Let $A_{i,h,j}=\mathbf1\{L_{i,h,j}\le\widetilde y_{i,h,j}\le U_{i,h,j}\}$. Report separately:

1. Per-output, per-horizon coverage $N_{eval}^{-1}\sum_i A_{i,h,j}$, and per-output averages over the evaluated grid.
2. Per-output trajectory coverage $N_{eval}^{-1}\sum_i\prod_h A_{i,h,j}$.
3. Horizon complete multivariate coverage $N_{eval}^{-1}\sum_i\prod_j A_{i,h,j}$.
4. Trajectory complete multivariate coverage $N_{eval}^{-1}\sum_i\prod_{h,j}A_{i,h,j}$.

For phase-restricted trajectory diagnostics replace the all-time set by the supported times in that phase and exclude samples with no phase support. Keep the full-trajectory objective and phase-restricted diagnostic labels distinct. Pooled calibration gives no separate family-conditional guarantee. Analyze low-coverage horizons/output pairs descriptively, retaining the common time axis and maneuver timing; inspecting many cells does not yield simultaneous significance claims.

**Widths and interval scores.** Use full width $W=U-L$. Convert target-specific widths and scores to physical units with $b_j^Y$; retain scaled units for aggregates. Define matched width ratio $\overline W_{MACE}/\overline W_{absolute}$ and reduction $1-\overline W_{MACE}/\overline W_{absolute}$, recomputing the ratio from means rather than averaging individual ratios. Use

$$
IS_\alpha(L,U;y)=(U-L)+\frac2\alpha(L-y)\mathbf1\{y<L\}
+\frac2\alpha(y-U)\mathbf1\{y>U\}.
$$

Cite Gneiting and Raftery (2007), as verified in the source document's methods references. Lower interval score is better. For trajectory constructions, this is a point-averaged diagnostic applied to the band endpoints; it is not a proper scoring rule for the joint trajectory event or proof that a simultaneous coverage target was met.

Pool sums/counts over supported time points for pointwise accuracy, width, and score metrics; different dwell lengths therefore receive their actual time-point weight. Average output-scaled summaries with equal output weights for aggregates. Count each supported trajectory once for trajectory coverage. Show phase support counts so long holds cannot obscure transient misses and zero-support samples cannot inflate coverage.

**Paired whole-trajectory bootstrap.** Resample evaluation trajectory IDs with replacement 10,000 times (retained operational seed 20261011). Retain every selected sample's complete time series, output variables, family label, phase masks, and all methods/calibration arms. Use the same index draw for every comparison and recompute reductions, interval-score differences, and coverage differences within each replicate. Report percentile 95% intervals. For family-specific metrics restrict each common draw to that family's sampled members, as in the implementation; do not describe this as a fixed-count stratified bootstrap. Keep the models and calibration quantiles fixed. These intervals describe evaluation-sampling uncertainty conditional on the fitted models and calibration sets, not training/calibration uncertainty or simultaneous inference across every diagnostic cell.

**Numerical reproducibility.** Use one inclusive scaled-space coverage convention everywhere. The current evaluation code stores truth/mean/spread and constructs conformal scores/quantiles/endpoints in float32, then promotes endpoints/truth to float64 for width and interval-score arithmetic. Native physical truth is retained for accuracy. Do not claim float64 conformal score construction unless the implementation/results are changed accordingly. Inverse transformation is used to display physical values, not to rerun a different rounded coverage decision.

Require exact comparison of arrays with matching dtype/precision. When precisions differ, report the explicit conversion, exact equality after that conversion where applicable, and maximum absolute differences in both native and converted representations. Reconstruct a few actual physical/scaled samples for steam quality, another output, and drum angle; independently recompute scaler extrema from the raw training split; verify recurrence inputs; regenerate coverage from saved endpoints. Describe audit details in the supplement rather than making them a separate claimed algorithmic contribution.

## 4. Figure-production specifications

### Shared production rules and figure map

Use journal-neutral editable vector masters for new diagrams and scientific plots (PDF/SVG), plus 300-dpi or better PNG previews. This is manuscript production; do not change the retained operational plotting workflow, which already uses individual PNGs and one consolidated PDF. Use a white background, consistent math notation, approximately 8–9 pt final-size labels, panel letters outside data, and no captions baked into the image. Suggested double-column width: 178 mm. Check legibility at the journal's actual column width before final export. Use color plus line style, not color alone; avoid red/green-only comparisons.

| Main-text figure | Asset/status | First reference |
|---|---|---|
| **1. Stochastic control-trajectory sampling** | New, four scientific panels | §2.2 |
| **2. LSTM architecture and member-specific autoregression** | Adapt/combine supplied LSTM Architecture and NeuralAutoregression figures | §2.3 |
| **3. MACE calibration and coverage events** | New diagram with score reductions and event illustrations | §2.4 |
| **4. Microreactor plant schematic** | Use/adapt supplied `hpmr-bop-viz` | §2.5 |
| **5. Generated multivariate transient dataset** | Use supplied `timeseries_batch_0003_ALL_PROFILES`, preserving 4×4 layout | §2.5 |
| **6. Calibration comparison and operation-informed controls** | New study diagram plus four input plots | §2.6 |
| **A1. Recursive forecast algorithm** | Correct supplied Rolling Forecast figure, then move to Appendix A | Appendix A |

Figure numbers are reserved for this proposed ordering; renumber automatically if the journal reorganizes methods. Keep accuracy/UQ performance plots in the results section. The attached operational forecast PDF is a source for selecting result illustrations, not an introduction/methodology figure to reproduce in full. A floor-sensitivity result figure is supplementary and conditional on recurrence-consistent data (§5).

### 4.1. Figure 1 — stochastic control-trajectory sampling (new)

**Purpose:** show how correlated velocity becomes temporally coherent control and distinguish design dispersion from initialized sampling. Size: 178×125 mm; 2×2 grid, panel order (a),(b) top row and (c),(d) bottom row. Proposed master name: `fig01_control_trajectory_sampling.pdf`.

- **(a) Covariance/regularity knob:** plot the unconditioned Matérn-5/2 velocity correlation against lag, 0–60 s, with $\ell=5,10,20$ s. Use dashed gray/solid black/dotted gray curves respectively; highlight 10 s as the retained configuration. Y-axis “Velocity correlation,” range 0–1. These are analytic parameter illustrations, not three training datasets. Include the kernel formula in the caption or nearby body, not a dense annotation inside the axes.
- **(b) Sampled interval velocity:** use eight accepted original stochastic control trajectories chosen by ascending profile ID from the documented batch-3 manifest, with no selection by outcome. Plot their stored interval velocities on [0,200) s in degree/s, using the same eight colors in (c). Their trailing undefined velocity entry is omitted, not replaced with zero. Y-limits are symmetric about zero with 10% padding beyond the largest absolute selected velocity. Mark the zero line. If original velocity tables cannot be recovered, generate eight explicitly labeled *illustrative* trajectories with the exact batch configuration and seed 123; do not pretend these replacements are training samples.
- **(c) Integrated control trajectory:** plot the matching eight angle trajectories, 0–200 s, degree, with a gray dashed 45° baseline. Set limits from selected paths with 10% padding, including the baseline. Do not shade this panel as MACE UQ and do not mark 45±2.5° as a hard envelope. Preserve one-to-one velocity/angle colors and profile IDs.
- **(d) Dispersion and initialization:** calculate standard deviation versus time from $BKB^T$ and from the implemented zero-initial-velocity transformation of the regularized covariance. Plot unconditioned design SD dashed black and conditioned SD solid blue; x-axis 0–200 s, y-axis “Control-angle standard deviation (degree),” lower limit zero. Mark the design endpoint (200 s,2.5°) and annotate the actual conditioned endpoint numerically from calculation. These covariance curves are before bounded-path rejection. If the regularization affects the last decimals, state its magnitude; do not force the conditioned curve to terminate at 2.5°.

**Required inputs:** batch generation record; original time/angle/velocity tables and selected IDs; kernel/variance/initialization code at the reported commit. Record paths, IDs, seed, and covariance settings in a sidecar. Do not create extra physics simulations for this figure.

**Caption content:** velocity GP and integrated control; Matérn correlation settings; 0.1-s control grid and 45° initialization; selected batch/IDs; unconditioned design versus conditioned covariance; rejection bounds. Example opening: “Construction of stochastic control trajectories from temporally correlated interval velocities.”

### 4.2. Figure 2 — LSTM architecture and autoregression (adapt existing)

**Purpose:** retain both required neural-network figures while showing the implemented shapes and recurrence. Size: 178×110 mm, (a) architecture occupying roughly 40% width, (b) autoregression 60%. Proposed name: `fig02_lstm_autoregression.pdf`. Rebuild labels as editable text if the supplied raster becomes illegible.

**Panel (a):** exact sequence: input tensor $B_{batch}\times13\times16$ → one LSTM(64) → last sequence output $B_{batch}\times64$ → Linear(64→64) → ReLU → Linear(64→15). Label history features “15 output variables in input coordinates” and control “one prescribed control.” Avoid the old generic multilayer illustration implying an architecture sweep. Use $B_{batch}$ for batch size so it cannot be confused with the integration matrix $B$.

**Panel (b):** draw two adjacent forecast steps for one member $m$. At the first step show the 13-row output history and shifted control window entering $f_m$, then $\widehat{\widetilde{\mathbf y}}^{Y}_{m,h}$. Route this prediction through a box “Output → input coordinate map,” then append it to the member's next history after dropping the oldest row. Feed prescribed controls forward independently. Repeat the network for the next step. Above/alongside the panel, show that this process is repeated for ten members and that their completed predictions yield $\mu_h,\sigma_h$; ensemble aggregation has no arrow back into a member history.

Use solid blue arrows for forecast/feedback and dashed black arrows for prescribed controls. Label history rows with the exact indices from §2.3, or illustrate a three-row window clearly marked “schematic; implemented window = 13 rows.” No future reference-response arrow should enter the rollout.

**Caption content:** one-step training versus recursive inference, member-specific feedback, known controls, input/output coordinate conversion, final ensemble mean and population spread. Identify the corrected shapes explicitly. Retain the conceptual content of `LSTM Architecture.png` and `NeuralAutoregression.png`.

### 4.3. Figure 3 — MACE scores, quantiles, and coverage events (new)

**Purpose:** make the three calibration constructions and the meaning of “complete multivariate” unambiguous. Size: 178×145 mm. Proposed name: `fig03_mace_calibration_events.pdf`. This is a diagram, not an empirical coverage plot.

**Upper panel (a), about 35% height:** nodes “Frozen ensemble forecasts” → “Mean $\mu$, spread $\sigma$” → “Positive scale $d=\sigma+\tau$”. A separate “Calibration response $\widetilde y$” arrow joins at “Normalized absolute error $s=|\widetilde y-\mu|/d$”. Route score output down to the three lower rows. Keep node text short; no training update or evaluation-truth arrow enters calibration.

**Lower panel (b), three horizontal rows:** row labels “MACE-Horizon,” “MACE-Trajectory,” and “Joint calibration.” Each row has score reduction → order statistic → quantile shape → an event illustration. Use labels “none; retain (h,j),” “max over h,” and “max over h,j”; rank box $k=951$ of 1,000; quantile labels $q_{h,j}$, $q_j$, and $q_{all}$. At far right draw an identical 5-horizon×3-output grid for each row, with columns time and rows outputs. Highlight a single cell, one full output row, and the full grid respectively. Add “schematic dimensions” so 5×3 cannot be mistaken for the actual 501×15 grid.

Below the grid illustrations add a thin outlined column of one horizon, labeled “Horizon complete multivariate event: all outputs at one horizon; diagnostic under ordinary MACE-Horizon.” Do not imply a fourth evaluated calibration construction. Define grid highlighting as the set of coordinates required to succeed, not as observed hits/misses. Use the same neutral fill for all event sets; no false appearance that one method achieved greater coverage in this conceptual figure.

Place the interval formula $I=\mu\pm q(\sigma+\tau)$ in a shared footer. The caption states that the quantile is selected across independent calibration trajectory samples, whereas maxima are taken within each sample. Describe the exchangeability condition in text, not as an unconditional “95% guaranteed” stamp in the figure.

### 4.4. Figure 4 — microreactor plant schematic (existing)

Use `hpmr-bop-viz(1).png` as the source, approximately 178×65 mm at its existing wide aspect ratio, on white so its dark annotations remain readable. Proposed export name: `fig04_microreactor_plant.png` (vector only if editable source is available).

Preserve the heat-transfer path fuel → moderator/structure → heat pipe → nitrogen loop → active steam-generator metal → feedwater/steam. Ensure labels match $T_f,T_m,T_{hp},T_{N2},T_{sg}$; replace ambiguous `Tg`/`TN` labels if present. Identify the prescribed drum angle $\theta(t)$, fixed feedwater flow and outlet pressure, and calculated $T_{steam,out},x_{steam,out}$. Do not visually label outlet temperature as a prescribed boundary. Avoid adding unmodeled turbine/controller dynamics or implying a spatially resolved simulator.

**Caption content:** reduced coupled kinetics/thermal model, synchronized drums, five thermal nodes, prescribed secondary boundaries, and calculated outlet variables. Point to updated Appendix B for equations and parameters.

### 4.5. Figure 5 — generated transient dataset (existing, required 4×4)

Retain `timeseries_batch_0003_ALL_PROFILES.png`, approximately 178×119 mm. Proposed export name: `fig05_generated_transient_dataset.png`. Re-render from the original batch arrays only if necessary for readable labels; preserve the 4×4 subplot structure and sample population. This is a dataset illustration with simulator traces, not forecasts or uncertainty bounds.

Keep the supplied display order: row 1 $\theta,T_{N2},T_m,T_{hp}$; row 2 $T_f,T_{sg},T_{steam,out},x_{steam,out}$; row 3 $c_1,c_2,c_3,c_4$; row 4 $c_5,c_6,P,\rho_{\$}$. Use time 0–200 s, temperature K, angle degree, physical power MW, quality dimensionless, reactivity dollars, and verified precursor normalization units. State explicitly that $P=6n$ MW is displayed although $n$ is forecast/stored. The control panel is not one of the 15 output variables.

Use exact batch/sample count from the underlying generation records; the filename “ALL_PROFILES” is insufficient to establish a count or training membership. If not all illustrated samples are selected for the training split, say “generated batch” rather than “training set.” Do not add an impossible per-trace legend for thousands of paths. The caption identifies the common initial condition and distinguishes display order from native output storage order.

### 4.6. Figure 6 — study arms and control families (new)

**Purpose:** show what remains fixed, what is recalibrated, and the temporal structure of the retained operation-informed controls. Size: 178×170 mm, (a) study design across the upper 40%, (b–e) 2×2 input plots across the lower 60%. Proposed name: `fig06_calibration_distributions.pdf`.

**Panel (a):** at top place “Stochastic training (5,000) + validation (1,000)” → “Frozen ten-member ensemble + training scaler + common floor.” From the frozen block branch into two score-calibration boxes: “Original stochastic calibration (1,000)” and “Operation-informed calibration (1,000).” Each yields its own horizon, per-output trajectory, and joint trajectory quantiles. Both quantile boxes and the frozen ensemble feed the same “Operation-informed evaluation (2,000)” box. Label the common predictor-to-evaluation arrow “same forecast mean and spread”; label quantile arrows “different calibrated bounds.” Mark original-to-operational comparison “distribution-transfer diagnostic” and operation-informed arm “independent calibration/evaluation draws from one prespecified mixture.” Add a separate small stochastic-evaluation (2,000) box for the original-distribution study linked to the frozen ensemble and original calibration; flag its result provenance as required in §5, not an invented completed result.

Do not draw an arrow from evaluation responses back to calibration, training, or floor selection. No model/scaler update loop belongs here. List the three interval constructions once in a shared legend rather than repeating nine boxes.

**Panels (b–e):** B, C, D, F in that order. Plot three controls from each family selected by lowest evaluation profile IDs in the retained manifest, without selection by forecast performance. These are the same selection rule as the existing twelve forecast panels; record the exact IDs. All axes: time 0–200 s, control angle 44–49°, ticks at integer degrees, gray dashed 45° baseline. Use black solid, blue dashed, orange dotted for the three samples in each panel. Label only with family/descriptive name and compact ID legend. Keep identical axes across families. Do not include ±30° trajectories or adjust the controls to make more dramatic examples.

Draw unobtrusive short horizontal phase bars below each sample's curve plot only if the layout permits clear association with its ID; otherwise omit phase shading from this crowded figure. Describe the shared phase definition in the caption and demonstrate it with one dedicated example in the results/supplement. Do not shade one profile's move times behind all three lines as if they were common.

**Required inputs:** retained `operational_manifest.json` and its matching profile tables/recipe; original split manifest counts; frozen checkpoint/scaler identities; original-calibration corrected-forecast provenance. The figure uses input data and protocol only; it needs no new physics simulation or empirical performance values.

**Caption content:** frozen components, separate calibration distributions, common evaluation set, equal-probability family mixture, ID-based illustrative selection, and that stated coverage objectives apply to the pooled exchangeable arm rather than every family/phase.

### 4.7. Figure A1 — rolling forecast algorithm (correct then relocate)

The supplied Rolling Forecast figure is useful in Appendix A after rewriting its pseudocode. Use `figA1_recursive_forecast_algorithm.pdf`, roughly single-column width 85 mm, with readable code text. Required steps: load frozen member; initialize a 13×15 input-scaled history; combine with a 13×1 shifted prescribed-control window; predict a 15-vector in output coordinates; save it; convert this vector to input coordinates; drop oldest history row and append the mapped vector; repeat through all 501 grid points; aggregate completed member forecasts. Use consistent $K,J,M,H$ notation. Correct old `S=13` output dimensions, control slicing, full-matrix appending, and the omitted coordinate map. State that the schematic algorithm is per member and does not consume future reference outputs.

## 5. Evidence dependencies to resolve before the numerical manuscript draft

These are targeted provenance requirements for the paper, not reasons to omit the stochastic-study narrative or to retrain models automatically.

1. **One recursive forecast definition throughout.** The retained scaler-use audit reports historical direct output-to-input feedback mismatches despite exact historical replay. The corrected operational workflow maps native Y to X coordinates and recomputes original calibration forecasts/quantiles, but its metadata explicitly says historical forecasts were not replaced. Therefore, do not combine historical stochastic-evaluation accuracy/UQ or historical floor-sweep numbers with corrected operational results as though they came from one recurrence. Locate a corrected stochastic-evaluation archive, or regenerate forecasts from the same frozen models/scaler and recalibrate on the existing original partitions before locking those numbers. This requires no new training or new stochastic physics test set.
2. **Floor evidence version.** The historical fixed-forecast sweep remains a record of that historical workflow. A stability claim for the corrected primary method needs a sweep using corrected forecasts and recalculated quantiles at all seven floors. Otherwise retain the analytical floor interpretation and a limited fixed-floor evaluation, without claiming empirical corrected-method floor robustness.
3. **Original batch provenance.** Recover batch-wise generation settings, seeds, accepted counts, and partition membership. Current batch-3 settings do not establish settings for every original sample. Confirm family-independent original trajectory sampling/lineage and the source population/count in Figure 5.
4. **Actual executed plant bindings.** Bind the Appendix B parameter table to the Modelica source plus run wrapper/configuration overrides. Retain the selected old appendix's useful physics exposition, but update the fifth thermal node, calculated outlet closures, and specific executed steam-generator capacitance. Do not resurrect the outdated fixed-outlet model.
5. **Figures and numerical conventions.** Correct architecture/window/feedback labels, verify physical units and the $P$ versus $n$ distinction, and save figure source/selection sidecars. Regenerate coverage from saved scaled endpoints under one convention before quoting it. Keep target-specific physical metrics and aggregate scaled metrics separate.

Do not hide these requirements inside result captions. A single supplementary reproducibility table can identify each result group, checkpoint/scaler hashes, simulation manifest, forecast recurrence ID, quantile file, coverage convention, and evaluation code commit.

## 6. Source files anchoring the method

All paths below refer to the implementation reference commit identified at the top; current branch content can later evolve. Literature citations belong in the companion verified-source document; these files establish what the present study actually executed.

- `src/rabl/variography/DrumVariography.py`: Matérn covariance, left-endpoint integration, velocity initialization conditioning, numerical regularization, bounds rejection, and dispersion helper.
- `scripts/config.py`; `scripts/run_dymola_batch.py`: current sampling/simulation configuration and generation flow.
- `src/rabl/machine_learning/build_lstm_dataset.py`: output order, prepared steady rows, 13-row windows, and shifted control timing.
- `scripts/lstm_ensemble_training_config.json`; `src/rabl/machine_learning/lstm_pipeline.py`; `src/rabl/machine_learning/bagging_ensemble.py`: training settings, architecture, and member subsampling.
- `scripts/_mace_scaler_use.py`; `scripts/_mace_operational_forecasts.py`: native coordinate mapping and corrected member-specific recursive forecasting.
- `scripts/_mace_floor_study.py`; `scripts/run_mace_profile_validation.py`: normalized scores, horizon/per-output trajectory quantiles, joint maximum calibration, endpoints, and coverage arithmetic.
- `scripts/_mace_operational_profiles.py`; `scripts/_mace_operational_metrics.py`; `scripts/mace_operational_validation_config.json`: retained B/C/D/F recipe, phase definitions, comparison statistics, and paired whole-trajectory bootstrap.
- `src/modelica/MicroreactorPK/Models/HPMicroPK.mo`; `src/modelica/MicroreactorPK/Experiments/RunOneProfile.mo`; `src/rabl/interface/pymola.py`: plant equations, experiment bindings, and Dymola execution.

## 7. Drafting order

Write §2.1 definitions and §2.4 calibration objectives first, then §§2.2–2.3 sampling/forecasting and §2.5 current physics. Finalize study arms, metrics, and provenance before writing numerical conclusions. Create Figures 1–3 from the locked notation, adapt Figures 4–5, and build Figure 6 from retained input manifests. Then draft the introduction around the implemented contributions and supported results in the scientific style described above. No introduction statement should depend on the excluded broader-angle study.

## 8. Executable figure production

Use `scripts/generate_mace_paper_figures.py` with `scripts/mace_paper_figures_config.json`; see [the figure-generation workflow](../../scripts/mace_paper_figures_workflow.md). Stable IDs select sampling (Figure 1), dataset (Figure 5), operational control panels (Figure 6b–e), supplied graphics, and the retained operational forecast report. The runner generates scientific plots and exports original graphics unchanged; pure diagrams and their manual corrections remain figure-production tasks. In particular, Figure 6a is assembled separately from its generated data panels. New data-figure renderers can be registered as later manuscript sections are developed.
