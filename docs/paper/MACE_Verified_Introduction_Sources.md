# Verified literature for the MACE introduction

Verified 12 October 2026. Recommended citation pool: 21 core papers and 7 supporting papers, organized around the introduction's argument. The aim is a cohesive review, not 28 disconnected paper summaries. The supplied refs.bib is preserved; its useful references are retained here with corrected metadata.

## Verification scope

For each included paper, the identity, author attribution, year, venue or preprint status, and DOI/official identifier were checked against a primary publisher/proceedings page, author-deposited paper, or originating institutional record. Abstracts and accessible paper text were used to check the narrow claim proposed below. Some publishers block direct page access; in those cases, publisher-indexed records or author/institutional copies supplied the evidence. This is not a claim that every paper was read exhaustively or that all its empirical conclusions were independently reproduced. No numerical accuracy or speedup claim from the old draft is carried over automatically.

Preprints are labeled explicitly. A repository deposit date is not substituted for a conference/publication year. Books, blogs and Wikipedia are outside this papers-only list. Suggested citation keys are new-manuscript keys; nothing has been silently overwritten in the uploaded bibliography.

## Plant-model and nomenclature decisions

The current Modelica equations, experiment wrapper and Dymola runner in LoganAlexanderBurnett/RABL are the plant-model authority. Read at commit `5b0f9f6a2702ab4b1b88a723d40cdba293d3c02a` on `mace-floor-sensitivity`:

- [Models/HPMicroPK.mo](https://github.com/LoganAlexanderBurnett/RABL/blob/5b0f9f6a2702ab4b1b88a723d40cdba293d3c02a/src/modelica/MicroreactorPK/Models/HPMicroPK.mo)
- [Experiments/RunOneProfile.mo](https://github.com/LoganAlexanderBurnett/RABL/blob/5b0f9f6a2702ab4b1b88a723d40cdba293d3c02a/src/modelica/MicroreactorPK/Experiments/RunOneProfile.mo)
- [Dymola runner](https://github.com/LoganAlexanderBurnett/RABL/blob/5b0f9f6a2702ab4b1b88a723d40cdba293d3c02a/src/rabl/interface/pymola.py)

The current plant adds a dynamic steam-generator metal node to the reactor thermal network and calculates outlet temperature and quality from heat supplied per unit feedwater mass. Outlet pressure and feedwater flow are prescribed; outlet temperature and quality are not fixed. The experiment wrapper passes parameter values to the plant and overrides some defaults, including steam-generator metal capacitance. Package overview comments and the old appendix do not supersede the actual equations and experiment bindings. These repository files are implementation evidence, not substitutes for literature citations.

Accepted manuscript terminology: control trajectory; output variables; multivariate response trajectory; trajectory sample; forecast trajectory; calibration distribution; evaluation distribution; MACE-Horizon; MACE-Trajectory; Horizon and Trajectory complete multivariate coverage. Coverage events must be defined separately: complete multivariate coverage at a horizon requires every output to be covered at that horizon; complete multivariate trajectory coverage requires every output at every evaluated horizon. Per-output calibration alone does not automatically guarantee either complete multivariate event.

## Recommended literature-review progression

1. Physical surrogates require carefully designed simulation data, especially temporally structured control trajectories.
2. Sequence models can learn useful response dynamics, but recursive forecasting needs evaluation beyond one-step fit.
3. Ensembles supply forecast-dependent disagreement; accurate means or useful disagreement do not establish calibrated bounds.
4. Conformal prediction supplies a calibration framework, with prior work on normalized scores, ensembles and multivariate/multi-step forecasting.
5. Calibration assumptions become consequential when the evaluation distribution changes. Operation-informed nuclear control examples motivate the application; they do not make this paper a control-policy study.

The sampling construction, accurate autoregressive model, MACE formulation and recalibration study should all contribute to the narrative. Operational transients are the final application test, not the whole story. Steam quality is a brief example of behavior with very-low-variance output variables, not a separate literature-review theme.

## A. Surrogate modeling and control-trajectory sampling

### S01. Porcu et al. (2024) — Matérn covariance foundation [new; core]

**Reference:** Emilio Porcu, Moreno Bevilacqua, Robert Schaback and Chris J. Oates. *The Matérn Model: A Journey Through Statistics, Numerical Analysis and Machine Learning*. Statistical Science **39**(3), 469–492 (2024).

**DOI:** [10.1214/24-STS923](https://doi.org/10.1214/24-STS923).

**Verification:** Publisher issue/PDF at [Institute of Mathematical Statistics](https://www.imstat.org/publications/sts/sts_39_3/sts_39_3.pdf); originating-author institutional records agree on the article metadata. The verified DOI is **24-STS923**.

**Use:** Explain why a Matérn covariance is a reasonable way to specify correlation and regularity in sampled control velocity. This supports the kernel foundation, not novelty of our integrated-control sampling construction or guaranteed hard angle/rate bounds.

**Suggested key:** `porcu2024matern`.

### S02. Pestourie et al. (2020) — physical-surrogate data acquisition [retained; core]

**Reference:** Raphaël Pestourie, Youssef Mroueh, Thanh V. Nguyen, Payel Das and Steven G. Johnson. *Active learning of deep surrogates for PDEs: application to metasurface design*. npj Computational Materials **6**, 164 (2020).

**DOI / primary record:** [10.1038/s41524-020-00431-2](https://www.nature.com/articles/s41524-020-00431-2).

**Use:** Establish that choosing simulation data deliberately matters for learned physical surrogates. Its active-learning method and metasurface application differ from our prespecified control-trajectory sampling. Avoid importing its speedup or simulation-budget results into claims about this study.

**Correction:** The supplied bibliography names the wrong journal and gives an incomplete/wrong author entry. Suggested key: `pestourie2020surrogates`.

### S03. Kapadia, Feng and Benner (2024) — nonlinear surrogate/data-design context [retained; core]

**Reference:** Harshit Kapadia, Lihong Feng and Peter Benner. *Active-learning-driven surrogate modeling for efficient simulation of parametric nonlinear systems*. Computer Methods in Applied Mechanics and Engineering **419**, 116657 (2024).

**DOI:** [10.1016/j.cma.2023.116657](https://doi.org/10.1016/j.cma.2023.116657). [Originating institute paper](https://pure.mpg.de/rest/items/item_3514920_9/component/file_3558958/content).

**Use:** Provide an example of adapting simulation data to build nonlinear surrogates when conventional residual-based selection is unavailable. It selects parametric snapshots/POD subspaces; it is not the same control-trajectory construction or uncertainty method as MACE.

**Correction:** Harshit/Lihong replace the supplied Rohan/Nai; the journal year is 2024 even though the DOI includes 2023. Suggested key: `kapadia2024surrogate`.

### S04. Zimmer, Meister and Nguyen-Tuong (2018) — temporally structured input design [retained; core]

**Reference:** Christoph Zimmer, Mona Meister and Duy Nguyen-Tuong. *Safe Active Learning for Time-Series Modeling with Gaussian Processes*. Advances in Neural Information Processing Systems **31** (NeurIPS 2018).

**Primary record:** [Official proceedings](https://papers.neurips.cc/paper_files/paper/2018/hash/b197ffdef2ddc3308584dce7afa3661b-Abstract.html).

**Use:** Motivate control trajectories as the objects of experiment design, preserving temporal structure and accounting for physical constraints. Their sequential safe exploration is context; our study does not demonstrate their safety guarantees or an equivalent active-learning algorithm.

**Correction:** The supplied 2024 year, Nils first name and placeholder arXiv identifier are incorrect. Suggested key: `zimmer2018safe`.

### S05. Lundby et al. (2023) — trajectory excitation for nonlinear identification [retained; core; preprint]

**Reference:** Erlend Torje Berg Lundby, Adil Rasheed, Ivar Johan Halvorsen, Dirk Reinhardt, Sebastien Gros and Jan Tommy Gravdahl. *Deep active learning for nonlinear system identification*. arXiv:**2302.12667** (2023).

**Primary record:** [Author-deposited preprint](https://arxiv.org/abs/2302.12667); [SINTEF record](https://www.sintef.no/en/publications/publication/0198cc62ea23-331f1492-979b-4d51-b276-7ff0f45935eb/).

**Use:** Explain that learning dynamical systems depends on excitation over state-control trajectories. Their local optimal-control exploration and ensemble-based acquisition provide a useful comparison to prespecified stochastic sampling. This source is a verified preprint, not a verified journal publication.

**Correction:** Replace the supplied first name and arXiv placeholder. Suggested key: `lundby2023identification`.

### S06. Xie and Bemporad (2025) — control excitation as experiment design [retained; core; preprint]

**Reference:** Kui Xie and Alberto Bemporad. *Online design of experiments by active learning for nonlinear system identification*. arXiv:**2506.21754** (2025).

**Primary record:** [Author-deposited preprint](https://arxiv.org/abs/2506.21754).

**Use:** Connect input/control-trajectory selection to nonlinear system identification and constraints. Distinguish their online excitation design from our sampled trajectories. Do not confuse this preprint with their differently titled CDC 2024 autoregressive-model paper.

**Correction:** Kui, not Xia; the supplied arXiv identifier is a placeholder. Suggested key: `xie2025design`.

## B. Deep learning for physical time series

### S07. Hochreiter and Schmidhuber (1997) — LSTM foundation [new; core]

**Reference:** Sepp Hochreiter and Jürgen Schmidhuber. *Long Short-Term Memory*. Neural Computation **9**(8), 1735–1780 (1997).

**DOI / primary record:** [10.1162/neco.1997.9.8.1735](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory).

**Use:** Give the architectural foundation for learning temporal dependencies. It does not establish the accuracy, recursive stability, or uncertainty calibration of this particular model.

**Suggested key:** `hochreiter1997lstm`.

### S08. Radaideh et al. (2020) — nuclear transient forecasting [retained; core]

**Reference:** Majdi I. Radaideh, Connor Pigg, Tomasz Kozlowski, Yujia Deng and Annie Qu. *Neural-based time series forecasting of loss of coolant accidents in nuclear power plants*. Expert Systems with Applications **160**, 113699 (2020).

**DOI:** [10.1016/j.eswa.2020.113699](https://doi.org/10.1016/j.eswa.2020.113699). [Publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0957417420305236).

**Use:** Establish nuclear-physics transient prediction as an existing application of neural sequence models. Their accident setting and forecasting design differ from our prescribed-control, recursively deployed ensemble. We can cite the application without reproducing their reported accuracy percentages or speedup as if they measured the same task.

**Correction:** Replace the supplied author list. The publisher uses “Annie Qu”; some institutional records display a different given-name form, so use the publisher's byline consistently. Suggested key: `radaideh2020loca`.

### S09. Appiah et al. (2024) — LSTM physical system identification [retained; core]

**Reference:** Rita Appiah, Alexander Heifetz, Derek Kultgen, Lefteri H. Tsoukalas and Richard B. Vilim. *Dynamic Control of Sodium Cold Trap Purification Temperature Using LSTM System Identification*. Energies **17**(24), 6257 (2024).

**DOI / primary record:** [10.3390/en17246257](https://www.mdpi.com/1996-1073/17/24/6257). [Author laboratory publication listing](https://engineering.purdue.edu/AISL/publications).

**Use:** Provide a physical/nuclear example connecting LSTM prediction, system identification and control. Their LSTM-MPC setting is relevant context; it does not show that our model has been evaluated in closed-loop control.

**Correction:** The supplied authors, issue and article number are incorrect. Suggested key: `appiah2024coldtrap`.

## C. Ensemble uncertainty and behavior under distribution change

### S10. Lakshminarayanan, Pritzel and Blundell (2017) — deep ensembles [retained; core]

**Reference:** Balaji Lakshminarayanan, Alexander Pritzel and Charles Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. Advances in Neural Information Processing Systems **30** (NeurIPS 2017).

**Primary record:** [Official proceedings](https://papers.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html).

**Use:** Establish ensembles as a practical predictive-uncertainty approach. Distinguish their probabilistic predictive models from the spread of our deterministic point-prediction networks. Ensemble spread is an adaptation scale in MACE, not an automatic Gaussian 95% interval or an identified decomposition of all physical uncertainty.

**Suggested key:** `lakshminarayanan2017deep`.

### S11. Ovadia et al. (2019) — uncertainty does not automatically transfer [new; core]

**Reference:** Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan and Jasper Snoek. *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*. Advances in Neural Information Processing Systems **32** (NeurIPS 2019).

**Primary records:** [Official proceedings](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html); [author-deposited paper](https://arxiv.org/abs/1906.02530).

**Use:** Motivate evaluation of uncertainty when the evaluation distribution changes. Its main benchmarks are classification, so it supports the broad concern rather than proving anything about reactor regression or conformal trajectory coverage.

**Suggested key:** `ovadia2019shift`.

## D. Conformal prediction, adaptive intervals and temporal/multivariate coverage

### S12. Angelopoulos and Bates (2023) — calibration foundation [new; core]

**Reference:** Anastasios N. Angelopoulos and Stephen Bates. *Conformal Prediction: A Gentle Introduction*. Foundations and Trends in Machine Learning **16**(4), 494–591 (2023).

**DOI / primary publisher citation:** [10.1561/2200000101](https://doi.org/10.1561/2200000101); [publisher summary/PDF](https://www.nowpublishers.com/article/DownloadSummary/MAL-101). A related author preprint is [arXiv:2107.07511](https://arxiv.org/abs/2107.07511), under a longer title; use the chosen publication version consistently.

**Use:** Introduce conformal calibration around a pre-trained predictor, finite-sample marginal coverage under appropriate assumptions, and conformalization of scalar uncertainty estimates. State exchangeability explicitly; distribution-free does not mean valid under every distribution shift.

**Suggested key:** `angelopoulos2023conformal`.

### S13. Lei et al. (2018) — predictive regression inference [new; core]

**Reference:** Jing Lei, Max G'Sell, Alessandro Rinaldo, Ryan J. Tibshirani and Larry Wasserman. *Distribution-Free Predictive Inference for Regression*. Journal of the American Statistical Association **113**(523), 1094–1111 (2018).

**DOI:** [10.1080/01621459.2017.1307116](https://doi.org/10.1080/01621459.2017.1307116). [Author-hosted paper](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf).

**Use:** Give a primary statistical foundation for split conformal regression and locally varying interval lengths. Important support for separating prediction-model training from interval calibration.

**Suggested key:** `lei2018regression`.

### S14. Papadopoulos, Gammerman and Vovk (2008) — normalized-score precedent [new; core]

**Reference:** Harris Papadopoulos, Alex Gammerman and Vladimir Vovk. *Normalized Nonconformity Measures for Regression Conformal Prediction*. Proceedings of the IASTED International Conference on Artificial Intelligence and Applications, 64–69 (2008).

**Primary records:** [Publisher abstract](https://actapress.com/Abstract.aspx?paperId=32280); [originating institution metadata](https://pure.royalholloway.ac.uk/en/publications/normalized-nonconformity-measures-for-regression-conformal-predic/).

**Use:** Cite the established idea of normalizing regression nonconformity according to expected prediction difficulty to obtain variable-width intervals. Their underlying predictor is nearest-neighbor regression, not our autoregressive ensemble. No DOI has been added without verification.

**Suggested key:** `papadopoulos2008normalized`.

### S15. Romano, Patterson and Candès (2019) — adaptive conformal intervals [new; core]

**Reference:** Yaniv Romano, Evan Patterson and Emmanuel J. Candès. *Conformalized Quantile Regression*. Advances in Neural Information Processing Systems **32** (NeurIPS 2019).

**Primary record:** [Official proceedings](https://papers.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).

**Use:** Establish that adaptation to local variability and conformal coverage are compatible goals. It learns conditional quantiles; MACE instead uses a frozen ensemble mean and disagreement scale. Do not call MACE a quantile-regression method.

**Suggested key:** `romano2019cqr`.

### S16. Stankevičiūtė, Alaa and van der Schaar (2021) — conformal sequence forecasting [new; core]

**Reference:** Kamilė Stankevičiūtė, Ahmed M. Alaa and Mihaela van der Schaar. *Conformal Time-series Forecasting*. Advances in Neural Information Processing Systems **34** (NeurIPS 2021).

**Primary record:** [Official proceedings](https://papers.neurips.cc/paper_files/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html).

**Use:** Position MACE within existing multi-horizon conformal forecasting. Compare calibration units, score construction and coverage events precisely; do not claim that adding conformal bounds to sequence models is itself new.

**Suggested key:** `stankeviciute2021forecasting`.

### S17. Xu and Xie (2021) — ensembles plus conformal inference [new; core]

**Reference:** Chen Xu and Yao Xie. *Conformal prediction interval for dynamic time-series*. Proceedings of the 38th International Conference on Machine Learning, PMLR **139**, 11559–11569 (2021).

**Primary record:** [Official ICML/PMLR proceedings](https://proceedings.mlr.press/v139/xu21h.html).

**Use:** Identify EnbPI as prior work combining bootstrap ensembles with conformal-related sequential intervals. Its nonexchangeable time-series setting and approximate coverage results under dependence assumptions differ from split calibration over exchangeable whole trajectory samples. This is an important comparison when describing how MACE combines ensemble and conformal strengths.

**Version note:** The authors identify a journal extension, *Conformal Prediction for Time Series*, IEEE TPAMI **45**(10), 11575–11587 (2023), DOI [10.1109/TPAMI.2023.3272339](https://doi.org/10.1109/TPAMI.2023.3272339), on their [publication page](https://www2.isye.gatech.edu/~yxie77/Pub_timeseries.html) and [official code repository](https://github.com/hamrel-cxu/EnbPI). Use the conference version above for the directly verified proceedings claim, or use the journal version consistently after consulting that text; do not treat them as independent methods.

**Suggested key:** `xu2021enbpi`.

### S18. Sun and Yu (2024) — multivariate, multi-step coverage [new; core]

**Reference:** Sophia Sun and Rose Yu. *Copula Conformal Prediction for Multi-step Time Series Forecasting*. International Conference on Learning Representations (ICLR 2024).

**Primary records:** [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html); [conference paper](https://openreview.net/pdf/76af3e4f5f8bb8b344fcb41b96512669856618c4.pdf).

**Use:** Discuss dependence-aware conformal regions for multivariate, multi-step forecasts. Important context for Horizon and Trajectory complete multivariate coverage. Its copula method is not our maximum-score construction.

**Version/title note:** The paper PDF uses “Forecasting”; the proceedings landing page displays “Copula Conformal prediction for multi-step time series prediction.” These identify the same paper. The original preprint is arXiv:2212.03281 (2022); the verified conference is ICLR 2024, not ICML 2023.

**Suggested key:** `sun2024copulacpts`.

### S19. Barber et al. (2023) — exchangeability and distribution change [new; core]

**Reference:** Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas and Ryan J. Tibshirani. *Conformal prediction beyond exchangeability*. The Annals of Statistics **51**(2), 816–845 (2023).

**DOI / primary paper:** [10.1214/23-AOS2276](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.pdf).

**Use:** Explain why changes between the calibration distribution and evaluation distribution require care, and identify existing approaches beyond ordinary exchangeability. MACE's studied batch recalibration is not their weighted/randomized robustness procedure; their guarantees must not be attributed to our unchanged calibration arm.

**Suggested key:** `barber2023exchangeability`.

## E. Artificial intelligence for control and the nuclear application

### S20. Price, Radaideh and Kochunas (2022) — microreactor control context [retained; core]

**Reference:** Dean Price, Majdi I. Radaideh and Brendan Kochunas. *Multiobjective optimization of nuclear microreactor reactivity control system operation with swarm and evolutionary algorithms*. Nuclear Engineering and Design **393**, 111776 (2022).

**DOI:** [10.1016/j.nucengdes.2022.111776](https://doi.org/10.1016/j.nucengdes.2022.111776). [Publisher record](https://www.sciencedirect.com/science/article/pii/S0029549322001303); [originating research-group listing](https://nuram.engin.umich.edu/publications/journal-publications/).

**Use:** Establish practical interest in drum control and neural surrogates for reactor optimization. It optimizes configurations/reactivity/power distribution; it does not validate our dynamic plant model, MACE bounds, or autonomous control.

**Suggested key:** `price2022microreactor`.

### S21. Tunkle et al. (2025) — transient drum-control context [retained; core]

**Reference:** Leo Tunkle, Kamal Abdulraheem, Linyu Lin and Majdi I. Radaideh. *Nuclear microreactor transient and load-following control with deep reinforcement learning*. Energy Conversion and Management: X **27**, 101090 (2025).

**DOI:** [10.1016/j.ecmx.2025.101090](https://doi.org/10.1016/j.ecmx.2025.101090). [Originating laboratory record](https://inl.elsevierpure.com/en/publications/nuclear-microreactor-transient-and-load-following-control-with-de/).

**Use:** Motivate control-driven microreactor transients and the need to assess predictions in operation-informed settings. It develops controllers, whereas our surrogate forecasts responses to prescribed control trajectories. Its reactor/time-scale assumptions are not automatically our plant assumptions.

**Correction:** Fill the actual author list, volume and DOI. Suggested key: `tunkle2025microreactor`.

## Supporting papers: use selectively rather than adding seven more summaries

### S22. Vandenberg-Rodes and Shahbaba (2015) [retained; supporting; preprint]

**Reference:** Alexander Vandenberg-Rodes and Babak Shahbaba. *Dependent Matérn Processes for Multivariate Time Series*. arXiv:**1502.03466** (2015). [Author-deposited source](https://arxiv.org/abs/1502.03466).

**Use:** Additional Matérn-process/time-series context if needed. Porcu et al. provide a broader published kernel foundation for the main review. This paper does not establish novelty or bounds of our specific control generator.

**Correction:** Replace Arash/2020/arXiv:2007.00000. Suggested key: `vandenberg2015matern`.

### S23. Kim et al. (2025) [retained; supporting]

**Reference:** Yegon Kim, Hyunsu Kim, Gyeonghoon Ko and Juho Lee. *Active Learning with Selective Time-Step Acquisition for PDEs*. ICML, PMLR **267**, 30199–30223 (2025). [Official proceedings](https://proceedings.mlr.press/v267/kim25m.html).

**Use:** A short contrast between acquisition of entire trajectories and selective time-step acquisition. Do not imply this study adopts partial-trajectory solver acquisition.

**Correction:** Replace the supplied first author and incomplete venue. Suggested key: `kim2025acquisition`.

### S24. Shahid et al. (2024 meeting; manuscript deposited 2023) [retained; supporting]

**Reference:** Muhammad Bilal Shahid, Robert Robison, Tom Shafer, Victor Diloreto, Natalia M. Alexandrov and Cody Fleming. *Uncertainty Quantification using Deep Ensembles for Decision Making in Cyber-Physical-Human Systems*. Author/institutional conference manuscript for the **2024 AIAA SciTech Forum**, NASA NTRS document **20230017659**, deposited December 2023. [NASA record](https://ntrs.nasa.gov/citations/20230017659); [deposited manuscript](https://ntrs.nasa.gov/api/citations/20230017659/downloads/Unc_Quan_NASA_Final_revised.pdf?attachment=true).

**Use:** Additional evidence for ensemble transition models and multistep uncertainty propagation in physical dynamical systems. The manuscript uses probabilistic ensemble machinery; it does not give automatic coverage to deterministic ensemble spread.

**Verification limit:** This reference intentionally identifies the NASA-deposited manuscript and meeting, whose metadata were verified. Do not silently attach a differently titled AIAA publisher record or unverified publisher DOI. The supplied placeholder arXiv reference is removed. Suggested key: `shahid2024ensembles`.

### S25. Kherad et al. (2025) [retained; supporting]

**Reference:** Mahdi Kherad, Mohammad Kazem Moayyedi, Faranak Fotouhi-Ghazvini, Maryam Vahabi and Hossein Fotouhi. *Sensor-Driven Surrogate Modeling and Control of Nonlinear Dynamical Systems Using FAE-CAE-LSTM and Deep Reinforcement Learning*. Sensors **25**(16), 5149 (2025). [Publisher record / DOI 10.3390/s25165149](https://www.mdpi.com/1424-8220/25/16/5149).

**Use:** Broader example of a learned dynamical surrogate used inside a control framework. Useful to distinguish surrogate forecasting from policy learning, but optional if the nuclear examples already establish the motivation.

**Suggested key:** `kherad2025control`.

### S26. Zolman et al. (2025) [retained; supporting]

**Reference:** Nicholas Zolman, Christian Lagemann, Urban Fasel, J. Nathan Kutz and Steven L. Brunton. *SINDy-RL for interpretable and efficient model-based reinforcement learning*. Nature Communications **16**, 10714 (2025). [Publisher record / DOI 10.1038/s41467-025-65738-4](https://www.nature.com/articles/s41467-025-65738-4).

**Use:** Optional control-literature contrast involving interpretable learned dynamics, rewards and policies. MACE neither learns a policy nor uses a SINDy surrogate.

**Correction:** Use the final 2025 publication instead of the supplied placeholder arXiv entry/title. Suggested key: `zolman2025sindyrl`.

### S27. Tibshirani et al. (2019) [new; supporting]

**Reference:** Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel J. Candès and Aaditya Ramdas. *Conformal Prediction Under Covariate Shift*. Advances in Neural Information Processing Systems **32** (NeurIPS 2019). [Official proceedings](https://papers.nips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html).

**Use:** Identify weighted conformal inference as one alternative for particular forms of shift. Its covariate-shift/likelihood-ratio conditions are more specific than arbitrary environmental change. Our batch recalibration does not implement this weighting method.

**Suggested key:** `tibshirani2019shift`.

### S28. Xu and Xie (2023) [new; supporting]

**Reference:** Chen Xu and Yao Xie. *Sequential Predictive Conformal Inference for Time Series*. ICML, PMLR **202**, 38707–38727 (2023). [Official proceedings](https://proceedings.mlr.press/v202/xu23r.html).

**Use:** Explain that online sequential residual/conditional-quantile adaptation is a distinct meaning of adaptive uncertainty. Our studied recalibration uses a separate calibration dataset; it is not SPCI or evidence of its conditional-coverage properties.

**Suggested key:** `xu2023spci`.

## Citation placement and emphasis

| Literature-review topic | Recommended papers | Transition into this paper |
|---|---|---|
| Surrogate modeling and data generation | S02–S03 | Physical simulation data are useful but costly; their design matters. |
| Data generation strategies / control trajectories | S01, S04–S06; S22–S23 if needed | We sample temporally structured control trajectories with specified covariance/dispersion instead of performing active acquisition. |
| Deep learning for physical time series | S07–S09 | Train a control-conditioned surrogate and test genuine autoregressive forecasts. |
| Ensemble methods and uncertainty | S10–S11; S24 if needed | Forecast-dependent disagreement provides an adaptation scale, but needs calibration. |
| Conformal uncertainty and coverage objectives | S12–S19; S27–S28 if needed | State normalization, calibration unit, horizon/trajectory events and distribution assumptions explicitly. |
| AI for control / microreactor application | S20–S21; S25–S26 if needed | Accurate, calibrated forecasts are potentially useful inputs to control, while this study does not learn or validate a controller. |

The introduction should synthesize themes and contrasts rather than give every source equal paragraph space. For example, one sentence can cite Zimmer/Lundby/Xie as evidence that control-trajectory design is an established identification problem, followed by a clear explanation of the particular construction evaluated here. The conformal paragraph should give especially clear attention to normalized scores, EnbPI and multivariate/multi-step forecasting because they are close methodological precedents.

## Corrections and exclusions from the supplied bibliography

| Supplied key | Action |
|---|---|
| `radaideh2020loca` | Retain paper; replace incorrect author list with publisher byline. |
| `appiah2024coldtrap` | Retain paper; correct authors, 17(24), article 6257 and DOI. |
| `lakshminarayanan2017deep` | Retain; rewrite the old +/-2sigma coverage assertion. |
| `vandenberg2020matern` | Retain optionally as the verified 2015 preprint; repair author/year/identifier. |
| `Zimmer2024SafeActiveLearning` | Retain as Zimmer/Meister/Nguyen-Tuong, NeurIPS 2018. |
| `Lundby2023DeepActiveLearning` | Retain verified arXiv:2302.12667; correct authors. |
| `Xie2025OnlineDesign` | Retain verified Kui Xie / Bemporad arXiv:2506.21754. |
| `Kapadia2023ActLearnPODKSNN` | Retain journal publication in 2024; correct authors. |
| `Pestourie2020DeepSurrogates` | Retain in npj Computational Materials, not Nature Machine Intelligence; correct author list. |
| `Kim2025SelectiveTimeSteps` | Supporting only; correct first author and PMLR metadata. |
| `Shahid2023DeepEnsembles` | Supporting only; replace placeholder arXiv with the identified NASA manuscript/2024 meeting. |
| `Tunkle2025MicroreactorRL` | Retain; complete author/volume/DOI metadata. |
| `Price2022MultiobjectiveMicroreactor` | Retain; add verified DOI. |
| `Zolman2025SINDyRL` | Supporting only; use the final Nature Communications publication. |
| `Kherad2025SurrogateControl` | Supporting only; complete authors, issue/article and DOI. |
| `banerjee2015spatial` | Not included in this papers-only list. The supplied entry needs a separate book/chapter correction if retained in methodology. |
| `wikipediaBagging`, `Teo2017ActiveLearning` | Exclude from the research-paper list. Describe the actual training resampling scheme directly. |
| `Cacciarelli2023ActiveLearningDataStreams`, `Holtz2025ActiveLearningAnomaly` | Not selected for the present narrative; do not preserve their placeholder/incomplete metadata merely because they appeared in the old file. Their full identities were not audited in this selected-paper pass. |

## Writing boundaries established by the verified literature

- Matérn covariance and GP sampling are established foundations. Define precisely what is distinctive about the variance-controlled, integrated control-trajectory procedure rather than asserting that Gaussian trajectory sampling itself is new.
- Ensembles plus conformal prediction and calibrated multi-horizon/multivariate forecasting already exist. MACE must be distinguished by its actual scale, score reductions, control-conditioned autoregressive deployment, calibration/recalibration design and demonstrated performance.
- A predictor may be frozen while calibration changes, but ordinary split-conformal guarantees still require the appropriate exchangeability of calibration and evaluation trajectory samples. Published shift corrections are not automatically part of our implementation.
- Mean accuracy, per-output coverage, complete multivariate coverage, adaptive width and control performance are different claims; one source or metric cannot establish all of them.
- Existing closed-loop control papers supply application motivation. They do not show that MACE has been evaluated as a controller or that its intervals represent every source of real-plant uncertainty.

No steam-quality-specific review is needed. Keep any very-low-variance example proportional to its role in explaining a general normalization limitation.

## Methods reference outside the introduction pool

For the interval-score definition, use Tilmann Gneiting and Adrian E. Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation*, Journal of the American Statistical Association **102**(477), 359–378 (2007), DOI [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437), with the [author-hosted paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). It belongs primarily in evaluation methodology; a point-average interval score for trajectory bands should not be described as a proper joint-trajectory scoring rule.
