# Corrected stochastic forecasts: matched autoregression

Use the existing ten frozen model checkpoints, both original train-fitted scalers,
and exactly the original 1,000 calibration/2,000 test trajectories. The corrected
stochastic test forecasts call the SAME \`corrected_rollout\` used for
operation-informed forecasting. Each member uses its own prior Y-scaled
prediction mapped into X-scaled coordinates at each recursive step.

This does not simulate, retrain, refit scalers, modify historical archives,
or change the original profile partitions. The corrected original stochastic
calibration is reused only if the completed operation-informed study's hashes,
identities, feedback label, runtime settings and prediction sources pass validation.

The corrected stochastic forecast runner saves the historical *reader schema*
under a new directory; the files explicitly carry a corrected feedback marker.
Control values and physical truth are reconstructed using the original scaled
arrays and float32 scaler convention expected by the fixed-study readers.
The original operation-informed HDF5 remains untouched.

## Commands (PowerShell, repository root)

\`\`\`powershell
git switch mace-floor-sensitivity
git pull --ff-only origin mace-floor-sensitivity
conda activate pymola
python -m unittest discover -s tests -p "test_mace_*.py" -q
python scripts/run_mace_stochastic_forecasts.py --config scripts/mace_stochastic_forecasts_config.json
python scripts/run_lstm_floor_sensitivity.py --config scripts/lstm_floor_sensitivity_corrected_config.json
python scripts/run_mace_profile_validation.py --config scripts/mace_profile_validation_corrected_config.json
\`\`\`

Run the three scientific commands in order. Each creates its own new \`_v1\`
output directory and refuses to overwrite an existing directory. On interruption,
retain the incomplete directory for diagnostics and choose a new destination for
a fresh attempt. Do not rerun the completed operational experiment.

The corrected floor sensitivity script verifies the saved corrected forecast hashes,
manifest IDs, targets and feedback labels and recomputes ALL absolute/MACE
quantiles for all seven floors. It deliberately does not compare against the old
historical conformal endpoints, which were produced by another autoregression.
The profile-validation stage recomputes joint/trajectory/horizon metrics and
10,000 whole-profile paired bootstrap intervals for the corrected stochastic
distribution. The original historical scripts retain their original mode.

No local experimental HDF5 was available in GitHub, so these runs require the
saved study outputs on the user's workstation. Review the final JSON metadata's
\`completed\` field and the output CSVs before treating new numbers as validated.
