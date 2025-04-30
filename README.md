# Polaris ASAP competition baselines

This repository defines
two simple baselines for small molecule machine learning and applies
them to datasets from the [Polaris antiviral competition](https://polarishub.io/competitions/asap-discovery/antiviral-drug-discovery-2025).

All content in `evaluation/` is copied directly from the
[official repository](https://github.com/asapdiscovery/asap-polaris-blind-challenge-examples).

## Setup and running the baselines

```bash
# create exact environment used for blog post
mamba create -f env.lock.yaml
# or just install basic dependencies
# mamba create -f env.yaml

mamba activate ib_polaris_baselines

# ensure you're logged in to polaris to pull the data
# polaris login --overwrite

python train_admet_baselines.py
python train_potency_baselines.py
```

## Baseline details

Two baselines are included, one based on Morgan fingerprints
and one based on RDKit descriptors.

### Fingerprint baseline

This baseline uses Morgan Fingerprints, setting a radius
of 3 and using counts.
The fingerprints are passed to a support vector regressor (SVR).
SVR is chosen for its effectiveness of capturing the similarity
structure among molecular fingerprints via the RBF kernel. We
tuned the C parameter automatically via grid search on the
random-split training set, and left the other parameters at their
default values.

### Descriptor baseline

This baseline uses the full set of available RDKit descriptors.
The descriptors are passed to a random forest regressor, chosen
for its ability to handle input variables with vastly differing
distributions. The random forest regressor's hyperparameters were
left to their default values except for the number of trees, which
was increased to 500 from the default of 100 (larger forests
are generally somewhat more accurate at the cost of speed).

## Expected baseline output

### expected ADMET baseline output

```
Training fingerprint_baseline...
 - Training HLM...
 - Training KSOL...
 - Training LogD...
 - Training MDR1-MDCKII...
 - Training MLM...
Evaluating fingerprint_baseline...
Summary metrics for fingerprint_baseline:
   Target Label               Metric      mean       std
30   aggregated          kendall_tau  0.415026  0.025227
31   aggregated  mean_absolute_error  0.361006  0.011894
32   aggregated   mean_squared_error  0.218776  0.013597
33   aggregated             pearsonr  0.643275  0.030673
34   aggregated                   r2  0.330518  0.061861
35   aggregated            spearmanr  0.564284  0.031893
Training descriptor_baseline...
 - Training HLM...
 - Training KSOL...
 - Training LogD...
 - Training MDR1-MDCKII...
 - Training MLM...
Evaluating descriptor_baseline...
Summary metrics for descriptor_baseline:
   Target Label               Metric      mean       std
30   aggregated          kendall_tau  0.403504  0.024961
31   aggregated  mean_absolute_error  0.344717  0.012596
32   aggregated   mean_squared_error  0.215298  0.015424
33   aggregated             pearsonr  0.632719  0.032773
34   aggregated                   r2  0.341459  0.048763
35   aggregated            spearmanr  0.550684  0.031241
```

### expected potency baseline output

```
Training fingerprint_baseline...
 - Training pIC50 (MERS-CoV Mpro)...
 - Training pIC50 (SARS-CoV-2 Mpro)...
Evaluating fingerprint_baseline...
Summary metrics for fingerprint_baseline:
  Target Label               Metric      mean       std
0   aggregated          kendall_tau  0.633199  0.016574
1   aggregated  mean_absolute_error  0.517889  0.021382
2   aggregated   mean_squared_error  0.512912  0.061926
3   aggregated             pearsonr  0.806679  0.027728
4   aggregated                   r2  0.606813  0.042371
5   aggregated            spearmanr  0.817025  0.018797
Training descriptor_baseline...
 - Training pIC50 (MERS-CoV Mpro)...
 - Training pIC50 (SARS-CoV-2 Mpro)...
Evaluating descriptor_baseline...
Summary metrics for descriptor_baseline:
  Target Label               Metric      mean       std
0   aggregated          kendall_tau  0.584956  0.018500
1   aggregated  mean_absolute_error  0.576950  0.022463
2   aggregated   mean_squared_error  0.600975  0.062375
3   aggregated             pearsonr  0.765832  0.028549
4   aggregated                   r2  0.539815  0.041359
5   aggregated            spearmanr  0.767204  0.022099
```
