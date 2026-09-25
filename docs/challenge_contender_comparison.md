# ACDC challenge contender comparison

This note benchmarks this repository against the *other* segmentation papers that
entered the MICCAI 2017 Automated Cardiac Diagnosis Challenge (ACDC). It is not a
general survey of papers that later used the ACDC dataset.

## Source and scope

The challenge organizers report that 10 teams submitted meaningful segmentation
results during the contest. Their challenge paper describes the architectures and
publishes one uniform evaluation of all 10 submissions on the hidden test set;
this is the source for the table below. [Bernard et al., 2018, Table II and
Table III](https://www.creatis.insa-lyon.fr/Challenge/acdc/files/tmi_2018_bernard.pdf).
The official leaderboard remains available and says its final update was November
2022: [ACDC test-results leaderboard](https://www.creatis.insa-lyon.fr/Challenge/acdc/results.html).

The contest test set contains 50 patients. Each entry below has Dice and
Hausdorff distance (HD, mm) at end diastole (ED) and end systole (ES), for LV,
RV and myocardium (Myo). `Macro Dice` is calculated here as the unweighted mean
of the six displayed Dice values; it was **not** a contest ranking metric.

## Results of the other challenge papers

| Paper / challenge method | Concise method description | LV ED / ES Dice | RV ED / ES Dice | Myo ED / ES Dice | Macro Dice (derived) | Mean HD across six cells (derived, mm) |
|---|---|---:|---:|---:|---:|---:|
| Isensee et al., *Automatic Cardiac Disease Assessment on Cine-MRI via Time-Series Segmentation and Domain Specific Features* | Ensemble of 2D and anisotropic 3D U-Nets; Dice loss | .968 / .931 | .946 / .899 | .902 / .919 | **.928** | 9.0 |
| Khened et al., *Densely Connected Fully Convolutional Network for Short-Axis Cardiac Cine MR Image Segmentation and Heart Diagnosis Using Random Forest* | Dense 2D U-Net with inception first layer; combined Dice + CE loss | .964 / .917 | .935 / .879 | .889 / .898 | .914 | 11.2 |
| Jang et al., *Automatic Segmentation of LV and RV in Cardiac MRI* | 2D M-Net; weighted cross-entropy | .959 / .921 | .929 / .885 | .875 / .895 | .911 | 9.7 |
| Zotti et al., *GridNet with Automatic Shape Prior Registration for Automatic MRI Cardiac Segmentation* | 2D GridNet/U-Net extension with automatically registered shape prior | .957 / .905 | .941 / .882 | .884 / .896 | .911 | 9.6 |
| Wolterink et al., *Automatic Segmentation and Disease Classification Using Cardiac Cine MR Images* | 2D dilated CNN jointly processing corresponding ED and ES slices | .961 / .918 | .928 / .872 | .875 / .894 | .908 | 10.7 |
| Patravali, Jain & Chilamkurthy, *2D-3D Fully Convolutional Neural Networks for Cardiac MR Segmentation* | Best submission: 2D U-Net with Dice loss | .955 / .885 | .911 / .819 | .882 / .897 | .892 | 12.1 |
| Rohé, Sermesant & Pennec, *Automatic Multi-Atlas Segmentation of Myocardium with SVF-Net* | Multi-atlas label fusion; encoder–decoder registration module | .957 / .900 | .916 / .845 | .867 / .869 | .892 | 12.1 |
| Tziritas & Grinias, *Fast Fully-Automatic Localization of Left Ventricle and Myocardium in MRI Using MRF Model Optimization, Sub-structures Tracking and B-Spline Smoothing* | Chan–Vese level set, MRF graph cut and B-spline smoothing | .948 / .865 | .863 / .743 | .794 / .801 | .836 | 15.8 |
| Yang et al., *Class-Balanced Deep Neural Network for Automatic Ventricular Structure Segmentation* | Residual 3D U-Net with multiclass Dice loss; no myocardium submission | .864 / .775 | .789 / .770 | N/A | .800† | 40.6† |

All raw numbers in this table come from the organizers’ [test-set Table
III](https://www.creatis.insa-lyon.fr/Challenge/acdc/files/tmi_2018_bernard.pdf).
The architecture summaries likewise follow its Table II; the paper’s reference
list gives the full proceedings citations for every contender. For two useful
direct paper copies, see [Zotti et al. (arXiv)](https://arxiv.org/abs/1705.08943)
and [Wolterink et al. (arXiv)](https://arxiv.org/abs/1708.01141). †Yang did not
submit myocardium results, so both derived values average only the four available
LV/RV phase cells and should not be ranked against complete three-structure
submissions.

## How these results relate to this repository

The reference paper reproduced here is Baumgartner et al., *An Exploration of
2D and 3D Deep Learning Techniques for Cardiac MR Image Segmentation*. Its
challenge submission was a 2D U-Net trained with cross-entropy. On the official
test set it scored LV `.963/.911`, RV `.932/.883`, and Myo `.892/.901` at
ED/ES (derived macro Dice `.914`, mean HD `10.4` mm). It therefore sits in the
upper group of the original 10 submissions, though below the 2D+3D U-Net
ensemble of Isensee et al. The same official paper describes these choices and
the competition evaluation: [Baumgartner result in Bernard et al.](https://www.creatis.insa-lyon.fr/Challenge/acdc/files/tmi_2018_bernard.pdf).

This repository's headline result is a modified 2D U-Net with weighted CE:
mean foreground Dice `.875` (RV `.859`, Myo `.846`, LV `.921`); its anisotropic
3D U-Net reaches `.862` (RV `.841`, Myo `.817`, LV `.928`). Those are calculated
on 40 reconstructed validation volumes from 20 training-set patients, as
documented in [the local results README](../results/README.md) and
[project README](../README.md).

Consequently, the two result blocks should be treated as contextual rather than
as a strict leaderboard comparison:

- The contender table is organizer-scored, hidden **50-patient test** data and
  reports each structure separately at ED and ES.
- The local figures are selected-run **20-patient validation** results and pool
  the reconstructed ED/ES volumes into one score per structure.
- A valid ranking claim would require producing predictions for the official
  held-out test set and applying the challenge evaluator with the same
  post-processing and submission protocol. Do not compare the `.875` local
  validation mean directly to the contender-table macro Dice values.

## Takeaways for comparison

The original challenge evidence favors three practical conclusions. First,
2D U-Net variants were already strong: the Baumgartner, Jang, Khened and
Patravali/Jain entries form much of the top half. Second, the best original
submission combined 2D and anisotropic 3D U-Nets rather than selecting one
dimension alone. Third, myocardium and ES RV are consistently harder than LV,
which makes the repository's lower Myo/RV scores more informative than its
strong LV score. These are cross-method patterns in a shared hidden-test
evaluation, not causal proof that any individual architectural feature alone
produced the difference.
