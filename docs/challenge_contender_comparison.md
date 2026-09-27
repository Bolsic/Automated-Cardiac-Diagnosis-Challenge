# ACDC challenge contender comparison

This note places the current five-fold experiments in the context of methods
submitted to the MICCAI 2017 Automated Cardiac Diagnosis Challenge (ACDC). It
is not a general survey of later work using the ACDC dataset, and the local
metrics are not treated as organizer-scored leaderboard results.

## Source and scope

The challenge organizers report that 10 teams submitted meaningful
segmentation results during the contest. Their challenge paper describes the
architectures and provides one uniform evaluation of all 10 submissions on the
hidden test set; it is the source of the values below: [Bernard et al., 2018,
Tables II and III](https://www.creatis.insa-lyon.fr/Challenge/acdc/files/tmi_2018_bernard.pdf).
The [official ACDC test-results leaderboard](https://www.creatis.insa-lyon.fr/Challenge/acdc/results.html)
states that its final update was November 2022.

The contest test set contains 50 patients. Each entry below reports Dice and
Hausdorff distance (HD, mm) at end diastole (ED) and end systole (ES) for the
left ventricle (LV), right ventricle (RV), and myocardium (Myo). `Macro Dice`
is the unweighted mean of the six displayed Dice values and is derived here;
it was not an official contest ranking metric.

## Results of the original challenge methods

| Paper / challenge method | Concise method description | LV ED / ES Dice | RV ED / ES Dice | Myo ED / ES Dice | Macro Dice (derived) | Mean HD across six cells (derived, mm) |
|---|---|---:|---:|---:|---:|---:|
| Isensee et al., *Automatic Cardiac Disease Assessment on Cine-MRI via Time-Series Segmentation and Domain Specific Features* | Ensemble of 2D and anisotropic 3D U-Nets; Dice loss | .968 / .931 | .946 / .899 | .902 / .919 | **.928** | 9.0 |
| Khened et al., *Densely Connected Fully Convolutional Network for Short-Axis Cardiac Cine MR Image Segmentation and Heart Diagnosis Using Random Forest* | Dense 2D U-Net with inception first layer; combined Dice + CE loss | .964 / .917 | .935 / .879 | .889 / .898 | .914 | 11.2 |
| Baumgartner et al., *An Exploration of 2D and 3D Deep Learning Techniques for Cardiac MR Image Segmentation* | 2D U-Net; cross-entropy | .963 / .911 | .932 / .883 | .892 / .901 | .914 | 10.4 |
| Jang et al., *Automatic Segmentation of LV and RV in Cardiac MRI* | 2D M-Net; weighted cross-entropy | .959 / .921 | .929 / .885 | .875 / .895 | .911 | 9.7 |
| Zotti et al., *GridNet with Automatic Shape Prior Registration for Automatic MRI Cardiac Segmentation* | 2D GridNet/U-Net extension with automatically registered shape prior | .957 / .905 | .941 / .882 | .884 / .896 | .911 | 9.6 |
| Wolterink et al., *Automatic Segmentation and Disease Classification Using Cardiac Cine MR Images* | 2D dilated CNN jointly processing corresponding ED and ES slices | .961 / .918 | .928 / .872 | .875 / .894 | .908 | 10.7 |
| Patravali, Jain & Chilamkurthy, *2D-3D Fully Convolutional Neural Networks for Cardiac MR Segmentation* | Best submission: 2D U-Net with Dice loss | .955 / .885 | .911 / .819 | .882 / .897 | .892 | 12.1 |
| Rohé, Sermesant & Pennec, *Automatic Multi-Atlas Segmentation of Myocardium with SVF-Net* | Multi-atlas label fusion; encoder–decoder registration module | .957 / .900 | .916 / .845 | .867 / .869 | .892 | 12.1 |
| Tziritas & Grinias, *Fast Fully-Automatic Localization of Left Ventricle and Myocardium in MRI Using MRF Model Optimization, Sub-structures Tracking and B-Spline Smoothing* | Chan–Vese level set, MRF graph cut, and B-spline smoothing | .948 / .865 | .863 / .743 | .794 / .801 | .836 | 15.8 |
| Yang et al., *Class-Balanced Deep Neural Network for Automatic Ventricular Structure Segmentation* | Residual 3D U-Net with multiclass Dice loss; no myocardium submission | .864 / .775 | .789 / .770 | N/A | .800† | 40.6† |

All source values are from the organizers' test-set Table III. Architecture
summaries follow Table II of the same paper. Two useful direct paper copies are
[Zotti et al.](https://arxiv.org/abs/1705.08943) and
[Wolterink et al.](https://arxiv.org/abs/1708.01141). †Yang did not submit
myocardium results, so its derived values average only the four available
LV/RV phase cells and should not be ranked against complete three-structure
submissions.

## Current repository results

The current repository results replace the earlier single-split comparison
with diagnosis-stratified five-fold validation over patients 001–100. Every
training patient is used for validation exactly once. Final local testing uses
a five-model softmax ensemble on patients 101–150.

| Model | Five-fold validation Dice, mean ± SD | Local ensemble Dice on 100 ED/ES volumes |
|---|---:|---:|
| FCN-8 | 0.8810 ± 0.0105 | 0.8934 |
| **2D U-Net** | **0.9025 ± 0.0064** | **0.9109** |
| Modified 2D U-Net | 0.9006 ± 0.0043 | 0.9085 |
| Anisotropic 3D U-Net | 0.8407 ± 0.0176 | 0.8583 |

The standard 2D U-Net is the strongest current model. Its local ensemble Dice
of `0.9109` is numerically close to the derived `0.914` macro Dice of the
Baumgartner challenge entry, but the numbers are not interchangeable.

## Why the numbers are contextual rather than a leaderboard ranking

- The challenge table is organizer-scored on the original hidden test set and
  reports each structure separately at ED and ES.
- The local CV value averages reconstructed volumes within each fold and then
  reports the mean and population standard deviation across five folds.
- The local ensemble value pools the 100 ED/ES volumes and three foreground
  structures into a single macro result after repository-specific
  largest-component post-processing.
- The repository evaluation was run locally; it was not submitted to or scored
  by the official challenge server.

A strict ranking claim would require generating a submission and using the
organizers' exact evaluation protocol. The local `0.9109` therefore must not be
inserted directly into the challenge table as though it were an official
leaderboard score.

## Takeaways

The original challenge evidence and the current experiments point in the same
general direction: 2D U-Net variants are strong on ACDC, and RV and myocardium
are usually more difficult than LV. In this repository, the standard 2D U-Net
slightly outperforms the parameter-reduced modified model, while the local
anisotropic 3D model is less accurate and more variable across folds. These are
empirical observations under the recorded protocols, not proof that model
dimensionality alone causes the differences.
