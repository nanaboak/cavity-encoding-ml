---
title: 'cavity-encoding-ml, a Python package for cavity encoding and leakage-safe validation in multi-cavity injection molding'
tags:
  - Python
  - injection molding
  - multi-cavity mold
  - machine learning
  - quality prediction
  - categorical encoding
  - data leakage
  - cross-validation
authors:
  - name: Prince Nana Boakye
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Hadyan Hafizh
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Pintu Lohar
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Anwar P. P. Abdul Majeed
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: Faculty of Engineering and Technology, Sunway University, Bandar Sunway, Petaling Jaya, Selangor 47500, Malaysia
    index: 1
date: 30 October 2026
bibliography: paper.bib
---

# Summary

During the injection molding process, a cycle or 'shot' comprises the injection of plastic material into a cavity to form the desired part. To optimize production costs, a mold can feature multiple cavities to produce several parts simultaneously. When machine learning predicts part quality from process data, the analyst must decide whether the cavity that formed a part should enter the model, how the cavity label should be encoded, and how the model should be validated when the parts of one shot share the same machine settings. `cavity-encoding-ml` is a Python package that guides these three decisions in four steps. It first uses an F test to check whether the cavities differ systematically. If they do, it encodes the cavity label, by default with one-hot encoding and alternatively with target or integer encoding. It then validates models with and without the label on folds that keep every shot together and raises an alert when ordinary K-fold cross-validation separates the parts of a shot. Finally, it compares the largest error reduction that the cavity label can deliver with the fold-to-fold jitter of the cavity-blind model and reports whether the label is worth keeping. It implements the method of @boakye2026cavity and works with any machine learning library that offers a fit and predict interface.

# Statement of need

Multi-cavity molds are the economic norm in serial production, because the fixed cost of a cycle is divided among all parts formed in it [@kazmer2022]. In the resulting data every shot occupies several rows that share one machine-state vector. This structure creates two problems. First, systematic differences between cavities can be large enough to improve prediction, but they can also be statistically detectable and still too small to exploit. Second, when parts are assigned to folds at random, a part is often tested while its siblings from the same shot are in the training set, and the error estimate becomes optimistic. Such leakage from dependent observations is described by @kaufman2012 and @kapoor2023, and @roberts2017 showed that resampling must then be grouped by the unit of dependence.

The original study of the method [@boakye2026cavity] quantified both problems on four quality characteristics of a polyamide gear produced in a four-cavity mold. Encoding the cavity label reduced the root mean squared error (RMSE) by 12.9% to 21.6% for gear diameter, tip diameter and solution viscosity. Tensile strength showed a significant cavity effect with an F statistic of 11.4, yet no cavity-aware model improved significantly on the cavity-blind one. The largest possible gain of 1.1% in RMSE was smaller than the 5.1% by which the error of the cavity-blind model varied from one fold to the next. Ordinary K-fold cross-validation understated the RMSE by up to 44.8% compared with shot-grouped cross-validation. `cavity-encoding-ml` turns the analysis behind these findings into a tested tool for new data. It is intended for process engineers and researchers who build quality prediction models from injection molding data, and it applies in the same way to other processes in which one machine cycle produces several parts in parallel positions.

# State of the field

General machine learning libraries provide the building blocks but not the decision. scikit-learn [@pedregosa2011] offers grouped splitters such as `GroupKFold`, a one-hot encoder and, since version 1.3, a target encoder, while category_encoders [@mcginnis2018] and Feature-engine [@galli2021] add many further encoders and feature transformers. None of them tells the user whether a categorical variable with a physical meaning deserves to be encoded, and none warns when a random split separates dependent rows. Statistical packages such as statsmodels [@seabold2010] and Pingouin [@vallat2018] can fit the two-way analysis of variance behind the cavity test, but the user must still translate the result into noise floors and an expected error reduction. Validation suites such as Deepchecks [@chorev2022] inspect a finished split for problems such as duplicated samples, but they have no notion of a shot. Machine learning studies that consider multi-cavity molds remain rare, with @parizs2022 and @boakye2026kan among the few examples, and to the knowledge of the authors no open-source package addresses the cavity dimension of injection molding data. `cavity-encoding-ml` connects the cavity-effect test, the choice of encoding, shot-aware validation and a decision rule in one workflow.

# Software design

The package mirrors the four steps of the method. Each step works on its own or through the guided `CavityWorkflow` class (\autoref{fig-workflow}), and all components follow the estimator conventions of scikit-learn [@buitinck2013].

The diagnostic (`cavity_diagnostic`) needs only the measured response and the shot and cavity identifiers. It computes the pooled within-shot standard deviation of the raw response, $\sigma_\text{blind}$, and after removal of the cavity offsets, $\sigma_\text{noise}$, together with the F statistic of the cavity effect. For a balanced layout with $S$ shots and $k$ cavities,
$$F = \frac{(S-1)\,(\sigma_\text{blind}^2-\sigma_\text{noise}^2)}{\sigma_\text{noise}^2},$$
with $k-1$ and $(k-1)(S-1)$ degrees of freedom. The implementation uses an exact within-shot least-squares fit, which gives the same value for balanced data and remains valid when single parts are missing. The step passes when the effect is significant and its partial eta squared reaches a minimum size, by default the conventional lower bound of a small effect [@cohen1988]. The floor separation $1-\sigma_\text{noise}/\sigma_\text{blind}$ is reported as the largest fractional RMSE reduction that cavity information can deliver.

Machine logs usually store the sensor reading of every cavity on every row of a shot. A `CavityConfig` records this layout once, and the `CavityMapper` transformer builds from it the five cavity representations of the original study without hand-written feature engineering (\autoref{tab-reps}). One-hot encoding (R3) is the default because it has no fitted parameters and shared the best mean rank with target encoding (R4) in that study. Target encoding with empirical Bayes smoothing [@micci2001] keeps the input narrow for molds with many cavities and is refitted on the training rows of each fold. Integer encoding (R2) is retained for comparison with earlier models of the same machine, although it imposes an order on the cavities that has no physical meaning. In its default strict mode the mapper raises an error when it has to transform a part whose shot sibling was used for fitting, so a leaky split fails instead of silently inflating the accuracy.

| Code | Representation | Model inputs |
|:-----|:---------------|:-------------|
| R0 | Cavity-blind baseline | Machine features and the shot average of each cavity sensor |
| R1 | Cavity readings | Machine features and the sensors of the producing cavity |
| R2 | Integer label | R1 and the cavity number as one numeric column |
| R3 | One-hot label (default) | R1 and one binary column per cavity |
| R4 | Target-encoded label | R1 and the smoothed cavity mean of the response |

Table: Cavity representations of `CavityMapper`, coded as in the original study. \label{tab-reps}

The validation step (`validate_cavity_encoding`) trains a fresh copy of the user's model on identical folds for the baseline (R0), for the cavity readings without the label (R1) and for the encoded label. This separates the value of the label from the value of cavity-level sensors. Folds come from `ShotGroupKFold`, a scikit-learn compatible splitter, and any other split is checked for sibling leakage and reported with a `LeakageWarning`. Models from any library are accepted as an object with `fit` and `predict` methods, as a factory, or as a `FunctionModel` that wraps two functions for networks written in PyTorch, TensorFlow or JAX.

The decision (`decide`) compares the floor separation with the fold-to-fold jitter, defined as the standard deviation of the fold RMSE of the cavity-blind model divided by its mean. When the separation is smaller than twice the jitter, the package reports that cavity information adds noise and recommends leaving the label out. The default factor lies between the ratios observed in the original study for tensile strength (0.2) and for the three characteristics that benefited from the label (3.4 to 15), and it can be changed. Every result prints a plain-text report, and a command-line interface serves users who do not write code. The core depends only on NumPy [@harris2020], pandas [@mckinney2010], SciPy [@virtanen2020] and scikit-learn, with Matplotlib [@hunter2007] as an optional dependency for figures.

![Workflow of `cavity-encoding-ml` (a) and the decision for the tip diameter and tensile strength data of the original study (b). The floor separation is the largest RMSE reduction the cavity label can deliver. The fold-to-fold jitter is the relative spread of the fold RMSE of the cavity-blind random forest under shot-grouped cross-validation.\label{fig-workflow}](figure1.png)

# Research impact statement

The method implemented in the package was developed and validated by @boakye2026cavity, which builds on the shot-grouped validation of @boakye2026kan for the same machine. The package was checked against the original code on the tip-diameter data of that study (384 parts from 96 shots). With the same fixed feature set, the five representations of \autoref{tab-reps} and the shot-grouped folds of the package reproduce the pooled RMSE of the original code for random forest, gradient boosting, support vector regression and multilayer perceptron models to machine precision, and exactly for the random forest. `cavity_diagnostic` reproduces the published values exactly, with F(3, 285) = 326.4, $\sigma_\text{blind}$ = 10.40 $\mu$m, $\sigma_\text{noise}$ = 4.94 $\mu$m and a floor separation of 52.5%. In the guided workflow, the one-hot encoded random forest reached a shot-grouped RMSE of 13.68 $\mu$m against 15.70 $\mu$m for the cavity-blind model, close to the published 13.89 $\mu$m and 15.92 $\mu$m that relied on nested feature selection. The floor separation was 5.9 times the fold-to-fold jitter of 8.9%, and the workflow recommended the cavity label. Under ordinary K-fold cross-validation, 98.6% of the test parts had a sibling in the training fold, and the RMSE of the cavity-blind model was understated by 24.3%. For tensile strength the package detected the significant cavity effect (F = 11.4) but recommended leaving the label out, because the floor separation of 1.1% was below the fold-to-fold jitter of 3.7%, in agreement with the original study. As the production data are confidential, the repository ships the verification scripts and a synthetic data set with the published variance structure of the tip diameter.

# AI usage disclosure

AI tools were used to assist with language editing, and the authors remain fully responsible for the content and integrity of the paper.

# Acknowledgements

The authors thank the production and quality departments of the industrial partner for access to the machine and measurement records used in the original study. This work received no specific funding.

# References
