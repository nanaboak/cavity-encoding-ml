# Method

The package implements the method of Boakye, Hafizh, Lohar and Abdul Majeed (accepted, *Journal of Intelligent Manufacturing*, 2026).

## Noise floors

Parts of one shot are formed under the same machine settings. The spread inside a shot is therefore the part-to-part scatter that no model with only shot-level inputs can explain.

- `sigma_blind` is the pooled within-shot standard deviation of the response.
- `sigma_noise` is the same quantity after the systematic cavity offsets (cavity mean minus grand mean) have been removed.

## F test

For S shots and k cavities, in a balanced layout

F = (S - 1) (sigma_blind^2 - sigma_noise^2) / sigma_noise^2

with k - 1 and (k - 1)(S - 1) degrees of freedom. The package uses the exact within-shot least-squares form, which gives the same value for balanced data and remains valid when single parts are missing. The step passes when p < alpha (default 0.05) and the partial eta squared is at least 0.01.

## Floor separation

1 - sigma_noise / sigma_blind is the largest relative RMSE reduction that cavity information can deliver when the cavity-blind and the cavity-aware model have the same excess error.

## Representations

| Code | Model inputs |
|---|---|
| R0 | Machine features and the shot average of each cavity sensor (cavity-blind) |
| R1 | Machine features and the sensors of the producing cavity |
| R2 | R1 and the cavity number as one numeric column |
| R3 | R1 and one binary column per cavity (default) |
| R4 | R1 and the smoothed cavity mean of the response |

Target encoding uses `(n_c * mean_c + m * mean_all) / (n_c + m)` with m = 10 and is fitted on the training rows of every fold.

## Validation

`ShotGroupKFold` shuffles the shots, not the parts, and splits them into folds. The default is 4 folds and 3 repeats, with seeds 42, 1042 and 2042. The out-of-fold predictions of the repeats are averaged for every part before the pooled RMSE is computed.

## Decision rule

The jitter is the standard deviation of the fold RMSE of the cavity-blind model divided by its mean. The package reports that the cavity information adds noise when

floor separation < margin x jitter

The default margin is 2. In the original study the ratio was about 0.2 for tensile strength and between 3.4 and 15 for the three characteristics that benefited from the label, so the default lies between them. It can be changed.

## References

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Erlbaum.

Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press.
