# Beginner guide

This page explains the idea without formulas.

## The problem

A multi-cavity mold makes, for example, four parts in every machine cycle. The four parts are made with the same machine settings, but they are not identical. Cavity 3 may always make a slightly larger part than cavity 2. That systematic difference is a *cavity effect*.

When you train a model to predict part quality, you can

- leave the cavity out (the model is *cavity-blind*), or
- tell the model which cavity made the part (the model is *cavity-aware*).

Telling the model sounds like a free improvement, but it is not always one. If the cavities barely differ, the extra input only adds noise.

## The four steps

1. **Diagnostic.** An F test asks whether the cavities differ more than chance allows. It also computes the *floor separation*. This number is the best improvement in RMSE (the typical prediction error) that cavity information could ever give.
2. **Encoding.** A model cannot read "cavity 3" as a name, so the label must become numbers. *One-hot* gives one 0/1 column per cavity and is the default. *Target encoding* gives one number per cavity, the average response of that cavity. *Integer* just uses 1, 2, 3, 4, which wrongly suggests that cavity 4 is "more" than cavity 1.
3. **Validation.** To test a model honestly, you hide some data from it. All four parts of a shot share their machine settings, so if one part is hidden and its sibling is not, the model can almost copy the answer. This is *leakage*. The package keeps every shot together (`ShotGroupKFold`) and warns you when you use ordinary K-fold.
4. **Decision.** Repeat the test on different splits and the error changes a little each time. That spread is the *fold-to-fold jitter*. If the possible gain (the floor separation) is not clearly larger than this jitter, the gain cannot be told apart from luck, and the package tells you that the cavity information adds noise.

## How to read the report

```
STEP 1  F(3, 285) = 339.6 ... floor separation 53.2 %      <- cavities differ a lot
STEP 3  R0 cavity-blind          0.0175
        R1 cavity readings       0.0170   +2.6 %           <- sensors alone help a little
        R3 + onehot label        0.0155   +11.6 %          <- the label helps more
        fold-to-fold jitter 11.1 %
STEP 4  separation 53.2 % is 4.8 times the jitter -> keep the label
```

## Words you will meet

| Word | Meaning |
|---|---|
| Shot | One machine cycle, which forms several parts |
| Cavity | One of the mold cavities, numbered 1 to k |
| RMSE | Root mean squared error, the typical size of a prediction error |
| Fold | One part of the data that is hidden from the model during a test |
| Leakage | The model sees information in training that it would not have in real use |
| Jitter | How much the RMSE changes from fold to fold |
