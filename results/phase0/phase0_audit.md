# Phase 0 audit (CD-MSC development data)

- wav files on disk: **271380** (README says 271,380 plus the dataset has 271380 ids across splits)
- ids missing on disk: **0**
- split sizes: {"Training": 213647, "Validation": 30516, "Test": 27217, "TrainVal": 244163}
- official unseen domain per species: {"Ae.aeg": "D3", "Ae.alb": "D2", "Cx.qui": "D1", "An.gam": "D1", "An.ara": "D1", "An.dir": "D4", "Cx.pip": "D3", "An.min": "D2", "An.ste": "D4"}
- agreement of official unseen flag with "(species, domain) pair absent from Training+Validation" on test clips: **1.0000**
- official-unseen clips found in Training/Validation: **0** (should be 0)

## Clips per species x domain (all splits)

| species   |   D1 |   D2 |   D3 |   D4 |     D5 |    All |
|:----------|-----:|-----:|-----:|-----:|-------:|-------:|
| Ae.aeg    |  123 |    0 |  192 |   22 |  81250 |  81587 |
| Ae.alb    |   79 |  419 |   19 |    0 |  18000 |  18517 |
| An.ara    | 1820 |    0 |    0 |    0 |  19297 |  21117 |
| An.dir    |   87 |    0 |    0 |   40 |      0 |    127 |
| An.gam    |  818 |    0 |    0 |    0 |  46180 |  46998 |
| An.min    |  200 |   99 |  200 |   51 |      0 |    550 |
| An.ste    |  200 |  200 |  200 |   74 |      0 |    674 |
| Cx.pip    |   66 |   66 |   68 |    0 |  29554 |  29754 |
| Cx.qui    |  672 |    0 |    0 |   13 |  71371 |  72056 |
| All       | 4065 |  784 |  679 |  200 | 265652 | 271380 |

## Training clips per species x domain

| species   |   D1 |   D2 |   D3 |   D4 |     D5 |    All |
|:----------|-----:|-----:|-----:|-----:|-------:|-------:|
| Ae.aeg    |   95 |    0 |    0 |   19 |  64137 |  64251 |
| Ae.alb    |   67 |    0 |   15 |    0 |  14500 |  14582 |
| An.ara    |    0 |    0 |    0 |    0 |  16630 |  16630 |
| An.dir    |   76 |    0 |    0 |    0 |      0 |     76 |
| An.gam    |    0 |    0 |    0 |    0 |  37011 |  37011 |
| An.min    |  176 |    0 |  170 |   49 |      0 |    395 |
| An.ste    |  168 |  178 |  179 |    0 |      0 |    525 |
| Cx.pip    |   52 |   52 |    0 |    0 |  23328 |  23432 |
| Cx.qui    |    0 |    0 |    0 |   12 |  56733 |  56745 |
| All       |  634 |  230 |  364 |   80 | 212339 | 213647 |

## Test clips: seen vs unseen (official definition)

| species   |   seen |   unseen |   All |
|:----------|-------:|---------:|------:|
| Ae.aeg    |   7967 |      192 |  8159 |
| Ae.alb    |   1433 |      419 |  1852 |
| An.ara    |    292 |     1820 |  2112 |
| An.dir    |      0 |       40 |    40 |
| An.gam    |   3882 |      818 |  4700 |
| An.min    |      0 |       99 |    99 |
| An.ste    |      0 |       74 |    74 |
| Cx.pip    |   2907 |       68 |  2975 |
| Cx.qui    |   6534 |      672 |  7206 |
| All       |  23015 |     4202 | 27217 |

## Test clips per species x domain

| species   |   D1 |   D2 |   D3 |   D4 |    D5 |   All |
|:----------|-----:|-----:|-----:|-----:|------:|------:|
| Ae.aeg    |   12 |    0 |  192 |    2 |  7953 |  8159 |
| Ae.alb    |    6 |  419 |    2 |    0 |  1425 |  1852 |
| An.ara    | 1820 |    0 |    0 |    0 |   292 |  2112 |
| An.dir    |    0 |    0 |    0 |   40 |     0 |    40 |
| An.gam    |  818 |    0 |    0 |    0 |  3882 |  4700 |
| An.min    |    0 |   99 |    0 |    0 |     0 |    99 |
| An.ste    |    0 |    0 |    0 |   74 |     0 |    74 |
| Cx.pip    |    7 |    6 |   68 |    0 |  2894 |  2975 |
| Cx.qui    |  672 |    0 |    0 |    1 |  6533 |  7206 |
| All       | 3335 |  524 |  262 |  117 | 22979 | 27217 |

## Audio properties per domain (sample of up to 150 clips each)

|   domain |   n | sample_rates         |   dur_mean |   dur_min |   dur_max | channels   |
|---------:|----:|:---------------------|-----------:|----------:|----------:|:-----------|
|        1 | 150 | [8000, 44100]        |       7.45 |      0.04 |     53.76 | [1]        |
|        2 | 150 | [8000]               |       2.68 |      0.44 |     34.64 | [1]        |
|        3 | 150 | [8000]               |      12.42 |      2    |     36.46 | [1]        |
|        4 | 150 | [8000, 44100, 48000] |      50.53 |      0.2  |    293.89 | [1]        |
|        5 | 150 | [8000]               |       0.62 |      0.62 |      0.62 | [1]        |

## split_summary.json (first 3000 characters)

```json
{
  "train_test_seed": 42,
  "validation_seed": 43,
  "validation_ratio_within_trainval": 0.12498330208712395,
  "counts": {
    "train": 213647,
    "validation": 30516,
    "test": 27217,
    "trainval": 244163,
    "all_total": 271380
  },
  "unseen_domain_by_species": {
    "Aedes aegypti": "D3",
    "Aedes albopictus": "D2",
    "Culex quinquefasciatus": "D1",
    "Anopheles gambiae": "D1",
    "Anopheles arabiensis": "D1",
    "Anopheles dirus": "D4",
    "Culex pipiens": "D3",
    "Anopheles minimus": "D2",
    "Anopheles stephensi": "D4"
  },
  "species_summary": [
    {
      "species": "Aedes aegypti",
      "unseen_domain": "D3",
      "species_total": 81587,
      "test_total": 8159,
      "trainval_total": 73428
    },
    {
      "species": "Aedes albopictus",
      "unseen_domain": "D2",
      "species_total": 18517,
      "test_total": 1852,
      "trainval_total": 16665
    },
    {
      "species": "Culex quinquefasciatus",
      "unseen_domain": "D1",
      "species_total": 72056,
      "test_total": 7206,
      "trainval_total": 64850
    },
    {
      "species": "Anopheles gambiae",
      "unseen_domain": "D1",
      "species_total": 46998,
      "test_total": 4700,
      "trainval_total": 42298
    },
    {
      "species": "Anopheles arabiensis",
      "unseen_domain": "D1",
      "species_total": 21117,
      "test_total": 2112,
      "trainval_total": 19005
    },
    {
      "species": "Anopheles dirus",
      "unseen_domain": "D4",
      "species_total": 127,
      "test_total": 40,
      "trainval_total": 87
    },
    {
      "species": "Culex pipiens",
      "unseen_domain": "D3",
      "species_total": 29754,
      "test_total": 2975,
      "trainval_total": 26779
    },
    {
      "species": "Anopheles minimus",
      "unseen_domain": "D2",
      "species_total": 550,
      "test_total": 99,
      "trainval_total": 451
    },
    {
      "species": "Anopheles stephensi",
      "unseen_domain": "D4",
      "species_total": 674,
      "test_total": 74,
      "trainval_total": 600
    }
  ]
}
```
