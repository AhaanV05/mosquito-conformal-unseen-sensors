# Phase 4: sensor-conditional (Mondrian) conformal sets on the whole dev test

| rule | alpha | overall cov | species-macro cov | mean size | D5 cov / size | mic cov / size | unseen cov / size | seen cov | unseen groups >= 0.80 |
|---|---|---|---|---|---|---|---|---|---|
| global C0 (validation) | 0.10 | 0.785 | 0.594 | 1.54 | 0.900 / 1.52 | 0.164 / 1.64 | 0.158 / 1.65 | 0.900 | 2/9 |
| global C1 (pooled pseudo-unseen) | 0.10 | 0.973 | 0.960 | 6.81 | 1.000 / 6.60 | 0.824 / 7.94 | 0.822 / 7.94 | 1.000 | 8/9 |
| Mondrian by sensor (detected) mic=C1 | 0.10 | 0.886 | 0.878 | 2.48 | 0.900 / 1.52 | 0.807 / 7.70 | 0.805 / 7.71 | 0.900 | 7/9 |
| Mondrian by sensor (oracle) mic=C1 | 0.10 | 0.888 | 0.910 | 2.52 | 0.900 / 1.52 | 0.824 / 7.94 | 0.822 / 7.94 | 0.900 | 8/9 |
| Mondrian by sensor (detected) mic=C2b | 0.10 | 0.901 | 0.897 | 2.58 | 0.900 / 1.52 | 0.904 / 8.29 | 0.903 / 8.30 | 0.900 | 8/9 |
| Mondrian by sensor (oracle) mic=C2b | 0.10 | 0.904 | 0.930 | 2.62 | 0.900 / 1.52 | 0.923 / 8.55 | 0.922 / 8.56 | 0.900 | 9/9 |
| Mondrian by sensor (detected) mic=C2 | 0.10 | 0.912 | 0.918 | 2.64 | 0.900 / 1.52 | 0.980 / 8.72 | 0.980 / 8.72 | 0.900 | 8/9 |
| Mondrian by sensor (oracle) mic=C2 | 0.10 | 0.916 | 0.951 | 2.69 | 0.900 / 1.52 | 1.000 / 9.00 | 1.000 / 9.00 | 0.900 | 9/9 |
| global C0 (validation) | 0.20 | 0.702 | 0.532 | 1.13 | 0.808 / 1.11 | 0.128 / 1.19 | 0.121 / 1.19 | 0.808 | 1/9 |
| global C1 (pooled pseudo-unseen) | 0.20 | 0.930 | 0.829 | 4.57 | 0.998 / 4.32 | 0.558 / 5.92 | 0.554 / 5.93 | 0.998 | 3/9 |
| Mondrian by sensor (detected) mic=C1 | 0.20 | 0.767 | 0.701 | 1.83 | 0.808 / 1.12 | 0.542 / 5.73 | 0.538 / 5.74 | 0.809 | 3/9 |
| Mondrian by sensor (oracle) mic=C1 | 0.20 | 0.769 | 0.736 | 1.86 | 0.808 / 1.12 | 0.558 / 5.92 | 0.554 / 5.93 | 0.809 | 3/9 |
| Mondrian by sensor (detected) mic=C2b | 0.20 | 0.805 | 0.827 | 2.12 | 0.808 / 1.12 | 0.785 / 7.57 | 0.783 / 7.58 | 0.809 | 7/9 |
| Mondrian by sensor (oracle) mic=C2b | 0.20 | 0.808 | 0.863 | 2.16 | 0.808 / 1.12 | 0.805 / 7.82 | 0.804 / 7.83 | 0.809 | 8/9 |
| Mondrian by sensor (detected) mic=C2 | 0.20 | 0.834 | 0.869 | 2.29 | 0.808 / 1.12 | 0.974 / 8.66 | 0.974 / 8.66 | 0.809 | 8/9 |
| Mondrian by sensor (oracle) mic=C2 | 0.20 | 0.838 | 0.906 | 2.34 | 0.808 / 1.12 | 0.997 / 8.97 | 0.997 / 8.97 | 0.809 | 9/9 |
