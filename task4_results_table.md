## Grid Search Results

The following table shows the test error corresponding to the minimum validation error for each hyperparameter combination.

**Note:** The best combination is highlighted in bold.

| Hidden Size | Batch Size | Learning Rate | Min Val Error | Test Error | Best Epoch |
| ---------- | ---------- | ------------ | ------------- | ---------- | ---------- |
| 500.0 | 50.0 | 0.001000 | 0.0766 | 0.0744 | 4 |
| **<span style="background-color: yellow">300.0</span>** | **<span style="background-color: yellow">50.0</span>** | **<span style="background-color: yellow">0.001000</span>** | **<span style="background-color: yellow">0.0862</span>** | **<span style="background-color: yellow">0.0734</span>** | **<span style="background-color: yellow">4</span>** |
| 500.0 | 100.0 | 0.001000 | 0.0885 | 0.0773 | 4 |
| 500.0 | 200.0 | 0.001000 | 0.0959 | 0.0846 | 4 |
| 300.0 | 100.0 | 0.001000 | 0.0995 | 0.0849 | 4 |
| 300.0 | 200.0 | 0.010000 | 0.1053 | 0.0987 | 2 |
| 100.0 | 50.0 | 0.001000 | 0.1058 | 0.0897 | 4 |
| 300.0 | 200.0 | 0.001000 | 0.1112 | 0.0964 | 4 |
| 500.0 | 200.0 | 0.010000 | 0.1119 | 0.1093 | 1 |
| 100.0 | 200.0 | 0.010000 | 0.1213 | 0.1092 | 1 |
| 100.0 | 100.0 | 0.001000 | 0.1218 | 0.1066 | 4 |
| 100.0 | 100.0 | 0.010000 | 0.1225 | 0.1169 | 1 |
| 500.0 | 100.0 | 0.010000 | 0.1350 | 0.1253 | 1 |
| 300.0 | 100.0 | 0.010000 | 0.1373 | 0.1210 | 3 |
| 100.0 | 50.0 | 0.010000 | 0.1503 | 0.1479 | 1 |
| 100.0 | 200.0 | 0.001000 | 0.1549 | 0.1366 | 4 |
| 300.0 | 50.0 | 0.010000 | 0.1618 | 0.1562 | 2 |
| 500.0 | 50.0 | 0.010000 | 0.1679 | 0.1525 | 2 |
| 500.0 | 50.0 | 0.000100 | 0.1846 | 0.1653 | 4 |
| 300.0 | 50.0 | 0.000100 | 0.2167 | 0.1973 | 4 |
| 500.0 | 100.0 | 0.000100 | 0.2207 | 0.2015 | 4 |
| 300.0 | 100.0 | 0.000100 | 0.2509 | 0.2324 | 4 |
| 500.0 | 200.0 | 0.000100 | 0.2639 | 0.2444 | 4 |
| 100.0 | 50.0 | 0.000100 | 0.2757 | 0.2561 | 4 |
| 300.0 | 200.0 | 0.000100 | 0.2940 | 0.2749 | 4 |
| 100.0 | 100.0 | 0.000100 | 0.3097 | 0.2885 | 4 |
| 100.0 | 200.0 | 0.000100 | 0.3598 | 0.3375 | 4 |

### Best Hyperparameters

- Hidden Size: 500
- Batch Size: 50
- Learning Rate: 0.001000
- Minimum Validation Error: 0.0766
- Corresponding Test Error: 0.0744
- Best Epoch: 4
- Best Test Accuracy: 97.81%
