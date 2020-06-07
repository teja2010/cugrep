# cugrep

grep in cuda.
An exercise in understanding CUDA and Regex.


----
wc : 19058252

`fo*s` : 40506 (0.2%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |1.95    |
| grep        |3.64    |
| cugrep 0.1  |2.42    |
| cugrep 0.2  |0.84    |



PATTERN `ss*i` : 1945310 (1%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |1.65    |
| grep        |3.99    |
| cugrep 0.1  |4.17    |
| cugrep 0.2  |2.41    |

PATTERN `s*a`  : 14534739 (7.6%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |11.2    |
| grep        |11.0    |
| cugrep 0.1  |13.5    |
| cugrep 0.2  |11.2    |


