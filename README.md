# cugrep

grep in cuda.
An exercise in understanding CUDA and Regex.


----
wc : 19058252

`fo*s` : 40506 (1.8%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |1.95    |
| grep        |3.64    |
| cugrep 0.1  |2.42    |
| cugrep 0.2  |0.84    |  find offsets in kernel
| cugrep 0.3  |0.87    |  move nfa to shared mem



PATTERN `ss*i` : 1945310 (11%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |1.65    |
| grep        |3.99    |
| cugrep 0.1  |4.17    |
| cugrep 0.2  |2.41    |
| cugrep 0.3  |2.40    |

PATTERN `s*a`  : 14534739 (76%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |11.2    |
| grep        |11.0    |
| cugrep 0.1  |13.5    |
| cugrep 0.2  |11.2    |
| cugrep 0.3  |11.2    |



