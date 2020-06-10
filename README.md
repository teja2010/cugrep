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

`i*on` : 6296087 (33%)

PATTERN `s*a`  : 14534739 (76%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |11.2    |
| grep        |11.0    |
| cugrep 0.1  |13.5    |
| cugrep 0.2  |11.2    |
| cugrep 0.3  |11.2    |

----

PATTERN `big`  : 27830 (0.14%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.52    |
| grep        |0.76    |
| cugrep      |1.14    |

drive: 22263 (0.11%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.48    |
| grep        |0.66    |
| cugrep      |1.15    |

ordinary : 21270 (0.11%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.54    |
| grep        |0.55    |
| cugrep      |1.13    |

----

can : 384220 (2%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.78    |
| grep        |1.76    |
| cugrep      |1.68    |

was : 1418654 (7.4%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |1.10    |
| grep        |1.48    |
| cugrep      |2.31    |


her : 2285987 (11%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |1.78    |
| grep        |1.88    |
| cugrep      |2.92    |

--------

american & british

`colou*r` :  34753 (0.18%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.58    |
| grep        |1.15    |
| cugrep      |1.08    |

`lite*re*`
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.91    |
| grep        |1.79    |
| cugrep      |1.16    |

`honou*r` : 41035 (0.21%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.77    |
| grep        |1.20    |
| cugrep      |1.11    |

---------

| len var                  | time                      |
|:-------------------------|:--------------------------|
| 50  (0 , 12, 25, 38)     | 1.084,1.081, 1.131, 1.106 |
| 


--------

wc : 19058252

common : 43231 (0.22)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.60    |
| grep        |0.95    |
| cugrep      |1.21    |

extraordinary : 9027 (0.04%)
| prog        | time   |
|:------------|:-------|
| ripgrep     |0.63    |
| grep        |0.46    |
| cugrep      |1.11    |




