# Source and license notices

- `datasets/training/fd_torch.py`, `datasets/kimberlina/fd_torch.py`, and
  `benchmarks/marmousi/vendor/fd_torch.py` preserve the FD10 solver from
  [DeepWave-KAUST/GenWP-pub](https://github.com/DeepWave-KAUST/GenWP-pub),
  Shijun Cheng and Tariq Alkhalifah. The upstream project uses the MIT license;
  its original MIT license text and attribution are retained in
  [`LICENSE_GENWP.txt`](LICENSE_GENWP.txt).
  Other adapted training/data-conversion files should be checked against that
  upstream repository before final publication.
- NeuralOperator is **not** redistributed here. Obtain its source at the
  recorded commit and follow its own license terms.
- The four Red Sea prepared arrays, three Pluto prepared velocity arrays, and
  Marmousi and Kimberlina preparation instructions derive from previously
  published geological datasets. The study code license does not override any
  original data-source conditions. Review the corresponding dataset terms and
  attribution before publishing the included binary velocity arrays.
