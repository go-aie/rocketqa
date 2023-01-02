# go-rocketqa

[![Go Reference](https://pkg.go.dev/badge/RussellLuo/go-rocketqa/vulndb.svg)][2]

Go Inference API for [RocketQA][1].


## Installation

1. Install [Paddle Inference Go API][3]
2. Generate [the inference models](cli/README.md#save-inference-model)

    ```bash
    $ python3 cli/cli.py save zh_dureader_de_v2 --out-path=testdata/zh_dureader_de_v2
    $ python3 cli/cli.py save zh_dureader_ce_v2 --out-path=testdata/zh_dureader_ce_v2
    ```
   
3. Install `go-rocketqa`

    ```bash
    $ go get -u github.com/RussellLuo/go-rocketqa
    ```


## Documentation

Check out the [documentation][2].


## Testing and Benchmarking

Run tests:

```bash
$ go test -v -race | grep -E 'go|Test'
=== RUN   TestCrossEncoder_Rank
--- PASS: TestCrossEncoder_Rank (0.69s)
=== RUN   TestDualEncoder_EncodeQuery
--- PASS: TestDualEncoder_EncodeQuery (0.97s)
=== RUN   TestDualEncoder_EncodePara
--- PASS: TestDualEncoder_EncodePara (0.77s)
ok  	github.com/RussellLuo/go-rocketqa	3.094s
```

Run benchmarks:

```bash
$ go test -bench=. -benchmem | grep -E 'go|Benchmark'
goos: darwin
goarch: arm64
pkg: github.com/RussellLuo/go-rocketqa
BenchmarkCrossEncoder_Rank/C-1-10      	      12	  95231254 ns/op	   21522 B/op	     557 allocs/op
BenchmarkCrossEncoder_Rank/C-2-10      	      24	  48904571 ns/op	   21224 B/op	     555 allocs/op
BenchmarkCrossEncoder_Rank/C-4-10      	      49	  25722997 ns/op	   21154 B/op	     554 allocs/op
BenchmarkCrossEncoder_Rank/C-8-10      	      87	  13775692 ns/op	   21003 B/op	     557 allocs/op
BenchmarkCrossEncoder_Rank/C-16-10     	      96	  12942040 ns/op	   20965 B/op	     555 allocs/op
BenchmarkCrossEncoder_Rank/C-32-10     	      96	  12374102 ns/op	   20869 B/op	     554 allocs/op
BenchmarkCrossEncoder_Rank/C-64-10     	      96	  12106736 ns/op	   20867 B/op	     554 allocs/op
BenchmarkDualEncoder_EncodeQuery/C-1-10         	      13	  87274330 ns/op	   91977 B/op	     463 allocs/op
BenchmarkDualEncoder_EncodeQuery/C-2-10         	      24	  46361825 ns/op	   92004 B/op	     464 allocs/op
BenchmarkDualEncoder_EncodeQuery/C-4-10         	      42	  24531967 ns/op	   92054 B/op	     465 allocs/op
BenchmarkDualEncoder_EncodeQuery/C-8-10         	      84	  13078837 ns/op	   91868 B/op	     463 allocs/op
BenchmarkDualEncoder_EncodeQuery/C-16-10        	      92	  12734853 ns/op	   91831 B/op	     460 allocs/op
BenchmarkDualEncoder_EncodeQuery/C-32-10        	      92	  12949048 ns/op	   91796 B/op	     462 allocs/op
BenchmarkDualEncoder_EncodeQuery/C-64-10        	      96	  12571607 ns/op	   91839 B/op	     462 allocs/op
BenchmarkDualEncoder_EncodePara/C-1-10          	      10	 102307329 ns/op	   97323 B/op	     600 allocs/op
BenchmarkDualEncoder_EncodePara/C-2-10          	      22	  52587379 ns/op	   97290 B/op	     599 allocs/op
BenchmarkDualEncoder_EncodePara/C-4-10          	      40	  27110265 ns/op	   97255 B/op	     597 allocs/op
BenchmarkDualEncoder_EncodePara/C-8-10          	      75	  15437987 ns/op	   97171 B/op	     597 allocs/op
BenchmarkDualEncoder_EncodePara/C-16-10         	      84	  14117255 ns/op	   97148 B/op	     597 allocs/op
BenchmarkDualEncoder_EncodePara/C-32-10         	      81	  14174034 ns/op	   97116 B/op	     596 allocs/op
BenchmarkDualEncoder_EncodePara/C-64-10         	      80	  14531767 ns/op	   97123 B/op	     597 allocs/op
ok  	github.com/RussellLuo/go-rocketqa	55.900s
```

## Known Issues

### BLAS error

When using a high `MaxConcurrency` (e.g. running benchmarks), sometimes you will get a BLAS error:

```
BLAS : Program is Terminated. Because you tried to allocate too many memory regions.
```

#### Solution

Set OpenBLAS to use a single thread:

```bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

See also:
- https://github.com/autogluon/autogluon/issues/1020
- https://groups.google.com/g/cmu-openface/c/CwVFyKJPWP4


## License

[MIT](LICENSE)


[1]: https://github.com/PaddlePaddle/RocketQA
[2]: https://pkg.go.dev/github.com/RussellLuo/go-rocketqa
[3]: https://www.paddlepaddle.org.cn/inference/master/guides/install/go_install.html
