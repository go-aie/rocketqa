# go-rocketqa

[![Go Reference](https://pkg.go.dev/badge/RussellLuo/go-rocketqa/vulndb.svg)][2]

Go Inference API for [RocketQA][1].

Checkout the [documentation][2] for more information.


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


## License

[MIT](LICENSE)


[1]: https://github.com/PaddlePaddle/RocketQA
[2]: https://pkg.go.dev/github.com/RussellLuo/go-rocketqa
[3]: https://www.paddlepaddle.org.cn/inference/master/guides/install/go_install.html
