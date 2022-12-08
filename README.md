# Go RocketQA

Go Inference API for [RocketQA][1].


## Installation

1. Install [Paddle Inference Go API][2]
2. Generate [the inference models](cli#save-inference-model)

    ```bash
    $ python3 cli.py save zh_dureader_de_v2 --out-prefix=testdata/zh_dureader_de_v2
    $ python3 cli.py save zh_dureader_ce_v2 --out-prefix=testdata/zh_dureader_ce_v2
    ```
   
3. Install `go-rocketqa`

    ```bash
    $ go get -u github.com/RussellLuo/go-rocketqa
    ```


## Documentation

Checkout the [Godoc][3].


## License

[MIT](LICENSE)


[1]: https://github.com/PaddlePaddle/RocketQA
[2]: https://www.paddlepaddle.org.cn/inference/master/guides/install/go_install.html
[3]: https://pkg.go.dev/github.com/RussellLuo/go-rocketqa
