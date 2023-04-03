module github.com/go-aie/rocketqa/examples/es

go 1.19

replace github.com/go-aie/rocketqa => ../../

require (
	github.com/elastic/go-elasticsearch/v8 v8.5.0
	github.com/go-aie/rocketqa v0.0.0-00010101000000-000000000000
	golang.org/x/exp v0.0.0-20230212135524-a684f29349b6
)

require (
	github.com/elastic/elastic-transport-go/v8 v8.0.0-20211216131617-bbee439d559c // indirect
	github.com/go-aie/paddle v0.0.0-20230213030711-67518e191570 // indirect
	github.com/jackc/puddle/v2 v2.2.0 // indirect
	github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi v0.0.0-20221116023434-3fa7a736e325 // indirect
	golang.org/x/sync v0.1.0 // indirect
	gonum.org/v1/gonum v0.12.0 // indirect
)
