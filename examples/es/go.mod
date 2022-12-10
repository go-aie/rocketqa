module github.com/RussellLuo/go-rocketqa/examples/es

go 1.19

replace github.com/RussellLuo/go-rocketqa => ../../

require (
	github.com/RussellLuo/go-rocketqa v0.0.0-00010101000000-000000000000
	github.com/elastic/go-elasticsearch/v8 v8.5.0
	golang.org/x/exp v0.0.0-20221207211629-99ab8fa1c11f
)

require (
	github.com/elastic/elastic-transport-go/v8 v8.0.0-20211216131617-bbee439d559c // indirect
	github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi v0.0.0-20221116023434-3fa7a736e325 // indirect
	gonum.org/v1/gonum v0.12.0 // indirect
)
