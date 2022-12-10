package internal

import (
	"fmt"

	paddle "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
)

// Engine is an inference engine.
type Engine struct {
	predictor *paddle.Predictor
}

func NewEngine(model, params string) *Engine {
	config := paddle.NewConfig()
	config.SetModel(model, params)
	return &Engine{
		predictor: paddle.NewPredictor(config),
	}
}

func (e *Engine) Infer(inputs []Tensor) (outputs []Tensor) {
	inputNames := e.predictor.GetInputNames()
	if len(inputs) != len(inputNames) {
		panic(fmt.Errorf("inputs mismatch the length of %v", inputNames))
	}

	// Set the inference input.
	for i, name := range e.predictor.GetInputNames() {
		inputHandle := e.predictor.GetInputHandle(name)
		inputHandle.Reshape(inputs[i].Shape)
		inputHandle.CopyFromCpu(inputs[i].Data)
	}

	// Run the inference engine.
	e.predictor.Run()

	// Get the inference output.
	for _, name := range e.predictor.GetOutputNames() {
		outputHandle := e.predictor.GetOutputHandle(name)
		outputData := make([]float32, numElements(outputHandle.Shape()))
		outputHandle.CopyToCpu(outputData)

		outputs = append(outputs, Tensor{Shape: outputHandle.Shape(), Data: outputData})
	}

	return
}

type Tensor struct {
	Shape []int32
	Data  interface{}
}

func NewInputTensor[E any](value [][]E) Tensor {
	if len(value) == 0 {
		return Tensor{}
	}

	var flattened []E
	for _, d := range value {
		flattened = append(flattened, d...)
	}

	batchSize, dataSize := len(value), len(value[0])
	return Tensor{Shape: []int32{int32(batchSize), int32(dataSize), 1}, Data: flattened}
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}
