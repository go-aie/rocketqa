package internal

import (
	"fmt"

	paddle "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
)

// Engine is an inference engine.
type Engine struct {
	predictorPool *PredictorPool
}

func NewEngine(model, params string, maxConcurrency int) *Engine {
	config := paddle.NewConfig()
	config.SetModel(model, params)
	config.EnableMemoryOptim(true) // Enable the memory optimization

	return &Engine{
		predictorPool: NewPredictorPool(config, maxConcurrency),
	}
}

func (e *Engine) Infer(inputs []Tensor) (outputs []Tensor) {
	predictor, put := e.predictorPool.Get()
	defer put()

	inputNames := predictor.GetInputNames()
	if len(inputs) != len(inputNames) {
		panic(fmt.Errorf("inputs mismatch the length of %v", inputNames))
	}

	// Set the inference input.
	for i, name := range predictor.GetInputNames() {
		inputHandle := predictor.GetInputHandle(name)
		inputHandle.Reshape(inputs[i].Shape)
		inputHandle.CopyFromCpu(inputs[i].Data)
	}

	// Run the inference engine.
	predictor.Run()

	// Get the inference output.
	for _, name := range predictor.GetOutputNames() {
		outputHandle := predictor.GetOutputHandle(name)
		outputData := make([]float32, numElements(outputHandle.Shape()))
		outputHandle.CopyToCpu(outputData)

		outputs = append(outputs, Tensor{
			Shape: outputHandle.Shape(),
			Data:  outputData,
		})
	}

	// Clear all temporary tensors to release the memory.
	//
	// See also:
	// - https://github.com/PaddlePaddle/Paddle/issues/43346
	// - https://github.com/PaddlePaddle/PaddleOCR/discussions/6977
	predictor.ClearIntermediateTensor()
	predictor.TryShrinkMemory()

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
	return Tensor{
		Shape: []int32{int32(batchSize), int32(dataSize), 1},
		Data:  flattened,
	}
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}
