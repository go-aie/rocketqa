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

func (e *Engine) Infer(inputs []Input, outputIndex int) Output {
	inputNames := e.predictor.GetInputNames()
	if len(inputs) != len(inputNames) {
		panic(fmt.Errorf("inputs mismatch the length of %v", inputNames))
	}

	// Set the inference input.
	for i, name := range e.predictor.GetInputNames() {
		inputHandle := e.predictor.GetInputHandle(name)
		inputHandle.Reshape(inputs[i].Shape)
		inputHandle.CopyFromCpu(inputs[i].Value)
	}

	// Run the inference engine.
	e.predictor.Run()

	// Get the inference output.
	outputNames := e.predictor.GetOutputNames()
	outputHandle := e.predictor.GetOutputHandle(outputNames[outputIndex])
	outputData := make([]float32, numElements(outputHandle.Shape()))
	outputHandle.CopyToCpu(outputData)

	return outputData
}

type Input struct {
	Value interface{}
	Shape []int32
}

func NewInput(value interface{}) Input {
	switch t := value.(type) {
	case []int64:
		return Input{Value: value, Shape: []int32{1, int32(len(t)), 1}}
	case []float32:
		return Input{Value: value, Shape: []int32{1, int32(len(t)), 1}}
	default:
		panic(fmt.Errorf("unsupported type: %T", t))
	}
}

type Output []float32

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}
