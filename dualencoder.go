package rocketqa

import (
	"math"

	"github.com/RussellLuo/go-rocketqa/internal"
)

type DualEncoderConfig struct {
	ModelPath, ParamsPath string
	VocabFile             string
	DoLowerCase           bool
	QueryMaxSeqLength     int
	ParaMaxSeqLength      int
	ForCN                 bool
}

type DualEncoder struct {
	engine    *internal.Engine
	generator *internal.Generator
}

func NewDualEncoder(cfg *DualEncoderConfig) (*DualEncoder, error) {
	generator, err := internal.NewGenerator(internal.GeneratorConfig{
		VocabFile:         cfg.VocabFile,
		DoLowerCase:       cfg.DoLowerCase,
		QueryMaxSeqLength: cfg.QueryMaxSeqLength,
		ParaMaxSeqLength:  cfg.ParaMaxSeqLength,
		ForCN:             cfg.ForCN,
	})
	if err != nil {
		return nil, err
	}

	return &DualEncoder{
		engine:    internal.NewEngine(cfg.ModelPath, cfg.ParamsPath),
		generator: generator,
	}, nil
}

func (de *DualEncoder) EncodeQuery(query string) Vector {
	data := de.generator.GenerateDE(internal.NewExampleFromQuery(query))
	inputs := de.getInputs(data)
	output := de.engine.Infer(inputs, 0)
	return Vector(output)
}

func (de *DualEncoder) EncodePara(para, title string) Vector {
	data := de.generator.GenerateDE(internal.NewExampleFromPara(para, title))
	inputs := de.getInputs(data)
	output := de.engine.Infer(inputs, 1)
	return Vector(output)
}

func (de *DualEncoder) getInputs(data internal.Data) []internal.Input {
	var inputs []internal.Input

	inputs = append(inputs, internal.NewInput(data.Query.TokenIDs))
	inputs = append(inputs, internal.NewInput(data.Query.TextTypeIDs))
	inputs = append(inputs, internal.NewInput(data.Query.PositionIDs))
	inputs = append(inputs, internal.NewInput(data.Query.InputMask))

	inputs = append(inputs, internal.NewInput(data.Para.TokenIDs))
	inputs = append(inputs, internal.NewInput(data.Para.TextTypeIDs))
	inputs = append(inputs, internal.NewInput(data.Para.PositionIDs))
	inputs = append(inputs, internal.NewInput(data.Para.InputMask))

	return inputs
}

type Vector []float32

func (v Vector) Unitize() Vector {
	var quadraticSum float64
	for _, value := range v {
		quadraticSum += float64(value * value)
	}
	length := math.Sqrt(quadraticSum)

	var unitized []float32
	for _, value := range v {
		unitized = append(unitized, float32(float64(value)/length))
	}
	return unitized
}

func (v Vector) ToFloat64() []float64 {
	var result []float64
	for _, value := range v {
		result = append(result, float64(value))
	}
	return result
}
