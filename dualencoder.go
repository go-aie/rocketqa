package rocketqa

import (
	"fmt"

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

func (de *DualEncoder) EncodeQuery(queries []string) []Vector {
	if len(queries) == 0 {
		return nil
	}

	var dataSet []internal.Data
	for _, query := range queries {
		dataSet = append(dataSet, de.generator.GenerateDE(internal.NewExampleFromQuery(query)))
	}

	inputs := de.getInputs(dataSet)
	outputs := de.engine.Infer(inputs)

	result := outputs[0] // 0: q_rep, 1: p_rep
	m := internal.NewMatrix(result)
	return newVectors(m.Rows())
}

func (de *DualEncoder) EncodePara(paras, titles []string) ([]Vector, error) {
	n := len(paras)
	if n == 0 {
		return nil, nil
	}
	if len(titles) != n {
		return nil, fmt.Errorf("len(titles) does not equal len(paras)")
	}

	var dataSet []internal.Data
	for i := 0; i < n; i++ {
		title := ""
		if len(titles) > 0 {
			title = titles[i]
		}
		dataSet = append(dataSet, de.generator.GenerateDE(internal.NewExampleFromPara(paras[i], title)))
	}

	inputs := de.getInputs(dataSet)
	outputs := de.engine.Infer(inputs)

	result := outputs[1] // 0: q_rep, 1: p_rep
	m := internal.NewMatrix(result)
	return newVectors(m.Rows()), nil
}

func (de *DualEncoder) getInputs(dataSet []internal.Data) []internal.Tensor {
	var queryTokenIDs [][]int64
	var queryTextTypeIDs [][]int64
	var queryPositionIDs [][]int64
	var paraTokenIDs [][]int64
	var paraTextTypeIDs [][]int64
	var paraPositionIDs [][]int64

	for _, d := range dataSet {
		queryTokenIDs = append(queryTokenIDs, d.Query.TokenIDs)
		queryTextTypeIDs = append(queryTextTypeIDs, d.Query.TextTypeIDs)
		queryPositionIDs = append(queryPositionIDs, d.Query.PositionIDs)
		paraTokenIDs = append(paraTokenIDs, d.Para.TokenIDs)
		paraTextTypeIDs = append(paraTextTypeIDs, d.Para.TextTypeIDs)
		paraPositionIDs = append(paraPositionIDs, d.Para.PositionIDs)
	}

	queryTokenIDs, queryInputMasks := de.generator.Pad(queryTokenIDs)
	queryTextTypeIDs, _ = de.generator.Pad(queryTextTypeIDs)
	queryPositionIDs, _ = de.generator.Pad(queryPositionIDs)
	paraTokenIDs, paraInputMasks := de.generator.Pad(paraTokenIDs)
	paraTextTypeIDs, _ = de.generator.Pad(paraTextTypeIDs)
	paraPositionIDs, _ = de.generator.Pad(paraPositionIDs)

	return []internal.Tensor{
		internal.NewInputTensor(queryTokenIDs),
		internal.NewInputTensor(queryTextTypeIDs),
		internal.NewInputTensor(queryPositionIDs),
		internal.NewInputTensor(queryInputMasks),
		internal.NewInputTensor(paraTokenIDs),
		internal.NewInputTensor(paraTextTypeIDs),
		internal.NewInputTensor(paraPositionIDs),
		internal.NewInputTensor(paraInputMasks),
	}
}

type Vector []float32

func (v Vector) Norm() Vector {
	m := internal.NewMatrix(internal.Tensor{Shape: []int32{1, int32(len(v))}, Data: []float32(v)})
	return m.Norm().RawData()
}

func (v Vector) ToFloat64() []float64 {
	return internal.Float32To64(v)
}

func newVectors(value [][]float32) []Vector {
	var vectors []Vector
	for _, v := range value {
		vectors = append(vectors, v)
	}
	return vectors
}
