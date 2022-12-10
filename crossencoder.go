package rocketqa

import (
	"fmt"

	"github.com/RussellLuo/go-rocketqa/internal"
)

type CrossEncoderConfig struct {
	ModelPath, ParamsPath string
	VocabFile             string
	DoLowerCase           bool
	MaxSeqLength          int
	ForCN                 bool
}

type CrossEncoder struct {
	engine    *internal.Engine
	generator *internal.Generator
}

func NewCrossEncoder(cfg *CrossEncoderConfig) (*CrossEncoder, error) {
	generator, err := internal.NewGenerator(internal.GeneratorConfig{
		VocabFile:    cfg.VocabFile,
		DoLowerCase:  cfg.DoLowerCase,
		MaxSeqLength: cfg.MaxSeqLength,
		ForCN:        cfg.ForCN,
	})
	if err != nil {
		return nil, err
	}

	return &CrossEncoder{
		engine:    internal.NewEngine(cfg.ModelPath, cfg.ParamsPath),
		generator: generator,
	}, nil
}

func (ce *CrossEncoder) Rank(queries, paras, titles []string) ([]float32, error) {
	n := len(queries)
	if n == 0 {
		return nil, nil
	}
	if len(paras) != n {
		return nil, fmt.Errorf("len(paras) does not equal len(queries)")
	}
	if len(titles) > 0 && len(titles) != n {
		return nil, fmt.Errorf("len(titles) does not equal len(queries)")
	}

	var records []internal.Record
	for i := 0; i < n; i++ {
		e := &internal.Example{
			Query: queries[i],
			Para:  paras[i],
		}
		if len(titles) > 0 {
			e.Title = titles[i]
		}
		records = append(records, ce.generator.GenerateCE(e))
	}

	inputs := ce.getInputs(records)
	outputs := ce.engine.Infer(inputs)

	// We only care the first (also the only one) output.
	result := outputs[0]
	// Extract the second column (Assume that joint_training == 0)
	return internal.NewMatrix(result).Col(1), nil
}

func (ce *CrossEncoder) getInputs(records []internal.Record) []internal.Tensor {
	var tokenIDs [][]int64
	var textTypeIDs [][]int64
	var positionIDs [][]int64

	for _, r := range records {
		tokenIDs = append(tokenIDs, r.TokenIDs)
		textTypeIDs = append(textTypeIDs, r.TextTypeIDs)
		positionIDs = append(positionIDs, r.PositionIDs)
	}

	tokenIDs, inputMasks := ce.generator.Pad(tokenIDs)
	textTypeIDs, _ = ce.generator.Pad(textTypeIDs)
	positionIDs, _ = ce.generator.Pad(positionIDs)

	var inputs []internal.Tensor
	inputs = append(inputs, internal.NewInputTensor(tokenIDs))
	inputs = append(inputs, internal.NewInputTensor(textTypeIDs))
	inputs = append(inputs, internal.NewInputTensor(positionIDs))
	inputs = append(inputs, internal.NewInputTensor(inputMasks))
	return inputs
}
