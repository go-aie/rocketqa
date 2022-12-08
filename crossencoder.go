package rocketqa

import "github.com/RussellLuo/go-rocketqa/internal"

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

func (ce *CrossEncoder) Rank(query, para, title string) float32 {
	e := &internal.Example{
		Query: query,
		Para:  para,
		Title: title,
	}
	record := ce.generator.GenerateCE(e)
	inputs := ce.getInputs(record)
	output := ce.engine.Infer(inputs, 0)

	// Extract the element at index 1 (Assume that joint_training == 0)
	return output[1]
}

func (ce *CrossEncoder) getInputs(r internal.Record) []internal.Input {
	var inputs []internal.Input

	inputs = append(inputs, internal.NewInput(r.TokenIDs))
	inputs = append(inputs, internal.NewInput(r.TextTypeIDs))
	inputs = append(inputs, internal.NewInput(r.PositionIDs))
	inputs = append(inputs, internal.NewInput(r.InputMask))

	return inputs
}
