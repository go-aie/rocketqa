package rocketqa_test

import (
	"testing"

	"github.com/go-aie/rocketqa"
	"github.com/google/go-cmp/cmp"
)

func TestCrossEncoder_Rank(t *testing.T) {
	ce, err := newCrossEncoder(1)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inQPTs     rocketqa.QPTs
		wantScores []float32
	}{
		{
			inQPTs: rocketqa.QPTs{
				{
					Query: "你好，世界！",
					Para:  "这是一段较长的文本。",
				},
				{
					Query: "Hello, World!",
					Para:  "This is a long paragraph.",
				},
			},
			wantScores: []float32{
				0.05041640,
				0.07384859,
			},
		},
	}
	for _, tt := range tests {
		gotScores, _ := ce.Rank(tt.inQPTs.Q(), tt.inQPTs.P(), tt.inQPTs.T())
		if !cmp.Equal(gotScores, tt.wantScores) {
			diff := cmp.Diff(gotScores, tt.wantScores)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func BenchmarkCrossEncoder_Rank(b *testing.B) {
	inQPTs := rocketqa.QPTs{
		{
			Query: "你好，世界！",
			Para:  "这是一段较长的文本。",
		},
		{
			Query: "Hello, World!",
			Para:  "This is a long paragraph.",
		},
	}

	tests := []struct {
		name           string
		maxConcurrency int
	}{
		{"C-1", 1},
		{"C-2", 2},
		{"C-4", 4},
		{"C-8", 8},
		{"C-16", 16},
		{"C-32", 32},
		{"C-64", 64},
	}
	for _, tt := range tests {
		ce, err := newCrossEncoder(tt.maxConcurrency)
		if err != nil {
			b.Fatal(err)
		}

		b.Run(tt.name, func(b *testing.B) {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					_, _ = ce.Rank(inQPTs.Q(), inQPTs.P(), inQPTs.T())
				}
			})
		})
	}
}

func newCrossEncoder(maxConcurrency int) (*rocketqa.CrossEncoder, error) {
	return rocketqa.NewCrossEncoder(&rocketqa.CrossEncoderConfig{
		ModelPath:      "./testdata/zh_dureader_ce_v2.pdmodel",
		ParamsPath:     "./testdata/zh_dureader_ce_v2.pdiparams",
		VocabFile:      "./testdata/zh_vocab.txt",
		DoLowerCase:    true,
		MaxSeqLength:   384,
		ForCN:          true,
		MaxConcurrency: maxConcurrency,
	})
}
