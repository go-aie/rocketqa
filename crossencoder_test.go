package rocketqa_test

import (
	"testing"

	"github.com/RussellLuo/go-rocketqa"
	"github.com/google/go-cmp/cmp"
)

func TestCrossEncoder_Rank(t *testing.T) {
	ce, err := rocketqa.NewCrossEncoder(&rocketqa.CrossEncoderConfig{
		ModelPath:    "./testdata/zh_dureader_ce_v2.pdmodel",
		ParamsPath:   "./testdata/zh_dureader_ce_v2.pdiparams",
		VocabFile:    "./testdata/zh_vocab.txt",
		DoLowerCase:  true,
		MaxSeqLength: 384,
		ForCN:        true,
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inQueries  []string
		inParas    []string
		inTitles   []string
		wantScores []float32
	}{
		{
			inQueries: []string{
				"你好，世界！",
				"Hello, World!",
			},
			inParas: []string{
				"这是一段较长的文本。",
				"This is a long paragraph.",
			},
			wantScores: []float32{
				0.05041640,
				0.07384859,
			},
		},
	}
	for _, tt := range tests {
		gotScores, _ := ce.Rank(tt.inQueries, tt.inParas, tt.inTitles)
		if !cmp.Equal(gotScores, tt.wantScores) {
			diff := cmp.Diff(gotScores, tt.wantScores)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
