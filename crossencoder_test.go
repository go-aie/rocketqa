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
		inQuery  string
		inPara   string
		wantProb float32
	}{
		{
			inQuery:  "你好，世界！",
			inPara:   "这是一段较长的文本。",
			wantProb: 0.05041640,
		},
		{
			inQuery:  "Hello, World!",
			inPara:   "This is a long paragraph.",
			wantProb: 0.07384859,
		},
	}
	for _, tt := range tests {
		gotProb := ce.Rank(tt.inQuery, tt.inPara, "")
		if !cmp.Equal(gotProb, tt.wantProb) {
			diff := cmp.Diff(gotProb, tt.wantProb)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
