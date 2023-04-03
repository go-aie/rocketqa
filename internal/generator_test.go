package internal_test

import (
	"testing"

	"github.com/go-aie/rocketqa/internal"
	"github.com/google/go-cmp/cmp"
)

func TestGenerator_GenerateDE(t *testing.T) {
	g, err := internal.NewGenerator(internal.GeneratorConfig{
		VocabFile:         "../testdata/zh_vocab.txt",
		DoLowerCase:       true,
		QueryMaxSeqLength: 32,
		ParaMaxSeqLength:  384,
		ForCN:             true,
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inExample *internal.Example
		wantData  internal.Data
	}{
		{
			inExample: internal.NewExampleFromQuery("你好，世界！"),
			wantData: internal.Data{
				Query: internal.Record{
					TokenIDs:    []int64{1, 226, 170, 4, 203, 280, 12044, 2},
					TextTypeIDs: []int64{0, 0, 0, 0, 0, 0, 0, 0},
					PositionIDs: []int64{0, 1, 2, 3, 4, 5, 6, 7},
				},
				Para: internal.Record{
					TokenIDs:    []int64{1, 17963, 2, 17963, 2},
					TextTypeIDs: []int64{0, 0, 0, 1, 1},
					PositionIDs: []int64{0, 1, 2, 3, 4},
				},
			},
		},
		{
			inExample: internal.NewExampleFromQuery("Hello, World!"),
			wantData: internal.Data{
				Query: internal.Record{
					TokenIDs:    []int64{1, 6368, 30, 4604, 12046, 2},
					TextTypeIDs: []int64{0, 0, 0, 0, 0, 0},
					PositionIDs: []int64{0, 1, 2, 3, 4, 5},
				},
				Para: internal.Record{
					TokenIDs:    []int64{1, 17963, 2, 17963, 2},
					TextTypeIDs: []int64{0, 0, 0, 1, 1},
					PositionIDs: []int64{0, 1, 2, 3, 4},
				},
			},
		},
		{
			inExample: internal.NewExampleFromPara("这是一段较长的文本。", ""),
			wantData: internal.Data{
				Query: internal.Record{
					TokenIDs:    []int64{1, 17963, 2},
					TextTypeIDs: []int64{0, 0, 0},
					PositionIDs: []int64{0, 1, 2},
				},
				Para: internal.Record{
					TokenIDs:    []int64{1, 2, 47, 10, 7, 613, 420, 84, 5, 68, 89, 12043, 2},
					TextTypeIDs: []int64{0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
					PositionIDs: []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				},
			},
		},
		{
			inExample: internal.NewExampleFromPara("This is a long paragraph.", ""),
			wantData: internal.Data{
				Query: internal.Record{
					TokenIDs:    []int64{1, 17963, 2},
					TextTypeIDs: []int64{0, 0, 0},
					PositionIDs: []int64{0, 1, 2},
				},
				Para: internal.Record{
					TokenIDs:    []int64{1, 2, 3730, 11345, 10314, 10684, 10349, 9501, 11366, 42, 2},
					TextTypeIDs: []int64{0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
					PositionIDs: []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
				},
			},
		},
	}
	for _, tt := range tests {
		gotData := g.GenerateDE(tt.inExample)
		if !cmp.Equal(gotData, tt.wantData) {
			diff := cmp.Diff(gotData, tt.wantData)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func TestGenerator_GenerateCE(t *testing.T) {
	g, err := internal.NewGenerator(internal.GeneratorConfig{
		VocabFile:    "../testdata/zh_vocab.txt",
		DoLowerCase:  true,
		MaxSeqLength: 384,
		ForCN:        true,
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inExample  *internal.Example
		wantRecord internal.Record
	}{
		{
			inExample: &internal.Example{
				Query: "你好，世界！",
				Para:  "这是一段较长的文本。",
			},
			wantRecord: internal.Record{
				TokenIDs:    []int64{1, 226, 170, 4, 203, 280, 12044, 2, 47, 10, 7, 613, 420, 84, 5, 68, 89, 12043, 2},
				TextTypeIDs: []int64{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
				PositionIDs: []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
			},
		},
		{
			inExample: &internal.Example{
				Query: "Hello, World!",
				Para:  "This is a long paragraph.",
			},
			wantRecord: internal.Record{
				TokenIDs:    []int64{1, 6368, 30, 4604, 12046, 2, 3730, 11345, 10314, 10684, 10349, 9501, 11366, 42, 2},
				TextTypeIDs: []int64{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
				PositionIDs: []int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
			},
		},
	}
	for _, tt := range tests {
		gotRecord := g.GenerateCE(tt.inExample)
		if !cmp.Equal(gotRecord, tt.wantRecord) {
			diff := cmp.Diff(gotRecord, tt.wantRecord)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}

func TestGenerator_Pad(t *testing.T) {
	g, err := internal.NewGenerator(internal.GeneratorConfig{
		VocabFile:    "../testdata/zh_vocab.txt",
		DoLowerCase:  true,
		MaxSeqLength: 384,
		ForCN:        true,
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inInstances   [][]int64
		wantPadded    [][]int64
		wantInputMask [][]float32
	}{
		{
			inInstances: [][]int64{
				{1, 226, 170, 4, 203, 280, 12044, 2, 47, 10, 7, 613, 420, 84, 5, 68, 89, 12043, 2},
				{1, 6368, 30, 4604, 12046, 2, 3730, 11345, 10314, 10684, 10349, 9501, 11366, 42, 2},
			},
			wantPadded: [][]int64{
				{1, 226, 170, 4, 203, 280, 12044, 2, 47, 10, 7, 613, 420, 84, 5, 68, 89, 12043, 2},
				{1, 6368, 30, 4604, 12046, 2, 3730, 11345, 10314, 10684, 10349, 9501, 11366, 42, 2, 0, 0, 0, 0},
			},
			wantInputMask: [][]float32{
				{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
				{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},
			},
		},
	}
	for _, tt := range tests {
		gotPadded, gotInputMask := g.Pad(tt.inInstances)
		if !cmp.Equal(gotPadded, tt.wantPadded) {
			diff := cmp.Diff(gotPadded, tt.wantPadded)
			t.Errorf("Padded (Want - Got): %s", diff)
		}
		if !cmp.Equal(gotInputMask, tt.wantInputMask) {
			diff := cmp.Diff(gotInputMask, tt.wantInputMask)
			t.Errorf("InputMask (Want - Got): %s", diff)
		}
	}
}
