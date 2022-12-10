package rocketqa_test

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/RussellLuo/go-rocketqa"
	"github.com/google/go-cmp/cmp"
)

func TestDualEncoder_EncodeQuery(t *testing.T) {
	de := newDualEncoder(t)

	tests := []struct {
		inType    string
		inQueries []string
	}{
		{
			inType: "query",
			inQueries: []string{
				"你好，世界！",
				"Hello, World!",
			},
		},
	}
	for _, tt := range tests {
		for i, vector := range de.EncodeQuery(tt.inQueries) {
			var gotEmb []string
			for _, v := range vector {
				gotEmb = append(gotEmb, fmt.Sprintf("%.8f", v))
			}

			wantEmb := getEmbedding(t, tt.inType, tt.inQueries[i])
			if !cmp.Equal(gotEmb, wantEmb) {
				diff := cmp.Diff(gotEmb, wantEmb)
				t.Errorf("Want - Got: %s", diff)
			}
		}
	}
}

func TestDualEncoder_EncodePara(t *testing.T) {
	de := newDualEncoder(t)

	tests := []struct {
		inType   string
		inParas  []string
		inTitles []string
	}{
		{
			inType: "para",
			inParas: []string{
				"这是一段较长的文本。",
				"This is a long paragraph.",
			},
		},
	}
	for _, tt := range tests {
		vectors, _ := de.EncodePara(tt.inParas, tt.inTitles)
		for i, vector := range vectors {
			var gotEmb []string
			for _, v := range vector {
				gotEmb = append(gotEmb, fmt.Sprintf("%.8f", v))
			}

			wantEmb := getEmbedding(t, tt.inType, tt.inParas[i])
			if !cmp.Equal(gotEmb, wantEmb) {
				diff := cmp.Diff(gotEmb, wantEmb)
				t.Errorf("Want - Got: %s", diff)
			}
		}
	}
}

func newDualEncoder(t *testing.T) *rocketqa.DualEncoder {
	de, err := rocketqa.NewDualEncoder(&rocketqa.DualEncoderConfig{
		ModelPath:         "./testdata/zh_dureader_de_v2.pdmodel",
		ParamsPath:        "./testdata/zh_dureader_de_v2.pdiparams",
		VocabFile:         "./testdata/zh_vocab.txt",
		DoLowerCase:       true,
		QueryMaxSeqLength: 32,
		ParaMaxSeqLength:  384,
		ForCN:             true,
	})
	if err != nil {
		t.Fatal(err)
	}
	return de
}

func getEmbedding(t *testing.T, typ string, text string) []string {
	content, err := os.ReadFile("./testdata/embeddings.json")
	if err != nil {
		t.Error(err)
	}

	var m map[string]map[string][]string
	if err := json.Unmarshal(content, &m); err != nil {
		t.Error(err)
	}

	return m[typ][text]
}
