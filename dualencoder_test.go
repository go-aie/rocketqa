package rocketqa_test

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/go-aie/rocketqa"
	"github.com/google/go-cmp/cmp"
)

func TestDualEncoder_EncodeQuery(t *testing.T) {
	de, err := newDualEncoder(1)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inType string
		inQPTs rocketqa.QPTs
	}{
		{
			inType: "query",
			inQPTs: rocketqa.QPTs{
				{
					Query: "你好，世界！",
				},
				{
					Query: "Hello, World!",
				},
			},
		},
	}
	for _, tt := range tests {
		for i, vector := range de.EncodeQuery(tt.inQPTs.Q()) {
			var gotEmb []string
			for _, v := range vector {
				gotEmb = append(gotEmb, fmt.Sprintf("%.8f", v))
			}

			wantEmb := getEmbedding(t, tt.inType, tt.inQPTs[i].Query)
			if !cmp.Equal(gotEmb, wantEmb) {
				diff := cmp.Diff(gotEmb, wantEmb)
				t.Errorf("Want - Got: %s", diff)
			}
		}
	}
}

func TestDualEncoder_EncodePara(t *testing.T) {
	de, err := newDualEncoder(1)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inType string
		inQPTs rocketqa.QPTs
	}{
		{
			inType: "para",
			inQPTs: rocketqa.QPTs{
				{
					Para: "这是一段较长的文本。",
				},
				{
					Para: "This is a long paragraph.",
				},
			},
		},
	}
	for _, tt := range tests {
		vectors, _ := de.EncodePara(tt.inQPTs.P(), tt.inQPTs.T())
		for i, vector := range vectors {
			var gotEmb []string
			for _, v := range vector {
				gotEmb = append(gotEmb, fmt.Sprintf("%.8f", v))
			}

			wantEmb := getEmbedding(t, tt.inType, tt.inQPTs[i].Para)
			if !cmp.Equal(gotEmb, wantEmb) {
				diff := cmp.Diff(gotEmb, wantEmb)
				t.Errorf("Want - Got: %s", diff)
			}
		}
	}
}

func BenchmarkDualEncoder_EncodeQuery(b *testing.B) {
	inQPTs := rocketqa.QPTs{
		{
			Query: "你好，世界！",
		},
		{
			Query: "Hello, World!",
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
		de, err := newDualEncoder(tt.maxConcurrency)
		if err != nil {
			b.Fatal(err)
		}

		b.Run(tt.name, func(b *testing.B) {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					_ = de.EncodeQuery(inQPTs.Q())
				}
			})
		})
	}
}

func BenchmarkDualEncoder_EncodePara(b *testing.B) {
	inQPTs := rocketqa.QPTs{
		{
			Para: "这是一段较长的文本。",
		},
		{
			Para: "This is a long paragraph.",
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
		de, err := newDualEncoder(tt.maxConcurrency)
		if err != nil {
			b.Fatal(err)
		}

		b.Run(tt.name, func(b *testing.B) {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					_, _ = de.EncodePara(inQPTs.P(), inQPTs.T())
				}
			})
		})
	}
}

func newDualEncoder(maxConcurrency int) (*rocketqa.DualEncoder, error) {
	return rocketqa.NewDualEncoder(&rocketqa.DualEncoderConfig{
		ModelPath:         "./testdata/zh_dureader_de_v2.pdmodel",
		ParamsPath:        "./testdata/zh_dureader_de_v2.pdiparams",
		VocabFile:         "./testdata/zh_vocab.txt",
		DoLowerCase:       true,
		QueryMaxSeqLength: 32,
		ParaMaxSeqLength:  384,
		ForCN:             true,
		MaxConcurrency:    maxConcurrency,
	})
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
