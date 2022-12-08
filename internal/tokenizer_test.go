package internal_test

import (
	"testing"

	"github.com/RussellLuo/go-rocketqa/internal"
	"github.com/google/go-cmp/cmp"
)

func TestTokenizer_Tokenize(t *testing.T) {
	tokenizer, err := internal.NewTokenizer("../testdata/zh_vocab.txt", true)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		inText     string
		wantTokens []string
	}{
		{
			inText:     "你好，世界！",
			wantTokens: []string{"你", "好", "，", "世", "界", "！"},
		},
		{
			inText:     "Hello, World!",
			wantTokens: []string{"hello", ",", "world", "!"},
		},
		{
			inText:     "这是一段较长的文本。",
			wantTokens: []string{"这", "是", "一", "段", "较", "长", "的", "文", "本", "。"},
		},
		{
			inText:     "Thisisalongparagraph.",
			wantTokens: []string{"this", "##isa", "##lon", "##gp", "##ara", "##g", "##raph", "."},
		},
	}
	for _, tt := range tests {
		gotTokens := tokenizer.Tokenize(tt.inText)
		if !cmp.Equal(gotTokens, tt.wantTokens) {
			diff := cmp.Diff(gotTokens, tt.wantTokens)
			t.Errorf("Want - Got: %s", diff)
		}
	}
}
