package internal

import (
	"bufio"
	"os"
	"strings"
	"unicode"
)

type Tokenizer struct {
	vocab
	invVocab

	wordpiece   *wordpieceTokenizer
	doLowerCase bool
}

func NewTokenizer(vocabFile string, doLowerCase bool) (*Tokenizer, error) {
	v, err := newVocab(vocabFile)
	if err != nil {
		return nil, err
	}
	iv := newInvVocab(v)
	return &Tokenizer{
		vocab:       v,
		invVocab:    iv,
		wordpiece:   newWordpieceTokenizer(v),
		doLowerCase: doLowerCase,
	}, nil
}

func (t *Tokenizer) Tokenize(text string) []string {
	text = t.cleanText(text)
	text = t.tokenizeChineseAndPunctuation(text)

	var result []string
	for _, token := range strings.Split(text, " ") {
		for _, subToken := range t.wordpiece.Tokenize(token) {
			result = append(result, subToken)
		}
	}
	return result
}

// cleanText performs invalid character removal and whitespace cleanup on text.
func (t *Tokenizer) cleanText(text string) string {
	var output []string
	for _, r := range text {
		s := string(r)
		if unicode.IsControl(r) {
			continue
		}
		if unicode.IsSpace(r) {
			output = append(output, " ")
		} else {
			output = append(output, s)
		}
	}
	return strings.Join(output, "")
}

// tokenizeChineseAndPunctuation adds whitespace around any Chinese and
// punctuation character.
func (t *Tokenizer) tokenizeChineseAndPunctuation(text string) string {
	var output []string
	for _, r := range text {
		s := string(r)
		if unicode.Is(unicode.Han, r) || unicode.IsPunct(r) {
			s = " " + s + " "
		}
		if t.doLowerCase {
			s = strings.ToLower(s)
		}
		output = append(output, s)
	}
	return strings.Join(output, "")
}

type wordpieceTokenizer struct {
	vocab                vocab
	unkToken             string
	maxInputCharsPerWord int
}

func newWordpieceTokenizer(vocab vocab) *wordpieceTokenizer {
	return &wordpieceTokenizer{
		vocab:                vocab,
		unkToken:             "[UNK]",
		maxInputCharsPerWord: 100,
	}
}

// Tokenize tokenizes a piece of text into its word pieces.
//
// This uses a greedy longest-match-first algorithm to perform tokenization
// using the given vocabulary.
//
// Example:
//
//	input = "unaffable"
//	output = ["un", "##aff", "##able"]
func (t *wordpieceTokenizer) Tokenize(text string) []string {
	var result []string

	chars := strings.Split(text, "")

	if len(chars) > t.maxInputCharsPerWord {
		result = append(result, t.unkToken)
		return result
	}

	isBad := false
	start := 0
	var subTokens []string

	for start < len(chars) {
		end := len(chars)
		var curSubstr string

		for start < end {
			substr := strings.Join(chars[start:end], "")
			if start > 0 {
				substr = "##" + substr
			}
			if _, ok := t.vocab[substr]; ok {
				curSubstr = substr
				break
			}
			end -= 1
		}

		if curSubstr == "" {
			isBad = true
			break
		}

		subTokens = append(subTokens, curSubstr)
		start = end
	}

	if isBad {
		result = append(result, t.unkToken)
	} else {
		result = append(result, subTokens...)
	}

	return result
}

type vocab map[string]int64

func newVocab(filename string) (vocab, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	v := make(map[string]int64)
	scanner := bufio.NewScanner(file)
	for i := 0; scanner.Scan(); i++ {
		v[scanner.Text()] = int64(i)
	}

	return v, scanner.Err()
}

func (v vocab) TokensToIDs(tokens []string) (ids []int64) {
	for _, token := range tokens {
		if id, ok := v[token]; ok {
			ids = append(ids, id)
		}
	}
	return
}

type invVocab map[int64]string

func newInvVocab(v vocab) invVocab {
	iv := make(map[int64]string)
	for token, id := range v {
		iv[id] = token
	}
	return iv
}

func (iv invVocab) IDsToTokens(ids []int64) (tokens []string) {
	for _, id := range ids {
		if token, ok := iv[id]; ok {
			tokens = append(tokens, token)
		}
	}
	return
}
