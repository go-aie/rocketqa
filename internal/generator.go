package internal

import (
	"strings"
)

type Example struct {
	Query string
	Title string
	Para  string
}

func NewExampleFromQuery(query string) *Example {
	return &Example{
		Query: query,
		Title: "-",
		Para:  "-",
	}
}

func NewExampleFromPara(para, title string) *Example {
	return &Example{
		Query: "-",
		Title: title,
		Para:  para,
	}
}

func (e *Example) Clean() {
	e.Query = removeAllSpaces(e.Query)
	e.Title = removeAllSpaces(e.Title)
	e.Para = removeAllSpaces(e.Para)
}

type Data struct {
	Query Record
	Para  Record
}

type Record struct {
	TokenIDs    []int64
	TextTypeIDs []int64
	PositionIDs []int64
}

type GeneratorConfig struct {
	VocabFile         string
	DoLowerCase       bool
	QueryMaxSeqLength int
	ParaMaxSeqLength  int
	MaxSeqLength      int
	ForCN             bool
}

type Generator struct {
	tokenizer         *Tokenizer
	queryMaxSeqLength int
	paraMaxSeqLength  int
	maxSeqLength      int
	forCN             bool
}

func NewGenerator(config GeneratorConfig) (*Generator, error) {
	tokenizer, err := NewTokenizer(config.VocabFile, config.DoLowerCase)
	if err != nil {
		return nil, err
	}
	return &Generator{
		tokenizer:         tokenizer,
		queryMaxSeqLength: config.QueryMaxSeqLength,
		paraMaxSeqLength:  config.ParaMaxSeqLength,
		maxSeqLength:      config.MaxSeqLength,
		forCN:             config.ForCN,
	}, nil
}

// GenerateDE generates data for dual encoder.
func (g *Generator) GenerateDE(e *Example) Data {
	if g.forCN {
		e.Clean()
	}

	queryTokens := g.tokenizer.Tokenize(e.Query)
	queryRecord := g.generate(queryTokens, nil, g.queryMaxSeqLength)

	titleTokens := g.tokenizer.Tokenize(e.Title)
	paraTokens := g.tokenizer.Tokenize(e.Para)
	paraRecord := g.generate(titleTokens, paraTokens, g.paraMaxSeqLength)

	return Data{Query: queryRecord, Para: paraRecord}
}

// GenerateCE generates data for cross encoder.
func (g *Generator) GenerateCE(e *Example) Record {
	if g.forCN {
		e.Clean()
	}

	tokensA := g.tokenizer.Tokenize(e.Query)

	tokensB := g.tokenizer.Tokenize(e.Title)
	tokensB = append(tokensB, g.tokenizer.Tokenize(e.Para)...)

	return g.generate(tokensA, tokensB, g.maxSeqLength)
}

// Pad pads the instances to the max sequence length in batch, and generate
// the corresponding input mask, which is used to avoid attention on paddings.
func (g *Generator) Pad(insts [][]int64) (padded [][]int64, inputMask [][]float32) {
	padID := g.tokenizer.vocab["[PAD]"]

	maxLen := 0
	for _, inst := range insts {
		if len(inst) > maxLen {
			maxLen = len(inst)
		}
	}

	for _, inst := range insts {
		paddedInst := inst
		var mask []float32
		for i := 0; i < len(inst); i++ {
			mask = append(mask, 1)
		}

		diffLen := maxLen - len(inst)
		if diffLen > 0 {
			paddedInst = make([]int64, len(inst))
			copy(paddedInst, inst)

			for i := 0; i < diffLen; i++ {
				paddedInst = append(paddedInst, padID)
				mask = append(mask, 0)
			}
		}

		padded = append(padded, paddedInst)
		inputMask = append(inputMask, mask)
	}

	return
}

// generate converts tokensA and tokens B into a Record.
//
// The convention in BERT/ERNIE is:
// (a) For sequence pairs:
//
//	tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
//	type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
//
// (b) For single sequences:
//
//	tokens:   [CLS] the dog is hairy . [SEP]
//	type_ids: 0     0   0   0  0     0 0
//
// Where "type_ids" are used to indicate whether this is the first
// sequence or the second sequence. The embedding vectors for `type=0` and
// `type=1` were learned during pre-training and are added to the wordpiece
// embedding vector (and position vector). This is not *strictly* necessary
// since the [SEP] token unambiguously separates the sequences, but it makes
// it easier for the model to learn the concept of sequences.
//
// For classification tasks, the first vector (corresponding to [CLS]) is
// used as the "sentence vector". Note that this only makes sense because
// the entire model is fine-tuned.
func (g *Generator) generate(tokensA, tokensB []string, maxSeqLength int) Record {
	padTokens := []string{"[CLS]", "[SEP]"}
	if len(tokensB) > 0 {
		padTokens = append(padTokens, "[SEP]")
	}
	tokensA, tokensB = g.truncateSeqPair(tokensA, tokensB, maxSeqLength-len(padTokens))

	var tokens []string
	var textTypeIDs []int64

	tokens = append(tokens, padTokens[0])
	textTypeIDs = append(textTypeIDs, 0)

	for _, token := range tokensA {
		tokens = append(tokens, token)
		textTypeIDs = append(textTypeIDs, 0)
	}
	tokens = append(tokens, padTokens[1])
	textTypeIDs = append(textTypeIDs, 0)

	if len(tokensB) > 0 {
		for _, token := range tokensB {
			tokens = append(tokens, token)
			textTypeIDs = append(textTypeIDs, 1)
		}
		tokens = append(tokens, padTokens[2])
		textTypeIDs = append(textTypeIDs, 1)
	}

	ids := g.tokenizer.TokensToIDs(tokens)

	var positionIDs []int64
	for i := 0; i < len(ids); i++ {
		positionIDs = append(positionIDs, int64(i))
	}

	return Record{
		TokenIDs:    ids,
		TextTypeIDs: textTypeIDs,
		PositionIDs: positionIDs,
	}
}

// truncateSeqPair truncates a sequence pair in place to the maximum length.
//
// This is a simple heuristic which will always truncate the longer sequence
// one token at a time. This makes more sense than truncating an equal percent
// of tokens from each, since if one sequence is very short then each token
// that's truncated likely contains more information than a longer sequence.
func (g *Generator) truncateSeqPair(tokensA []string, tokensB []string, maxLen int) (a []string, b []string) {
	for {
		aLen, bLen := len(tokensA), len(tokensB)
		if (aLen + bLen) <= maxLen {
			break
		}

		if aLen > bLen {
			if aLen > 0 {
				tokensA = tokensA[:aLen-1]
			}
		} else {
			if bLen > 0 {
				tokensB = tokensB[:bLen-1]
			}
		}
	}
	return tokensA, tokensB
}

func removeAllSpaces(s string) string {
	return strings.ReplaceAll(s, " ", "")
}
