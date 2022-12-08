package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/RussellLuo/go-rocketqa"
	"github.com/elastic/go-elasticsearch/v8"
	"github.com/elastic/go-elasticsearch/v8/esutil"
)

type Item struct {
	Title string
	Para  string
}

type Indexer struct {
	es *elasticsearch.Client
	de *rocketqa.DualEncoder
}

func NewIndexer(es *elasticsearch.Client, de *rocketqa.DualEncoder) *Indexer {
	return &Indexer{
		es: es,
		de: de,
	}
}

func (i *Indexer) Index(index string, items []Item) error {
	bulk, err := esutil.NewBulkIndexer(esutil.BulkIndexerConfig{
		Client: i.es,
		Index:  index,
	})
	if err != nil {
		return err
	}

	ctx := context.Background()

	for idx, item := range items {
		vector := i.de.EncodePara(item.Para, item.Title).Unitize().ToFloat64()

		b, err := json.Marshal(map[string]interface{}{
			"title":     item.Title,
			"paragraph": item.Para,
			"vector":    vector,
		})
		if err != nil {
			return err
		}

		err = bulk.Add(
			ctx,
			esutil.BulkIndexerItem{
				Action:     "index",
				DocumentID: strconv.Itoa(idx + 1),
				Body:       bytes.NewReader(b),
				OnSuccess: func(ctx context.Context, item esutil.BulkIndexerItem, res esutil.BulkIndexerResponseItem) {
					fmt.Printf("[%d] %s test/%s\n", res.Status, res.Result, item.DocumentID)
				},
				OnFailure: func(ctx context.Context, item esutil.BulkIndexerItem, res esutil.BulkIndexerResponseItem, err error) {
					if err != nil {
						log.Printf("ERROR: %s\n", err)
					} else {
						log.Printf("ERROR: %s: %s\n", res.Error.Type, res.Error.Reason)
					}
				},
			},
		)
		if err != nil {
			return err
		}
	}

	return bulk.Close(ctx)
}

func getItems(dataFile string) ([]Item, error) {
	file, err := os.Open(dataFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var items []Item

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(strings.TrimSpace(line), "\t")
		if len(parts) != 2 {
			log.Printf("bad line %q", line)
			continue
		}

		items = append(items, Item{
			Title: parts[0],
			Para:  parts[1],
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return items, nil
}

func main() {
	var indexName, dataFile string
	flag.StringVar(&indexName, "index", "", "The index name")
	flag.StringVar(&dataFile, "data", "", "The data file")
	flag.Parse()

	if indexName == "" {
		log.Fatalf(`argument "index" is required`)
	}
	if dataFile == "" {
		log.Fatalf(`argument "data" is required`)
	}

	items, err := getItems(dataFile)
	if err != nil {
		log.Fatal(err)
	}

	es, err := elasticsearch.NewClient(elasticsearch.Config{
		Addresses: []string{"https://localhost:9200"},
		Username:  "elastic",
		Password:  "123456",
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	de, err := rocketqa.NewDualEncoder(&rocketqa.DualEncoderConfig{
		ModelPath:         "../../../testdata/zh_dureader_de_v2.pdmodel",
		ParamsPath:        "../../../testdata/zh_dureader_de_v2.pdiparams",
		VocabFile:         "../../../testdata/zh_vocab.txt",
		DoLowerCase:       true,
		QueryMaxSeqLength: 32,
		ParaMaxSeqLength:  384,
		ForCN:             true,
	})
	if err != nil {
		log.Fatal(err)
	}

	indexer := NewIndexer(es, de)
	if err := indexer.Index(indexName, items); err != nil {
		log.Fatal(err)
	}
}
