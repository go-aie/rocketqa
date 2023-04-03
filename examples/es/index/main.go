package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/elastic/go-elasticsearch/v8"
	"github.com/elastic/go-elasticsearch/v8/esutil"
	"github.com/go-aie/rocketqa"
)

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

func (i *Indexer) Index(index string, qptsC <-chan rocketqa.QPTs) error {
	bulk, err := esutil.NewBulkIndexer(esutil.BulkIndexerConfig{
		Client: i.es,
		Index:  index,
	})
	if err != nil {
		return err
	}

	ctx := context.Background()

	var baseIdx int
	for qpts := range qptsC {
		vectors, _ := i.de.EncodePara(qpts.P(), qpts.T())

		for idx, item := range qpts {
			vector := vectors[idx].Norm().ToFloat64()

			b, err := json.Marshal(map[string]interface{}{
				"title":     item.Title,
				"paragraph": item.Para,
				"vector":    vector,
			})
			if err != nil {
				return err
			}

			item := esutil.BulkIndexerItem{
				Action:     "index",
				DocumentID: strconv.Itoa(baseIdx + idx + 1),
				Body:       bytes.NewReader(b),
				OnSuccess: func(ctx context.Context, item esutil.BulkIndexerItem, res esutil.BulkIndexerResponseItem) {
					log.Printf("[%d] %s test/%s\n", res.Status, res.Result, item.DocumentID)
				},
				OnFailure: func(ctx context.Context, item esutil.BulkIndexerItem, res esutil.BulkIndexerResponseItem, err error) {
					if err != nil {
						log.Printf("ERROR: %s\n", err)
					} else {
						log.Printf("ERROR: %s: %s\n", res.Error.Type, res.Error.Reason)
					}
				},
			}
			if err = bulk.Add(ctx, item); err != nil {
				return err
			}
		}

		baseIdx += len(qpts)
	}

	return bulk.Close(ctx)
}

type Reader struct {
	C chan rocketqa.QPTs

	dataFile  string
	batchSize int
}

func NewReader(dataFile string, batchSize int) *Reader {
	if batchSize <= 0 {
		batchSize = 1
	}
	return &Reader{
		C:         make(chan rocketqa.QPTs),
		dataFile:  dataFile,
		batchSize: batchSize,
	}
}

func (r *Reader) Read() {
	file, err := os.Open(r.dataFile)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	var qpts []rocketqa.QPT

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(strings.TrimSpace(line), "\t")
		if len(parts) != 2 {
			log.Printf("bad line %q", line)
			continue
		}

		qpts = append(qpts, rocketqa.QPT{
			Title: parts[0],
			Para:  parts[1],
		})

		if len(qpts) == r.batchSize {
			// Reach the batch size, copy and send all the buffered records.
			temp := make([]rocketqa.QPT, r.batchSize)
			copy(temp, qpts)
			r.C <- temp

			// Clear the buffer.
			qpts = qpts[:0]
		}
	}

	// Send all the remaining records, if any.
	if len(qpts) > 0 {
		r.C <- qpts
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	close(r.C)
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

	reader := NewReader(dataFile, 100)
	go reader.Read()

	indexer := NewIndexer(es, de)
	if err := indexer.Index(indexName, reader.C); err != nil {
		log.Fatal(err)
	}
}
