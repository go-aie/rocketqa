# Elasticsearch

This example illustrates how to use `go-rocketqa` along with [Elasticsearch](https://www.elastic.co/).


## Prerequisites

### Install Dependencies

```console
$ go mod tidy
```

Also see [Installation](../../README.md#installation).

### Run Elasticsearch

Run Elasticsearch in development mode:

```console
$ docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "ELASTIC_PASSWORD=123456" elasticsearch:8.4.2
```

## Usage

### Index

Prepare the data (stored at `data/test.tsv`) in the following format:

```
title_1\tparagraph_1\n
title_2\tparagraph_2\n
...
```

Create the index and save the data into the index:

```console
$ curl -XPUT -u elastic:123456 -k -H "Content-Type: application/json" https://localhost:9200/test-index -d @mappings.json
$ cd index
$ go run main.go -index=test-index -data=data/test.tsv
```

### Query

```console
$ cd query
$ go run main.go -index=test-index
```