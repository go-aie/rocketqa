# Command Line Interface


## Prerequisites

Install dependencies:

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```

## Train model

```bash
$ python3 cli.py train zh_dureader_ce_v2 ./cross.train.tsv
```


## Save inference model

Save the inference model from the default model for dual encoder:

```bash
$ python3 cli.py save zh_dureader_de_v2
```

Save the inference model from the default model for cross encoder:

```bash
$ python3 cli.py save zh_dureader_ce_v2
```

Save the inference model from a custom model:

```bash
$ python3 cli.py save /path/to/model/config.json --out-path=mymodel
```
