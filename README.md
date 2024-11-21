## Install Requirements

```sh
pip install -r requirements.txt
```

## Training a GNN
Follow instructions in ```gnn-training/```

## Local Play
Local play without network dependencies is possible, running:

```sh
PYTHONPATH=$(pwd) python scripts/local-play.py \
    --iter-1 200 \
    --iter-2 100 \
    --model-type-1 MCTS \
    --model-type-2 Random \
    --output-dir results
```

## Remote Play
Remote play is currently not possible, as the API communication is not working. Might fix it in the future.
