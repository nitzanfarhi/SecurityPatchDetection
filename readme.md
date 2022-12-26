# Security Patch Detection Repository

## Creating the dataset
` python data_collection\create_dataset.py --cve -o data_collection\data`
## Running the training
`python train.py  -a before --model conv1d -k 10 --metadata -c`
