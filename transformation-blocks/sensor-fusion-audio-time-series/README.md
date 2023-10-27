# Merge CSV files

This transformation job example takes a CSV file as an input and an audio file and merge them together in a CSV file. A 8kHz resampling is applied by default. Feel free to edit the transformation to make any changes.

![Run job](https://github.com/edgeimpulse/transformation-blocks/blob/main/assets/merge-audio-time-series-csv/run-merge-audio-csv.png?raw=true)

## Setup

**Operating mode**: Data item (`--in-directory`)

**Parameters**:

```
[
    {
        "name": "CSV file",
        "type": "string",
        "param": "csv",
        "value": "",
        "help": "CSV file name to be merged with audio"
    },
    {
        "name": "Audio file",
        "type": "string",
        "param": "audio",
        "value": "",
        "help": "Audio file name to be merged with CSV"
    }
]
```

## Test the transformation block locally

You can try this transformation with this dataset, using the `/raw` folder:
[Coffee Machine Stages](https://cdn.edgeimpulse.com/datasets/coffee-machine-stages.zip)

Install the dependencies:
```
pip3 install -r requirement.txt
```
Run the script:
```
python transform.py --in-directory ../dataset/CoffeeMachine-2023-10-20_14-15-51 --out-directory output --audio Microphone.mp4 --csv Accelerometer.csv
```