# DSC-Test-Run

## Environment Used
Python 3.10.11
Windows 11

## Project Structure

```
DSC-Test-Run/
│
├── data/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ... (more images)
│   ├── test/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ... (more images)
│   ├── test.csv
│   └── train.csv
│
├── src/
│   ├── Dhanay_J/
│   │   ├── README.md
│   │   ├── train.py
│   │   ├── test.py
│   │   └── config.yaml
```

## How to Use

### Training

To train the model, run the following command:

```bash
python src/Dhanay_J/train.py
```

### Testing

To test the model and generate the results in `submission.csv`, run the following command:

```bash
python src/Dhanay_J/test.py
```

## Configuration

The configuration for the project is stored in `config.yaml`. Make sure to update this file with the appropriate settings before running the training or testing scripts.
