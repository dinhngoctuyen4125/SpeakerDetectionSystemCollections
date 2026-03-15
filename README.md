## 1. Add and activate venv

```bash
conda create -n spk python=3.11.13
conda activate spk
```

## 2. Install dependencies

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run training

```bash
bash run_ECAPA.sh
```