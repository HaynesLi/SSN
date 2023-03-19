The Dataset is down from "https://www.woven-planet.global/en/data/prediction-dataset"




We support deterministic build through pipenv.

1: pip3 install pipenv

Once you’ve installed pipenv (or made it available in your env) run:

2: pipenv sync --dev

If you don’t care about determinist builds or you’re having troubles with packages resolution (Windows, Python<3.7, etc..), you can install directly from the setup.py by running:

3: pip3 install -e ."[dev]"

After install necessary packages, cd BSCCN folder

4:python3 train.py
