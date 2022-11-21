# Predict Gender from First Names

This project trains a model using `Multinomial Naive Bayes` algorithm to predict gender of a person from his/her first name. For this project, we used a dataset
downloaded from [data.gov](https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data) which contains a zip file containing
142 `txt` files. There are files for every year from 1800 to 2021.

###**Instruction**

1. Clone this repository:
```
git clone https://github.com/taeefnajib/predict-gender-from-first-name
```

2. Download the zip file from [data.gov](https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data) and unzip the `names` folder.
 Place it in the working directory.
 
3. Install all the dependencies:
```
pip install -r requirements.txt
```

4. `data.py` prepare a `csv` file from all the `txt` files and pre-processes the dataset. You don't need to run it in the command line.

5. `train.py` builds a model and trains it on the dataset. The repository contains the files `data.csv` and `model.pkl`. If you remove them and run `train.py`, 
this file will create the files `data.csv` and `model.pkl`

6. `test.py` uses `argparse` to allow users to predict genders from first names in the command line. Use `--name` or `-n` followed by the name you want to predict
 gender for. Example:
```
python test.py --name Josh
```
7. If you want to use `FastAPI` instead, you can do it:
```
uvicorn main:app --reload
```
This will open Swagger UI interface at 127.0.0.1 using port 8080 (if it is available). If you use the first name as a `string` it will reuturn a dictionary 
for `Gender` and `Probability`
