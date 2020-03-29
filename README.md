# Yogyakarta-Food-Production-Prediction
This Repository is about build Regression model to try predict values of Paddy Production commodity in tons from Some years before (Respectively) using Keras(Tensorflow) Sequence modeling.

## Dataset
We got the dataset from ['Produksi Pertanian Indonesia BPS'](https://www.kaggle.com/lintangwisesa/produksi-pertanian-indonesia-bps-19932015)

## Prerequisites
What things you need to install some python package before you run the notebook or python code
```
- Tensforflow Package
- Sklearn Package
- Numpy Package
- Pandas Package
- Matplotlib Package
```
## Contents
1. Folder [data](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/tree/master/data) 
(it's contain dataset Indonesia food production from 1993 until 2015 provided by BPS such as Corn, Paddy, Soy, and etc.)
2. Folder [model](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/tree/master/model) 
(it's contain Regression model with trained weights from modeling in Modeling notebook)
3. Folder [Plot](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/tree/master/plot) 
(it's contain MAE and MSE result during train and validation process also Regression and Prediction graph of Paddy Prodcution)
4. Modeling Notebook
  - [Final Model](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/blob/master/Final_regression_model.ipynb)
  - [Model with kfold and early stopping](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/blob/master/sequence_regression_model.ipynb)
  - [Protoype Model](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/blob/master/Regression_food_production_indonesia.ipynb)
5. Model Python file
  - [Main file](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/blob/master/main.py)
  - [Preprocessing dataset](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/blob/master/preprocessing.py)

```
*Note : 
- Modeling we used is [Final Model](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/blob/master/Final_regression_model.ipynb)
- Model Python file (main.py) is same model and workflow as Final Model, but we not update it until final model like Final_regression_model.ipynb
- preprocessing.py is very necessery file to proceed raw dataset of Padi.csv (Paddy Production dataset) to dataset that can be used to make regression model in modeling file. 
```
## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/BangkitProjectYOG4/Yogyakarta-Food-Production-Prediction/blob/master/LICENSE) file for details

## Authors
- Reza Anugrah Prakasa - [Reza Anugrah Prakasa](https://github.com/Yakagi17)
- Candra Dewi Jodistiara - [Candra Dewi Jodistiara](https://github.com/jodistiara)
- M.Alfa Riza - [M.Alfa Riza](https://github.com/AlfaRiza)
