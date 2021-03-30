# Description of the project

[Project](https://github.com/EdoardoGruppi/IoT_Assignment)

This project aims to analyse and model the information retrieved by a real-world smart home energy IoT system in order to understand which are the relationships in the dataset chosen between the dependent variables and the selected target feature. To help the understanding of the inner working of the models adopted, several machine learning interpretability techniques are exploited.

## Packages required

A comprehensive list of the packages needed to run the project is provided in the file [requirements.txt](https://github.com/EdoardoGruppi/IoT_Assignment/blob/main/requirements.txt). The latter can also be directly used to install the packages by typing the specific command on the terminal.

## Dataset

For the assignment, the dataset is downloaded directly from the Db2 database by the bespoke function at the beginning of the main code. It is also submitted within the code folder.

Alternatively, it can also be downloaded from this [link](https://www.kaggle.com/taranvee/smart-home-dataset-with-weather-information).

## How to start

Once all the necessary packages have been installed you can run the code by typing the following line on the terminal or by the specific command within the IDE.

```
python main.py
```

The packages required for the execution of the code along with the role of each file and the software used are described in the Sections below.

## Dashboard

The dashboard can be visualised typing the following line on the terminal or by running the specific command within the IDE.

```
python dashboard.py
```

Hereafter, two gifs introduce how the dashboard appears.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/48513387/112954769-00f5ad80-913f-11eb-8e9a-7b171c471cd3.gif)

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/48513387/112954810-0bb04280-913f-11eb-84cc-30793f5250d6.gif)

## Role of each file

**main.py** is the starting point of the entire project. It defines the order in which instructions are executed. More precisely, it is responsible to call functions from other .py files in order to download and divide the dataset, pre-process data as well as to instantiate, train and test the models.

**model.py** provides a series of functions to instantiate, train, cross validate and test a list of available models.

**config.py** makes available all the global variables used in the project.

**data_preparation** provides crucial functions to divide and prepare the dataset before feeding data to the models.

**ML_interpretability.py** includes helpful functions to interpret and to understand the inner working of the ML models adopted.

**utilities.py** is used to drop or combine some features from the original dataset as well as to compute the metrics on the model predictions.

**visualization.py** comprises useful functions to visualize data across all the entire duration of the project, i.e. from the data acquisition phase to the inference process.

**acquisition.py** enables the user to directly download the dataset from a db2 cloud database.

## Software used

> <img src="https://financesonline.com/uploads/2019/08/PyCharm_Logo1.png" width="200" alt="pycharm">

PyCharm is an integrated development environment (IDE) for Python programmers: it was chosen because it is one of the most advanced working environments and for its ease of use.

> <img src="https://user-images.githubusercontent.com/674621/71187801-14e60a80-2280-11ea-94c9-e56576f76baf.png" width="80" alt="vscode">

Visual Studio Code is a code editor optimized for building and debugging modern web and cloud applications.

> <img src="https://cdn-images-1.medium.com/max/1200/1*Lad06lrjlU9UZgSTHUoyfA.png" width="140" alt="colab">

Google Colab is an environment that enables to run python notebook entirely in the cloud. It supports many popular machine learning libraries and it offers GPUs where you can execute the code as well.
