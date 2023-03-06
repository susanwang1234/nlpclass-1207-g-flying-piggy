# Evaluating the output files

1. Our evaluation phase uses Kaggle API. We will submit output .csv files to Kaggle and get a list of scores from Kaggle for each model. Please make sure you have a Kaggle account, and have created Kaggle API Token. Following show how to do that. Please skip it if you already have done that before.

> Go to the Account Tab ( `https://www.kaggle.com/<username>/account` ) and click ‘Create API Token’. A file named **kaggle.json** will be downloaded. Move this file in to `~/.kaggle/ `folder in Mac and Linux or to `C:\Users\<username>\.kaggle\ `on windows. This is required for authentication and do not skip this step.



2. Install Kaggle API package. 

    (We have already included this package in requirements.txt if you run in a virtual environment)

``` shell
pip install kaggle
```



3. Run check.py

```shell
python ./check.py
```



4. Done! You can see the scores of each model. If you see "Pending", please run:

``` shell
kaggle competitions submissions jigsaw-toxic-comment-classification-challenge
```



You can see a sample output in `scores.txt`. 



Note: you can download submission_xxx.csv files from our Goggle Drive to simply test the score of our output files. Or you can run the codes in source and generate the submission_xxx.csv files, and then evaluate them.



