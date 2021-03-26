# install requirements

    !pip install -r requirements.txt

# Run
You can run our three approaches that we mentioned in the report.
## 1. Completeness and locality evaluation 
The  `execute.py` file contains commented lines of code for calculating completeness and locality results mentioned in the report.

## 2. Fid score calculation 
This generated the table with FID scores for the horse2zebra dataset.

    python execute-fid-scores.py

## 3. Paired datasets evaluation
Cityscapes and Maps are the two available datasets. 
The training is to be run on a cluster, which trains 8 models in parallel. 
The evalution runs the models on the test set and computes their respective MSE's. The initial idea was to use FCN Models for evaluation but we weren't able to run them.

    python execute-cityscapes-train.py
    python execute-cityscapes-eval.py
    python execute-maps-train.py 
    python execute-maps-eval.py


