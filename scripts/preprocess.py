import numpy as np
import pandas as pd

from pyspark import SparkContext
from pyspark.sql import SQLContext
from sparkdl import readImages
# from sparkdl.image.image import ImageSchema
from pyspark.sql.functions import lit
#import keras

import load_data

from pyspark.sql.functions import *

sc = SparkContext.getOrCreate()
sqlCtx = SQLContext(sc)

img_dir = "/Users/macbook/Desktop/PSTAT194/spotted_dogfish/data_gen/"

#Read images and Create training & test DataFrames for transfer learning
affenpin = readImages(img_dir + "train/affenpinscher").withColumn("label", lit(0))
afghan_hound = readImages(img_dir + "train/afghan_hound").withColumn("label", lit(1))
african_hunt = readImages(img_dir + "train/african_hunting_dog").withColumn("label", lit(2))

# train_sub = readImages(img_dir + "subset") #<class 'pyspark.sql.dataframe.DataFrame'>
# labels_df = pd.read_csv(img_dir + "labels.csv")
# test_df = readImages(img_dir + "test")
# ss = pd.read_csv(img_dir + "sample_submission.csv")

#labels_df = sqlCtx.createDataFrame(labels_df)

affenpin_test, affenpin_train = affenpin.randomSplit([0.1, 0.9], seed = 420)
afghan_test, afghan_train = afghan_hound.randomSplit([0.1, 0.9], seed = 420)
af_hunt_test, af_hunt_train = african_hunt.randomSplit([0.1, 0.9], seed = 420)

train_df = affenpin_train.unionAll(afghan_train).unionAll(af_hunt_train)
test_df = affenpin_test.unionAll(afghan_test).unionAll(af_hunt_test)


from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="outFeatures", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)
