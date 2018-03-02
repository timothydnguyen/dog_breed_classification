import numpy as np
import pandas as pd
import h5py as h5py
from pyspark import SparkContext
from pyspark.sql import SQLContext
from sparkdl import readImages
from pyspark.sql.functions import lit

sc = SparkContext.getOrCreate()
sqlCtx = SQLContext(sc)
img_dir = "~/Desktop/PSTAT194/spotted_dogfish/data"

#Read images and Create training & test DataFrames for transfer learning
train_sub = readImages(img_dir + "/subset")
labels_df = pd.read_csv(img_dir + "/labels.csv")
test_df = readImages(img_dir + "/test")
ss = pd.read_csv(img_dir + "/sample_submission.csv")

def count_the_unique_values (data_set, column):
    description = data_set.describe(include="all")
    unique = description.iloc[1][column]
    return unique

### Make the unique values data as a row
def process_the_label_data(column,num_classes,sort,data_set):
    selected_breed_list = list(data_set.groupby(column).count().sort_values(by=sort, ascending=False).head(num_classes).index)
    data_set = data_set[data_set[column].isin(selected_breed_list)]
    data_set['target'] = 1
    #labels['rank'] = labels.groupby(column).rank()[sort]
    labels_pivot = data_set.pivot(sort, column, 'target').reset_index().fillna(0)
    return labels_pivot

'''
print (labels_df.head())

word_counter = {}
for word in labels_df['breed']:
    if word in word_counter:
        word_counter[word] +=1
    else:
        word_counter[word] = 1

popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
top_8 = popular_words[:8]

print(top_8)

'''
labels_df = sqlCtx.createDataFrame(labels_df)

#['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher',
#'bernese_mountain_dog', 'shih-tzu', 'great_pyrenees', 'pomeranian']
from pyspark.ml.feature import OneHotEncoder, StringIndexer

training, test = labels_df.randomSplit([0.6, 0.4], seed = 420)
print ("MADE TRAINING AND TEST SET LOL FUCK YEA")

stringIndexer = StringIndexer(inputCol="breed", outputCol="breedIndex")
model = stringIndexer.fit(labels_df)
indexed = model.transform(labels_df)

encoder = OneHotEncoder(inputCol="breed", outputCol="breedVec")
encoded = encoder.transform(indexed)

print ("ONE HOT ENCODER???")

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="breed")
p = Pipeline(stages=[encoder, lr])
p_model = p.fit(train_sub)

print("DID THIS PIPELINE WORK????")





#labels_df_col = labels_df.withColumn('filePath', sf.concat(sf.lit('file:/Users/macbook/Desktop/PSTAT194/spotted_dogfish/data/subset/'),sf.col('id'), sf.lit(".jpg")))

#print(train_sub.columns) #filepath, image
#training, test = train_sub.randomSplit([0.6, 0.4], seed = 420)
#print (labels_df.take(2))

#labels_rdd = labels_df.rdd

#combined = df1.join(df2, df1["c_id"] == df2["c_id"])
#combined.show()


# from pyspark.ml.feature import StringIndexer
#
# indexer = StringIndexer(inputCol="image", outputCol="imageIndex")
# indexed = indexer.fit(exam).transform(exam)
#
# indexed.show()
# from pyspark.ml.feature import OneHotEncoder
#
# encoder = OneHotEncoder(inputCol="image", outputCol="imageVec")
# encoded = encoder.transform(exam)
# print(encoded.head(1))

# training, test = indexed.randomSplit([0.6, 0.4], seed = 420)
#
#
# from pyspark.ml.classification import MultilayerPerceptronClassifier
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#
# #Train neural network
# # specify layers for the neural network:
# # input layer of size 11 (features), two intermediate of size 5 and 4
# # and output of size 7 (classes)
# layers = [11, 5, 4, 4, 3 , 7]
#
# # create the trainer and set its parameters
# FNN = MultilayerPerceptronClassifier(labelCol="breed", featuresCol="features",\
#                                          maxIter=100, layers=layers, blockSize=128, seed=420)
#
# model = FNN.fit(training)
#
#
# # Convert indexed labels back to original labels.
# labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
#                                labels=labelIndexer.labels)
# # Chain indexers and forest in a Pipeline
# from pyspark.ml import Pipeline
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, FNN, labelConverter])
# # train the model
# # Train model.  This also runs the indexers.
# model = pipeline.fit(trainingData)

#
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml import Pipeline
# from sparkdl import DeepImageFeaturizer
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#
# featurizer = DeepImageFeaturizer(inputCol="image", outputCol="outFeatures", modelName="InceptionV3")
# lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
# p = Pipeline(stages=[featurizer, lr])
# p_model = p.fit(test)
