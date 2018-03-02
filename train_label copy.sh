#!/bin/bash
#Read images and Create training & test DataFrames for transfer learning
#train_sub = readImages(img_dir + "subset") #<class 'pyspark.sql.dataframe.DataFrame'>
#labels_df = pd.read_csv(img_dir + "labels.csv")
#test_df = readImages(img_dir + "test")
#ss = pd.read_csv(img_dir + "sample_submission.csv")
IMAGES="/Users/macbook/Desktop/PSTAT194/spotted_dogfish/data/subset"
UNLABELED="/Users/macbook/Desktop/PSTAT194/spotted_dogfish/data/labels.csv"
TRAININGDATA="/Users/macbook/Desktop/PSTAT194/spotted_dogfish/data/training"
#rm "$UNLABELED"

for i in $(ls -1 "$IMAGES")
do
  for j in $(ls -1 "$IMAGES"/"$i")
    do
      rating=`identify -verbose "$IMAGES"/"$i"/"$j" | grep xmp:Rating | cut -d':' -f3`
      rtng=`echo "$rating" | awk '{$1=$1};1'`
      case "$rtng" in
        echo "this is breed 1"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/breed1/
             ;;
        echo "this is breed 2"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/breed2/
             ;;
        echo "this is breed 3"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/breed3/
             ;;
        echo "this is breed 4"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/breed4/
             ;;
        echo "this is someting else"
          echo "$j" >> "$UNLABELED"
             ;;
      esac
    done
done
