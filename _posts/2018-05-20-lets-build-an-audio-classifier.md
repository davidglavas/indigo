---
title: "Let's build an audio classifier"
layout: post
date: 2018-05-13 12:08
mathjax: true
headerImage: true
tag:
- audio classification
- TensorFlow
star: false
category: blog
author: davidglavas
description: I use a neural network to build a simple audio classifier and evaluate its performance on the UrbanSound8K dataset.
---

<p align="center">
  <img src="https://github.com/davidglavas/davidglavas.github.io/blob/master/_posts/Figures/2018-05-20-lets-build-an-audio-classifier/Pipeline.png?raw=true" width ="1100" height="230">
</p>

## TL;DR
I use a neural network to build a simple audio classifier and evaluate its performance on the UrbanSound8K dataset.

For this post, our goal is learning how to:
-	turn sound recordings into feature vectors which a neural network can use,
-	build a simple classifier with TensoFlow’s estimator API,
-	run experiments and interpret results.



We will start by taking a look the dataset. Then, we will learn how to extract meaningful information from audio signals to obtain a compact, yet expressive description that the classifier can use. Then, we will learn how to use TensorFlow’s estimator API to build a simple neural network with which we will classify the extracted features. Finally, we will learn how to run experiments, interpret the results, and use what we have learned in future projects.

## The Data
In order for the neural network to generalize well we need a sufficiently large amount of varied labeled data. Creating such datasets is challenging as it usually involves first finding quality sources of data, then setting up a system to automate the collection process from various sources, and finally manually labeling the data. [Salamon and Bello](https://serv.cusp.nyu.edu/projects/urbansounddataset/salamon_urbansound_acmmm14.pdf) did just that. Their efforts resulted in the [UrbanSound8K dataset](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html), which we will use in this post. I chose this dataset because it is relatively large and easy to work with. 

The UrbanSound8K dataset is a collection of 8732 short (~4 second) labeled audio recordings (.wav files) of various urban sounds. Each recording belongs to one of the following 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, or street_music. The names of the audio files contain various meta-data of which we will use the class id, in other words, the label of each audio recording is contained in the file name. The recordings are conveniently pre-sorted into 10 folds to help with the reproduction and comparison of results.

## Feature Extraction
In this section we will turn our audio recordings into feature vectors. We do this because of the network structure we use to build the audio classifier—our data must conform to the input layer’s structure. Hence, we need to turn our audio recordings into arrays of floating point numbers—the feature vectors.

We must also keep in mind that the length of our feature vector is equal to the number of units in the network’s input layer—we therefore have full control over it. However, the cost of training a network (time and compute) usually increases with the number of units in the network. Therefore we will try to keep our feature vectors as small as possible.

A major challenge during the development of audio classifiers is the identification of appropriate content-based features for the representation of the audio recordings. The exact number of different features that people have used up to this point is unknown and irrelevant--the point is that there’s a lot of different ways to concisely represent audio signals, [this](https://www.sciencedirect.com/science/article/pii/S0065245810780037) is a great overview of the most used features. Choosing which feature(s) to extract depends on the nature of the audio sources. Different types of audio sources have different characteristics, the goal is to find features that capture relevant differences between our audio recordings which then the neural network can use during classification.

Our recordings are a kind of environmental sound. An overview of effective features for such audio recordings can be found [here]( https://ac.els-cdn.com/S0167865503001478/1-s2.0-S0167865503001478-main.pdf?_tid=8901d567-2f27-44bd-ba5b-b48290fd89c8&acdnat=1525870758_f2ee1c0c2b690073c98a4232873a5974). The feature we will use is inspired by the human auditory system and has proven to be very effective—the Mel-frequency cepstrum (MFC). For the purposes of this post we need to know that we extract one MFC per audio recording, that an MFC is made up of Mel-frequency cepstral coefficients (MFCCs), and that an MFC can be stored as a matrix. Each column in an MFCC matrix represents the MFCCs for one frame, and each row represents the extracted MFCCs across all frames (note that some other libraries reverse the columns and rows). So an MFCC matrix is a sequence of MFCCs. In case you are interested how the matrix is computed, see [this](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/). Next we will describe how to turn the MFCC matrix into a feature vector by summarizing the extracted sqeuences of MFCCs (rows).

Let’s assume that we extracted all the MFCCs for our audio recordings, now we have one matrix per audio recording which is supposed to be a good representation of the recording’s content. Finally, we will obtain the feature vectors by summarizing each MFCC matrix into one vector. For example, we can summarize one MFCC matrix into a feature vector by computing the mean of every sequence of MFCCs (row in the matrix)—the feature vector would be made up of the means and its length would be equal to the number of MFCCs seqeuences (rows in the matrix). We will go a step further and use multiple summary statistics (minimum, maximum, median, mean, variance, skewness, kurtosis) and obtain the final feature vector by concatenating the vectors we obtained for each of the summary statistics. Therefore, the length of our final feature vector will be the number of coefficients (rows) in our MFCC matrix multiplied by the number of summary statistics we use—in our case we have a feature vector of length 20 * 7 = 140. Note that I chose 20 as the number of coefficients through experimenting with different values.

To extract the features we will use [LibROSA](https://librosa.github.io/librosa/index.html)—a package for music and audio analysis. Given the path to one of the recordings, we can compute the corresponding MFCC matrix and create our feature vector as follows:

``` python
def extract_features_from_file(file_name):
    raw_sound, sample_rate = librosa.load(file_name)

    # one row per extracted coefficient, one column per frame
    mfccs = librosa.feature.mfcc(y=raw_sound, sr=sample_rate, n_mfcc=20)

    mfccs_min = np.min(mfccs, axis=1)  # row-wise summaries
    mfccs_max = np.max(mfccs, axis=1)
    mfccs_median = np.median(mfccs, axis=1)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_variance = np.var(mfccs, axis=1)
    mfccs_skeweness = skew(mfccs, axis=1)
    mfccs_kurtosis = kurtosis(mfccs, axis=1)

    return mfccs_min, mfccs_max, mfccs_median, mfccs_mean, mfccs_variance, mfccs_skeweness, mfccs_kurtosis
```

Let’s take a closer look at a feature vector we create with the above function. Explicitly, the first feature in our feature vector is the minimum of the first coefficient (first MFCCs sequence) of the recording’s MFCC matrix. The second feature is the minimum value of the second coefficient in the recording’s MFCC matrix. Assume that the MFCC matrix has $k$ rows and start index is $1$, the $k+1$-th feature in our feature vector is the maximum (second summary statistic) value of the first coefficient in the recording’s MFCC matrix. The $2k+1$-th feature is the median of the first coefficient in the recording’s MFCC matrix. And so on for the rest of the summary statistics we chose: the $m$-th summary statistic makes up the $k$ features starting from the $(m-1)*k + 1$-th feature in the feature vector.

The following example shows how to use librosa for performing the feature extraction I described above. 

``` python
# First we load the audio file.
raw_sound, sample_rate = librosa.load(file_name)  # file must be in the root folder of your project

print("raw_sound:", raw_sound)
print("raw_sound.shape:", raw_sound.shape)
print("\n")

mfccs = librosa.feature.mfcc(y=raw_sound, sr=sample_rate, n_mfcc=20)  # compute the MFCC matrix

# Next we compute the summary statistics, each of them summarizes the MFCC matrix in its own way.
mfccs_min = np.min(mfccs, axis=1)  # row-wise minimum, etc
mfccs_max = np.max(mfccs, axis=1)
mfccs_median = np.median(mfccs, axis=1)
mfccs_mean = np.mean(mfccs, axis=1)
mfccs_variance = np.var(mfccs, axis=1)
mfccs_skeweness = skew(mfccs, axis=1)
mfccs_kurtosis = kurtosis(mfccs, axis=1)

# We obtain the feature vector by concatenating the different summaries.
finalFeatureVector = np.concatenate([mfccs_min, mfccs_max, mfccs_median, mfccs_mean, mfccs_variance, mfccs_skeweness, mfccs_kurtosis])

print("mfccs:", mfccs)
print("mfccs.shape:", mfccs.shape)
print("\n")

print("mfccs_min:", mfccs_min)
print("mfccs_min.shape:", mfccs_min.shape)
print("\n")

print("mfccs_max:", mfccs_max)
print("mfccs_max.shape:", mfccs_max.shape)
print("\n")

print("mfccs_median:", mfccs_median)
print("mfccs_median.shape:", mfccs_median.shape)
print("\n")

print("mfccs_mean:", mfccs_mean)
print("mfccs_mean.shape:", mfccs_mean.shape)
print("\n")

print("mfccs_variance:", mfccs_variance)
print("mfccs_variance.shape:", mfccs_variance.shape)
print("\n")

print("mfccs_skeweness:", mfccs_skeweness)
print("mfccs_skeweness.shape:", mfccs_skeweness.shape)
print("\n")

print("mfccs_kurtosis:", mfccs_kurtosis)
print("mfccs_kurtosis.shape:", mfccs_kurtosis.shape)
print("\n")

print("finalFeatureVector", finalFeatureVector)
print("finalFeatureVector.shape:", finalFeatureVector.shape)
```

The output of the above example shows the data we are working with during all stages of the feature extraction process I described earlier:

``` python
raw_sound: [-0.05454996 -0.13038099 -0.1837227  ...  0.02945998  0.00751382
 -0.03976963]
raw_sound.shape: (88200,)


mfccs: [[-9.78217290e+01 -1.15200264e+02 -1.40122382e+02 ... -1.96356880e+02
  -1.88242525e+02 -1.70682403e+02]
 [ 1.32125684e+02  1.27208858e+02  1.15394368e+02 ...  1.34137467e+02
   1.27680399e+02  1.22533017e+02]
 [-6.74225340e+01 -6.66515689e+01 -5.92767531e+01 ... -3.84596698e+01
  -4.27690513e+01 -4.38633941e+01]
 ...
 [-2.10589355e+00  2.17217352e+00  1.09952991e+01 ...  1.37435825e+01
   1.12389371e+01  1.08770824e+01]
 [-5.70890730e-02  1.68226600e+00  6.01086590e+00 ...  3.57389057e+00
   3.50404748e+00  3.94450867e+00]
 [ 6.35962364e+00  3.37899134e+00  6.57503920e+00 ...  7.80786112e+00
   6.53761178e+00  7.81919162e+00]]
mfccs.shape: (20, 173)


mfccsMin: [-262.1882601   103.68769703  -70.92737613   -7.06636488   -2.57441136
  -14.41774564    3.2479818     5.35776817  -28.04321332   -5.00591745
   -6.77451869  -19.54877245  -13.61704643    1.02649618  -16.86513529
   -2.19287018   -5.88764237   -6.10989292   -6.86591663   -0.98465032]
mfccsMin.shape: (20,)


mfccsMax: [-97.82172897 168.4061548  -28.47245793  38.02142843  21.11869576
  16.15369493  30.22304018  30.83948912   7.8262849   23.61299204
  12.44763379   6.71378187   9.72676893  18.96199983  16.05955284
  24.25329661  23.63521651  21.56967718  14.08734175  18.52652466]
mfccsMax.shape: (20,)


mfccsMedian: [-211.31858874  143.49744481  -54.27675658   20.94282905    9.16580307
   -0.21533032   12.92539716   19.48127578  -15.44112234    8.92052176
    2.95877848   -9.21938462   -3.30998802   12.52612904    5.7662368
   10.03349768    7.50772457    6.01753001    2.9066586    10.10142573]
mfccsMedian.shape: (20,)


mfccsMean: [-2.07550980e+02  1.41899004e+02 -5.35410878e+01  2.05936852e+01
  9.21844173e+00  1.36437309e-01  1.33962109e+01  1.92881863e+01
 -1.49302789e+01  9.21802490e+00  3.25152418e+00 -8.93898900e+00
 -3.07507361e+00  1.17549153e+01  4.85579146e+00  9.71038003e+00
  7.07648620e+00  6.55743907e+00  2.49942263e+00  9.86277491e+00]
mfccsMean.shape: (20,)


mfccsVariance: [889.79984189 268.69156081  88.54314337  75.46131586  25.79757966
  32.35109778  36.47155209  27.3671436   42.67346134  35.26865155
  14.07979971  26.54864055  19.22512523  15.06969567  35.02728229
  20.04404124  21.50347683  29.0669686   15.21941274  14.67586581]
mfccsVariance.shape: (20,)


mfccsSkeweness: [ 0.87052288 -0.47742533  0.35495499 -0.09604224 -0.1575763   0.25362987
  0.50156373 -0.12891151  0.49370992  0.00782589  0.10513811  0.37190028
  0.17411075 -0.45545584 -0.59793439  0.19273781 -0.054743    0.27577162
 -0.07832378 -0.30175746]
mfccsSkeweness.shape: (20,)


mfccsKurtosis: [ 0.90751202 -0.4865782  -0.6830255  -0.64238508 -0.3210699   0.08463533
 -0.26465951 -0.29554489  0.11507486 -0.42076076 -0.32018804 -0.18306213
  0.03231722 -0.42274134  0.1122558   0.02620816  0.24541489 -0.14746111
  0.13099534 -0.15450902]
mfccsKurtosis.shape: (20,)


finalFeatureVector [-2.62188260e+02  1.03687697e+02 -7.09273761e+01 -7.06636488e+00
 -2.57441136e+00 -1.44177456e+01  3.24798180e+00  5.35776817e+00
 -2.80432133e+01 -5.00591745e+00 -6.77451869e+00 -1.95487725e+01
 -1.36170464e+01  1.02649618e+00 -1.68651353e+01 -2.19287018e+00
 -5.88764237e+00 -6.10989292e+00 -6.86591663e+00 -9.84650318e-01
 -9.78217290e+01  1.68406155e+02 -2.84724579e+01  3.80214284e+01
  2.11186958e+01  1.61536949e+01  3.02230402e+01  3.08394891e+01
  7.82628490e+00  2.36129920e+01  1.24476338e+01  6.71378187e+00
  9.72676893e+00  1.89619998e+01  1.60595528e+01  2.42532966e+01
  2.36352165e+01  2.15696772e+01  1.40873418e+01  1.85265247e+01
 -2.11318589e+02  1.43497445e+02 -5.42767566e+01  2.09428290e+01
  9.16580307e+00 -2.15330319e-01  1.29253972e+01  1.94812758e+01
 -1.54411223e+01  8.92052176e+00  2.95877848e+00 -9.21938462e+00
 -3.30998802e+00  1.25261290e+01  5.76623680e+00  1.00334977e+01
  7.50772457e+00  6.01753001e+00  2.90665860e+00  1.01014257e+01
 -2.07550980e+02  1.41899004e+02 -5.35410878e+01  2.05936852e+01
  9.21844173e+00  1.36437309e-01  1.33962109e+01  1.92881863e+01
 -1.49302789e+01  9.21802490e+00  3.25152418e+00 -8.93898900e+00
 -3.07507361e+00  1.17549153e+01  4.85579146e+00  9.71038003e+00
  7.07648620e+00  6.55743907e+00  2.49942263e+00  9.86277491e+00
  8.89799842e+02  2.68691561e+02  8.85431434e+01  7.54613159e+01
  2.57975797e+01  3.23510978e+01  3.64715521e+01  2.73671436e+01
  4.26734613e+01  3.52686516e+01  1.40797997e+01  2.65486406e+01
  1.92251252e+01  1.50696957e+01  3.50272823e+01  2.00440412e+01
  2.15034768e+01  2.90669686e+01  1.52194127e+01  1.46758658e+01
  8.70522876e-01 -4.77425325e-01  3.54954990e-01 -9.60422388e-02
 -1.57576301e-01  2.53629873e-01  5.01563735e-01 -1.28911511e-01
  4.93709923e-01  7.82588861e-03  1.05138109e-01  3.71900277e-01
  1.74110753e-01 -4.55455841e-01 -5.97934394e-01  1.92737805e-01
 -5.47430015e-02  2.75771615e-01 -7.83237770e-02 -3.01757456e-01
  9.07512022e-01 -4.86578202e-01 -6.83025497e-01 -6.42385082e-01
 -3.21069903e-01  8.46353279e-02 -2.64659508e-01 -2.95544886e-01
  1.15074865e-01 -4.20760761e-01 -3.20188037e-01 -1.83062131e-01
  3.23172225e-02 -4.22741337e-01  1.12255804e-01  2.62081612e-02
  2.45414889e-01 -1.47461115e-01  1.30995342e-01 -1.54509024e-01]
finalFeatureVector.shape: (140,)
```

Note that the feature vectors will potentially vary a lot due to the nature of the recordings (ex. maximum for a gun shot recording will potentially be much larger than for an air conditioner recording). Hence, we will mean normalize the extracted features as follows:

``` python
def mean_normalize(featureMatrix):
    mean = np.mean(featureMatrix, axis=0)  # compute mean of each column (feature)
    std = np.std(featureMatrix, axis=0, ddof=1)  # compute sample std of each column (feature)

    featureMatrix -= mean  # subtract each column's mean from every value in the corresponding column
    featureMatrix /= std  # divide values in each column with the corresponding sample std for that column

    return featureMatrix
```

The feature extraction phase tends to take long—my PC takes about 6 minutes to extract feature vectors of length 140 for one fold of the UrbanSound8K dataset. We will store the extracted features. This way we only need to pay the price of extracting features once, and can reuse them while experimenting during classification later on. 

I plan to do a 10-fold cross validation, the following two functions make it easy to extract and store the features systematically. I use `extract_feature_from_directories` to iterate through the folds and extract features from the recordings they contain--all the extracted feature vectors are combined into one feature matrix (rows are feature vectors, columns are features). 

``` python
def extract_features_from_directories(parent_dir, sub_dirs, file_ext="*.wav"):
    feature_matrix, labels = np.empty((0, featureVectorLength)), np.empty(0)

    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs_min, mfccs_max, mfccs_median, mfccs_mean, mfccs_variance, mfccs_skeweness, mfccs_kurtosis = extract_features_from_file(fn)
                print("Finished processing file: ", fn)
            except Exception as e:
                print("Error while processing file: ", fn)
                continue

            # concatenate extracted features
            new_feature_vector = np.hstack([mfccs_min, mfccs_max, mfccs_median, mfccs_mean, mfccs_variance, mfccs_skeweness, mfccs_kurtosis])

            # add current feature vector as last row in feature matrix
            feature_matrix = np.vstack([feature_matrix, new_feature_vector])

            # extracts label from the file name. Change '\\' to  '/' on Unix systems
            labels = np.append(labels, fn.split('\\')[2].split('-')[1])

    return np.array(feature_matrix), np.array(labels, dtype=np.int)
```

I use `prepare_features` to extract and store the training and validation set for one run of the 10-fold cross validation. 

``` python
def prepare_features(training_dirs, validation_dirs, training_name, validation_name):
    parent_dir = 'Sound-Data'  # name of the directory which contains the recordings
    training_sub_dirs = training_dirs
    validation_sub_dirs = validation_dirs

    # ndarrays
    training_features, training_labels = extract_features_from_directories(parent_dir, training_sub_dirs)
    test_features, test_labels = extract_features_from_directories(parent_dir, validation_sub_dirs)

    # convert ndarray to pandas dataframe
    training_examples = pd.DataFrame(training_features, columns=list(range(1, featureVectorLength+1)))
    # convert ndarray to pandas series
    training_labels = pd.Series(training_labels.tolist())

    # convert ndarray to pandas dataframe
    validation_examples = pd.DataFrame(test_features, columns=list(range(1, featureVectorLength+1)))
    # convert ndarray to pandas series
    validation_labels = pd.Series(test_labels.tolist())

    # store extracted training data
    training_examples.to_pickle('Extracted_Features\\' + training_name + '_features.pkl')
    training_labels.to_pickle('Extracted_Features\\' + training_name + '_labels.pkl')

    # store extracted validation data
    validation_examples.to_pickle('Extracted_Features\\' + validation_name + '_features.pkl')
    validation_labels.to_pickle('Extracted_Features\\' + validation_name + '_labels.pkl')
```

The following snippet uses the above functions to create the training and validation sets that are used during the first run of the 10-fold cross-validation:

``` python
# On the first run I use the first 9 folds for training, the tenth for validation.
training_dirs = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9"]
validation_dirs = ["fold10"]

# extracts and stores training and validation sets that are used for the first run of the 10-fold cross-validation.
prepare_features(training_dirs, validation_dirs, 'notFold10', 'fold10')
```

To summarize, we extracted one MFCC matrix per recording. The features we will use during classification are the summary statistics of the MFCC matrix’s coefficients (rows). By concatenating the different summaries of the MFCC matrix we obtain a compact representation of the original audio recording—the final feature vector. The code for this section can be found [here](https://gist.github.com/davidglavas/c33a9eb5bec736e47438ec546f629520).

Next, we will discuss the classification phase and how the neural network will use the feature vectors we developed in this section.

## Classification
In this section we will use a neural network to classify the audio recordings. We want to classify each audio recording into one of 10 classes mentioned earlier. Hence, we are dealing with a multi-class classification problem where the classes are mutually exclusive (a recording can belong to only one class). Therefore, we want to build a neural network with a softmax layer at the top.

Our goal is to create a simple but flexible framework which we can use for experiments. We want to control hyperparameters such as the learning rate, the regularization strength, the number of steps the optimization algorithm makes, the batch size, the number of hidden layers, and the number of hidden units in each layer. We also want to leave the doors open for trying out different types of regularization (L1, L2, Dropout), activation functions, to change the number of classes, and to test different optimization algorithms. We want this flexibility while minimizing the number of errors when switching between configurations during experiments. We don’t want to deal with issues such as the initialization of weights, connecting individual units, and cost functions. We are fine with reasonable defaults, as long as it allows for enough freedom to quickly try out interesting configurations.

TensorFlow’s [Estimator API](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) offers everything we need. Next we will see how to use the estimator API's [DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) to build our network. The code that follows is a modified version of [this]( https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/programming-exercise).

We have to let our estimator know what type of data it can expect, we do this by defining a [feature column](https://www.tensorflow.org/get_started/feature_columns). Then, we have to let our estimator know how to fetch data from our dataset—we do this by defining an [input function]( https://www.tensorflow.org/get_started/premade_estimators#create_input_functions). Finally, we will initialize the classifier and setup the training loop—here we will set up the monitoring of metrics we are interested in while training the classifier (loss curves, confusion matrix). 

We start by defining the feature column:

``` python
def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """
    return set([tf.feature_column.numeric_column('audioFeatures', shape=featureVectorSize)])
```

Next we will define two functions for creating the input functions, one for fetching data from the training set, the other for the validation set:

``` python
def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
    """A custom input_fn for sending our feature vectors to the estimator for training.

    Args:
      features: The training features.
      labels: The training labels.
      batch_size: Batch size to use during training.

    Returns:
      A function that returns batches of training features and labels during training.
    """

    def _input_fn(num_epochs=num_epochs, shuffle=True):
        idx = np.random.permutation(features.index)
        raw_features = {"audioFeatures": features.reindex(idx)}
        raw_labels = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_labels))
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        # Returns the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def create_predict_input_fn(features, labels, batch_size):
    """A custom input_fn for sending our feature vectors to the estimator for predictions.

    Args:
      features: The features to base predictions on.
      labels: The labels of the prediction examples.

    Returns:
      A function that returns features and labels for predictions.
    """

    def _input_fn():
        raw_features = {"audioFeatures": features.values}
        raw_labels = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_labels))
        ds = ds.batch(batch_size)

        # Returns the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn
```

Next we will define the function that will train the model while periodically printing loss metrics to guide our hyperparameter search during experiments. We will divide the number of training steps into 10 periods. For each period, we train the `DNNClassifier` for $steps/10$ steps. Then we print loss metrics and continue with the next period. We repeat this 10 times, each time we continue training the `DNNClassifier` where the previous period left off. The last period gives us the final `DNNClassifier` which represents our final model.

``` python
def train_nn_classification_model(
        learning_rate,
        regularization_strength,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_labels,
        validation_examples,
        validation_labels,
        model_Name='no_Name'):
    """Trains a neural network classification model.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.

    Args:
      learning_rate: An `int`, the learning rate to use.
      regularization_strength: A float, the regularization strength.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of units in each layer.
      training_examples: A `DataFrame` containing the training features.
      training_labels: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_labels: A `DataFrame` containing the validation labels.
      model_Name: A `string` containing the model's name which is used when storing the loss curve and confusion
       matrix plots.

    Returns:
      The trained `DNNClassifier` object.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_labels, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_labels, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_labels, batch_size)

    # Create feature columns.
    feature_columns = construct_feature_columns()

    # Create a DNNClassifier object.
    my_optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=learning_rate,
        l2_regularization_strength=regularization_strength  # can be swapped for l1 regularization
    )

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Use the current model to make predictions on both, the training and validation set.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Use predictions to compute training and validation errors.
        training_log_loss = metrics.log_loss(training_labels, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_labels, validation_pred_one_hot)

        # Print validation error of current model.
        print("  period %02d : %0.2f" % (period, validation_log_loss))

        # Store loss metrics so we can plot them later.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)

    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Compute predictions of final model.
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    # Evaluate predictions of final model.
    accuracy = metrics.accuracy_score(validation_labels, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    # plt.show()  # blocks execution
    plt.savefig('Results\\' + model_Name + '_loss_curve.png', bbox_inches='tight')
    plt.gcf().clear()

    # Create a confusion matrix.
    cm = metrics.confusion_matrix(validation_labels, final_predictions)

    # Normalize the confusion matrix by the number of samples in each class (rows).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # plt.show()  # blocks execution
    plt.savefig('Results\\' + model_Name + '_confusion_matrix.png', bbox_inches='tight')
    plt.gcf().clear()

    return classifier
```

We train the model by calling the training function we defined above like this:

``` python
# unpickle and prepare training data
training_examples = mean_normalize(pd.read_pickle('Extracted_Features\\notFold10_features.pkl'))
training_labels = pd.read_pickle('Extracted_Features\\notFold10_labels.pkl')

# unpickle and prepare validation data
validation_examples = mean_normalize(pd.read_pickle('Extracted_Features\\fold10_features.pkl'))
validation_labels = pd.read_pickle('Extracted_Features\\fold10_labels.pkl')

train_nn_classification_model(
    learning_rate=0.003,
    regularization_strength=1.0,
    steps=5000,
    batch_size=32,
    hidden_units=[120],  # One layer with 120 units, for more layers simply add more integers to the list.
    training_examples=training_examples,
    training_labels=training_labels,
    validation_examples=validation_examples,
    validation_labels=validation_labels)
 ```
 
To summarize, we used the Estimator's API `DNNClassifier` to build a simple neural network with which we classified the feature vectors we created earlier. The code for this section can be found [here](https://gist.github.com/davidglavas/60d102bb236cda4f2ff129324352dc86).


## Results

In order to estimate how a certain model configuration will perform on unseen data we will do a 10-fold cross-validation using the already provided folds in the UrbanSound8K dataset. My machine takes about 4 minutes (on average across different configurations with 1 layer) to train a `DNNClassifier` on 9 folds and evaluate it on 1 fold. A 10 fold cross-validation takes my machine about 40 minutes. This severely limits my hyperparameter search, I therefore searched for a good model configuration while training on 9 folds and validating on 1. Then, once I found a configuration which performed well, I ran a 10-fold cross-validation to see how it generalizes.

To perform a 10-fold cross-validation we first have to load the features we extracted and stored earlier. Then we use the training function to train models for the different folds:

``` python
def k_fold_cross_validation(training_set_names, validation_set_names):
    """
    Performs a k-fold cross validation. Trains k different models and lets you know how they perform by using the
    corresponding validation set.
    :param training_set_names: List of training sets stored as tuples. Each tuple is a pair of strings, first
    element is the name of the training examples, second element is the name of the corresponding training labels.
    :param validation_set_names: List of validation sets stored as tuples. Each tuple is a pair of strings, first
     element is the name of the validation examples, second element is the name of the corresponding validation labels.
    """

    # group each training set with its corresponding validation set
    folds = zip(training_set_names, validation_set_names)

    for (training_name, validation_name) in folds:

        training_examples, training_labels = load_features(training_name)
        validation_examples, validation_labels = load_features(validation_name)

        print("#####################################################################################")
        print("Model is trained with ", training_name[0], "and validated with", validation_name[0])
        train_nn_classification_model(
            learning_rate=0.003,
            regularization_strength=0.1,
            steps=5000,
            batch_size=32,
            hidden_units=[120],
            training_examples=training_examples,
            training_labels=training_labels,
            validation_examples=validation_examples,
            validation_labels=validation_labels,
            model_Name=training_name[0])
 

def load_features(dataset_name):
    """
    Unpickles the given examples and labels. Mean normalizes the examples.
    :param dataset_name: Pair of names referring to an example and corresponding label set.
    :return: Actual dataset as a pair, first element are the mean normalized examples (pandas DataFrame), second
     element are the corresponding labels (pandas Series).
    """

    examples_path = 'Extracted_Features\\' + dataset_name[0]
    # unpickles and mean normalizes examples
    examples = mean_normalize(pd.read_pickle(examples_path))

    # unpickles labels
    labels_path = 'Extracted_Features\\' + dataset_name[1]
    labels = pd.read_pickle(labels_path)

    return examples, labels
 ```

Finally, we perform the 10-fold cross validation by specifying the folds and calling the above `k_fold_cross_validation` function. Note that the `k` is equal to the number of pairs in `training_set_names` and `validation_set_names`. Further, the first cross validation run will use the first tuple in `training_set_names` (corresponds to the first 9 folds) for training, and the first tuple in `validation_set_names` (corresponds to the tenth fold) for validation. Similarly for the rest of the runs.

``` python
# order in training_set_names matches the order in validation_set_names
training_set_names = [('notFold1_features.pkl', 'notFold1_labels.pkl'),
                      ('notFold2_features.pkl', 'notFold2_labels.pkl'),
                      ('notFold3_features.pkl', 'notFold3_labels.pkl'),
                      ('notFold4_features.pkl', 'notFold4_labels.pkl'),
                      ('notFold5_features.pkl', 'notFold5_labels.pkl'),
                      ('notFold6_features.pkl', 'notFold6_labels.pkl'),
                      ('notFold7_features.pkl', 'notFold7_labels.pkl'),
                      ('notFold8_features.pkl', 'notFold8_labels.pkl'),
                      ('notFold9_features.pkl', 'notFold9_labels.pkl'),
                      ('notFold10_features.pkl', 'notFold10_labels.pkl')]

validation_set_names = [('fold1_features.pkl', 'fold1_labels.pkl'),
                        ('fold2_features.pkl', 'fold2_labels.pkl'),
                        ('fold3_features.pkl', 'fold3_labels.pkl'), 
                        ('fold4_features.pkl', 'fold4_labels.pkl'),
                        ('fold5_features.pkl', 'fold5_labels.pkl'), 
                        ('fold6_features.pkl', 'fold6_labels.pkl'),
                        ('fold7_features.pkl', 'fold7_labels.pkl'), 
                        ('fold8_features.pkl', 'fold8_labels.pkl'),
                        ('fold9_features.pkl', 'fold9_labels.pkl'),
                        ('fold10_features.pkl', 'fold10_labels.pkl')]


k_fold_cross_validation(training_set_names, validation_set_names)
 ```

 In my (fairly limited) experiments, the best configuration I found achieved an average accuracy of 70.3% across all 10 folds:

 ``` python
classifier = train_nn_classification_model(
    learning_rate=0.003,
    regularization_strength=0.2,
    steps=10000,
    batch_size=32,
    hidden_units=[120],
    training_examples=training_examples,
    training_labels=training_labels,
    validation_examples=validation_examples,
    validation_labels=validation_labels,
    model_Name=training_name[0])
 ```

The loss curves and confusion matrices are similar across folds. The following is obtained by training on the first 9 folds and validating on the 10th:

<p align="center">
  <img src="https://github.com/davidglavas/davidglavas.github.io/blob/master/_posts/Figures/2018-05-20-lets-build-an-audio-classifier/classification_results.png?raw=true">
</p>

While trying to compare results I found only one [article]( http://iaser.org/Vol-2/Session%203/02EEECS212.pdf) that used simple fully connected neural networks to classify the UrbanSound8K dataset. Their best classification accuracy is 72.2% using 2 hidden layers and 3000 units per layer.

Note that there are much better approaches for classifying the UrbanSound8K dataset. I used a simple fully connected neural network with one layer and few hidden units due to hardware constraints. The most common deep learning based approach for classification of sounds is to convert the audio file to an image (ex. spectrogram, MFCC, CRP), and then use a convolutional neural network to classify the image. The best classification accuracy on the UrbanSound8K dataset I could find is 93%, the approach is described [here](https://www.sciencedirect.com/science/article/pii/S1877050917316599).

The approach I used in this post is by no means a good one when it comes to maximizing classification performance. It's great for learning because it can easily be transferred to similar problems to get working prototypes quickly. In case you wish to use a different dataset of audio recordings you could research what features work well for your data and adapt the feature extraction process. In case you use fewer or more classes, simply change the `n_classes` parameter. In case you want to try out more layers and/or units, simply change the `hidden_units` parameter. You can swap `l2_regularization_strength` for `l1_regularization_strength` or add `dropout=0.5` to the initialization phase of the `DNNClassifier`. You can swap the `proximalAdagradOptimizer` with another [optimizer]( https://www.tensorflow.org/api_docs/python/tf/train/Optimizer) (ex.  `tf.train.AdagradOptimizer(learning_rate=learning_rate)`).

To conclude, we learned how to use librosa to extract features from audio files. Then we learned how to build a simple but flexible framework for quick experiments with the Estimator API. Finally we learned how to evaluate results and compared it with the results of others. Feel free to comment on what I could have done better or what you would have done differently and why.
