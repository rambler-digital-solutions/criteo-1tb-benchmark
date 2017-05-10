# Criteo 1 TiB benchmark



## Table of contents

* [Introduction](#introduction)
* [Task and data](#task-and-data)
* [Algorithms](#algorithms)
* [Setup](#setup)
* [Experiment layout](#experiment-layout)
  * [Hyperparameter optimization](#hyperparameter-optimization)
  * [Data format for Spark.ML](#data-format-for-sparkml)
  * [Code](#code)
* [Results](#results)
  * [Local training - Vowpal Wabbit & XGBoost](#local-training---vowpal-wabbit--xgboost)
  * [Distributed training - Spark.ML](#distributed-training---sparkml)
  * [Distributed training - Time vs. Cores](#distributed-training---time-vs-cores)
  * [Comparison of local vs. remote](#comparison-of-local-vs-remote)
* [Conclusion](#conclusion)
* [Resources](#resources)


## Introduction
[_(back to toc)_](#table-of-contents)

This project is a minimal benchmark of applicability of several implementations of machine learning algorithms to training on big data. Our main focus is [Spark.ML](http://spark.apache.org/mllib/) and how it compares to commonly used single-node machine learning tools Vowpal Wabbit and XGBoost in terms of scaling to terabyte (billions of lines) train data. Quick web search shows that many people tested Spark but not on tasks requiring a cluster so they are mostly single-node tests.

This project is inspired by https://github.com/szilard/benchm-ml but is focused on training models on billions of lines of train data, including Spark.ML in multinode cluster environment.



## Task and data
[_(back to toc)_](#table-of-contents)

Our target application is prediction of click-through ratio (CTR) of banners in online advertising. [Criteo released](http://labs.criteo.com/2015/03/criteo-releases-its-new-dataset/) an industry-standard open dataset which represents banner impressions in online advertising during the timespan of 24 days. It is more than 1 terabyte in size and consists of more than 4 billion lines of data. Each line represents a banner impression and contains 40 columns separated by tabulation:

- the first column is a label - {0, 1} - 1 meaning the banner was clicked and 0 otherwise;
- 13 numeric columns;
- 26 categorical columns with categories being 32-bit hashes.

This is how it looks like:

![Dataset schema](images/dataset.png)

All the data except the last day was concatenated and sampled into training sets of 10ⁿ and 3×10ⁿ lines with `n ∈ {4, 5, ..., 9}` (i.e. train samples' sizes are 10k, 30k, 100k, ..., 1kkk, 3kkk lines). The last day was used for testing - a sample of one million lines was taken from it. All samples were converted to

- LibSVM format for training XGBoost models and as a source for the transformation to Spark.ML DataFrame;
- Vowpal Wabbit data format.

Data for Spark.ML models was processed on-the-fly from LibSVM format:

- into a dataset of tuples of "label" (integer) and "features" (SparseVector of size 10⁵ using [hashing trick](https://en.wikipedia.org/wiki/Feature_hashing#Feature_vectorization_using_the_hashing_trick) for all features) for Spark.ML LogisticRegression;
- into a dataset of tuples of "label" (integer) and "features" (SparseVector of size 39 taken as-is from corresponding columns, see below) for Spark.ML RandomForestClassifier.



## Algorithms
[_(back to toc)_](#table-of-contents)

Historically, we make use of [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) and [XGBoost](https://github.com/dmlc/xgboost) exploiting "Local train + Distributed apply" scenario. Our task was to run a performance test of our currently used approach and Spark.ML library algorithms.

We used the following non-distributed algorithms:

- Vowpal Wabbit - it implements logistic regression with a hashing trick and reads the data only once never keeping more than one sample in memory (it is an out-of-core implementation);
- in-memory XGBoost - gradient-boosted trees implementation that (by default) loads the whole data into memory (which is faster than multiple reads from disk, but we are limited in size by machine memory);
- out-of-core XGBoost - a variant of XGBoost training which uses an on-disk cache; this is slower (compared to the in-memory variant) but potentially we can train on the data limited in size only by the size of HDD.

Spark.ML contains following classification algorithms:

- [LogisticRegression](http://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression),
- [RandomForestClassifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier),
- [NaiveBayes](http://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes),
- [DecisionTreeClassifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier),
- [GBTClassifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier),
- [MultilayerPerceptronClassifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier).

Our preliminary research shows that four of the algorithms are not well-suited for our task of CTR prediction:

- NaiveBayes provides significantly worse logistic loss (which is an essential metric of CTR models' quality) than all other models;
- DecisionTreeClassifier suffers in quality in comparison to the RandomForestClassifier but still requires roughly the same amount of time to train;
- GBTClassifier (Spark.ML implementation of gradient-boosted trees) and MultilayerPerceptronClassifier do not support prediction of probabilities that are required by the task (these two models are not shown on graphs above).

![ROC AUC](images/roc_auc.cluster_selection.png) ![Log loss](images/log_loss.cluster_selection.png) ![Training time](images/train_time.cluster_selection.png)

Thus we use only LogisticRegression and RandomForestClassifier for our testing purposes.



## Setup
[_(back to toc)_](#table-of-contents)

Local models were trained on a 12-core (24-thread) machine with 128 GiB of memory. Distributed training was performed on our production cluster (total capacity is approximately 2000 cores and 10 TiB of memory); for the experiment a small part of resources has been allocated - 256 cores and 1 TiB of memory for training on datasets upto 300 million of lines and 512 cores and 2 TiB of memory for training on one billion and 3 billion lines of train data. 4 cores and 16 TiB of memory per Spark executor was used.

For the experiment we used Vowpal Wabbit 8.3.0, XGBoost 0.4 and Spark 2.1.0 running on a Hadoop 2.6 cluster (using YARN as a cluster manager).



## Experiment layout

### Hyperparameter optimization
[_(back to toc)_](#table-of-contents)

Our first idea was to skip models' hyperparameters optimization completely, but unfortunately XGBoost's default hyperparameters are not good enough for training even on million lines of data - the default number of trees is only 10, and it hits the ceiling quite soon:

![ROC AUC](images/roc_auc.why_optimize.png) ![Log loss](images/log_loss.why_optimize.png)

These figures reminded us that production usage of any machine learning model is associated with optimization of its hyperparameters, and in our experiment we should do the same. For optimization of models' hyperparameters (including Spark.ML ones) we used the million-line sample of train data and 5-fold cross validation for metric (log loss) averaging.



### Data format for Spark.ML
[_(back to toc)_](#table-of-contents)

We tried to use [one-hot-encoding](https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science) of categorical features, but due to very large number of unique values it turned out to be very time and memory consuming, so for Spark.ML we decided to try the hashing trick. Spark.ML LogisticRegression was trained using this approach. We sticked to hashing space of 10⁵ hashes as it turned out to give about the same quality as VW on large samples. Taking less hashes usually leads to better quality on smaller data (because of less overfitting) and worse quality on bigger data (because some patterns in data are consumed by collisions in hashing space):

![ROC AUC](images/roc_auc.lr_hash_size.png) ![Log loss](images/log_loss.lr_hash_size.png)

RandomForestClassifier was very slow to train even with a thousand hashes, so we used "as-is" format for it:

- all numeric features were converted to elements of SparseVector as-is;
- all categorical features were converted to elements of SparseVector by interpreting the hashes as 32 bit numbers.



### Code
[_(back to toc)_](#table-of-contents)

All work was performed in Jupyter notebooks in Python. Notebooks:

- [experiment_local.ipynb](notebooks/experiment_local.ipynb) was used for preparing the data and training of the local models;
- [experiment_spark_lr.ipynb](notebooks/experiment_spark_lr.ipynb) and [experiment_spark_rf.ipynb](notebooks/experiment_spark_rf.ipynb) for training Spark.ML LogisticRegression and RandomForestClassifier accordingly.



## Results

### Local training - Vowpal Wabbit & XGBoost
[_(back to toc)_](#table-of-contents)

![ROC AUC](images/roc_auc.local.png) ![Log loss](images/log_loss.local.png) ![Train time](images/train_time.local.png) ![Maximum memory](images/maximum_memory.local.png) ![CPU load](images/cpu_load.local.png)

Some observations:

- our main concern about an out-of-core training of XGBoost was that it would not produce the same quality as its in-memory variant due to the approximate splitting algorithm; however, in-memory XGBoost and out-of-core XGBoost turned out to provide about the same level of quality, but out-of-core variant is about an order of magnitude slower;
- in-memory XGBoost is about an order of magnitude slower than Vowpal Wabbit on the same amount of train data;
- Vowpal Wabbit was able to give about the same quality as XGBoost trained on an order of magnitude smaller sample.



### Distributed training - Spark.ML
[_(back to toc)_](#table-of-contents)

![ROC AUC](images/roc_auc.cluster.png) ![Log loss](images/log_loss.cluster.png) ![Train time](images/train_time.cluster.png)

We made the following conclusions:

- RandomForestClassifier is quite slow, and it is even slower when the data consists of large vectors;
- LogisticRegression is hard to set up for smaller samples and for bigger samples at the same time - either it overfits on small data or it cannot extract patterns due to more aggressive hashing trick.



### Distributed training - Time vs. Cores
[_(back to toc)_](#table-of-contents)

To check how model training scales to multi-core setup we made a quick test where we increased the number of cores and measured training time for every step. To make it fast we used a 10⁷ sample of train data and checked training time for a number of cores from 5 to 50 in steps of 5. In order to eliminate the uncertainty brought forth by running the test in parallel with production tasks, for this test we created a standalone Spark cluster running on three machines with a total of ≈50 cores and ≈200 GiB of memory.

![Time vs. cores](images/time_vs_cores.png)

Training time dropped quite fast when we went from 5 to 15 cores but slowed down afterwards and completely ceased to improve by the mark of 40 cores (even growing a little on transition from 40 to 45 cores). The main idea we have extracted from this figure is that one should not increase amount of resources beyond minimum required, so that work distribution and aggregation overhead would be cheaper than potential improvement of speed by parallelization.



### Comparison of local vs. remote
[_(back to toc)_](#table-of-contents)

![ROC AUC](images/roc_auc.local_and_cluster.png) ![Log loss](images/log_loss.local_and_cluster.png) ![Train time](images/train_time.local_and_cluster.png)

We can see that:

- on large datasets (100 million of lines and more) Spark.ML is faster than both Vowpal Wabbit and XGBoost; maybe it is possible to make it even faster by finding the best cluster setup for each size of training sample (we had not done this work);
- however it is slow when working with large vectors - steps should be taken in order to find a balance between quality and speed;
- for small tasks Spark introduces overhead that can be more expensive than it is possible to gain by computing the task in parallel (well, this is true for parallel computing in general).



## Conclusion
[_(back to toc)_](#table-of-contents)

The best quality measured by logarithmic loss (which is a metric of choice for CTR prediction) was achived using XGBoost - no matter in-memory or out-of-core, as they both seem to be equal in quality - on a sample smaller in size than other models required for the same level of quality.
However XGBoost is very slow on big samples in out-of-core setup thus it was not rational to test it on a 300kk sample and above (training the in-memory setup on large samples was also not possible due to memory restrictions).
The highest ROC AUC was reached by Vowpal Wabbit on one-billion-line train sample, strangely decreasing in quality by three-billion-line sample.
Spark.ML LogisticRegression is quite close in quality to Vowpal Wabbit, and maybe it can be made even better by increasing the feature space size (which is 100k hashes in current setup).
Spark.ML LogisticRegression appeared to be considerably faster than VW on billion-line samples and maybe it can be made even faster by optimizing the allocated resources.
Spark.ML RandomForestClassifier stopped increasing in quality quite early and it is also quite slow.



## Resources
[_(back to toc)_](#table-of-contents)

Results in table format can be found ![here](results). Scala scripts used for faster conversion and sampling can be found ![here](scripts/conversion) - these scripts can be used from ![spark-shell](http://spark.apache.org/docs/latest/quick-start.html#basics) using `:load` command. Scripts for running VW & XGBoost and plotting outside of the Jupyter notebooks can be found ![here](scripts/running).
