# APISeqFewShot
This is the code repository for up-coming paper "*A Novel Few-Shot Malware Classification Approach for Unknown Family Recognition with Multi-Prototype Modeling*" submit to ***Computer & Security***. All the model implementation details and baselines are included and all the models are implemented with **Python 3.7+** and **PyTorch 1.4(cuda in default)**.


## Declaration
This code repository has always been used for experiments of our research, thus it is designed for engineering purpose, where some code logic is hard to explained clearly in short words. **We DO NOT PROMISE it work properly on other people's machines.** We publicate this repository to faciliate the few-shot malware classification research and if you use our code in your work, please refer to our paper. If there are any problems, please raise them in 'issues'.

Implementation of our SIMPLE model referred to IMP implementation: [IMP](https://github.com/k-r-allen/imp)


## How to Run this Code

### Note
**This repository DOES NOT contain any data files for size limitation.** To replicate our work, visit [VirusShare website](https://virusshare.com) or [APIMDS dataset release website](http://ocslab.hksecurity.net/apimds-dataset) to download the datasets and follow the preprocessing steps in our paper to make suitable datasets to run this code.

### Preprocessing Instructions
Almost all the preprocessing-related code are located in *preliminaries* package. **We assume all the sequence data files are in 'JSON' form, where the api sequence is a list with key 'apis'**. We DO NOT RECOMMAND use our code to do preprocessing work because we divide the process into many subprocesses and dump log for each subprocess, instead of making them a end-to-end pipeline. This brings in confusion when using our code when preprocessing, thus we recommand to write your own code to do preprocessing, including several main steps:

1. Run malware binary samples in Cuckoo sandbox to get api sequence in JSON form
   
2. Drop sequence files where sequence has length less than 10, according to our paper

3. Remove redundancy in api sequence (successive invocation of the same api twice or more), which can also be done by calling *preliminaries/preprocessing.py/removeApiRedundance* (set 'class_dir' to 'False' assumes that each JSON file are located in an independent directory, with the same name of JSON file). This function will overwrite the original JSON file, so please remember to backup.

4. Extract N-gram (*N=3* in our paper) of api sequence and replace the api sequence item with api N-gram item
   
5. Calculate TF-IDF value of each N-gram item (you may refer to our code *extractors/TFIDF.py*). Leave only top-k (*k=2000* in our paper) TF-IDF value N-gram item in sequence and **repeat Step.2**

6. Use some scan tools (like VirusTotal) to make analysis report for each malware sample, then extract a sole label for each malware sample by some tools (like AVClass). Collect samples with the same family tag

7. Sample a fixed number (20 in our paper) of sequence files from each family to a directory, whose name is the same as the family

8. Run our code *preliminaries/dataset.py/makeDatasetDirStruct* to make dataset directory structure, and move all the family folders to 'all' directory

9. Train GloVe embedding on all the family folders and output a mapping file and a embedding matrix, you may refer to our code *uitls/GloVe.py* (it requires for Python2 env and glove-python package). Move the index mapping file (rename to wordMap.json) and word embedding matrix (rename to matrix.npy, NumPy.ndarray type) to 'data' folder.

10. Split the families in 'all' dataset to 'train', 'validate' and 'test', according to our paper (move family folders)

11. Run our code *run/make_data_files.py* to generate torch data files to be loaded (change the parameters before running)

After these steps, the whole dataset directory structure look like this:

```
(Your dataset name)
    |-- all
        |-- (all family folders...)
    |-- train
        |-- (train family folders...)
    |-- validate
        |-- (validate family folders...)
    |-- test
        |-- (test family folders...)
    |-- doc    
    |-- models
    |-- data
        |-- train
            |-- data.npy
            |-- idxMapping.json
            |-- seqLength.json
        |-- validate
            |-- data.npy
            |-- idxMapping.json
            |-- seqLength.json
        |-- test
            |-- data.npy
            |-- idxMapping.json
            |-- seqLength.json
        |-- wordMap.json
        |-- matrix.npy
```

### Hyper-Parameter Setting
To ensure reproducibility, hyper-parameters are configured by JSON form config file, located in *run/runConfig.json*. Everytime you start to train a model, a doc directory will be made in 'doc' with the name of 'version' parameter. **So remember to change the 'version' when you intend to train a new model to prevent from unexpected overwriting**. Other important parameters are introduced in the following sections. Trained torch model parameter state dictionaries are saved under 'models'. 

**Before running, add an item to 'Ns' key-value object in runConfig.json, where key equals your dataset name and value equals how many items are there in each family. Also, please add an object to 'platform-node' item, where key equals the host name, value equals the base path your datasets are located in.** (host name can be obtained by calling *platform.node()*)


### Train
Just make your datasets and run *run/run.py*, it will load the data files and run config to configure the running settings. If you want to visualize the training process, set 'useVisdom' in runConfig.json to 'true', and turn on the visdom server by 
``` shell
python -m visdom.server
```
Then visit [localhost:8097](http://localhost:8097/).

Training statistics will display in console like: 
```
----------------Test in Epoch 99--------------------
***********************************
train acc:  0.642
train loss:  5.168325721025467
----------------------------------
val acc: 0.5968000000000001
val loss: 2.0870852395892143
----------------------------------
best val loss: 2.0870852395892143
best epoch: 99
time consuming:  19.70339059829712
time remains:  01:38:11
***********************************
```

### Test
It is very similar to training models but you need to change testConfig.json to configure test settings. Note that if you want to run testing on 'train' or 'validate' subset, set 'testSubdataset' to 'train' or 'validate', but it is 'test' in default.

Testing statistics will display in console like: 
```
200 Epoch
--------------------------------------------------
Acc: 0.926400
Loss: 0.231254
Current Avg Acc: 0.922600
Time: 3.38
time remains:  00:00:27
--------------------------------------------------
```

 

## Project Origanization
Code files are organized by function and locate in different folders.


### run
This folder mainly contains the lauching script for training and testing experiment and  run-time parameter controlling related files. 

- **config.py** 
    Most relates to the running configuration saving and version controlling.

- **run.py**
    Actual lauching script for training experiments. This file contains training config reading, manager initialization, dataset parsing, model running/validating, line plotting and result reporting. It will read the *runConfig.py* file to load the training parameters.

- **test.py**
    Actual lauching script for testing experiments. This file is highly similar to *run.py* but only focus on testing, so some parts are removed.

- **finetuning.py**
    Actual lauching script for finetuning experiments. It reads the *ftConfig.py* to load the fine-tuning parameters and uses SGD optimizer to fine-tune an untrained model(SIMPLE) for several iterations. Then the fine-tuned model is runned for testing to produce the final results. Note that we use a linear layer to generate the classification results after sequence embedding.

- **runConfig.json**
    Configuration file for training experiments. Most important items include: model, shot(k), way(n), dataset, maximum sequence length, optimizer and learning rate. Note that the base paths of datasets are required to state in this file, as "platform-node" item. On different hosts, dataset bases can be altered by modifying this item.

- **testConfig.json**
    Configuration file for testing experiments. This file is highly similart to *runConfig.json*, except for: testing version and model name, testing iteration and testing subdataset(train, validate or test).

- **starter.py**
    Actual lauching script for some trivial dataset manipulation programs. Typical manipulations include: data file collection and generating(sequence length setting), dataset splitting, dataset preprocessing and etc. In short, this file calls the functions in other modules to complete the target.

---

### preliminaries
This folder mainly contains the modular scripts for dataset preprocessing, dataset formalization, malware scanning and malware labeling. 
    
  - **avclass.py**
  It most contains the auxiliary scripts for supporting avclass tool.

  - **dataset.py**
  It contains the implementations for making and splitting dataset. Original data is located at "*(dataset)/train*", "*(dataset)/validate*" and "*(dataset)/test*". Then these distributed data are collected and wrapped in a single matrix file and sequence lengths are also recored in another json file, which are all located in "*(dataset)/data/train/*", "*(dataset)/data/validate/*" and "*(dataset)/data/test/*".

  - **embedding.py**
  Aggregates the malware sequences in the dataset and trains the W2V embedding model.

  - **preprocessing.py**
  It contains the implementations for: API alias mapping, sequence redundancy removing, API frequency statistics, API repeated subsequence removing and class-wise API data file collecting and etc.

  - **virustotal.py**
  Most relates to some malware scanning operations using the service provided by VirusTotal website.

---

### utils
It contains the utiliy modules to facilate the dataset generating, training/testing experiments, path managing, statistics managing and functional utilities. Each file relates to one particular functional utility.

  - **color.py**
  It contains some color managing code to provide some convenient APIs to get some 
  plotting colors.

  - **error.py**
  It contains a Reporter class to manage the run-time errors. It will record the message of the error/warning and report the final statistics of the whole process.

  - **file.py**
  Some file-related operations, such as JSON reading and dumping, directory deleting and list dumping.

  - **GloVe.py**
  Actual lauching file for GloVe embedding training. It relies on the *glove* module and runs under *Python 2.7*. It receives the whole sequence matrix and output the word embedding matrix, which will be saved as a NumPy file. 

  - **init.py**
  Model initilization functions.

  - **magic.py**
  Randomization implementation utilies, such as random seed, random list and  sampling.

  - **manager.py**
  Some useful managers to simplify the operations.
    - **PathManager**
    Dataset path manager. Given the dataset name, it will automatically read the dataset bases from *runConfig.py* and generate a series of paths, such as data file path(given the subdataset type), word embedding path, sequence length path, running document path and etc.
    - **TrainStatManager**
    Statistics manager for training/validating experiments. It can record the accuracy/loss and epoch time consuming data during training/validating. Besides, it will save the best-performed model state dictionary after validating. At last, it can report the overall training statistics.
    - **TestStatManager**
    Statistics manager for testing experiments. It has similar functions as TrainStatManager but used in testing stage.
    - **TrainConfigManager**
    Configuration reading manager of *runConfig.json* or *testConfig.json*. It integrates some parameters together and return them as a whole part.

  - **matrix.py**
  Some torch matrix-related functions, such as batch-dot and matrix reduce.

  - **plot.py**
  It contains a Visdom visulization class *VisdomPlot* and a line plot function of  matplotlib. Visdom interactions can be done through the static class.

  - **stat.py**
  Some statistic utility function like parameter statistics and confidence interval calculation.

  - **timer.py**
  A step timer class that compute the step time interval to record the time slice between consecutive validating or testing displays and report the estimated remaining running time.

  - **training.py**
  Some utility functions called during training, like making sequence batch, parse the task parameters(shot,way...), dynamic routing, sequence masking and etc.

  - **zip.py**
  Utility functions to unzip the zipped files.

---

### components
Many neural network and meta-learning components including sampler, episode task, dataset entity classes, training procedures, sequence embeddings and etc. 

- **datasets.py**
It contains the torch dataset entity class *SeqFileDataset*, which reads the sequence data in a single file into the memory. It also reads in the sequence length information to output the sequence and length in pair.

- **sampler.py**
It contains the episode-training task sampler class *EpisodeSampler*. Given the label space(sampled classes) and class-wise seed, it samples support set and query set from the dataset and keep these two sets nonoverlapped.

- **task.py**
Some model-dependent task entity classes which integrates label space sampling, class-wise sampling(by generating class seeds for samplers), dataloader instantiation(use the batch making utility function in *utils/training.py*) and label normalization. It leaves a function *episode()* to be implemented for different models, which randomly samples a task from the dataset and returns the episode data as support/query set form. 

- **procedure.py**
Detailed training procedures for different kind of models, such as fast-adaption based(MAML family), meta-loss-optimization based(mostly metric-based), infinite mixture prototypes based(IMP and our SIMPLE) and etc. These methods have varied ways to forward the model or compute the loss value, so they may have different implementaions.