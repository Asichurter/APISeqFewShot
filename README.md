# APISeqFewShot
This is the code repository for up-coming paper "*Recognize Unknown Malware Families Using Limited Samples through Few-Shot Multi-Prototype Modeling*". All the model implementation details and baselines are included and all the models are implemented with **Python 3.7+** and **PyTorch 1.4(cuda)**.

1. Project Origanization
Code files are organized by function and locate in different folders.

- run
This folder mainly contains the lauching script for training and testing experiment and  run-time parameter controlling related files. 
    - config.py
    Most relates to the running configuration saving and version controlling.
    - run.py
    **Actual lauching script for training experiments**. This file contains training config reading, manager initialization, dataset parsing, model running/validating, line plotting and result reporting. It will read the *runConfig.py* file to load the training parameters.
    - test.py
    **Actual lauching script for testing experiment**s. This file is highly similar to *run.py* but only focus on testing, so some parts are removed.
    - finetuning.py
    **Actual lauching script for finetuning experiments.** It reads the *ftConfig.py* to load the fine-tuning parameters and uses SGD optimizer to fine-tune an untrained model(SIMPLE) for several iterations. Then the fine-tuned model is runned for testing to produce the final results. Note that we use a linear layer to generate the classification results after sequence embedding.
    - runConfig.json
    **Configuration file for training experiments.** Most important items include: model, shot(k), way(n), dataset, maximum sequence length, optimizer and learning rate. Note that the base paths of datasets are required to state in this file, as "platform-node" item. On different hosts, dataset bases can be altered by modifying this item.
    - testConfig.json
    **Configuration file for testing experiments.** This file is highly similart to *runConfig.json*, except for: testing version and model name, testing iteration and testing subdataset(train, validate or test).
    - starter.py
    **Actual lauching script for some trivial dataset manipulation programs.** Typical manipulations include: data file collection and generating(sequence length setting), dataset splitting, dataset preprocessing and etc. In short, this file calls the functions in other modules to complete the target.

- preliminaries
This folder mainly contains the modular scripts for dataset preprocessing, dataset formalization, malware scanning and malware labeling. 
    - avclass.py
    It most contains the auxiliary scripts for supporting avclass tool.
    - dataset.py
    It contains the implementations for making and splitting dataset. Original data is located at "(dataset)/train", "(dataset)/validate" and "(dataset)/test". Then these distributed data are collected and wrapped in a single matrix file and sequence lengths are also recored in another json file, which are all located in "(dataset)/data/train/", "(dataset)/data/validate/" and "(dataset)/data/test/".
    - embedding.py
    Aggregates the malware sequences in the dataset and trains the W2V embedding model.
    - preprocessing.py
    It contains the implementations for: API alias mapping, sequence redundancy removing, API frequency statistics, API repeated subsequence removing and class-wise API data file collecting and etc.
    - virustotal.py
    Most relates to some malware scanning operations using the service provided by VirusTotal website.

- utils
It contains the utiliy modules to facilate the dataset generating, training/testing experiments, path managing, statistics managing and functional utilities. Each file relates to one particular functional utility.
    - color.py
    It contains some color managing code to provide some convenient APIs to get some plotting colors.
    - error.py
    It contains a Reporter class to manage the run-time errors. It will record the message of the error/warning and report the final statistics of the whole process.
    - file.py
    Some file-related operations, such as JSON reading and dumping, directory deleting and list dumping.
    - GloVe.py
    Actual lauching file for GloVe embedding training. It relies on the *glove* module and runs under *Python 2.7*. It receives the whole sequence matrix and output the word embedding matrix, which will be saved as a NumPy file. 
    - init.py
    Model initilization functions.
    - magic.py
    Randomization implementation utilies, such as random seed, random list and  sampling.
    - manager.py
    Some useful managers to simplify the operations.
      - PathManager
      Dataset path manager. Given the dataset name, it will automatically read the dataset bases from *runConfig.py* and generate a series of paths, such as data file path(given the subdataset type), word embedding path, sequence length path, running document path and etc.
      - TrainStatManager
      Statistics manager for training/validating experiments. It can record the accuracy/loss and epoch time consuming data during training/validating. Besides, it will save the best-performed model state dictionary after validating. At last, it can report the overall training statistics.
      - TestStatManager
      Statistics manager for testing experiments. It has similar functions as TrainStatManager but used in testing stage.
      - TrainConfigManager
      Configuration reading manager of *runConfig.json* or *testConfig.json*. It integrates some parameters together and return them as a whole part.
    - matrix.py
    Some torch matrix-related functions, such as batch-dot and matrix reduce.
    - plot.py
    It contains a Visdom visulization class *VisdomPlot* and a line plot function of  matplotlib. Visdom interactions can be done through the static class.
    - stat.py
    Some statistic utility function like parameter statistics and confidence interval calculation.
    - timer.py
    A step timer class that compute the step time interval to record the time slice between consecutive validating or testing displays and report the estimated remaining running time.
    - training.py
    Some utility functions called during training, like making sequence batch, parse the task parameters(shot,way...), dynamic routing, sequence masking and etc.
    - zip.py
    Utility functions to unzip the zipped files.

- components
Many neural network and meta-learning components including sampler, episode task, dataset entity classes, training procedures, sequence embeddings and etc. 
    - datasets.py
    It contains the torch dataset entity class *SeqFileDataset*, which reads the sequence data in a single file into the memory. It also reads in the sequence length information to output the sequence and length in pair.
    - sampler.py
    It contains the episode-training task sampler class *EpisodeSampler*. Given the label space(sampled classes) and class-wise seed, it samples support set and query set from the dataset and keep these two sets nonoverlapped.
    - task.py
    Some model-dependent task entity classes which integrates label space sampling, class-wise sampling(by generating class seeds for samplers), dataloader instantiation(use the batch making utility function in *utils/training.py*) and label normalization. It leaves a function *episode()* to be implemented for different models, which randomly samples a task from the dataset and returns the episode data as support/query set form. 
    