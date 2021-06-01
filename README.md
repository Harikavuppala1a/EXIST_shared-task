This repository contains code for our paper titled "Knowledge-based Neural Framework for SexismDetection and Classification". 


Abstract: Sexism, a prejudice that causes enormous suffering, manifests in blatant as well as subtle ways. As sexist content towards women is increasingly spread on social networks, the automatic detection and categorization of these tweets/posts can help social scientists and policymakers in research, thereby combating sexism. In this paper, we explore the problem of detecting whether a Twitter/Gab post is sexist or not. We further discriminate the detected sexist post into one of the fine-grained sexist categories. We propose a neural model that combines tweet representations obtained using the RoBERTa model and linguistic features such as Empath, Hurtlex, and Perspective API by involving recurrent components. We also leverage the unlabeled sexism data to infuse the domain-specific transformer model into our framework. Our proposed framework also features a knowledge module comprised of emoticon and hashtag representations from the tweet to infuse the external knowledge-specific features into the learning process. Several proposed methods outperform various baselines across several standard metrics.


Instructions for using our code: 
* To run the DL baselines, run python main.py data/config_dl.txt.
* For traditional baselines, run python TraditionalML.py data/config_traditional_ML.txt 
* To run the proposed methods, run python models.py config.txt. 
* All the hyperparameters are available in the Tuned hyper-parameters.pdf file.
* analysis.py - To generate the analysis. class_wise_analysis_chart.py - To generate the class wise F scores. 
* features.py - It has all the linguistic features.
* load_data.py - To load the data. models.py - It has all the proposed models
* preprocessing.py - It has code to preprocess the data. translate.py - To translate the spanish posts to english posts.
