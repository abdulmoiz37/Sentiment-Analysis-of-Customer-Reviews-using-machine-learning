To implement this project author has used following Steps:
1) Data Collection: Using this module we will upload AMAZON reviews dataset to application
2) Data Preprocessing: using this module we will read all reviews and then remove stop words, special symbols, punctuation and numeric data from all reviews and after applying Preprocessing we will extract features from all reviews.
3) Features Extraction: here we will apply TF-IDF (term frequency Inverse Document Frequency) algorithm to convert string reviews into numeric vector. Each word count will be put in vector in place of words.
4) Run SVM Algorithm: We will apply SVM algorithm on TF-IDF vector to train SVM algorithm and then we apply test data on SVM trained model to calculate SVM prediction accuracy
5) Run Naïve Bayes Algorithm: We will apply Naïve Bayes algorithm on TF-IDF vector to train Naïve Bayes algorithm and then we apply test data on Naïve Bayes trained model to calculate Naïve Bayes prediction accuracy
6) Run Decision Tree Algorithm: We will apply Decision Tree algorithm on TF-IDF vector to train Decision Tree algorithm and then we apply test data on Decision Tree trained model to calculate Decision Tree prediction accuracy
7) Detect Sentiment from Test Reviews: Using this module we will upload test reviews and then ML algorithm will predict sentiment for each review and in below test reviews dataset we can see there is no sentiment value and ML will predict sentiment for each test value
8) Getting The accuracy Graph: The Three Algorithms can be displayed using a visual bar Graph