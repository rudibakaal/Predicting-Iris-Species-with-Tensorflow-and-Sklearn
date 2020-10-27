
# Predicting-Iris-Species-with-Tensorflow-and-Sklearn

## Motivation
Machine learning multi-label classification task on the famous Iris Dataset.

The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other[1].

All features were standardised via sklearn's StandardScaler class. The categorical labels were also numerically converted using sklearn's Labelencoder class. Finaly, the converted class vector integer labels were converted into a binary matrix to facilitate categorical crossentropy with keras' to_categorical class.

## Neural Network Topology and Results Summary

The categorical crossentropy loss function was leveraged along with the Adam optimizer for this classification problem.

![model](https://user-images.githubusercontent.com/48378196/96961401-4be81500-1550-11eb-9cd2-4e0f682c3b56.png)

After 200 epochs, the training and validation set classifiers reach 97% and 93% accuracy, respectively, in the predicting Iris species. 

![iris](https://user-images.githubusercontent.com/48378196/97241055-112df780-1844-11eb-9e4a-72037a48e828.png)

## License
[MIT](https://choosealicense.com/licenses/mit/) 

## References
[1]  Fisher,R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950)
