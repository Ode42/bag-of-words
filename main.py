from cgi import test
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"
    
train_x = ["I love the book", "This is a great book", "The fit is great", "I like the shoes"]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))
train_x_vectors = vectorizer.fit_transform(train_x)

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

test_x = vectorizer.transform(["This is a fine book"])

print(clf_svm.predict(test_x))