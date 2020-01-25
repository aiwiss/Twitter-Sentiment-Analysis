from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

# declare classifiers
classifier_names = [
    'KNN',
    'D-Tree',
    'RF',
    'BernoulliNB',
    'MultinomialNB',
    'LinearSVC',
    'MLP'
]

# declare the list of classifiers used within the project
classifiers = [
    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    DecisionTreeClassifier(max_depth=7, min_samples_split=4),
    RandomForestClassifier(max_depth=120, n_estimators=150, n_jobs=-1),
    BernoulliNB(alpha=0.1),
    MultinomialNB(alpha=0.01),
    LinearSVC(C=3),
    MLPClassifier(alpha=0.1, hidden_layer_sizes=(10,10))
]

# number of features per classifier, index matches classifer name and the object itself
n_feats = [600, 1000, 2500, 1500, 2500, 2500, 2500]