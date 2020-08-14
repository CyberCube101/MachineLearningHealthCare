import numpy as np
import pandas as pd
import sklearn
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
ConvergenceWarning('ignore')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection

# E. coli promoter gene sequences (DNA) classification
# If class positive, it is a promoter

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'

names = ['Class', 'id', 'Sequence']

df = pd.read_csv(url, names=names)

# print(df.iloc[:5])  # there is a \t which we need to remove in preprocessing
# Each column in df is a series

classes = df.loc[:, 'Class']  ## all rows for column class

# generate list of DNA sequences

sequences = list(df.loc[:, 'Sequence'])
dataset = {}
# now we loop through list and split into individual nucleotides and remove tab chars

for i, seq in enumerate(sequences):
    nucleotides = list(seq)
    nucleotides = [x for x in nucleotides if x != '\t']
    nucleotides.append(classes[i])
    dataset[i] = nucleotides

# turn back into df

dframe = pd.DataFrame(dataset)
dframe = dframe.transpose()

dframe.rename(columns={57: 'Class'}, inplace=True)

# record value counts for each sequence

series = []

for name in dframe.columns:
    series.append((dframe[name].value_counts()))

info = pd.DataFrame(series)
details = info.transpose()

# switch to numerical data using pd.get_dummies()

num_df = pd.get_dummies(dframe)

# remove 1 of the class columns since we dont need both

df = num_df.drop(columns=['Class_-'])
df.rename(columns={'Class_+': 'Class'}, inplace=True)
# print(df.iloc[60])

###### Data preprocessing complete, now for the ML work

scoring = 'accuracy'

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])
seed = 1

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=seed)

names = ["KNN", "GP", "DT", "RF", "NN", "AB", "NB", "SVM_Linear", "SVM_RBF", "SVM_SIGMOID"]

classifiers = [KNeighborsClassifier(n_neighbors=3),
               GaussianProcessClassifier(1.0 * RBF(1.0)),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               MLPClassifier(alpha=1),
               AdaBoostClassifier(),
               GaussianNB(),
               SVC(kernel='linear'),
               SVC(kernel='rbf'),
               SVC(kernel='sigmoid')]

models = zip(names, classifiers)

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(" Training {0}: {1} ({2})".format(name, cv_results.mean(), cv_results.std()))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
