import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydotplus
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    create_dataset = True
    if create_dataset:
        df = pd.read_csv("students.csv", error_bad_lines=False)  #eliminated all the bad data
        df.dropna(axis=0, inplace=True)

        gender = df["gender"]
        all_gender = []  # see all the strings in the effects row
        for index, row in df.iterrows():  # iterate
            string_ef = row['gender']  # choose the row effects
            for genders in string_ef.split(" "):  # split the strings
                if genders not in all_gender:
                    all_gender.append(genders)

        raw_data = {'female': [], 'male': []} #Create a library with the info of the columns of gender
        for index, row in df.iterrows():
            gender_row = row['gender'].split(' ') #split the strings
            for gender in all_gender:
                if gender in gender_row:      #append a 1 or a 0 if the label is in the row
                    raw_data[gender].append(1)
                else:
                    raw_data[gender].append(0)
        for gender, values in raw_data.items():
            df[gender] = values

        df['parental level of education'] = df['parental level of education'].str.replace(r"[\"\',]", '')                #quit ' from the names
        paren = df["parental level of education"]
        all_parental = []  # see all the strings in the effects row
        for index, row in df.iterrows():  # iterate
            string_ep = row['parental level of education']  # choose the row effects
            for parents in string_ep.split("  "):  # split the strings
                if parents not in all_parental:
                    all_parental.append(parents)

        raw_data = {'bachelors degree': [], 'some college': [], 'masters degree': [], 'associates degree': [],
                    'high school': [],'some high school': []}
        for index, row in df.iterrows():
            paren_row = row['parental level of education'].split('  ')  # split the strings
            for parent in all_parental:
                if parent in paren_row:
                    raw_data[parent].append(1)
                else:
                    raw_data[parent].append(0)
        for parent, values in raw_data.items():
            df[parent] = values

        lunch = df["lunch"]
        all_lunch = []  # see all the strings in the effects row
        for index, row in df.iterrows():  # iterate
            string_l = row['lunch']  # choose the row effects
            for lunches in string_l.split("  "):  # split the strings
                if lunches not in all_lunch:
                    all_lunch.append(lunches)
         # print effects split

        raw_data = {'standard': [], 'free/reduced': []}
        for index, row in df.iterrows():
            lunch_row = row['lunch'].split(' ')  # split the strings
            for lunch in all_lunch:
                if lunch in lunch_row:
                    raw_data[lunch].append(1)
                else:
                    raw_data[lunch].append(0)
        for lunch, values in raw_data.items():
            df[lunch] = values

        preparation = df["test preparation course"]
        all_preparation = []  # see all the strings in the effects row
        for index, row in df.iterrows():  # iterate
            string_pre = row['test preparation course']  # choose the row effects
            for tps in string_pre.split(" "):  # split the strings
                if tps not in all_preparation:
                    all_preparation.append(tps)
         # print effects split

        raw_data = {'none': [], 'completed': []}
        for index, row in df.iterrows():
            test_row = row['test preparation course'].split(' ')  # split the strings
            for preparation in all_preparation:
                if preparation in test_row:
                    raw_data[preparation].append(1)
                else:
                    raw_data[preparation].append(0)
        for preparation, values in raw_data.items():
            df[preparation] = values

        df.drop(['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'], axis=1,
                inplace=True)  # drop the columns

        df['average'] = (df['math score']+df['writing score']+df['reading score'])/3

        df['average'] = df['average'].apply(lambda x: round(x))

        df['labels'] = df['average']

        df['labels'] = pd.cut(x=df['labels'], bins=[0, 70, 100], labels=['Failed', 'Accredited'])

        df.drop(['math score', 'reading score', 'writing score', 'average'], axis=1, inplace=True)

        df.to_csv('Clean.csv')
        data_top = df.head()

        X = df.drop(['labels'], axis=1) #create my X varibles "Features" for the tree
        y = df['labels'] #create my y variables "label" for the tree

        #Binary_Decision_Tree
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=195)
        tree = DecisionTreeClassifier(max_depth=7, random_state=195)
        tree.fit(x_train, y_train)
        print('Accuracy on the training subset on the Binary :', format(tree.score(x_train, y_train)))
        print('Accuracy on the test subset on the Binary:',  format(tree.score(x_val, y_val)))
        clf = tree

        dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True,
                                   class_names=['Failed', 'Accredited'])
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("Students tree")

        Xnew = ([[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0]])  # Female, Male | bachelors degree, some college, master degree,
                                                         # associates degree, high school, some highschool | standard,
                                                         # free/reduced | none, complete
        y_predict = tree.predict(Xnew)
        print('Row selected is:', Xnew[0], y_predict)
        input("Press Enter to continue...")

        n_features = X.shape[1]
        plt.barh(range(n_features), tree.feature_importances_)
        plt.yticks(np.arange(n_features), df.columns.values)
        plt.xlabel('Features Importance')
        plt.ylabel('Feature')
        plt.show()

        #Random_Forest
        x_train, x_val, y_train, y_val = train_test_split(X, y)
        forest = RandomForestClassifier(n_estimators=80)
        forest.fit(x_train, y_train)
        n_features = X.shape[1]
        plt.barh(range(n_features), forest.feature_importances_)
        plt.yticks(np.arange(n_features), df.columns.values)
        plt.xlabel('Features Importance')
        plt.ylabel('Feature')
        plt.show()

        # KNeighborsClassifier
        clf = DecisionTreeClassifier()
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        print(knn)
        print('Accuracy of KNN n-5, on the training set:', format(knn.score(x_train, y_train)))
        print('Accuracy of KNN n-5, on the test set:', format(knn.score(x_val, y_val)))

        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=66)
        training_accuracy = []
        test_accuracy = []
        neighbors_settings = range(1, 20)
        for n_neighbors in neighbors_settings:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(x_train, y_train)
            training_accuracy.append(clf.score(x_train, y_train))
            test_accuracy.append(clf.score(x_val, y_val))
        plt.plot(neighbors_settings, training_accuracy, label='Accuracy of the Training set')
        plt.plot(neighbors_settings, test_accuracy, label='Accuracy of the Test set')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Neighbors')
        plt.legend()
        plt.show()









































































