import pandas as pd
import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from tqdm import tqdm

from models import LDA, QDA, Dataloader, Dataset


class Validation:
    ds = ('letter', 'digits', 'sat', 'vowel', 'iris', )

    def __init__(self):
        self.result = []

    def __call__(self):
        for d in tqdm(self.ds):
            if d == 'iris':
                folds = Dataset(d)
                self.cross_validation(folds)
            else:
                X_train, X_test, y_train, y_test = Dataset(d)
                lda_b, qda_b = self.baseline(X_train, X_test, y_train, y_test)
                lda, lda_d, qda = self.validation(X_train, X_test, y_train, y_test)
                self.result.append({'Exp.': d, 'Baseline LDA': lda_b, 'Baseline QDA': qda_b, 'LDA no diag': lda,
                                    'LDA diag': lda_d, 'QDA': qda})
        df = pd.DataFrame(self.result)
        df.to_csv(f'result-{time.strftime("%m-%d_%H")}.csv', index=False)
        return df

    @staticmethod
    def baseline(x_train, x_test, y_train, y_test):
        linear_clf = LinearDiscriminantAnalysis()
        linear_clf.fit(x_train, y_train)
        linear_score = linear_clf.score(x_test, y_test)

        quad_clf = QuadraticDiscriminantAnalysis()
        quad_clf.fit(x_train, y_train)
        quad_score = quad_clf.score(x_test, y_test)

        return linear_score, quad_score

    @staticmethod
    def validation(x_train, x_test, y_train, y_test):
        lda = LDA(x_train, y_train, diag=False)
        ls = lda.score(x_test, y_test)

        lda_d = LDA(x_train, y_train, diag=True)
        ls_d = lda_d.score(x_test, y_test)

        qda = QDA(x_train, y_train)
        qs = qda.score(x_test, y_test)

        return ls, ls_d, qs

    def cross_validation(self, dataset):
        res = np.zeros((5, 5))
        for idx, (X_train, X_test, y_train, y_test) in enumerate(dataset):
            lda, lda_d, qda = self.validation(X_train, X_test, y_train, y_test)
            lda_b, qda_b = self.baseline(X_train, X_test, y_train, y_test)
            self.result.append({'Exp.': f'iris fold-{idx}', 'Baseline LDA': lda_b, 'Baseline QDA': qda_b,
                                'LDA no diag': lda, 'LDA diag': lda_d, 'QDA': qda})
            res[idx] = [lda_b, qda_b, lda, lda_d, qda]
        rs = np.mean(res, axis=0)
        self.result.append({'Exp.': 'iris', 'Baseline LDA': rs[0], 'Baseline QDA': rs[1], 'LDA no diag': rs[2],
                            'LDA diag': rs[3], 'QDA': rs[4]})


if __name__ == '__main__':
    val = Validation()
    val()
