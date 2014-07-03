from sklearn import cross_validation, metrics, svm
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import numpy as np

import time, utm, itertools, os

import psycopg2
import psycopg2.extras

from local_db_settings import DB_SETTINGS

from abc import ABCMeta, abstractmethod, abstractproperty

def _build_query(auth_user_id):
    return 'select * from skilo_sc.user_location_track  where ' \
        'auth_user_id = {0} order by ts'.format(auth_user_id)

def _fetch_sorted_user_data(auth_user_id):
    with psycopg2.connect(**DB_SETTINGS) as conn:
        psycopg2.extras.register_hstore(conn)
        with conn.cursor() as cur:
            cur.execute(_build_query(auth_user_id))

            # convert python's datetime to timestamp
            dt_to_ts = lambda dt: \
                       time.mktime(dt.timetuple()) + dt.microsecond / 1E6

            data_all = [
                dict(id=t[0], auth_user_id=t[1], ts=dt_to_ts(t[2]),
                     type=t[3], x=t[4], y=t[5], z=t[6], m=t[7],
                     track_id=t[8], sensor=t[9], classification=t[11],
                     categorizers=t[12])
                for t in cur.fetchall()
            ]
    return data_all


def _store_to_db(errors, experiment_name):
    with psycopg2.connect(**DB_SETTINGS) as conn:
        psycopg2.extras.register_hstore(conn)
        with conn.cursor() as cur:
            for data in errors:
                cur.execute('UPDATE skilo_sc.user_location_track \
                SET errors = errors || hstore(%s,%s) \
                WHERE id=%s',[experiment_name,
                              str(data['class']),
                              data['id']])

class ExperimentParamsSelectionBase(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def WINDOW_SIZE(self):
        return

    @abstractproperty
    def FEATURES_PER_SAMPLE(self):
        return

    @abstractproperty
    def KERNEL_TYPE(self):
        return

    @abstractmethod
    def _extract_features(self, points):
        pass

    def _build_dataset(self, auth_ids):
        X = []
        y = []
        for auth_id in auth_ids:
            points = _fetch_sorted_user_data(auth_id)
            for i in range(self.WINDOW_SIZE/2, len(points)-self.WINDOW_SIZE/2):
                try:
                    yi = int(points[i]['classification'])
                except TypeError:
                    continue
                if (yi in [0,1,2]):
                    Xi = self._extract_features(points[i-self.WINDOW_SIZE/2:
                                                       i+self.WINDOW_SIZE/2+1])
                    if Xi:
                        assert (len(Xi) ==
                                self.FEATURES_PER_SAMPLE)
                        X.append({'id' : points[i]['id'],
                                  'features' : Xi})
                        y.append(yi)
        return X, y

    def run(self, auth_ids, experiment_name, subsampling=1.0):
        print 'Running experiment '+experiment_name
        #open log
        log = open(experiment_name+".log","w")
        try:
            os.makedirs(experiment_name)
        except OSError:
            pass

        #build dataset
        print 'Building dataset...'
        X, y = self._build_dataset(auth_ids)
        assert len(X)==len(y)

        #subsample
        print 'Subsampling dataset, using '+str(subsampling)+' of the data'
        X = X[:int(len(X)*subsampling)]
        y = y[:int(len(y)*subsampling)]

        #keep only features
        X = [e['features'] for e in X]

        #scale data
        print 'Scale data...'
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        #select parameters
        C_range = 10.0 ** np.arange(-2, 9)
        gamma_range = 10.0 ** np.arange(-5, 4)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedKFold(y=y, n_folds=3)
        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
        grid.fit(X, y)

        s = "The best classifier is: "+str(grid.best_estimator_)
        print s
        log.write(s)

        #close log
        log.close()
