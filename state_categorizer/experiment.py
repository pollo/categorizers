from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn import svm

import time, utm

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

class ExperimentBase(object):
    __metaclass__ = ABCMeta

    WINDOW_SIZE = 11

    @abstractproperty
    def FEATURES_PER_POINT(self):
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
                                self.WINDOW_SIZE*self.FEATURES_PER_POINT)
                        X.append(Xi)
                        y.append(yi)
        return X, y

    def run(self, auth_ids, dump_filename):
        #build dataset
        print 'Building dataset...'
        X, y = self._build_dataset(auth_ids)

        X = X[:10000]
        y = y[:10000]

        #split dataset
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.4, random_state=0)

        #train model
        print 'Training model...'
        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        joblib.dump(clf, dump_filename)

        #test model
        print 'Test model...'
        print clf.score(X_test, y_test)
