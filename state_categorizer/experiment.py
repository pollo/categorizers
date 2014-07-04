from sklearn import cross_validation, metrics, svm
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

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

class ExperimentBase(object):
    __metaclass__ = ABCMeta

    scale = False

    @abstractproperty
    def C(self):
        return

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

    def _test_model(self, clf, X_train, X_test, y_train, y_test, log):
        #compute training score
        training_score = clf.score([e['features'] for e in X_train],
                                   y_train)
        #compute validation score
        validation_score = clf.score([e['features'] for e in X_test],
                                     y_test)
        #compute confusion matrix
        y_pred = clf.predict([e['features'] for e in X_test])
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        s =  "Training score: \n"
        s += str(training_score)+"\n"
        s += "Validation score\n"
        s += str(validation_score)+"\n"
        s += "Confusion matrix\n"
        s += str(confusion_matrix)+"\n"
        print s
        log.write(s)

    def _find_errors(self, clf, X_test, y_test, log):
        standing_samples = 0
        skiing_samples = 0
        ascending_samples = 0
        correct_standing_samples = 0
        correct_skiing_samples = 0
        correct_ascending_samples = 0
        errors = []
        for X, y in itertools.izip(X_test, y_test):
            predicted_y = clf.predict(X['features'])[0]
            if y==0:
                standing_samples += 1
                if predicted_y==y:
                    correct_standing_samples += 1
                else:
                    errors.append({'id' : X['id'],
                                   'class' : predicted_y})
            if y==1:
                skiing_samples += 1
                if predicted_y==y:
                    correct_skiing_samples += 1
                else:
                    errors.append({'id' : X['id'],
                                   'class' : predicted_y})
            if y==2:
                ascending_samples += 1
                if predicted_y==y:
                    correct_ascending_samples += 1
                else:
                    errors.append({'id' : X['id'],
                                   'class' : predicted_y})
        s =  "Fraction of correctly classified standing: \n"
        s += str(float(correct_standing_samples)/standing_samples)+"\n"
        s += "Fraction of correctly classified skiing: \n"
        s += str(float(correct_skiing_samples)/skiing_samples)+"\n"
        s += "Fraction of correctly classified ascending: \n"
        s += str(float(correct_ascending_samples)/ascending_samples)+"\n"
        print s
        log.write(s)

        return errors

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

        #scale data
        if (self.scale):
            print 'Scale data...'
            scaler = StandardScaler()
            scaler.fit([e['features'] for e in X])
            X = [{'id':e['id'],
                  'features':scaler.transform(e['features'])} for e in X]

        #split dataset
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.4, random_state=0)

        #train model
        print 'Training model...'
        t0 = time.time()
        clf = svm.SVC(kernel=self.KERNEL_TYPE,C=1)
        clf.fit([e['features'] for e in X_train],
                y_train)
        t1 = time.time()
        print 'Trained in '+str(t1-t0)+' seconds.'
        log.write('Trained in '+str(t1-t0)+' seconds.')
        joblib.dump(clf, experiment_name+"/clf.dump")

        #test model
        print 'Test model...'
        self._test_model(clf, X_train, X_test, y_train, y_test, log)

        #find errors
        print 'Find errors...'
        errors = self._find_errors(clf, X_test, y_test, log)

        #write errors to db
        print 'Storing errors on db'
        _store_to_db(errors, experiment_name)

        #close log
        log.close()
