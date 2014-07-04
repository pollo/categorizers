import psycopg2
import psycopg2.extras
import numpy
import sys
import time
import utm

from state_categorizer.experiment4 import Experiment
from sklearn.externals import joblib

from local_db_settings import DB_SETTINGS

def build_query(auth_user_id):
    return 'select * from skilo_sc.user_location_track  where auth_user_id = {0}' \
           ' order by ts'.format(auth_user_id)

def fetch_data(auth_user_id):
    with psycopg2.connect(**DB_SETTINGS) as conn:
        psycopg2.extras.register_hstore(conn)
        with conn.cursor() as cur:
            cur.execute(build_query(auth_user_id))

            # convert python's datetime to timestamp
            dt_to_ts = lambda dt: \
                time.mktime(dt.timetuple()) + dt.microsecond / 1E6

            data_all = [
                dict(id=t[0], auth_user_id=t[1], ts=dt_to_ts(t[2]), type=t[3],
                     x=t[4], y=t[5], z=t[6], m=t[7], track_id=t[8], sensor=t[9],
                     categorizers=t[12])
                for t in cur.fetchall()
            ]
    return data_all

def store_data(buf):
    with psycopg2.connect(**DB_SETTINGS) as conn:
        psycopg2.extras.register_hstore(conn)
        with conn.cursor() as cur:
            for data in buf:
                cur.execute('UPDATE skilo_sc.user_location_track \
                SET categorizers = %s \
                WHERE id=%s',[data['categorizers'],data['id']])

def run(auth_id):
    buf = fetch_data(auth_id)

    #load classifier
    clf = joblib.load("state_categorizer/experiment4/clf.dump")

    e = Experiment()

    for i in range(len(buf)):
        if not buf[i]['categorizers']:
            buf[i]['categorizers'] = {}

        points = buf[i-e.WINDOW_SIZE/2:
                     i+e.WINDOW_SIZE/2+1]

        if len(points) < e.WINDOW_SIZE:
            state = 0
        else:
            #extract features
            features = e._extract_features(points)

            #classify
            if features:
                state = clf.predict(features)[0]
            else:
                state = 0

        # add this for future selection of next points
        # this should never be written to the database!
        buf[i]['classification'] = str(state)

        buf[i]['categorizers']['ml_state'] = str(state)

    store_data(buf)


if __name__ == '__main__':
    auth_ids = [32, 51]
    for auth_id in auth_ids:
        run(auth_id)
