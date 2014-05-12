import psycopg2
import psycopg2.extras
import numpy
import sys
import time
import utm

from local_db_settings import DB_SETTINGS


def distance_3d(lon1, lat1, h1, lon2, lat2, h2):
    ym1, xm1, _, _ = utm.from_latlon(lat1, lon1)
    ym2, xm2, _, _ = utm.from_latlon(lat2, lon2)

    arr1 = numpy.array((xm1, ym1, h1))
    arr2 = numpy.array((xm2, ym2, h2))

    dist = numpy.linalg.norm(arr1-arr2)
    return dist

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

    for i in range(len(buf)):
        try:
            prev_item = buf[i-1]
            item = buf[i]
            suc_item = buf[i+1]
        except IndexError:
            continue

        print item

        #compute speed: previous and successive points are considered
        distance_prev = distance_3d(
            item['x'], item['y'], item['z'],
            prev_item['x'], prev_item['y'], prev_item['z']
        )
        distance_suc = distance_3d(
            item['x'], item['y'], item['z'],
            suc_item['x'], suc_item['y'], suc_item['z']
        )

        time_delta = abs(suc_item['ts'] - prev_item['ts'])

        speed = (distance_prev+distance_suc) / time_delta

        if not buf[i]['categorizers']:
            buf[i]['categorizers'] = {}

        buf[i]['categorizers']['speed'] = str(speed)

    store_data(buf)


if __name__ == '__main__':
    auth_id = 32
    #auth_id = 51
    run(auth_id)
