import psycopg2
import psycopg2.extras
import numpy
import sys
import time
import utm

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
                     lon=t[4], lat=t[5], z=t[6], m=t[7], track_id=t[8], sensor=t[9],
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
            item = buf[i]
            suc_item = buf[i+1]
        except IndexError:
            continue

        print item

        #transform from utm to lat lon
        x1, y1, _, _ = utm.from_latlon(item['lat'], item['lon'])
        x2, y2, _, _ = utm.from_latlon(suc_item['lat'], suc_item['lon'])

        #compute delta
        p1 = numpy.array((x1, y1, item['z']))
        p2 = numpy.array((x2, y2, suc_item['z']))
        delta_p = p2 - p1
        delta_t = suc_item['ts'] - item['ts']

        #compute velocity
        velocity = delta_p / delta_t

        #compute acceleration
        acceleration = velocity / delta_t

        if not buf[i]['categorizers']:
            buf[i]['categorizers'] = {}

        buf[i]['categorizers']['vel'] = str(numpy.linalg.norm(velocity))
        buf[i]['categorizers']['velx'] = str(velocity[0])
        buf[i]['categorizers']['vely'] = str(velocity[1])
        buf[i]['categorizers']['velz'] = str(velocity[2])

        buf[i]['categorizers']['acc'] = str(numpy.linalg.norm(acceleration))
        buf[i]['categorizers']['accx'] = str(acceleration[0])
        buf[i]['categorizers']['accy'] = str(acceleration[1])
        buf[i]['categorizers']['accz'] = str(acceleration[2])

    store_data(buf)

if __name__ == '__main__':
    auth_ids = [51, 32]
    for auth_id in auth_ids:
        run(auth_id)
