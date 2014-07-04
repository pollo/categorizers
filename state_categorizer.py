import psycopg2
import psycopg2.extras
import numpy
import sys
import time
import utm

from local_db_settings import DB_SETTINGS

#window of points to consider to classify the current point
#WINDOW/2 successive points and WINDOW/2 previous points are considered
WINDOW = 12
#minimum average speed that the skier should have moving from the first point
#of the window to the last to be considered moving (m/s)
MINIMUM_AVERAGE_SPEED = 0.5
#minimum average speed that the skier should have moving from the first point
#of the window to the last in low slope section to be considered skiing (m/s)
MINIMUM_DESCENDING_SPEED = 8
#minimum slope between first and last point of the window to consider the
#skier as most likely descending the ski slope
MINIMUM_SLOPE = -0.01
#maximum difference in height between last and first point to not be
#automatically considered as ascending
MAXIMUM_HEIGHT_DESC = 5
#maximu time intervall between first and last point to be considered part
#of the same track
MAXIMUM_TIME_DELTA = 5*60

def distance_3d(lon1, lat1, h1, lon2, lat2, h2):
    ym1, xm1, _, _ = utm.from_latlon(lat1, lon1)
    ym2, xm2, _, _ = utm.from_latlon(lat2, lon2)

    arr1 = numpy.array((xm1, ym1, h1))
    arr2 = numpy.array((xm2, ym2, h2))

    dist = numpy.linalg.norm(arr1-arr2)
    return dist

def distance_2d(lon1, lat1, lon2, lat2):
    ym1, xm1, _, _ = utm.from_latlon(lat1, lon1)
    ym2, xm2, _, _ = utm.from_latlon(lat2, lon2)

    arr1 = numpy.array((xm1, ym1))
    arr2 = numpy.array((xm2, ym2))

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
        if not buf[i]['categorizers']:
            buf[i]['categorizers'] = {}

        try:
            first_item = buf[i-WINDOW/2]
            item = buf[i]
            last_item = buf[i+WINDOW/2]
        except IndexError:
            state = "standing"
        else:
            print item

            #compute average speed from first to last oint of the window
            distance = distance_3d(
                first_item['x'], first_item['y'], first_item['z'],
                last_item['x'], last_item['y'], last_item['z']
            )
            time_delta = abs(last_item['ts'] - first_item['ts'])
            speed = (distance) / time_delta

            if time_delta > MAXIMUM_TIME_DELTA:
                #track is starting or ending
                state = "standing"

            if speed < MINIMUM_AVERAGE_SPEED:
                #if not moving enough fast the skier is considered standing
                state = "standing"
            else:
                #compute slope between first and last point of the window
                distance = distance_2d(
                    first_item['x'], first_item['y'],
                    last_item['x'], last_item['y'],
                )
                height = last_item['z']-first_item['z']
                slope = height/distance

                if height>MAXIMUM_HEIGHT_DESC:
                    state = "ascending"
                elif slope > MINIMUM_SLOPE and speed < MINIMUM_DESCENDING_SPEED:
                    state = "ascending"
                else:
                    state = "descending"

        buf[i]['categorizers']['state'] = state

    store_data(buf)


if __name__ == '__main__':
    auth_id = 32
    #auth_id = 51
    run(auth_id)
