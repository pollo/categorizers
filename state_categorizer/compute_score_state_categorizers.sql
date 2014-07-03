SELECT classification, COUNT(classification)
FROM skilo_sc.user_location_track
WHERE classification != state AND
      classification != -1
GROUP BY classification

SELECT classification, COUNT(classification)
FROM skilo_sc.user_location_track
WHERE classification != -1
GROUP BY classification