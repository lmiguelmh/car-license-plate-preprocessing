"""
@author lmiguelmh
@since 20170415
"""

import cv2

# HAAR classifier
haar_path = 'D:\\projects\\car-license-plate-recognition\\config\\haarcascade_russian_plate_number.xml'
cascade = cv2.CascadeClassifier(haar_path)


def haar_plate_detection(plate_path, scale_factor=1.3, min_neighbors=3, min_size=(0, 0)):
    img = cv2.imread(plate_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale
    plates = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    coordinates = []
    for (x, y, w, h) in plates:
        coordinates.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return coordinates


from openalpr import Alpr

alpr_home = 'c:\\OpenALPR\\openalpr_64\\'
eu_alpr = Alpr('eu', alpr_home + 'openalpr.conf', alpr_home + 'runtime_data')
us_alpr = Alpr('us', alpr_home + 'openalpr.conf', alpr_home + 'runtime_data')


# TODO: use other countries
# au_alpr = Alpr('au', alpr_home + 'openalpr.conf', alpr_home + 'runtime_data')
# gb_alpr = Alpr('gb', alpr_home + 'openalpr.conf', alpr_home + 'runtime_data')
# kr_alpr = Alpr('kr', alpr_home + 'openalpr.conf', alpr_home + 'runtime_data')
# mx_alpr = Alpr('mx', alpr_home + 'openalpr.conf', alpr_home + 'runtime_data')
# sg_alpr = Alpr('sg', alpr_home + 'openalpr.conf', alpr_home + 'runtime_data')


def alpr_plate_detection(plate_path, country='eu', top_n=5, default_region='wa', detect_region=False):
    # Invalid pattern provided: wa
    # Valid patterns are located in the eu.patterns file
    if country == 'eu':
        alpr = eu_alpr
    else:
        alpr = us_alpr

    if not alpr.is_loaded():
        return []

    alpr.set_top_n(top_n)
    alpr.set_default_region(default_region)
    alpr.set_detect_region(detect_region)

    jpeg_bytes = open(plate_path, 'rb').read()
    results = alpr.recognize_array(jpeg_bytes)
    coordinates = []
    for result in results['results']:
        coordinates.append([(result['coordinates'][0]['x'], result['coordinates'][0]['y']),
                            (result['coordinates'][1]['x'], result['coordinates'][1]['y']),
                            (result['coordinates'][2]['x'], result['coordinates'][2]['y']),
                            (result['coordinates'][3]['x'], result['coordinates'][3]['y'])])
    return coordinates


def get_plates_coordinates(plate_path):
    """
    :param plate_path: 
    :return: [[(260, 256), (409, 229), (415, 264), (263, 291)]] 
    """
    coordinates = alpr_plate_detection(plate_path, country='eu', top_n=1)
    if len(coordinates) > 0:
        return coordinates

    coordinates = alpr_plate_detection(plate_path, country='us', top_n=1)
    if len(coordinates) > 0:
        return coordinates

    coordinates = haar_plate_detection(plate_path)
    if len(coordinates) > 0:
        return coordinates

    return []
