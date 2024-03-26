import cv2
from pyzbar.pyzbar import decode

def barcode_reader(img): # takes barcode img array as an input
    result_holder = list()
    detected_code = decode(img)

    if not detected_code:
        return False
    else:
        flag = 0
        for code in detected_code:
            result_holder.append({'idx' : flag, 'data' : code.data.decode(), 'type' : code.type})
            flag += 1

    return result_holder