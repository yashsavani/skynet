from load_svhn import get_datum_iterator

for datum in get_datum_iterator():
    image = datum.display_image()
    cv2.imwrite("test.png", image)
    break
