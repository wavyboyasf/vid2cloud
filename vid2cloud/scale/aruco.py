"""Detekcja markerów ArUco na pełnej rozdzielczości klatki.

Będzie tu:
    detect()           — cv2.aruco.ArucoDetector (API OpenCV >= 4.7) + cornerSubPix
                         na obrazie w skali szarości; zwraca {id: ndarray(4, 2)}.
    rescale_corners()  — przeliczenie współrzędnych narożników z pełnej rozdzielczości
                         na rozdzielczość wejścia modelu (pointmapa jest w niej).

Detekcja idzie po pełnej klatce (4K), bo marker 12 cm z 3 m ma przy 512 px około
15 pikseli. Dopiero gotowe narożniki skalujemy w dół — patrz scale/lift.py.
Zależności (opencv-contrib-python) instalujemy w env vid2cloud-tools, nie mast3r-slam.
"""
