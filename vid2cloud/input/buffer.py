"""Bufor klatek między dekoderem a silnikiem rekonstrukcji.

Będzie tu:
    RingBuffer    — bufor cykliczny o stałym rozmiarze, bezpieczny wątkowo.
                    Przy przepełnieniu nadpisuje najstarszą klatkę i zlicza drop-y.
    producer()    — wątek producenta: czyta z FrameSource i wrzuca do RingBuffer,
                    żeby dekodowanie nie czekało na inferencję.

Rozmiar bufora i liczba porzuconych klatek trafiają do logu przebiegu — przy
strumieniu na żywo to jedna z miar, czy pipeline nadąża za kamerą.
"""
