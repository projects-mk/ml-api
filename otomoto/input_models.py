from pydantic import BaseModel


class OtomotoInputData(BaseModel):
    marka: str
    model: str
    rok_produkcji: int
    przebieg: float
    pojemnosc_skokowa: float
    moc: int
    rodzaj_paliwa: str
    skrzynia_biegow: str
    naped: str
    spalanie_w_miescie: float
    nadwozie: str
    liczba_drzwi: int
    liczba_miejsc: int
    bezwypadkowy: str
    serwisowany_w_aso: str
    stan: str
