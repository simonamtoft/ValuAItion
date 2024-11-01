def zip_code_mapper(zip_code: int) -> str:
    if zip_code < 1000:
        return 'Special'
    if zip_code < 3000:
        return 'Copenhagen'
    if zip_code < 3700:
        return 'North Zealand'
    if zip_code < 3800:
        return 'Bornholm'
    if zip_code < 3900:
        return 'Faroe Islands'
    if zip_code < 4000:
        return 'Greenland'
    if zip_code < 5000:
        return 'Zealand'
    if zip_code < 6000:
        return 'Funen'
    if zip_code < 10000:
        return 'Jutland'
    return 'N/A'


def municipality_code_mapper(municipality_code: int) -> str:
    if municipality_code < 150:
        return '100'
    if municipality_code < 200:
        return '150'
    if municipality_code < 250:
        return '200'
    if municipality_code < 300:
        return '250'
    if municipality_code < 350:
        return '300'
    if municipality_code < 400:
        return '350'
    if municipality_code < 420:
        return '400'
    if municipality_code < 500:
        return '420'
    if municipality_code < 550:
        return '500'
    if municipality_code < 600:
        return '550'
    if municipality_code < 650:
        return '600'
    if municipality_code < 700:
        return '650'
    if municipality_code < 760:
        return '700'
    if municipality_code < 800:
        return '760'
    if municipality_code < 900:
        return '800'
    return 'N/A'


def transform_floor(x: str) -> int:
    if x == 'st':
        return 0
    elif x == 'kl':
        return -1
    return x


def facility_clean(text: str) -> str:
    return text.split(':')[0]
