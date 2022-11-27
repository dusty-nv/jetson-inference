
from dash import Input

from .card import create_card, card_generator


@card_generator(Input('navbar_test_card', 'n_clicks'), Input('navbar_test_card_2', 'n_clicks'))
def create_test_card(n, n2):
    n += n2
    print(f"create_test_card n_clicks={n}")
    return create_card([f"Test card {n} body"], title=f"Test card {n}", id=f"test_{n}")
    