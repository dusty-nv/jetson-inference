
import dash
from dash import Input, ALL

from .card import create_card, card_callback


def create_test_card(n):
    print(f"create_test_card n={n}")
    return create_card([f"Test card {n} body"], title=f"Test card {n}", id=f"test_{n}")

@card_callback(
    Input('navbar_test_card', 'n_clicks'), 
    Input('navbar_test_card_2', 'n_clicks')
)
def on_test_card(n1, n2):
    print(f"on_test_card({n1}, {n2})")
    print(dash.ctx.triggered[0])
    if n1 or n2:
        return create_test_card(n1+n2)
    return None
    

@card_callback(
    Input({'type': f'navbar_menu_test_card', 'index': ALL}, 'n_clicks')
)
def on_test_card_menu(n):
    print(f"on_test_card_menu:")
    print(n)
    print(f"on_test_card_menu dash triggered dict type {dash.ctx.triggered_id['type']}")
    print(f"on_test_card_menu triggered_index {dash.ctx.triggered_id['index']}")
    return create_test_card(dash.ctx.triggered_id['index'])