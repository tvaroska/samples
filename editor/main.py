from fasthtml.common import fast_app, serve
from fasthtml.common import Div, P

app,rt = fast_app()

@rt('/')
def get(): 
    return Div(P('Hello World!'), hx_get="/change")

@rt('/change')
def get(): 
    return P('Nice to be here!')

serve()