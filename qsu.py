# utility functions for use in demo notebooks
from IPython.display import HTML


class qsu:
    @staticmethod
    def print_pauli(pauli):
        text = str(pauli)
        text = text.replace(
            'X', '<span style="color:red; font-weight:bold">X</span>')
        text = text.replace(
            'Y', '<span style="color:magenta; font-weight:bold">Y</span>')
        text = text.replace(
            'Z', '<span style="color:blue; font-weight:bold">Z</span>')
        display(HTML(
            '<div class="highlight"><pre style="line-height:1!important;">{}</pre></div>'.format(text)))
