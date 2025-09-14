def test_imports():
    import streamlit
    import pandas as pd
    import numpy as np
    from src.generate_data import synth_wafer
    assert synth_wafer(kind="center").shape == (28,28)