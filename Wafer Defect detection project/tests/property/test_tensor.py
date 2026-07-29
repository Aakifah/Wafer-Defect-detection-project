import numpy as np
from hypothesis import given
from hypothesis import strategies as st


@given(
    batch_size=st.integers(min_value=1, max_value=64),
    height=st.just(52),
    width=st.just(52),
    channels=st.just(1)
)
def test_input_tensor_dimensionality(batch_size: 
    int, height: int, width: int, channels: int) -> None:
    mock_tensor = np.zeros((batch_size, height, width, channels))
    assert mock_tensor.shape == (batch_size, 52, 52, 1)
    assert mock_tensor.ndim == 4