"""Small reusable result displays for the Streamlit application."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def show_confusion_matrix(matrix, labels) -> None:
    """Render a labeled confusion matrix with a color scale."""
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "Actual"
    frame.columns.name = "Predicted"
    st.dataframe(frame.style.background_gradient(cmap="Blues"), use_container_width=True)
