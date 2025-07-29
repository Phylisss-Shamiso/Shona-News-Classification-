import streamlit as st
import pickle
import time

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from embedding_utils import EmbeddingGenerator
import pandas as pd


class CalibratedClassifierPipeline(ClassifierMixin, BaseEstimator):
    def __init__(self, base_estimator, method='sigmoid', cv=5):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.calibrated_classifier = None

    def fit(self, X, y):
        self.calibrated_classifier = CalibratedClassifierCV(
            self.base_estimator, method=self.method, cv=self.cv
        )
        self.calibrated_classifier.fit(X, y)
        self.classes_ = self.calibrated_classifier.classes_
        return self

    def predict(self, X):
        return self.calibrated_classifier.predict(X)

    def predict_proba(self, X):
        return self.calibrated_classifier.predict_proba(X)

    def get_params(self, deep=True):
        return {
            'base_estimator': self.base_estimator,
            'method': self.method,
            'cv': self.cv
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    @property
    def classes_(self):
        if hasattr(self.calibrated_classifier, 'classes_'):
            return self.calibrated_classifier.classes_
        if hasattr(self.base_estimator, 'classes_'):
            return self.base_estimator.classes_
        raise AttributeError("'CalibratedClassifierPipeline' object has no attribute 'classes_'")


model= pickle.load(open('shonanews1.pkl','rb'))

# Custom CSS for larger text area
st.markdown("""
<style>
    .stTextArea textarea {
        min-height: 250px;
        font-size: 16px;
        line-height: 1.5;
        padding: 15px;
    }
    .stTextArea label {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title('📰 ChiShona News Classification')

# Create input area
message = st.text_area(
    'Enter any Shona article text:',
    height=100,
    placeholder='Paste the full Shona article here...'
)

submit = st.button('Predict', type='primary')

if submit and message:
    # Get prediction and probabilities
    predicted_label = model.predict([message])[0]  # This returns a string like 'business'
    probabilities = model.predict_proba([message])[0]

    # Get the classifier from the pipeline
    classifier = model.named_steps['classifier']

    # Find the index of the predicted label
    try:
        prediction_idx = list(classifier.classes_).index(predicted_label)
    except ValueError:
        st.error(f"Unknown prediction label: {predicted_label}")
        st.stop()

    confidence = probabilities[prediction_idx]

    # Define your class labels and messages
    CLASS_INFO = {
        0: ('business', '💼 related article'),
        1: ('health', '🩺 related article'),
        2: ('politics', '🏛️ related article'),
        3: ('sports', '⚽ related article')
    }

    # Get class info
    if prediction_idx in CLASS_INFO:
        class_name, custom_message = CLASS_INFO[prediction_idx]
    else:
        class_name = predicted_label
        custom_message = "No custom message defined"

    # Show results
    if confidence > 0.7:
        st.success(f"Category: {class_name.capitalize()}  {custom_message}")
    elif confidence > 0.5:
        st.warning(f"Category: {class_name.capitalize()}  {custom_message}")
    else:
        st.error(f"Category: {class_name.capitalize()} cle{custom_message}")
    if prediction_idx in CLASS_INFO:
        class_name, custom_message = CLASS_INFO[prediction_idx]
    else:
        class_name = predicted_label
        custom_message = "No custom message defined"

        # Show results



    #Show probability distribution
    st.subheader("Class Distribution:")

    # Create placeholders for progress bars and text
    progress_bars = []
    text_placeholders = []
    for i, prob in enumerate(probabilities):
        class_name = CLASS_INFO.get(i, (f"Class {i}", ""))[0]
        text_placeholder = st.empty()
        progress_bar = st.progress(0)
        text_placeholders.append(text_placeholder)
        progress_bars.append(progress_bar)

    # Animation parameters
    total_steps = 100
    delay = 0.02  # Adjust for faster/slower animation

    # Animate all progress bars simultaneously
    for step in range(total_steps + 1):
        for i, prob in enumerate(probabilities):
            # Calculate current progress for this step
            current_progress = min(prob, step / total_steps)

            # Update progress bar
            progress_bars[i].progress(current_progress)

            # Update text
            class_name = CLASS_INFO.get(i, (f"Class {i}", ""))[0]
            text_placeholders[i].text(f"{class_name}: {current_progress:.2%}")

        time.sleep(delay)

    # Final update to exact values
    for i, prob in enumerate(probabilities):
        progress_bars[i].progress(prob)
        class_name = CLASS_INFO.get(i, (f"Class {i}", ""))[0]
        text_placeholders[i].text(f"{class_name}: {prob:.2%}")
