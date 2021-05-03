# noinspection PyUnresolvedReferences
import pytest
from os.path import exists
from joblib import load


@pytest.fixture(scope="session")
def loads_model_history():
    MODEL_PATH = '../face_mask_model'
    HISTORY_PATH = MODEL_PATH + '/history.joblib'
    model = load(HISTORY_PATH)

    return model


# Checks if the specified file path exists and as such there is a trained model.
def test_trained_model_exists():
    # Setup
    MODEL_PATH = '../face_mask_model'
    HISTORY_PATH = MODEL_PATH + '/history.joblib'

    # Exercise
    check = exists(HISTORY_PATH)

    # Verify
    assert check


def test_val_accuracy_trained_model_is_90(loads_model_history):
    # Setup
    accuracy = loads_model_history['val_accuracy'][-1]

    # Exercise
    check = accuracy >= 0.90

    # Verify
    assert check


def test_val_loss_trained_model_is_20(loads_model_history):
    # Setup
    accuracy = loads_model_history['val_loss'][-1]

    # Exercise
    check = accuracy <= 0.20

    # Verify
    assert check


def test_accuracy_trained_model_is_90(loads_model_history):
    # Setup
    accuracy = loads_model_history['accuracy'][-1]

    # Exercise
    check = accuracy >= 0.90

    # Verify
    assert check


def test_loss_trained_model_is_20(loads_model_history):
    # Setup
    accuracy = loads_model_history['loss'][-1]

    # Exercise
    check = accuracy <= 0.20

    # Verify
    assert check