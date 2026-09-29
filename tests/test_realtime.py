import numpy as np

from backend.realtime import (
    encode_row,
    assign_tda_node,
    preprocess_realtime,
    mappings,
    tda_centers,
    feature_columns,
)


def sample_customer():
    return {
        "gender": "Male",
        "paymentmethod": "UPI",
        "industry": "Ecommerce",
        "age": 45,
        "tenure": 24,
        "monthlycharges": 65.0,
    }


def test_encode_row_returns_expected_shape():
    result = encode_row(sample_customer())

    assert isinstance(result, np.ndarray)
    assert result.dtype.kind == "f"
    assert len(result) == len(mappings) + 3


def test_encode_row_handles_unknown_category():
    customer = sample_customer()
    customer["gender"] = "UnknownGender"

    result = encode_row(customer)

    gender_index = list(mappings.keys()).index("gender")
    assert result[gender_index] == -1


def test_assign_tda_node_returns_valid_node():
    center = tda_centers[0]

    node = assign_tda_node(center)

    assert isinstance(node, (int, np.integer))
    assert 0 <= node < len(tda_centers)


def test_preprocess_realtime_output_shapes():
    final_features, kmeans_features, explainer_vector = preprocess_realtime(
        sample_customer()
    )

    assert final_features.shape[0] == len(feature_columns)
    assert kmeans_features.shape[0] > 0
    assert explainer_vector.shape == (6,)