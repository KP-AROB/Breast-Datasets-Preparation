import pytest
import numpy as np
from src.processing.pipeline import BasePipeline

def test_add_valid_operation():
    pipeline = BasePipeline()
    pipeline.add_operation(lambda x: x + 1)
    assert len(pipeline.operations) == 1

# def test_add_invalid_operation():
#     pipeline = BasePipeline()
#     with pytest.raises(ValueError):
#         pipeline.add_operation("not_a_function")

# def test_process_with_multiple_operations():
#     pipeline = BasePipeline()
#     pipeline.add_operation(invert)
#     pipeline.add_operation(double)

#     input_img = np.array([[100, 150], [200, 250]], dtype=np.uint8)
#     expected = double(invert(input_img))
#     output = pipeline.process(input_img)

#     assert np.array_equal(output, expected)

# def test_process_with_loader_adapter():
#     pipeline = BasePipeline()
#     pipeline.add_operation(read_dicom_adapter("dummy_path"))
#     pipeline.add_operation(double)

#     output = pipeline.process(None)
#     expected = np.ones((2, 2), dtype=np.float32) * 200

#     assert np.array_equal(output, expected)

# def test_process_raises_on_faulty_op():
#     pipeline = BasePipeline()
#     pipeline.add_operation(invert)
#     pipeline.add_operation(faulty_op)

#     with pytest.raises(RuntimeError, match="Intentional failure"):
#         pipeline.process(np.ones((2, 2), dtype=np.uint8))

# def test_process_skips_faulty_op_in_non_strict_mode():
#     pipeline = BasePipeline(strict=False)
#     pipeline.add_operation(invert)
#     pipeline.add_operation(faulty_op)
#     pipeline.add_operation(double)

#     input_img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
#     expected = double(invert(input_img))

#     output = pipeline.process(input_img)
#     assert np.array_equal(output, expected)
