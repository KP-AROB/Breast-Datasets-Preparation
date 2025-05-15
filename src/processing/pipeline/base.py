from typing import List, Callable

class BasePipeline:
    def __init__(self):
        """
        Initializes the Pipeline with an empty list of operations.
        """
        self.operations: List[Callable] = []
        
    def add_operation(self, operation: Callable):
        """
        Adds a pre-processing operation to the pipeline.

        Parameters:
        operation (Callable): A function that takes an image as input and returns the processed image.
        """
        if not callable(operation):
            raise ValueError("The operation must be a callable function.")
        self.operations.append(operation)

    def process(self, image):
        """
        Processes an image through the pipeline.

        Parameters:
        image: The input image to be processed.

        Returns:
        The processed image after applying all operations in the pipeline.
        """
        for i, operation in enumerate(self.operations):
            try:
                image = operation(image)
            except Exception as e:
                raise RuntimeError(f"Operation {i} ({operation.__name__}) failed: {e}")
        return image
