"""
Batch Processor for Ultimate Rotoscopy
=======================================

High-performance batch processing with parallel execution.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

from ultimate_rotoscopy.core.engine import RotoscopyEngine, ProcessingResult
from ultimate_rotoscopy.models.sam3 import SegmentationPrompt


class BatchProcessor:
    """
    High-performance batch processor.

    Features:
    - Thread-based parallelism
    - Process-based parallelism option
    - Memory management
    - Progress tracking
    """

    def __init__(
        self,
        engine: Optional[RotoscopyEngine] = None,
        num_workers: int = 4,
        use_multiprocessing: bool = False,
    ):
        self.engine = engine
        self.num_workers = num_workers
        self.use_multiprocessing = use_multiprocessing

    def process_images(
        self,
        images: List[Union[np.ndarray, Path, str]],
        prompts: Optional[List[SegmentationPrompt]] = None,
        callback=None,
    ) -> List[ProcessingResult]:
        """
        Process multiple images in parallel.

        Args:
            images: List of images or image paths
            prompts: Optional prompts for each image
            callback: Progress callback (current, total)

        Returns:
            List of ProcessingResult
        """
        results = [None] * len(images)
        total = len(images)

        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor

        with executor_class(max_workers=self.num_workers) as executor:
            futures = {}

            for i, image in enumerate(images):
                prompt = prompts[i] if prompts else None
                future = executor.submit(self._process_single, image, prompt)
                futures[future] = i

            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                completed += 1

                if callback:
                    callback(completed, total)

        return results

    def _process_single(
        self,
        image: Union[np.ndarray, Path, str],
        prompt: Optional[SegmentationPrompt]
    ) -> ProcessingResult:
        """Process a single image."""
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image))

        return self.engine.process(image, prompt=prompt)
