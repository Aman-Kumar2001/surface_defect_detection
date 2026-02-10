<h1>Exploratory Data Analysis (EDA)</h1>

The dataset contains six types of industrial surface defects:

1. Crazing
2. Pitted Surface
3. Scratches
4. Inclusion
5. Patches
6. Rolled-in Scale

<h3> Dataset-level observations </h3>

- The dataset is balanced across all six defect classes, which reduces bias during training.

- All images have a uniform spatial resolution of 200 × 200 pixels.

- Pixel intensity values range from 0 to 255, indicating that normalization will be required during model training.

<h3>Visual analysis of defect characteristics</h3>

- Patches, Inclusion, Scratches, and Crazing are generally clearly visible and distinguishable in most images.

- Pitted Surface defects are subtle and small, making them harder to visually identify and potentially challenging for a model.

- Rolled-in Scale defects are visible but exhibit lower contrast, which may affect detection reliability.

- Patches and Inclusion show similar grayscale texture patterns, which could lead to class confusion.

- Certain samples of Crazing and Rolled-in Scale also appear visually similar, suggesting potential misclassification between these classes.

<h3>Implications for modeling </h3>

- The presence of visually similar defect classes indicates that the model will need to learn fine-grained texture differences.

- Subtle defects (e.g., Pitted Surface) may require stronger augmentation, careful loss design, and error analysis.
