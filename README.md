# DeepFace Face Recognition Tool

A robust and accurate face verification system that leverages state-of-the-art deep learning models to compare and verify faces across different images. This tool uses the DeepFace library and provides multiple models, detectors, and visualization options.

![Example Verification](output/verification_result.png)

## Features

- **Multiple Face Recognition Models**:
  - ArcFace (default) - One of the most accurate face recognition models
  - VGG-Face, Facenet, OpenFace, DeepID, and more
  - Ensemble option that uses multiple models and votes on the result

- **Advanced Face Detection Backends**:
  - RetinaFace (default) - State-of-the-art face detector
  - MTCNN, SSD, MediaPipe, Dlib, and OpenCV options
  - Automatic fallbacks if detection fails

- **Comprehensive Analysis**:
  - Similarity scores and confidence levels
  - Distance metrics with proper thresholds
  - Detailed results with multiple metrics
  - Visual comparison with automated decision

- **Flexible Options**:
  - Choose distance metrics (cosine, euclidean)
  - Toggle face detection enforcement
  - Create visualizations of results
  - Debug options for troubleshooting

## Installation

### Prerequisites

- Python 3.6+
- pip package manager

### Quick Install

```bash
# Clone this repository
git clone https://github.com/your-username/deepface-recognition.git
cd deepface-recognition

# Install dependencies
pip install -r requirements.txt
```

The tool will automatically attempt to install DeepFace if not found on your system.

### Manual Dependencies Installation

If you prefer to install dependencies manually:

```bash
pip install deepface opencv-python matplotlib numpy
```

## Usage

### Basic Usage

```bash
python deepface_recognition.py image1.jpg image2.jpg
```

This will:
1. Compare the faces in the two images
2. Generate a JSON report with similarity scores and confidence
3. Create a visualization image in the output directory

### Advanced Usage

#### Use a specific face recognition model:

```bash
python deepface_recognition.py image1.jpg image2.jpg --model ArcFace
```

Available models: `VGG-Face`, `Facenet`, `Facenet512`, `OpenFace`, `DeepFace`, `DeepID`, `ArcFace`, `Dlib`, `SFace`, `ensemble`

#### Use a different face detector:

```bash
python deepface_recognition.py image1.jpg image2.jpg --detector retinaface
```

Available detectors: `opencv`, `ssd`, `dlib`, `mtcnn`, `retinaface`, `mediapipe`

#### Change the distance metric:

```bash
python deepface_recognition.py image1.jpg image2.jpg --metric cosine
```

Available metrics: `cosine`, `euclidean`, `euclidean_l2`

#### Handle challenging images:

```bash
python deepface_recognition.py image1.jpg image2.jpg --no-enforce-detection
```

#### Use ensemble of multiple models (most accurate):

```bash
python deepface_recognition.py image1.jpg image2.jpg --model ensemble
```

#### Disable visualization:

```bash
python deepface_recognition.py image1.jpg image2.jpg --no-visualize
```

#### Enable debug output:

```bash
python deepface_recognition.py image1.jpg image2.jpg --debug
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `image1` | Path to first image (required) |
| `image2` | Path to second image (required) |
| `--model` | Face recognition model to use (default: `ArcFace`) |
| `--detector` | Face detector backend (default: `retinaface`) |
| `--metric` | Distance metric (default: `cosine`) |
| `--no-enforce-detection` | Don't enforce face detection (helpful for challenging images) |
| `--no-visualize` | Don't create visualization |
| `--debug` | Enable debug output |

## Example Output

```json
{
  "match": false,
  "distance": 0.5623421072959899,
  "threshold": 0.40000000000000036,
  "similarity": 0.4376578927040101,
  "confidence": "low",
  "model": "ArcFace",
  "detector": "retinaface",
  "distance_metric": "cosine",
  "processing_time": 2.345918893814087,
  "message": "Match not found with low confidence (0.4377)"
}
```

## Output Files

The tool generates the following files in the `output` directory:

- `verification_result.png`: Visual comparison of the two images with matching results
- `original_img1.jpg` and `original_img2.jpg`: Original input images
- `face1.jpg` and `face2.jpg`: Extracted face regions (if detection succeeds)

## Troubleshooting

### Face detection fails

If face detection fails, try the following:
- Use `--no-enforce-detection` flag
- Try a different detector with `--detector mtcnn` or `--detector ssd`
- Check if the image is too low resolution or the face is too small

### DeepFace installation issues

If you encounter issues with automatic DeepFace installation:
1. Try manual installation: `pip install deepface`
2. Check Python version (3.6+ required)
3. Ensure you have proper GPU drivers if using GPU acceleration

### Out of memory errors

If you get out of memory errors:
1. Try a lighter model: `--model VGG-Face` or `--model OpenFace`
2. Use a less resource-intensive detector: `--detector opencv`

## Requirements

- Python 3.6+
- DeepFace
- OpenCV
- Matplotlib
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) library by Sefik Ilkin Serengil
- Face recognition models: ArcFace, VGG-Face, FaceNet, etc.
